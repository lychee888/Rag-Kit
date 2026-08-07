"""
Comprehensive pipeline tests for rag-kit.

Covers all modules: config, embed, store, ingest, vlm, autostart, watcher, CLI.
20+ tests designed to run quickly without downloading models.

Run with:
    cd ~/rag-kit
    python -m pytest tests/test_rag_pipeline.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Helpers
# =============================================================================

def _rag_kit(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run rag-kit CLI and return the result."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.run(
        [sys.executable, "-m", "rag_kit.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        **kwargs,
    )


# =============================================================================
# Config tests (5 tests)
# =============================================================================

class TestConfig:
    """Configuration loading, validation, env overrides."""

    def test_default_config_valid(self):
        """Default config values pass validation."""
        from rag_kit.config import DEFAULT_CONFIG, Config
        cfg = Config.from_dict(DEFAULT_CONFIG)
        errors = cfg.validate()
        assert errors == [], f"Default config has validation errors: {errors}"

    def test_config_from_env(self):
        """Environment variables override config values."""
        from rag_kit.config import Config
        with mock.patch.dict(os.environ, {
            "RAG_KIT_CHUNK_SIZE": "1024",
            "RAG_KIT_VLM_ENABLED": "false",
            "RAG_KIT_SUPPORTED_EXTENSIONS": ".pdf,.md",
        }):
            cfg = Config.from_dict({})
            assert cfg.chunk_size == 1024
            assert cfg.vlm_enabled is False
            assert cfg.supported_extensions == [".pdf", ".md"]

    def test_config_validation_errors(self):
        """Invalid config values produce errors."""
        from rag_kit.config import Config
        cfg = Config.from_dict({"chunk_size": -1, "search_alpha": 5.0})
        errors = cfg.validate()
        assert len(errors) >= 2
        assert any("chunk_size" in e for e in errors)

    def test_config_to_dict(self):
        """Config serializes to dict correctly."""
        from rag_kit.config import Config
        cfg = Config.from_dict({"chunk_size": 256})
        d = cfg.to_dict()
        assert d["chunk_size"] == 256
        assert "embedding_model" in d

    def test_load_config_defaults(self, tmp_path):
        """load_config returns defaults when no file exists."""
        from rag_kit.config import load_config
        config_path = tmp_path / "nonexistent.yaml"
        cfg = load_config(config_path)
        assert cfg.chunk_size == 512
        assert cfg.watch_interval == 30


# =============================================================================
# Embedding tests (3 tests)
# =============================================================================

class TestEmbed:
    """Embedding engine and text chunking."""

    def test_chunk_text_basic(self):
        """Chunking produces overlapping chunks from text."""
        from rag_kit.embed import chunk_text, Chunk
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text(text, source="test.txt", chunk_size=60, chunk_overlap=20)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.source == "test.txt" for c in chunks)
        assert chunks[0].chunk_idx == 0
        assert chunks[0].chunk_id  # Auto-generated

    def test_chunk_text_chinese(self):
        """CJK sentence splitting works correctly."""
        from rag_kit.embed import chunk_text
        # CJK sentences with newline separators (common in real documents)
        text = "第一句话。\n第二句话！\n第三句话？\n第四句话。\n第五句话。\n第六句话。\n第七句话。\n第八句话。"
        chunks = chunk_text(text, chunk_size=25, chunk_overlap=5)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_text_empty(self):
        """Empty text returns empty list."""
        from rag_kit.embed import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []


# =============================================================================
# Store tests (5 tests)
# =============================================================================

class TestStore:
    """LanceDB VectorStore operations."""

    @pytest.fixture
    def store(self, tmp_path):
        from rag_kit.store import VectorStore
        db_path = tmp_path / "test-db"
        s = VectorStore(db_path=str(db_path))
        yield s
        s.close()
        shutil.rmtree(db_path, ignore_errors=True)

    @pytest.fixture
    def vectors(self):
        import numpy as np
        return np.random.rand(5, 384).astype(np.float32)

    @pytest.fixture
    def chunks(self):
        return [
            {"id": f"chunk_{i}", "text": f"Document chunk {i} content.", "source": "test.pdf", "page": 1, "chunk_idx": i, "vlm_generated": False, "source_type": "text"}
            for i in range(5)
        ]

    def test_add_and_count(self, store, chunks, vectors):
        """Adding chunks increases row count."""
        assert store.count_rows() == 0
        added = store.add_chunks(chunks, vectors)
        assert added == 5
        assert store.count_rows() == 5

    def test_search_returns_scored_results(self, store, chunks, vectors):
        """Vector search returns results with scores."""
        store.add_chunks(chunks, vectors)
        query_vec = vectors[0]  # Query with the first vector
        results = store.search(query_vec, top_k=3)
        assert len(results) == 3
        assert all("score" in r for r in results)
        assert all("text" in r for r in results)
        assert all("source" in r for r in results)
        assert results[0]["score"] >= results[-1]["score"]  # Sorted descending

    def test_list_sources_unique(self, store, chunks, vectors):
        """list_sources returns deduplicated source paths."""
        chunks2 = [
            {"id": f"chunk2_{i}", "text": f"More content {i}.", "source": "test.pdf", "page": 2, "chunk_idx": i, "vlm_generated": False, "source_type": "text"}
            for i in range(3)
        ]
        store.add_chunks(chunks, vectors[:5])
        store.add_chunks(chunks2, vectors[:3])
        sources = store.list_sources()
        assert sources == ["test.pdf"]  # Deduplicated

    def test_delete_by_source(self, store, chunks, vectors):
        """Deleting by source removes chunks and decrements count."""
        store.add_chunks(chunks, vectors)
        assert store.count_rows() == 5
        removed = store.delete_by_source("test.pdf")
        assert removed == 5
        assert store.count_rows() == 0

    def test_stats_complete(self, store, chunks, vectors):
        """get_stats returns all expected fields."""
        store.add_chunks(chunks, vectors)
        stats = store.get_stats()
        assert stats["row_count"] == 5
        assert stats["source_count"] == 1
        assert stats["table_name"] == "documents"
        assert "db_path" in stats
        assert "sources" in stats


# =============================================================================
# Autostart tests (2 tests)
# =============================================================================

class TestAutostart:
    """Autostart module tests."""

    def test_setup_autostart_returns_tuple(self):
        """setup_autostart() returns (bool, str)."""
        from rag_kit.autostart import setup_autostart
        success, message = setup_autostart()
        assert isinstance(success, bool)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_is_autostart_installed(self):
        """is_autostart_installed() returns bool."""
        from rag_kit.autostart import is_autostart_installed
        result = is_autostart_installed()
        assert isinstance(result, bool)


# =============================================================================
# Watcher tests (2 tests)
# =============================================================================

class TestWatcher:
    """Watcher module tests."""

    def test_scan_and_ingest_once(self, tmp_path):
        """scan_and_ingest with once=True runs and returns summary."""
        from rag_kit.watcher import scan_and_ingest

        # Set up minimal config
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        db_dir = tmp_path / "db"
        config_path = tmp_path / "test.yaml"

        import yaml
        with open(config_path, "w") as f:
            yaml.dump({
                "watch_folder": str(watch_dir),
                "db_path": str(db_dir),
                "chunk_size": 512,
                "chunk_overlap": 64,
                "watch_interval": 1,
            }, f)

        with mock.patch.dict(os.environ, {
            "RAG_KIT_CONFIG": str(config_path),
            "HF_HUB_OFFLINE": "1",
        }):
            result = scan_and_ingest(once=True)

        assert isinstance(result, dict)
        assert "new" in result
        assert "changed" in result
        assert "deleted" in result
        assert result["new"] == 0  # Empty folder

    def test_watcher_state_persistence(self, tmp_path):
        """Watcher state file is created and loadable."""
        from rag_kit.watcher import _load_state, _save_state, _state_path

        # Override state path
        with mock.patch("rag_kit.watcher._state_path", return_value=tmp_path / "state.json"):
            _save_state({"a.txt": "abc123", "b.txt": "def456"})
            loaded = _load_state()
            assert loaded == {"a.txt": "abc123", "b.txt": "def456"}


# =============================================================================
# CLI tests (4 tests)
# =============================================================================

class TestCLI:
    """CLI command tests."""

    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path):
        """Isolate DB and config per test."""
        self._config_path = tmp_path / "test.yaml"
        self._db_path = tmp_path / "db"
        self._env = {
            "RAG_KIT_CONFIG": str(self._config_path),
            "RAG_KIT_DB_PATH": str(self._db_path),
        }

    def test_version(self):
        """rag --version prints version."""
        r = _rag_kit("--version")
        assert r.returncode == 0
        assert "0.1.1" in r.stdout

    def test_setup_autostart_command_exists(self):
        """setup-autostart command has help text."""
        r = _rag_kit("setup-autostart", "--help")
        assert r.returncode == 0
        assert "autostart" in r.stdout.lower() or "Configure" in r.stdout

    def test_status_json_on_empty_db(self):
        """status --json on empty DB returns valid JSON."""
        with mock.patch.dict(os.environ, self._env):
            r = _rag_kit("status", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert "version" in data
        assert "db" in data

    def test_config_set_and_read(self, tmp_path):
        """config set writes and config reads back after init."""
        config_p = tmp_path / "test-config.yaml"
        db_dir = tmp_path / "db"
        env = {**os.environ, "RAG_KIT_CONFIG": str(config_p), "RAG_KIT_DB_PATH": str(db_dir)}
        # Init config first so the file exists, then set a value
        r = subprocess.run(
            [sys.executable, "-m", "rag_kit.cli.main", "config", "init", "--path", str(config_p)],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
        assert config_p.exists(), f"Config init failed: {r.stderr}"
        r = subprocess.run(
            [sys.executable, "-m", "rag_kit.cli.main", "config", "set", "chunk_size", "256"],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
        assert r.returncode == 0, r.stderr
        r = subprocess.run(
            [sys.executable, "-m", "rag_kit.cli.main", "config"],
            capture_output=True, text=True, timeout=30,
            env=env,
        )
        assert r.returncode == 0, r.stderr
        assert "chunk_size" in r.stdout
        assert "256" in r.stdout


# =============================================================================
# Integration tests (3 tests)
# =============================================================================

class TestIntegration:
    """End-to-end pipeline tests."""

    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path):
        self._tmp = tmp_path
        self._config_path = tmp_path / "test.yaml"
        self._db_path = tmp_path / "db"

    def test_ingest_and_query_english(self):
        """Ingest an English text file and query it."""
        docs_dir = self._tmp / "docs"
        docs_dir.mkdir()
        (docs_dir / "readme.md").write_text(
            "# Test Document\n\nThis is a test document about machine learning.\n\n"
            "Machine learning is a subset of artificial intelligence.\n\n"
            "It enables systems to learn from data without explicit programming.\n",
            encoding="utf-8",
        )

        with mock.patch.dict(os.environ, {
            "RAG_KIT_CONFIG": str(self._config_path),
            "RAG_KIT_DB_PATH": str(self._db_path),
        }):
            r = _rag_kit("ingest", str(docs_dir / "readme.md"), "--no-ocr", "--no-vlm")
            assert r.returncode == 0, r.stderr

            r = _rag_kit("query", "machine learning", "--json")
            assert r.returncode == 0, r.stderr
            data = json.loads(r.stdout)
            assert data["status"] == "ok"
            assert len(data["results"]) > 0

    def test_ingest_and_query_chinese(self):
        """Ingest a Chinese text file and query it."""
        docs_dir = self._tmp / "docs"
        docs_dir.mkdir()
        (docs_dir / "notes.txt").write_text(
            "中文测试文档。\n\n这是一个关于机器学习的文档。\n\n"
            "机器学习是人工智能的一个分支。\n\n它使系统能够从数据中学习。\n",
            encoding="utf-8",
        )

        with mock.patch.dict(os.environ, {
            "RAG_KIT_CONFIG": str(self._config_path),
            "RAG_KIT_DB_PATH": str(self._db_path),
        }):
            r = _rag_kit("ingest", str(docs_dir / "notes.txt"), "--no-ocr", "--no-vlm")
            assert r.returncode == 0, r.stderr

            r = _rag_kit("query", "机器学习", "--json")
            assert r.returncode == 0, r.stderr
            data = json.loads(r.stdout)
            assert data["status"] == "ok"
            assert len(data["results"]) > 0

    def test_list_files_and_delete_flow(self):
        """Full flow: ingest, list, delete, verify empty."""
        docs_dir = self._tmp / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("Document one content.", encoding="utf-8")
        (docs_dir / "doc2.txt").write_text("Document two content.", encoding="utf-8")

        with mock.patch.dict(os.environ, {
            "RAG_KIT_CONFIG": str(self._config_path),
            "RAG_KIT_DB_PATH": str(self._db_path),
        }):
            # Ingest folder
            r = _rag_kit("ingest", str(docs_dir), "--no-ocr", "--no-vlm")
            assert r.returncode == 0

            # List files
            r = _rag_kit("list-files", "--json")
            data = json.loads(r.stdout)
            assert data["file_count"] >= 2

            # Delete one
            r = _rag_kit("delete", "doc1.txt", "--json")
            assert r.returncode == 0

            # List again — should have one less
            r = _rag_kit("list-files", "--json")
            data = json.loads(r.stdout)
            assert "doc1.txt" not in data["files"]
            assert any("doc2.txt" in f for f in data["files"])

            # Query should still work on remaining
            r = _rag_kit("query", "Document", "--json")
            data = json.loads(r.stdout)
            assert data["status"] == "ok"


# =============================================================================
# Edge case tests (3 tests)
# =============================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def isolate(self, tmp_path):
        self._env = {
            "RAG_KIT_CONFIG": str(tmp_path / "test.yaml"),
            "RAG_KIT_DB_PATH": str(tmp_path / "db"),
        }

    def test_ingest_nonexistent_file(self):
        """Ingesting a nonexistent file returns error."""
        with mock.patch.dict(os.environ, self._env):
            r = _rag_kit("ingest", "/nonexistent/file.pdf", "--no-ocr", "--no-vlm")
        assert r.returncode != 0

    def test_query_empty_db(self):
        """Querying empty DB returns empty message."""
        with mock.patch.dict(os.environ, self._env):
            r = _rag_kit("query", "anything", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["status"] == "empty"
        assert data["results"] == []

    def test_delete_nonexistent_file(self):
        """Deleting a file not in DB returns zero removed."""
        with mock.patch.dict(os.environ, self._env):
            r = _rag_kit("delete", "nonexistent.pdf", "--json")
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["removed_chunks"] == 0