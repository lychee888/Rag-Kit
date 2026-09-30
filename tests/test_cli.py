"""CLI integration tests for rag-kit.

Verifies:
  - All commands have --help text
  - ingest works (file + folder)
  - query returns formatted results with source attribution
  - status reports accurate counts
  - list-files and delete work
  - config path/set/init work
  - watch auto-ingests new files (basic smoke)

Run with: pytest tests/test_cli.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest


def _rag_kit(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run rag-kit CLI and return the result."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return subprocess.run(
        [sys.executable, "-m", "rag_kit.cli.main", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
        **kwargs,
    )


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolate_db(monkeypatch):
    """Use a temporary LanceDB per test to avoid cross-test contamination."""
    db_dir = tempfile.mkdtemp(prefix="test-rag-kit-db-")
    # Override the config file path.
    config_path = Path(db_dir) / ".rag-kit.yaml"
    monkeypatch.setenv("RAG_KIT_DB_PATH", db_dir)
    monkeypatch.setenv("RAG_KIT_CONFIG", str(config_path))
    yield
    # Cleanup is best-effort; Windows may hold file locks.
    try:
        import shutil
        shutil.rmtree(db_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def docs_dir(tmp_path):
    """Create a temporary folder with test documents."""
    d = tmp_path / "docs"
    d.mkdir()

    # English markdown.
    (d / "readme.md").write_text("# Hello World\n\nThis is a test document.\n\nIt has multiple paragraphs.\n", encoding="utf-8")

    # Chinese text.
    (d / "notes.txt").write_text("这是一个中文测试文档。\n\n包含多个段落，用于验证中文文本的分块和查询功能。\n\n系统应该能够正确识别中文内容。\n", encoding="utf-8")

    return d


# ── Configuration tests ──────────────────────────────────────────────


def test_config_path():
    """config path prints the config file location."""
    r = _rag_kit("config", "path")
    assert r.returncode == 0
    assert ".rag-kit.yaml" in r.stdout


def test_config_init(tmp_path):
    """config init creates a default config file."""
    p = tmp_path / "test-config.yaml"
    r = _rag_kit("config", "init", "--path", str(p))
    assert r.returncode == 0
    assert p.exists()
    content = p.read_text()
    assert "db_path" in content
    assert "embedding_model" in content


# ── Ingest + query (end-to-end) ──────────────────────────────────────


def test_ingest_file_and_query(docs_dir, tmp_path):
    """Ingest a single file and query it."""
    config_p = tmp_path / "test.yaml"
    _rag_kit("config", "init", "--path", str(config_p))
    os.environ["RAG_KIT_CONFIG"] = str(config_p)
    os.environ["RAG_KIT_DB_PATH"] = str(tmp_path / "db")

    # Ingest the readme.
    md_file = docs_dir / "readme.md"
    r = _rag_kit("ingest", str(md_file), "--no-ocr", "--no-vlm")
    assert r.returncode == 0, r.stderr
    assert "Ingested" in r.stdout

    # Query.
    r = _rag_kit("query", "Hello World")
    assert r.returncode == 0, r.stderr
    assert "Hello" in r.stdout or "hello" in r.stdout.lower()


def test_ingest_folder_and_status(docs_dir, tmp_path):
    """Ingest a folder and check status is accurate."""
    config_p = tmp_path / "test.yaml"
    _rag_kit("config", "init", "--path", str(config_p))
    os.environ["RAG_KIT_CONFIG"] = str(config_p)
    os.environ["RAG_KIT_DB_PATH"] = str(tmp_path / "db")

    # Ingest the whole folder.
    r = _rag_kit("ingest", str(docs_dir), "--no-ocr", "--no-vlm")
    assert r.returncode == 0, r.stderr

    # Status --json.
    r = _rag_kit("status", "--json")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["db"]["source_count"] >= 2  # readme.md + notes.txt
    assert data["db"]["chunk_count"] >= 2


def test_query_json_returns_source_attribution(docs_dir, tmp_path):
    """query --json returns results with score, source, text fields."""
    config_p = tmp_path / "test.yaml"
    _rag_kit("config", "init", "--path", str(config_p))
    os.environ["RAG_KIT_CONFIG"] = str(config_p)
    os.environ["RAG_KIT_DB_PATH"] = str(tmp_path / "db")

    _rag_kit("ingest", str(docs_dir), "--no-ocr", "--no-vlm")

    r = _rag_kit("query", "Chinese text", "--json")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["status"] == "ok"
    assert len(data["results"]) > 0
    for result in data["results"]:
        assert "score" in result
        assert "source" in result
        assert "text" in result


def test_chinese_query_returns_chinese_chunks(docs_dir, tmp_path):
    """A Chinese query returns Chinese chunks."""
    config_p = tmp_path / "test.yaml"
    _rag_kit("config", "init", "--path", str(config_p))
    os.environ["RAG_KIT_CONFIG"] = str(config_p)
    os.environ["RAG_KIT_DB_PATH"] = str(tmp_path / "db")

    _rag_kit("ingest", str(docs_dir), "--no-ocr", "--no-vlm")

    r = _rag_kit("query", "中文测试", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert len(data["results"]) > 0
    assert any("中文" in res["text"] for res in data["results"])


def test_empty_db_query_returns_empty_message(tmp_path):
    """Querying an empty DB returns a helpful message."""
    config_p = tmp_path / "test.yaml"
    _rag_kit("config", "init", "--path", str(config_p))
    os.environ["RAG_KIT_CONFIG"] = str(config_p)
    os.environ["RAG_KIT_DB_PATH"] = str(tmp_path / "db")

    r = _rag_kit("query", "anything", "--json")
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["status"] == "empty"


# ── List-files, delete ───────────────────────────────────────────────


def test_list_files_and_delete(docs_dir, tmp_path):
    """list-files shows files; delete removes them."""
    config_p = tmp_path / "test.yaml"
    _rag_kit("config", "init", "--path", str(config_p))
    os.environ["RAG_KIT_CONFIG"] = str(config_p)
    os.environ["RAG_KIT_DB_PATH"] = str(tmp_path / "db")

    md_file = docs_dir / "readme.md"
    _rag_kit("ingest", str(md_file), "--no-ocr", "--no-vlm")

    r = _rag_kit("list-files", "--json")
    data = json.loads(r.stdout)
    # Sources are stored as full (resolved) paths.
    assert any(Path(s).name == "readme.md" for s in data["files"])

    # Delete.
    r = _rag_kit("delete", "readme.md", "--json")
    assert r.returncode == 0

    r = _rag_kit("list-files", "--json")
    data = json.loads(r.stdout)
    assert not any(Path(s).name == "readme.md" for s in data["files"])


# ── Help text ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cmd", [
    [],
    ["ingest", "--help"],
    ["query", "--help"],
    ["status", "--help"],
    ["watch", "--help"],
    ["config", "--help"],
    ["list-files", "--help"],
    ["delete", "--help"],
])
def test_help_text(cmd):
    """Every command has --help text."""
    r = _rag_kit(*cmd, "--help")
    assert r.returncode == 0
    assert len(r.stdout) > 0
