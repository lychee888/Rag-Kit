"""Regression tests for the 2026-10-21 audit fixes (card t_77d0d976).

Covers F2 (hybrid score contract), F3 (keyword degradation is loud),
F5 (deterministic chunk ids + content-hash skip), F6 (config path and
validation hardening), and F7 (poll daemon lock + per-folder state).
No embedding model or network required: engines are fakes, stores are
real LanceDB on tmp paths, and watcher state/lock dirs are redirected
into tmp via monkeypatched module globals.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from rag_kit import config as cfg_mod
from rag_kit import content_state, watchlock
from rag_kit.config import Config
from rag_kit.store import VectorStore


@pytest.fixture(autouse=True)
def isolate_state(tmp_path, monkeypatch):
    """Keep watcher buckets, legacy state, and locks inside tmp_path."""
    monkeypatch.setattr(content_state, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(content_state, "LEGACY_STATE_PATH", tmp_path / "legacy.json")
    monkeypatch.setattr(watchlock, "_LOCK_DIR", tmp_path / "locks")


def make_store(tmp_path, rows, dim=8):
    store = VectorStore(db_path=str(tmp_path / "db"))
    chunks = [
        {"id": f"c{i}", "text": text, "source": src, "page": 1, "chunk_idx": i}
        for i, (text, src) in enumerate(rows)
    ]
    rng = np.random.default_rng(42 + len(rows))
    vectors = rng.random((len(chunks), dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    store.add_chunks(chunks, vectors)
    return store, vectors


# --------------------------------------------------------------------------- #
# F2 — hybrid score contract
# --------------------------------------------------------------------------- #

class TestF2ScoreContract:
    ROWS = [
        ("quantum entanglement experiment notes", "/docs/a.pdf"),
        ("recipe for chocolate cake with frosting", "/docs/b.pdf"),
        ("quarterly financial report highlights", "/docs/c.pdf"),
    ]

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.99, 1.0])
    def test_scores_within_unit_interval_and_ordered(self, tmp_path, alpha):
        store, vectors = make_store(tmp_path, self.ROWS)
        results = store.search_hybrid(
            "chocolate cake recipe", vectors[1], top_k=3, alpha=alpha
        )
        scores = [r["score"] for r in results]
        assert scores, "expected results"
        assert all(0.0 <= s <= 1.0 for s in scores), f"score out of [0,1]: {scores}"
        assert scores == sorted(scores, reverse=True), "scores must be descending"

    def test_fusion_exposes_raw_rrf_and_similarity(self, tmp_path):
        store, vectors = make_store(tmp_path, self.ROWS)
        results = store.search_hybrid("chocolate cake recipe", vectors[1],
                                      top_k=3, alpha=0.5)
        top = results[0]
        assert "rrf_score" in top, "hybrid results must expose the raw RRF value"
        assert top["score"] == pytest.approx(min(1.0, top["rrf_score"] * 61), abs=1e-3)
        # The document that is rank 1 in both legs tops the blend and keeps
        # its cosine similarity available under "similarity".
        assert top["source"] == "/docs/b.pdf"
        assert 0.0 <= top["similarity"] <= 1.0

    def test_keyword_only_scores_are_normalized(self, tmp_path):
        # Raw BM25 _score is unbounded; the contract requires [0,1].
        store, vectors = make_store(tmp_path, self.ROWS)
        results = store.search_hybrid("chocolate cake recipe", vectors[0],
                                      top_k=3, alpha=0.0)
        assert results[0]["score"] == pytest.approx(1.0)
        assert all("_score" not in r for r in results)
        assert all(r["score"] <= 1.0 for r in results)

    def test_alpha_one_matches_vector_search(self, tmp_path):
        store, vectors = make_store(tmp_path, self.ROWS)
        hybrid = store.search_hybrid("anything", vectors[2], top_k=3, alpha=1.0)
        plain = store.search(vectors[2], top_k=3)
        assert [r["score"] for r in hybrid] == [r["score"] for r in plain]


# --------------------------------------------------------------------------- #
# F3 — keyword degradation is never silent
# --------------------------------------------------------------------------- #

class TestF3Degradation:
    def _broken_fts(self, store, monkeypatch):
        table = store._ensure_table()
        orig_search = table.search

        def fake_search(query, **kwargs):
            if kwargs.get("query_type") == "fts":
                raise RuntimeError("fts index exploded")
            return orig_search(query, **kwargs)

        monkeypatch.setattr(table, "search", fake_search)

    def test_hybrid_marks_degraded_vector_fallback(self, tmp_path, monkeypatch):
        store, vectors = make_store(tmp_path, [("chocolate cake", "/a.pdf")])
        self._broken_fts(store, monkeypatch)
        results = store.search_hybrid("chocolate cake", vectors[0], top_k=2, alpha=0.5)
        assert results, "vector leg still returns results"
        meta = store.last_search
        assert meta["degraded"] is True
        assert meta["keyword_status"] == "failed"
        assert "exploded" in meta["keyword_error"]
        # Degraded fallback scores are vector cosine similarities.
        assert meta["mode"] in ("hybrid", "vector")

    def test_keyword_only_fails_loud(self, tmp_path, monkeypatch):
        store, vectors = make_store(tmp_path, [("chocolate cake", "/a.pdf")])
        self._broken_fts(store, monkeypatch)
        with pytest.raises(RuntimeError, match="Keyword search unavailable"):
            store.search_hybrid("chocolate cake", vectors[0], top_k=2, alpha=0.0)

    def test_clean_search_not_degraded(self, tmp_path):
        store, vectors = make_store(tmp_path, [("chocolate cake", "/a.pdf")])
        store.search_hybrid("chocolate cake", vectors[0], top_k=2, alpha=0.5)
        assert store.last_search["degraded"] is False
        assert store.last_search["keyword_status"] == "ok"

    def test_cli_json_surfaces_degradation(self, tmp_path, monkeypatch):
        from rag_kit.cli import main as cli

        cfg = Config(watch_folder=str(tmp_path), db_path=str(tmp_path / "db"),
                     search_alpha=0.5)
        monkeypatch.setattr(cli, "_ensure_config", lambda: cfg)
        fake_store = SimpleNamespace(
            count_rows=lambda: 3,
            list_sources=lambda: ["/a.pdf"],
            search_hybrid=lambda *a, **k: [
                {"score": 0.4, "source": "/a.pdf", "page": 1, "chunk_idx": 0,
                 "text": "body"}],
            last_search={"mode": "hybrid", "keyword_status": "failed",
                         "degraded": True, "keyword_error": "boom"},
        )
        monkeypatch.setattr(cli, "create_store_from_config", lambda _: fake_store)
        monkeypatch.setattr(cli, "create_engine_from_config", lambda _: SimpleNamespace(
            embed_query=lambda t: np.zeros(4, dtype=np.float32)))
        out = CliRunner().invoke(cli.app, ["query", "--json", "anything"])
        assert out.exit_code == 0
        payload = json.loads(out.stdout)
        assert payload["degraded"] is True
        assert payload["keyword_status"] == "failed"
        assert payload["keyword_error"] == "boom"


# --------------------------------------------------------------------------- #
# F5 — deterministic chunk ids + content-hash skip
# --------------------------------------------------------------------------- #

class TestF5IncrementalIndexing:
    def test_chunk_ids_are_content_derived(self):
        from rag_kit.embed import Chunk

        a = Chunk(text="hello world", source="/x.pdf", page=1, chunk_idx=0)
        b = Chunk(text="hello world", source="/x.pdf", page=1, chunk_idx=0)
        c = Chunk(text="hello there", source="/x.pdf", page=1, chunk_idx=0)
        d = Chunk(text="hello world", source="/x.pdf", page=2, chunk_idx=0)
        assert a.chunk_id == b.chunk_id
        assert a.chunk_id != c.chunk_id
        assert a.chunk_id != d.chunk_id
        assert len(a.chunk_id) == 16

    def test_reingest_unchanged_doc_is_row_stable(self, tmp_path):
        from rag_kit.ingest.pipeline import ingest_file

        doc = tmp_path / "note.txt"
        doc.write_text("Alpha sentence. Beta sentence.", encoding="utf-8")
        first = ingest_file(str(doc), use_ocr=False, use_vlm=False)
        second = ingest_file(str(doc), use_ocr=False, use_vlm=False)
        assert [c["id"] for c in first] == [c["id"] for c in second]

    def _watcher_cfg(self, tmp_path, watch_dir):
        return Config(watch_folder=str(watch_dir), db_path=str(tmp_path / "db"),
                      vlm_enabled=False, supported_extensions=[".txt"],
                      chunk_size=512, chunk_overlap=64, watch_interval=1)

    def _scan(self, tmp_path, monkeypatch, cfg):
        from rag_kit import embed, watcher

        calls = {"embed": 0}

        def fake_engine(_cfg):
            def embed_texts(texts):
                calls["embed"] += 1
                return np.tile(np.eye(8, dtype=np.float32)[0], (len(texts), 1))
            return SimpleNamespace(embed_texts=embed_texts)

        monkeypatch.setattr("rag_kit.config.get_config", lambda: cfg)
        monkeypatch.setattr(embed, "create_engine_from_config", fake_engine)
        result = watcher.scan_and_ingest(once=True)
        return result, calls

    def test_poll_scan_skips_unchanged_and_detects_changes(self, tmp_path, monkeypatch):
        watch = tmp_path / "watch"
        watch.mkdir()
        (watch / "a.txt").write_text("First document body.", encoding="utf-8")
        cfg = self._watcher_cfg(tmp_path, watch)

        result, calls = self._scan(tmp_path, monkeypatch, cfg)
        assert result["new"] == 1 and not result["errors"]
        assert calls["embed"] == 1

        # Rescan unchanged: zero embedding work (the F5 boot symptom).
        result, calls = self._scan(tmp_path, monkeypatch, cfg)
        assert (result["new"], result["changed"], result["deleted"]) == (0, 0, 0)
        assert calls["embed"] == 0, "unchanged files must not re-embed"

        (watch / "a.txt").write_text("Edited document body!", encoding="utf-8")
        result, calls = self._scan(tmp_path, monkeypatch, cfg)
        assert result["changed"] == 1
        assert calls["embed"] == 1

    def test_empty_store_and_empty_folder_boot_clean(self, tmp_path, monkeypatch):
        watch = tmp_path / "watch"
        watch.mkdir()
        result, calls = self._scan(tmp_path, monkeypatch,
                                   self._watcher_cfg(tmp_path, watch))
        assert result["status"] == "ok" and not result["errors"]
        assert calls["embed"] == 0

    def test_rows_deleted_manually_forces_reingest(self, tmp_path, monkeypatch):
        watch = tmp_path / "watch"
        watch.mkdir()
        (watch / "a.txt").write_text("Body text here.", encoding="utf-8")
        cfg = self._watcher_cfg(tmp_path, watch)
        self._scan(tmp_path, monkeypatch, cfg)

        store = VectorStore(cfg.db_path)
        store.delete_by_source(str((watch / "a.txt").resolve()))

        result, calls = self._scan(tmp_path, monkeypatch, cfg)
        assert result["new"] == 1, "hash match must not skip when rows are gone"
        assert calls["embed"] == 1

    def test_legacy_state_migration_adopts_matching_entries(self, tmp_path, monkeypatch):
        watch = tmp_path / "watch"
        watch.mkdir()
        doc = watch / "a.txt"
        doc.write_text("Already indexed content.", encoding="utf-8")
        digest = hashlib.md5(doc.read_bytes()).hexdigest()
        other = str(tmp_path / "elsewhere" / "b.txt")
        # Pre-F7 global state file: one entry under this root, one elsewhere.
        content_state.LEGACY_STATE_PATH.write_text(
            json.dumps({"files": {str(doc.resolve()): digest, other: "0" * 32}}),
            encoding="utf-8",
        )
        cfg = self._watcher_cfg(tmp_path, watch)
        store = VectorStore(cfg.db_path)
        store.replace_source(str(doc.resolve()),
                             [{"id": "x1", "text": "Already indexed content.",
                               "source": str(doc.resolve()), "page": 0,
                               "chunk_idx": 0}],
                             np.ones((1, 8), dtype=np.float32))

        result, calls = self._scan(tmp_path, monkeypatch, cfg)
        assert result["new"] == 0 and result["changed"] == 0
        assert calls["embed"] == 0, "adopted hashes must skip re-embedding"
        assert not result["errors"], "unrelated legacy entries must not be treated as deletes"

        # The folder bucket now exists and contains only this root's entry.
        bucket = content_state.load_bucket(watch)
        assert bucket == {str(doc.resolve()): digest}


# --------------------------------------------------------------------------- #
# F6 — config hardening
# --------------------------------------------------------------------------- #

class TestF6Config:
    def _load(self, tmp_path, data):
        path = tmp_path / "rag-kit.yaml"
        path.write_text(yaml.dump(data) if not isinstance(data, str) else data,
                        encoding="utf-8")
        return cfg_mod.load_config(path)

    def test_tilde_paths_expanded(self, tmp_path):
        cfg = self._load(tmp_path, {"db_path": "~/lancedb",
                                    "watch_folder": "~/Documents/rag-ingest",
                                    "model_dir": "~/models"})
        home = str(Path.home())
        assert cfg.db_path == home + str(Path("/lancedb"))
        assert cfg.watch_folder.startswith(home)
        assert cfg.model_dir.startswith(home)

    def test_non_dict_yaml_raises_actionable_error(self, tmp_path):
        with pytest.raises(ValueError, match="mapping"):
            self._load(tmp_path, ["db_path: ~/lancedb"])

    def test_invalid_yaml_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not valid YAML"):
            self._load(tmp_path, "db_path: [unclosed\n")

    def test_unknown_keys_warn(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger="rag_kit.config"):
            self._load(tmp_path, {"db_pth": "~/typo", "chunk_size": 512})
        assert any("db_pth" in rec.getMessage() and "unknown" in rec.getMessage().lower()
                   for rec in caplog.records), caplog.text

    def test_cli_config_set_warns_on_unknown_key(self, tmp_path, monkeypatch):
        from rag_kit.cli import main as cli

        monkeypatch.setenv("RAG_KIT_CONFIG", str(tmp_path / "rag-kit.yaml"))
        result = CliRunner().invoke(cli.config_app, ["set", "anygarbage", "x"],
                                    catch_exceptions=False)
        assert result.exit_code == 0
        combined = (result.output or "") + getattr(result, "stderr", "")
        assert "not a known rag-kit config key" in combined


# --------------------------------------------------------------------------- #
# F7 — poll daemon respects the lock; state is per-folder
# --------------------------------------------------------------------------- #

class TestF7WatcherIsolation:
    def _cfg(self, watch, db):
        return Config(watch_folder=str(watch), db_path=str(db),
                      vlm_enabled=False, supported_extensions=[".txt"],
                      watch_interval=1)

    def _fake_engine(self, monkeypatch):
        from rag_kit import embed, watcher  # noqa: F401

        monkeypatch.setattr(embed, "create_engine_from_config", lambda _c: SimpleNamespace(
            embed_texts=lambda texts: np.ones((len(texts), 8), dtype=np.float32)))

    def test_poll_daemon_defers_to_watchlock_owner(self, tmp_path, monkeypatch):
        from rag_kit import watcher

        watch = tmp_path / "watch"
        watch.mkdir()
        (watch / "a.txt").write_text("Contended body.", encoding="utf-8")
        monkeypatch.setattr("rag_kit.config.get_config",
                            lambda: self._cfg(watch, tmp_path / "db"))
        self._fake_engine(monkeypatch)

        lock = watchlock.acquire(watch)  # pretend `rag watch` owns the folder
        assert lock is not None
        result = watcher.scan_and_ingest(once=True)
        assert result["status"] == "already_running"
        assert VectorStore(str(tmp_path / "db")).count_rows() == 0, \
            "second watcher must not ingest while locked"
        watchlock.release(lock)
        result = watcher.scan_and_ingest(once=True)
        assert result["new"] == 1

    def test_folder_switch_does_not_wipe_other_folder(self, tmp_path, monkeypatch):
        from rag_kit import watcher

        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        doc_b = b / "keep.txt"
        doc_b.write_text("Belongs to folder B.", encoding="utf-8")
        db = tmp_path / "db"
        store = VectorStore(str(db))
        store.replace_source(str(doc_b.resolve()),
                             [{"id": "b1", "text": "Belongs to folder B.",
                               "source": str(doc_b.resolve()), "page": 0,
                               "chunk_idx": 0}],
                             np.ones((1, 8), dtype=np.float32))
        monkeypatch.setattr("rag_kit.config.get_config", lambda: self._cfg(a, db))
        self._fake_engine(monkeypatch)

        result = watcher.scan_and_ingest(once=True)
        assert result["deleted"] == 0, \
            "scanning folder A must not treat folder B's state as deletions"
        assert store.count_rows() == 1, "folder B rows must survive"
