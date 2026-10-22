"""
Poll-based folder watcher — scan the ingest directory for new, changed, or
deleted files on a fixed interval.

Unlike the watchdog-based ``rag watch`` command (which uses OS-level file
events), this module polls the filesystem every *WATCH_INTERVAL* seconds.
It's designed to run as a lightweight background daemon started via
``pythonw -m rag_kit.watcher`` on Windows or ``nohup rag watch &``
on Linux.

State lives in a per-watch-folder bucket file (``rag_kit.content_state``)
so restarts skip unchanged files AND so two different watch folders can
never cross-wipe each other's rows (audit F7; the old single global
``~/.rag-kit-watcher-state.json`` is adopted once per folder for
migration).  The daemon respects the same per-folder ``watchlock`` that
``rag watch`` uses: running both watchers on one folder exits cleanly
instead of double-ingesting.

Usage:
    python -m rag_kit.watcher          # Run once with defaults
    python -m rag_kit.watcher --interval 30  # Poll every 30 seconds
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from rag_kit import content_state

# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #

_logger = logging.getLogger("rag_kit.watcher")
_logger.setLevel(logging.INFO)
_console = logging.StreamHandler(sys.stderr)
_console.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
)
_logger.addHandler(_console)


# --------------------------------------------------------------------------- #
# State tracking (per-folder buckets; folder=None keeps the legacy global file)
# --------------------------------------------------------------------------- #

def _state_path(folder: "str | Path | None" = None) -> Path:
    """Return the state file path for *folder* (or the legacy global file)."""
    if folder is not None:
        return content_state.bucket_path(folder)
    return content_state.LEGACY_STATE_PATH


def _load_state(folder: "str | Path | None" = None) -> dict:
    """Load persisted watcher state (file_path -> md5_hash) for *folder*."""
    if folder is None:
        stored = content_state.load_state_file(_state_path())
        return stored if stored is not None else {}
    stored = content_state.load_state_file(_state_path(folder))
    if stored is not None:
        return stored
    adopted = content_state.legacy_entries_under(folder)
    if adopted:
        _save_state(adopted, folder)
    return adopted


def _save_state(state: dict, folder: "str | Path | None" = None) -> None:
    """Persist watcher state to disk for *folder* (or the legacy file)."""
    content_state.save_state_file(
        _state_path(folder), state,
        root=folder if folder is not None else None,
    )


def _file_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    return content_state.file_hash(file_path)


# --------------------------------------------------------------------------- #
# Main scan loop
# --------------------------------------------------------------------------- #


def scan_and_ingest(once: bool = False) -> dict:
    """Scan the configured ingest directory and process any changes.

    Args:
        once:   If True, run one scan and return.  If False, loop forever
                (the daemon mode).

    Returns:
        Summary dict with keys ``new``, ``changed``, ``deleted``,
        ``errors``, and ``status`` (``already_running`` when another
        watcher holds this folder's lock).
    """
    from rag_kit.config import get_config

    try:
        config = get_config()
    except Exception as exc:
        _logger.error("Failed to load config: %s", exc)
        return {"new": 0, "changed": 0, "deleted": 0, "errors": [str(exc)]}

    ingest_dir = Path(config.watch_folder).expanduser().resolve()
    interval = config.watch_interval
    supported_exts = set(config.supported_extensions)

    # Lazy imports to avoid loading heavy modules at import time.
    from rag_kit import watchlock
    from rag_kit.embed import create_engine_from_config
    from rag_kit.ingest import SUPPORTED_EXTENSIONS, ingest_file
    from rag_kit.store import create_store_from_config

    # One watcher per folder, regardless of surface (audit F7).
    lock = watchlock.acquire(ingest_dir)
    if lock is None:
        _logger.warning(
            "Another watcher already holds the lock for %s; "
            "exiting without scanning.", ingest_dir,
        )
        return {"new": 0, "changed": 0, "deleted": 0, "errors": [],
                "status": "already_running"}

    _logger.info(
        "Watcher started: folder=%s interval=%ds extensions=%s",
        ingest_dir, interval, supported_exts,
    )

    engine = None
    store = None

    def _ensure_engine():
        nonlocal engine
        if engine is None:
            engine = create_engine_from_config(config)
        return engine

    def _ensure_store():
        nonlocal store
        if store is None:
            store = create_store_from_config(config)
        return store

    try:
        while True:
            summary = {"new": 0, "changed": 0, "deleted": 0,
                       "errors": [], "status": "ok"}

            # --- Discover files on disk ---
            disk_files = {}  # abs_path -> md5
            if ingest_dir.exists():
                for ext in supported_exts:
                    if ext not in SUPPORTED_EXTENSIONS:
                        # Avoid ingesting unsupported extensions even if misconfigured.
                        continue
                    for f in ingest_dir.rglob("*" + ext):
                        if f.is_file():
                            abs_path = str(f.resolve())
                            disk_files[abs_path] = _file_md5(f)

            # --- Load previous state (this folder's bucket only) ---
            prev_state = _load_state(ingest_dir)

            # --- Detect new and changed files ---
            to_ingest = []
            for abs_path, md5_hash in disk_files.items():
                if md5_hash == "":  # unreadable
                    continue
                if abs_path not in prev_state:
                    to_ingest.append(abs_path)
                    summary["new"] += 1
                elif prev_state[abs_path] != md5_hash:
                    to_ingest.append(abs_path)
                    summary["changed"] += 1
                elif not _ensure_store().has_source(abs_path):
                    # Hash says unchanged, but the rows are gone (manual
                    # delete / rebuilt store): re-ingest instead of skipping.
                    to_ingest.append(abs_path)
                    summary["new"] += 1

            # --- Detect deleted files ---
            for abs_path in list(prev_state.keys()):
                if abs_path not in disk_files:
                    _logger.info("File deleted from disk, removing from DB: %s", abs_path)
                    try:
                        removed = _ensure_store().delete_by_source(abs_path)
                        _logger.info("  Removed %d chunk(s)", removed)
                    except Exception as exc:
                        _logger.error("  Failed to remove from DB: %s", exc)
                        summary["errors"].append(str(exc))
                    del prev_state[abs_path]
                    summary["deleted"] += 1

            # --- Ingest new/changed files ---
            for abs_path in to_ingest:
                _logger.info("Ingesting: %s", abs_path)
                try:
                    file_path = Path(abs_path)
                    chunks = ingest_file(
                        file_path,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                        languages=config.languages,
                        use_ocr=True,
                        use_vlm=config.vlm_enabled,
                    )
                except Exception as exc:
                    _logger.error("  Failed to ingest %s: %s", abs_path, exc)
                    from rag_kit.ingest.pipeline import IncompleteExtractionError
                    if isinstance(exc, IncompleteExtractionError):
                        summary["status"] = "partial"
                    summary["errors"].append(str(exc))
                    continue

                # Convert to store format.
                store_chunks = []
                for i, c in enumerate(chunks):
                    store_chunks.append({
                        "id": c.get("id", c.get("chunk_id", "")),
                        "text": c["text"],
                        "source": c.get("source", c.get("source_file", "")),
                        "page": int(c.get("page", 0)),
                        "chunk_idx": int(c.get("chunk_idx", i)),
                        "vlm_generated": bool(c.get("vlm_generated", False)),
                        "source_type": str(c.get("source_type", "text")),
                    })

                try:
                    eng = _ensure_engine()
                    texts = [c["text"] for c in store_chunks]
                    vectors = eng.embed_texts(texts) if texts else []
                    added = _ensure_store().replace_source(abs_path, store_chunks, vectors)
                    _logger.info("  %d chunks stored", added)
                except Exception as exc:
                    _logger.error("  Failed to store %s: %s", abs_path, exc)
                    summary["errors"].append(str(exc))
                    continue

                # Update content hash (re-read: the file may have changed
                # while we were ingesting it).
                prev_state[abs_path] = _file_md5(Path(abs_path))

            # --- Persist updated state ---
            if to_ingest or summary["deleted"] > 0:
                _save_state(prev_state, ingest_dir)

            if (summary["new"] > 0 or summary["changed"] > 0
                    or summary["deleted"] > 0):
                _logger.info(
                    "Scan summary: new=%d changed=%d deleted=%d errors=%d",
                    summary["new"], summary["changed"], summary["deleted"],
                    len(summary["errors"]),
                )
            else:
                _logger.debug("No changes detected")

            if summary["errors"] and summary["status"] == "ok":
                summary["status"] = "error"
            if once:
                return summary

            time.sleep(interval)
    finally:
        watchlock.release(lock)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Entry point for ``python -m rag_kit.watcher``.

    Accepts ``--once`` to run one scan, ``--interval <N>`` to override
    the config polling interval.
    """
    # Parse CLI args minimally (don't pull in Typer for a daemon process).
    once_flag = "--once" in sys.argv

    interval_override = None
    for i, arg in enumerate(sys.argv):
        if arg == "--interval" and i + 1 < len(sys.argv):
            try:
                interval_override = int(sys.argv[i + 1])
            except ValueError:
                pass

    if interval_override is not None:
        # Override env for the config loader to pick up.
        os.environ["RAG_KIT_WATCH_INTERVAL"] = str(interval_override)

    _logger.info("rag-kit watcher starting")
    result = scan_and_ingest(once=once_flag)

    if once_flag:
        print(json.dumps(result, indent=2))
    else:
        # The loop never exits; this is reached only if scan_and_ingest
        # returns (e.g., via error or once=True).
        _logger.info("Watcher exited")


if __name__ == "__main__":
    main()
