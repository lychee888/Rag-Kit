"""
Poll-based folder watcher — scan the ingest directory for new, changed, or
deleted files on a fixed interval.

Unlike the watchdog-based ``rag watch`` command (which uses OS-level file
events), this module polls the filesystem every *WATCH_INTERVAL* seconds.
It's designed to run as a lightweight background daemon started via
``pythonw -m rag_kit.watcher`` on Windows or ``nohup rag watch &
on Linux.

State is tracked in a JSON file (~/.rag-kit-watcher-state.json) so the
watcher can survive restarts without re-ingesting unchanged files.

Usage:
    python -m rag_kit.watcher          # Run once with defaults
    python -m rag_kit.watcher --interval 30  # Poll every 30 seconds
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

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
# State tracking
# --------------------------------------------------------------------------- #

def _state_path() -> Path:
    """Return the path to the watcher state file."""
    return Path.home() / ".rag-kit-watcher-state.json"


def _load_state() -> dict[str, Any]:
    """Load persisted watcher state (file_path → md5_hash)."""
    sp = _state_path()
    if not sp.exists():
        return {}
    try:
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("files", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    """Persist watcher state to disk."""
    sp = _state_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"files": state, "updated_at": time.time()}, f, indent=2)


def _file_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file's contents."""
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Main scan loop
# --------------------------------------------------------------------------- #


def scan_and_ingest(once: bool = False) -> dict[str, Any]:
    """Scan the configured ingest directory and process any changes.

    Args:
        once:   If True, run one scan and return.  If False, loop forever
                (the daemon mode).

    Returns:
        Summary dict with keys ``new``, ``changed``, ``deleted``, ``errors``.
    """
    from rag_kit.config import get_config

    try:
        config = get_config()
    except Exception as exc:
        _logger.error("Failed to load config: %s", exc)
        return {"new": 0, "changed": 0, "deleted": 0, "errors": [str(exc)]}

    ingest_dir = Path(config.watch_folder)
    interval = config.watch_interval
    supported_exts = set(config.supported_extensions)

    # Lazy imports to avoid loading heavy modules at import time.
    from rag_kit.embed import create_engine_from_config
    from rag_kit.ingest import SUPPORTED_EXTENSIONS, ingest_file
    from rag_kit.store import create_store_from_config

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

    while True:
        summary: dict[str, Any] = {"new": 0, "changed": 0, "deleted": 0, "errors": []}

        # --- Discover files on disk ---
        disk_files: dict[str, str] = {}  # abs_path → md5
        if ingest_dir.exists():
            for ext in supported_exts:
                if ext not in SUPPORTED_EXTENSIONS:
                    # Avoid ingesting unsupported extensions even if misconfigured.
                    continue
                for f in ingest_dir.rglob(f"*{ext}"):
                    if f.is_file():
                        abs_path = str(f.resolve())
                        disk_files[abs_path] = _file_md5(f)

        # --- Load previous state ---
        prev_state = _load_state()

        # --- Detect new and changed files ---
        to_ingest: list[str] = []
        for abs_path, md5_hash in disk_files.items():
            if md5_hash == "":  # unreadable
                continue
            if abs_path not in prev_state:
                to_ingest.append(abs_path)
                summary["new"] += 1
            elif prev_state[abs_path] != md5_hash:
                to_ingest.append(abs_path)
                summary["changed"] += 1

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
                summary["errors"].append(str(exc))
                continue

            # Convert to store format.
            store_chunks: list[dict[str, Any]] = []
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

            # Update MD5 hash.
            prev_state[abs_path] = _file_md5(Path(abs_path))

        # --- Persist updated state ---
        if to_ingest or summary["deleted"] > 0:
            _save_state(prev_state)

        if summary["new"] > 0 or summary["changed"] > 0 or summary["deleted"] > 0:
            _logger.info(
                "Scan summary: new=%d changed=%d deleted=%d errors=%d",
                summary["new"], summary["changed"], summary["deleted"], len(summary["errors"]),
            )
        else:
            _logger.debug("No changes detected")

        if once:
            return summary

        time.sleep(interval)


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

    interval_override: int | None = None
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