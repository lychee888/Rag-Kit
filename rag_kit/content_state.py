"""Per-root content-hash state shared by every ingest surface.

Each watch folder (or explicit ``rag ingest`` target) owns a *bucket* file
under ``~/.cache/rag-kit/watcher-state/`` keyed by the hashed, fully
resolved root path.  A bucket maps absolute file path -> md5 content hash,
letting watchers skip re-extracting and re-embedding files whose bytes have
not changed.

Why per-root buckets instead of one global state file (audit F7): a scan of
folder A that sees folder B's paths in the shared state concludes those
files were *deleted* and wipes their store rows.  Namespacing state by root
makes each scan blind to other folders.  The legacy global file
(``~/.rag-kit-watcher-state.json``) is still read once for migration:
entries that live under the bucket's root are adopted; everything else is
left for its own folder to adopt.

Why md5: the legacy state files store md5 digests; keeping the algorithm
makes adoption incremental (unchanged files stay unchanged).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

# Overridable for tests (same pattern as watchlock._LOCK_DIR).
_STATE_DIR = Path.home() / ".cache" / "rag-kit" / "watcher-state"
LEGACY_STATE_PATH = Path.home() / ".rag-kit-watcher-state.json"


def _norm(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def bucket_key(root: str | Path) -> str:
    """Return the stable bucket digest for a watch/ingest root."""
    return hashlib.sha1(_norm(root).encode()).hexdigest()[:16]


def bucket_path(root: str | Path) -> Path:
    """Return the state file path for *root*'s bucket."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR / f"{bucket_key(root)}.json"


def file_hash(path: str | Path) -> str:
    """MD5 of a file's bytes, or ``""`` when unreadable."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def _files_of(data: Any) -> dict[str, str] | None:
    """Extract the ``{path: hash}`` mapping from a state payload."""
    if isinstance(data, dict):
        files = data.get("files", data)
        if isinstance(files, dict):
            return {str(k): str(v) for k, v in files.items()}
    return None


def _safe_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_state_file(path: str | Path) -> dict[str, str] | None:
    """Return the state mapping in *path*, or None when missing/corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    return _files_of(_safe_json(p))


def save_state_file(path: str | Path, state: dict[str, str],
                    root: str | Path | None = None) -> None:
    """Persist *state* to *path* atomically (tmp file + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    payload: dict[str, Any] = {"files": state, "updated_at": time.time()}
    if root is not None:
        payload["root"] = _norm(root)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, p)


def legacy_entries_under(root: str | Path) -> dict[str, str]:
    """Entries of the pre-F7 global state file that live under *root*."""
    if not LEGACY_STATE_PATH.exists():
        return {}
    legacy = _files_of(_safe_json(LEGACY_STATE_PATH)) or {}
    root_norm = _norm(root)
    return {
        path_str: digest
        for path_str, digest in legacy.items()
        if _norm(path_str) == root_norm
        or _norm(path_str).startswith(root_norm + os.sep)
    }


def load_bucket(root: str | Path) -> dict[str, str]:
    """Return ``{abs_path: md5}`` for *root*'s bucket (migrating if needed)."""
    stored = load_state_file(bucket_path(root))
    if stored is not None:
        return stored
    adopted = legacy_entries_under(root)
    if adopted:
        save_bucket(root, adopted)
    return adopted


def save_bucket(root: str | Path, state: dict[str, str]) -> None:
    """Persist *root*'s bucket."""
    save_state_file(bucket_path(root), state, root)
