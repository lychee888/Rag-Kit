"""Per-folder single-instance lock for the state-change watcher.

Allows ``rag watch <folder>`` to be started by both the installer AND the
boot-time autostart without two watchers racing on the same folder (which
would double-ingest). The first live process holding the lock wins; every
later start on the same folder exits cleanly.

Locks live under ``~/.cache/rag-kit/watcher-locks/`` keyed by the hashed,
fully-resolved folder path, and store the owning PID.
"""

from __future__ import annotations

import hashlib
import errno
import os
from pathlib import Path

_LOCK_DIR = Path.home() / ".cache" / "rag-kit" / "watcher-locks"
_HANDLES: dict[Path, object] = {}


def lock_path(folder: Path) -> Path:
    """Return the lockfile path for a watched *folder*."""
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(os.path.normcase(str(folder.resolve())).encode()).hexdigest()[:16]
    return _LOCK_DIR / f"{digest}.lock"


def acquire(folder: Path) -> Path | None:
    """Hold an OS lock until release/process exit; None means already owned.

    Files stay in place to avoid inode races. I/O failures raise rather than
    allowing an unlocked watcher to run.
    """
    lock = lock_path(folder)
    if lock in _HANDLES:
        return None
    handle = open(lock, "a+b")
    try:
        if lock.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return None
        raise
    try:
        handle.truncate(0)
        handle.write(str(os.getpid()).encode())
        handle.flush()
        _HANDLES[lock] = handle
    except Exception:
        handle.close()
        raise
    return lock


def release(lock: Path | None) -> None:
    """Release the lock if we still own it."""
    handle = _HANDLES.pop(lock, None)
    if handle is not None:
        handle.close()
