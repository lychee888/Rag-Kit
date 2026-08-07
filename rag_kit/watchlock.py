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
import os
import sys
from pathlib import Path

_LOCK_DIR = Path.home() / ".cache" / "rag-kit" / "watcher-locks"


def lock_path(folder: Path) -> Path:
    """Return the lockfile path for a watched *folder*."""
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:16]
    return _LOCK_DIR / f"{digest}.lock"


def _pid_alive(pid: int) -> bool:
    """Best-effort cross-platform liveness check."""
    if sys.platform == "win32":
        try:
            import ctypes

            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            return True  # can't tell; assume alive
    # POSIX
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True  # exists (maybe different user)
    except Exception:
        return True


def acquire(folder: Path) -> Path | None:
    """Take the lock for *folder*. Returns the lock path, or ``None`` if a
    live watcher already holds it (stale locks are reclaimed).
    """
    lock = lock_path(folder)
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except (ValueError, OSError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            return None  # already being watched by a live process
        try:
            lock.unlink()  # stale
        except OSError:
            pass
    try:
        lock.write_text(str(os.getpid()))
    except OSError:
        pass  # lock dir may be read-only; don't block watching on lock failure
    return lock


def release(lock: Path | None) -> None:
    """Release the lock if we still own it."""
    if lock is None:
        return
    try:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass
