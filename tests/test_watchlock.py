"""Tests for the watcher single-instance lock (watchlock)."""

from pathlib import Path
import pytest

from rag_kit import watchlock


@pytest.fixture(autouse=True)
def isolate_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(watchlock, '_LOCK_DIR', tmp_path / 'locks')


def test_lock_blocks_second_instance(tmp_path):
    folder = tmp_path / "watch"
    folder.mkdir()
    l1 = watchlock.acquire(folder)
    assert l1 is not None, "first watcher should acquire the lock"
    try:
        assert l1.exists()
        l2 = watchlock.acquire(folder)
        assert l2 is None, "second watcher on same folder should be denied"
    finally:
        watchlock.release(l1)


def test_lock_released_allows_reacquire(tmp_path):
    folder = tmp_path / "watch"
    folder.mkdir()
    l1 = watchlock.acquire(folder)
    assert l1 is not None
    watchlock.release(l1)
    l2 = watchlock.acquire(folder)
    assert l2 is not None, "after release the lock should be free again"
    watchlock.release(l2)


def test_lock_isolated_per_folder(tmp_path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    la = watchlock.acquire(a)
    assert la is not None
    lb = None
    try:
        lb = watchlock.acquire(b)
        assert lb is not None, "different folders use different locks"
    finally:
        watchlock.release(la)
        watchlock.release(lb)


def test_acquire_generates_deterministic_path():
    f = Path("C:\\tmp\\folder")
    p1 = watchlock.lock_path(f)
    p2 = watchlock.lock_path(f)
    assert p1 == p2
    assert p1.name.endswith(".lock")


def test_concurrent_processes_have_one_owner_and_crash_releases(tmp_path):
    import subprocess
    import sys
    folder = tmp_path / 'watch'
    folder.mkdir()
    script = '''
import os, sys
from pathlib import Path
from rag_kit import watchlock
watchlock._LOCK_DIR = Path(sys.argv[2])
lock = watchlock.acquire(Path(sys.argv[1]))
print(int(lock is not None), flush=True)
if lock:
    sys.stdin.readline()
    os._exit(0)  # Exit without running release/finally handlers.
'''
    processes = [subprocess.Popen([sys.executable, '-c', script, str(folder), str(tmp_path / 'locks')],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(6)]
    try:
        outcomes = [int(p.stdout.readline().strip()) for p in processes]
        assert sum(outcomes) == 1
        winner = processes[outcomes.index(1)]
        winner.communicate(input='exit\n', timeout=5)
        lock = watchlock.acquire(folder)
        assert lock is not None
        watchlock.release(lock)
    finally:
        for p in processes:
            if p.poll() is None:
                p.communicate(input='exit\n', timeout=5)
