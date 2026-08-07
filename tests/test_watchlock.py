"""Tests for the watcher single-instance lock (watchlock)."""

from pathlib import Path

from rag_kit import watchlock


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
    try:
        lb = watchlock.acquire(b)
        assert lb is not None, "different folders use different locks"
    finally:
        watchlock.release(la)


def test_acquire_generates_deterministic_path():
    f = Path("C:\\tmp\\folder")
    p1 = watchlock.lock_path(f)
    p2 = watchlock.lock_path(f)
    assert p1 == p2
    assert p1.name.endswith(".lock")
