"""Tests for the Windows autostart generator (self-healing hidden chain).

Regression guard for 2026-10-22: the old Windows path was a one-shot
`schtasks /SC ONLOGON rag watch` — denied to non-elevated users on Win11 and
with no restart semantics, so the watcher died silently overnight. The
generator must now emit a startup .vbs + restart-looping .bat instead.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rag_kit import autostart


@pytest.fixture()
def fake_env(tmp_path, monkeypatch):
    """Fake venv layout + APPDATA so nothing touches the real machine."""
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    fake_exe = scripts / "rag.exe"
    fake_exe.write_text("fake")
    appdata = tmp_path / "AppData"
    (appdata / r"Microsoft\Windows\Start Menu\Programs\Startup").mkdir(parents=True)
    monkeypatch.setenv("APPDATA", str(appdata))
    calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: calls.append(list(cmd))
    )
    return tmp_path, fake_exe, appdata, calls


@pytest.mark.skipif(not hasattr(autostart, "_setup_windows"), reason="no windows impl")
def test_windows_generator_emits_self_healing_chain(fake_env, monkeypatch):
    tmp_path, fake_exe, appdata, calls = fake_env
    monkeypatch.setattr(autostart, "_find_rag_exe", lambda: str(fake_exe))

    ok, msg = autostart._setup_windows("RagKitWatcher")

    assert ok, msg
    autostart_dir = tmp_path / "venv" / "autostart"
    bat = (autostart_dir / "watch_loop.bat").read_text()
    vbs = (autostart_dir / "hidden_watcher.vbs").read_text()

    # restart semantics live in the .bat: label, log, backoff, loop
    assert ":loop" in bat
    assert "rag.exe\" watch" in bat or str(fake_exe) in bat
    assert "watch.log" in bat
    assert "goto loop" in bat
    # launcher hides the console and points at the bat
    assert "Wscript.Shell" in vbs
    assert ", 0, False" in vbs
    assert "watch_loop.bat" in vbs
    # startup folder copy = the logon hook (no elevation needed)
    startup_vbs = (
        appdata / r"Microsoft\Windows\Start Menu\Programs\Startup" / "RagKitWatcher.vbs"
    )
    assert startup_vbs.exists()
    # legacy scheduled task removed, watcher started immediately
    assert any("/delete" in c for c in calls)
    assert any(c and c[0].lower().endswith("wscript.exe") for c in calls)


@pytest.mark.skipif(not hasattr(autostart, "_setup_windows"), reason="no windows impl")
def test_no_legacy_onlogon_task_created(fake_env, monkeypatch):
    """The broken /SC onlogon creation path must be gone entirely."""
    _, fake_exe, _, calls = fake_env
    monkeypatch.setattr(autostart, "_find_rag_exe", lambda: str(fake_exe))
    autostart._setup_windows("RagKitWatcher")
    # exact-token check: the tmp_path contains the test name ("...onlogon...")
    assert not any(arg == "onlogon" or arg == "/SC" for c in calls for arg in c)
