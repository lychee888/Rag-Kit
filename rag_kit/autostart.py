"""
Autostart support — configure the rag-kit watcher to start on boot.

Provides:
    setup_autostart() → (bool, str)
        Returns (success, message). Sets up a scheduler task (Windows)
        or systemd user service (Linux) that runs ``rag watch`` on startup.

    is_autostart_installed() → bool
        Checks whether the autostart entry exists.

    remove_autostart() → (bool, str)
        Removes the autostart entry.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


def _find_rag_exe() -> str | None:
    """Return the absolute path to the rag/rag-kit CLI executable."""
    # In a venv, sys.executable's parent/Scripts/rag.exe (Windows) or bin/rag (Linux).
    venv_dir = Path(sys.executable).parent

    candidates = [
        venv_dir / "rag.exe",
        venv_dir / "rag",
        venv_dir / "rag-kit.exe",
        venv_dir / "rag-kit",
    ]

    for c in candidates:
        if c.is_file():
            return str(c.resolve())

    # Fallback: search PATH.
    import shutil
    for name in ("rag.exe", "rag", "rag-kit.exe", "rag-kit"):
        found = shutil.which(name)
        if found:
            return found

    return None


def setup_autostart(
    interval: int = 30,
    task_name: str = "rag-kit-watcher",
) -> tuple[bool, str]:
    """Configure the rag-kit watcher to start automatically on boot.

    Windows:
        Creates a scheduled task via ``schtasks`` that runs on user logon.
        Task: ``rag-kit-watcher`` — runs ``rag watch <watch_folder>``.

    Linux:
        Creates a systemd user service at
        ``~/.config/systemd/user/rag-kit-watcher.service`` and enables it
        with ``loginctl enable-linger``.

    Args:
        interval:   Reserved (kept for backward compat). The watch command
                    is state-change based (watchdog) and needs no poll interval.
        task_name:  Name of the task/service.

    Returns:
        ``(success: bool, message: str)``
    """
    system = platform.system()

    if system == "Windows":
        return _setup_windows(task_name)
    elif system == "Linux":
        return _setup_linux(task_name)
    else:
        return False, f"Autostart not supported on {system}"


def is_autostart_installed(task_name: str = "rag-kit-watcher") -> bool:
    """Check whether the autostart entry exists on this system.

    Returns True if the scheduled task (Windows) or systemd service (Linux)
    is configured.
    """
    system = platform.system()

    if system == "Windows":
        try:
            result = subprocess.run(
                ["schtasks", "/query", "/tn", task_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    elif system == "Linux":
        service_file = (
            Path.home() / ".config" / "systemd" / "user" / f"{task_name}.service"
        )
        return service_file.exists()

    return False


def remove_autostart(task_name: str = "rag-kit-watcher") -> tuple[bool, str]:
    """Remove the autostart entry.

    Returns:
        ``(success: bool, message: str)``
    """
    system = platform.system()

    if system == "Windows":
        try:
            subprocess.run(
                ["schtasks", "/delete", "/tn", task_name, "/f"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return True, f"Removed Windows scheduled task '{task_name}'"
        except subprocess.CalledProcessError as exc:
            return False, f"Failed to remove task: {exc.stderr.strip() if exc.stderr else exc}"
        except Exception as exc:
            return False, f"Failed to remove task: {exc}"

    elif system == "Linux":
        service_file = (
            Path.home() / ".config" / "systemd" / "user" / f"{task_name}.service"
        )
        if not service_file.exists():
            return True, f"No service file at {service_file} (already removed)"

        # Stop and disable the service.
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", task_name],
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["systemctl", "--user", "disable", task_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass  # Service may not be running — safe to ignore.

        service_file.unlink(missing_ok=True)

        # Reload systemd.
        try:
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass

        return True, f"Removed systemd user service '{task_name}'"

    return False, f"Autostart removal not supported on {system}"


# --------------------------------------------------------------------------- #
# Windows implementation
# --------------------------------------------------------------------------- #


def _setup_windows(task_name: str) -> tuple[bool, str]:
    """Create a Windows scheduled task for the watcher."""

    rag_exe = _find_rag_exe()
    if rag_exe is None:
        return (
            False,
            "Could not find rag.exe. Ensure rag-kit is installed in a venv.",
        )

    # Run the state-change watcher on the configured watch folder.
    command = f'"{rag_exe}" watch'

    try:
        # First, delete any existing task with the same name.
        subprocess.run(
            ["schtasks", "/delete", "/tn", task_name, "/f"],
            capture_output=True,
            timeout=10,
        )

        # Create the task.  /SC ONLOGON runs the task when the user logs in.
        # /RL HIGHEST requests elevation for the task (if admin).
        result = subprocess.run(
            [
                "schtasks",
                "/create",
                "/tn", task_name,
                "/tr", command,
                "/sc", "onlogon",
                "/rl", "limited",
                "/f",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode == 0:
            return True, f"Created Windows scheduled task '{task_name}' (runs on logon)"
        else:
            error = result.stderr.strip() if result.stderr else result.stdout.strip()
            # If we need admin rights, try with elevated prompt hint.
            if "access is denied" in error.lower() or "0x80070005" in error:
                return (
                    False,
                    f"Permission denied. Run as Administrator to create the scheduled task.\n"
                    f"Command: schtasks /create /tn {task_name} /tr \"{command}\" /sc onlogon /f",
                )
            return False, f"Failed to create task: {error}"

    except subprocess.TimeoutExpired:
        return False, "schtasks command timed out"
    except FileNotFoundError:
        return False, "schtasks.exe not found (not a supported Windows version?)"
    except Exception as exc:
        return False, f"Unexpected error: {exc}"


# --------------------------------------------------------------------------- #
# Linux implementation
# --------------------------------------------------------------------------- #


def _setup_linux(task_name: str) -> tuple[bool, str]:
    """Create a systemd user service for the watcher."""

    rag_exe = _find_rag_exe()
    if rag_exe is None:
        return (
            False,
            "Could not find rag executable. Ensure rag-kit is installed in a venv.",
        )

    # Ensure systemd user directory exists.
    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)

    service_file = service_dir / f"{task_name}.service"

    service_content = f"""[Unit]
Description=rag-kit watcher — auto-ingest documents
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={rag_exe} watch
Restart=on-failure
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""

    try:
        service_file.write_text(service_content)

        # Enable linger so the user service starts at boot (even before login).
        import getpass
        subprocess.run(
            ["loginctl", "enable-linger", getpass.getuser()],
            capture_output=True,
            timeout=10,
        )

        # Reload systemd and enable the service.
        subprocess.run(
            ["systemctl", "--user", "daemon-reload"],
            capture_output=True,
            timeout=10,
            check=True,
        )
        subprocess.run(
            ["systemctl", "--user", "enable", task_name],
            capture_output=True,
            timeout=10,
            check=True,
        )
        subprocess.run(
            ["systemctl", "--user", "start", task_name],
            capture_output=True,
            timeout=10,
            check=True,
        )

        return (
            True,
            f"Created and started systemd user service '{task_name}' "
            f"(autostart enabled via linger).",
        )

    except subprocess.CalledProcessError as exc:
        error = exc.stderr.decode() if exc.stderr else str(exc)
        return False, f"Failed to configure systemd service: {error}"
    except Exception as exc:
        return False, f"Unexpected error: {exc}"