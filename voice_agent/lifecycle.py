"""Start/stop the voice app as one background instance (used by the SessionStart hook, `voice` command, /voice)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from .config import MUTED_PATH, ROOT

APP_PID_PATH = ROOT / "voice-app.pid"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)
_STILL_ACTIVE = 259


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)  # signal 0: existence check only
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes  # Windows only
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == _STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def running_pid() -> int:
    try:
        pid = int(APP_PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return 0
    return pid if pid_alive(pid) else 0


def register() -> None:
    """Called by the app itself at startup."""
    APP_PID_PATH.write_text(str(os.getpid()))


def unregister() -> None:
    try:
        if APP_PID_PATH.read_text().strip() == str(os.getpid()):
            APP_PID_PATH.unlink()
    except OSError:
        pass


def start(cwd: str | None = None, extra: list[str] | None = None) -> bool:
    """Launch the app hidden. False if one is already running."""
    try:
        MUTED_PATH.unlink()  # an explicit start un-mutes the hooks
    except OSError:
        pass
    if running_pid():
        return False
    log = open(ROOT / "voice-app.log", "a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, str(ROOT / "run.py"), "--background", *(extra or [])],
        cwd=cwd or str(Path.cwd()), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        creationflags=_NO_WINDOW | _DETACHED, close_fds=True,
        start_new_session=sys.platform != "win32")  # POSIX: survive the launching terminal
    return True


def _stray_pids() -> list[int]:
    """Other voice-app processes (e.g. a manual `python run.py`) that the pid file does not know about."""
    me = os.getpid()
    marker = str(ROOT / "run.py").lower().replace("\\", "/")
    found: list[int] = []
    if sys.platform == "win32":
        ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
              "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             creationflags=_NO_WINDOW).stdout
        rows = [line.split("\t", 1) for line in out.splitlines() if "\t" in line]
    else:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
        rows = [line.strip().split(None, 1) for line in out.splitlines() if line.strip()]
    for row in rows:
        if len(row) == 2 and row[0].strip().isdigit() and marker in row[1].lower().replace("\\", "/"):
            if int(row[0]) != me:
                found.append(int(row[0]))
    return found


def _kill(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=_NO_WINDOW)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def stop() -> bool:
    """Stop every voice-app process and mute the hooks. True if something was running."""
    try:
        MUTED_PATH.write_text("muted", encoding="utf-8")  # hook speech + auto-start stay off until `voice start`
    except OSError:
        pass
    pids = {p for p in [running_pid(), *_stray_pids()] if p}
    for pid in pids:
        _kill(pid)
    try:
        APP_PID_PATH.unlink()
    except OSError:
        pass
    return bool(pids)


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "start":
        print("Voice started." if start() else "Voice is already running.")
    elif cmd == "tray":
        print("Voice tray started." if start(extra=["--tray"]) else "Voice is already running.")
    elif cmd == "stop":
        print("Voice stopped." if stop() else "Voice was not running.")
    elif cmd == "restart":
        stop()
        start()
        print("Voice restarted.")
    else:
        pid = running_pid()
        print(f"Voice running (pid {pid})." if pid else "Voice is not running.")
    return 0
