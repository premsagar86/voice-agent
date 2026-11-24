"""Inject keystrokes into another process's console, without needing keyboard focus (Windows).

Runs as a tiny helper process because AttachConsole detaches the caller from its own console:
    python -m voice_agent.agents.console_input <pid> '<json list of ["text"|"key", value]>'
`find_cli_pid()` and `ConsoleKeys` are the parent-side API used by the interactive adapter.
"""
from __future__ import annotations

import json
import subprocess
import sys

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)  # helper must start with no console at all
_SKIP_ARGS = ("--chrome-native-host", " -p", "--print", "--output-format", "--mcp", "--no-session-persistence")


_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])
_FOOTER = __import__("re").compile(r"(auto mode|plan mode|accept edits|bypass permissions) on", __import__("re").I)
_FOOTER_NAMES = {"auto mode": "auto", "plan mode": "plan", "accept edits": "acceptEdits",
                 "bypass permissions": "bypassPermissions"}


def mode_from_screen(rows: list[str]) -> str | None:
    """Permission mode shown in the CLI footer; "default" when no mode is shown; None if unreadable."""
    if not rows:
        return None
    for row in reversed(rows[-4:]):  # the footer sits at the bottom of the screen
        m = _FOOTER.search(row)
        if m:
            return _FOOTER_NAMES[m.group(1).lower()]
    return "default"


# ------------------------------------------------------------------ parent side
def pick_cli_pid(procs: list[dict]) -> int:
    """From Win32_Process rows, the interactive Claude Code CLI (not hosts or headless `-p` runs)."""
    ok = []
    for p in procs:
        cmd = str(p.get("CommandLine") or "")
        name = str(p.get("Name") or "").lower()
        if name == "node.exe" and "claude" not in cmd.lower():
            continue
        if any(a in f" {cmd}" for a in _SKIP_ARGS):
            continue
        ok.append(int(p["ProcessId"]))
    return max(ok) if ok else 0  # newest-looking (highest) PID


def find_cli_pid() -> int:
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'claude.exe','node.exe' } | "
          "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=20, creationflags=_NO_WINDOW).stdout
        rows = json.loads(out or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0
    return pick_cli_pid([rows] if isinstance(rows, dict) else rows)


class ConsoleKeys:
    """Keys sent straight to the CLI's console. Same write/send API as the keyboard fallback."""

    def __init__(self, pid: int):
        self.pid = pid

    def _retarget(self) -> bool:
        """The CLI was restarted: find its new PID. True if the target changed."""
        new = find_cli_pid()
        if new and new != self.pid:
            self.pid = new
            return True
        return False

    def _run(self, actions: list, retry: bool = True) -> None:
        r = subprocess.run([sys.executable, "-m", "voice_agent.agents.console_input", str(self.pid),
                            json.dumps(actions)], capture_output=True, text=True, timeout=20,
                           creationflags=_DETACHED, cwd=_ROOT)
        if r.returncode != 0:
            if retry and r.returncode == 2 and self._retarget():  # stale PID (AttachConsole error)
                return self._run(actions, retry=False)
            raise RuntimeError((r.stdout + r.stderr).strip() or f"console input failed ({r.returncode})")

    def read_screen(self, retry: bool = True) -> list[str]:
        """Last visible non-empty rows of the CLI's screen ([] if unreadable)."""
        r = subprocess.run([sys.executable, "-m", "voice_agent.agents.console_input", str(self.pid),
                            json.dumps([["read"]])], capture_output=True, text=True, encoding="utf-8",
                           timeout=20, creationflags=_DETACHED, cwd=_ROOT)
        if retry and r.returncode == 2 and self._retarget():
            return self.read_screen(retry=False)
        try:
            return list(json.loads(r.stdout)) if r.returncode == 0 else []
        except ValueError:
            return []

    def write(self, text: str) -> None:
        self._run([["text", text]])

    def send(self, combo: str) -> None:
        self._run([["key", combo]])


# ------------------------------------------------------------------ helper side
def _records(actions: list):
    import ctypes
    from ctypes import wintypes

    class KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [("bKeyDown", wintypes.BOOL), ("wRepeatCount", wintypes.WORD),
                    ("wVirtualKeyCode", wintypes.WORD), ("wVirtualScanCode", wintypes.WORD),
                    ("uChar", wintypes.WCHAR), ("dwControlKeyState", wintypes.DWORD)]

    class _U(ctypes.Union):
        _fields_ = [("KeyEvent", KEY_EVENT_RECORD)]

    class INPUT_RECORD(ctypes.Structure):
        _anonymous_ = ("Event",)
        _fields_ = [("EventType", wintypes.WORD), ("Event", _U)]

    def rec(vk: int, ch: str, down: bool, state: int):
        r = INPUT_RECORD()
        r.EventType = 0x0001
        r.KeyEvent = KEY_EVENT_RECORD(down, 1, vk, 0, ch, state)
        return r

    keys = {"enter": (0x0D, "\r", 0), "esc": (0x1B, "\x1b", 0), "shift+tab": (0x09, "\t", 0x10), "tab": (0x09, "\t", 0)}
    out = []
    for kind, value in actions:
        if kind == "key":
            vk, ch, st = keys[value]
            out += [rec(vk, ch, True, st), rec(vk, ch, False, st)]
        else:
            for ch in value:
                if ch in "\r\n":
                    continue
                scan = ctypes.windll.user32.VkKeyScanW(ord(ch))
                vk, st = (scan & 0xFF), (0x10 if scan & 0x100 else 0)
                vk = 0 if vk == 0xFF else vk
                out += [rec(vk, ch, True, st), rec(vk, ch, False, st)]
    return INPUT_RECORD, out


def _read_screen(k, pid: int) -> int:
    """Print the last non-empty visible rows of the target console as JSON."""
    import ctypes
    from ctypes import wintypes

    class COORD(ctypes.Structure):
        _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

    class SMALL_RECT(ctypes.Structure):
        _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short),
                    ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

    class CSBI(ctypes.Structure):
        _fields_ = [("dwSize", COORD), ("dwCursorPosition", COORD), ("wAttributes", wintypes.WORD),
                    ("srWindow", SMALL_RECT), ("dwMaximumWindowSize", COORD)]

    k.FreeConsole()
    if not k.AttachConsole(pid):
        return 2
    h = k.CreateFileW("CONOUT$", 0xC0000000, 3, None, 3, 0, None)
    info = CSBI()
    if not k.GetConsoleScreenBufferInfo(h, ctypes.byref(info)):
        k.FreeConsole()
        return 3
    rows = []
    for y in range(info.srWindow.Top, info.srWindow.Bottom + 1):
        buf = ctypes.create_unicode_buffer(info.dwSize.X + 1)
        n = wintypes.DWORD(0)
        k.ReadConsoleOutputCharacterW(h, buf, info.dwSize.X, COORD(0, y), ctypes.byref(n))
        rows.append(buf.value.rstrip())
    k.CloseHandle(h)
    k.FreeConsole()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps([r for r in rows if r][-6:], ensure_ascii=False))
    return 0


def main(argv: list[str]) -> int:
    import ctypes
    from ctypes import wintypes

    pid, actions = int(argv[1]), json.loads(argv[2])
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = wintypes.HANDLE
    if actions and actions[0][0] == "read":
        return _read_screen(k, pid)
    record_type, records = _records(actions)
    k.FreeConsole()
    if not k.AttachConsole(pid):
        print(f"AttachConsole({pid}) failed: error {ctypes.get_last_error()}")
        return 2
    h = k.CreateFileW("CONIN$", 0xC0000000, 3, None, 3, 0, None)
    arr = (record_type * len(records))(*records)
    written = wintypes.DWORD(0)
    ok = k.WriteConsoleInputW(h, arr, len(records), ctypes.byref(written))
    k.CloseHandle(h)
    k.FreeConsole()
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
