"""Send keystrokes to a Claude Code CLI running inside tmux (macOS / Linux).

Same API as console_input.ConsoleKeys (write / send / read_screen), so InteractiveAdapter works unchanged.
Run `claude` inside a tmux pane; the voice app finds that pane by itself (or set `tmux_pane` in config.json).
"""
from __future__ import annotations

import shutil
import subprocess

_KEYS = {"enter": "Enter", "esc": "Escape", "shift+tab": "BTab", "tab": "Tab"}


def _tmux(*args: str, timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=True, text=True, timeout=timeout)


def pick_pane(rows: list[str]) -> str:
    """From `pane_id<TAB>current_command<TAB>pane_pid` rows, the pane running the Claude Code CLI ("" if none)."""
    for row in rows:
        parts = row.split("\t")
        if len(parts) >= 2 and parts[1].strip().lower() in ("claude", "node"):
            return parts[0].strip()
    return ""


def find_cli_pane() -> str:
    if not shutil.which("tmux"):
        return ""
    try:
        out = _tmux("list-panes", "-a", "-F", "#{pane_id}\t#{pane_current_command}\t#{pane_pid}")
    except (OSError, subprocess.SubprocessError):
        return ""
    return pick_pane(out.stdout.splitlines()) if out.returncode == 0 else ""


class TmuxKeys:
    def __init__(self, pane: str):
        self.pane = pane

    def write(self, text: str) -> None:
        text = text.replace("\r", " ").replace("\n", " ")
        r = _tmux("send-keys", "-t", self.pane, "-l", "--", text)  # -l: literal, never parsed as key names
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "tmux send-keys failed")

    def send(self, combo: str) -> None:
        r = _tmux("send-keys", "-t", self.pane, _KEYS[combo])
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "tmux send-keys failed")

    def read_screen(self) -> list[str]:
        try:
            r = _tmux("capture-pane", "-p", "-t", self.pane)
        except (OSError, subprocess.SubprocessError):
            return []
        rows = [line.rstrip() for line in r.stdout.splitlines()] if r.returncode == 0 else []
        return [line for line in rows if line][-12:]
