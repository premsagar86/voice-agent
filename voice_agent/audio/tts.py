"""Text-to-speech: Windows SAPI (PowerShell), macOS `say`, Linux espeak-ng / espeak / spd-say.
Interruptible from this or another process."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile

from ..config import PID_PATH

_IS_WIN = sys.platform == "win32"
_SPEECH_PROGRAMS = {"say", "espeak", "espeak-ng", "spd-say"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_current: subprocess.Popen | None = None  # the PowerShell child speaking right now (this process)

_PS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.Rate = [int]$env:VA_RATE;"
    "if ($env:VA_VOICE) { try { $s.SelectVoice($env:VA_VOICE) } catch {} };"
    "$s.Speak([IO.File]::ReadAllText($env:VA_FILE, [Text.Encoding]::UTF8))"
)


def stop_speaking() -> None:
    """Stop speech: our own child process, and a previous process recorded in the pid file."""
    proc = _current
    if proc is not None and proc.poll() is None:
        proc.kill()
    try:
        pid = int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return
    if pid != os.getpid():
        try:
            if _IS_WIN:
                # The pid file holds the speaking *PowerShell child*. The image filter makes sure a recycled
                # pid (run.py, a terminal, anything else) can never be killed by mistake.
                subprocess.run(["taskkill", "/FI", f"PID eq {pid}", "/FI", "IMAGENAME eq powershell.exe", "/F"],
                               stdin=subprocess.DEVNULL, capture_output=True, creationflags=_NO_WINDOW)
            elif _is_speech_process(pid):
                os.kill(pid, signal.SIGTERM)
        except OSError:  # stopping speech must never crash the caller
            pass


def _is_speech_process(pid: int) -> bool:
    """POSIX: only kill the pid if it really is one of the speech programs (guards against pid reuse)."""
    try:
        name = subprocess.run(["ps", "-p", str(pid), "-o", "comm="], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return os.path.basename(name) in _SPEECH_PROGRAMS


def _speech_command(path: str, cfg: dict) -> tuple[list[str], dict]:
    """Per-OS command that reads `path` aloud, plus the environment it needs."""
    env = dict(os.environ)
    if _IS_WIN:
        env.update(VA_RATE=str(cfg["rate"]), VA_VOICE=cfg["voice"], VA_FILE=path)
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_SCRIPT], env
    wpm = str(max(80, min(400, 175 + int(cfg["rate"]) * 15)))  # config rate is -10..10 (SAPI scale)
    if sys.platform == "darwin":
        cmd = ["say", "-r", wpm, "-f", path]
        if cfg["voice"]:
            cmd[1:1] = ["-v", cfg["voice"]]
        return cmd, env
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            cmd = [exe, "-s", wpm, "-f", path]
            if cfg["voice"]:
                cmd[1:1] = ["-v", cfg["voice"]]
            return cmd, env
    if shutil.which("spd-say"):
        return ["spd-say", "--wait", "-e"], env  # reads text from stdin, see speak()
    raise RuntimeError("No text-to-speech program found. Install espeak-ng (Linux) or use macOS `say`.")


def speak(text: str, cfg: dict) -> None:
    """Speak `text` and block until it finishes (or stop_speaking() is called)."""
    global _current
    stop_speaking()
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="va_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        cmd, env = _speech_command(path, cfg)
        via_stdin = cmd[0] == "spd-say"
        _current = subprocess.Popen(
            cmd, env=env, stdin=subprocess.PIPE if via_stdin else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW,
        )
        if via_stdin:
            try:
                _current.stdin.write(text.encode("utf-8"))
                _current.stdin.close()
            except OSError:
                pass
        try:
            PID_PATH.write_text(str(_current.pid))  # the speaking child, not this process
        except OSError:
            pass
        try:
            _current.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            _current.kill()
    finally:
        _current = None
        try:
            os.remove(path)
        except OSError:
            pass
