"""Shared paths, config loading and logging."""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
LOG_PATH = ROOT / "speak_hook.log"
PID_PATH = ROOT / ".speaking.pid"
SESSIONS_PATH = ROOT / "sessions.json"
MODE_PATH = ROOT / "mode.json"  # spoken-mode choice shared by the app and the Stop hook
SESSION_PATH = ROOT / "session.stamp"  # touched by the SessionStart hook: a new CLI session closes the open conversation
MUTED_PATH = ROOT / "voice.muted"  # exists after `voice stop`: hooks stay silent and nothing auto-starts
VALID_MODES = ("full", "summary", "keypoints")
GUARD_ENV = "VOICE_AGENT_SUMMARIZING"  # set in child processes so the Stop hook stays silent

DEFAULT_CONFIG = {
    "enabled": True,
    "auto_start": True,  # SessionStart hook launches the voice app whenever a Claude CLI session starts
    "read_mode": "full",  # "full" = whole reply start to end, "summary" = 1-3 sentences, "keypoints" = key points
    "listen_mode": "push",  # "push" = hold the hotkey, "auto" = hands-free (detects when you speak)
    "vad_threshold": 0.02,  # hands-free: mic level that counts as speech
    "vad_silence": 1.0,  # hands-free: seconds of quiet that end an utterance
    "barge_in": False,  # speaking over the voice stops it (use headphones)
    "barge_in_threshold": 0.08,
    "wake_word": "",  # hands-free only: e.g. "hey claude"; empty = every utterance counts
    "wake_timeout": -1,  # after the wake word: -1 = open until the session ends, N = seconds of quiet, 0 = say it every time
    "persona": "",  # "jarvis" = spoken summaries in a calm, dry, butler-like voice; empty = plain
    "target": "interactive",  # "interactive" = type into the focused Claude Code CLI, "headless" = hidden claude -p
    "cli_pid": 0,  # interactive target: PID of the Claude CLI (0 = find it automatically)
    "tmux_pane": "",  # macOS/Linux: tmux pane id running Claude (e.g. "%3"); empty = find it automatically
    "stop_hotkey": "f10",  # global key: silence speech instantly (most reliable stop)
    "stop_whisper_model": "tiny.en",  # small/fast model used by the hook to hear "stop"
    "done_mode": "follow",  # when a task finishes: "follow" = obey the read mode (say "full" / "brief"), or force "summary" / "keypoints" / "full"
    "stop_listen_hook": False,  # hook speech: also listen for the word "stop" (F10 always works)
    "stop_listen": True,  # saying "stop" while it works/speaks silences and cancels (use headphones)
    "stop_threshold": 0.03,  # mic level that triggers a check for the word "stop"
    "guard_destructive": True,  # risky requests need a spoken "confirm"
    "read_code": False,  # full mode: True reads code blocks aloud, False says "Code block skipped"
    "summarizer": "claude",  # "claude" (claude -p, uses your login) or "local"
    "summarizer_model": "haiku",
    "summarizer_timeout": 25,
    "max_chars": 300,
    "rate": 1,  # SAPI rate, -10..10
    "voice": "",  # empty = system default
    "speak_notifications": True,
    "hotkey": "f9",
    "whisper_model": "base.en",
    "whisper_compute_type": "int8",
    "agent_timeout": 600,
}


def log(msg: str) -> None:
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    mode = load_mode()
    if mode:
        cfg["read_mode"] = mode  # the spoken choice wins over config.json
    if MUTED_PATH.exists():
        cfg["enabled"] = False  # `voice stop` mutes the hooks too, not just the app
    return cfg


def load_mode() -> str:
    try:
        mode = json.loads(MODE_PATH.read_text(encoding="utf-8")).get("read_mode", "")
    except (OSError, ValueError, AttributeError):
        return ""
    return mode if mode in VALID_MODES else ""


def save_mode(mode: str) -> None:
    if mode in VALID_MODES:
        MODE_PATH.write_text(json.dumps({"read_mode": mode}), encoding="utf-8")
