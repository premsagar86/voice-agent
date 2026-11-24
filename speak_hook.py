"""Claude Code hook: speak a short summary of Claude's reply (Windows, stdlib only).

Usage (configured in ~/.claude/settings.json):
    python speak_hook.py stop      # Stop hook: Claude finished a reply
    python speak_hook.py notify    # Notification hook: Claude needs attention

The hook itself returns immediately: it extracts the text, hands it to a
detached worker (`_speak`) that summarizes and speaks, so Claude Code is
never blocked by the (slow) summarizer or by audio playback.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_agent.audio.tts import _NO_WINDOW, speak  # noqa: E402,F401
from voice_agent.config import GUARD_ENV, load_config, log  # noqa: E402,F401
from voice_agent.speech.filters import clean_for_speech  # noqa: E402,F401
from voice_agent.speech.summarizer import speech_text, summarize  # noqa: E402,F401


# ---------------------------------------------------------------- extraction
def _tool_text(block: dict) -> str:
    """Speakable text hidden in tool calls: the plan shown for approval, or a question to the user."""
    inp = block.get("input") or {}
    if block.get("name") == "ExitPlanMode":
        return str(inp.get("plan", "")).strip()
    if block.get("name") == "AskUserQuestion":
        qs = inp.get("questions") or []
        return "\n".join(str(q.get("question", "")) for q in qs if isinstance(q, dict)).strip()
    return ""


def _text_of(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out = []
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                out.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                out.append(_tool_text(b))
        return "\n".join(t for t in out if t).strip()
    return ""


def last_assistant_text(transcript_path: str, whole_turn: bool = True) -> str:
    """Text of the current turn: every assistant text message after the last user prompt, in order.

    whole_turn=False returns only the final assistant message that has text.
    """
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    parts: list[str] = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        msg = entry.get("message") if isinstance(entry, dict) else None
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            if _text_of(msg):  # a real prompt (tool results have no text) ends the turn
                break
            continue
        if msg.get("role") != "assistant":
            continue
        text = _text_of(msg)
        if text:
            parts.append(text)
            if not whole_turn:
                break
    return "\n\n".join(reversed(parts))


def plan_in_turn(transcript_path: str) -> str:
    """The plan shown for approval (ExitPlanMode) in the current turn, or "" if there is none."""
    try:
        lines = Path(transcript_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        try:
            msg = (json.loads(line) or {}).get("message")
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            if _text_of(msg):
                break
            continue
        content = msg.get("content")
        for b in content if isinstance(content, list) else []:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "ExitPlanMode":
                return _tool_text(b)
    return ""


def pick_speech(transcript_path: str) -> tuple[str, str]:
    """One thing per event: the plan (alone) if one was presented, else the final message of the turn."""
    plan = plan_in_turn(transcript_path)
    if plan:
        return "plan", plan
    return "done", last_assistant_text(transcript_path, whole_turn=False)


# ---------------------------------------------------------------------- main
def _stop_controls(cfg: dict):
    """F10 and the spoken word "stop" both silence this speech. Best effort: never raises."""
    import contextlib
    from voice_agent.audio import mic
    from voice_agent.audio.tts import stop_speaking

    stack = contextlib.ExitStack()
    try:
        stack.enter_context(mic.StopHotkey(stop_speaking, cfg["stop_hotkey"]))
        if cfg.get("stop_listen_hook"):  # off by default: loads Whisper per reply, hears its own speaker
            from voice_agent.stt.whisper import Transcriber
            stack.enter_context(mic.StopWordMonitor(
                stop_speaking, Transcriber(cfg["stop_whisper_model"], cfg["whisper_compute_type"]).transcribe,
                cfg["stop_threshold"]))
    except Exception as e:
        log(f"stop controls unavailable: {e!r}")
    return stack


def worker(text_file: str, kind: str) -> None:
    cfg = load_config()
    try:
        text = Path(text_file).read_text(encoding="utf-8")
        os.remove(text_file)
    except OSError:
        return
    if kind == "notify":
        spoken = text
    elif kind == "plan":  # key points, unless you said "full" (then the whole plan is read)
        spoken = speech_text(text, cfg) if cfg["read_mode"] == "full" else summarize(text, dict(cfg, read_mode="keypoints"))
    else:  # task finished: "follow" (default) obeys the spoken read mode; "summary"/"full" force one
        done = cfg.get("done_mode", "follow")
        done = cfg["read_mode"] if done == "follow" else done
        if done == "full":
            spoken = speech_text(text, dict(cfg, read_mode="full"))
        else:
            spoken = summarize(text, dict(cfg, read_mode="keypoints" if done == "keypoints" else "summary"))
    if spoken:
        with _stop_controls(cfg):
            speak(spoken, cfg)


def hook(kind: str) -> None:
    if os.environ.get(GUARD_ENV):
        return
    cfg = load_config()
    if not cfg["enabled"]:
        return
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        payload = {}
    if payload.get("stop_hook_active"):
        return
    if kind == "stop":
        kind, text = pick_speech(payload.get("transcript_path", ""))
    elif kind == "notify" and cfg["speak_notifications"]:
        text = clean_for_speech(str(payload.get("message", "")), cfg["max_chars"])
    else:
        text = ""
    if not text:
        return
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="va_in_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "_speak", kind, path],
        creationflags=_NO_WINDOW | getattr(subprocess, "DETACHED_PROCESS", 0),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def main(argv: list[str]) -> int:
    try:
        if len(argv) >= 4 and argv[1] == "_speak":
            worker(argv[3], argv[2])
        elif len(argv) >= 2 and argv[1] == "hush":  # UserPromptSubmit: silence old speech, note the mode
            from voice_agent.audio.tts import stop_speaking
            stop_speaking()
            try:
                mode = json.loads(sys.stdin.read() or "{}").get("permission_mode", "")
                if mode:
                    from voice_agent.agents.interactive import save_state
                    save_state(mode)
            except (ValueError, OSError):
                pass
        elif len(argv) >= 2 and argv[1] == "start":  # SessionStart: bring the voice app up if it is not running
            cfg = load_config()
            if cfg["enabled"] and cfg["auto_start"] and not os.environ.get(GUARD_ENV):
                try:
                    info = json.loads(sys.stdin.read() or "{}")
                except ValueError:
                    info = {}
                cwd = info.get("cwd") or None
                if info.get("source") != "compact":  # a new/cleared/resumed session: wake word needed again
                    try:
                        from voice_agent.config import SESSION_PATH
                        SESSION_PATH.write_text(str(time.time()), encoding="utf-8")
                    except OSError:
                        pass
                from voice_agent import lifecycle
                lifecycle.start(cwd)
        elif len(argv) >= 2 and argv[1] in ("stop", "notify"):
            hook(argv[1])
    except Exception as e:  # a hook must never break Claude Code
        log(f"error: {e!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
