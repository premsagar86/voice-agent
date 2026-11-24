"""Turn an agent message into 1-3 spoken sentences (Haiku via `claude -p`, or local fallback)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from ..config import GUARD_ENV, log
from .filters import clean_for_speech

SUMMARY_PROMPT = (
    "Rewrite the assistant message below so it can be spoken aloud to a developer "
    "who is not looking at the screen. Rules: 1 to 3 short plain sentences, no markdown, "
    "no code, no file paths read out in full (say the file name only), no symbols. "
    "Lead with the outcome. If the message asks the user a question or needs a decision, "
    "end by asking it. Output only the spoken text.\n\n--- MESSAGE ---\n"
)


KEYPOINTS_PROMPT = (
    "Turn the assistant message below into its key points, to be spoken aloud to a developer "
    "who is not looking at the screen. Say only the important points (outcome, decisions, "
    "next steps, any question for the user) in 2 to 5 short plain sentences. No markdown, "
    "no code, no symbols, no full file paths. Output only the spoken text.\n\n--- MESSAGE ---\n"
)


PERSONA_STYLES = {
    "jarvis": (
        "Speak in the style of JARVIS from Iron Man: calm, polished, dry wit, brief, addressing the user "
        "as \"sir\" now and then. Example: \"Done, sir. Three tests were failing; all pass now.\" "
        "Never change the facts, and keep any question for the user.\n\n"
    ),
}


def build_prompt(cfg: dict) -> str:
    """Summary prompt for the current read mode, plus the persona's style line if one is configured."""
    base = KEYPOINTS_PROMPT if cfg.get("read_mode") == "keypoints" else SUMMARY_PROMPT
    style = PERSONA_STYLES.get(str(cfg.get("persona", "")).lower(), "")
    head, sep, _ = base.partition("--- MESSAGE ---\n")
    return head + style + sep


def summarize(text: str, cfg: dict) -> str:
    fallback = clean_for_speech(text, cfg["max_chars"])
    if cfg["summarizer"] != "claude":
        return fallback
    # Short, already-clean messages don't need a model round trip.
    if len(text) <= 160 and fallback == text.strip():
        return fallback
    try:
        env = dict(os.environ, **{GUARD_ENV: "1"})  # stop our own call re-triggering the hook
        proc = subprocess.run(
            [shutil.which("claude") or "claude", "-p", "--model", cfg["summarizer_model"],
             "--no-session-persistence", "--disable-slash-commands"],
            input=build_prompt(cfg) + text[:6000],  # via stdin: multi-line args break on Windows
            capture_output=True, text=True, encoding="utf-8", timeout=cfg["summarizer_timeout"],
            env=env, cwd=tempfile.gettempdir(),
        )
        out = proc.stdout.strip()
        if proc.returncode == 0 and out:
            return out[: cfg["max_chars"] * 2]
        log(f"summarizer rc={proc.returncode} stderr={proc.stderr.strip()[:200]}")
    except (OSError, subprocess.SubprocessError) as e:
        log(f"summarizer failed: {e!r}")
    return fallback


def speech_text(text: str, cfg: dict) -> str:
    """What to say for an agent message: everything ("full") or a short summary ("summary")."""
    if cfg.get("read_mode", "full") == "full":
        return clean_for_speech(text, 0, "read" if cfg.get("read_code") else "note")
    return summarize(text, cfg)
