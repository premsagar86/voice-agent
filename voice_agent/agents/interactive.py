"""Drive the REAL interactive Claude Code CLI by typing into the focused window.

Replies are spoken by the Stop/Notification hooks, so this adapter never produces events.
Permission modes are changed with Shift+Tab, the key Claude Code itself uses.
"""
from __future__ import annotations

import json
import time
from typing import Callable, Iterator

from ..config import ROOT
from .base import Event
from ..approvals import is_destructive
from .console_input import (mode_from_screen, plan_option_selected, plan_prompt_on_screen,
                            tool_prompt_on_screen)

STATE_PATH = ROOT / "mode_state.json"  # last permission mode seen by the UserPromptSubmit hook
CYCLE = ["default", "acceptEdits", "plan", "auto"]  # Shift+Tab order in Claude Code
ALIASES = {"manual": "default", "": "default"}


def load_state(path=STATE_PATH) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("permission_mode", ""))
    except (OSError, ValueError, AttributeError):
        return ""


def save_state(mode: str, path=STATE_PATH) -> None:
    path.write_text(json.dumps({"permission_mode": mode}), encoding="utf-8")


def presses_needed(current: str, target: str) -> int:
    cur, tgt = ALIASES.get(current, current), ALIASES.get(target, target)
    if cur not in CYCLE or tgt not in CYCLE:
        return 0
    return (CYCLE.index(tgt) - CYCLE.index(cur)) % len(CYCLE)


class _Keys:
    """Real keyboard output, replaceable in tests."""

    def write(self, text: str) -> None:
        import keyboard
        keyboard.write(text, delay=0.005)

    def send(self, combo: str) -> None:
        import keyboard
        keyboard.send(combo)


class InteractiveAdapter:
    external_output = True  # the hooks speak; the voice loop must stay quiet

    def __init__(self, keys: _Keys | None = None, state_path=STATE_PATH, pause: float = 0.15,
                 sleep: Callable[[float], None] = time.sleep):
        self.keys, self.state_path, self.pause, self.sleep = keys or _Keys(), state_path, pause, sleep
        self.cwd = "the focused Claude Code window"
        self.permission_mode = ""

    def _screen(self) -> list[str] | None:
        reader = getattr(self.keys, "read_screen", None)
        if reader is None:
            return None
        try:
            return reader()
        except Exception:
            return None

    def _plan_waiting(self) -> bool:
        rows = self._screen()
        return bool(rows and plan_prompt_on_screen(rows))

    def _select_feedback_option(self) -> bool:
        """Move the cursor to option 3 ("Tell Claude what to change") and confirm it on screen."""
        for _ in range(5):
            rows = self._screen()
            if not rows:
                return False
            cur = plan_option_selected(rows)
            if cur == 3:
                return True
            if cur is None:
                return False
            try:
                self.keys.send("down")
            except Exception:
                return False
            self.sleep(0.2)
        return False

    def send(self, prompt: str) -> Iterator[Event]:
        if self._plan_waiting() and not self._select_feedback_option():
            # Typing + Enter on option 1 would ACCEPT the plan: if option 3 can't be reached, send nothing.
            return iter(())
        self.keys.write(prompt)
        self.sleep(self.pause)
        self.keys.send("enter")
        return iter(())

    def is_working(self) -> bool | None:
        """True if the footer shows "esc to interrupt"; None if the screen can't be read."""
        reader = getattr(self.keys, "read_screen", None)
        if reader is None:
            return None
        try:
            rows = reader()
        except Exception:
            return None
        return any("esc to interrupt" in r.lower() for r in rows[-4:]) if rows else None

    def interrupt(self) -> None:
        """Send Esc, but only while Claude works: Esc twice on an idle prompt opens the rewind menu."""
        if self.is_working() is False and not self._plan_waiting():
            return
        try:
            self.keys.send("esc")
        except Exception:
            pass

    def _select_option(self, number: int) -> bool:
        """Move the cursor to option `number` (1-based) and confirm it on screen."""
        for _ in range(6):
            rows = self._screen()
            cur = plan_option_selected(rows) if rows else None
            if cur is None:
                return False
            if cur == number:
                return True
            try:
                self.keys.send("down" if cur < number else "up")
            except Exception:
                return False
            self.sleep(0.2)
        return False

    def approve_prompt(self) -> str:
        """Approve what is on screen (used by a double clap). Returns "plan", "tool", "risky", "none" or
        "unreadable". Plan screen: option 2 (manual approve). Tool permission: option 1 (Yes). Anything that
        looks destructive is NEVER approved this way."""
        rows = self._screen()
        if rows is None:
            return "unreadable"
        if plan_prompt_on_screen(rows):
            kind, number = "plan", 2
        elif tool_prompt_on_screen(rows):
            kind, number = "tool", 1
            if is_destructive(" ".join(rows[-25:])):
                return "risky"
        else:
            return "none"
        if not self._select_option(number):
            return "unreadable"
        try:
            self.keys.send("enter")
        except Exception:
            return "unreadable"
        return kind

    def reject_plan(self) -> bool | None:
        """Go to option 3 of the plan dialog ("Tell Claude what to change") so the next sentence is typed as
        feedback. True if there, False if no plan is waiting, None if the screen can't be read."""
        rows = self._screen()
        if rows is None:
            return None
        if not plan_prompt_on_screen(rows):
            return False
        return True if self._select_feedback_option() else None

    def new_session(self) -> None:
        for _ in self.send("/clear"):
            pass

    NAMES = {"default": "Manual", "acceptEdits": "Accept edits on", "plan": "Plan", "auto": "Auto",
             "bypassPermissions": "Bypass permissions"}

    def current_mode(self) -> str | None:
        """The mode actually shown on the CLI's screen, or None if it can't be read."""
        reader = getattr(self.keys, "read_screen", None)
        if reader is None:
            return None
        try:
            return mode_from_screen(reader())
        except Exception:
            return None

    def set_mode(self, target: str) -> str:
        if getattr(self.keys, "read_screen", None) is not None:
            return self._set_mode_checked(ALIASES.get(target, target))
        return self._set_mode_blind(target)

    def _set_mode_checked(self, target: str) -> str:
        """Press Shift+Tab until the screen shows the target mode (max 6 presses)."""
        pressed, seen = 0, None
        for _ in range(7):
            seen = self.current_mode()
            if seen is None:
                return "I could not read the Claude screen, so I cannot confirm the mode."
            if seen == target:
                save_state(target, self.state_path)
                self.permission_mode = target
                return "Already on." if pressed == 0 else (
                    f"Confirmed on screen after {pressed} Shift Tab press{'es' if pressed != 1 else ''}.")
            if pressed == 6:
                break
            try:
                self.keys.send("shift+tab")
            except Exception as e:
                return f"I could not reach the Claude window. {str(e)[:80]}"
            pressed += 1
            self.sleep(0.4)
        return f"I pressed Shift Tab {pressed} times but the screen shows {self.NAMES.get(seen, seen)}."

    def _set_mode_blind(self, target: str) -> str:
        current = load_state(self.state_path)
        n = presses_needed(current, target)
        try:
            for _ in range(n):
                self.keys.send("shift+tab")
                self.sleep(self.pause)
        except Exception as e:  # could not reach the CLI: never claim the mode changed
            return f"I could not reach the Claude window. {str(e)[:80]}"
        save_state(ALIASES.get(target, target), self.state_path)
        self.permission_mode = target
        known = "" if current else " I did not know the starting mode, so I assumed default."
        if n == 0:
            return "Already in that mode." + known
        return f"Pressed Shift Tab {n} time{'s' if n != 1 else ''}. I will confirm on your next prompt.{known}"
