"""Voice loop: speak -> speech-to-text -> Claude Code -> spoken reply, with voice commands and a safety guard.

Run from the project you want Claude to work in:
    python run.py
"""
from __future__ import annotations

import contextlib
import sys
import time
from typing import Callable

from .agents.base import AgentAdapter, Done, Error, Text, ToolUse
from .approvals import is_confirmation, is_destructive
from .audio import tts
from .commands import is_approval, normalize, parse_chain, parse_command, strip_wake_word
from .config import SESSION_PATH, load_config, log, save_mode
from .speech.summarizer import speech_text

# voice command -> (claude --permission-mode value, spoken name)
PERMISSION_MODES = {
    "mode_auto": ("auto", "Auto"),
    "mode_plan": ("plan", "Plan"),
    "mode_accept": ("acceptEdits", "Accept edits on"),
    "mode_manual": ("manual", "Manual"),
}
MODE_HELP = {
    "auto": "It will act without asking.",
    "plan": "It will only read and plan, and will not change any files.",
    "acceptEdits": "It will edit files without asking, but still asks for other actions.",
    "manual": "It will ask before editing or running anything.",
    "": "Claude's default behaviour.",
}

IDLE, LISTENING, TRANSCRIBING, WORKING, SUMMARIZING, SPEAKING = (
    "idle", "listening", "transcribing", "working", "summarizing", "speaking")


CLOSE_PHRASES = {"goodbye", "good bye", "bye", "that s all", "thats all", "thanks jarvis",
                 "thank you jarvis", "bye jarvis", "goodbye jarvis"}  # end the open conversation


UNMUTE_PHRASES = {"unmute", "un mute", "unmuted", "on mute", "unmute mic", "unmute the mic", "unmute please"}
_UNMUTE_PREFIX = ("hey claude", "claude", "hey jarvis", "jarvis")  # "Claude, unmute" works too


def is_unmute_phrase(text: str) -> bool:
    t = normalize(text)
    for prefix in _UNMUTE_PREFIX:
        if t.startswith(prefix + " "):
            t = t[len(prefix):].strip()
            break
    return t in UNMUTE_PHRASES


def wake_phrases(cfg: dict) -> list[str]:
    """All phrases that wake the app: `wake_words` (list) plus the older single `wake_word`."""
    words = list(cfg.get("wake_words") or [])
    if cfg.get("wake_word") and cfg["wake_word"] not in words:
        words.insert(0, cfg["wake_word"])
    return words


class VoiceLoop:
    """One spoken turn per `run_turn`. All I/O is injected so it can be tested."""

    def __init__(self, cfg: dict, record: Callable, transcribe: Callable, agent: AgentAdapter,
                 speak: Callable = tts.speak, show: Callable[[str], None] = print,
                 summarise: Callable = speech_text, monitor: Callable | None = None,
                 stop_monitor: Callable | None = None, clapper=None):
        self.clapper = clapper  # mic.ClapDetector or None: a double clap approves the prompt on screen
        self.cfg, self.record, self.transcribe = cfg, record, transcribe
        self.agent, self.speak, self.show, self.summarise = agent, speak, show, summarise
        self.monitor = monitor  # factory returning a context manager that watches for barge-in
        self.stop_monitor = stop_monitor  # factory: listens for the word "stop" (calls stop_now)
        self.cancelled = False  # set when the user said stop during this turn
        self.state = IDLE
        self.last_spoken = ""
        self.pending = ""  # a risky request waiting for a spoken "confirm"
        self.approval: tuple = ()  # ((allow_spec, label), ...) tools waiting for a spoken "approve"
        self._session_seen = self._session_stamp()  # a newer stamp = a new CLI session started
        self.mic_muted = False  # "mute": the mic is closed completely (no OS mic icon) until the unmute hotkey
        self._announce_unmute = False
        self._awake_until = 0.0  # after the wake word, sentences without it are accepted until this time

    @staticmethod
    def _session_stamp() -> float:
        try:
            return SESSION_PATH.stat().st_mtime
        except OSError:
            return 0.0

    def _open_window(self, timeout: float) -> None:
        """timeout > 0: seconds; < 0: until the session ends; 0: wake word every time."""
        now = time.monotonic()
        self._awake_until = float("inf") if timeout < 0 else (now + timeout if timeout > 0 else 0.0)

    def _set(self, state: str) -> None:
        self.state = state
        mode = getattr(self.agent, "permission_mode", "") if state == LISTENING else ""
        self.show(f"[{state}]" + (f" [mode: {mode}]" if mode else ""))

    def _on_clap(self) -> None:
        """Double clap: approve the plan screen or a tool permission, if one is waiting (never risky ones)."""
        approve = getattr(self.agent, "approve_prompt", None)
        result = approve() if approve else "none"
        if result == "plan":
            self._say("Plan approved, manual edits.")
        elif result == "tool":
            self._say("Approved.")
        elif result == "risky":
            self._say("That looks risky. Approve it by hand.")
        elif result == "unreadable":
            self._say("I could not read the Claude screen to approve.")
        else:
            self.show("(double clap: nothing waiting for approval)")

    def unmute(self) -> None:
        """Called from the unmute hotkey (another thread): the next turn reopens the mic and says Listening."""
        if self.mic_muted:
            self.mic_muted, self._announce_unmute = False, True

    def _mic_monitor(self, factory):
        """A listener that opens the mic, or nothing while muted (so no mic icon shows)."""
        return factory() if factory and not self.mic_muted else contextlib.nullcontext()

    def _say(self, text: str) -> None:
        self.show(f"Claude: {text}")
        self._set(SPEAKING)
        with self._mic_monitor(self.monitor):
            self.speak(text, self.cfg)
        self._set(IDLE)

    def stop_now(self) -> None:
        """The user said stop: silence speech and abandon the agent's reply."""
        self.cancelled = True
        tts.stop_speaking()
        if self.agent is not None:
            self.agent.interrupt()

    # ------------------------------------------------------------- commands
    def _command(self, name: str) -> None:
        if name == "repeat":
            self._say(self.last_spoken or "There is nothing to repeat yet.")
        elif name == "listen":  # open the conversation: no wake word needed until the session ends
            self._open_window(float(self.cfg.get("wake_timeout", -1)) or -1)
            self._say("Listening.")
        elif name == "mute":  # close the mic completely until the unmute hotkey
            self._awake_until = 0.0
            self.mic_muted = True
            if self.cfg.get("mute_mode", "listen") == "off":
                self.show(f"(muted, mic closed: press {self.cfg.get('unmute_hotkey', 'f8').upper()} to unmute)")
            else:
                self.show("(muted: only \"unmute\" or \"Claude unmute\" is heard)")
            self._say("Muted.")  # spoken confirmation; the mic is already muted, so no listener opens for it
        elif name == "chat":  # a plan is waiting: do not accept it, then take the changes you say next
            rejected = self.agent.reject_plan() if hasattr(self.agent, "reject_plan") else None
            if rejected:
                self._open_window(float(self.cfg.get("wake_timeout", -1)) or -1)
                self._say("Plan not accepted. Tell me what to change.")
            elif rejected is False:
                self._say("There is no plan waiting.")
            else:
                self._say("I could not read the Claude screen to find a plan.")
        elif name in ("stop", "esc"):  # "esc" = press Escape in the CLI: interrupt what Claude is doing
            tts.stop_speaking()
            self.agent.interrupt()
            self._set(IDLE)
        elif name in ("brief", "detail", "summary"):
            mode = {"brief": "keypoints", "detail": "full", "summary": "summary"}[name]
            self.cfg["read_mode"] = mode
            save_mode(mode)  # the Stop hook reads this too
            self._say({"keypoints": "Okay, key points only.", "full": "Okay, I will read everything.",
                       "summary": "Okay, short summaries."}[mode])
        elif name == "new_session":
            self.agent.new_session()
            self._awake_until = 0.0  # new session: say the wake word again
            self._say("Started a new session.")
        elif name in PERMISSION_MODES:
            mode, label = PERMISSION_MODES[name]
            announce = self.cfg.get("announce_mode", False)
            if hasattr(self.agent, "set_mode"):  # real CLI: press Shift+Tab
                note = self.agent.set_mode(mode)
                failed = note.startswith(("I could not", "I pressed"))
                if announce or failed:  # a failed change is never silent
                    self._say(f"{label} mode. {note}")
                else:
                    self.show(f"(mode: {label}. {note})")
            else:
                self.agent.permission_mode = mode
                if announce:
                    self._say(f"{label} mode. {MODE_HELP[mode]}")
                else:
                    self.show(f"(mode: {label})")
        elif name == "mode_query":
            if hasattr(self.agent, "current_mode"):  # real CLI: what its screen shows right now
                seen = self.agent.current_mode()
                if seen is None:
                    self._say("I could not read the Claude screen.")
                else:
                    self._say(f"{self.agent.NAMES.get(seen, seen)} mode.")
                return
            mode = getattr(self.agent, "permission_mode", "")
            label = next((l for m, l in PERMISSION_MODES.values() if m == mode), "Default")
            self._say(f"{label} mode.")
        elif name == "cancel":
            self.pending = ""
            self._say("Cancelled.")

    # ----------------------------------------------------------------- turn
    def run_turn(self) -> bool:
        """Returns False if nothing was heard."""
        if self.mic_muted:
            if self.cfg.get("mute_mode", "listen") == "off":  # mic fully closed: only the unmute hotkey wakes it
                time.sleep(0.2)
                return False
            # muted: the mic stays open but everything is ignored except "unmute" / "Claude, unmute"
            audio = self.record()
            if self.clapper and self.clapper.consume():
                return False  # muted: claps are ignored too
            heard = (self.transcribe(audio) or "").strip()
            if heard and is_unmute_phrase(heard):
                self.mic_muted = False
                self._open_window(float(self.cfg.get("wake_timeout", -1)) or -1)
                self._say("Listening.")
                return True
            if heard:
                self.show("(muted: say unmute or Claude unmute)")
            return False
        if self._announce_unmute:
            self._announce_unmute = False
            self._open_window(float(self.cfg.get("wake_timeout", -1)) or -1)
            self._say("Listening.")
            return True
        self._set(LISTENING)
        audio = self.record()
        if self.clapper and self.clapper.consume():
            self._on_clap()
            self._set(IDLE)
            return True
        self._set(TRANSCRIBING)
        text = (self.transcribe(audio) or "").strip()
        if not text:
            self.show("You: (nothing heard)")
            self._set(IDLE)
            return False
        wake = wake_phrases(self.cfg)
        if wake and self.cfg.get("listen_mode") == "auto":  # hands-free: only speech addressed to us counts
            timeout = float(self.cfg.get("wake_timeout", -1))
            stamp = self._session_stamp()
            if stamp != self._session_seen:  # new session: the wake word is needed again
                self._session_seen, self._awake_until = stamp, 0.0
            awake = time.monotonic() < self._awake_until
            # "chat", "listen", "unmute", "mute" and "esc" work without a wake word (and "listen" alone is
            # the command, not a wake word followed by nothing)
            rest = text if parse_command(text) in ("chat", "listen", "mute", "esc") else strip_wake_word(text, wake)
            if rest is None:
                if not awake:
                    self.show(f"(ignored, no wake word: {text})")
                    self._set(IDLE)
                    return False
                rest = text  # conversation is open: no need to say the wake word again
            elif not rest:
                self._open_window(timeout)
                self._say("Yes?")
                return True
            text = rest
            if awake and normalize(text) in CLOSE_PHRASES:
                self._awake_until = 0.0
                self._say("Very good.")
                return True
            self._open_window(timeout)
            if parse_command(text) == "mute":
                self._awake_until = 0.0  # muting must not re-open the conversation
        self.show(f"You: {text}")
        tts.stop_speaking()  # a new request cuts off the previous answer

        if self.approval:  # the agent was refused some tools and is waiting for approve / deny
            grants, self.approval = self.approval, ()
            if is_approval(text):
                self.agent.grant = [spec for spec, _ in grants]
                text = "Permission granted. Go ahead and do what you could not do before."
                self.show("(approved)")
            else:
                self._say("Denied.")
                return True
        elif self.pending:  # waiting for confirmation of a risky request
            if is_confirmation(text):
                text, self.pending = self.pending, ""
            else:
                self.pending = ""
                self._say("Cancelled.")
                return True
        else:
            chain = parse_chain(text)
            if chain:  # "plan mode, then auto mode": run each command in order
                for name in chain:
                    self._command(name)
                return True
            cmd = parse_command(text)
            if cmd:
                self._command(cmd)
                return True
            if self.cfg.get("guard_destructive", True) and is_destructive(text):
                self.pending = text
                self._say("That sounds destructive. Say confirm to go ahead, or anything else to cancel.")
                return True

        if getattr(self.agent, "external_output", False):  # typed into the real CLI; hooks speak the reply
            for _ in self.agent.send(text):
                pass
            self.show(f"(sent to Claude: {text})")
            self._set(IDLE)
            return True

        self._set(WORKING)
        texts, final, error, denials = [], "", "", ()
        self.cancelled = False
        with self._mic_monitor(self.stop_monitor):
            for ev in self.agent.send(text):
                if self.cancelled:
                    break
                if isinstance(ev, ToolUse):
                    self.show(f"  - {ev.name} {ev.summary}".rstrip())
                elif isinstance(ev, Text):
                    texts.append(ev.text)
                elif isinstance(ev, Done):
                    final, denials = ev.result, ev.denials
                elif isinstance(ev, Error):
                    error = ev.message
        if self.cancelled:  # said stop while working: no explanation, back to listening
            self.cancelled = False
            self.show("Stopped.")
            self._set(IDLE)
            return True
        if texts:  # everything the agent said this turn, in order
            final = "\n\n".join(texts)
        if error and not final:
            final = f"The agent reported an error. {error}"
        self._set(SUMMARIZING)
        spoken = self.summarise(final, self.cfg) if final else "I got no answer."
        self.last_spoken = spoken
        if denials:  # headless mode cannot ask, so ask out loud
            self.approval = tuple(denials)
            labels = " and ".join(label for _, label in denials[:3])
            spoken += f" I need permission to {labels}. Say approve to allow it, or anything else to deny."
        self._say(spoken)
        return True


def main() -> int:
    from .agents.claude_code import ClaudeCodeAdapter
    from .audio import mic
    from .stt.whisper import Transcriber

    cfg = load_config()
    stt = Transcriber(cfg["whisper_model"], cfg["whisper_compute_type"],
                      ", ".join(wake_phrases(cfg)) if cfg.get("listen_mode") == "auto" else "")
    print("Loading speech model (first run downloads it, this can take a while)...", flush=True)
    stt.load()
    print("Speech model ready.", flush=True)

    def heard(audio):
        print(f"heard {len(audio) / mic.SAMPLE_RATE:.1f} s", flush=True)
        return audio

    if "--test-mic" in sys.argv:
        import numpy as np
        import sounddevice as sd
        print("Say something for 4 seconds...", flush=True)
        audio = sd.rec(4 * mic.SAMPLE_RATE, samplerate=mic.SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = np.asarray(audio).reshape(-1)
        print(f"mic level {mic.rms(audio):.4f} (quiet room is about 0.001)")
        print(f"You: {stt.transcribe(audio) or '(nothing heard)'}")
        return 0

    import faulthandler
    from . import lifecycle
    from .config import ROOT
    other = lifecycle.running_pid()
    if other and other != __import__("os").getpid():
        print(f"Voice is already running (pid {other}). Use `voice stop` first.")
        return 0
    lifecycle.register()
    import atexit
    atexit.register(lifecycle.unregister)
    faulthandler.enable(open(ROOT / "crash.log", "a"))  # a hard crash leaves a trace instead of silence
    if cfg["target"] == "interactive":
        from .agents.interactive import InteractiveAdapter
        if sys.platform == "win32":
            from .agents.console_input import ConsoleKeys, find_cli_pid
            pid = int(cfg.get("cli_pid") or 0) or find_cli_pid()
            keys, where, alive = (ConsoleKeys(pid) if pid else None), f"process {pid}", (lambda: lifecycle.pid_alive(keys.pid))
        else:  # macOS / Linux: Claude must run inside tmux
            from .agents.tmux_input import TmuxKeys, find_cli_pane
            pane = cfg.get("tmux_pane") or find_cli_pane()
            keys, where = (TmuxKeys(pane) if pane else None), f"tmux pane {pane}"
            alive = lambda: bool(find_cli_pane() or cfg.get("tmux_pane"))  # noqa: E731
        if keys:
            agent = InteractiveAdapter(keys=keys)
            agent.cwd = f"Claude Code CLI ({where}), no focus needed"
            if "--background" in sys.argv:  # auto-started: quit when that CLI closes
                import os
                import threading
                import time

                def _watch() -> None:
                    while alive():
                        time.sleep(3)
                    log("claude CLI closed, voice exiting")
                    lifecycle.unregister()
                    os._exit(0)
                threading.Thread(target=_watch, daemon=True).start()
        else:
            print("Could not find a running Claude Code CLI (on macOS/Linux run `claude` inside tmux); "
                  "falling back to typing into the focused window.")
            agent = InteractiveAdapter()
    else:
        agent = ClaudeCodeAdapter(timeout=cfg["agent_timeout"])
    clapper = mic.ClapDetector(float(cfg.get("clap_threshold", 0.15))) if cfg.get("clap_approve") else None
    if cfg["listen_mode"] == "auto":
        record = lambda: heard(mic.record_auto(cfg["vad_threshold"], cfg["vad_silence"], clapper))  # noqa: E731
        hint = "Just speak"
    else:
        record = lambda: heard(mic.record_while_pressed(  # noqa: E731
            cfg["hotkey"], on_start=lambda: print("[recording]", flush=True)))
        hint = f"Hold {cfg['hotkey'].upper()} and speak"
    monitor = stop_monitor = None
    loop: VoiceLoop | None = None
    if cfg["stop_listen"]:  # say "stop" while it works or speaks (other noise is ignored)
        stop_monitor = lambda: mic.StopWordMonitor(loop.stop_now, stt.transcribe, cfg["stop_threshold"])  # noqa: E731
        monitor = stop_monitor
    elif cfg["barge_in"]:
        monitor = lambda: mic.BargeInMonitor(tts.stop_speaking, cfg["barge_in_threshold"])  # noqa: E731
    loop = VoiceLoop(cfg, record, stt.transcribe, agent, monitor=monitor, stop_monitor=stop_monitor, clapper=clapper)
    hotkey = mic.StopHotkey(loop.stop_now, cfg["stop_hotkey"])
    hotkey.__enter__()  # F10 anywhere = stop
    unmute_key = mic.StopHotkey(loop.unmute, cfg.get("unmute_hotkey", "f8"))
    unmute_key.__enter__()  # F8 anywhere = unmute (the mic is fully closed while muted)
    print(f"{hint}. Press {cfg['stop_hotkey'].upper()} (or say stop) to silence. Ctrl+C to quit. Working in: {agent.cwd}")
    print('Commands: "repeat that", "stop", "brief", "explain in detail", "new session".')
    print('Modes: "auto mode", "plan mode", "accept edits on mode", "manual mode".')
    if "--tray" in sys.argv:
        from .tray import run_tray
        return run_tray(loop, cfg)
    try:
        while True:
            try:
                loop.run_turn()
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # one bad turn (e.g. lost Claude window) must not kill the app
                log(f"turn error: {e!r}")
                print(f"Error: {e}", file=sys.stderr)
                __import__("time").sleep(1)
    except KeyboardInterrupt:
        agent.interrupt()
        tts.stop_speaking()
        print("Bye.")
        return 0
    except Exception as e:  # keep the reason visible and logged
        log(f"main error: {e!r}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        hotkey.__exit__(None, None, None)
        unmute_key.__exit__(None, None, None)


if __name__ == "__main__":
    sys.exit(main())
