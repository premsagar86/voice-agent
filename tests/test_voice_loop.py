import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from voice_agent.agents.base import Done, Error, Text, ToolUse  # noqa: E402
from voice_agent.agents.claude_code import ClaudeCodeAdapter, parse_event  # noqa: E402
from voice_agent.config import DEFAULT_CONFIG  # noqa: E402
from voice_agent.main import VoiceLoop  # noqa: E402
from voice_agent.speech.filters import clean_for_speech  # noqa: E402

FAKE = [sys.executable, str(HERE / "fake_agent.py")]


def test_parse_event_text_tool_result():
    t = parse_event('{"type":"assistant","message":{"content":[{"type":"text","text":"hi"},'
                    '{"type":"tool_use","name":"Edit","input":{"file_path":"a.py"}}]}}')
    assert t == [Text("hi"), ToolUse("Edit", "a.py")]
    assert parse_event('{"type":"result","result":"ok","session_id":"s"}') == [Done("ok", "s")]
    assert isinstance(parse_event('{"type":"result","is_error":true,"result":"bad"}')[0], Error)
    assert parse_event("not json") == []


def test_adapter_contract_and_resume(tmp_path):
    sp = tmp_path / "sessions.json"
    a = ClaudeCodeAdapter(cwd=str(tmp_path), command=FAKE, sessions_path=sp)
    evs = list(a.send("hello"))
    assert any(isinstance(e, ToolUse) for e in evs)
    assert isinstance(evs[-1], Done) and "Echo: hello resume=" in evs[-1].result
    assert a.session_id == "sess-1" and sp.exists()
    # second adapter on same cwd picks up the saved session and resumes it
    b = ClaudeCodeAdapter(cwd=str(tmp_path), command=FAKE, sessions_path=sp)
    assert "resume=sess-1" in list(b.send("again"))[-1].result


def test_adapter_missing_binary_yields_error(tmp_path):
    a = ClaudeCodeAdapter(cwd=str(tmp_path), command=["definitely-not-a-binary-xyz"],
                          sessions_path=tmp_path / "s.json")
    assert isinstance(list(a.send("x"))[-1], Error)


def test_loop_full_turn(tmp_path):
    spoken, shown = [], []
    agent = ClaudeCodeAdapter(cwd=str(tmp_path), command=FAKE, sessions_path=tmp_path / "s.json")
    cfg = dict(DEFAULT_CONFIG, summarizer="local")
    loop = VoiceLoop(cfg, record=lambda: np.ones(8000, dtype="float32"),
                     transcribe=lambda audio: "list the files", agent=agent,
                     speak=lambda text, c: spoken.append(text), show=shown.append)
    assert loop.run_turn() is True
    assert spoken == ["Echo: list the files resume="]
    assert loop.state == "idle" and "You: list the files" in shown


def test_loop_ignores_silence(tmp_path):
    loop = VoiceLoop(dict(DEFAULT_CONFIG), record=lambda: np.zeros(0, dtype="float32"),
                     transcribe=lambda a: "", agent=None, speak=lambda *a: None, show=lambda s: None)
    assert loop.run_turn() is False


def test_filters_module_matches_hook():
    assert clean_for_speech("```\nx\n```") == ""


def test_stop_while_working_cancels_reply():
    from contextlib import nullcontext

    class Slow:
        interrupted = False

        def send(self, prompt):
            yield Text("Long explanation.")
            yield Done("Long explanation.")

        def interrupt(self):
            self.interrupted = True

    agent, spoken = Slow(), []
    loop = VoiceLoop(dict(DEFAULT_CONFIG), record=lambda: np.ones(8000, dtype="float32"),
                     transcribe=lambda a: "go", agent=agent,
                     speak=lambda t, c: spoken.append(t), show=lambda s: None)

    class SaysStop:  # user says "stop" as soon as the agent starts working
        def __enter__(self):
            loop.stop_now()

        def __exit__(self, *exc):
            pass

    loop.stop_monitor = SaysStop
    assert loop.run_turn() is True
    assert spoken == [] and agent.interrupted and loop.state == "idle" and not loop.cancelled


def test_stop_phrases_are_commands_but_not_prompts():
    from voice_agent.commands import parse_command
    for p in ("stop", "Stop it.", "enough", "That's enough", "hush"):
        assert parse_command(p) == "stop"
    assert parse_command("stop the dev server") is None


def test_loop_speaks_all_text_events_in_full_mode(tmp_path):
    from voice_agent.agents.base import Done, Text

    class Two:
        def send(self, prompt):
            yield Text("Part one.")
            yield Text("Part two.")
            yield Done("Part two.")

        def interrupt(self):
            pass

    spoken = []
    cfg = dict(DEFAULT_CONFIG, read_mode="full")
    loop = VoiceLoop(cfg, record=lambda: np.ones(8000, dtype="float32"), transcribe=lambda a: "go",
                     agent=Two(), speak=lambda t, c: spoken.append(t), show=lambda s: None)
    loop.run_turn()
    assert "Part one." in spoken[0] and "Part two." in spoken[0]


def test_plan_dialog_detection_and_reject_plan():
    from voice_agent.agents.console_input import plan_prompt_on_screen
    from voice_agent.agents.interactive import InteractiveAdapter
    plan_rows = ["Ready to code?", "1. Yes, auto-accept edits", "3. No, keep planning"]
    assert plan_prompt_on_screen(plan_rows) and not plan_prompt_on_screen(["> hello", "esc to interrupt"])

    class Keys:
        def __init__(self, rows):
            self.rows, self.sent = rows, []
        def read_screen(self):
            return self.rows
        def send(self, combo):
            self.sent.append(combo)

    k = Keys(plan_rows)
    assert InteractiveAdapter(keys=k).reject_plan() is True and k.sent == ["esc"]
    k = Keys(["> hello"])
    assert InteractiveAdapter(keys=k).reject_plan() is False and k.sent == []  # never presses Esc without a plan
