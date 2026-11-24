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


class PlanScreenKeys:
    """Fake CLI: the real plan dialog, a cursor that moves with Down, and a log of keys/typing."""

    def __init__(self, waiting=True):
        self.cursor, self.waiting, self.log = 1, waiting, []

    def read_screen(self):
        if not self.waiting:
            return ["> hello", "esc to interrupt"]
        marks = [">" if i == self.cursor else " " for i in (1, 2, 3)]
        return ["Claude has written up a plan and is ready to execute. Would you like to proceed?",
                f"{marks[0]} 1. Yes, and use auto mode", f"{marks[1]} 2. Yes, manually approve edits",
                f"{marks[2]} 3. Tell Claude what to change"]

    def send(self, combo):
        self.log.append(combo)
        if combo == "down":
            self.cursor = min(3, self.cursor + 1)

    def write(self, text):
        self.log.append(text)


def test_plan_dialog_detection_and_option_cursor():
    from voice_agent.agents.console_input import plan_option_selected, plan_prompt_on_screen
    k = PlanScreenKeys()
    assert plan_prompt_on_screen(k.read_screen()) and plan_option_selected(k.read_screen()) == 1
    k.cursor = 3
    assert plan_option_selected(k.read_screen()) == 3
    assert not plan_prompt_on_screen(["> hello", "esc to interrupt"])


def test_chat_goes_to_option_3_and_next_sentence_is_typed_there():
    from voice_agent.agents.interactive import InteractiveAdapter
    k = PlanScreenKeys()
    a = InteractiveAdapter(keys=k, sleep=lambda s: None)
    assert a.reject_plan() is True and k.cursor == 3 and k.log == ["down", "down"]
    list(a.send("make it shorter"))
    assert k.log == ["down", "down", "make it shorter", "enter"]  # never on option 1 or 2
    assert InteractiveAdapter(keys=PlanScreenKeys(waiting=False), sleep=lambda s: None).reject_plan() is False


def test_any_sentence_in_plan_dialog_goes_to_option_3_not_accept():
    from voice_agent.agents.interactive import InteractiveAdapter
    k = PlanScreenKeys()
    list(InteractiveAdapter(keys=k, sleep=lambda s: None).send("let's start"))
    assert k.log == ["down", "down", "let's start", "enter"] and k.cursor == 3
    k = PlanScreenKeys(waiting=False)
    list(InteractiveAdapter(keys=k, sleep=lambda s: None).send("hello"))
    assert k.log == ["hello", "enter"]  # no plan waiting: plain typing


def test_cannot_find_cursor_means_nothing_is_typed():
    from voice_agent.agents.interactive import InteractiveAdapter

    class Blind(PlanScreenKeys):
        def read_screen(self):
            return ["Would you like to proceed?", "1. Yes", "2. Yes", "3. Tell Claude what to change"]

    k = Blind()
    assert list(InteractiveAdapter(keys=k, sleep=lambda s: None).send("hi")) == [] and k.log == []


def test_escape_interrupts_the_plan_dialog():
    from voice_agent.agents.interactive import InteractiveAdapter
    k = PlanScreenKeys()
    InteractiveAdapter(keys=k, sleep=lambda s: None).interrupt()
    assert k.log == ["esc"]
    k = PlanScreenKeys(waiting=False)
    k.read_screen = lambda: ["> hello"]  # idle prompt: Esc twice opens the rewind menu, so nothing is sent
    InteractiveAdapter(keys=k, sleep=lambda s: None).interrupt()
    assert k.log == []
