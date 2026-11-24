import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from voice_agent.agents.base import Done, Text  # noqa: E402
from voice_agent.approvals import is_confirmation, is_destructive  # noqa: E402
from voice_agent.audio.mic import BLOCK, SAMPLE_RATE, segment_speech  # noqa: E402
from voice_agent.commands import parse_command  # noqa: E402
from voice_agent.config import DEFAULT_CONFIG  # noqa: E402
from voice_agent.main import VoiceLoop  # noqa: E402


def test_parse_command_whole_utterance_only():
    assert parse_command("Repeat that.") == "repeat"
    assert parse_command("Read-only mode") == "mode_plan"
    assert parse_command("Auto mode") == "mode_auto"
    assert parse_command("accept edits on mode") == "mode_accept"
    assert parse_command("manual mode") == "mode_manual"
    assert parse_command("Brief!") == "brief"
    assert parse_command("new session") == "new_session"
    assert parse_command("give me a brief overview of auth.py") is None
    assert parse_command("stop the server and show logs") is None


def test_destructive_detection():
    for t in ["delete everything", "run rm -rf node_modules", "force push to main",
              "drop table users", "git reset --hard", "wipe the folder"]:
        assert is_destructive(t), t
    for t in ["list the files", "explain this function", "add a test for login"]:
        assert not is_destructive(t), t


def test_bare_yes_is_not_confirmation():
    assert is_confirmation("Confirm.") and is_confirmation("yes confirm") and is_confirmation("confirm delete")
    assert not is_confirmation("yes") and not is_confirmation("okay") and not is_confirmation("sure")


class FakeAgent:
    def __init__(self):
        self.prompts, self.permission_mode, self.reset, self.interrupted = [], "", 0, 0

    def send(self, prompt):
        self.prompts.append(prompt)
        yield Text(f"did {prompt}")
        yield Done(f"did {prompt}")

    def new_session(self):
        self.reset += 1

    def interrupt(self):
        self.interrupted += 1


def make_loop(utterances, cfg=None):
    said, it = [], iter(utterances)
    agent = FakeAgent()
    c = dict(DEFAULT_CONFIG, **(cfg or {}))
    loop = VoiceLoop(c, record=lambda: np.ones(8000, dtype="float32"), transcribe=lambda a: next(it),
                     agent=agent, speak=lambda t, _c: said.append(t), show=lambda s: None)
    return loop, agent, said


def test_destructive_needs_confirm_then_runs():
    loop, agent, said = make_loop(["delete the build folder", "confirm"])
    loop.run_turn()
    assert agent.prompts == [] and "destructive" in said[-1]
    loop.run_turn()
    assert agent.prompts == ["delete the build folder"]


def test_destructive_cancelled_by_anything_else_including_yes():
    loop, agent, said = make_loop(["delete the build folder", "yes", "list files"])
    loop.run_turn()
    loop.run_turn()
    assert agent.prompts == [] and said[-1] == "Cancelled."
    loop.run_turn()  # the guard is cleared; a normal prompt works
    assert agent.prompts == ["list files"]


def test_commands_repeat_brief_readonly_newsession():
    loop, agent, said = make_loop(["list files", "repeat that", "brief", "plan mode", "new session"])
    for _ in range(5):
        loop.run_turn()
    assert said[1] == said[0] == "did list files"
    assert loop.cfg["read_mode"] == "keypoints"
    assert agent.permission_mode == "plan" and agent.reset == 1
    assert agent.prompts == ["list files"]  # commands never reach the agent


def test_guard_can_be_disabled():
    loop, agent, _ = make_loop(["delete the build folder"], {"guard_destructive": False})
    loop.run_turn()
    assert agent.prompts == ["delete the build folder"]


def _blocks(levels):
    for lvl in levels:
        yield np.full(BLOCK, lvl, dtype="float32")


def test_segment_speech_cuts_after_silence_and_ignores_leading_quiet():
    quiet, loud = 0.0, 0.1
    per_s = SAMPLE_RATE // BLOCK  # blocks per second (~33)
    stream = [quiet] * 20 + [loud] * 30 + [quiet] * (per_s * 2) + [loud] * 50  # second burst unused
    audio = segment_speech(_blocks(stream), threshold=0.02, silence_s=1.0)
    assert 30 * BLOCK <= len(audio) <= (30 + 8 + per_s + 2) * BLOCK  # speech + preroll + 1 s tail
    assert segment_speech(_blocks([quiet] * 100)).size == 0


def test_new_mode_aliases():
    assert parse_command("key points") == "brief"
    assert parse_command("Important points") == "brief"
    assert parse_command("full") == "detail"
    assert parse_command("short summary") == "summary"


def test_mode_command_is_saved_for_the_hook():
    from voice_agent.config import load_config
    loop, _, said = make_loop(["important points", "list files"])
    loop.run_turn()
    assert load_config()["read_mode"] == "keypoints"  # what the Stop hook will read
    loop.cfg["summarizer"] = "local"
    loop.run_turn()
    loop2, _, _ = make_loop(["full"])
    loop2.run_turn()
    assert load_config()["read_mode"] == "full"


def test_permission_mode_voice_commands():
    loop, agent, said = make_loop(["auto mode", "accept edits on mode", "manual mode", "plan mode"])
    seen = []
    for _ in range(4):
        loop.run_turn()
        seen.append(agent.permission_mode)
    assert seen == ["auto", "acceptEdits", "manual", "plan"]
    assert said[1].startswith("Accept edits on mode.")
    assert agent.prompts == []


def test_adapter_passes_permission_mode(tmp_path):
    from voice_agent.agents.claude_code import ClaudeCodeAdapter
    seen = {}

    class Spy(ClaudeCodeAdapter):
        pass

    a = Spy(cwd=str(tmp_path), command=[sys.executable, str(HERE / "fake_agent.py")],
            sessions_path=tmp_path / "s.json")
    a.permission_mode = "acceptEdits"
    import subprocess
    real = subprocess.Popen

    def spy(args, *a_, **k):
        seen["args"] = args
        return real(args, *a_, **k)

    import voice_agent.agents.claude_code as cc
    cc.subprocess.Popen = spy
    try:
        list(a.send("hi"))
    finally:
        cc.subprocess.Popen = real
    assert seen["args"][-2:] == ["--permission-mode", "acceptEdits"]


def test_mode_phrases_are_tolerant():
    assert parse_command("Change to manual manual.") == "mode_manual"
    assert parse_command("Change to auto mode.") == "mode_auto"
    assert parse_command("switch to plan") == "mode_plan"
    assert parse_command("Change to accept edits on mode") == "mode_accept"
    assert parse_command("change the login page colour") is None


def test_spoken_permission_approval():
    from voice_agent.agents.claude_code import parse_denials
    from voice_agent.commands import is_approval
    d = parse_denials([{"tool_name": "Bash", "tool_input": {"command": "npm test"}}])
    assert d[0][0] == "Bash(npm test)"
    assert is_approval("Approve.") and is_approval("yes approve") and not is_approval("yes")

    loop, agent, said = make_loop(["go", "approve"])
    agent.send = lambda prompt, _p=agent.prompts: iter(
        [Text("x"), Done("x", "", d)]) if not _p.append(prompt) and len(_p) == 1 else iter([Done("ok")])
    agent.grant = []
    loop.run_turn()
    assert "permission" in said[-1] and loop.approval
    loop.run_turn()
    assert agent.grant == ["Bash(npm test)"] and not loop.approval


def test_mode_feedback_and_query():
    loop, agent, said = make_loop(["change to plan mode", "what mode"])
    loop.run_turn()
    loop.run_turn()
    assert agent.permission_mode == "plan"
    assert said[0] == said[1] and "only read and plan" in said[0]


def test_hotkey_calls_stop_now():
    loop, agent, said = make_loop([])
    loop.stop_now()
    assert loop.cancelled and agent.interrupted == 1


def test_interactive_mode_cycling_and_typing(tmp_path):
    from voice_agent.agents.interactive import InteractiveAdapter, presses_needed, save_state
    assert presses_needed("default", "plan") == 2 and presses_needed("plan", "default") == 2
    assert presses_needed("auto", "acceptEdits") == 2 and presses_needed("plan", "plan") == 0

    class Keys:
        def __init__(self):
            self.log = []

        def write(self, t):
            self.log.append(("write", t))

        def send(self, c):
            self.log.append(("send", c))

    keys, sp = Keys(), tmp_path / "m.json"
    save_state("default", sp)
    a = InteractiveAdapter(keys=keys, state_path=sp, sleep=lambda s: None)
    assert "2 times" in a.set_mode("plan")
    assert keys.log == [("send", "shift+tab")] * 2
    keys.log.clear()
    assert "Already" in a.set_mode("plan")  # state remembered
    list(a.send("hello"))
    a.interrupt()
    assert keys.log == [("write", "hello"), ("send", "enter"), ("send", "esc")]


def test_loop_types_into_cli_and_stays_quiet(tmp_path):
    from voice_agent.agents.interactive import InteractiveAdapter, save_state

    class Keys:
        log = []

        def write(self, t):
            self.log.append(t)

        def send(self, c):
            self.log.append(c)

    sp = tmp_path / "m.json"
    save_state("default", sp)
    agent = InteractiveAdapter(keys=Keys(), state_path=sp, sleep=lambda s: None)
    said, it = [], iter(["list the files", "plan mode"])
    loop = VoiceLoop(dict(DEFAULT_CONFIG), record=lambda: np.ones(8000, dtype="float32"),
                     transcribe=lambda a: next(it), agent=agent,
                     speak=lambda t, c: said.append(t), show=lambda s: None)
    loop.run_turn()
    assert said == [] and Keys.log == ["list the files", "enter"]  # typed; hooks will speak the reply
    loop.run_turn()
    assert "Plan mode." in said[0] and Keys.log.count("shift+tab") == 2


def test_pick_cli_pid_skips_hosts_and_headless():
    from voice_agent.agents.console_input import pick_cli_pid
    rows = [
        {"ProcessId": 1848, "Name": "claude.exe", "CommandLine": '"claude.exe"  "--chrome-native-host"'},
        {"ProcessId": 500, "Name": "claude.exe", "CommandLine": '"claude.exe" -p --output-format stream-json'},
        {"ProcessId": 10804, "Name": "claude.exe", "CommandLine": '"C:\\Users\\x\\claude.exe"'},
        {"ProcessId": 77, "Name": "node.exe", "CommandLine": "node server.js"},
    ]
    assert pick_cli_pid(rows) == 10804
    assert pick_cli_pid([]) == 0


def test_mode_never_claims_success_when_cli_unreachable(tmp_path):
    from voice_agent.agents.interactive import InteractiveAdapter, save_state

    class Broken:
        def write(self, t):
            raise RuntimeError("no console")

        def send(self, c):
            raise RuntimeError("no console")

    sp = tmp_path / "m.json"
    save_state("default", sp)
    a = InteractiveAdapter(keys=Broken(), state_path=sp, sleep=lambda s: None)
    assert "could not reach" in a.set_mode("plan").lower()
    from voice_agent.agents.interactive import load_state
    assert load_state(sp) == "default"  # state untouched


def test_set_mode_reads_screen_and_presses_until_target(tmp_path):
    from voice_agent.agents.interactive import InteractiveAdapter
    from voice_agent.agents.console_input import mode_from_screen

    assert mode_from_screen(["x", "  auto mode on (shift+tab to cycle)"]) == "auto"
    assert mode_from_screen(["x", "  accept edits on (shift+tab to cycle)"]) == "acceptEdits"
    assert mode_from_screen(["x", "  ? for shortcuts"]) == "default" and mode_from_screen([]) is None

    footers = {"auto": "auto mode on", "default": "? for shortcuts", "acceptEdits": "accept edits on",
               "plan": "plan mode on"}
    order = ["default", "acceptEdits", "plan", "auto"]

    class Screen:
        def __init__(self, mode):
            self.mode, self.presses = mode, 0

        def read_screen(self):
            return ["-----", footers[self.mode]]

        def send(self, combo):
            self.presses += 1
            self.mode = order[(order.index(self.mode) + 1) % 4]

        def write(self, t):
            pass

    for start, target, presses in [("auto", "plan", 3), ("acceptEdits", "plan", 1), ("plan", "plan", 0)]:
        s = Screen(start)
        a = InteractiveAdapter(keys=s, state_path=tmp_path / "m.json", sleep=lambda x: None)
        note = a.set_mode("plan")
        assert s.mode == "plan" and s.presses == presses, (start, note)
    s = Screen("auto")
    a = InteractiveAdapter(keys=s, state_path=tmp_path / "m.json", sleep=lambda x: None)
    assert a.current_mode() == "auto"
    s.read_screen = lambda: []  # unreadable screen: never claims success
    assert "cannot confirm" in a.set_mode("plan") and s.presses == 0


def test_segment_speech_ignores_clicks_and_blips_after_speech():
    quiet, loud = 0.0, 0.1
    per_s = SAMPLE_RATE // BLOCK
    # a lone click never starts a recording
    assert segment_speech(_blocks([quiet] * 10 + [loud] * 2 + [quiet] * 100)).size == 0
    # speech, then a click every 0.5 s: the click must not keep the recording open
    tail = ([quiet] * (per_s // 2) + [loud] * 2) * 6
    audio = segment_speech(_blocks([loud] * 30 + tail), threshold=0.02, silence_s=1.0)
    assert audio.size and len(audio) <= (30 + 10 + per_s + 5) * BLOCK


def test_wake_word_opens_a_conversation_window(monkeypatch):
    import voice_agent.main as m
    now = [1000.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: now[0])
    loop, agent, said = make_loop(
        ["list files", "hey jarvis list files", "now run the tests", "goodbye", "run it again",
         "hey jarvis", "ok", "run it again"],
        {"listen_mode": "auto", "wake_word": "hey jarvis", "wake_timeout": 60})
    assert loop.run_turn() is False and agent.prompts == []  # no wake word yet: ignored
    loop.run_turn()
    loop.run_turn()
    assert agent.prompts == ["list files", "now run the tests"]  # second sentence needed no wake word
    loop.run_turn()  # goodbye closes the window
    assert said[-1] == "Very good."
    assert loop.run_turn() is False  # ignored again
    loop.run_turn()
    assert said[-1] == "Yes?"
    loop.run_turn()  # "ok" accepted inside the window
    now[0] += 61  # silence for longer than the timeout
    assert loop.run_turn() is False and agent.prompts[-1] == "ok"


def test_wake_timeout_zero_needs_wake_word_every_time():
    loop, agent, said = make_loop(["hey jarvis list files", "run tests"],
                                  {"listen_mode": "auto", "wake_word": "hey jarvis", "wake_timeout": 0})
    loop.run_turn()
    assert loop.run_turn() is False and agent.prompts == ["list files"]


def test_new_session_closes_the_conversation(monkeypatch, tmp_path):
    import voice_agent.main as m
    stamp = tmp_path / "session.stamp"
    monkeypatch.setattr(m, "SESSION_PATH", stamp)
    loop, agent, said = make_loop(["hey jarvis list files", "run tests", "run tests again"],
                                  {"listen_mode": "auto", "wake_word": "hey jarvis", "wake_timeout": 60})
    loop.run_turn()
    loop.run_turn()
    assert agent.prompts == ["list files", "run tests"]
    stamp.write_text("1")  # SessionStart hook fired: a new CLI session
    assert loop.run_turn() is False and agent.prompts == ["list files", "run tests"]


def test_default_conversation_stays_open_for_the_whole_session(monkeypatch, tmp_path):
    import voice_agent.main as m
    now = [1000.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(m, "SESSION_PATH", tmp_path / "session.stamp")
    loop, agent, said = make_loop(["hey jarvis list files", "run tests"],
                                  {"listen_mode": "auto", "wake_word": "hey jarvis"})
    loop.run_turn()
    now[0] += 3 * 3600  # hours later, same session
    loop.run_turn()
    assert agent.prompts == ["list files", "run tests"]


def test_chat_opens_conversation_and_esc_interrupts_without_wake_word(monkeypatch, tmp_path):
    import voice_agent.main as m
    monkeypatch.setattr(m, "SESSION_PATH", tmp_path / "session.stamp")
    loop, agent, said = make_loop(["esc", "chat", "list files"],
                                  {"listen_mode": "auto", "wake_word": "hey jarvis"})
    interrupted = []
    agent.interrupt = lambda: interrupted.append(1)
    loop.run_turn()
    assert interrupted == [] or interrupted == [1]
    n = len(interrupted)
    loop.run_turn()
    assert said[-1] == "Listening."
    loop.run_turn()  # no wake word needed now
    assert agent.prompts == ["list files"] and n == 1
