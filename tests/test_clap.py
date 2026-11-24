import numpy as np

from voice_agent.agents.interactive import InteractiveAdapter
from voice_agent.audio.mic import BLOCK, SAMPLE_RATE, ClapDetector, segment_speech
from voice_agent.config import DEFAULT_CONFIG
from voice_agent.main import VoiceLoop

PER_S = SAMPLE_RATE // BLOCK  # blocks per second (about 33)


def blocks(levels):
    for lvl in levels:
        yield np.full(BLOCK, lvl, dtype="float32")


def clap(n=2):
    return [0.5] * n + [0.0] * 3


def test_double_clap_is_detected_but_single_clap_and_speech_are_not():
    c = ClapDetector()
    assert not any(c.feed(b) for b in blocks(clap() + [0.0] * 40))  # one clap
    c = ClapDetector()
    assert any(c.feed(b) for b in blocks(clap() + [0.0] * 5 + clap()))  # two claps, 0.3 s apart
    c = ClapDetector()
    assert not any(c.feed(b) for b in blocks(clap() + [0.0] * 60 + clap()))  # 2 s apart: two single claps
    c = ClapDetector()
    assert not any(c.feed(b) for b in blocks(([0.1] * 20 + [0.0] * 5) * 3))  # speech-level, long: not a clap
    c = ClapDetector()
    assert not any(c.feed(b) for b in blocks(([0.5] * 12 + [0.0] * 5) * 3))  # loud but too long


def test_segment_speech_returns_empty_and_flags_a_double_clap():
    c = ClapDetector()
    audio = segment_speech(blocks([0.0] * 10 + clap() + [0.0] * 5 + clap() + [0.0] * 50), clapper=c)
    assert audio.size == 0 and c.consume() and not c.consume()


class Keys:
    def __init__(self, rows, cursor=1):
        self.rows, self.cursor, self.log = rows, cursor, []

    def read_screen(self):
        out = []
        for r in self.rows:
            if r[:1].isdigit() and r[1] == ".":
                n = int(r[0])
                out.append(("> " if n == self.cursor else "  ") + r)
            else:
                out.append(r)
        return out

    def send(self, combo):
        self.log.append(combo)
        if combo == "down":
            self.cursor += 1
        if combo == "up":
            self.cursor -= 1

    def write(self, text):
        self.log.append(text)


PLAN = ["Would you like to proceed?", "1. Yes, and use auto mode", "2. Yes, manually approve edits",
        "3. Tell Claude what to change"]
TOOL = ["Do you want to proceed?", "1. Yes", "2. Yes, and do not ask again", "3. No"]


def adapter(keys):
    return InteractiveAdapter(keys=keys, sleep=lambda s: None)


def test_clap_approves_plan_with_manual_edits_option_2():
    k = Keys(PLAN)
    assert adapter(k).approve_prompt() == "plan" and k.cursor == 2 and k.log == ["down", "enter"]


def test_clap_approves_tool_permission_with_option_1():
    k = Keys(TOOL)
    assert adapter(k).approve_prompt() == "tool" and k.cursor == 1 and k.log == ["enter"]
    k = Keys(TOOL, cursor=3)  # cursor elsewhere: move to Yes first, never accept "No" or "do not ask again"
    assert adapter(k).approve_prompt() == "tool" and k.cursor == 1 and k.log == ["up", "up", "enter"]


def test_clap_never_approves_a_risky_command_or_an_empty_screen():
    k = Keys(["Bash command", "rm -rf build", "Do you want to proceed?", "1. Yes", "2. No"])
    assert adapter(k).approve_prompt() == "risky" and "enter" not in k.log
    k = Keys(["> hello", "esc to interrupt"])
    assert adapter(k).approve_prompt() == "none" and k.log == []


def test_loop_double_clap_approves_and_speaks():
    said = []
    c = ClapDetector()
    c.fired = True
    loop = VoiceLoop(dict(DEFAULT_CONFIG, listen_mode="auto"), record=lambda: np.zeros(0, dtype="float32"),
                     transcribe=lambda a: "", agent=adapter(Keys(TOOL)), speak=lambda t, cfg: said.append(t),
                     show=lambda s: None, clapper=c)
    assert loop.run_turn() is True and said == ["Approved."]
