import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import speak_hook as sh  # noqa: E402

SAMPLE = HERE / "sample_transcript.jsonl"


def test_last_assistant_text_picks_final_text_message():
    text = sh.last_assistant_text(str(SAMPLE), whole_turn=False)
    assert text.startswith("## Done")
    assert "hmm" not in text  # thinking blocks ignored


def test_last_assistant_text_missing_file():
    assert sh.last_assistant_text(str(HERE / "nope.jsonl")) == ""


def test_clean_removes_code_markdown_urls_paths():
    out = sh.clean_for_speech(sh.last_assistant_text(str(SAMPLE), whole_turn=False), 300)
    assert "```" not in out and "def login" not in out
    assert "https://" not in out and "**" not in out and "#" not in out
    assert "D:\\proj" not in out and "auth.py" in out
    assert out.startswith("Done")


def test_clean_caps_length_on_word_boundary():
    out = sh.clean_for_speech("word " * 200, 50)
    assert len(out) <= 52


def test_clean_empty_and_code_only():
    assert sh.clean_for_speech("```\nx=1\n```", 300) == ""
    assert sh.clean_for_speech("", 300) == ""


def test_clean_table_and_diff_lines_dropped():
    out = sh.clean_for_speech("| a | b |\n|---|---|\n+++ b/file\n@@ -1 +1 @@\nAll good.", 300)
    assert out.endswith("All good.") and "@@" not in out and "+++" not in out


def test_full_read_keeps_everything_no_truncation():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    out = sh.clean_for_speech(text, 0)
    assert "Sentence number 0 " in out and "Sentence number 39 " in out


def test_code_block_modes():
    src = "Before.\n```py\nx = 1\n```\nAfter."
    assert "Code block skipped" in sh.clean_for_speech(src, 0, "note")
    assert "x = 1" not in sh.clean_for_speech(src, 0, "note")
    read = sh.clean_for_speech(src, 0, "read")
    assert "x = 1" in read and "```" not in read
    assert "Code block" not in sh.clean_for_speech(src, 0, "drop")


def test_whole_turn_collects_all_text_messages_in_order(tmp_path):
    import json
    rows = [
        {"message": {"role": "user", "content": "old prompt"}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "old answer"}]}},
        {"message": {"role": "user", "content": "new prompt"}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "First part."}]}},
        {"message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Read", "input": {}}]}},
        {"message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "Second part."}]}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert sh.last_assistant_text(str(p)) == "First part.\n\nSecond part."
    assert sh.last_assistant_text(str(p), whole_turn=False) == "Second part."


def test_whole_turn_default_includes_earlier_messages():
    assert sh.last_assistant_text(str(SAMPLE)).startswith("Looking at the code.")


def test_plan_and_question_text_from_tool_calls_is_spoken(tmp_path):
    import json
    rows = [
        {"message": {"role": "user", "content": "plan it"}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "Here is the plan."},
            {"type": "tool_use", "name": "ExitPlanMode", "input": {"plan": "# Plan\n- Do thing one\n- Do thing two"}},
        ]}},
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "AskUserQuestion", "input": {"questions": [{"question": "Which option?"}]}},
            {"type": "tool_use", "name": "Bash", "input": {"command": "rm -rf /"}},
        ]}},
    ]
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = sh.last_assistant_text(str(p))
    assert "Do thing one" in out and "Do thing two" in out and "Which option?" in out
    assert "rm -rf" not in out  # other tool calls are never spoken


def test_plan_turn_speaks_only_the_plan_other_turns_only_final_message(tmp_path):
    import json
    plan_rows = [
        {"message": {"role": "user", "content": "plan it"}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "Chatter."}]}},
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "ExitPlanMode", "input": {"plan": "- step one"}}]}},
    ]
    p = tmp_path / "plan.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in plan_rows), encoding="utf-8")
    assert sh.pick_speech(str(p)) == ("plan", "- step one")

    done_rows = [
        {"message": {"role": "user", "content": "do it"}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "Working on it."}]}},
        {"message": {"role": "assistant", "content": [{"type": "text", "text": "All done."}]}},
    ]
    q = tmp_path / "done.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in done_rows), encoding="utf-8")
    assert sh.pick_speech(str(q)) == ("done", "All done.")
    old = [{"message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "ExitPlanMode", "input": {"plan": "old plan"}}]}}]
    r = tmp_path / "old.jsonl"
    r.write_text("\n".join(json.dumps(x) for x in old + done_rows), encoding="utf-8")
    assert sh.pick_speech(str(r)) == ("done", "All done.")  # an earlier turn's plan is not repeated


def test_voice_stop_mutes_hooks_and_start_unmutes(tmp_path, monkeypatch):
    from voice_agent import config, lifecycle
    monkeypatch.setattr(config, "MUTED_PATH", tmp_path / "voice.muted")
    monkeypatch.setattr(lifecycle, "MUTED_PATH", tmp_path / "voice.muted")
    monkeypatch.setattr(lifecycle, "APP_PID_PATH", tmp_path / "app.pid")
    monkeypatch.setattr(lifecycle, "_stray_pids", lambda: [])
    assert config.load_config()["enabled"] is True
    assert lifecycle.stop() is False
    assert config.load_config()["enabled"] is False  # hook speech is muted
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "ROOT", tmp_path)
    assert lifecycle.start() is True
    assert config.load_config()["enabled"] is True


def test_stop_kills_stray_instances(tmp_path, monkeypatch):
    from voice_agent import lifecycle
    monkeypatch.setattr(lifecycle, "MUTED_PATH", tmp_path / "voice.muted")
    monkeypatch.setattr(lifecycle, "APP_PID_PATH", tmp_path / "app.pid")
    monkeypatch.setattr(lifecycle, "_stray_pids", lambda: [111, 222])
    killed = []
    monkeypatch.setattr(lifecycle, "_kill", killed.append)
    assert lifecycle.stop() is True
    assert sorted(killed) == [111, 222]


def test_persona_adds_style_and_empty_persona_is_unchanged():
    from voice_agent.speech import summarizer as s
    assert s.build_prompt({"read_mode": "summary", "persona": ""}) == s.SUMMARY_PROMPT
    assert s.build_prompt({"read_mode": "keypoints"}) == s.KEYPOINTS_PROMPT
    p = s.build_prompt({"read_mode": "summary", "persona": "jarvis"})
    assert "JARVIS" in p and p.endswith("--- MESSAGE ---\n")


def test_hey_jarvis_wake_word_is_stripped():
    from voice_agent.commands import strip_wake_word
    assert strip_wake_word("Hey, Jarvis. Open the tests", "hey jarvis") == "Open the tests"
    assert strip_wake_word("open the tests", "hey jarvis") is None


def test_fuzzy_wake_word():
    from voice_agent.commands import strip_wake_word
    w = "hey jarvis"
    assert strip_wake_word("Hey Jarvas, open the tests", w) == "open the tests"
    assert strip_wake_word("Jarvis open the tests", w) == "open the tests"
    assert strip_wake_word("Jarvis", w) == ""
    assert strip_wake_word("what's in this project", w) is None
    assert strip_wake_word("HR is watching this project", w) is None


def test_summarizer_runs_claude_without_a_visible_window(monkeypatch):
    from voice_agent.speech import summarizer as s
    seen = {}

    class R:
        returncode, stdout, stderr = 0, "spoken text", ""

    def fake_run(cmd, **kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(s.subprocess, "run", fake_run)
    cfg = dict(s.__dict__.get("DEFAULT_CONFIG", {}), summarizer="claude", summarizer_model="haiku",
               summarizer_timeout=5, max_chars=300, read_mode="summary", persona="")
    assert s.summarize("x" * 400, cfg) == "spoken text"
    assert seen["creationflags"] == s._NO_WINDOW
