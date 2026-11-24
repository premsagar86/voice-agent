"""Installer, tmux picker and per-OS speech command (no real settings, tmux or speech involved)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import install  # noqa: E402
from voice_agent.agents.tmux_input import TmuxKeys, pick_pane  # noqa: E402
from voice_agent.audio import tts  # noqa: E402


def test_install_adds_hooks_and_keeps_others(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "sonnet", "hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "echo mine"}]},
        {"hooks": [{"type": "command", "command": "python /old/place/speak_hook.py stop"}]}]}}))
    install.update_settings(p)
    install.update_settings(p)  # idempotent
    s = json.loads(p.read_text())
    assert s["model"] == "sonnet"
    cmds = [h["command"] for g in s["hooks"]["Stop"] for h in g["hooks"]]
    assert cmds[0] == "echo mine" and len(cmds) == 2 and "/old/place" not in cmds[1]
    assert "\\" not in cmds[1] and set(s["hooks"]) == set(install.EVENTS)
    assert (tmp_path / "settings.json.voice-backup").exists()


def test_install_creates_missing_settings_and_uninstall_cleans(tmp_path):
    p = tmp_path / "sub" / "settings.json"
    install.update_settings(p)
    assert "hooks" in json.loads(p.read_text())
    install.update_settings(p, remove_only=True)
    assert json.loads(p.read_text()) == {}


def test_install_refuses_to_overwrite_broken_settings(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{ not json")
    try:
        install.update_settings(p)
    except ValueError:
        pass
    else:
        raise AssertionError("broken settings.json must not be overwritten")
    assert p.read_text() == "{ not json"


def test_pick_pane():
    rows = ["%0\tzsh\t100", "%2\tclaude\t200", "%3\tnode\t300"]
    assert pick_pane(rows) == "%2" and pick_pane(["%0\tvim\t1"]) == ""


def test_tmux_keys_literal_text(monkeypatch):
    calls = []

    class R:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr("voice_agent.agents.tmux_input._tmux", lambda *a, **k: calls.append(a) or R())
    k = TmuxKeys("%2")
    k.write("run ls\n")
    k.send("shift+tab")
    assert calls[0] == ("send-keys", "-t", "%2", "-l", "--", "run ls ") and calls[1][-1] == "BTab"


def test_speech_command_per_os(monkeypatch):
    cfg = {"rate": 0, "voice": ""}
    monkeypatch.setattr(tts, "_IS_WIN", False)
    monkeypatch.setattr(tts.sys, "platform", "darwin")
    assert tts._speech_command("/tmp/x.txt", cfg)[0][0] == "say"
    monkeypatch.setattr(tts.sys, "platform", "linux")
    monkeypatch.setattr(tts.shutil, "which", lambda e: "/usr/bin/" + e if e == "espeak-ng" else None)
    assert tts._speech_command("/tmp/x.txt", cfg)[0][0] == "espeak-ng"
    monkeypatch.setattr(tts.shutil, "which", lambda e: None)
    try:
        tts._speech_command("/tmp/x.txt", cfg)
    except RuntimeError as e:
        assert "espeak" in str(e)
    else:
        raise AssertionError("expected a clear error when no TTS program exists")


def test_console_keys_retargets_on_stale_pid(monkeypatch):
    from types import SimpleNamespace
    from voice_agent.agents import console_input as ci

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[3])
        if cmd[3] == "111":  # stale PID -> AttachConsole failure
            return SimpleNamespace(returncode=2, stdout="AttachConsole(111) failed: error 87", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ci.subprocess, "run", fake_run)
    monkeypatch.setattr(ci, "find_cli_pid", lambda: 222)
    keys = ci.ConsoleKeys(111)
    keys.write("hi")
    assert keys.pid == 222 and calls == ["111", "222"]


def test_install_and_remove_voice_slash_command(tmp_path):
    import install
    settings = tmp_path / "settings.json"
    install.update_command(settings)
    cmd = (tmp_path / "commands" / "voice.md").read_text(encoding="utf-8")
    assert "voice.py $ARGUMENTS" in cmd and "\\" not in cmd.split("---")[2]
    install.update_command(settings, remove_only=True)
    assert not (tmp_path / "commands" / "voice.md").exists()
