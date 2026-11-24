import pytest


@pytest.fixture(autouse=True)
def no_real_speech_control(monkeypatch):
    """Tests must never kill real processes via the pid file."""
    monkeypatch.setattr("voice_agent.audio.tts.stop_speaking", lambda: None)


@pytest.fixture(autouse=True)
def isolated_mode_file(monkeypatch, tmp_path):
    """The spoken-mode file must not leak between tests or into the real app."""
    monkeypatch.setattr("voice_agent.config.MODE_PATH", tmp_path / "mode.json")
