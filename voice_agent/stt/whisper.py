"""Local speech-to-text with faster-whisper (CPU, int8)."""
from __future__ import annotations

import threading

import numpy as np

CODE_VOCAB = ("Claude Code, pytest, git, npm, pip, refactor, repo, README, JSON, TypeScript, "
              "Python, commit, branch, function, API, config, bug, test.")


class Transcriber:
    def __init__(self, model: str = "small.en", compute_type: str = "int8", hint: str = ""):
        self.model_name, self.compute_type = model, compute_type
        self.prompt = f"{hint.title()}. {CODE_VOCAB}" if hint else CODE_VOCAB  # nudges Whisper to spell the wake word
        self._model = None
        self._lock = threading.Lock()  # the monitor and the main loop may both transcribe

    def load(self) -> None:
        if self._model is None:  # the first load can download the model (slow)
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device="cpu", compute_type=self.compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        with self._lock:
            self.load()
            segments, _ = self._model.transcribe(
                audio, language="en", initial_prompt=self.prompt, beam_size=1, vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()
