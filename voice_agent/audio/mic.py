"""Microphone capture (16 kHz mono float32): push-to-talk, hands-free, and a barge-in monitor."""
from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Iterable

import numpy as np

SAMPLE_RATE = 16000
BLOCK = 480  # 30 ms


def record_while_pressed(hotkey: str = "f9", min_seconds: float = 0.3,
                         on_start: Callable[[], None] | None = None) -> np.ndarray:
    """Block until `hotkey` is pressed, record while held, return audio (empty if too short)."""
    import keyboard
    import sounddevice as sd

    while not keyboard.is_pressed(hotkey):
        time.sleep(0.02)
    if on_start:
        on_start()
    chunks: list[np.ndarray] = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=lambda indata, *_: chunks.append(indata.copy())):
        while keyboard.is_pressed(hotkey):
            time.sleep(0.02)
    if not chunks:
        return np.zeros(0, dtype="float32")
    audio = np.concatenate(chunks).reshape(-1)
    return audio if len(audio) >= min_seconds * SAMPLE_RATE else np.zeros(0, dtype="float32")


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0


class ClapDetector:
    """Double-clap detector over 30 ms blocks. A clap is a very loud burst of at most ~120 ms; two of them
    0.2 to 1.2 s apart count. Speech and long noises are too long or too quiet to match."""

    def __init__(self, threshold: float = 0.15, max_burst: int = 4, gap=(0.2, 1.2)):
        self.threshold, self.max_burst, self.gap = threshold, max_burst, gap
        self.t = 0.0  # time in seconds, advanced by the blocks themselves (testable without a clock)
        self.burst = 0
        self.last_clap = -99.0
        self.fired = False

    def feed(self, block: np.ndarray) -> bool:
        """Process one block; True when a double clap was just completed."""
        self.t += BLOCK / SAMPLE_RATE
        if rms(np.asarray(block, dtype="float32").reshape(-1)) >= self.threshold:
            self.burst += 1
            return False
        done, self.burst = self.burst, 0
        if not 1 <= done <= self.max_burst:
            return False  # nothing, or too long to be a clap
        if self.gap[0] <= self.t - self.last_clap <= self.gap[1]:
            self.last_clap, self.fired = -99.0, True
            return True
        self.last_clap = self.t
        return False

    def consume(self) -> bool:
        """True once per detected double clap."""
        fired, self.fired = self.fired, False
        return fired


def segment_speech(blocks: Iterable[np.ndarray], threshold: float = 0.02, silence_s: float = 1.0,
                   max_s: float = 30.0, preroll_blocks: int = 8, min_s: float = 0.3,
                   clapper: "ClapDetector | None" = None) -> np.ndarray:
    """Pure energy-based voice detection over a stream of 30 ms blocks (testable without a mic).

    Waits for a block above `threshold`, keeps recording until `silence_s` of quiet follows
    (or `max_s`), and returns the utterance. Returns empty audio if the stream ends first.
    """
    # A click or knock is 1-2 loud blocks; speech is a sustained run. Blips must neither start a recording
    # nor keep it open after you finish talking.
    sustain = 3  # consecutive loud blocks (~90 ms) that count as real sound
    pre: list[np.ndarray] = []
    voiced: list[np.ndarray] = []
    quiet = 0.0
    speaking = False
    run = 0  # current streak of loud blocks
    loud_blocks = 0  # loud blocks in the utterance, to reject recordings that were only noise
    for b in blocks:
        b = np.asarray(b, dtype="float32").reshape(-1)
        if clapper is not None and clapper.feed(b):  # a double clap: stop recording, the caller handles it
            return np.zeros(0, dtype="float32")
        loud = rms(b) >= threshold
        run = run + 1 if loud else 0
        if not speaking:
            pre.append(b)
            pre = pre[-(preroll_blocks + sustain):]
            if run >= sustain:
                speaking, voiced, quiet, loud_blocks = True, list(pre), 0.0, run
            continue
        voiced.append(b)
        if loud:
            loud_blocks += 1
        quiet = 0.0 if run >= sustain else quiet + len(b) / SAMPLE_RATE
        if quiet >= silence_s or len(voiced) * BLOCK / SAMPLE_RATE >= max_s:
            break
    if not voiced or loud_blocks * BLOCK / SAMPLE_RATE < min_s * 0.7:
        return np.zeros(0, dtype="float32")
    audio = np.concatenate(voiced)
    return audio if len(audio) >= min_s * SAMPLE_RATE else np.zeros(0, dtype="float32")


def _mic_blocks():
    import sounddevice as sd

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK) as stream:
        while True:
            data, _ = stream.read(BLOCK)
            yield data


def record_auto(threshold: float = 0.02, silence_s: float = 1.0, clapper: "ClapDetector | None" = None) -> np.ndarray:
    """Hands-free: block until someone speaks, return the utterance (empty after a double clap)."""
    return segment_speech(_mic_blocks(), threshold, silence_s, clapper=clapper)


class StopHotkey:
    """Global hotkey (default F10) that calls `on_stop`, from any window. Silently off if unavailable."""

    def __init__(self, on_stop: Callable[[], None], key: str = "f10"):
        self.on_stop, self.key, self._handle = on_stop, key, None

    def __enter__(self):
        try:
            import keyboard
            self._handle = keyboard.add_hotkey(self.key, self.on_stop)
        except Exception:
            self._handle = None
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            try:
                import keyboard
                keyboard.remove_hotkey(self._handle)
            except Exception:
                pass
            self._handle = None


class _MicListener:
    """One callback-based input stream feeding a queue that a worker thread consumes.

    The stream is opened and closed on the caller's thread (never from the worker), and `__exit__`
    joins the worker, so PortAudio is never touched from two threads at once (that crashed the app).
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._thread: threading.Thread | None = None

    def _handle(self, block: np.ndarray) -> bool:
        """Process one 30 ms block; return True to stop listening."""
        raise NotImplementedError

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    block = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._handle(block):
                    return
        except Exception:  # never let a listener take the app down
            return

    def __enter__(self):
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
                callback=lambda indata, *_: self._q.put(indata.copy().reshape(-1)))
            self._stream.start()
        except Exception:  # no mic / device busy: listening simply stays off
            self._stream = None
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class StopWordMonitor(_MicListener):
    """While the agent speaks or works, listen for the spoken word "stop" and call `on_stop`.

    Other noise or speech is ignored: each loud utterance is transcribed and only a whole-utterance
    stop command counts. Keeps listening after a false alarm.
    """

    def __init__(self, on_stop: Callable[[], None], transcribe: Callable[[np.ndarray], str],
                 threshold: float = 0.03, silence_s: float = 0.5, max_s: float = 3.0):
        super().__init__()
        self.on_stop, self.transcribe = on_stop, transcribe
        self.threshold, self.silence_s, self.max_s = threshold, silence_s, max_s
        self._pre: list[np.ndarray] = []
        self._voiced: list[np.ndarray] = []
        self._quiet = 0.0

    def _handle(self, block: np.ndarray) -> bool:
        from ..commands import parse_command

        loud = rms(block) >= self.threshold
        if not self._voiced:
            self._pre = (self._pre + [block])[-8:]
            if loud:
                self._voiced, self._quiet = list(self._pre), 0.0
            return False
        self._voiced.append(block)
        self._quiet = 0.0 if loud else self._quiet + len(block) / SAMPLE_RATE
        if self._quiet < self.silence_s and len(self._voiced) * BLOCK / SAMPLE_RATE < self.max_s:
            return False
        audio, self._voiced, self._pre = np.concatenate(self._voiced), [], []
        if self._stop.is_set():
            return True
        if parse_command(self.transcribe(audio) or "") == "stop":
            self.on_stop()
            return True
        return False


class BargeInMonitor(_MicListener):
    """While the agent speaks, watch the mic; loud sound for `hold_s` calls `on_speech` once.

    Use headphones: without echo cancellation the speakers can trigger this.
    """

    def __init__(self, on_speech: Callable[[], None], threshold: float = 0.08, hold_s: float = 0.3):
        super().__init__()
        self.on_speech, self.threshold, self.hold_s = on_speech, threshold, hold_s
        self._loud_for = 0.0

    def _handle(self, block: np.ndarray) -> bool:
        self._loud_for = self._loud_for + BLOCK / SAMPLE_RATE if rms(block) >= self.threshold else 0.0
        if self._loud_for >= self.hold_s:
            self.on_speech()
            return True
        return False
