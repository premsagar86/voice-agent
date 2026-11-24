"""System-tray front end: run the voice loop in a thread and control it from a tray menu."""
from __future__ import annotations

import os
import sys
import threading

from .audio import tts
from .config import CONFIG_PATH, load_mode, log, save_mode

COLORS = {"idle": (90, 90, 90), "listening": (40, 160, 70), "transcribing": (200, 160, 30),
          "working": (200, 160, 30), "summarizing": (200, 160, 30), "speaking": (50, 110, 210)}


def _icon(state: str):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (64, 64), (30, 30, 30))
    ImageDraw.Draw(img).ellipse((10, 10, 54, 54), fill=COLORS.get(state, COLORS["idle"]))
    return img


def run_tray(loop, cfg: dict) -> int:
    try:
        import pystray
    except ImportError:
        print("The tray needs: pip install pystray pillow", file=sys.stderr)
        return 1

    running = threading.Event()
    running.set()
    quit_flag = threading.Event()

    def worker() -> None:
        while not quit_flag.is_set():
            if not running.is_set():
                quit_flag.wait(0.3)
                continue
            try:
                loop.run_turn()
            except Exception as e:  # keep the tray alive, and keep the reason
                log(f"tray loop error: {e!r}")
                quit_flag.wait(1.0)

    def set_read_mode(mode: str):
        def _do(icon, item):
            cfg["read_mode"] = mode
            save_mode(mode)
        return _do

    def is_mode(mode: str):
        return lambda item: (load_mode() or cfg.get("read_mode")) == mode

    def toggle(icon, item) -> None:
        if running.is_set():
            running.clear()
            tts.stop_speaking()
        else:
            running.set()

    def stop_now(icon, item) -> None:
        loop.stop_now()

    def open_config(icon, item) -> None:
        os.startfile(str(CONFIG_PATH))  # type: ignore[attr-defined]

    def quit_app(icon, item) -> None:
        quit_flag.set()
        loop.stop_now()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Listening", toggle, checked=lambda item: running.is_set()),
        pystray.MenuItem("Stop speaking", stop_now),
        pystray.MenuItem("Read mode", pystray.Menu(
            pystray.MenuItem("Full", set_read_mode("full"), checked=is_mode("full"), radio=True),
            pystray.MenuItem("Summary", set_read_mode("summary"), checked=is_mode("summary"), radio=True),
            pystray.MenuItem("Key points", set_read_mode("keypoints"), checked=is_mode("keypoints"), radio=True))),
        pystray.MenuItem("Open config", open_config),
        pystray.MenuItem("Quit", quit_app),
    )
    icon = pystray.Icon("voice-agent", _icon("idle"), "Voice agent", menu)

    original_set = loop._set

    def set_with_icon(state: str) -> None:  # the tray icon colour follows the loop state
        original_set(state)
        try:
            icon.icon = _icon(state)
            icon.title = f"Voice agent: {state}"
        except Exception:
            pass

    loop._set = set_with_icon  # type: ignore[method-assign]
    threading.Thread(target=worker, daemon=True).start()
    icon.run()
    return 0
