"""One-step setup on any PC:  python install.py   (add --uninstall to remove the hooks again).

1. installs the Python packages          (skip with --no-pip)
2. creates config.json from the example  (only if you do not have one)
3. adds the voice hooks to ~/.claude/settings.json, using THIS folder's real path (other hooks are kept),
   and installs the /voice slash command (start | stop | restart | status | tray)
4. downloads the speech model            (skip with --no-model)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVENTS = {"Stop": "stop", "UserPromptSubmit": "hush", "Notification": "notify", "SessionStart": "start"}
MARKER = "speak_hook.py"


def _posix(p: Path | str) -> str:
    return str(p).replace("\\", "/")  # Claude runs hooks through a POSIX shell (Git Bash on Windows)


def hook_command(action: str) -> str:
    return f'"{_posix(sys.executable)}" "{_posix(ROOT / MARKER)}" {action}'


def merge_hooks(settings: dict, remove_only: bool = False) -> dict:
    """Return `settings` with our four hooks added (or removed); every other hook is left untouched."""
    hooks = settings.setdefault("hooks", {})
    for event, action in EVENTS.items():
        groups = []
        for group in hooks.get(event, []):
            kept = [h for h in group.get("hooks", []) if MARKER not in str(h.get("command", ""))]
            if kept:
                groups.append({**group, "hooks": kept})
        if not remove_only:
            groups.append({"hooks": [{"type": "command", "command": hook_command(action)}]})
        if groups:
            hooks[event] = groups
        else:
            hooks.pop(event, None)
    if not hooks:
        settings.pop("hooks", None)
    return settings


def update_settings(path: Path, remove_only: bool = False) -> None:
    settings = {}
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if text.strip():
            settings = json.loads(text)  # a broken file raises here instead of being overwritten
        shutil.copy2(path, path.with_name(path.name + ".voice-backup"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merge_hooks(settings, remove_only), indent=2) + "\n", encoding="utf-8")


VOICE_COMMAND = """---
description: Control the voice layer (start, stop, restart, status, tray)
argument-hint: start | stop | restart | status | tray
allowed-tools: Bash(python {voice}:*)
---
Run `python {voice} $ARGUMENTS` (use `status` if no argument was given) and reply with its one-line output only.
"""


def command_path(settings: Path) -> Path:
    return settings.parent / "commands" / "voice.md"


def update_command(settings: Path, remove_only: bool = False) -> None:
    """Install (or remove) the /voice slash command next to settings.json."""
    path = command_path(settings)
    if remove_only:
        if path.exists() and "voice.py" in path.read_text(encoding="utf-8"):
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(VOICE_COMMAND.format(voice=_posix(ROOT / "voice.py")), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", type=Path, default=Path.home() / ".claude" / "settings.json")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    args = ap.parse_args(argv)

    if args.uninstall:
        update_settings(args.settings, remove_only=True)
        update_command(args.settings, remove_only=True)
        print(f"Voice hooks and /voice command removed from {args.settings.parent}")
        return 0

    if sys.version_info < (3, 9):
        print("Python 3.9 or newer is required.")
        return 1
    if not args.no_pip:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    cfg_path, example = ROOT / "config.json", ROOT / "config.example.json"
    if not cfg_path.exists() and example.exists():
        shutil.copy2(example, cfg_path)
        print("Created config.json")
    update_settings(args.settings)
    update_command(args.settings)
    print(f"/voice command installed in {command_path(args.settings)}")
    print(f"Hooks installed in {args.settings} (backup: settings.json.voice-backup)")
    if not args.no_model:
        sys.path.insert(0, str(ROOT))
        from voice_agent.config import load_config
        from voice_agent.stt.whisper import Transcriber
        cfg = load_config()
        print(f"Downloading speech model '{cfg['whisper_model']}' (first run only)...")
        Transcriber(cfg["whisper_model"], cfg["whisper_compute_type"]).load()
    print("Done. Open Claude Code and run:  python voice.py status")
    if sys.platform != "win32":
        print("macOS/Linux: run `claude` inside tmux so voice can type into it; Linux also needs espeak-ng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
