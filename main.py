"""Shortcut: `python main.py` starts the voice agent, same as `python run.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_agent.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
