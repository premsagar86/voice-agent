"""Launcher: python run.py (run from the project Claude should work in)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_agent.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
