"""Control the voice app: voice start | stop | restart | status | tray"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_agent.lifecycle import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv))
