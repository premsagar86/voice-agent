"""Safety guard: risky spoken requests need an explicit spoken "confirm" on the next turn."""
from __future__ import annotations

import re

from .commands import normalize

_DESTRUCTIVE = [
    r"\brm\s+-?r?f?\b", r"\brm rf\b", r"\bdelete\b", r"\bremove (all|everything|the (whole )?(folder|directory|repo|repository|database))\b",
    r"\bwipe\b", r"\berase\b", r"\bformat (the )?(drive|disk|c|d)\b", r"\bdrop (table|database)\b",
    r"\bforce[ -]?push\b", r"\bpush (with )?--?force\b", r"\bgit reset (--)?hard\b", r"\bhard reset\b",
    r"\btruncate\b", r"\bshut ?down\b", r"\buninstall\b",
]
_PATTERN = re.compile("|".join(_DESTRUCTIVE))

# Bare "yes" is deliberately NOT enough: a misheard word must never approve a destructive action.
_CONFIRM = re.compile(r"^(yes )?confirm( (delete|it|that|the action|action|go ahead))?$")


def is_destructive(text: str) -> bool:
    return bool(_PATTERN.search(normalize(text)))


def is_confirmation(text: str) -> bool:
    return bool(_CONFIRM.match(normalize(text)))
