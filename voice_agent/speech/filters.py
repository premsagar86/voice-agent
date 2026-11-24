"""Make agent text speakable: drop code, markdown, URLs, paths, diffs, tables."""
from __future__ import annotations

import re

_FENCE = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`([^`]*)`")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.\-@]+){2,}")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _unfence(m: re.Match) -> str:
    return "\n".join(l for l in m.group(0).splitlines() if not l.strip().startswith("```"))


def clean_for_speech(text: str, max_chars: int = 300, code: str = "drop") -> str:
    """Strip things that sound terrible. max_chars<=0 keeps everything (full read).

    code: "drop" removes code blocks, "note" says that a block was skipped, "read" keeps the code text.
    """
    fence_sub = {"drop": " ", "note": " Code block skipped. "}.get(code)
    t = _FENCE.sub(_unfence if fence_sub is None else fence_sub, text)
    t = _LINK.sub(r"\1", t)
    t = _URL.sub(" ", t)
    t = _INLINE.sub(r"\1", t)
    t = _PATH.sub(lambda m: re.split(r"[\\/]", m.group(0))[-1], t)
    kept = []
    for line in t.splitlines():
        s = line.strip()
        if not s or set(s) <= set("|-+=:_ "):  # blanks, table rules, hr
            continue
        if s.startswith(("diff ", "@@", "+++", "---", "index ")):
            continue
        s = re.sub(r"^\s*(#{1,6}|[-*>]|\d+[.)])\s+", "", s)  # headings, bullets, quotes
        s = s.replace("|", ", ")
        s = re.sub(r"[*_~#>]+", "", s)
        kept.append(s)
    t = re.sub(r"\s+", " ", " ".join(kept)).strip()
    if not t:
        return ""
    if max_chars <= 0:
        return t
    out = ""
    for sentence in _SENT.split(t):
        if len(out) + len(sentence) > max_chars and out:
            break
        out = f"{out} {sentence}".strip()
        if len(out) >= max_chars:
            break
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "."
    return out
