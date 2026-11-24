"""Spoken control commands. Only a whole short utterance counts, so "brief overview of X" is a prompt."""
from __future__ import annotations

import difflib
import re

# name -> phrases (whole-utterance match after lowercasing and stripping punctuation)
_COMMANDS: dict[str, tuple[str, ...]] = {
    "repeat": ("repeat that", "repeat", "say that again", "say it again", "what did you say"),
    "stop": ("stop", "stop it", "skip", "be quiet", "quiet", "stop talking", "skip that",
             "enough", "that s enough", "thats enough", "hush"),
    "mute": ("mute", "mute mic", "mute the mic", "go quiet", "mute yourself"),  # close the conversation, stay muted
    "esc": ("esc", "escape", "press escape", "hit escape", "press esc"),
    "chat": ("chat", "chart", "chat mode", "let s chat", "lets chat", "let chat", "let chart", "let s chart", "lets chart", "let us chat", "let us chart",
             "change the plan"),  # reject the plan, say what to change
    "listen": ("listen", "listen to me", "start listening", "start chat", "start chatting", "unmute", "un mute",
               "unmute mic", "unmute the mic"),  # open the conversation
    "brief": ("brief", "be brief", "keep it short", "key points", "just the key points",
              "important points", "the important points", "only important points"),
    "summary": ("summary", "short summary", "summarize", "summarise"),
    "detail": ("explain in detail", "in detail", "full", "full script", "read it all", "read it exactly",
               "normal", "read everything", "read the full thing"),
    "new_session": ("new session", "start over", "new conversation"),
    "mode_auto": ("auto mode", "auto", "switch to auto mode", "change to auto mode"),
    "mode_plan": ("plan mode", "plan", "read only mode", "read only", "switch to plan mode",
                  "change to plan mode"),
    "mode_accept": ("accept edits on mode", "accept edits on", "accept edits mode", "accept edits",
                    "switch to accept edits on mode", "change to accept edits on mode",
                    "auto accept", "auto accept mode", "auto accept edits"),
    "mode_manual": ("manual mode", "manual", "switch to manual mode", "change to manual mode",
                    "write mode"),
    "mode_query": ("what mode", "which mode", "current mode", "what mode are you in", "what mode is it",
                   "which mode am i in", "what mode is claude in", "current mode please", "mode"),
    "cancel": ("cancel", "cancel that", "never mind", "nevermind", "no"),
}

_LOOKUP = {phrase: name for name, phrases in _COMMANDS.items() for phrase in phrases}

_FUZZY = [p for n, ps in _COMMANDS.items() if n.startswith("mode_") for p in ps]  # never "stop": too risky


def normalize(text: str) -> str:
    t = re.sub(r"[^a-z0-9\s-]", " ", text.lower())
    return re.sub(r"[\s-]+", " ", t).strip()


_FILLER = re.compile(r"^(please )?(change|switch|go|set|turn)( it)?( to| on)?( the)? ")
_MODE_WORDS = {"mode_auto": ("auto",), "mode_plan": ("plan", "read only"),
               "mode_accept": ("accept edits", "accept edits on"), "mode_manual": ("manual",)}


def _dedupe(t: str) -> str:
    """Whisper sometimes stutters ("manual manual"): collapse repeated neighbouring words."""
    words = t.split()
    return " ".join(w for i, w in enumerate(words) if i == 0 or w != words[i - 1])


def strip_wake_word(text: str, wake) -> str | None:
    """`wake` is one phrase or a list of them; the first phrase that matches wins."""
    phrases = [wake] if isinstance(wake, str) else [w for w in (wake or []) if w]
    if not phrases:
        return text
    for w in phrases:
        rest = _strip_one(text, w)
        if rest is not None:
            return rest
    return None


def _strip_one(text: str, wake: str) -> str | None:
    """Text after the wake phrase ("hey claude list files" -> "list files"), or None if absent."""
    w, t = normalize(wake), normalize(text)
    if not w:
        return text
    if t == w or t.startswith(w + " "):
        words = len(w.split())
        return " ".join(re.sub(r"^[^\w]+", "", text).split()[words:]).strip(" ,.:;!") or ""
    # Whisper often mishears the name ("hey jarvas", "jarvis" without "hey"): accept sound-alikes.
    words = re.sub(r"^[^\w]+", "", text).split()
    wl = w.split()
    for n in dict.fromkeys((len(wl), 1)):
        cand = normalize(" ".join(words[:n])) if n <= len(words) else ""
        if len(wl) == 1 and abs(len(cand) - len(w)) > 1:
            continue  # one-word wake phrase: "list" must not pass for "listen"
        if cand and difflib.SequenceMatcher(None, cand, w).ratio() >= (0.7 if len(wl) == 1 and len(w) >= 5 else 0.75):
            return " ".join(words[n:]).strip(" ,.:;!") or ""
    name = wl[-1]
    if len(wl) > 1 and len(name) >= 4 and words and \
            difflib.SequenceMatcher(None, normalize(words[0]), name).ratio() >= 0.75:
        return " ".join(words[1:]).strip(" ,.:;!") or ""
    return None


_CHAIN_SPLIT = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b|\bafter that\b|\band\b)\s*")
_CHAINABLE = ("mode_", "brief", "detail", "summary", "new_session")


def parse_chain(text: str) -> list[str] | None:
    """"plan mode, then auto mode" -> ["mode_plan", "mode_auto"]. None unless EVERY part is a command
    (so a normal sentence containing "and" is never split), and at least two parts."""
    parts = [p for p in _CHAIN_SPLIT.split(text.lower()) if p.strip(" .!?")]
    if len(parts) < 2 or len(parts) > 4:
        return None
    names = [parse_command(re.sub(r"^(to|the)\s+", "", p.strip())) for p in parts]
    if any(n is None or not n.startswith(_CHAINABLE) for n in names):
        return None
    return names  # type: ignore[return-value]


def parse_command(text: str) -> str | None:
    """Return the command name, or None if this is a normal prompt for the agent."""
    t = normalize(text).replace("automode", "auto mode")
    if t in _LOOKUP:
        return _LOOKUP[t]
    t = _dedupe(t)
    if t in _LOOKUP:
        return _LOOKUP[t]
    if _FILLER.match(t):  # "change to manual mode", "switch to plan", "set auto mode"
        rest = _FILLER.sub("", t)
        for cand in (rest, rest.removesuffix(" mode").strip(), rest + " mode"):
            if _LOOKUP.get(cand, "").startswith("mode_"):
                return _LOOKUP[cand]
        for name, words in _MODE_WORDS.items():
            if rest.removesuffix(" mode").strip() in words:
                return name
    if len(t.split()) <= 3:  # misheard short mode/stop words: "plan more" -> plan mode
        close = difflib.get_close_matches(t, _FUZZY, n=1, cutoff=0.85)
        if close:
            return _LOOKUP[close[0]]
    return None


_APPROVE = re.compile(r"^(yes )?(approve|allow|grant|go ahead|do it)( it| that)?$")


def is_approval(text: str) -> bool:
    """Bare "yes" is not enough: a misheard word must not grant a permission."""
    return bool(_APPROVE.match(normalize(text)))
