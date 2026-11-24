"""Common agent interface and event types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol


@dataclass
class Text:
    text: str


@dataclass
class ToolUse:
    name: str
    summary: str = ""


@dataclass
class Error:
    message: str


@dataclass
class Done:
    result: str  # final assistant message
    session_id: str = ""
    denials: tuple = ()  # ((allow_spec, spoken_label), ...) tools refused for lack of permission


Event = Text | ToolUse | Error | Done


class AgentAdapter(Protocol):
    def send(self, prompt: str) -> Iterator[Event]:
        """Send a prompt and yield events until Done (or Error)."""

    def interrupt(self) -> None:
        """Stop the running request, if any."""
