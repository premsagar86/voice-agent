"""Claude Code adapter: headless `claude -p` with stream-json output and --resume."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

from ..config import GUARD_ENV, SESSIONS_PATH
from .base import Done, Error, Event, Text, ToolUse


def parse_event(line: str) -> list[Event]:
    """Map one stream-json line to zero or more events."""
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict):
        return []
    kind = obj.get("type")
    if kind == "assistant":
        out: list[Event] = []
        content = (obj.get("message") or {}).get("content") or []
        for b in content if isinstance(content, list) else []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and b.get("text", "").strip():
                out.append(Text(b["text"]))
            elif b.get("type") == "tool_use":
                inp = b.get("input") or {}
                hint = next((str(inp[k]) for k in ("file_path", "command", "pattern", "path") if k in inp), "")
                out.append(ToolUse(b.get("name", "tool"), hint[:120]))
        return out
    if kind == "result":
        if obj.get("is_error"):
            return [Error(str(obj.get("result") or obj.get("subtype") or "agent error"))]
        return [Done(str(obj.get("result") or ""), str(obj.get("session_id") or ""),
                     parse_denials(obj.get("permission_denials")))]
    return []


def parse_denials(raw) -> tuple:
    """permission_denials from the result event -> ((allow_spec, spoken_label), ...)."""
    out = []
    for d in raw if isinstance(raw, list) else []:
        if not isinstance(d, dict) or not d.get("tool_name"):
            continue
        name, inp = str(d["tool_name"]), d.get("tool_input") or {}
        if name == "Bash" and inp.get("command"):
            cmd = str(inp["command"])
            out.append((f"Bash({cmd})", f"run the command {cmd[:80]}"))
        else:
            hint = next((str(inp[k]) for k in ("file_path", "path", "pattern") if k in inp), "")
            out.append((name, f"use {name} {hint[:80]}".strip()))
    return tuple(out)


def load_sessions(path: Path = SESSIONS_PATH) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_session(cwd: str, session_id: str, path: Path = SESSIONS_PATH) -> None:
    data = load_sessions(path)
    data[cwd] = session_id
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class ClaudeCodeAdapter:
    def __init__(self, cwd: str | None = None, command: list[str] | None = None,
                 timeout: int = 600, sessions_path: Path = SESSIONS_PATH):
        self.cwd = str(Path(cwd or os.getcwd()).resolve())
        self.command = command or [shutil.which("claude") or "claude"]
        self.timeout = timeout
        self.sessions_path = sessions_path
        self.session_id = load_sessions(sessions_path).get(self.cwd, "")
        self._proc: subprocess.Popen | None = None
        self._interrupted = False
        self.grant: list[str] = []  # tools approved by voice, allowed for the next request only
        self.permission_mode = ""  # "", "auto", "plan", "acceptEdits" or "manual" (passed to --permission-mode)

    def new_session(self) -> None:
        self.session_id = ""

    def send(self, prompt: str) -> Iterator[Event]:
        args = [*self.command, "-p", "--output-format", "stream-json", "--verbose"]
        if self.permission_mode:
            args += ["--permission-mode", self.permission_mode]
        if self.grant:
            args += ["--allowedTools", *self.grant]
            self.grant = []
        if self.session_id:
            args += ["--resume", self.session_id]
        env = dict(os.environ, **{GUARD_ENV: "1"})  # the Stop hook must not also speak
        self._interrupted = False
        try:
            self._proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", cwd=self.cwd, env=env,
            )
            proc = self._proc
            proc.stdin.write(prompt)  # via stdin: multi-line args break on Windows
            proc.stdin.close()
            finished = False
            for line in proc.stdout:
                if self._interrupted:
                    break  # user said stop: drop the rest of the reply
                for ev in parse_event(line):
                    if isinstance(ev, Done):
                        finished = True
                        if ev.session_id:
                            self.session_id = ev.session_id
                            save_session(self.cwd, ev.session_id, self.sessions_path)
                    yield ev
            proc.wait(timeout=self.timeout)
            if not finished and not self._interrupted:
                yield Error(proc.stderr.read().strip()[:300] or f"agent exited with code {proc.returncode}")
        except (OSError, subprocess.SubprocessError) as e:
            yield Error(f"could not run agent: {e}")
        finally:
            self._proc = None

    def interrupt(self) -> None:
        self._interrupted = True
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
