"""Fake `claude` CLI for contract tests: prints stream-json like `claude -p --output-format stream-json`.

Usage as the adapter command: [sys.executable, fake_agent.py]. Reads the prompt from stdin.
"""
import json
import sys

prompt = sys.stdin.read()
resume = sys.argv[sys.argv.index("--resume") + 1] if "--resume" in sys.argv else ""
print(json.dumps({"type": "system", "subtype": "init", "session_id": "sess-1"}))
print(json.dumps({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Read", "input": {"file_path": "D:/proj/auth.py"}}]}}))
print(json.dumps({"type": "assistant", "message": {"content": [
    {"type": "text", "text": f"Echo: {prompt.strip()} resume={resume}"}]}}))
print(json.dumps({"type": "result", "is_error": False,
                  "result": f"Echo: {prompt.strip()} resume={resume}", "session_id": "sess-1"}))
