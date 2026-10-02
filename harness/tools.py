"""Sandboxed tools + the text protocol agents use to call them.

Agents emit fenced blocks:

    ```tool
    {"name": "read_file", "arguments": {"path": "notes/todo.txt"}}
    ```

and inter-agent messages as:

    ```message
    {"to": "coder", "text": "the API is ready, you can wire the UI now"}
    ```

Everything file-related is jailed inside the workspace root.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Tuple

import requests

TOOL_BLOCK_RE = re.compile(r"```tool\s*\n(.*?)```", re.DOTALL)
MESSAGE_BLOCK_RE = re.compile(r"```message\s*\n(.*?)```", re.DOTALL)
STRIP_BLOCKS_RE = re.compile(r"```(?:tool|message)\s*\n.*?```", re.DOTALL)


def parse_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract (tool_name, arguments) pairs from ```tool blocks."""
    calls = []
    for m in TOOL_BLOCK_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1).strip())
            name = obj.get("name", "")
            args = obj.get("arguments", {}) or {}
            if name:
                calls.append((name, args))
        except json.JSONDecodeError:
            continue
    return calls


def parse_messages(text: str) -> List[Tuple[str, str]]:
    """Extract (recipient, text) pairs from ```message blocks."""
    out = []
    for m in MESSAGE_BLOCK_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1).strip())
            to, msg = obj.get("to", ""), obj.get("text", "")
            if to and msg:
                out.append((to, msg))
        except json.JSONDecodeError:
            continue
    return out


def strip_protocol_blocks(text: str) -> str:
    """Remove tool/message blocks so final answers read cleanly."""
    return STRIP_BLOCKS_RE.sub("", text or "").strip()


class ToolError(Exception):
    pass


class Tool:
    def __init__(self, name: str, description: str, usage: str, func: Callable[..., str]):
        self.name = name
        self.description = description
        self.usage = usage  # short JSON example for the prompt
        self.func = func


class ToolRegistry:
    def __init__(self, workspace: str):
        self.root = os.path.abspath(workspace)
        os.makedirs(self.root, exist_ok=True)
        self.tools: Dict[str, Tool] = {}
        self._register_defaults()

    # -- sandbox ---------------------------------------------------------
    def _safe(self, path: str) -> str:
        """Resolve a workspace-relative path, rejecting escapes."""
        if os.path.isabs(path):
            raise ToolError("absolute paths are not allowed; use workspace-relative paths")
        full = os.path.abspath(os.path.join(self.root, path))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise ToolError(f"path escapes the workspace: {path!r}")
        return full

    # -- tool implementations --------------------------------------------
    def _read_file(self, path: str) -> str:
        full = self._safe(path)
        if not os.path.isfile(full):
            raise ToolError(f"file not found: {path}")
        with open(full, encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 60000:
            content = content[:60000] + "\n...[truncated at 60k chars]..."
        return content

    def _write_file(self, path: str, content: str = "") -> str:
        full = self._safe(path)
        os.makedirs(os.path.dirname(full) or self.root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} chars to {path}"

    def _edit_file(self, path: str, old: str, new: str) -> str:
        full = self._safe(path)
        if not os.path.isfile(full):
            raise ToolError(f"file not found: {path}")
        with open(full, encoding="utf-8") as f:
            content = f.read()
        count = content.count(old)
        if count == 0:
            raise ToolError("`old` text not found in file")
        if count > 1:
            raise ToolError(f"`old` text matches {count} places; make it unique")
        with open(full, "w", encoding="utf-8") as f:
            f.write(content.replace(old, new, 1))
        return f"edited {path} (1 replacement)"

    def _list_dir(self, path: str = ".") -> str:
        full = self._safe(path)
        if not os.path.isdir(full):
            raise ToolError(f"not a directory: {path}")
        entries = []
        for e in sorted(os.listdir(full)):
            ep = os.path.join(full, e)
            entries.append(e + ("/" if os.path.isdir(ep) else ""))
        return "\n".join(entries) or "(empty)"

    def _delete_file(self, path: str) -> str:
        full = self._safe(path)
        if os.path.isdir(full):
            shutil.rmtree(full)
            return f"deleted directory {path}"
        if os.path.isfile(full):
            os.remove(full)
            return f"deleted {path}"
        raise ToolError(f"not found: {path}")

    def _run_shell(self, command: str, timeout: int = 30) -> str:
        timeout = max(1, min(int(timeout), 120))
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {timeout}s")
        out = (proc.stdout or "") + (proc.stderr or "")
        if len(out) > 20000:
            out = out[:20000] + "\n...[truncated]..."
        return f"[exit {proc.returncode}]\n{out}".strip()

    def _web_fetch(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ToolError("url must start with http:// or https://")
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "mini-harness/1.0"})
            r.raise_for_status()
        except requests.RequestException as e:
            raise ToolError(f"fetch failed: {e}")
        text = r.text
        # crude HTML -> text
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 15000:
            text = text[:15000] + " ...[truncated]..."
        return text or "(empty page)"

    # -- registry ---------------------------------------------------------
    def _register_defaults(self):
        self.add("read_file", "Read a text file from the workspace.",
                 '{"path": "notes/todo.txt"}', lambda path: self._read_file(path))
        self.add("write_file", "Create or overwrite a file in the workspace.",
                 '{"path": "hello.py", "content": "print(123)"}',
                 lambda path, content="": self._write_file(path, content))
        self.add("edit_file", "Replace one unique occurrence of `old` with `new` in a file.",
                 '{"path": "app.py", "old": "x = 1", "new": "x = 2"}',
                 lambda path, old, new: self._edit_file(path, old, new))
        self.add("list_dir", "List files in a workspace directory.",
                 '{"path": "."}', lambda path=".": self._list_dir(path))
        self.add("delete_file", "Delete a file or directory in the workspace.",
                 '{"path": "old.txt"}', lambda path: self._delete_file(path))
        self.add("run_shell", "Run a shell command in the workspace (cwd = workspace root).",
                 '{"command": "python3 main.py", "timeout": 30}',
                 lambda command, timeout=30: self._run_shell(command, timeout))
        self.add("web_fetch", "Fetch a URL and return its text content.",
                 '{"url": "https://example.com"}', lambda url: self._web_fetch(url))

    def add(self, name: str, description: str, usage: str, func: Callable[..., str]):
        self.tools[name] = Tool(name, description, usage, func)

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(sorted(self.tools))}"
        try:
            return tool.func(**args)
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}"
        except ToolError as e:
            return f"ERROR: {e}"
        except Exception as e:  # never let a tool crash the loop
            return f"ERROR: {type(e).__name__}: {e}"

    def prompt_block(self) -> str:
        lines = ["Available tools (call with a ```tool block):"]
        for t in self.tools.values():
            lines.append(f"- {t.name}: {t.description}\n  Example: {t.usage}")
        return "\n".join(lines)
