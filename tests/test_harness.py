"""Tests for the harness core: protocol parsing, sandbox, tools, agent loop, team."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.agent import Agent
from harness.team import Team
from harness.tools import (
    ToolRegistry,
    parse_messages,
    parse_tool_calls,
    strip_protocol_blocks,
)


class FakeClient:
    """Canned-response client: returns scripted replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, model, **kw):
        on_token = kw.get("on_token")
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if on_token:
            on_token(text)
        return text


def test_parse_tool_calls():
    text = 'thinking\n```tool\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n```\ndone'
    calls = parse_tool_calls(text)
    assert calls == [("read_file", {"path": "a.txt"})], calls
    assert parse_tool_calls("no blocks here") == []
    # broken JSON is skipped, not fatal
    assert parse_tool_calls('```tool\n{not json}\n```') == []


def test_parse_messages():
    text = '```message\n{"to": "coder", "text": "go build it"}\n```'
    assert parse_messages(text) == [("coder", "go build it")]
    assert strip_protocol_blocks("hi\n```tool\n{}\n```\nthere") == "hi\n\nthere"


def test_sandbox_escape_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        reg = ToolRegistry(os.path.join(tmp, "ws"))
        out = reg.execute("read_file", {"path": "../secret.txt"})
        assert out.startswith("ERROR"), out
        out = reg.execute("write_file", {"path": "/etc/evil", "content": "x"})
        assert out.startswith("ERROR"), out


def test_write_read_edit_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        reg = ToolRegistry(os.path.join(tmp, "ws"))
        assert reg.execute("write_file", {"path": "a.txt", "content": "hello"}).startswith("wrote")
        assert reg.execute("read_file", {"path": "a.txt"}) == "hello"
        assert "edited" in reg.execute("edit_file", {"path": "a.txt", "old": "hello", "new": "bye"})
        assert reg.execute("read_file", {"path": "a.txt"}) == "bye"
        # ambiguous edit is rejected
        reg.execute("write_file", {"path": "b.txt", "content": "x x"})
        assert reg.execute("edit_file", {"path": "b.txt", "old": "x", "new": "y"}).startswith("ERROR")


def test_shell_runs_in_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        reg = ToolRegistry(os.path.join(tmp, "ws"))
        out = reg.execute("run_shell", {"command": "pwd"})
        assert os.path.join(tmp, "ws") in out, out


def test_agent_loop_uses_tools():
    with tempfile.TemporaryDirectory() as tmp:
        reg = ToolRegistry(os.path.join(tmp, "ws"))
        reg.execute("write_file", {"path": "data.txt", "content": "42"})
        client = FakeClient([
            'I will read the file.\n```tool\n{"name": "read_file", "arguments": {"path": "data.txt"}}\n```',
            "The file says 42. Done.",
        ])
        agent = Agent("tester", "test role", "fake-model", client, reg)
        events = []
        out = agent.run_turn("what is in data.txt?", on_event=events.append)
        assert out["final"] == "The file says 42. Done.", out
        assert out["tool_calls"] == 1
        kinds = [e["type"] for e in events]
        assert "tool_call" in kinds and "tool_result" in kinds


def test_agent_emits_inter_agent_message():
    with tempfile.TemporaryDirectory() as tmp:
        reg = ToolRegistry(os.path.join(tmp, "ws"))
        client = FakeClient([
            '```message\n{"to": "coder", "text": "please write main.py"}\n```\nAssigned.',
        ])
        agent = Agent("lead", "lead", "fake-model", client, reg)
        out = agent.run_turn("assign work")
        assert out["messages_out"] == [("coder", "please write main.py")]
        assert "please write main.py" not in out["final"]  # protocol stripped


def test_team_routes_messages():
    with tempfile.TemporaryDirectory() as tmp:
        reg = ToolRegistry(os.path.join(tmp, "ws"))
        # lead assigns to coder; coder writes a file; lead synthesizes.
        lead_replies = [
            'Plan: coder builds.\n```message\n{"to": "coder", "text": "write hello.py printing hi"}\n```',
            "Mission complete: hello.py written.",
        ]
        coder_replies = [
            'On it.\n```tool\n{"name": "write_file", "arguments": {"path": "hello.py", "content": "print(\'hi\')"}}\n```',
            "Wrote hello.py.",
        ]
        clients = {"lead": FakeClient(lead_replies), "coder": FakeClient(coder_replies)}

        orig_agent = Agent

        class PatchedAgent(orig_agent):
            def __init__(self, name, **kw):
                super().__init__(name, kw["role"], kw["model"], clients[name],
                                 kw["registry"], kw.get("max_inner_iters", 8))

        import harness.team as team_mod
        team_mod.Agent = PatchedAgent
        try:
            team = Team(clients["lead"], reg, max_rounds=2)
            result = team.run_mission("say hi", [
                {"name": "lead", "role": "lead", "model": "fake"},
                {"name": "coder", "role": "coder", "model": "fake"},
            ])
        finally:
            team_mod.Agent = orig_agent
        assert os.path.isfile(os.path.join(tmp, "ws", "hello.py"))
        assert "hello.py" in result["summary"]


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"PASS {name}")
    print("all tests passed")
