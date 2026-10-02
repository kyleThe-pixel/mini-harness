"""The agent loop: a single agent that thinks, calls tools, and replies.

The loop is model-agnostic — it uses a plain-text tool protocol (fenced
```tool blocks) instead of native function calling, so it works with any
OpenRouter chat model, including the free tier.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .openrouter import OpenRouterClient
from .tools import (
    ToolRegistry,
    parse_messages,
    parse_tool_calls,
    strip_protocol_blocks,
)

SYSTEM_TEMPLATE = """You are {name}, {role}.
You are part of a small team of AI agents sharing one workspace (root: {workspace_root}).
Use your tools to make progress on your assignment; do not just describe what you would do.

TOOL PROTOCOL
Call a tool by emitting a fenced block (one call per block, several blocks per reply are fine):

```tool
{{"name": "read_file", "arguments": {{"path": "notes/todo.txt"}}}}
```

Tool results come back as <tool_result name="...">...</tool_result> messages. Continue until the
task is done, then write your final summary with no tool blocks.

TALKING TO TEAMMATES
Send a message to another agent with a fenced block:

```message
{{"to": "teammate-name", "text": "what you want to share, ask, or hand off"}}
```

Messages are delivered between turns. Share useful findings promptly and ask for what you need.

RULES
- Work only inside the workspace root. Paths are relative to it.
- Read a file before editing it. Prefer small, verifiable changes; run code to check your work.
- If a tool call errors, read the error and try a different approach instead of repeating it.
- Be concise in chat; be thorough in files. Do not emit tool blocks "for show".

{tool_catalog}"""


class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        model: str,
        client: OpenRouterClient,
        registry: ToolRegistry,
        max_inner_iters: int = 8,
    ):
        self.name = name
        self.role = role
        self.model = model
        self.client = client
        self.registry = registry
        self.max_inner_iters = max_inner_iters

    def system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(
            name=self.name,
            role=self.role,
            workspace_root=self.registry.root,
            tool_catalog=self.registry.prompt_block(),
        )

    def run_turn(
        self,
        user_text: str,
        on_event: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """Run one turn: loop model -> tools until no more tool calls.

        Returns {"final": str, "messages_out": [(to, text)], "tool_calls": int}.
        Events emitted: token, tool_call, tool_result, error.
        """
        def emit(ev: Dict):
            ev.setdefault("agent", self.name)
            if on_event:
                on_event(ev)

        messages: List[Dict] = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user_text},
        ]
        messages_out: List = []
        tool_call_count = 0

        for _ in range(self.max_inner_iters):
            try:
                text = self.client.chat(
                    messages,
                    self.model,
                    stream=True,
                    on_token=lambda t: emit({"type": "token", "text": t}),
                )
            except Exception as e:
                emit({"type": "error", "text": str(e)})
                return {"final": f"(error: {e})", "messages_out": messages_out,
                        "tool_calls": tool_call_count}

            messages.append({"role": "assistant", "content": text})
            calls = parse_tool_calls(text)
            for to, msg in parse_messages(text):
                messages_out.append((to, msg))
                emit({"type": "agent_message", "to": to, "text": msg})

            if not calls:
                return {"final": strip_protocol_blocks(text),
                        "messages_out": messages_out,
                        "tool_calls": tool_call_count}

            for tool_name, args in calls:
                tool_call_count += 1
                emit({"type": "tool_call", "tool": tool_name, "args": args})
                result = self.registry.execute(tool_name, args)
                emit({"type": "tool_result", "tool": tool_name,
                      "ok": not result.startswith("ERROR"),
                      "text": result[:4000]})
                messages.append({
                    "role": "user",
                    "content": f'<tool_result name="{tool_name}">\n{result}\n</tool_result>',
                })

        return {"final": "(stopped: tool iteration limit reached)",
                "messages_out": messages_out, "tool_calls": tool_call_count}
