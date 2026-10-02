"""Multi-agent orchestration: a team of agents that talk to each other.

Flow:
  1. The lead agent breaks the goal into subtasks and assigns them via
     ```message blocks to teammates.
  2. In rounds, every agent with inbox messages (or fresh work) runs a turn:
     it uses tools and may message teammates.
  3. A shared bulletin accumulates progress notes all agents can see.
  4. The lead synthesizes a final summary when the team goes quiet or
     max_rounds is reached.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .agent import Agent
from .openrouter import OpenRouterClient
from .tools import ToolRegistry

LEAD_BRIEF = """MISSION GOAL:
{goal}

You are the team lead. The team is: {roster}.
Break the goal into concrete subtasks and assign each one to a teammate by sending
them a ```message block with their name in "to" and a clear assignment in "text".
Do the first subtask-relevant step yourself only if no teammate fits it.
End your reply with a short PLAN: section listing who does what."""


class Team:
    def __init__(
        self,
        client: OpenRouterClient,
        registry: ToolRegistry,
        max_inner_iters: int = 8,
        max_rounds: int = 4,
    ):
        self.client = client
        self.registry = registry
        self.max_inner_iters = max_inner_iters
        self.max_rounds = max_rounds

    def run_mission(
        self,
        goal: str,
        agent_specs: List[Dict],
        on_event: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """Run a mission. agent_specs: [{name, role, model}]; first is the lead."""
        def emit(ev: Dict):
            if on_event:
                on_event(ev)

        if not agent_specs:
            raise ValueError("at least one agent is required")
        agents: Dict[str, Agent] = {}
        for spec in agent_specs:
            agents[spec["name"]] = Agent(
                name=spec["name"],
                role=spec.get("role", "team member"),
                model=spec.get("model") or "qwen/qwen3-coder:free",
                client=self.client,
                registry=self.registry,
                max_inner_iters=self.max_inner_iters,
            )
        lead_name = agent_specs[0]["name"]
        roster = ", ".join(f'{n} ("{a.role}")' for n, a in agents.items())

        emit({"type": "mission_start", "goal": goal,
              "agents": [{"name": n, "role": a.role, "model": a.model}
                         for n, a in agents.items()]})

        inboxes: Dict[str, List[str]] = {n: [] for n in agents}
        assignments: Dict[str, str] = {n: "" for n in agents}
        bulletin: List[str] = []
        finals: Dict[str, str] = {}

        def route(messages_out: List, sender: str) -> int:
            delivered = 0
            for to, text in messages_out:
                if to in inboxes:
                    inboxes[to].append(f"[{sender}]: {text}")
                    delivered += 1
                else:
                    bulletin.append(f"(note for unknown agent '{to}' from {sender}: {text})")
            return delivered

        # Round 0: lead plans and assigns.
        emit({"type": "round_start", "round": 0})
        emit({"type": "agent_start", "agent": lead_name})
        out = agents[lead_name].run_turn(
            LEAD_BRIEF.format(goal=goal, roster=roster), on_event=on_event)
        finals[lead_name] = out["final"]
        n_msg = route(out["messages_out"], lead_name)
        # Messages that read like assignments also become the recipient's brief.
        for to, text in out["messages_out"]:
            if to in assignments and not assignments[to]:
                assignments[to] = text
        bulletin.append(f"Lead plan: {out['final'][:800]}")
        emit({"type": "agent_done", "agent": lead_name,
              "final": out["final"], "messages_sent": n_msg})

        # Rounds: agents work their inbox + assignment, talk to each other.
        for rnd in range(1, self.max_rounds + 1):
            emit({"type": "round_start", "round": rnd})
            round_activity = False
            for name, agent in agents.items():
                inbox = inboxes[name]
                if not inbox and not assignments[name] and rnd > 1:
                    continue  # nothing new for this agent
                prompt_parts = []
                if assignments[name]:
                    prompt_parts.append(f"YOUR ASSIGNMENT:\n{assignments[name]}")
                if inbox:
                    prompt_parts.append("MESSAGES FROM TEAMMATES:\n" + "\n".join(inbox))
                    inboxes[name] = []
                if bulletin:
                    prompt_parts.append("TEAM BULLETIN (what others found):\n" + "\n".join(bulletin[-8:]))
                prompt_parts.append(
                    "Work your assignment now. Use tools as needed, message teammates with "
                    "```message blocks when you need something or have something to share. "
                    "When your part is done, reply with a concise summary of what you did and "
                    "where the results are.")
                emit({"type": "agent_start", "agent": name})
                out = agent.run_turn("\n\n".join(prompt_parts), on_event=on_event)
                finals[name] = out["final"]
                n_msg = route(out["messages_out"], name)
                bulletin.append(f"{name}: {out['final'][:600]}")
                if out["tool_calls"] or n_msg or out["final"].strip():
                    round_activity = True
                emit({"type": "agent_done", "agent": name, "final": out["final"],
                      "messages_sent": n_msg, "tool_calls": out["tool_calls"]})
            if not round_activity:
                break

        # Synthesis by the lead.
        emit({"type": "agent_start", "agent": lead_name})
        synth = agents[lead_name].run_turn(
            "MISSION GOAL:\n" + goal + "\n\nTEAM RESULTS:\n" + "\n\n".join(
                f"## {n}\n{f}" for n, f in finals.items()
            ) + "\n\nWrite the final mission summary: what was accomplished, key files "
                 "changed/created, and anything left undone. No tool blocks needed.",
            on_event=on_event)
        emit({"type": "mission_done", "summary": synth["final"]})
        return {"summary": synth["final"], "finals": finals,
                "bulletin": bulletin}
