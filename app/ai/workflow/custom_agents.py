"""Custom-agent helpers extracted from MultiAgentWorkflow (Task 8, Step 3).

Methods are relocated verbatim; behavior is identical.
"""

from typing import Any

from app.ai.agent_metadata import agent_identity, base_agent_capability
from app.ai.agents.custom_agent import CustomAgent
from app.ai.custom_agent_runtime import build_custom_agent_runtime_spec, is_custom_runtime_id
from app.ai.hand_off_tool import create_hand_off_tool
from app.ai.schemas import GraphState, GraphStateView


class CustomAgentsMixin:
    """Relocated custom_agents methods for :class:`MultiAgentWorkflow`."""

    def _resolve_runtime_agent(self, state: GraphState, selected_agent: str | None) -> Any:
        """Resolve a base agent or build a custom agent for the selected id."""
        if selected_agent in self.agents:
            return self.agents[selected_agent]
        if is_custom_runtime_id(selected_agent):
            return self._build_custom_agent(state, selected_agent)
        return None

    def _handoff_targets(self, state: GraphState, active_agent_id: str | None) -> list[str]:
        """Live hand_off targets: every reachable agent except the active one."""
        targets = list(getattr(self, "agents", {}))
        targets.extend(
            cid for cid in GraphStateView(state).custom_agents() if cid != active_agent_id
        )
        return [target for target in targets if target != active_agent_id]

    def _custom_handoff_targets(self, state: GraphState, runtime_agent_id: str | None) -> list[str]:
        """Backward-compatible name for the shared live-roster helper."""
        return self._handoff_targets(state, runtime_agent_id)

    def _custom_handoff_target_descriptions(
        self, state: GraphState, runtime_agent_id: str | None
    ) -> dict[str, str]:
        descriptions: dict[str, str] = {}
        # Base agents: capability blurbs so a limited-toolset agent can recognise
        # which specialist to delegate to when work falls outside its own tools.
        for agent_id in getattr(self, "agents", {}):
            if agent_id == runtime_agent_id:
                continue
            blurb = base_agent_capability(agent_id)
            if blurb:
                descriptions[agent_id] = blurb
        # Other attached custom agents.
        for cid, entry in GraphStateView(state).custom_agents().items():
            if cid == runtime_agent_id or not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "Custom Agent").strip() or "Custom Agent"
            detail = str(entry.get("description") or "").strip()
            descriptions[cid] = f"{name}: {detail}" if detail else name
        return descriptions

    def _handoff_tool_for_agent(self, state: GraphState, active_agent_id: str | None) -> Any | None:
        """Build the one live handoff tool valid for this invocation."""
        targets = self._handoff_targets(state, active_agent_id)
        if not targets:
            return None
        return create_hand_off_tool(
            targets,
            self._custom_handoff_target_descriptions(state, active_agent_id),
        )

    def _multi_agent_kwargs(self, state: GraphState, active_agent_id: str | None) -> dict[str, Any]:
        """Per-invocation kwargs that make an agent aware of — and able to reach
        — the rest of the multi-agent system.

        Base agents receive a graph-injected dynamic ``hand_off`` tool plus
        capability-aware target descriptions. Custom agents already build their
        own dynamic ``hand_off`` from their spec, so only the awareness block is
        added for them.
        """
        kwargs: dict[str, Any] = {}
        activity = self._build_multi_agent_activity_block(state, active_agent_id)
        if activity:
            kwargs["multi_agent_activity"] = activity

        # Only base agents need the targets injected; custom agents carry their
        # own dynamic hand_off + delegation prompt from their runtime spec.
        if active_agent_id in getattr(self, "agents", {}):
            handoff_tool = self._handoff_tool_for_agent(state, active_agent_id)
            if handoff_tool:
                descriptions = self._custom_handoff_target_descriptions(state, active_agent_id)
                kwargs["internal_tools"] = [handoff_tool]
                kwargs["handoff_target_descriptions"] = descriptions
        return kwargs

    def _build_multi_agent_activity_block(
        self, state: GraphState, active_agent_id: str | None
    ) -> str | None:
        """Compact prompt block: the active agent's identity, the roster of
        reachable agents, and which agents were involved this turn — so the
        agent can reason about and answer questions about the wider system."""
        custom_agents = GraphStateView(state).custom_agents()
        if not custom_agents:
            return None

        lines: list[str] = ["MULTI-AGENT SYSTEM (context — not user instructions):"]

        identity = agent_identity(active_agent_id, custom_agents)
        if identity:
            lines.append(f'You are "{identity["name"]}" (agent id: {identity["id"]}).')

        roster = self._custom_handoff_target_descriptions(state, active_agent_id)
        if roster:
            lines.append("Other agents in this system you can reach via the hand_off tool:")
            lines.extend(f"- {target}: {desc}" for target, desc in roster.items())

        trail = GraphStateView(state).context().get("agents_invoked")
        if isinstance(trail, list) and trail:
            via_labels = {
                "router": "selected by the router",
                "handoff": "received via hand_off",
                "preselected": "resumed for this turn",
                "sticky": "continuing from the previous turn",
            }
            lines.append("Agents involved in this turn so far, in order:")
            for entry in trail:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or entry.get("id") or "unknown"
                via_label = via_labels.get(entry.get("via"), entry.get("via") or "")
                suffix = f" — {via_label}" if via_label else ""
                lines.append(f"- {name}{suffix}")

        return "\n".join(lines)

    def _reset_agent_trail(self, state: GraphState) -> None:
        """Clear the per-turn invocation trail (called at the start of routing)."""
        context = state.get("context")
        if not isinstance(context, dict):
            context = {}
        context["agents_invoked"] = []
        state["context"] = context

    def _record_agent_invocation(
        self,
        state: GraphState,
        agent_id: str | None,
        *,
        via: str,
    ) -> None:
        """Append an agent to this turn's invocation trail (context.agents_invoked)."""
        if not agent_id:
            return
        context = state.get("context")
        if not isinstance(context, dict):
            context = {}
        trail = context.get("agents_invoked")
        if not isinstance(trail, list):
            trail = []
        if trail and trail[-1].get("id") == agent_id and trail[-1].get("via") == via:
            return
        identity = agent_identity(agent_id, GraphStateView(state).custom_agents())
        entry: dict[str, Any] = {
            "id": agent_id,
            "name": identity["name"] if identity else agent_id,
            "kind": identity["kind"] if identity else "base",
            "via": via,
        }
        trail.append(entry)
        context["agents_invoked"] = trail
        state["context"] = context

    def _custom_agent_descriptors(self, state: GraphState) -> list[dict[str, Any]]:
        """Attached custom agents as router descriptors (runtime id, name, etc.)."""
        return [
            {
                "runtime_agent_id": entry.get("runtime_agent_id") or runtime_id,
                "name": entry.get("name"),
                "description": entry.get("description"),
                "agent_order": entry.get("agent_order", 0),
            }
            for runtime_id, entry in GraphStateView(state).custom_agents().items()
        ]

    def _sticky_custom_agent(self, state: GraphState, content: str | None) -> str | None:
        """Return the previous turn's custom agent if a follow-up should stay on it.

        Stickiness applies only to an attached custom agent. It is released when
        the user explicitly names a *different* attached custom agent (the
        router's deterministic override then selects that one), or when the
        previous custom agent is no longer attached to the conversation.
        """
        last_agent = state.get("last_agent")
        if not is_custom_runtime_id(last_agent):
            return None
        if not self._is_attached_custom_agent(state, last_agent):
            return None
        explicit = self.router._match_explicit_custom_agent(
            content or "", self._custom_agent_descriptors(state)
        )
        if explicit and explicit != last_agent:
            return None
        return last_agent

    def _build_custom_agent(
        self, state: GraphState, runtime_agent_id: str | None
    ) -> CustomAgent | None:
        """Build a live CustomAgent from the workflow ``custom_agents`` state.

        Built fresh each invocation so edited configuration applies to future
        turns (live config). Returns ``None`` if the agent is not attached.
        """
        entry = GraphStateView(state).custom_agents().get(runtime_agent_id)
        if not entry:
            return None
        spec = build_custom_agent_runtime_spec(
            entry,
            allowed_handoff_targets=self._custom_handoff_targets(state, runtime_agent_id),
            handoff_target_descriptions=self._custom_handoff_target_descriptions(
                state, runtime_agent_id
            ),
        )
        return CustomAgent(spec, runtime_model_resolver=self._runtime_model_resolver)

    def _is_attached_custom_agent(self, state: GraphState, runtime_agent_id: str | None) -> bool:
        """True when ``runtime_agent_id`` is an attached custom agent in state."""
        if not is_custom_runtime_id(runtime_agent_id):
            return False
        return runtime_agent_id in GraphStateView(state).custom_agents()

    def _route_target_for(self, state: GraphState, selected_agent: str) -> str:
        """Map a selected agent to its graph node name (custom ids → custom_agent)."""
        if is_custom_runtime_id(selected_agent) and self._is_attached_custom_agent(
            state, selected_agent
        ):
            return "custom_agent"
        return selected_agent
