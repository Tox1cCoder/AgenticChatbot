"""Custom-agent helpers extracted from MultiAgentWorkflow (Task 8, Step 3).

Methods are relocated verbatim; behavior is identical.
"""

from typing import Any

from app.ai.agent_metadata import agent_identity, base_agent_capability
from app.ai.agents.custom_agent import CustomAgent, build_custom_specialist_definition
from app.ai.custom_agent_runtime import build_custom_agent_runtime_spec, is_custom_runtime_id
from app.ai.hand_off_tool import create_hand_off_tool
from app.ai.schemas import GraphState, GraphStateView
from app.ai.workflow.contracts import AgentTransition


class CustomAgentsMixin:
    """Relocated custom_agents methods for :class:`MultiAgentWorkflow`."""

    def _resolve_runtime_agent(self, state: GraphState, active_agent_id: str | None) -> Any:
        """Resolve a base agent or build a custom agent for the selected id."""
        if active_agent_id in self.agents:
            return self.agents[active_agent_id]
        if is_custom_runtime_id(active_agent_id):
            return self._build_custom_agent(state, active_agent_id)
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
            source_agent_id=str(active_agent_id),
            allowed_targets=targets,
            target_descriptions=self._custom_handoff_target_descriptions(state, active_agent_id),
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

        lines.extend(self._agent_trail_lines(state, custom_agents))
        return "\n".join(lines)

    @staticmethod
    def _agent_trail_lines(state: GraphState, custom_agents: dict[str, Any]) -> list[str]:
        """Who has held this turn so far, read from its ``agent_history``.

        The accepted-transition audit trail the router and the transition
        resolver append to, so it cannot disagree with what actually ran. It is
        turn-local by construction: every turn runs on its own checkpoint
        thread.
        """
        history = [
            transition
            for transition in (state.get("agent_history") or [])
            if isinstance(transition, AgentTransition)
        ]
        if not history:
            return []
        via_labels = {
            "router": "selected by the router",
            "handoff": "received via hand_off",
            "resume": "resumed for this turn",
        }
        lines = ["Agents involved in this turn so far, in order:"]
        for transition in history:
            identity = agent_identity(transition.to_agent_id, custom_agents)
            name = identity["name"] if identity else transition.to_agent_id
            lines.append(f"- {name} — {via_labels.get(transition.source, transition.source)}")
        return lines

    def _build_custom_agent(
        self,
        state: GraphState,
        runtime_agent_id: str | None,
        *,
        allow_handoff: bool = True,
    ) -> CustomAgent | None:
        """Build a live CustomAgent from the workflow ``custom_agents`` state.

        Built fresh each invocation so edited configuration applies to future
        turns (live config). Returns ``None`` if the agent is not attached.
        ``allow_handoff=False`` builds it with no handoff targets, so it gets
        neither the ``hand_off`` tool nor the delegation prompt.
        """
        entry = GraphStateView(state).custom_agents().get(runtime_agent_id)
        if not entry:
            return None
        spec = build_custom_agent_runtime_spec(
            entry,
            allowed_handoff_targets=(
                self._custom_handoff_targets(state, runtime_agent_id) if allow_handoff else []
            ),
            handoff_target_descriptions=(
                self._custom_handoff_target_descriptions(state, runtime_agent_id)
                if allow_handoff
                else {}
            ),
        )
        return CustomAgent(
            spec,
            runtime_model_resolver=self._runtime_model_resolver,
            recorder=self._model_usage_recorder,
        )

    def _custom_specialist_definition(self, request: Any) -> Any | None:
        """The definition one custom-agent invocation runs, from its own roster.

        The specialist factory calls this per request instead of keeping a
        registry, because the factory is shared by every turn. A public turn's
        request carries the graph state; a Planning worker's carries the
        parent's roster snapshot. A worker is built without handoff targets: it
        returns a private result to Planning and never performs a parent-level
        handoff.
        """
        if request.extras.get("worker"):
            roster = {"custom_agents": request.extras.get("custom_agents") or {}}
            agent = self._build_custom_agent(roster, request.agent_id, allow_handoff=False)
        else:
            agent = self._build_custom_agent(request.state, request.agent_id)
        if agent is None:
            return None
        return build_custom_specialist_definition(agent)
