"""CustomAgent — a thin BaseAgent adapter driven by an AgentRuntimeSpec.

Runtime identity (``custom_agent:<uuid>``) and model identity (the generic
``custom`` key) are kept separate: the runtime id keys tool/deferred state,
streaming metadata, and the response ``agent_id``, while ``custom`` only drives
model resolution. The per-agent provider/model/temperature come from the spec's
``model_request`` override.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from ...core.config import settings
from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..custom_agent_runtime import AgentRuntimeSpec, filter_tools_for_custom_agent
from ..hand_off_tool import create_hand_off_tool
from ..schemas import AgentType
from ..skills_tool import create_activate_skill_tool
from ..tool_search_tool import create_tool_search_tool_for_custom_agent
from .base_agent import BaseAgent

CUSTOM_MODEL_AGENT_KEY = "custom"


class CustomAgent(BaseAgent):
    """Adapter exposing a persisted custom agent as a graph-invocable agent."""

    def __init__(
        self,
        spec: AgentRuntimeSpec,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
    ):
        model_request = spec.model_request or {}
        super().__init__(
            model_name=model_request.get("model"),
            agent_config_key=CUSTOM_MODEL_AGENT_KEY,
            runtime_model_resolver=runtime_model_resolver,
        )
        self._spec = spec
        # Deferred/loaded tool state is keyed by the runtime id so two custom
        # agents in the same conversation never share loaded tools.
        self.tool_state_key = spec.runtime_agent_id
        self._runtime_warnings: list[str] = []

    # ------------------------------------------------------------- identity

    @property
    def agent_type(self) -> AgentType:
        # Custom responses are CHAT-compatible for existing AgentResponse
        # handling; the real identity lives in agent_id + metadata.
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return self._spec.runtime_agent_id

    @property
    def spec(self) -> AgentRuntimeSpec:
        return self._spec

    def _get_base_system_prompt(self) -> str:
        return self._spec.prompt or ""

    # ------------------------------------------------------- model + prompt

    def _resolve_model_request(self, model_request: dict[str, Any] | None) -> dict[str, Any] | None:
        # Custom agents always resolve to their own saved provider/model,
        # independent of any incoming per-base-agent model_request map.
        return self._spec.model_request

    def _build_skills_suffix(
        self,
        *,
        user_id: str | None = None,
        device_id: str | None = None,
        allowed_skill_refs: list[dict[str, Any]] | None = None,
    ) -> str:
        # Always restrict to this agent's selected skills.
        return super()._build_skills_suffix(
            user_id=user_id,
            device_id=device_id,
            allowed_skill_refs=self._spec.allowed_skill_refs,
        )

    # ------------------------------------------------------------- metadata

    def set_runtime_warnings(self, warnings: list[str]) -> None:
        """Record warnings (e.g. unavailable selected tools/skills) for metadata."""
        self._runtime_warnings = list(warnings or [])

    def _augment_response_metadata(self, metadata: dict[str, Any]) -> None:
        metadata["runtime_agent_id"] = self._spec.runtime_agent_id
        metadata["custom_agent_id"] = (
            str(self._spec.custom_agent_id) if self._spec.custom_agent_id else None
        )
        metadata["custom_agent_name"] = self._spec.display_name
        if self._runtime_warnings:
            metadata["custom_agent_warnings"] = list(self._runtime_warnings)

    # ---------------------------------------------------- restricted tools

    def _get_tools_for_binding(
        self,
        conversation_id: str | None = None,
        internal_tools: list[BaseTool] | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        tool_scope: str | None = None,
        include_hand_off: bool | None = None,
    ) -> list[BaseTool]:
        """Bind exactly the restricted toolset (same set used for execution).

        Candidates = initialized MCP tools + active client runtime tools, then
        filtered to this agent's runtime policy (backend MCP tools by default
        plus exactly the selected client tools). Restricted internal tools
        (tool_search, activate_skill) are always included. Unavailable selected
        tools surface as response-metadata warnings.
        """
        candidates: list[BaseTool] = list(self.tools or [])
        if not (settings.enable_client_runtime_bridge and not device_id):
            candidates.extend(self._get_client_runtime_tools(user_id=user_id, device_id=device_id))

        external, warnings = filter_tools_for_custom_agent(
            candidates, self._spec, request_device_id=device_id
        )
        self.set_runtime_warnings(warnings)

        tools: list[BaseTool] = list(
            self.restricted_internal_tools(user_id=user_id, device_id=device_id)
        )
        for tool in internal_tools or []:
            tools.append(tool)
        tools.extend(external)

        seen: set[str] = set()
        deduped: list[BaseTool] = []
        for tool in tools:
            name = getattr(tool, "name", None)
            if name in seen:
                continue
            seen.add(name)
            deduped.append(tool)
        return deduped

    def restricted_internal_tools(
        self,
        *,
        user_id: str | None,
        device_id: str | None,
    ) -> list[Any]:
        """Internal tools every custom agent gets: restricted tool_search + skills.

        The dynamic ``hand_off`` tool is injected by the graph (Task 10) from the
        spec's allowed targets; external client tools are appended after exact
        filtering by the graph's tool node (Task 9).
        """
        tools: list[Any] = [
            create_tool_search_tool_for_custom_agent(self._spec),
            create_activate_skill_tool(
                user_id=user_id,
                device_id=device_id,
                allowed_skill_refs=self._spec.allowed_skill_refs,
            ),
        ]
        # Dynamic hand_off scoped to this agent's valid targets (base + other
        # attached custom agents). Same tool object the graph re-validates.
        if self._spec.allowed_handoff_targets:
            tools.append(create_hand_off_tool(self._spec.allowed_handoff_targets))
        return tools
