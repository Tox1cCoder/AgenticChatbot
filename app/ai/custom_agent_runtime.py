"""Runtime spec and restricted tool/skill policy for custom agents.

A single :class:`AgentRuntimeSpec` is the source of truth for an invocation:
the same object drives model binding, tool-map construction, tool execution,
prompt-skill visibility, and handoff target validation. Keeping one object
prevents drift between the tools shown to the model and the tools that can
actually run.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.custom_agent import (
    CUSTOM_AGENT_PREFIX,
    CUSTOM_MODEL_AGENT_KEY,
    runtime_agent_id_for,
)

__all__ = [
    "AgentRuntimeSpec",
    "BASE_AGENT_IDS",
    "is_custom_runtime_id",
    "parse_custom_agent_id",
    "build_base_agent_runtime_spec",
    "build_custom_agent_runtime_spec",
    "filter_tools_for_custom_agent",
    "rebase_client_tool_refs",
    "client_ref_matches_metadata",
    "list_selected_skill_summaries",
]

BASE_AGENT_IDS: frozenset[str] = frozenset(
    {
        "chat_agent",
        "rag_agent",
        "search_agent",
        "image_generator_agent",
        "planning_agent",
        "canvas_agent",
    }
)

_BASE_RUNTIME_TO_MODEL_KEY: dict[str, str] = {
    "chat_agent": "chat",
    "rag_agent": "rag",
    "search_agent": "search",
    "image_generator_agent": "image_generator",
    "planning_agent": "planning",
    "canvas_agent": "canvas",
}

# Exact-identity keys that a persisted client-tool ref and a bound tool's
# metadata must agree on before the tool may be used by a custom agent.
_CLIENT_MATCH_KEYS = (
    "device_id",
    "session_id",
    "catalog_version",
    "tool_instance_id",
    "qualified_tool_id",
)


class AgentRuntimeSpec(BaseModel):
    """The resolved runtime policy for a single agent invocation."""

    runtime_agent_id: str
    model_agent_key: str
    custom_agent_id: UUID | None = None
    display_name: str
    description: str | None = None
    prompt: str | None = None
    model_request: dict[str, Any] | None = None
    allowed_handoff_targets: list[str] = Field(default_factory=list)
    handoff_target_descriptions: dict[str, str] = Field(default_factory=dict)
    allowed_server_tool_refs: list[dict[str, Any]] = Field(default_factory=list)
    allowed_client_tool_refs: list[dict[str, Any]] = Field(default_factory=list)
    allowed_skill_refs: list[dict[str, Any]] = Field(default_factory=list)
    allow_all_server_tools: bool = False

    @property
    def is_custom(self) -> bool:
        return self.custom_agent_id is not None

    def allowed_qualified_tool_ids(self) -> set[str]:
        """Qualified tool ids the model may discover via tool_search."""
        ids: set[str] = set()
        for ref in (*self.allowed_server_tool_refs, *self.allowed_client_tool_refs):
            qid = ref.get("qualified_tool_id")
            if qid:
                ids.add(str(qid))
        return ids

    def tool_search_allowlist(self) -> list[str]:
        """Allowlist (names + qualified ids) for the restricted tool_search."""
        names: set[str] = set()
        server_allowlist = self.server_tool_search_allowlist()
        if server_allowlist:
            names.update(server_allowlist)
        names.update(self.client_tool_search_allowlist())
        return sorted(names)

    def server_tool_search_allowlist(self) -> list[str] | None:
        """Server-side search allowlist; ``None`` means every backend MCP tool."""
        if self.allow_all_server_tools:
            return None
        return sorted(
            str(ref.get("qualified_tool_id"))
            for ref in self.allowed_server_tool_refs
            if ref.get("qualified_tool_id")
        )

    def client_tool_search_allowlist(self) -> list[str]:
        """Exact client-side search allowlist.

        Prefer the persisted tool instance id so a current sidecar session with
        the same public tool name cannot be substituted for the saved tool.
        """
        names: set[str] = set()
        for ref in self.allowed_client_tool_refs:
            value = ref.get("tool_instance_id")
            if value:
                names.add(str(value))
                continue
            value = ref.get("qualified_tool_id")
            if value:
                names.add(str(value))
            value = ref.get("tool_name")
            if value:
                names.add(str(value))
        return sorted(names)


# --------------------------------------------------------------------------- #
# Runtime-id helpers
# --------------------------------------------------------------------------- #


def is_custom_runtime_id(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(CUSTOM_AGENT_PREFIX)


def parse_custom_agent_id(value: Any) -> UUID | None:
    if not is_custom_runtime_id(value):
        return None
    raw = value[len(CUSTOM_AGENT_PREFIX) :]
    try:
        return UUID(raw)
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------- #
# Spec builders
# --------------------------------------------------------------------------- #


def build_base_agent_runtime_spec(
    agent_id: str,
    *,
    display_name: str | None = None,
    allowed_handoff_targets: list[str] | None = None,
) -> AgentRuntimeSpec:
    """Runtime spec for a base agent (no tool/skill restriction)."""
    return AgentRuntimeSpec(
        runtime_agent_id=agent_id,
        model_agent_key=_BASE_RUNTIME_TO_MODEL_KEY.get(agent_id, agent_id),
        display_name=display_name or agent_id,
        allowed_handoff_targets=list(allowed_handoff_targets or []),
    )


def build_custom_agent_runtime_spec(
    state_entry: dict[str, Any],
    *,
    allowed_handoff_targets: list[str] | None = None,
    handoff_target_descriptions: dict[str, str] | None = None,
) -> AgentRuntimeSpec:
    """Build a custom-agent runtime spec from a ``custom_agents`` state entry."""
    tool_refs = state_entry.get("tool_refs") or []
    server_refs = [r for r in tool_refs if r.get("type") in {"server_default", "server_mcp"}]
    client_refs = [r for r in tool_refs if r.get("type") == "client"]

    custom_agent_id = state_entry.get("id")
    runtime_id = state_entry.get("runtime_agent_id") or (
        runtime_agent_id_for(custom_agent_id) if custom_agent_id else ""
    )
    return AgentRuntimeSpec(
        runtime_agent_id=runtime_id,
        model_agent_key=state_entry.get("model_agent_key") or CUSTOM_MODEL_AGENT_KEY,
        custom_agent_id=UUID(str(custom_agent_id)) if custom_agent_id else None,
        display_name=state_entry.get("name") or "Custom Agent",
        description=state_entry.get("description"),
        prompt=state_entry.get("prompt"),
        model_request=state_entry.get("model_request"),
        allowed_handoff_targets=list(allowed_handoff_targets or []),
        handoff_target_descriptions=dict(handoff_target_descriptions or {}),
        allowed_server_tool_refs=server_refs,
        allowed_client_tool_refs=client_refs,
        allowed_skill_refs=list(state_entry.get("skill_refs") or []),
        # Custom agents discover and use the full backend MCP server catalog via
        # tool_search (server tools are backend-shared, not client-specific, so
        # there is no cross-client leakage risk). Selected ``server`` tool_refs
        # are an optional hint, not a restriction. Client tools, by contrast, stay
        # strictly scoped to the exact selected instances and the active
        # device/session (see ``client_tool_search_allowlist`` and the client
        # branch of ``filter_tools_for_custom_agent``) so no client tool can leak
        # or be misinvoked across devices, sessions, or simultaneous sidecars.
        allow_all_server_tools=True,
    )


# --------------------------------------------------------------------------- #
# Exact client-tool filtering (binding + execution share this)
# --------------------------------------------------------------------------- #


def _tool_metadata(tool: Any) -> dict[str, Any]:
    return getattr(tool, "metadata", None) or {}


def client_ref_matches_metadata(ref: dict[str, Any], meta: dict[str, Any]) -> bool:
    """True only when every exact-identity field agrees."""
    return all(str(ref.get(key)) == str(meta.get(key)) for key in _CLIENT_MATCH_KEYS)


def _is_client_tool(meta: dict[str, Any]) -> bool:
    origin = str(meta.get("tool_origin") or meta.get("origin") or "")
    return "client" in origin


def _server_qualified_tool_id(tool: Any, meta: dict[str, Any]) -> str:
    qualified_id = str(meta.get("qualified_tool_id") or "").strip()
    if qualified_id:
        return qualified_id
    server_name = str(meta.get("server_name") or "").strip()
    tool_name = str(getattr(tool, "name", "") or "").strip()
    return f"{server_name}::{tool_name}" if server_name and tool_name else ""


def filter_tools_for_custom_agent(
    candidate_tools: list[Any],
    spec: AgentRuntimeSpec,
    *,
    request_device_id: str | None = None,
) -> tuple[list[Any], list[str]]:
    """Filter external (client/server) candidate tools to the spec's allowlist.

    Client tools are kept only when an allowed client-tool ref matches every
    exact-identity field AND (when supplied) the request device matches the
    stored device. Server tools are kept only by exact qualified id unless
    ``allow_all_server_tools`` is explicitly true. Returns ``(tools, warnings)``;
    a warning is emitted for each selected client tool that is unavailable.
    """
    server_ids = {
        str(r.get("qualified_tool_id"))
        for r in spec.allowed_server_tool_refs
        if r.get("qualified_tool_id")
    }
    server_ref_keys = {
        (str(r.get("server_name") or ""), str(r.get("tool_name") or ""))
        for r in spec.allowed_server_tool_refs
        if r.get("server_name") and r.get("tool_name")
    }
    allowed: list[Any] = []
    matched_ref_indexes: set[int] = set()

    for tool in candidate_tools:
        meta = _tool_metadata(tool)
        if _is_client_tool(meta):
            for index, ref in enumerate(spec.allowed_client_tool_refs):
                if not client_ref_matches_metadata(ref, meta):
                    continue
                if request_device_id and str(ref.get("device_id")) != str(request_device_id):
                    continue
                allowed.append(tool)
                matched_ref_indexes.add(index)
                break
        else:
            server_qualified_id = _server_qualified_tool_id(tool, meta)
            server_name = str(meta.get("server_name") or "").strip()
            tool_name = str(
                meta.get("aliased_from_tool_name") or getattr(tool, "name", "") or ""
            ).strip()
            if (
                spec.allow_all_server_tools
                or server_qualified_id in server_ids
                or (server_name, tool_name) in server_ref_keys
            ):
                allowed.append(tool)

    warnings: list[str] = [
        f"Selected client tool '{ref.get('qualified_tool_id')}' is unavailable "
        "in the active device session and was skipped."
        for index, ref in enumerate(spec.allowed_client_tool_refs)
        if index not in matched_ref_indexes
    ]
    return allowed, warnings


def rebase_client_tool_refs(
    refs: list[dict[str, Any]],
    live_tools: list[Any],
    *,
    request_device_id: str | None,
) -> list[dict[str, Any]]:
    """Rebase persisted client-tool refs onto the device's current live tools.

    A custom agent persists its selected client tools with the session-scoped
    identity (``session_id``, ``catalog_version``, ``tool_instance_id``) captured
    when the agent was saved. Those rotate on every sidecar reconnect/resync, so
    the exact-identity authorization match (:func:`client_ref_matches_metadata`)
    would reject the agent's own tools after a reconnect. This rebinds each
    persisted ref to the genuinely-current tool sharing its STABLE identity —
    same device and same ``qualified_tool_id`` — copying the live runtime fields
    so the strict matcher accepts the current tool. Refs whose tool is absent
    from the current device's live set are left unchanged (and reported
    unavailable downstream).

    Scoped strictly to ``request_device_id``: a tool on any other device is never
    used as a rebase source, so cross-device isolation is preserved. The strict
    matcher and dispatch-time validation are untouched — this only refreshes the
    volatile fields the matcher compares.
    """
    if not request_device_id or not refs:
        return refs

    live_by_qid: dict[str, dict[str, Any]] = {}
    for tool in live_tools:
        meta = _tool_metadata(tool)
        if not _is_client_tool(meta):
            continue
        if str(meta.get("device_id")) != str(request_device_id):
            continue
        qualified_id = str(meta.get("qualified_tool_id") or "")
        if qualified_id:
            live_by_qid[qualified_id] = meta

    rebased: list[dict[str, Any]] = []
    changed = False
    for ref in refs:
        qualified_id = str(ref.get("qualified_tool_id") or "")
        live_meta = live_by_qid.get(qualified_id)
        if live_meta is not None and str(ref.get("device_id")) == str(request_device_id):
            new_ref = dict(ref)
            new_ref["session_id"] = live_meta.get("session_id")
            new_ref["catalog_version"] = live_meta.get("catalog_version")
            new_ref["tool_instance_id"] = live_meta.get("tool_instance_id")
            rebased.append(new_ref)
            if new_ref != ref:
                changed = True
        else:
            rebased.append(ref)
    return rebased if changed else refs


# --------------------------------------------------------------------------- #
# Restricted skills
# --------------------------------------------------------------------------- #


def list_selected_skill_summaries(
    spec: AgentRuntimeSpec,
    *,
    user_id: str | None,
    device_id: str | None,
) -> list[dict[str, Any]]:
    """Prompt-safe summaries limited to the custom agent's selected skills."""
    from app.ai.skill_resolver import get_available_skill_summaries

    return get_available_skill_summaries(
        user_id=user_id,
        device_id=device_id,
        allowed_skill_refs=spec.allowed_skill_refs,
    )
