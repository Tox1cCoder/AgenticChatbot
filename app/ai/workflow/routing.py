"""Bounded, language-neutral routing context construction.

This module builds the single payload the router model reads. It contains no
intent rules: canvas state, documents, planning state, the previous agent,
custom-agent names, tools, skills, locale, and time are *descriptive context*
that the model interprets. Nothing here selects an agent.

Untrusted values (conversation text, filenames, personas, tool and skill
descriptions) are carried as JSON data in a separate ``HumanMessage``; they are
never interpolated into the router's ``SystemMessage``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.ai.workflow.inventory import AgentDescriptor, RoutingInventory

logger = logging.getLogger(__name__)

__all__ = [
    "RoutingAgentSummary",
    "RoutingCanvasDescriptor",
    "RoutingContext",
    "RoutingContextBuilder",
    "RoutingContextRequest",
    "RoutingDocumentDescriptor",
    "RoutingHistoryEntry",
    "RoutingPlanningDescriptor",
    "RoutingSkillSummary",
    "RoutingToolSummary",
]


class RoutingDocumentDescriptor(BaseModel):
    """Metadata-only view of an uploaded document. Never carries content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    filename: str
    file_type: str = ""
    status: str = ""
    upload_time: str | None = None


class RoutingCanvasDescriptor(BaseModel):
    """Identity of the active canvas. Never carries artifact source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    revision: int = 1
    title: str = ""
    message_id: str | None = None
    is_latest_assistant: bool = False


class RoutingPlanningDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    planning_mode_enabled: bool = False
    has_existing_plan: bool = False
    lifecycle: str | None = None
    summary: str = ""


class RoutingAgentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str
    display_name: str
    capability_description: str = ""
    kind: str = "base"


class RoutingToolSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""


class RoutingSkillSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source: str = ""
    description: str = ""


class RoutingHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str


class RoutingContextRequest(BaseModel):
    """Everything the builder is allowed to read for one routing call."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    message: str
    inventory: RoutingInventory
    conversation_id: str | None = None
    user_id: str | None = None
    device_id: str | None = None
    user_message_id: str | None = None
    persona: str | None = None
    active_canvas: dict[str, Any] | None = None
    planning: dict[str, Any] | None = None
    attachment_count: int = 0
    locale: str | None = None
    runtime_time: str | None = None
    allowed_skill_refs: list[dict[str, Any]] | None = None


class RoutingContext(BaseModel):
    """The bounded payload sent to the router model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: str
    conversation_summary: str = ""
    recent_messages: tuple[RoutingHistoryEntry, ...] = ()
    agents: tuple[RoutingAgentSummary, ...] = ()
    custom_agents: tuple[RoutingAgentSummary, ...] = ()
    documents: tuple[RoutingDocumentDescriptor, ...] = ()
    active_canvas: RoutingCanvasDescriptor | None = None
    planning: RoutingPlanningDescriptor = Field(default_factory=RoutingPlanningDescriptor)
    previous_final_agent_id: str | None = None
    tools: tuple[RoutingToolSummary, ...] = ()
    skills: tuple[RoutingSkillSummary, ...] = ()
    persona: str = ""
    attachment_count: int = 0
    locale: str | None = None
    runtime_time: str | None = None
    inventory_version: str = ""
    truncated_fields: tuple[str, ...] = ()
    serialized_json: str = ""

    def payload(self) -> dict[str, Any]:
        """Canonical JSON-safe body, excluding the serialized copy of itself."""
        return self.model_dump(mode="json", exclude={"serialized_json"})


class RoutingContextBuilder:
    """Builds one bounded ``RoutingContext`` per new user turn."""

    def __init__(
        self,
        *,
        history_provider: Any,
        document_repository: Any,
        settings: Any,
        skill_summary_provider: Callable[..., Sequence[dict[str, Any]]] | None = None,
        tool_summary_provider: Callable[..., Sequence[dict[str, Any]]] | None = None,
    ) -> None:
        self.history_provider = history_provider
        self.document_repository = document_repository
        self.settings = settings
        self._skill_summary_provider = skill_summary_provider
        self._tool_summary_provider = tool_summary_provider

    # -- bounds ---------------------------------------------------------

    @property
    def _field_max_chars(self) -> int:
        return int(getattr(self.settings, "router_context_field_max_chars", 500))

    @property
    def _total_max_chars(self) -> int:
        return int(getattr(self.settings, "router_context_max_chars", 24000))

    def _clip(self, value: Any, *, truncated: list[str], label: str) -> str:
        text = "" if value is None else str(value)
        limit = self._field_max_chars
        if len(text) > limit:
            truncated.append(label)
            return text[:limit]
        return text

    # -- construction ----------------------------------------------------

    async def build(self, request: RoutingContextRequest) -> RoutingContext:
        truncated: list[str] = []

        summary_text, recent = await self._history(request, truncated)
        documents = await self._documents(request, truncated)
        agents, custom_agents = self._agent_summaries(request.inventory, truncated)
        tools = self._tools(request, truncated)
        skills = self._skills(request, truncated)
        previous_final_agent_id = await self._previous_final_agent_id(request)

        canvas = None
        if isinstance(request.active_canvas, dict):
            canvas = RoutingCanvasDescriptor(
                artifact_id=str(request.active_canvas.get("artifact_id") or "canvas:main"),
                revision=int(request.active_canvas.get("revision") or 1),
                title=self._clip(
                    request.active_canvas.get("title"), truncated=truncated, label="canvas.title"
                ),
                message_id=(
                    str(request.active_canvas["message_id"])
                    if request.active_canvas.get("message_id")
                    else None
                ),
                is_latest_assistant=bool(request.active_canvas.get("is_latest_assistant")),
            )

        planning_raw = request.planning if isinstance(request.planning, dict) else {}
        planning = RoutingPlanningDescriptor(
            planning_mode_enabled=bool(planning_raw.get("planning_mode_enabled")),
            has_existing_plan=bool(planning_raw.get("has_existing_plan")),
            lifecycle=(
                str(planning_raw["lifecycle"])
                if planning_raw.get("lifecycle") is not None
                else None
            ),
            summary=self._clip(
                planning_raw.get("summary"), truncated=truncated, label="planning.summary"
            ),
        )

        context = RoutingContext(
            message=request.message or "",
            conversation_summary=summary_text,
            recent_messages=recent,
            agents=agents,
            custom_agents=custom_agents,
            documents=documents,
            active_canvas=canvas,
            planning=planning,
            previous_final_agent_id=previous_final_agent_id,
            tools=tools,
            skills=skills,
            persona=self._clip(request.persona, truncated=truncated, label="persona"),
            attachment_count=max(0, int(request.attachment_count or 0)),
            locale=request.locale,
            runtime_time=request.runtime_time,
            inventory_version=request.inventory.version,
            truncated_fields=tuple(dict.fromkeys(truncated)),
        )
        return self._with_serialized_payload(context)

    def _with_serialized_payload(self, context: RoutingContext) -> RoutingContext:
        """Attach the canonical JSON, shrinking bounded collections if needed."""
        payload = self.serialize(context)
        if len(payload) <= self._total_max_chars:
            return context.model_copy(update={"serialized_json": payload})

        truncated = list(context.truncated_fields)
        working = context
        # Drop the largest optional collections first, then clip the message.
        for label, field in (
            ("skills", "skills"),
            ("tools", "tools"),
            ("recent_messages", "recent_messages"),
            ("documents", "documents"),
            ("custom_agents", "custom_agents"),
        ):
            values = getattr(working, field)
            while values and len(self.serialize(working)) > self._total_max_chars:
                values = values[: max(0, len(values) - max(1, len(values) // 4))]
                working = working.model_copy(update={field: values})
                if label not in truncated:
                    truncated.append(label)
            if len(self.serialize(working)) <= self._total_max_chars:
                break

        payload = self.serialize(working)
        if len(payload) > self._total_max_chars:
            overflow = len(payload) - self._total_max_chars
            trimmed_message = working.message[: max(0, len(working.message) - overflow - 64)]
            if "message" not in truncated:
                truncated.append("message")
            working = working.model_copy(update={"message": trimmed_message})
            payload = self.serialize(working)

        working = working.model_copy(update={"truncated_fields": tuple(dict.fromkeys(truncated))})
        return working.model_copy(update={"serialized_json": self.serialize(working)})

    def serialize(self, context: RoutingContext) -> str:
        """Canonical JSON for the bounded context (no artifact source, no bodies)."""
        return json.dumps(
            context.payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def build_messages(
        self, context: RoutingContext, *, system_instruction: str
    ) -> list[SystemMessage | HumanMessage]:
        """Trusted instructions and untrusted data as two separate messages."""
        return [
            SystemMessage(content=system_instruction),
            HumanMessage(content=context.serialized_json or self.serialize(context)),
        ]

    # -- sources ---------------------------------------------------------

    async def _history(
        self, request: RoutingContextRequest, truncated: list[str]
    ) -> tuple[str, tuple[RoutingHistoryEntry, ...]]:
        if not (request.conversation_id and request.user_id and self.history_provider):
            return "", ()
        try:
            history = await self.history_provider.build_context(
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                current_message_id=request.user_message_id,
                agent_key="router",
            )
        except Exception as exc:
            logger.warning("Routing context could not load history: %s", exc)
            return "", ()

        summary_text = ""
        memory = getattr(history, "memory", None)
        if memory is not None:
            summary_text = self._clip(
                getattr(memory, "content", ""), truncated=truncated, label="conversation_summary"
            )

        entries: list[RoutingHistoryEntry] = []
        limit = int(getattr(self.settings, "router_history_max_messages", 12))
        for message in list(getattr(history, "messages", []) or [])[-limit:] if limit else []:
            role = getattr(getattr(message, "role", None), "value", None) or str(
                getattr(message, "role", "")
            )
            if role == "memory":
                continue
            entries.append(
                RoutingHistoryEntry(
                    role=role,
                    content=self._clip(
                        getattr(message, "content", ""),
                        truncated=truncated,
                        label="recent_messages",
                    ),
                )
            )
        return summary_text, tuple(entries)

    async def _previous_final_agent_id(self, request: RoutingContextRequest) -> str | None:
        if not (request.conversation_id and request.user_id and self.history_provider):
            return None
        lookup = getattr(self.history_provider, "get_previous_final_agent_id", None)
        if not callable(lookup):
            return None
        try:
            value = await lookup(conversation_id=request.conversation_id, user_id=request.user_id)
        except Exception as exc:
            logger.warning("Routing context could not load previous final agent: %s", exc)
            return None
        return str(value) if value else None

    async def _documents(
        self, request: RoutingContextRequest, truncated: list[str]
    ) -> tuple[RoutingDocumentDescriptor, ...]:
        if not (request.conversation_id and self.document_repository):
            return ()
        limit = int(getattr(self.settings, "router_context_max_documents", 20))
        lookup = getattr(self.document_repository, "aget_routing_descriptors", None)
        if not callable(lookup):
            return ()
        try:
            rows = await lookup(request.conversation_id, limit)
        except Exception as exc:
            logger.warning("Routing context could not load documents: %s", exc)
            return ()

        descriptors: list[RoutingDocumentDescriptor] = []
        for row in list(rows or [])[:limit]:
            data = dict(row)
            descriptors.append(
                RoutingDocumentDescriptor(
                    document_id=str(data.get("document_id") or ""),
                    filename=self._clip(
                        data.get("filename"), truncated=truncated, label="documents.filename"
                    ),
                    file_type=self._clip(
                        data.get("file_type"), truncated=truncated, label="documents.file_type"
                    ),
                    status=str(data.get("status") or ""),
                    upload_time=(
                        str(data["upload_time"]) if data.get("upload_time") is not None else None
                    ),
                )
            )
        return tuple(descriptors)

    def _agent_summaries(
        self, inventory: RoutingInventory, truncated: list[str]
    ) -> tuple[tuple[RoutingAgentSummary, ...], tuple[RoutingAgentSummary, ...]]:
        max_custom = int(getattr(self.settings, "router_context_max_custom_agents", 20))
        base: list[RoutingAgentSummary] = []
        custom: list[RoutingAgentSummary] = []
        for descriptor in inventory.agents:
            if not (descriptor.enabled and descriptor.attached):
                continue
            summary = self._agent_summary(descriptor, truncated)
            if descriptor.kind == "custom":
                custom.append(summary)
            else:
                base.append(summary)
        if len(custom) > max_custom:
            truncated.append("custom_agents")
            custom = custom[:max_custom]
        return tuple(base), tuple(custom)

    def _agent_summary(
        self, descriptor: AgentDescriptor, truncated: list[str]
    ) -> RoutingAgentSummary:
        return RoutingAgentSummary(
            agent_id=descriptor.agent_id,
            display_name=self._clip(
                descriptor.display_name, truncated=truncated, label="custom_agents.display_name"
            ),
            capability_description=self._clip(
                descriptor.capability_description,
                truncated=truncated,
                label="custom_agents.capability_description",
            ),
            kind=descriptor.kind,
        )

    def _tools(
        self, request: RoutingContextRequest, truncated: list[str]
    ) -> tuple[RoutingToolSummary, ...]:
        if self._tool_summary_provider is None:
            return ()
        limit = int(getattr(self.settings, "router_context_max_tools", 40))
        try:
            rows = self._tool_summary_provider(user_id=request.user_id, device_id=request.device_id)
        except Exception as exc:
            logger.warning("Routing context could not load tool summaries: %s", exc)
            return ()
        rows = list(rows or [])
        if len(rows) > limit:
            truncated.append("tools")
        return tuple(
            RoutingToolSummary(
                name=self._clip(row.get("name"), truncated=truncated, label="tools.name"),
                description=self._clip(
                    row.get("description"), truncated=truncated, label="tools.description"
                ),
            )
            for row in rows[:limit]
        )

    def _skills(
        self, request: RoutingContextRequest, truncated: list[str]
    ) -> tuple[RoutingSkillSummary, ...]:
        if self._skill_summary_provider is None:
            return ()
        limit = int(getattr(self.settings, "router_context_max_skills", 20))
        try:
            rows = self._skill_summary_provider(
                user_id=request.user_id,
                device_id=request.device_id,
                allowed_skill_refs=request.allowed_skill_refs,
            )
        except TypeError:
            rows = self._skill_summary_provider(
                user_id=request.user_id, device_id=request.device_id
            )
        except Exception as exc:
            logger.warning("Routing context could not load skill summaries: %s", exc)
            return ()
        rows = list(rows or [])
        if len(rows) > limit:
            truncated.append("skills")
        return tuple(
            RoutingSkillSummary(
                name=self._clip(
                    row.get("lookup_name") or row.get("name"),
                    truncated=truncated,
                    label="skills.name",
                ),
                source=str(row.get("source") or ""),
                description=self._clip(
                    row.get("description"), truncated=truncated, label="skills.description"
                ),
            )
            for row in rows[:limit]
        )

    # -- test support ----------------------------------------------------

    def build_sync_for_test(
        self,
        *,
        message: str,
        canvas_title: str = "",
        canvas_content: str = "",
        document_body: str = "",
    ) -> RoutingContext:
        """Build a context from literal values without any repository access.

        ``canvas_content`` and ``document_body`` are accepted and deliberately
        discarded so a test can prove they never reach the serialized payload.
        """
        del canvas_content, document_body
        context = RoutingContext(
            message=message,
            active_canvas=RoutingCanvasDescriptor(artifact_id="canvas:main", title=canvas_title),
            documents=(
                RoutingDocumentDescriptor(document_id="doc-1", filename="a.pdf", status="ready"),
            ),
        )
        return self._with_serialized_payload(context)
