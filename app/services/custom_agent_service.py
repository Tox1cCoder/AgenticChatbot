"""Service layer for custom agents: CRUD, attachments, validation, and locks."""

from __future__ import annotations

import contextlib
import logging
import re
import unicodedata
from collections.abc import Callable
from typing import Any
from uuid import UUID

from app.core.exceptions import (
    CustomAgentForbiddenError,
    CustomAgentInUseError,
    CustomAgentNotFoundError,
    CustomAgentValidationError,
)
from app.models.custom_agent import CustomAgent
from app.repositories.custom_agent import CustomAgentRepository
from app.schemas.custom_agent import (
    CUSTOM_MODEL_AGENT_KEY,
    ConversationCustomAgentsUpdate,
    CustomAgentCreate,
    CustomAgentOptions,
    CustomAgentRead,
    CustomAgentState,
    runtime_agent_id_for,
)
from app.utils.validation.conversation_validation import ConversationValidationUtils

logger = logging.getLogger(__name__)


def slugify(name: str) -> str:
    """Normalize a display name into a URL-safe slug."""
    value = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "custom-agent"


class CustomAgentService:
    """Owns custom-agent business rules: ownership, validation, locks, attachments."""

    def __init__(
        self,
        repository: CustomAgentRepository,
        conversation_validation_utils: ConversationValidationUtils,
        model_config_service: Any,
        *,
        generation_registry: Any | None = None,
        list_client_tool_refs: Callable[[str | None, str | None], list[dict[str, Any]]]
        | None = None,
        list_server_tool_refs: Callable[[], list[dict[str, Any]]] | None = None,
        list_skill_refs: Callable[[str | None, str | None], list[dict[str, Any]]] | None = None,
    ):
        self.repository = repository
        self.conversation_validation_utils = conversation_validation_utils
        self.model_config_service = model_config_service
        self.generation_registry = generation_registry
        self._list_client_tool_refs = list_client_tool_refs or _default_list_client_tool_refs
        self._list_server_tool_refs = list_server_tool_refs or _default_list_server_tool_refs
        self._list_skill_refs = list_skill_refs or _default_list_skill_refs

    # ------------------------------------------------------------------ reads

    def list_agents(self, owner_id: UUID) -> list[CustomAgentRead]:
        return [self._to_read(a) for a in self.repository.list_by_owner(owner_id)]

    def get_agent(self, owner_id: UUID, custom_agent_id: UUID) -> CustomAgentRead:
        return self._to_read(self._load_owned_or_raise(owner_id, custom_agent_id))

    # --------------------------------------------------------------- mutations

    def create_agent(
        self,
        owner_id: UUID,
        data: CustomAgentCreate,
        *,
        device_id: str | None = None,
    ) -> CustomAgentRead:
        fields = self._validated_fields(owner_id, data, device_id=device_id)
        slug = fields["slug"]
        if self.repository.find_live_by_slug(owner_id, slug) is not None:
            raise CustomAgentValidationError(
                detail=f"A custom agent named '{data.name}' already exists",
            )
        agent = self.repository.create(owner_id, fields)
        return self._to_read(agent)

    def update_agent(
        self,
        owner_id: UUID,
        custom_agent_id: UUID,
        data: Any,
        *,
        device_id: str | None = None,
    ) -> CustomAgentRead:
        existing = self._load_owned_or_raise(owner_id, custom_agent_id)
        self._assert_agent_not_in_use(owner_id, custom_agent_id)

        provided = data.model_dump(exclude_unset=True)
        fields: dict[str, Any] = {}

        new_name = provided.get("name", existing.name)
        new_provider = provided.get("provider_type", existing.provider_type)
        new_model = provided.get("model", existing.model)

        if "provider_type" in provided or "model" in provided:
            self._validate_model(owner_id, new_provider, new_model)

        if "tool_refs" in provided:
            tool_refs = self._dedupe_tool_refs([r.model_dump() for r in (data.tool_refs or [])])
            self._validate_tool_refs(owner_id, tool_refs, device_id)
            fields["tool_refs"] = tool_refs
        if "skill_refs" in provided:
            skill_refs = [r.model_dump() for r in (data.skill_refs or [])]
            self._validate_skill_refs(owner_id, skill_refs, device_id)
            fields["skill_refs"] = skill_refs

        for key in (
            "name",
            "description",
            "prompt",
            "provider_type",
            "model",
            "temperature",
            "reasoning_effort",
            "enabled",
        ):
            if key in provided:
                fields[key] = provided[key]

        if "name" in provided:
            new_slug = slugify(new_name)
            if (
                self.repository.find_live_by_slug(owner_id, new_slug, exclude_id=custom_agent_id)
                is not None
            ):
                raise CustomAgentValidationError(
                    detail=f"A custom agent named '{new_name}' already exists",
                )
            fields["slug"] = new_slug

        if not fields:
            return self._to_read(existing)

        agent = self.repository.update(owner_id, custom_agent_id, fields)
        if agent is None:
            raise CustomAgentNotFoundError()
        return self._to_read(agent)

    def delete_agent(self, owner_id: UUID, custom_agent_id: UUID) -> None:
        self._load_owned_or_raise(owner_id, custom_agent_id)
        self._assert_agent_not_in_use(owner_id, custom_agent_id)
        if not self.repository.delete_with_detach(owner_id, custom_agent_id):
            raise CustomAgentNotFoundError()

    # ------------------------------------------------------------- attachments

    def list_conversation_agents(
        self, owner_id: UUID, conversation_id: UUID
    ) -> list[CustomAgentRead]:
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        attachments = self.repository.list_attachments(owner_id, conversation_id)
        return [self._to_read(agent) for _attachment, agent in attachments]

    def set_conversation_agents(
        self,
        owner_id: UUID,
        conversation_id: UUID,
        payload: ConversationCustomAgentsUpdate,
    ) -> list[CustomAgentRead]:
        self.conversation_validation_utils.validate_conversation_access(owner_id, conversation_id)
        # Every id must be a live agent owned by this user.
        for custom_agent_id in payload.custom_agent_ids:
            self._load_owned_or_raise(owner_id, custom_agent_id)

        # Block detaching any agent currently active/paused in this conversation.
        current = {
            agent.id for _a, agent in self.repository.list_attachments(owner_id, conversation_id)
        }
        requested = set(payload.custom_agent_ids)
        for detached_id in current - requested:
            self._assert_attachment_not_in_use(owner_id, conversation_id, detached_id)

        self.repository.replace_attachments(owner_id, conversation_id, payload.custom_agent_ids)
        return self.list_conversation_agents(owner_id, conversation_id)

    async def get_options(self, owner_id: UUID, device_id: str | None = None) -> CustomAgentOptions:
        """Selectable providers, backend MCP tools, client tools, and skills.

        Providers/models come from the SAME refreshing snapshot the Models tab
        uses, so the custom-agent picker shows the same (and freshest) catalog.
        """
        await self.refresh_server_tool_catalog()
        client_tools = self._list_client_tool_refs(str(owner_id), device_id)
        return CustomAgentOptions(
            providers=await self._list_providers(owner_id),
            server_default_tools=list(self._list_server_tool_refs()),
            server_tools=[],
            client_tools=client_tools,
            client_servers=self._group_client_servers(client_tools),
            skills=self._list_skill_refs(str(owner_id), device_id),
        )

    @staticmethod
    def _group_client_servers(client_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse per-tool client refs into server-level picker entries.

        Sidecar MCP servers expose their tools individually in ``client_tools``;
        the picker needs a server-level node (parallel to ``server_default_tools``)
        so a user can attach a whole sidecar MCP server instead of hunting through
        individual tool rows. Derived from the same ``client_tools`` snapshot so the
        two views can never diverge. Keyed by ``(server_name, device_id)`` because the
        same server name can be live on more than one device.
        """
        servers: dict[tuple[str, str], dict[str, Any]] = {}
        for tool in client_tools:
            server_name = tool.get("server_name")
            if not server_name:
                continue
            device_id = tool.get("device_id")
            key = (str(server_name), str(device_id or ""))
            entry = servers.get(key)
            if entry is None:
                servers[key] = {
                    "server_name": server_name,
                    "device_id": device_id,
                    "tool_count": 1,
                }
            else:
                entry["tool_count"] += 1
        return sorted(
            servers.values(),
            key=lambda item: (str(item["server_name"]), str(item["device_id"] or "")),
        )

    async def refresh_server_tool_catalog(self) -> None:
        """Ensure backend MCP tools are initialized before options/validation reads."""
        if self._list_server_tool_refs is not _default_list_server_tool_refs:
            return
        try:
            from app.ai.mcp_registry import get_global_mcp_manager

            await get_global_mcp_manager()
        except Exception as exc:  # pragma: no cover - options can still render without MCP
            logger.debug("Backend MCP tool catalog unavailable for custom-agent options: %s", exc)

    async def _list_providers(self, owner_id: UUID) -> list[dict[str, Any]]:
        # Prefer the model-config snapshot (refreshes stale catalogs, identical
        # to the Models tab). Fall back to the cached catalog when unavailable.
        get_options = getattr(self.model_config_service, "get_model_config_options", None)
        if get_options is not None:
            try:
                snapshot = await get_options(owner_id)
                return [
                    {
                        "provider_type": entry.get("provider_type"),
                        "models": entry.get("models") or [],
                    }
                    for entry in (snapshot.get("providers") or [])
                ]
            except Exception:  # pragma: no cover - defensive
                pass

        provider_service = getattr(self.model_config_service, "provider_service", None)
        if provider_service is None:
            return []
        providers: list[dict[str, Any]] = []
        for provider_type in ("gemini", "openai"):
            try:
                models = provider_service.get_cached_provider_models(owner_id, provider_type)
            except Exception:  # pragma: no cover - defensive
                models = []
            providers.append({"provider_type": provider_type, "models": models})
        return providers

    def build_runtime_state(
        self, owner_id: UUID, conversation_id: UUID
    ) -> dict[str, dict[str, Any]]:
        """Build the ``custom_agents`` graph-state map keyed by runtime id."""
        attachments = self.repository.list_attachments(owner_id, conversation_id)
        state: dict[str, dict[str, Any]] = {}
        for attachment, agent in attachments:
            runtime_id = runtime_agent_id_for(agent.id)
            entry = CustomAgentState(
                id=str(agent.id),
                runtime_agent_id=runtime_id,
                model_agent_key=CUSTOM_MODEL_AGENT_KEY,
                name=agent.name,
                description=agent.description,
                prompt=agent.prompt,
                model_request={
                    "provider_type": agent.provider_type,
                    "model": agent.model,
                    "temperature": agent.temperature,
                    "reasoning_effort": agent.reasoning_effort,
                },
                tool_refs=list(agent.tool_refs or []),
                skill_refs=list(agent.skill_refs or []),
                agent_order=attachment.agent_order,
            )
            state[runtime_id] = entry.model_dump()
        return state

    # --------------------------------------------------------------- internals

    def _load_owned_or_raise(self, owner_id: UUID, custom_agent_id: UUID) -> CustomAgent:
        agent = self.repository.get_any(custom_agent_id)
        if agent is None:
            raise CustomAgentNotFoundError()
        if agent.owner_id != owner_id:
            raise CustomAgentForbiddenError()
        return agent

    def _validated_fields(
        self, owner_id: UUID, data: CustomAgentCreate, *, device_id: str | None
    ) -> dict[str, Any]:
        self._validate_model(owner_id, data.provider_type, data.model)
        tool_refs = self._dedupe_tool_refs([r.model_dump() for r in data.tool_refs])
        skill_refs = [r.model_dump() for r in data.skill_refs]
        self._validate_tool_refs(owner_id, tool_refs, device_id)
        self._validate_skill_refs(owner_id, skill_refs, device_id)
        return {
            "name": data.name,
            "slug": slugify(data.name),
            "description": data.description,
            "prompt": data.prompt,
            "provider_type": data.provider_type,
            "model": data.model,
            "temperature": data.temperature,
            "reasoning_effort": data.reasoning_effort,
            "tool_refs": tool_refs,
            "skill_refs": skill_refs,
            "enabled": data.enabled,
        }

    @staticmethod
    def _dedupe_tool_refs(tool_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse duplicate tool refs, preserving first-seen order.

        A tool selected both via its whole-server group and individually arrives
        twice; this keeps a single ref so the stored selection mirrors what the
        picker shows and no tool is bound twice. Server tools dedupe by qualified
        id; client tools by their full exact identity (qualified id +
        device/session/instance), matching the runtime client-tool match keys.
        """
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for ref in tool_refs:
            if ref.get("type") == "client":
                key: tuple[Any, ...] = (
                    "client",
                    ref.get("qualified_tool_id"),
                    ref.get("device_id"),
                    ref.get("session_id"),
                    ref.get("tool_instance_id"),
                )
            else:
                key = ("server", ref.get("qualified_tool_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped

    def _validate_model(self, owner_id: UUID, provider_type: str, model: str) -> None:
        try:
            self.model_config_service.validate_provider_model(owner_id, provider_type, model)
        except CustomAgentValidationError:
            raise
        except Exception as exc:
            raise CustomAgentValidationError(
                detail=f"Invalid model '{provider_type}/{model}': {exc}"
            ) from exc

    def _validate_tool_refs(
        self, owner_id: UUID, tool_refs: list[dict[str, Any]], device_id: str | None
    ) -> None:
        if not tool_refs:
            return
        available = {
            (t.get("qualified_tool_id"), t.get("device_id"), t.get("tool_instance_id"))
            for t in self._list_client_tool_refs(str(owner_id), device_id)
        }
        available_server_ids = {
            str(t.get("qualified_tool_id"))
            for t in self._list_server_tool_refs()
            if t.get("qualified_tool_id")
        }
        for ref in tool_refs:
            ref_type = ref.get("type")
            if ref_type in {"server_mcp", "server_default"}:
                if str(ref.get("qualified_tool_id")) not in available_server_ids:
                    raise CustomAgentValidationError(
                        detail=(
                            f"Server MCP tool '{ref.get('qualified_tool_id')}' is not available"
                        ),
                    )
            elif ref_type == "client":
                key = (
                    ref.get("qualified_tool_id"),
                    ref.get("device_id"),
                    ref.get("tool_instance_id"),
                )
                if key not in available:
                    raise CustomAgentValidationError(
                        detail=(
                            f"Client tool '{ref.get('qualified_tool_id')}' is not available "
                            "in the active device catalog"
                        ),
                    )
            else:
                raise CustomAgentValidationError(detail=f"Invalid tool ref type: {ref_type}")

    def _validate_skill_refs(
        self, owner_id: UUID, skill_refs: list[dict[str, Any]], device_id: str | None
    ) -> None:
        if not skill_refs:
            return
        available = {
            (s.get("source"), s.get("lookup_name"))
            for s in self._list_skill_refs(str(owner_id), device_id)
        }
        for ref in skill_refs:
            key = (ref.get("source"), ref.get("lookup_name"))
            if key not in available:
                raise CustomAgentValidationError(
                    detail=f"Skill '{ref.get('lookup_name')}' is not available",
                )

    def _assert_agent_not_in_use(self, owner_id: UUID, custom_agent_id: UUID) -> None:
        """Block edit/delete while the runtime agent is active or paused (Task 4)."""
        if self.generation_registry is None:
            return
        runtime_id = runtime_agent_id_for(custom_agent_id)
        if self.generation_registry.is_runtime_agent_in_use(str(owner_id), runtime_id):
            raise CustomAgentInUseError()
        # Conservative gate: any active run in a conversation this agent is
        # attached to whose selected agent is not yet known.
        for conversation_id in self.repository.list_attached_conversation_ids(custom_agent_id):
            if self.generation_registry.has_active_unknown_agent_in_conversation(
                str(owner_id), str(conversation_id)
            ):
                raise CustomAgentInUseError()

    def _assert_attachment_not_in_use(
        self, owner_id: UUID, conversation_id: UUID, custom_agent_id: UUID
    ) -> None:
        if self.generation_registry is None:
            return
        runtime_id = runtime_agent_id_for(custom_agent_id)
        if self.generation_registry.is_runtime_agent_in_use(
            str(owner_id), runtime_id, conversation_id=str(conversation_id)
        ):
            raise CustomAgentInUseError()

    @staticmethod
    def _to_read(agent: CustomAgent) -> CustomAgentRead:
        return CustomAgentRead.model_validate(agent)


# --------------------------------------------------------------------------- #
# Default lookups (production wiring). Tests inject fakes instead.
# --------------------------------------------------------------------------- #


def _default_list_client_tool_refs(
    user_id: str | None, device_id: str | None
) -> list[dict[str, Any]]:
    if not user_id or not device_id:
        return []
    try:
        from app.ai.client_tool_catalog import get_client_tool_catalog

        catalog = get_client_tool_catalog(device_id, user_id)
        descriptors = catalog.list_all()
    except Exception:  # pragma: no cover - defensive; no active catalog
        return []
    return [
        {
            "type": "client",
            "device_id": getattr(d, "device_id", None),
            "session_id": getattr(d, "session_id", None),
            "catalog_version": str(getattr(d, "catalog_version", "")),
            "tool_instance_id": getattr(d, "tool_instance_id", None),
            "server_name": getattr(d, "server_name", None),
            "qualified_tool_id": getattr(d, "qualified_tool_id", None),
            "tool_name": getattr(d, "tool_name", None),
        }
        for d in descriptors
    ]


def _default_list_server_tool_refs() -> list[dict[str, Any]]:
    try:
        from app.ai.mcp_registry import MCPRegistry

        manager = MCPRegistry.get_manager_sync()
    except Exception:  # pragma: no cover - defensive
        manager = None
    if manager is None:
        return []

    refs: list[dict[str, Any]] = []
    for server_name, tools in getattr(manager, "_server_tools", {}).items():
        for tool in tools or []:
            tool_name = str(getattr(tool, "name", "") or "").strip()
            if not tool_name:
                continue
            args_schema = {}
            with contextlib.suppress(Exception):
                args_schema = manager.get_tool_args_schema(tool)
            refs.append(
                {
                    "type": "server_mcp",
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "qualified_tool_id": f"{server_name}::{tool_name}",
                    "description": getattr(tool, "description", "") or "",
                    "args_schema": args_schema,
                }
            )
    refs.sort(key=lambda item: (str(item.get("server_name")), str(item.get("tool_name"))))
    return refs


def _default_list_skill_refs(user_id: str | None, device_id: str | None) -> list[dict[str, Any]]:
    try:
        from app.ai.skill_resolver import list_resolved_skills

        skills = list_resolved_skills(user_id=user_id, device_id=device_id)
    except Exception:  # pragma: no cover - defensive
        return []
    return [{"source": s.source, "lookup_name": s.lookup_name, "name": s.name} for s in skills]
