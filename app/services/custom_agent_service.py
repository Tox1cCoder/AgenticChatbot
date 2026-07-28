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
    CustomAgentAvailability,
    CustomAgentCreate,
    CustomAgentOptions,
    CustomAgentRead,
    CustomAgentState,
    DeviceCatalogSnapshot,
    runtime_agent_id_for,
)
from app.services.client_device_service import ClientDeviceService
from app.services.custom_agent_capability_resolver import (
    client_tool_binding_key,
    client_tool_logical_key,
    resolve_custom_agent_capabilities,
    skill_logical_key,
    skill_refs_match,
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
        get_device_snapshot: Callable[[str | None, str | None], dict[str, Any]] | None = None,
    ):
        self.repository = repository
        self.conversation_validation_utils = conversation_validation_utils
        self.model_config_service = model_config_service
        self.generation_registry = generation_registry
        self._list_client_tool_refs = list_client_tool_refs or _default_list_client_tool_refs
        self._list_server_tool_refs = list_server_tool_refs or _default_list_server_tool_refs
        self._list_skill_refs = list_skill_refs or _default_list_skill_refs
        self._get_device_snapshot = get_device_snapshot or _default_get_device_snapshot

    # ------------------------------------------------------------------ reads

    def list_agents(self, owner_id: UUID, *, device_id: str | None = None) -> list[CustomAgentRead]:
        snapshot, live_tools, live_skills = self._load_device_context(owner_id, device_id)
        return [
            self._to_read(
                agent,
                availability=self._availability_for(
                    agent,
                    snapshot=snapshot,
                    live_tools=live_tools,
                    live_skills=live_skills,
                ),
            )
            for agent in self.repository.list_by_owner(owner_id)
        ]

    def get_agent(
        self, owner_id: UUID, custom_agent_id: UUID, *, device_id: str | None = None
    ) -> CustomAgentRead:
        agent = self._load_owned_or_raise(owner_id, custom_agent_id)
        snapshot, live_tools, live_skills = self._load_device_context(owner_id, device_id)
        return self._to_read(
            agent,
            availability=self._availability_for(
                agent,
                snapshot=snapshot,
                live_tools=live_tools,
                live_skills=live_skills,
            ),
        )

    @staticmethod
    def _snapshot_identity(snapshot: dict[str, Any]) -> tuple[Any, ...]:
        return (
            snapshot.get("device_id"),
            snapshot.get("session_id"),
            snapshot.get("tool_catalog_version"),
            snapshot.get("skill_catalog_version"),
            snapshot.get("status"),
        )

    def _load_device_context(
        self, owner_id: UUID, device_id: str | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Read a caller-owned snapshot and its catalogs under one consistent identity.

        Retries once if a reconnect/resync rotates identity while catalogs are being
        materialized, then fails closed as unavailable rather than pairing one
        session's options with another session's cache key.
        """
        user_id = str(owner_id)
        snapshot = self._get_device_snapshot(user_id, device_id)
        if snapshot.get("status") != "ready":
            return snapshot, [], []

        for _attempt in range(2):
            before = snapshot
            live_tools = self._list_client_tool_refs(user_id, device_id)
            live_skills = self._list_skill_refs(user_id, device_id)
            after = self._get_device_snapshot(user_id, device_id)
            if after.get("status") == "ready" and self._snapshot_identity(
                before
            ) == self._snapshot_identity(after):
                return after, live_tools, live_skills
            snapshot = after
            if snapshot.get("status") != "ready":
                return snapshot, [], []

        unstable = dict(snapshot)
        unstable["status"] = "unavailable"
        return unstable, [], []

    @staticmethod
    def _availability_for(
        agent: CustomAgent,
        *,
        snapshot: dict[str, Any],
        live_tools: list[dict[str, Any]],
        live_skills: list[dict[str, Any]],
    ) -> CustomAgentAvailability:
        resolution = resolve_custom_agent_capabilities(
            selected_tool_refs=list(agent.tool_refs or []),
            selected_skill_refs=list(agent.skill_refs or []),
            live_tool_refs=live_tools,
            live_skill_refs=live_skills,
            request_device_id=snapshot.get("device_id"),
            device_available=snapshot.get("status") == "ready",
        )
        return CustomAgentAvailability(
            status=resolution.status,
            device_id=snapshot.get("device_id"),
            session_id=snapshot.get("session_id"),
            missing_tools=resolution.missing_tools,
            missing_skills=resolution.missing_skills,
            warnings=resolution.warnings,
        )

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
        new_reasoning_effort = provided.get("reasoning_effort", existing.reasoning_effort)

        if {"provider_type", "model", "reasoning_effort"} & provided.keys():
            new_reasoning_effort = self._validate_model(
                owner_id, new_provider, new_model, new_reasoning_effort
            )

        if "tool_refs" in provided:
            tool_refs = self._dedupe_tool_refs([r.model_dump() for r in (data.tool_refs or [])])
            self._validate_tool_refs(
                owner_id,
                tool_refs,
                device_id,
                existing_refs=list(existing.tool_refs or []),
            )
            fields["tool_refs"] = tool_refs
        if "skill_refs" in provided:
            skill_refs = self._dedupe_skill_refs([r.model_dump() for r in (data.skill_refs or [])])
            self._validate_skill_refs(
                owner_id,
                skill_refs,
                device_id,
                existing_refs=list(existing.skill_refs or []),
            )
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
        if {"provider_type", "model", "reasoning_effort"} & provided.keys():
            fields["reasoning_effort"] = new_reasoning_effort

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
        snapshot, client_tools, skills = self._load_device_context(owner_id, device_id)
        return CustomAgentOptions(
            providers=await self._list_providers(owner_id),
            server_default_tools=list(self._list_server_tool_refs()),
            server_tools=[],
            client_tools=client_tools,
            client_servers=self._group_client_servers(client_tools),
            skills=skills,
            device_snapshot=DeviceCatalogSnapshot.model_validate(snapshot),
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
        return self._runtime_state_from_attachments(attachments)

    async def abuild_runtime_state(
        self, owner_id: UUID, conversation_id: UUID
    ) -> dict[str, dict[str, Any]]:
        """Async twin of :meth:`build_runtime_state`."""
        attachments = await self.repository.alist_attachments(owner_id, conversation_id)
        return self._runtime_state_from_attachments(attachments)

    @staticmethod
    def _runtime_state_from_attachments(attachments) -> dict[str, dict[str, Any]]:
        """Pure projection of attachment rows into graph state. No database access."""
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
        reasoning_effort = self._validate_model(
            owner_id, data.provider_type, data.model, data.reasoning_effort
        )
        tool_refs = self._dedupe_tool_refs([r.model_dump() for r in data.tool_refs])
        skill_refs = self._dedupe_skill_refs([r.model_dump() for r in data.skill_refs])
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
            "reasoning_effort": reasoning_effort,
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
                logical_key = client_tool_logical_key(ref)
                key: tuple[Any, ...] = (
                    ("client", *logical_key)
                    if logical_key is not None
                    else (
                        "client-invalid",
                        ref.get("device_id"),
                        ref.get("tool_instance_id"),
                    )
                )
            else:
                key = ("server", ref.get("qualified_tool_id"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped

    @staticmethod
    def _dedupe_skill_refs(skill_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse duplicate skill refs by normalized logical identity, first-seen order.

        Legacy ``source="server"`` refs normalize to client (server-owned skills were
        removed), so a legacy ref and its current client ref for the same skill are
        one selection.
        """
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for ref in skill_refs:
            logical_key = skill_logical_key(ref)
            key = (
                ("skill", *logical_key)
                if logical_key is not None
                else (
                    "skill-invalid",
                    ref.get("source"),
                    ref.get("lookup_name"),
                    ref.get("name"),
                )
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped

    def _validate_model(
        self,
        owner_id: UUID,
        provider_type: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> str | None:
        try:
            return self.model_config_service.validate_provider_model(
                owner_id,
                provider_type,
                model,
                reasoning_effort=reasoning_effort,
            )
        except CustomAgentValidationError:
            raise
        except Exception as exc:
            raise CustomAgentValidationError(
                detail=f"Invalid model '{provider_type}/{model}': {exc}"
            ) from exc

    def _validate_tool_refs(
        self,
        owner_id: UUID,
        tool_refs: list[dict[str, Any]],
        device_id: str | None,
        *,
        existing_refs: list[dict[str, Any]] | None = None,
    ) -> None:
        if not tool_refs:
            return
        available_client_bindings = {
            key
            for tool in self._list_client_tool_refs(str(owner_id), device_id)
            if (key := client_tool_binding_key(tool)) is not None
        }
        existing_client_keys = {
            key
            for ref in (existing_refs or [])
            if (key := client_tool_logical_key(ref)) is not None
        }
        available_server_ids = {
            str(tool.get("qualified_tool_id"))
            for tool in self._list_server_tool_refs()
            if tool.get("qualified_tool_id")
        }
        for ref in tool_refs:
            ref_type = ref.get("type")
            if ref_type in {"server_mcp", "server_default"}:
                if str(ref.get("qualified_tool_id")) not in available_server_ids:
                    raise CustomAgentValidationError(
                        detail=f"Server MCP tool '{ref.get('qualified_tool_id')}' is not available",
                    )
                continue
            if ref_type != "client":
                raise CustomAgentValidationError(detail=f"Invalid tool ref type: {ref_type}")
            logical_key = client_tool_logical_key(ref)
            binding_key = client_tool_binding_key(ref)
            if logical_key is None or (
                logical_key not in existing_client_keys
                and binding_key not in available_client_bindings
            ):
                raise CustomAgentValidationError(
                    detail=(
                        f"Client tool '{ref.get('qualified_tool_id')}' is not available "
                        "in the active device catalog"
                    ),
                )

    def _validate_skill_refs(
        self,
        owner_id: UUID,
        skill_refs: list[dict[str, Any]],
        device_id: str | None,
        *,
        existing_refs: list[dict[str, Any]] | None = None,
    ) -> None:
        if not skill_refs:
            return
        available_skills = self._list_skill_refs(str(owner_id), device_id)
        existing_skills = list(existing_refs or [])
        for ref in skill_refs:
            retains_existing = any(skill_refs_match(ref, old) for old in existing_skills)
            is_exact_live_option = any(
                str(ref.get("source") or "") == str(live.get("source") or "")
                and str(ref.get("lookup_name") or "") == str(live.get("lookup_name") or "")
                and str(ref.get("name") or "") == str(live.get("name") or "")
                for live in available_skills
            )
            if not retains_existing and not is_exact_live_option:
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
    def _to_read(
        agent: CustomAgent,
        availability: CustomAgentAvailability | None = None,
    ) -> CustomAgentRead:
        value = CustomAgentRead.model_validate(agent)
        return value.model_copy(update={"availability": availability})


# --------------------------------------------------------------------------- #
# Default lookups (production wiring). Tests inject fakes instead.
# --------------------------------------------------------------------------- #


def _default_get_device_snapshot(user_id: str | None, device_id: str | None) -> dict[str, Any]:
    unavailable = {
        "device_id": device_id,
        "session_id": None,
        "tool_catalog_version": None,
        "skill_catalog_version": None,
        "status": "unavailable",
    }
    if not user_id or not device_id:
        return unavailable
    try:
        device_uuid = UUID(str(device_id))
    except (TypeError, ValueError, AttributeError):
        return unavailable
    session = ClientDeviceService.lookup_active_session(device_uuid)
    if session is None or str(session.user_id) != str(user_id):
        return unavailable
    if session.tool_catalog_version <= 0 or session.skill_catalog_version <= 0:
        return {
            "device_id": str(session.device_id),
            "session_id": session.session_id,
            "tool_catalog_version": session.tool_catalog_version,
            "skill_catalog_version": session.skill_catalog_version,
            "status": "unavailable",
        }
    return {
        "device_id": str(session.device_id),
        "session_id": session.session_id,
        "tool_catalog_version": session.tool_catalog_version,
        "skill_catalog_version": session.skill_catalog_version,
        "status": "ready",
    }


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
