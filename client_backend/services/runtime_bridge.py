"""
Client runtime bridge service.

Maintains the authenticated outbound runtime connection to the canonical server:
- registers the current device
- opens the persistent WebSocket
- syncs tool and skill catalogs
- executes client-local MCP tool requests
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from app.schemas.runtime_protocol import (
    RuntimeAckMessage,
    RuntimeErrorMessage,
    RuntimeHeartbeatMessage,
    RuntimeMessage,
    dump_runtime_message,
    parse_runtime_message,
)
from client_backend import __version__
from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.security import generate_device_identifier
from client_backend.schemas.runtime import (
    DeviceInfo,
    DeviceRegistrationResult,
    RuntimeErrorContext,
    RuntimeState,
    RuntimeStatus,
    ToolDispatchRequest,
    ToolDispatchResult,
)
from client_backend.services.local_mcp_manager import get_mcp_manager, shutdown_mcp_manager
from client_backend.services.local_skills_registry import (
    get_skills_registry,
    initialize_skills_registry,
)
from client_backend.services.server_api import ServerAPIClient, get_server_client
from client_backend.services.skill_runtime.execution import SkillExecutionEngine
from client_backend.services.skill_runtime.manager import SkillRuntimeManager
from shared.skills.errors import SkillRuntimeError


def _make_tool_instance_id(
    device_id: str,
    session_id: str,
    qualified_tool_id: str,
    catalog_version: int,
    source_hash: str = "",
) -> str:
    """
    Compute the opaque tool capability identifier that the server will echo
    back in ToolDispatchRequest for sidecar-side validation.
    """
    composite = f"{device_id}:{session_id}:{qualified_tool_id}:{catalog_version}:{source_hash}"
    return hashlib.sha256(composite.encode()).hexdigest()[:16]


logger = get_logger(__name__)


class ClientExecutionTimeoutError(TimeoutError):
    """The complete client runtime operation exceeded its execution deadline."""


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    """Retrieve the terminal state of work abandoned after its deadline."""

    with suppress(asyncio.CancelledError):
        task.exception()


class RuntimeBridgeService:
    """Owns the device registration + runtime WebSocket lifecycle."""

    def __init__(
        self,
        server_client: ServerAPIClient | None = None,
        websocket_base_url: str | None = None,
    ):
        self._server_client = server_client or get_server_client()
        self._websocket_base_url = websocket_base_url
        self._device_identifier = generate_device_identifier()
        self._runtime_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._stop_requested = False
        self._connected_event = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._websocket = None
        self._device_id: str | None = None
        self._session_id: str | None = None
        self._tool_catalog_version: int = 0  # Mirrors server-side catalog version counter
        # Last synced catalog keyed by qualified_id for validation
        self._current_tool_catalog: dict[str, dict] = {}
        self._state = RuntimeState(
            status=RuntimeStatus.DISCONNECTED,
            device_info=self.get_device_info(),
            server_url=self._server_client.base_url,
            connected_at=None,
            last_heartbeat=None,
            session_id=None,
            error_message=None,
        )

    def get_device_info(self) -> DeviceInfo:
        """Return local device metadata used for registration and status."""
        return DeviceInfo(
            device_id=self._device_id,
            device_identifier=self._device_identifier,
            device_name=client_settings.device_name,
            platform=platform.system().lower(),
            app_version=__version__,
            runtime_version=__version__,
        )

    def get_device_identifier(self) -> str:
        """Return the stable local installation identifier."""
        return self._device_identifier

    def get_registered_device_id(self) -> str | None:
        """Return the server-assigned device UUID when connected/registered."""
        return self._device_id

    def get_runtime_state(self) -> RuntimeState:
        """Return a copy-safe snapshot of runtime state."""
        return self._state.model_copy(deep=True)

    def is_connected(self) -> bool:
        return self._state.status == RuntimeStatus.CONNECTED and self._websocket is not None

    async def start(
        self,
        *,
        wait_for_connection: bool = True,
        timeout_seconds: int | None = None,
    ) -> bool:
        """
        Start the runtime bridge in the background.

        Returns True once the first connection succeeds when `wait_for_connection` is enabled.
        """
        if not self._server_client.is_authenticated():
            self._set_state(
                status=RuntimeStatus.DISCONNECTED,
                device_info=self.get_device_info(),
                error_message="Server authentication required before starting runtime bridge.",
            )
            return False

        if self._runtime_task and not self._runtime_task.done():
            if not self.is_connected():
                self._connected_event.clear()
            if not wait_for_connection:
                return True
            return await self._wait_for_initial_connection(timeout_seconds)

        self._stop_requested = False
        self._connected_event = asyncio.Event()
        self._runtime_task = asyncio.create_task(self._run_forever())

        if not wait_for_connection:
            return True

        return await self._wait_for_initial_connection(timeout_seconds)

    async def stop(self) -> None:
        """Stop the runtime bridge and close all local runtime resources."""
        self._stop_requested = True
        self._connected_event.clear()
        device_id = self._device_id

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        if self._websocket is not None:
            with suppress(Exception):
                await self._websocket.close()
            self._websocket = None

        if self._runtime_task:
            self._runtime_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._runtime_task
            self._runtime_task = None

        if device_id and self._server_client.is_authenticated():
            await self._wait_for_server_disconnect(device_id)

        await shutdown_mcp_manager()

        self._device_id = None
        self._session_id = None
        self._set_state(
            status=RuntimeStatus.DISCONNECTED,
            device_info=self.get_device_info(),
            connected_at=None,
            last_heartbeat=None,
            session_id=None,
            error_message=None,
        )

    async def refresh_catalogs(self) -> None:
        """Resync tool and skill catalogs to the server for the current runtime session."""
        if not self._device_id:
            return

        tool_catalog = await self._build_tool_catalog()
        skill_catalog = get_skills_registry().get_skill_catalog(include_content=False)

        tool_sync = await self._server_client.update_device_tool_catalog(
            device_id=self._device_id,
            catalog=tool_catalog,
        )
        self._tool_catalog_version += 1  # Mirror server-side increment
        # Cache catalog by qualified_id for fast validation in _validate_tool_request
        self._current_tool_catalog = {
            entry["qualified_id"]: entry
            for entry in tool_catalog.get("tools", [])
            if entry.get("qualified_id")
        }
        skill_sync = await self._server_client.update_device_skill_catalog(
            device_id=self._device_id,
            catalog=skill_catalog,
        )
        logger.debug(
            "Runtime catalogs synced for device %s (tools=%s, skills=%s)",
            self._device_id,
            tool_sync.tool_count,
            skill_sync.skill_count,
        )

    async def _wait_for_initial_connection(self, timeout_seconds: int | None = None) -> bool:
        timeout_value = timeout_seconds or client_settings.server_api_timeout_seconds
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout_value)
            return True
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for initial runtime bridge connection")
            return False

    def _set_state(self, **changes: Any) -> None:
        payload = self._state.model_dump()
        payload.update(changes)
        self._state = RuntimeState(**payload)

    @staticmethod
    def _build_runtime_error_context(
        exc: Exception,
        *,
        detail: dict[str, Any] | None = None,
    ) -> RuntimeErrorContext:
        if isinstance(exc, ClientExecutionTimeoutError):
            return RuntimeErrorContext(
                message="Client runtime operation timed out.",
                code="TIMEOUT_CLIENT_EXECUTION",
                detail=detail or None,
            )

        if isinstance(exc, SkillRuntimeError):
            merged_detail = dict(detail or {})
            if exc.repair is not None:
                merged_detail["repair"] = exc.repair
            return RuntimeErrorContext(
                message=exc.message,
                code=exc.code,
                detail=merged_detail or None,
            )

        return RuntimeErrorContext(
            message=str(exc),
            code=exc.__class__.__name__,
            detail=detail or None,
        )

    @staticmethod
    def _runtime_error_context_from_message(message: RuntimeErrorMessage) -> RuntimeErrorContext:
        if message.error_context is not None:
            return message.error_context

        return RuntimeErrorContext(
            message=message.message,
            code=message.code,
        )

    async def _run_forever(self) -> None:
        attempts = 0

        while not self._stop_requested:
            try:
                await self._initialize_local_runtime()
                registration = await self._register_device()
                self._device_id = registration.device_id
                self._session_id = registration.session_id
                self._tool_catalog_version = 0  # Reset on new session
                self._current_tool_catalog = {}

                self._set_state(
                    status=RuntimeStatus.CONNECTING,
                    device_info=self.get_device_info(),
                    session_id=self._session_id,
                    error_message=None,
                )

                await self._connect_and_serve()
                attempts = 0

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts += 1
                if exc.__class__.__name__.startswith("ConnectionClosed"):
                    logger.info("Runtime bridge WebSocket closed: %s", exc)
                else:
                    logger.warning("Runtime bridge loop error: %s", exc, exc_info=True)

                if self._stop_requested:
                    break

                if attempts >= client_settings.max_reconnect_attempts:
                    self._set_state(
                        status=RuntimeStatus.ERROR,
                        error_message=str(exc),
                    )
                    return

                self._set_state(
                    status=RuntimeStatus.RECONNECTING,
                    error_message=str(exc),
                )
                await asyncio.sleep(client_settings.reconnect_delay_seconds)

    async def _initialize_local_runtime(self) -> None:
        with suppress(Exception):
            await get_mcp_manager().initialize()
        await initialize_skills_registry()

    async def _wait_for_server_disconnect(self, device_id: str) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0

        while loop.time() < deadline:
            try:
                payload = await self._server_client.list_connected_devices()
            except Exception as exc:
                logger.debug("Runtime bridge disconnect verification failed: %s", exc)
                return

            if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
                payload = payload["data"]

            connected_devices = (
                payload.get("connected_devices", []) if isinstance(payload, dict) else []
            )
            if all(str(item.get("device_id") or "") != device_id for item in connected_devices):
                return

            await asyncio.sleep(0.2)

    async def _register_device(self) -> DeviceRegistrationResult:
        capabilities = {
            "local_mcp": True,
            "local_skills": True,
        }
        info = self.get_device_info()
        return await self._server_client.register_device(
            device_identifier=info.device_identifier,
            display_name=info.device_name,
            platform=info.platform,
            app_version=info.app_version,
            runtime_version=info.runtime_version,
            capabilities=capabilities,
        )

    async def _connect_and_serve(self) -> None:
        try:
            from websockets import connect as websocket_connect
        except ImportError as exc:
            raise RuntimeError(
                "The 'websockets' package is required for the client runtime bridge."
            ) from exc

        websocket_url = (
            self._build_websocket_url()
            if self._websocket_base_url
            else self._server_client.build_runtime_websocket_url(
                device_id=self._device_id or "",
                session_id=self._session_id or "",
            )
        )

        async with websocket_connect(websocket_url) as websocket:
            self._websocket = websocket

            raw_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=client_settings.server_api_timeout_seconds,
            )
            message = self._decode_runtime_message(raw_message)
            if not isinstance(message, RuntimeAckMessage):
                raise RuntimeError(
                    f"Unexpected runtime handshake message: {dump_runtime_message(message)}"
                )

            now = datetime.now(timezone.utc)
            self._set_state(
                status=RuntimeStatus.CONNECTED,
                connected_at=now,
                last_heartbeat=now,
                session_id=self._session_id,
                error_message=None,
            )
            self._connected_event.set()

            await self.refresh_catalogs()

            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            try:
                await self._receive_loop()
            finally:
                self._connected_event.clear()
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await self._heartbeat_task
                    self._heartbeat_task = None
                self._websocket = None

    def _build_websocket_url(self) -> str:
        base = self._websocket_base_url.rstrip("/")
        return f"{base}/device-runtime/{self._device_id}/connect?session_id={self._session_id}"

    async def _heartbeat_loop(self) -> None:
        while not self._stop_requested and self._websocket is not None:
            await asyncio.sleep(client_settings.heartbeat_interval_seconds)
            await self._send_runtime_message(RuntimeHeartbeatMessage())
            self._set_state(last_heartbeat=datetime.now(timezone.utc))

    async def _receive_loop(self) -> None:
        while not self._stop_requested and self._websocket is not None:
            raw_message = await self._websocket.recv()
            try:
                message = self._decode_runtime_message(raw_message)
            except Exception as exc:
                logger.warning("Runtime bridge received invalid WebSocket message: %s", exc)
                continue
            await self._handle_server_message(message)

    async def _handle_server_message(self, message: RuntimeMessage) -> None:
        if isinstance(message, ToolDispatchRequest):
            await self._handle_tool_request(message)
            return

        if isinstance(message, RuntimeAckMessage):
            return

        if isinstance(message, RuntimeHeartbeatMessage):
            self._set_state(last_heartbeat=datetime.now(timezone.utc))
            return

        if isinstance(message, RuntimeErrorMessage):
            error_context = self._runtime_error_context_from_message(message)
            self._set_state(error_message=error_context.message)
            logger.error(
                "Runtime bridge received server error (code=%s, detail=%s): %s",
                error_context.code,
                error_context.detail,
                error_context.message,
            )
            return

        logger.warning(
            "Runtime bridge received unsupported message: %s",
            message.type,
        )

    def _validate_tool_request(self, request: ToolDispatchRequest) -> str | None:
        """
        Validate an incoming tool request against current advertised capability record.

        Returns an error string if invalid, None if acceptable.
        """
        # Validate session_id if provided
        if request.expected_session_id and request.expected_session_id != self._session_id:
            return (
                f"Session mismatch: server expects session_id={request.expected_session_id!r} "
                f"but current session is {self._session_id!r}. "
                "The sidecar has reconnected; retry from the active session."
            )

        # Validate catalog version if provided
        if (
            request.expected_catalog_version is not None
            and request.expected_catalog_version != self._tool_catalog_version
        ):
            return (
                "Catalog version mismatch: server expects "
                f"version={request.expected_catalog_version} "
                f"but current version is {self._tool_catalog_version}. "
                "The tool catalog has changed; re-sync and retry."
            )

        qualified_id = request.qualified_tool_id
        if qualified_id == "client_skill::activate":
            if request.tool_name and request.tool_name != "activate_skill":
                return (
                    f"Tool name mismatch: request.tool_name={request.tool_name!r} "
                    "does not match reserved client skill activation request."
                )
            return None

        if qualified_id.startswith("skill::") and (
            not request.expected_session_id
            or request.expected_catalog_version is None
            or not request.tool_instance_id
        ):
            return (
                "Skill command binding is incomplete: session, catalog version, "
                "and tool instance are all required. Search the active device "
                "catalog again and retry."
            )

        catalog_entry = self._current_tool_catalog.get(qualified_id)

        if catalog_entry is None:
            return (
                f"Unknown tool: qualified_tool_id={qualified_id!r} is not in the "
                "current tool catalog. "
                "The tool may have been removed; re-sync and retry."
            )

        # Validate tool_instance_id if both sides provided it
        if (
            request.tool_instance_id
            and catalog_entry.get("tool_instance_id")
            and request.tool_instance_id != catalog_entry["tool_instance_id"]
        ):
            return (
                f"tool_instance_id mismatch for {qualified_id!r}: "
                f"server sent {request.tool_instance_id!r} "
                f"but catalog has {catalog_entry['tool_instance_id']!r}. "
                "The catalog has been updated; re-sync and retry."
            )

        # Validate tool_name matches catalog record
        catalog_name = catalog_entry.get("name")
        if catalog_name and catalog_name != request.tool_name:
            return (
                f"Tool name mismatch: request.tool_name={request.tool_name!r} "
                f"does not match catalog name={catalog_name!r} for {qualified_id!r}."
            )

        return None

    async def _handle_tool_request(self, request: ToolDispatchRequest) -> None:
        started_at = datetime.now(timezone.utc)

        try:
            validation_error = self._validate_tool_request(request)
            if validation_error:
                raise ValueError(f"Tool request rejected: {validation_error}")

            task = asyncio.create_task(self._execute_tool_request(request))
            try:
                done, _ = await asyncio.wait(
                    {task},
                    timeout=float(request.timeout_seconds),
                )
            except asyncio.CancelledError:
                task.cancel()
                task.add_done_callback(_consume_task_exception)
                raise
            if not done:
                task.cancel()
                task.add_done_callback(_consume_task_exception)
                raise ClientExecutionTimeoutError(
                    f"Client runtime execution exceeded {request.timeout_seconds}s"
                )
            result = task.result()
            duration_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            payload = ToolDispatchResult(
                request_id=request.request_id,
                success=True,
                result=result,
                execution_time_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            error_context = self._build_runtime_error_context(
                exc,
                detail={
                    "request_id": request.request_id,
                    "tool_name": request.tool_name,
                    "qualified_tool_id": request.qualified_tool_id,
                },
            )
            payload = ToolDispatchResult(
                request_id=request.request_id,
                success=False,
                error=error_context.message,
                error_context=error_context,
                execution_time_ms=duration_ms,
            )

        await self._send_runtime_message(payload)

    async def _execute_tool_request(self, request: ToolDispatchRequest) -> Any:
        qualified_tool_id = request.qualified_tool_id
        arguments = request.arguments
        timeout_seconds = float(
            request.timeout_seconds or client_settings.tool_call_timeout_seconds
        )

        if qualified_tool_id == "client_skill::activate":
            return await self._execute_client_skill_request(arguments=arguments)

        if qualified_tool_id.startswith("skill::"):
            return await self._execute_skill_capability(
                qualified_tool_id, arguments, timeout_seconds, request.mutation_approved
            )

        return await get_mcp_manager().call_tool(
            qualified_tool_id=qualified_tool_id,
            arguments=arguments,
            timeout=timeout_seconds,
        )

    async def _execute_skill_capability(
        self,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout_seconds: float,
        mutation_approved: bool = False,
    ) -> Any:
        context = {
            "device_id": self._device_id,
            "session_id": self._session_id,
            "timeout_seconds": timeout_seconds,
            "mutation_approved": mutation_approved,
        }
        # The engine independently requires this exact approval bit before it
        # resolves secrets or starts the skill-owned process.
        engine = SkillExecutionEngine()
        envelope = await engine.execute(qualified_tool_id, arguments, context)

        if not envelope.get("ok"):
            error = envelope.get("error") or {}
            raise SkillRuntimeError(
                error.get("code") or "RUNTIME_ERROR",
                error.get("message") or "skill execution failed",
                error.get("repair"),
            )

        return envelope.get("result")

    async def _execute_client_skill_request(
        self,
        *,
        arguments: dict[str, Any],
    ) -> Any:
        skill_name = str(arguments.get("skill_name") or "").strip()
        if not skill_name:
            raise ValueError("skill_name is required")

        skill = get_skills_registry().get_skill(skill_name)
        if skill is None:
            raise ValueError(f"Skill '{skill_name}' not found on this device.")
        if not skill.enabled:
            raise ValueError(f"Skill '{skill_name}' is disabled on this device.")

        readiness = SkillRuntimeManager().evaluate_readiness(skill)
        if readiness.status == "ready":
            runtime_footer = (
                f"Runtime status: ready\n"
                f"Command binding: `skill::{skill.name}::run_skill_command`.\n"
                "The activation service appends the exact model-callable tool name; "
                "use that exposed name with an argv array, not this internal binding id.\n"
                "Do not use Desktop Commander or search for another shell executor "
                "for this skill's commands."
            )
        elif readiness.status == "not_ready":
            runtime_footer = (
                f"Runtime status: {readiness.setup_status}\n"
                f"Repair hints: {json.dumps(readiness.repair_hints)}\n"
                "Do not guess npx, pip, or another package-manager command. "
                "Ask the user to approve this skill's local setup."
            )
        else:
            runtime_footer = (
                "Runtime status: instruction_only\n"
                "This skill has no bundled command. Follow its instructions without "
                "inventing an executable."
            )

        return (
            f"── Skill: {skill.name} ──\n\n{skill.content}\n\n"
            f"── Device-local skill runtime ──\n{runtime_footer}\n\n"
            f"── End Skill: {skill.name} ──"
        )

    async def _build_tool_catalog(self) -> dict[str, Any]:
        mcp_catalog = get_mcp_manager().get_tool_catalog()
        raw_tools = mcp_catalog.get("tools", []) if isinstance(mcp_catalog, dict) else []
        tools = [dict(entry) for entry in raw_tools if isinstance(entry, dict)]

        tools.extend(self._collect_skill_capability_tools())

        # Embed tool_instance_id in each entry using the NEXT catalog_version
        # (server increments catalog_version on receive, so use version+1)
        device_id = self._device_id or ""
        session_id = self._session_id or ""
        next_catalog_version = self._tool_catalog_version + 1
        for entry in tools:
            qid = entry.get("qualified_id", "")
            if qid:
                entry["tool_instance_id"] = _make_tool_instance_id(
                    device_id,
                    session_id,
                    qid,
                    next_catalog_version,
                    str(entry.get("source_hash") or ""),
                )

        return {
            "tools": tools,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tool_count": len(tools),
            "mcp_server_count": mcp_catalog.get("server_count", 0),
            "active_servers": mcp_catalog.get("active_servers", []),
        }

    @staticmethod
    def _collect_skill_capability_tools() -> list[dict[str, Any]]:
        """Build the single fixed command entry for every ready skill.

        A skill-runtime hiccup must never break MCP tool catalog sync.
        """
        try:
            skill_manager = SkillRuntimeManager()
            skill_tools: list[dict[str, Any]] = []
            for skill in get_skills_registry().get_enabled_skills():
                readiness = skill_manager.evaluate_readiness(skill)
                if readiness.status != "ready":
                    continue
                skill_tools.extend(skill_manager.capability_catalog_entries(skill, readiness))
            return skill_tools
        except Exception:
            # Swallow-and-continue so a skill-runtime hiccup never breaks MCP
            # tool sync; log with a traceback so a real bug here is still
            # debuggable rather than reduced to a one-line message.
            logger.exception("Failed to collect skill capability tools for catalog sync")
            return []

    async def _send_runtime_message(self, message: RuntimeMessage) -> None:
        await self._send_json(dump_runtime_message(message))

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._websocket is None:
            raise RuntimeError("Runtime WebSocket is not connected")

        async with self._send_lock:
            await self._websocket.send(json.dumps(payload))

    @staticmethod
    def _decode_message(raw_message: Any) -> dict[str, Any]:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        if isinstance(raw_message, str):
            return json.loads(raw_message)
        if isinstance(raw_message, dict):
            return raw_message
        raise ValueError(f"Unsupported WebSocket message payload: {type(raw_message)!r}")

    @classmethod
    def _decode_runtime_message(cls, raw_message: Any) -> RuntimeMessage:
        return parse_runtime_message(cls._decode_message(raw_message))


_runtime_bridge: RuntimeBridgeService | None = None


def get_runtime_bridge() -> RuntimeBridgeService:
    """Get the global runtime bridge singleton."""
    global _runtime_bridge
    if _runtime_bridge is None:
        _runtime_bridge = RuntimeBridgeService()
    return _runtime_bridge
