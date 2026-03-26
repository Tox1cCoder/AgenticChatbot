"""
Client runtime bridge service.

Maintains the authenticated outbound runtime connection to the canonical server:
- registers the current device
- opens the persistent WebSocket
- syncs tool and skill catalogs
- executes client-local tool requests
"""

from __future__ import annotations

import asyncio
import json
import platform
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from client_backend import __version__
from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.security import generate_device_identifier
from client_backend.schemas.runtime import DeviceInfo, RuntimeState, RuntimeStatus
from client_backend.services.filesystem_service import get_filesystem_service
from client_backend.services.local_mcp_manager import get_mcp_manager, shutdown_mcp_manager
from client_backend.services.local_skills_registry import (
    get_skills_registry,
    initialize_skills_registry,
)
from client_backend.services.server_api import ServerAPIClient, get_server_client
from client_backend.services.shell_runner import get_shell_runner

logger = get_logger(__name__)


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

        await self._server_client.update_device_tool_catalog(
            device_id=self._device_id,
            catalog=tool_catalog,
        )
        await self._server_client.update_device_skill_catalog(
            device_id=self._device_id,
            catalog=skill_catalog,
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

    async def _run_forever(self) -> None:
        attempts = 0

        while not self._stop_requested:
            try:
                await self._initialize_local_runtime()
                registration = await self._register_device()
                self._device_id = registration["device_id"]
                self._session_id = registration["session_id"]

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

    async def _register_device(self) -> dict[str, Any]:
        capabilities = {
            "native_tools": True,
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
            message = self._decode_message(raw_message)
            if message.get("type") != "ack":
                raise RuntimeError(f"Unexpected runtime handshake message: {message}")

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
            await self._send_json({"type": "heartbeat"})
            self._set_state(last_heartbeat=datetime.now(timezone.utc))

    async def _receive_loop(self) -> None:
        while not self._stop_requested and self._websocket is not None:
            raw_message = await self._websocket.recv()
            message = self._decode_message(raw_message)
            await self._handle_server_message(message)

    async def _handle_server_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")

        if message_type == "tool_request":
            await self._handle_tool_request(message)
            return

        if message_type == "ack":
            return

        if message_type == "error":
            logger.error("Runtime bridge received server error: %s", message.get("message"))
            return

        logger.warning("Runtime bridge received unsupported message: %s", message_type)

    async def _handle_tool_request(self, message: dict[str, Any]) -> None:
        request_id = str(message.get("request_id") or "")
        started_at = datetime.now(timezone.utc)

        try:
            result = await self._execute_tool_request(message)
            duration_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            payload = {
                "type": "tool_result",
                "request_id": request_id,
                "success": True,
                "result": result,
                "execution_time_ms": duration_ms,
            }
        except Exception as exc:
            duration_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
            payload = {
                "type": "tool_result",
                "request_id": request_id,
                "success": False,
                "error": str(exc),
                "execution_time_ms": duration_ms,
            }

        await self._send_json(payload)

    async def _execute_tool_request(self, message: dict[str, Any]) -> Any:
        qualified_tool_id = str(message.get("qualified_tool_id") or "")
        arguments = message.get("arguments") or {}
        timeout_seconds = int(
            message.get("timeout_seconds") or client_settings.shell_timeout_seconds
        )

        if qualified_tool_id.startswith("native::"):
            return await self._execute_native_tool(
                qualified_tool_id=qualified_tool_id,
                arguments=arguments,
                timeout_seconds=timeout_seconds,
            )

        return await get_mcp_manager().call_tool(
            qualified_tool_id=qualified_tool_id,
            arguments=arguments,
            timeout=timeout_seconds,
        )

    async def _execute_native_tool(
        self,
        *,
        qualified_tool_id: str,
        arguments: dict[str, Any],
        timeout_seconds: int,
    ) -> Any:
        filesystem = get_filesystem_service()
        shell_runner = get_shell_runner()

        if qualified_tool_id == "native::shell_execute":
            command = str(arguments.get("command") or "").strip()
            working_dir = arguments.get("working_dir")
            is_valid, reason = shell_runner.validate_command(
                command,
                working_dir=working_dir,
            )
            if not is_valid:
                raise ValueError(reason)

            result = await shell_runner.execute(
                command=command,
                working_dir=working_dir,
                timeout=int(arguments.get("timeout") or timeout_seconds),
                shell=str(arguments.get("shell") or self._default_shell()),
                env=arguments.get("env"),
            )
            return result.to_dict()

        if qualified_tool_id == "native::activate_skill":
            skill_name = str(arguments.get("skill_name") or "").strip()
            if not skill_name:
                raise ValueError("skill_name is required")

            skill = get_skills_registry().get_skill(skill_name)
            if skill is None:
                raise ValueError(f"Skill '{skill_name}' not found on this device.")
            if not skill.enabled:
                raise ValueError(f"Skill '{skill_name}' is disabled on this device.")

            return f"── Skill: {skill.name} ──\n\n{skill.content}\n\n── End Skill: {skill.name} ──"

        if qualified_tool_id == "native::filesystem_read_text":
            content = await filesystem.read_file(
                arguments["file_path"],
                encoding=str(arguments.get("encoding") or "utf-8"),
                max_size=arguments.get("max_size"),
            )
            return {"content": content}

        if qualified_tool_id == "native::filesystem_write_text":
            info = await filesystem.write_file(
                arguments["file_path"],
                content=str(arguments.get("content") or ""),
                encoding=str(arguments.get("encoding") or "utf-8"),
                create_dirs=bool(arguments.get("create_dirs", True)),
            )
            return info.to_dict()

        if qualified_tool_id == "native::filesystem_list_directory":
            entries = await filesystem.list_directory(
                arguments["dir_path"],
                recursive=bool(arguments.get("recursive", False)),
                pattern=arguments.get("pattern"),
            )
            return [entry.to_dict() for entry in entries]

        if qualified_tool_id == "native::filesystem_search_files":
            matches = await filesystem.search_files(
                arguments["root_path"],
                pattern=str(arguments.get("pattern") or "*"),
                content_pattern=arguments.get("content_pattern"),
                max_results=int(arguments.get("max_results") or 100),
            )
            return [
                {
                    "file": file_info.to_dict(),
                    "matching_lines": matching_lines or [],
                }
                for file_info, matching_lines in matches
            ]

        raise ValueError(f"Unsupported native tool: {qualified_tool_id}")

    @staticmethod
    def _default_shell() -> str:
        preferred = ["powershell", "bash", "sh", "cmd", "zsh"]
        for shell_name in preferred:
            if shell_name in client_settings.allowed_shells:
                return shell_name
        return client_settings.allowed_shells[0]

    async def _build_tool_catalog(self) -> dict[str, Any]:
        native_tools = self._build_native_tool_catalog()
        mcp_catalog = get_mcp_manager().get_tool_catalog()
        tools = list(native_tools)
        tools.extend(mcp_catalog.get("tools", []))

        return {
            "tools": tools,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "native_tool_count": len(native_tools),
            "mcp_server_count": mcp_catalog.get("server_count", 0),
            "active_servers": mcp_catalog.get("active_servers", []),
        }

    def _build_native_tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "shell_execute",
                "description": (
                    "Run a shell command on the local device. "
                    "Provide a working directory only when the command specifically needs one."
                ),
                "origin": "native",
                "qualified_id": "native::shell_execute",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "working_dir": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "shell": {"type": "string"},
                        "env": {"type": "object"},
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "filesystem_read_text",
                "description": "Read a UTF-8 text file from the local device.",
                "origin": "native",
                "qualified_id": "native::filesystem_read_text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "encoding": {"type": "string"},
                        "max_size": {"type": "integer"},
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "filesystem_write_text",
                "description": "Write a UTF-8 text file on the local device.",
                "origin": "native",
                "qualified_id": "native::filesystem_write_text",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                        "encoding": {"type": "string"},
                        "create_dirs": {"type": "boolean"},
                    },
                    "required": ["file_path", "content"],
                },
            },
            {
                "name": "filesystem_list_directory",
                "description": "List files and directories on the local device.",
                "origin": "native",
                "qualified_id": "native::filesystem_list_directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dir_path": {"type": "string"},
                        "recursive": {"type": "boolean"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["dir_path"],
                },
            },
            {
                "name": "filesystem_search_files",
                "description": "Search filenames and optional file contents on the local device.",
                "origin": "native",
                "qualified_id": "native::filesystem_search_files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "root_path": {"type": "string"},
                        "pattern": {"type": "string"},
                        "content_pattern": {"type": "string"},
                        "max_results": {"type": "integer"},
                    },
                    "required": ["root_path", "pattern"],
                },
            },
        ]

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


_runtime_bridge: RuntimeBridgeService | None = None


def get_runtime_bridge() -> RuntimeBridgeService:
    """Get the global runtime bridge singleton."""
    global _runtime_bridge
    if _runtime_bridge is None:
        _runtime_bridge = RuntimeBridgeService()
    return _runtime_bridge
