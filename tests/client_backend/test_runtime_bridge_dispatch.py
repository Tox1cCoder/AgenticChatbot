"""How the sidecar runs the tool requests the server sends it.

Requests arriving together run together, each within a budget that starts
when it arrives -- waiting for a free slot spends the same clock the server is
watching. A request the server cancels never starts if it has not started yet.
Every failure that happens around the tool, rather than inside it, carries a
code saying whether the tool could have run.
"""

from __future__ import annotations

import asyncio
import json

from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, ErrorData
from websockets.asyncio.server import serve

from app.schemas.runtime_protocol import RuntimeCancelMessage
from client_backend.schemas.runtime import ToolDispatchRequest
from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService


class _ServerClientStub:
    base_url = "http://server.test"

    def is_authenticated(self) -> bool:
        return True


class _ScriptedWebSocket:
    """Delivers queued server messages; records what the sidecar sends."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict] = []

    async def recv(self) -> str:
        return await self.incoming.get()

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _request(request_id: str, *, timeout: float = 5.0, **fields) -> ToolDispatchRequest:
    return ToolDispatchRequest(
        request_id=request_id,
        tool_name="run",
        qualified_tool_id="demo::run",
        arguments=fields.pop("arguments", {}),
        timeout_seconds=timeout,
        **fields,
    )


def _bridge(monkeypatch, *, slots: int = 4) -> RuntimeBridgeService:
    monkeypatch.setattr(runtime_bridge_module.client_settings, "max_concurrent_tool_calls", slots)
    bridge = RuntimeBridgeService(server_client=_ServerClientStub())
    bridge._current_tool_catalog = {"demo::run": {"qualified_id": "demo::run", "name": "run"}}
    return bridge


async def _serve_scripted(bridge: RuntimeBridgeService) -> tuple[_ScriptedWebSocket, asyncio.Task]:
    websocket = _ScriptedWebSocket()
    bridge._websocket = websocket
    return websocket, asyncio.create_task(bridge._receive_loop())


async def _stop(loop_task: asyncio.Task) -> None:
    loop_task.cancel()
    await asyncio.gather(loop_task, return_exceptions=True)


def _results(websocket: _ScriptedWebSocket) -> dict[str, dict]:
    return {message["request_id"]: message for message in websocket.sent}


async def _until(condition, seconds: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def test_requests_arriving_together_run_at_the_same_time(monkeypatch):
    bridge = _bridge(monkeypatch)
    started: dict[str, float] = {}
    loop = asyncio.get_running_loop()

    async def execute(request):
        started[request.request_id] = loop.time()
        await asyncio.sleep(0.3)
        return "done"

    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    websocket, loop_task = await _serve_scripted(bridge)
    for request_id in ("first", "second"):
        await websocket.incoming.put(_request(request_id).model_dump_json())

    await _until(lambda: len(websocket.sent) == 2)
    await _stop(loop_task)

    assert abs(started["first"] - started["second"]) < 0.1


async def test_request_that_cannot_start_within_its_budget_never_runs(monkeypatch):
    bridge = _bridge(monkeypatch, slots=1)
    executed: list[str] = []
    release = asyncio.Event()

    async def execute(request):
        executed.append(request.request_id)
        await release.wait()
        return "done"

    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    websocket, loop_task = await _serve_scripted(bridge)
    await websocket.incoming.put(_request("holder").model_dump_json())
    await websocket.incoming.put(_request("late", timeout=0.2).model_dump_json())

    await _until(lambda: "late" in _results(websocket))
    release.set()
    await _until(lambda: "holder" in _results(websocket))
    await _stop(loop_task)

    assert executed == ["holder"]
    assert _results(websocket)["late"]["error_context"]["code"] == "TIMEOUT_NOT_STARTED"


async def test_cancelled_request_that_has_not_started_never_runs(monkeypatch):
    bridge = _bridge(monkeypatch, slots=1)
    executed: list[str] = []
    release = asyncio.Event()

    async def execute(request):
        executed.append(request.request_id)
        await release.wait()
        return "done"

    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    websocket, loop_task = await _serve_scripted(bridge)
    await websocket.incoming.put(_request("holder").model_dump_json())
    await websocket.incoming.put(_request("queued").model_dump_json())
    await _until(lambda: executed == ["holder"])

    await websocket.incoming.put(RuntimeCancelMessage(request_id="queued").model_dump_json())
    await asyncio.sleep(0.05)
    release.set()
    await _until(lambda: "holder" in _results(websocket))
    await asyncio.sleep(0.05)
    await _stop(loop_task)

    assert executed == ["holder"]
    assert "queued" not in _results(websocket)


async def test_cancel_stops_a_running_request(monkeypatch):
    bridge = _bridge(monkeypatch)
    stopped = asyncio.Event()

    async def execute(_request):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.set()
            raise

    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    websocket, loop_task = await _serve_scripted(bridge)
    await websocket.incoming.put(_request("running").model_dump_json())
    await asyncio.sleep(0.05)

    await websocket.incoming.put(RuntimeCancelMessage(request_id="running").model_dump_json())
    await asyncio.wait_for(stopped.wait(), timeout=2)
    await _stop(loop_task)

    assert websocket.sent == []


async def test_rejected_request_reports_that_it_never_ran(monkeypatch):
    bridge = _bridge(monkeypatch)
    sent = []

    async def capture(payload):
        sent.append(payload)

    monkeypatch.setattr(bridge, "_send_runtime_message", capture)

    await bridge._handle_tool_request(
        ToolDispatchRequest(
            request_id="stale",
            tool_name="gone",
            qualified_tool_id="demo::gone",
            arguments={},
        )
    )

    assert sent[0].error_context.code == "SESSION_REQUEST_REJECTED"


async def test_tool_server_lost_mid_call_is_reported_as_possibly_run(monkeypatch):
    bridge = _bridge(monkeypatch)
    sent = []

    async def execute(_request):
        raise McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))

    async def capture(payload):
        sent.append(payload)

    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    monkeypatch.setattr(bridge, "_send_runtime_message", capture)

    await bridge._handle_tool_request(_request("lost"))

    assert sent[0].error_context.code == "TOOL_CONNECTION_LOST"


async def test_refused_credential_path_is_reported_as_a_permission_denial(monkeypatch):
    from client_backend.services.desktop_commander_policy import SensitivePathError

    bridge = _bridge(monkeypatch)
    sent = []

    async def execute(_request):
        raise SensitivePathError("Desktop Commander may not use '~/.ssh/config'")

    async def capture(payload):
        sent.append(payload)

    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    monkeypatch.setattr(bridge, "_send_runtime_message", capture)

    await bridge._handle_tool_request(_request("secret"))

    # PERMISSION_ codes classify as permission errors on the server: not
    # retryable, with a hint to ask the user rather than try again.
    assert sent[0].error_context.code == "PERMISSION_SENSITIVE_PATH"


async def test_oversized_result_is_capped_and_marked_truncated(monkeypatch):
    bridge = _bridge(monkeypatch)
    sent = []

    async def execute(_request):
        return [{"type": "text", "text": "x" * 50_000}]

    async def capture(payload):
        sent.append(payload)

    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    monkeypatch.setattr(bridge, "_send_runtime_message", capture)

    await bridge._handle_tool_request(_request("big", max_result_text_bytes=2_000))

    assert sent[0].success is True
    assert sent[0].truncated is True
    assert len(json.dumps(sent[0].result)) <= 2_000


async def _serve_bridge(monkeypatch, handler, execute):
    """Run the bridge's real WebSocket client against a local server."""

    server = await serve(handler, "127.0.0.1", 0, max_size=None)
    port = server.sockets[0].getsockname()[1]
    bridge = RuntimeBridgeService(
        server_client=_ServerClientStub(),
        websocket_base_url=f"ws://127.0.0.1:{port}",
    )
    bridge._device_id, bridge._session_id = "device-1", "session-1"
    bridge._current_tool_catalog = {"demo::run": {"qualified_id": "demo::run", "name": "run"}}

    async def catalogs_synced():
        return None

    monkeypatch.setattr(bridge, "_sync_initial_catalogs_and_mark_ready", catalogs_synced)
    monkeypatch.setattr(bridge, "_execute_tool_request", execute)
    return server, asyncio.create_task(bridge._connect_and_serve())


async def test_bridge_accepts_a_tool_request_over_one_mebibyte(monkeypatch):
    content = "x" * (2 * 1024 * 1024)
    replies: list[dict] = []

    async def handler(websocket):
        await websocket.send(json.dumps({"type": "ack"}))
        await websocket.send(_request("big", arguments={"content": content}).model_dump_json())
        replies.append(json.loads(await websocket.recv()))
        await websocket.wait_closed()

    async def execute(request):
        return len(request.arguments["content"])

    server, serving = await _serve_bridge(monkeypatch, handler, execute)
    try:
        await _until(lambda: replies)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        server.close()

    assert replies[0]["success"] is True
    assert replies[0]["result"] == len(content)


async def test_connection_loss_stops_requests_still_running(monkeypatch):
    stopped = asyncio.Event()

    async def handler(websocket):
        await websocket.send(json.dumps({"type": "ack"}))
        await websocket.send(_request("orphan").model_dump_json())
        await asyncio.sleep(0.1)
        await websocket.close()

    async def execute(_request):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            stopped.set()
            raise

    server, serving = await _serve_bridge(monkeypatch, handler, execute)
    try:
        await asyncio.wait_for(stopped.wait(), timeout=5)
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        server.close()
