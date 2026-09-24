"""Shared runtime WebSocket protocol models used by server and client runtimes."""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter

RUNTIME_MESSAGE_TOOL_REQUEST = "tool_request"
RUNTIME_MESSAGE_TOOL_RESULT = "tool_result"
RUNTIME_MESSAGE_HEARTBEAT = "heartbeat"
RUNTIME_MESSAGE_ERROR = "error"
RUNTIME_MESSAGE_ACK = "ack"
RUNTIME_MESSAGE_CANCEL = "cancel"

# Error codes for failures around a tool rather than inside it. They tell the
# server whether the tool could have run: a mutation that might have run must
# not be repeated blindly.
RUNTIME_ERROR_REQUEST_REJECTED = "SESSION_REQUEST_REJECTED"  # never ran
RUNTIME_ERROR_NOT_STARTED = "TIMEOUT_NOT_STARTED"  # never ran
RUNTIME_ERROR_EXECUTION_TIMEOUT = "TIMEOUT_CLIENT_EXECUTION"  # may have run
RUNTIME_ERROR_TOOL_CONNECTION_LOST = "TOOL_CONNECTION_LOST"  # may have run
RUNTIME_ERROR_DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"  # may have run
RUNTIME_ERROR_SENSITIVE_PATH = "PERMISSION_SENSITIVE_PATH"  # never ran

# Largest single WebSocket message either side accepts. It matches uvicorn's
# default ``ws_max_size`` on the server; the sidecar sets the same limit on
# its client, whose library default (1 MiB) would otherwise drop the whole
# connection on one large tool argument.
RUNTIME_MAX_MESSAGE_BYTES = 16 * 1024 * 1024

# Result budgets a sidecar applies when a request does not carry its own:
# text and structured content, and the decoded size of image and audio data.
DEFAULT_MAX_RESULT_TEXT_BYTES = 1024 * 1024
DEFAULT_MAX_RESULT_MEDIA_BYTES = 5 * 1024 * 1024


class RuntimeErrorContext(BaseModel):
    """Structured runtime error details preserved across the bridge."""

    message: str
    code: str | None = None
    detail: Any = None


class ToolDispatchRequest(BaseModel):
    """Canonical runtime tool dispatch request payload."""

    type: Literal["tool_request"] = RUNTIME_MESSAGE_TOOL_REQUEST
    request_id: str
    tool_name: str
    qualified_tool_id: str
    arguments: dict[str, Any]
    timeout_seconds: float = Field(default=30.0, gt=0)
    # Execution-scope validation fields - sidecar must reject if they mismatch.
    tool_instance_id: str | None = None
    expected_session_id: str | None = None
    expected_catalog_version: int | None = None
    # Server-set marker that a dispatched capability is a mutation the server
    # permitted to reach execution. It mirrors the capability's own mutation
    # flag: the server's HITL gate interrupts BEFORE the tools node, so a
    # mutation only reaches dispatch once it was approved by a human OR
    # pre-approved by policy OR the HITL master switch is off. The sidecar
    # therefore trusts this as "the server allowed this mutation" (the model
    # cannot forge it — it travels the authenticated server->sidecar channel),
    # NOT as independent proof a human clicked approve. The sidecar runs a
    # mutation only when this is True and STILL re-validates session/catalog/
    # tool_instance regardless of this flag.
    mutation_approved: bool = False
    # Budgets the sidecar caps the result to before replying (see
    # ``shared.runtime_results``). The server sends its configured values.
    max_result_text_bytes: int = Field(default=DEFAULT_MAX_RESULT_TEXT_BYTES, gt=0)
    max_result_media_bytes: int = Field(default=DEFAULT_MAX_RESULT_MEDIA_BYTES, ge=0)


class ToolDispatchResult(BaseModel):
    """Canonical runtime tool dispatch result payload."""

    type: Literal["tool_result"] = RUNTIME_MESSAGE_TOOL_RESULT
    request_id: str
    success: bool
    result: Any = None
    error: str | None = None
    error_context: RuntimeErrorContext | None = None
    execution_time_ms: int
    truncated: bool = False


class RuntimeHeartbeatMessage(BaseModel):
    """Heartbeat payload sent across the runtime bridge."""

    type: Literal["heartbeat"] = RUNTIME_MESSAGE_HEARTBEAT


class RuntimeErrorMessage(BaseModel):
    """Generic runtime error payload sent across the runtime bridge."""

    type: Literal["error"] = RUNTIME_MESSAGE_ERROR
    message: str
    code: str | None = None
    error_context: RuntimeErrorContext | None = None


class RuntimeAckMessage(BaseModel):
    """Acknowledgement payload sent across the runtime bridge."""

    type: Literal["ack"] = RUNTIME_MESSAGE_ACK
    message_id: str | None = None


class RuntimeCancelMessage(BaseModel):
    """Server to sidecar: stop a request the server no longer waits for.

    Sent when the server times out or the user stops the turn. A request that
    has not started never starts. One that is running is cancelled, which ends
    a skill command's process but only stops waiting on an MCP tool: MCP gives
    the sidecar no way to stop work a server has already begun.
    """

    type: Literal["cancel"] = RUNTIME_MESSAGE_CANCEL
    request_id: str


RuntimeMessage: TypeAlias = Annotated[
    ToolDispatchRequest
    | ToolDispatchResult
    | RuntimeHeartbeatMessage
    | RuntimeErrorMessage
    | RuntimeAckMessage
    | RuntimeCancelMessage,
    Field(discriminator="type"),
]

_RUNTIME_MESSAGE_ADAPTER = TypeAdapter(RuntimeMessage)


def parse_runtime_message(payload: Any) -> RuntimeMessage:
    """Validate one runtime WebSocket payload into the canonical message union."""

    return _RUNTIME_MESSAGE_ADAPTER.validate_python(payload)


def dump_runtime_message(message: RuntimeMessage) -> dict[str, Any]:
    """Serialize a validated runtime WebSocket model into a JSON-safe payload."""

    return message.model_dump(mode="json")
