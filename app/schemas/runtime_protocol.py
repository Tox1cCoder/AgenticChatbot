"""Shared runtime WebSocket protocol models used by server and client runtimes."""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter

RUNTIME_MESSAGE_TOOL_REQUEST = "tool_request"
RUNTIME_MESSAGE_TOOL_RESULT = "tool_result"
RUNTIME_MESSAGE_HEARTBEAT = "heartbeat"
RUNTIME_MESSAGE_ERROR = "error"
RUNTIME_MESSAGE_ACK = "ack"


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
    timeout_seconds: int = 30
    # Execution-scope validation fields - sidecar must reject if they mismatch.
    tool_instance_id: str | None = None
    expected_session_id: str | None = None
    expected_catalog_version: int | None = None
    # Set by the server for a dispatched skill mutation that has cleared the
    # HITL gate (approved, or pre-approved by policy). The sidecar allows the
    # mutation only when this is True, and still re-validates session/catalog/
    # tool_instance regardless of this flag.
    mutation_approved: bool = False


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


RuntimeMessage: TypeAlias = Annotated[
    ToolDispatchRequest
    | ToolDispatchResult
    | RuntimeHeartbeatMessage
    | RuntimeErrorMessage
    | RuntimeAckMessage,
    Field(discriminator="type"),
]

_RUNTIME_MESSAGE_ADAPTER = TypeAdapter(RuntimeMessage)


def parse_runtime_message(payload: Any) -> RuntimeMessage:
    """Validate one runtime WebSocket payload into the canonical message union."""

    return _RUNTIME_MESSAGE_ADAPTER.validate_python(payload)


def dump_runtime_message(message: RuntimeMessage) -> dict[str, Any]:
    """Serialize a validated runtime WebSocket model into a JSON-safe payload."""

    return message.model_dump(mode="json")
