from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import PlanLifecycle


class InterruptDecisionType(str, Enum):
    """Service-facing decision types for handling a tool interrupt."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    RESPOND = "respond"


class ToolInterruptRequest(BaseModel):
    """Request details for a tool awaiting human approval."""

    action: str = Field(..., description="Tool name")
    args: dict[str, Any] = Field(..., description="Tool arguments")
    description: str | None = Field(None, description="Description of what the tool will do")
    task_id: str | None = Field(None, description="Task identifier from interrupt mechanism")
    tool_call_id: str | None = Field(None, description="Tool call identifier for resume mapping")
    allowed_decisions: list[str] | None = Field(
        None,
        description="Which decision types are permitted for this tool",
    )


class InterruptDecision(BaseModel):
    """Service-facing decision for handling a tool interrupt."""

    type: InterruptDecisionType = Field(..., description="Type of decision")
    task_id: str | None = Field(None, description="Task ID to apply this decision to")
    tool_call_id: str | None = Field(
        None,
        description="Tool call ID to apply this decision to when the client uses toolCallId.",
    )
    action: str | None = Field(None, description="Tool/action this decision applies to")
    args: dict[str, Any] | None = Field(
        None,
        description="For EDIT: modified arguments. For REJECT: optional feedback message",
    )


class InterruptResponse(BaseModel):
    """Service/API boundary payload for a paused workflow interrupt."""

    interrupt_id: str = Field(..., description="Unique identifier for this interrupt")
    action_requests: list[ToolInterruptRequest] = Field(
        ...,
        description="List of tools awaiting approval",
    )
    thread_id: str = Field(..., description="Conversation thread ID for resuming")
    conversation_id: str = Field(..., description="Conversation ID")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the interrupt",
    )


class WorkflowPlanningContext(BaseModel):
    """Service-owned planning context passed into AI workflow execution."""

    planning_mode_enabled: bool = False
    has_existing_plan: bool = False
    current_task: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    plan_lifecycle: PlanLifecycle | None = None


class WorkflowExecutionRequest(BaseModel):
    """Service-owned workflow execution request."""

    message: str
    conversation_id: str | None = None
    user_id: str | None = None
    device_id: str | None = None
    thread_id: str | None = None
    persona: str | None = None
    attachments: list[Any] | None = None
    model_request: dict[str, Any] | None = None
    planning: WorkflowPlanningContext = Field(default_factory=WorkflowPlanningContext)
    # Stable database message identifiers that flow through the graph so the
    # current user turn can be excluded by ID rather than by tail position,
    # and the final assistant reply can be persisted with a known ID.
    user_message_id: str | None = Field(
        default=None,
        description="DB id of the persisted user message that triggered this execution",
    )
    assistant_message_id: str | None = Field(
        default=None,
        description="Reserved DB id for the assistant message this execution will produce",
    )
    inline_rich_response_v1: bool = Field(
        default=False,
        description=(
            "Capability flag from the API boundary: when true, the client opted "
            "into the inline rich-response v1 contract and the workflow may "
            "surface marker syntax / rich-item inventory to agents."
        ),
    )


class WorkflowResponseMessage(BaseModel):
    """Service-owned assistant message payload returned from the workflow boundary."""

    role: str = Field(default="assistant", description="Workflow message role")
    content: str = Field(default="", description="Rendered assistant content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-message metadata from the workflow runtime",
    )


class WorkflowResponse(BaseModel):
    """Service-owned workflow execution response."""

    agent_type: str = Field(default="chat", description="Agent type that produced the response")
    agent_id: str = Field(default="chat_agent", description="Agent identifier")
    message: WorkflowResponseMessage = Field(default_factory=WorkflowResponseMessage)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow execution metadata",
    )
    tool_artifacts: list[dict[str, Any]] | None = Field(
        default=None,
        description="Artifacts produced while executing tools",
    )
    error: str | None = Field(default=None, description="Workflow execution error")
    suggested_questions: list[str] | None = Field(
        default=None,
        description="Optional follow-up questions surfaced by the workflow",
    )
