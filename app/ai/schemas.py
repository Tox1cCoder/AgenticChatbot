from enum import Enum
from typing import Annotated, Any, NotRequired, TypedDict, cast

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.models.enums import PlanLifecycle


class AgentType(str, Enum):
    CHAT = "chat"
    RAG = "rag"
    SEARCH = "search"
    IMAGE_GENERATOR = "image_generator"
    PLANNING = "planning"
    CANVAS = "canvas"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class InterruptDecisionType(str, Enum):
    """Canonical decision types for handling a tool interrupt."""

    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"


class ToolInterruptRequest(BaseModel):
    """Request details for a tool awaiting human approval."""

    action: str = Field(..., description="Tool name")
    args: dict[str, Any] = Field(..., description="Tool arguments")
    description: str | None = Field(None, description="Description of what the tool will do")
    task_id: str | None = Field(None, description="Task identifier from interrupt mechanism")
    tool_call_id: str | None = Field(None, description="Tool call identifier for resume mapping")
    allowed_decisions: list[str] | None = Field(
        None, description="Which decision types are permitted for this tool"
    )


class InterruptDecision(BaseModel):
    """Decision for handling a tool interrupt."""

    type: InterruptDecisionType = Field(..., description="Type of decision")
    task_id: str | None = Field(None, description="Task ID to apply this decision to")
    tool_call_id: str | None = Field(
        None,
        description="Tool call ID to apply this decision to when the client uses toolCallId.",
    )
    action: str | None = Field(None, description="Name of the tool/action this decision applies to")
    args: dict[str, Any] | None = Field(
        None,
        description="For EDIT: modified arguments. For REJECT: optional feedback message",
    )


class InterruptResponse(BaseModel):
    """Response containing interrupt information for human approval."""

    interrupt_id: str = Field(..., description="Unique identifier for this interrupt")
    action_requests: list[ToolInterruptRequest] = Field(
        ..., description="List of tools awaiting approval"
    )
    thread_id: str = Field(..., description="Conversation thread ID for resuming")
    conversation_id: str = Field(..., description="Conversation ID")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the interrupt"
    )


class AgentMessage(BaseModel):
    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    attachments: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional image attachments with structure {name: str, mime: str, data: str (base64)}",
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of tool calls to be executed",
    )


class AgentConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: int | None = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0


class AgentResponse(BaseModel):
    agent_type: AgentType
    agent_id: str
    message: AgentMessage
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_artifacts: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of artifacts produced while executing tools",
    )
    error: str | None = None
    suggested_questions: list[str] | None = Field(
        default=None,
        description="0-3 follow-up question suggestions for continuing the conversation",
    )


class WorkflowPlanningContext(BaseModel):
    planning_mode_enabled: bool = False
    has_existing_plan: bool = False
    current_task: dict[str, Any] | None = None
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    plan_lifecycle: PlanLifecycle | None = None


class WorkflowExecutionRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    user_id: str | None = None
    device_id: str | None = None
    thread_id: str | None = None
    persona: str | None = None
    attachments: list[Any] | None = None
    model_request: dict[str, Any] | None = None
    planning: WorkflowPlanningContext = Field(default_factory=WorkflowPlanningContext)
    user_message_id: str | None = None
    assistant_message_id: str | None = None


class ContinuationSignal(TypedDict, total=False):
    should_continue: bool
    reason: str
    scope: str
    count: int
    limit: int


class GraphContext(TypedDict, total=False):
    attachments: list[Any]
    planning_mode_enabled: bool
    has_existing_plan: bool
    tool_artifacts: list[dict[str, Any]]
    tool_images: list[dict[str, Any]]
    pending_action_requests: list[dict[str, Any]]
    interrupt_metadata: dict[str, Any]
    generate_plan_response: bool
    generate_final_summary: bool
    final_summary_generated: bool
    plan_just_modified: bool
    continuation_signal: ContinuationSignal
    pause_reason: str
    continuation_round: int
    continuation_reason: str
    agentic_rag_iteration: int
    consecutive_errors: int
    all_tasks_completed: bool
    conversation_summarized: bool
    tool_provenance: dict[str, dict[str, Any]]
    force_final_response: bool
    tool_budget: dict[str, Any]


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    conversation_id: NotRequired[str | None]
    user_id: NotRequired[str | None]
    device_id: NotRequired[str | None]
    selected_agent: NotRequired[str | None]
    response: NotRequired[AgentResponse | None]
    context: NotRequired[GraphContext]
    persona: NotRequired[str | None]
    attachments: NotRequired[list[Any] | None]
    iteration_count: NotRequired[int | None]
    pending_tool_calls: NotRequired[list[Any] | None]
    # Multi-provider model configuration
    model_request: NotRequired[dict[str, Any] | None]
    # Task planning context fields
    task_plan_id: NotRequired[str | None]
    current_task: NotRequired[dict[str, Any] | None]
    all_tasks: NotRequired[list[dict[str, Any]] | None]
    planning_mode_enabled: NotRequired[bool | None]
    has_existing_plan: NotRequired[bool | None]
    # Dynamic todo tracking (for write_todos tool)
    todos: NotRequired[list[dict[str, Any]] | None]
    current_task_index: NotRequired[int | None]
    planning_call_count: NotRequired[int | None]
    # Planning phase: "planning" = create/edit only, "executing" = work through tasks
    planning_phase: NotRequired[str | None]
    # Persisted plan lifecycle (draft/ready/executing/paused/completed); None = no plan yet
    plan_lifecycle: NotRequired[PlanLifecycle | None]
    # Inter-agent delegation depth counter (reset each user turn)
    delegation_count: NotRequired[int | None]
    # Rolling conversation summary memory (checkpoint-backed)
    history_summary: NotRequired[str | None]
    history_summary_updated_at: NotRequired[str | None]
    summary_cursor_message_id: NotRequired[str | None]
    # Stable DB message identifiers for the current turn. ``user_message_id``
    # is the persisted prompt; ``assistant_message_id`` is reserved before
    # generation so the final ``AIMessage`` can carry the same ID that the
    # service later writes to the ``messages`` table.
    user_message_id: NotRequired[str | None]
    assistant_message_id: NotRequired[str | None]


class GraphStateView:
    """Typed access layer over LangGraph's dict-backed workflow state."""

    def __init__(self, state: GraphState | dict[str, Any] | None):
        self._state = state if isinstance(state, dict) else {}

    def messages(self) -> list[BaseMessage]:
        messages = self._state.get("messages", [])
        return messages if isinstance(messages, list) else []

    def conversation_id(self) -> str | None:
        value = self._state.get("conversation_id")
        return value if isinstance(value, str) or value is None else str(value)

    def user_id(self) -> str | None:
        value = self._state.get("user_id")
        return value if isinstance(value, str) or value is None else str(value)

    def device_id(self) -> str | None:
        value = self._state.get("device_id")
        return value if isinstance(value, str) or value is None else str(value)

    def selected_agent(self) -> str | None:
        value = self._state.get("selected_agent")
        return value if isinstance(value, str) or value is None else str(value)

    def iteration_count(self, default: int = 0) -> int:
        value = self._state.get("iteration_count")
        return int(value) if isinstance(value, int) else default

    def planning_call_count(self, default: int = 0) -> int:
        value = self._state.get("planning_call_count")
        return int(value) if isinstance(value, int) else default

    def context(self) -> GraphContext:
        context = self._state.get("context", {})
        if isinstance(context, dict):
            return cast(GraphContext, context)
        return cast(GraphContext, {})

    def context_copy(self) -> GraphContext:
        return cast(GraphContext, dict(self.context()))

    def attachments(self) -> list[Any]:
        attachments = self._state.get("attachments")
        if isinstance(attachments, list):
            return attachments
        context_attachments = self.context().get("attachments")
        return context_attachments if isinstance(context_attachments, list) else []

    def planning_flags(self) -> tuple[bool, bool]:
        planning_mode_enabled = self._state.get("planning_mode_enabled")
        has_existing_plan = self._state.get("has_existing_plan")
        if isinstance(planning_mode_enabled, bool) and isinstance(has_existing_plan, bool):
            return planning_mode_enabled, has_existing_plan

        context = self.context()
        return (
            bool(
                planning_mode_enabled
                if isinstance(planning_mode_enabled, bool)
                else context.get("planning_mode_enabled", False)
            ),
            bool(
                has_existing_plan
                if isinstance(has_existing_plan, bool)
                else context.get("has_existing_plan", False)
            ),
        )

    def tool_artifacts(self) -> list[dict[str, Any]]:
        value = self.context().get("tool_artifacts")
        return value if isinstance(value, list) else []

    def tool_images(self) -> list[dict[str, Any]]:
        value = self.context().get("tool_images")
        return value if isinstance(value, list) else []

    def pending_action_requests(self) -> list[dict[str, Any]]:
        value = self.context().get("pending_action_requests")
        return value if isinstance(value, list) else []

    def interrupt_metadata(self) -> dict[str, Any]:
        value = self.context().get("interrupt_metadata")
        return value if isinstance(value, dict) else {}

    def continuation_signal(self) -> ContinuationSignal:
        value = self.context().get("continuation_signal")
        if isinstance(value, dict):
            return cast(ContinuationSignal, value)
        return cast(ContinuationSignal, {})


# === Todo Management Schemas for write_todos tool ===


class TodoStatus(str, Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class TodoAction(str, Enum):
    """Action types for the write_todos tool."""

    SET_TODOS = "set_todos"  # Replace all todos with new list
    ADD_TODO = "add_todo"  # Add a single todo
    UPDATE_TODO = "update_todo"  # Update a todo's description or status
    COMPLETE_TODO = "complete_todo"  # Mark a todo as completed
    REMOVE_TODO = "remove_todo"  # Remove a todo
    START_TODO = "start_todo"  # Mark a todo as in progress


class TodoItem(BaseModel):
    """Represents a todo item in the planning workflow."""

    id: str = Field(..., description="Unique identifier for the todo")
    description: str = Field(..., description="What needs to be done")
    status: TodoStatus = Field(default=TodoStatus.PENDING, description="Current status of the todo")
    order: int = Field(..., description="Order/position in the todo list (0-indexed)")


class WriteTodosInput(BaseModel):
    """Input schema for the write_todos tool."""

    action: TodoAction = Field(..., description="The action to perform")
    todos: list[TodoItem] | None = Field(
        None, description="For SET_TODOS: the complete list of todos to set"
    )
    todo: TodoItem | None = Field(None, description="For ADD_TODO, UPDATE_TODO: the todo item")
    todo_id: str | None = Field(
        None,
        description="For COMPLETE_TODO, REMOVE_TODO, START_TODO: the ID of the todo to modify",
    )
    reason: str | None = Field(
        None,
        description="Optional reason for the action (e.g., why skipping/completing)",
    )


# === Document Exploration Schemas for search_documents tool ===


class DocumentAction(str, Enum):
    """Action types for the search_documents tool."""

    SCAN_ALL = "scan_all"  # Preview all documents in conversation
    READ_DOCUMENT = "read_document"  # Full content of specific document
    SEARCH_CHUNKS = "search_chunks"  # Vector search (existing functionality)
    GREP_DOCUMENT = "grep_document"  # Regex search in a document
    LIST_DOCUMENTS = "list_documents"  # List all documents in conversation
    VIEW_IMAGES = "view_images"  # Get images from a document


class SearchDocumentsInput(BaseModel):
    """Input schema for the search_documents tool."""

    action: DocumentAction = Field(..., description="The action to perform")
    document_id: str | None = Field(
        None, description="For READ_DOCUMENT, GREP_DOCUMENT: target document ID"
    )
    query: str | None = Field(None, description="For SEARCH_CHUNKS: semantic search query")
    pattern: str | None = Field(None, description="For GREP_DOCUMENT: regex pattern to search")
    reason: str | None = Field(None, description="Reasoning for the action (displayed to user)")
