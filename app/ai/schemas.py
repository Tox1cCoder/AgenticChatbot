from typing import Dict, List, Optional, Any, Annotated, TypedDict, NotRequired
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


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
    """Type of decision for handling a tool interrupt."""

    ACCEPT = "accept"
    APPROVE = "approve"
    EDIT = "edit"
    RESPOND = "respond"
    REJECT = "reject"


class ToolInterruptRequest(BaseModel):
    """Request details for a tool awaiting human approval."""

    action: str = Field(..., description="Tool name")
    args: Dict[str, Any] = Field(..., description="Tool arguments")
    description: Optional[str] = Field(
        None, description="Description of what the tool will do"
    )
    task_id: Optional[str] = Field(
        None, description="Task identifier from interrupt mechanism"
    )
    tool_call_id: Optional[str] = Field(
        None, description="Tool call identifier for resume mapping"
    )
    allowed_decisions: Optional[List[str]] = Field(
        None, description="Which decision types are permitted for this tool"
    )


class InterruptDecision(BaseModel):
    """Decision for handling a tool interrupt."""

    type: InterruptDecisionType = Field(..., description="Type of decision")
    task_id: Optional[str] = Field(
        None, description="Task ID to apply this decision to"
    )
    action: Optional[str] = Field(
        None, description="Name of the tool/action this decision applies to"
    )
    args: Optional[Dict[str, Any]] = Field(
        None,
        description="For EDIT: modified arguments. For RESPOND: feedback message",
    )


class InterruptResponse(BaseModel):
    """Response containing interrupt information for human approval."""

    interrupt_id: str = Field(..., description="Unique identifier for this interrupt")
    action_requests: List[ToolInterruptRequest] = Field(
        ..., description="List of tools awaiting approval"
    )
    thread_id: str = Field(..., description="Conversation thread ID for resuming")
    conversation_id: str = Field(..., description="Conversation ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata for the interrupt"
    )


class AgentMessage(BaseModel):
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional image attachments with structure {name: str, mime: str, data: str (base64)}",
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of tool calls to be executed",
    )


class AgentConfig(BaseModel):
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0


class AgentResponse(BaseModel):
    agent_type: AgentType
    agent_id: str
    message: AgentMessage
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_artifacts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of artifacts produced while executing tools",
    )
    error: Optional[str] = None
    suggested_questions: Optional[List[str]] = Field(
        default=None,
        description="0-3 follow-up question suggestions for continuing the conversation",
    )


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_id: NotRequired[Optional[str]]
    user_id: NotRequired[Optional[str]]
    selected_agent: NotRequired[Optional[str]]
    response: NotRequired[Optional[AgentResponse]]
    context: NotRequired[Dict[str, Any]]
    persona: NotRequired[Optional[str]]
    iteration_count: NotRequired[Optional[int]]
    pending_tool_calls: NotRequired[Optional[List[Any]]]
    # Multi-provider model configuration
    model_request: NotRequired[Optional[Dict[str, Any]]]
    # Task planning context fields
    task_plan_id: NotRequired[Optional[str]]
    current_task: NotRequired[Optional[Dict[str, Any]]]
    all_tasks: NotRequired[Optional[List[Dict[str, Any]]]]
    planning_mode_enabled: NotRequired[Optional[bool]]
    has_existing_plan: NotRequired[Optional[bool]]
    # Dynamic todo tracking (for write_todos tool)
    todos: NotRequired[Optional[List[Dict[str, Any]]]]
    current_task_index: NotRequired[Optional[int]]
    planning_call_count: NotRequired[Optional[int]]
    # Planning phase: "planning" = create/edit only, "executing" = work through tasks
    planning_phase: NotRequired[Optional[str]]
    # Inter-agent delegation depth counter (reset each user turn)
    delegation_count: NotRequired[Optional[int]]
    # Rolling conversation summary memory (checkpoint-backed)
    history_summary: NotRequired[Optional[str]]
    history_summary_updated_at: NotRequired[Optional[str]]
    summary_cursor_message_id: NotRequired[Optional[str]]


class Task(BaseModel):
    """Represents a single task in a plan."""

    description: str = Field(..., description="Clear, actionable task description")


class Plan(BaseModel):
    """Represents a complete task plan generated by the planning agent."""

    tasks: List[Task] = Field(..., description="Ordered list of tasks to complete")
    overall_goal: Optional[str] = Field(
        None, description="High-level description of what the plan achieves"
    )


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
    status: TodoStatus = Field(
        default=TodoStatus.PENDING, description="Current status of the todo"
    )
    order: int = Field(..., description="Order/position in the todo list (0-indexed)")


class WriteTodosInput(BaseModel):
    """Input schema for the write_todos tool."""

    action: TodoAction = Field(..., description="The action to perform")
    todos: Optional[List[TodoItem]] = Field(
        None, description="For SET_TODOS: the complete list of todos to set"
    )
    todo: Optional[TodoItem] = Field(
        None, description="For ADD_TODO, UPDATE_TODO: the todo item"
    )
    todo_id: Optional[str] = Field(
        None,
        description="For COMPLETE_TODO, REMOVE_TODO, START_TODO: the ID of the todo to modify",
    )
    reason: Optional[str] = Field(
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
    document_id: Optional[str] = Field(
        None, description="For READ_DOCUMENT, GREP_DOCUMENT: target document ID"
    )
    query: Optional[str] = Field(
        None, description="For SEARCH_CHUNKS: semantic search query"
    )
    pattern: Optional[str] = Field(
        None, description="For GREP_DOCUMENT: regex pattern to search"
    )
    reason: Optional[str] = Field(
        None, description="Reasoning for the action (displayed to user)"
    )
