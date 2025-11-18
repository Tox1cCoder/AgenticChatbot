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


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class InterruptDecisionType(str, Enum):
    """Type of decision for handling a tool interrupt."""

    ACCEPT = "accept"
    EDIT = "edit"
    RESPOND = "respond"


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


class InterruptDecision(BaseModel):
    """Decision for handling a tool interrupt."""

    type: InterruptDecisionType = Field(..., description="Type of decision")
    task_id: Optional[str] = Field(
        None, description="Task ID to apply this decision to"
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


class AgentMessage(BaseModel):
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    attachments: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional image attachments with structure {name: str, mime: str, data: str (base64)}",
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


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_id: NotRequired[Optional[str]]
    user_id: NotRequired[Optional[str]]
    selected_agent: NotRequired[Optional[str]]
    response: NotRequired[Optional[AgentResponse]]
    context: NotRequired[Dict[str, Any]]
    persona: NotRequired[Optional[str]]
    reasoning_steps: NotRequired[Optional[List[Dict[str, Any]]]]
    tool_results: NotRequired[Optional[Dict[str, Any]]]
    iteration_count: NotRequired[Optional[int]]
