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
    temperature: float = 0.7
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
