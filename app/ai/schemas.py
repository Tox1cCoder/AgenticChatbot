from typing import Dict, List, Optional, Any, Annotated, TypedDict, NotRequired
from enum import Enum
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentType(str, Enum):
    CHAT = "chat"
    RAG = "rag"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentMessage(BaseModel):
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    agent_type: AgentType
    agent_id: str
    message: AgentMessage
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_id: NotRequired[Optional[str]]
    user_id: NotRequired[Optional[str]]
    selected_agent: NotRequired[Optional[str]]
    response: NotRequired[Optional[AgentResponse]]
    context: NotRequired[Dict[str, Any]]
