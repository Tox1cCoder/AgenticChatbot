"""
AI System Schemas and Data Models

This module defines the core schemas and data models for the multi-agent system,
including agent communication, state management, and LangGraph integration.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class AgentType(str, Enum):
    """Available agent types in the system"""
    CHAT = "chat"
    RAG = "rag"
    ROUTER = "router"


class MessageRole(str, Enum):
    """Message roles for agent communication"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Types of messages in the agent system"""
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


class AgentCapability(str, Enum):
    """Capabilities that agents can provide"""
    CONVERSATION = "conversation"
    DOCUMENT_SEARCH = "document_search"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    ROUTING = "routing"
    ANALYSIS = "analysis"


class RoutingStrategy(str, Enum):
    """Strategies for routing requests to agents"""
    CONTENT_BASED = "content_based"
    INTENT_BASED = "intent_based"
    CAPABILITY_BASED = "capability_based"
    ROUND_ROBIN = "round_robin"


# Base schemas for agent communication
class BaseAgentMessage(BaseModel):
    """Base schema for all agent messages"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: MessageRole
    message_type: MessageType = MessageType.TEXT
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRequest(BaseModel):
    """Request sent to an agent"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    request_id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    user_id: UUID
    message: BaseAgentMessage
    context: Dict[str, Any] = Field(default_factory=dict)
    target_agent: Optional[AgentType] = None
    required_capabilities: List[AgentCapability] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Response from an agent"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    response_id: UUID = Field(default_factory=uuid4)
    request_id: UUID
    agent_type: AgentType
    agent_id: str
    message: BaseAgentMessage
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    processing_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ToolCall(BaseModel):
    """Schema for tool calls within agent execution"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    id: UUID = Field(default_factory=uuid4)
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolResult(BaseModel):
    """Schema for tool execution results"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    call_id: UUID
    name: str
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Agent configuration schemas
class AgentConfig(BaseModel):
    """Base configuration for all agents"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    agent_id: str
    agent_type: AgentType
    name: str
    description: str
    capabilities: List[AgentCapability]
    enabled: bool = True
    max_tokens: int = 4000
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    timeout_seconds: int = 30
    retry_count: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatAgentConfig(AgentConfig):
    """Configuration specific to chat agents"""
    system_prompt: str = "You are a helpful assistant."
    model_name: str = "gemini-2.5-flash"
    conversation_memory_limit: int = 20


class RAGAgentConfig(AgentConfig):
    """Configuration specific to RAG agents"""
    vector_store_path: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    similarity_threshold: float = 0.7
    max_documents: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RouterConfig(BaseModel):
    """Configuration for the agent router"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    default_strategy: RoutingStrategy = RoutingStrategy.CONTENT_BASED
    fallback_agent: AgentType = AgentType.CHAT
    confidence_threshold: float = 0.6
    enable_load_balancing: bool = True
    max_concurrent_requests: int = 10


# State management schemas for LangGraph
class ConversationState(BaseModel):
    """State maintained throughout a conversation"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conversation_id: UUID
    user_id: UUID
    messages: List[BaseAgentMessage] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    current_agent: Optional[AgentType] = None
    routing_history: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GraphState(BaseModel):
    """Complete state for LangGraph workflow"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Current request being processed
    current_request: Optional[AgentRequest] = None
    current_response: Optional[AgentResponse] = None
    
    # Conversation state
    conversation: ConversationState
    
    # Workflow state
    next_node: Optional[str] = None
    should_continue: bool = True
    error: Optional[str] = None
    
    # Agent selection and routing
    selected_agent: Optional[AgentType] = None
    routing_confidence: float = 0.0
    available_agents: List[AgentType] = Field(default_factory=list)
    
    # Performance tracking
    workflow_start_time: datetime = Field(default_factory=datetime.utcnow)
    node_execution_times: Dict[str, int] = Field(default_factory=dict)


# Integration schemas for existing system
class MessageServiceRequest(BaseModel):
    """Request format for integration with existing message service"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    conversation_id: UUID
    user_id: UUID
    user_message: str
    message_history: List[Dict[str, Any]] = Field(default_factory=list)
    system_context: Dict[str, Any] = Field(default_factory=dict)


class MessageServiceResponse(BaseModel):
    """Response format for integration with existing message service"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    response_text: str
    agent_used: AgentType
    confidence: float
    processing_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# Agent registration and discovery schemas
class AgentRegistration(BaseModel):
    """Schema for registering agents in the system"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    config: AgentConfig
    instance_id: str = Field(default_factory=lambda: str(uuid4()))
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    health_check_url: Optional[str] = None
    status: Literal["active", "inactive", "error"] = "active"


class AgentRegistry(BaseModel):
    """Registry of all available agents"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    agents: Dict[str, AgentRegistration] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentRegistration]:
        """Get all agents of a specific type"""
        return [
            agent for agent in self.agents.values()
            if agent.config.agent_type == agent_type and agent.status == "active"
        ]
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentRegistration]:
        """Get all agents with a specific capability"""
        return [
            agent for agent in self.agents.values()
            if capability in agent.config.capabilities and agent.status == "active"
        ]


# Error handling schemas
class AgentError(BaseModel):
    """Schema for agent-specific errors"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    error_id: UUID = Field(default_factory=uuid4)
    agent_id: str
    agent_type: AgentType
    error_type: str
    error_message: str
    error_details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[UUID] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None


# Export all schemas
__all__ = [
    # Enums
    "AgentType",
    "MessageRole", 
    "MessageType",
    "AgentCapability",
    "RoutingStrategy",
    
    # Base communication schemas
    "BaseAgentMessage",
    "AgentRequest",
    "AgentResponse",
    "ToolCall",
    "ToolResult",
    
    # Configuration schemas
    "AgentConfig",
    "ChatAgentConfig", 
    "RAGAgentConfig",
    "RouterConfig",
    
    # State management schemas
    "ConversationState",
    "GraphState",
    
    # Integration schemas
    "MessageServiceRequest",
    "MessageServiceResponse",
    
    # Registry schemas
    "AgentRegistration",
    "AgentRegistry",
    
    # Error schemas
    "AgentError",
]
