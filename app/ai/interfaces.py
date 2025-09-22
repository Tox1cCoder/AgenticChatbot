"""
Agent Interfaces and Abstract Base Classes

This module defines the core interfaces and abstract base classes for the multi-agent system,
following SOLID principles and supporting dependency injection.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
import logging

from app.ai.schemas import (
    AgentRequest,
    AgentResponse, 
    AgentConfig,
    AgentType,
    AgentCapability,
    BaseAgentMessage,
    ToolCall,
    ToolResult,
    ConversationState,
    AgentError,
)


class IAgent(ABC):
    """
    Interface for all agents in the multi-agent system.
    
    This interface defines the contract that all agents must implement,
    ensuring consistency and enabling polymorphic usage.
    """
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique identifier for this agent instance"""
        pass
    
    @property 
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Type of this agent"""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """List of capabilities this agent provides"""
        pass
    
    @property
    @abstractmethod
    def config(self) -> AgentConfig:
        """Configuration for this agent"""
        pass
    
    @abstractmethod
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process an incoming request and return a response.
        
        Args:
            request: The request to process
            
        Returns:
            AgentResponse: The response from the agent
            
        Raises:
            AgentError: If processing fails
        """
        pass
    
    @abstractmethod
    async def can_handle_request(self, request: AgentRequest) -> float:
        """
        Determine if this agent can handle the given request.
        
        Args:
            request: The request to evaluate
            
        Returns:
            float: Confidence score (0.0 to 1.0) indicating ability to handle request
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the agent is healthy and ready to process requests.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent and any required resources"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any resources used by the agent"""
        pass


class IAgentRouter(ABC):
    """
    Interface for agent routing functionality.
    
    The router is responsible for selecting the most appropriate agent
    for a given request based on various strategies.
    """
    
    @abstractmethod
    async def route_request(self, request: AgentRequest) -> AgentType:
        """
        Route a request to the most appropriate agent.
        
        Args:
            request: The request to route
            
        Returns:
            AgentType: The type of agent best suited for this request
            
        Raises:
            AgentError: If routing fails
        """
        pass
    
    @abstractmethod
    async def get_agent_scores(self, request: AgentRequest) -> Dict[AgentType, float]:
        """
        Get confidence scores for all available agents for a given request.
        
        Args:
            request: The request to evaluate
            
        Returns:
            Dict[AgentType, float]: Mapping of agent types to confidence scores
        """
        pass
    
    @abstractmethod
    def register_agent(self, agent: IAgent) -> None:
        """
        Register an agent with the router.
        
        Args:
            agent: The agent to register
        """
        pass
    
    @abstractmethod
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the router.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        pass


class ITool(ABC):
    """
    Interface for tools that can be used by agents.
    
    Tools provide specific functionality that agents can leverage
    to enhance their capabilities.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this tool"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this tool does"""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema for tool parameters"""
        pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Parameters for tool execution
            
        Returns:
            ToolResult: Result of tool execution
        """
        pass
    
    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate that parameters are correct for this tool.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            bool: True if parameters are valid
        """
        pass


class IAgentService(ABC):
    """
    Interface for the main agent service that orchestrates the multi-agent system.
    
    This service provides the main entry point for external systems
    to interact with the agent system.
    """
    
    @abstractmethod
    async def process_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a message through the multi-agent system.
        
        Args:
            conversation_id: ID of the conversation
            user_id: ID of the user
            message: The message to process
            context: Optional context information
            
        Returns:
            str: The response from the agent system
        """
        pass
    
    @abstractmethod
    async def get_conversation_state(self, conversation_id: UUID) -> ConversationState:
        """
        Get the current state of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            ConversationState: Current conversation state
        """
        pass
    
    @abstractmethod
    async def reset_conversation(self, conversation_id: UUID) -> None:
        """
        Reset a conversation to its initial state.
        
        Args:
            conversation_id: ID of the conversation to reset
        """
        pass


class BaseAgent(IAgent):
    """
    Abstract base class providing common functionality for all agents.
    
    This class implements common patterns and provides a foundation
    that specific agent implementations can build upon.
    """
    
    def __init__(self, config: AgentConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the base agent.
        
        Args:
            config: Configuration for this agent
            logger: Optional logger instance
        """
        self._config = config
        self._logger = logger or logging.getLogger(f"agent.{config.agent_id}")
        self._initialized = False
        self._tools: Dict[str, ITool] = {}
    
    @property
    def agent_id(self) -> str:
        return self._config.agent_id
    
    @property
    def agent_type(self) -> AgentType:
        return self._config.agent_type
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return self._config.capabilities
    
    @property
    def config(self) -> AgentConfig:
        return self._config
    
    @property
    def logger(self) -> logging.Logger:
        return self._logger
    
    async def initialize(self) -> None:
        """Initialize the agent"""
        if self._initialized:
            return
        
        self._logger.info(f"Initializing agent {self.agent_id}")
        await self._initialize_impl()
        self._initialized = True
        self._logger.info(f"Agent {self.agent_id} initialized successfully")
    
    async def cleanup(self) -> None:
        """Clean up agent resources"""
        if not self._initialized:
            return
        
        self._logger.info(f"Cleaning up agent {self.agent_id}")
        await self._cleanup_impl()
        self._initialized = False
        self._logger.info(f"Agent {self.agent_id} cleaned up successfully")
    
    async def health_check(self) -> bool:
        """Check agent health"""
        if not self._initialized:
            return False
        
        try:
            return await self._health_check_impl()
        except Exception as e:
            self._logger.error(f"Health check failed for agent {self.agent_id}: {e}")
            return False
    
    def register_tool(self, tool: ITool) -> None:
        """
        Register a tool with this agent.
        
        Args:
            tool: The tool to register
        """
        self._tools[tool.name] = tool
        self._logger.info(f"Registered tool {tool.name} with agent {self.agent_id}")
    
    def unregister_tool(self, tool_name: str) -> None:
        """
        Unregister a tool from this agent.
        
        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._logger.info(f"Unregistered tool {tool_name} from agent {self.agent_id}")
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            ToolResult: Result of the tool execution
        """
        if tool_call.name not in self._tools:
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                success=False,
                error=f"Tool {tool_call.name} not found",
                execution_time_ms=0
            )
        
        tool = self._tools[tool_call.name]
        
        try:
            # Validate parameters
            if not await tool.validate_parameters(tool_call.parameters):
                return ToolResult(
                    call_id=tool_call.id,
                    name=tool_call.name,
                    result=None,
                    success=False,
                    error="Invalid parameters",
                    execution_time_ms=0
                )
            
            # Execute tool
            result = await tool.execute(tool_call.parameters)
            return result
            
        except Exception as e:
            self._logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                success=False,
                error=str(e),
                execution_time_ms=0
            )
    
    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization"""
        pass
    
    @abstractmethod
    async def _cleanup_impl(self) -> None:
        """Implementation-specific cleanup"""
        pass
    
    @abstractmethod
    async def _health_check_impl(self) -> bool:
        """Implementation-specific health check"""
        pass


class BaseTool(ITool):
    """
    Abstract base class for tools.
    
    Provides common functionality and structure for tool implementations.
    """
    
    def __init__(self, name: str, description: str, parameters_schema: Dict[str, Any]):
        """
        Initialize the base tool.
        
        Args:
            name: Unique name for this tool
            description: Description of what this tool does
            parameters_schema: JSON schema for parameters
        """
        self._name = name
        self._description = description
        self._parameters_schema = parameters_schema
        self._logger = logging.getLogger(f"tool.{name}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return self._parameters_schema
    
    @property
    def logger(self) -> logging.Logger:
        return self._logger
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Basic parameter validation against schema.
        Override for more complex validation.
        """
        # Basic validation - check required fields exist
        required_fields = self._parameters_schema.get("required", [])
        for field in required_fields:
            if field not in parameters:
                return False
        return True


# Export all interfaces and base classes
__all__ = [
    "IAgent",
    "IAgentRouter", 
    "ITool",
    "IAgentService",
    "BaseAgent",
    "BaseTool",
]