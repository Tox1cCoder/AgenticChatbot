"""
Chat Agent Implementation

This module implements a conversational chat agent with Gemini integration
for handling general conversations and casual interactions.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import logging
import time

from google import genai

from app.ai.interfaces import BaseAgent
from app.ai.schemas import (
    AgentRequest,
    AgentResponse,
    ChatAgentConfig,
    AgentType,
    AgentCapability,
    BaseAgentMessage,
    MessageRole,
    MessageType,
    AgentError,
)
from app.core.config import settings


class ConversationMemory:
    """
    Manages conversation memory for the chat agent.
    """
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.max_messages = max_messages
        self.conversations: Dict[str, List[BaseAgentMessage]] = {}
        self.logger = logging.getLogger("chat_agent.memory")
    
    def add_message(self, conversation_id: str, message: BaseAgentMessage) -> None:
        """
        Add a message to conversation memory.
        
        Args:
            conversation_id: ID of the conversation
            message: Message to add
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append(message)
        
        # Trim to max_messages, keeping most recent
        if len(self.conversations[conversation_id]) > self.max_messages:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_messages:]
        
        self.logger.debug(f"Added message to conversation {conversation_id}")
    
    def get_conversation_history(self, conversation_id: str) -> List[BaseAgentMessage]:
        """
        Get conversation history.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List[BaseAgentMessage]: List of messages in chronological order
        """
        return self.conversations.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clear conversation memory.
        
        Args:
            conversation_id: ID of the conversation to clear
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.logger.info(f"Cleared conversation memory for {conversation_id}")
    
    def get_context_string(self, conversation_id: str) -> str:
        """
        Get conversation history as a formatted string for context.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            str: Formatted conversation history
        """
        messages = self.get_conversation_history(conversation_id)
        if not messages:
            return ""
        
        context_parts = []
        for msg in messages:
            role_str = "User" if msg.role == MessageRole.USER else "Assistant"
            context_parts.append(f"{role_str}: {msg.content}")
        
        return "\n".join(context_parts)


class ResponseGenerator:
    """
    Handles response generation using Gemini API.
    """
    
    def __init__(self, config: ChatAgentConfig):
        """
        Initialize response generator.
        
        Args:
            config: Chat agent configuration
        """
        self.config = config
        self.logger = logging.getLogger("chat_agent.generator")
        self._client = None
    
    async def initialize(self) -> None:
        """Initialize the Gemini client"""
        try:
            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API key not configured")
            
            # Handle environment variable format
            if api_key.startswith("GEMINI_API_KEY="):
                api_key = api_key.split("=", 1)[-1].strip()
            
            self._client = genai.Client(api_key=api_key)
            self.logger.info("Gemini client initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    async def generate_response(
        self,
        user_message: str,
        conversation_context: str = "",
        conversation_id: str = ""
    ) -> str:
        """
        Generate a response using Gemini API.
        
        Args:
            user_message: The user's message
            conversation_context: Previous conversation context
            conversation_id: ID of the conversation
            
        Returns:
            str: Generated response
            
        Raises:
            AgentError: If response generation fails
        """
        if not self._client:
            raise AgentError(
                agent_id="chat_agent",
                agent_type=AgentType.CHAT,
                error_type="client_not_initialized",
                error_message="Gemini client not initialized"
            )
        
        try:
            # Build the prompt
            prompt_parts = []
            
            # Add system prompt
            prompt_parts.append(self.config.system_prompt)
            
            # Add conversation context if available
            if conversation_context:
                prompt_parts.append(f"\nPrevious conversation:\n{conversation_context}")
            
            # Add current user message
            prompt_parts.append(f"\nUser: {user_message}")
            prompt_parts.append("Assistant:")
            
            full_prompt = "\n".join(prompt_parts)
            
            self.logger.debug(f"Generating response for conversation {conversation_id}")
            
            # Generate response
            response = self._client.models.generate_content(
                model=self.config.model_name,
                contents=full_prompt,
                config=genai.GenerateContentConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            
            if hasattr(response, 'text') and response.text:
                generated_text = response.text.strip()
            else:
                generated_text = str(response).strip()
            
            if not generated_text:
                raise ValueError("Empty response from Gemini API")
            
            self.logger.debug(f"Successfully generated response ({len(generated_text)} chars)")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            raise AgentError(
                agent_id="chat_agent",
                agent_type=AgentType.CHAT,
                error_type="generation_error",
                error_message=f"Failed to generate response: {str(e)}"
            )


class ChatAgent(BaseAgent):
    """
    Chat agent implementation for conversational interactions.
    
    This agent handles casual conversations, greetings, and general inquiries
    using the Gemini API for response generation.
    """
    
    def __init__(self, config: ChatAgentConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the chat agent.
        
        Args:
            config: Configuration for the chat agent
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.chat_config = config
        self.memory = ConversationMemory(max_messages=config.conversation_memory_limit)
        self.response_generator = ResponseGenerator(config)
        
        # Performance tracking
        self.total_requests = 0
        self.total_response_time = 0.0
        self.last_request_time: Optional[datetime] = None
    
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization"""
        await self.response_generator.initialize()
        self.logger.info(f"Chat agent {self.agent_id} initialized with model {self.chat_config.model_name}")
    
    async def _cleanup_impl(self) -> None:
        """Implementation-specific cleanup"""
        self.logger.info(f"Chat agent {self.agent_id} cleaned up")
    
    async def _health_check_impl(self) -> bool:
        """Implementation-specific health check"""
        try:
            # Simple health check: ensure we can access the API
            if not self.response_generator._client:
                return False
            
            # Check if we're within reasonable response time
            if self.last_request_time:
                time_since_last = (datetime.utcnow() - self.last_request_time).total_seconds()
                if time_since_last > 300:  # 5 minutes
                    self.logger.warning("Agent hasn't processed requests recently")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def can_handle_request(self, request: AgentRequest) -> float:
        """
        Determine if this agent can handle the given request.
        
        Args:
            request: The request to evaluate
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        message_content = request.message.content.lower().strip()
        
        # High confidence for greetings and casual conversation
        greeting_keywords = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "thanks", "thank you"
        ]
        
        casual_keywords = [
            "tell me about yourself", "who are you", "what can you do",
            "how do you work", "nice to meet you"
        ]
        
        confidence = 0.0
        
        # Check for greetings
        for keyword in greeting_keywords:
            if keyword in message_content:
                confidence += 0.4
                break
        
        # Check for casual conversation
        for keyword in casual_keywords:
            if keyword in message_content:
                confidence += 0.3
                break
        
        # Prefer shorter, conversational messages
        word_count = len(message_content.split())
        if word_count <= 10:
            confidence += 0.2
        elif word_count <= 20:
            confidence += 0.1
        
        # Check if it's not clearly a search/RAG query
        rag_indicators = ["search", "find", "lookup", "document", "explain", "what is"]
        has_rag_indicators = any(indicator in message_content for indicator in rag_indicators)
        
        if not has_rag_indicators:
            confidence += 0.1
        
        # Check required capabilities
        if request.required_capabilities:
            agent_capabilities = set(self.capabilities)
            required_capabilities = set(request.required_capabilities)
            
            if AgentCapability.CONVERSATION in required_capabilities:
                confidence += 0.2
            
            # Reduce confidence if we don't have other required capabilities
            missing_capabilities = required_capabilities - agent_capabilities
            if missing_capabilities:
                confidence *= 0.5
        
        return min(confidence, 1.0)
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process an incoming request and return a response.
        
        Args:
            request: The request to process
            
        Returns:
            AgentResponse: The response from the agent
        """
        start_time = time.time()
        self.last_request_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Processing request {request.request_id}")
            
            # Add user message to memory
            conversation_id = str(request.conversation_id)
            self.memory.add_message(conversation_id, request.message)
            
            # Get conversation context
            context = self.memory.get_context_string(conversation_id)
            
            # Generate response
            response_text = await self.response_generator.generate_response(
                user_message=request.message.content,
                conversation_context=context,
                conversation_id=conversation_id
            )
            
            # Create response message
            response_message = BaseAgentMessage(
                role=MessageRole.ASSISTANT,
                message_type=MessageType.TEXT,
                content=response_text
            )
            
            # Add response to memory
            self.memory.add_message(conversation_id, response_message)
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            
            # Update statistics
            self.total_requests += 1
            self.total_response_time += processing_time
            
            # Create agent response
            agent_response = AgentResponse(
                request_id=request.request_id,
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                message=response_message,
                confidence=await self.can_handle_request(request),
                processing_time_ms=processing_time,
                metadata={
                    "model_used": self.chat_config.model_name,
                    "temperature": self.chat_config.temperature,
                    "conversation_length": len(self.memory.get_conversation_history(conversation_id)),
                    "total_requests": self.total_requests,
                    "avg_response_time_ms": self.total_response_time / self.total_requests
                }
            )
            
            self.logger.info(
                f"Successfully processed request {request.request_id} "
                f"in {processing_time}ms"
            )
            
            return agent_response
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Failed to process request {request.request_id}: {e}")
            
            # Return error response
            error_message = BaseAgentMessage(
                role=MessageRole.ASSISTANT,
                message_type=MessageType.ERROR,
                content="I apologize, but I'm experiencing technical difficulties. Please try again."
            )
            
            return AgentResponse(
                request_id=request.request_id,
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                message=error_message,
                confidence=0.0,
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent performance statistics.
        
        Returns:
            Dict[str, Any]: Agent statistics
        """
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "total_requests": self.total_requests,
            "average_response_time_ms": avg_response_time,
            "last_request": self.last_request_time.isoformat() if self.last_request_time else None,
            "active_conversations": len(self.memory.conversations),
            "model_name": self.chat_config.model_name,
            "temperature": self.chat_config.temperature
        }


# Factory function for creating chat agent
def create_chat_agent(
    agent_id: str = "chat_agent_001",
    system_prompt: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    memory_limit: int = 20
) -> ChatAgent:
    """
    Create a configured chat agent.
    
    Args:
        agent_id: Unique identifier for the agent
        system_prompt: Custom system prompt
        model_name: Name of the model to use
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        memory_limit: Maximum messages to keep in memory
        
    Returns:
        ChatAgent: Configured chat agent instance
    """
    config = ChatAgentConfig(
        agent_id=agent_id,
        agent_type=AgentType.CHAT,
        name="Chat Agent",
        description="Conversational agent for casual interactions and general chat",
        capabilities=[AgentCapability.CONVERSATION],
        system_prompt=system_prompt or "You are a helpful and friendly assistant. Respond naturally and conversationally.",
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        conversation_memory_limit=memory_limit
    )
    
    return ChatAgent(config)


__all__ = [
    "ConversationMemory",
    "ResponseGenerator",
    "ChatAgent",
    "create_chat_agent",
]
