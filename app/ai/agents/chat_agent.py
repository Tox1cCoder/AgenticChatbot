from typing import Optional
from datetime import datetime
import logging

from ..interfaces import BaseAgent
from ..schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageType,
    AgentRequest,
    AgentConfig,
    ChatAgentConfig,
    AgentCapability,
)

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """Chat agent for general conversations."""

    def __init__(self):
        # Create a proper ChatAgentConfig instance
        config = ChatAgentConfig(
            agent_id="chat_agent",
            agent_type=AgentType.CHAT,
            name="Chat Agent",
            description="General purpose chat agent for conversations",
            capabilities=[AgentCapability.CONVERSATION],
            system_prompt="You are a helpful assistant.",
            model_name="gemini-2.5-flash",
        )
        super().__init__(config)
        self.logger = logging.getLogger("chat_agent")

    async def _initialize_impl(self) -> None:
        """Initialize the chat agent."""
        self.logger.info("Chat agent initialized successfully")

    async def _cleanup_impl(self) -> None:
        """Clean up chat agent resources."""
        self.logger.info("Chat agent cleaned up successfully")

    async def _health_check_impl(self) -> bool:
        """Check if the chat agent is healthy."""
        return True

    async def can_handle_request(self, request: AgentRequest) -> float:
        """Determine if this agent can handle the given request."""
        # Return high confidence for general chat requests
        return 0.8

    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process an agent request and return a response."""
        # Delegate to the existing process_message method
        return await self.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
        )

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a message using chat functionality."""

        try:
            user_input = message.content

            response_content = f"Hello! I'm your AI assistant. You said: '{user_input}'. I'm here to help with conversations, answer questions, and provide assistance. How can I help you today?"

            response_message = AgentMessage(
                role=message.role,
                content=response_content,
                message_type=message.message_type,
            )

            return AgentResponse(
                response_id=message.id,
                request_id=message.id,
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=response_message,
                confidence=0.95,
                processing_time_ms=100,
                metadata={
                    "agent_type": "chat",
                    "input_length": len(user_input),
                    "timestamp": datetime.now().isoformat(),
                },
            )

        except Exception as e:
            self.logger.error(f"Chat processing failed: {e}")

            error_message = AgentMessage(
                role=message.role,
                content="I apologize, but I'm having trouble processing your message right now. Please try again.",
                message_type=MessageType.ERROR,
            )

            return AgentResponse(
                response_id=message.id,
                request_id=message.id,
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=error_message,
                confidence=0.0,
                processing_time_ms=50,
                error=str(e),
            )
