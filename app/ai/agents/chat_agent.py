from typing import Optional
from datetime import datetime
import logging

from ..interfaces import BaseAgent
from ..schemas import AgentMessage, AgentResponse, AgentType

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """Chat agent for general conversations."""

    def __init__(self):
        super().__init__(AgentType.CHAT)
        self.logger = logging.getLogger("chat_agent")

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
