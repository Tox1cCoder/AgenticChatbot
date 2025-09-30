from typing import Optional
from datetime import datetime
from uuid import UUID
import logging

from google import genai

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
    MessageRole,
)
from ..memory import get_memory_manager
from ..prompts import (
    CHAT_AGENT_SYSTEM_PROMPT,
    ERROR_PROCESSING_MESSAGE,
)
from ...core.config import settings

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
            system_prompt=CHAT_AGENT_SYSTEM_PROMPT,
            model_name="gemini-2.5-flash",
        )
        super().__init__(config, logging.getLogger("chat_agent"))

        self._init_gemini_client()

    def _init_gemini_client(self) -> None:
        """Initialize the Gemini API client."""
        try:
            api_key = settings.gemini_api_key
            if not api_key:
                self.logger.error("Gemini API key not configured")
                self.gemini_client = None
                return

            # Clean up the API key if it has the prefix
            if api_key.startswith("GEMINI_API_KEY="):
                api_key = api_key.split("=", 1)[-1].strip()

            self.gemini_client = genai.Client(api_key=api_key)
            self.logger.info("Gemini client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            self.gemini_client = None

    async def _initialize_impl(self) -> None:
        """Initialize the chat agent."""
        self.logger.info("Chat agent initialized successfully")

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

            # Get memory manager and conversation memory
            memory_manager = get_memory_manager()
            conversation_memory = None
            context_messages = []

            if conversation_id and user_id:
                try:
                    conversation_memory = await memory_manager.get_memory(
                        UUID(conversation_id), UUID(user_id)
                    )

                    # Add current message to memory (for context only)
                    conversation_memory.add_message(message)

                    # Get recent conversation history
                    context_messages = conversation_memory.get_recent_messages(limit=10)

                except Exception as e:
                    self.logger.warning(f"Could not load conversation memory: {e}")

            # Build conversation prompt with history
            prompt = self._build_prompt_with_context(user_input, context_messages)

            # Generate response using Gemini
            response_content = await self._generate_with_gemini(prompt)

            response_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=response_content,
                message_type=MessageType.TEXT,
            )

            # Store response in memory (for context only)
            if conversation_memory:
                conversation_memory.add_message(response_message)

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
                    "memory_enabled": conversation_memory is not None,
                    "context_messages": len(context_messages),
                    "model": self.config.model_name,
                },
            )

        except Exception as e:
            self.logger.error(f"Chat processing failed: {e}")

            error_message = AgentMessage(
                role=MessageRole.ASSISTANT,
                content=ERROR_PROCESSING_MESSAGE,
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

    def _build_prompt_with_context(
        self, user_input: str, context_messages: list[AgentMessage]
    ) -> str:
        """Build a prompt with conversation context."""
        prompt_parts = [CHAT_AGENT_SYSTEM_PROMPT]

        # Add conversation history (excluding the current message)
        if len(context_messages) > 1:
            prompt_parts.append("\n\nConversation History:")
            for msg in context_messages[
                :-1
            ]:  # Exclude the last message (current user input)
                role_label = "User" if msg.role == MessageRole.USER else "Assistant"
                prompt_parts.append(f"{role_label}: {msg.content}")

        # Add current user input
        prompt_parts.append(f"\n\nUser: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API."""
        if not self.gemini_client:
            self.logger.error("Gemini client not initialized")
            return "[Error: Gemini API not configured]"

        try:
            response = self.gemini_client.models.generate_content(
                model=self.config.model_name, contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            return f"[Error: Failed to generate response - {str(e)}]"
