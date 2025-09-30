from typing import Optional, Dict, Any, List
from uuid import UUID
import logging
import asyncio
import concurrent.futures
from datetime import datetime

from ..ai.schemas import (
    AgentMessage,
    AgentResponse,
    MessageRole,
    MessageType,
    WorkflowConfig,
    ConversationState,
)
from ..ai.graph import Workflow, create_workflow
from ..ai.interfaces import IAgentService

logger = logging.getLogger(__name__)


class AIService(IAgentService):
    """
    Main AI service
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        Initialize the AI service with multi-agent workflow.

        Args:
            config: Optional workflow configuration
        """
        self.config = config or WorkflowConfig(
            max_iterations=3, max_retries=1, enable_logging=True, timeout_seconds=120
        )

        self.workflow = create_workflow(self.config)

        self.conversation_contexts: Dict[str, Dict[str, Any]] = {}

        self.stats = {
            "total_requests": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "average_response_time": 0.0,
        }

        logger.info("AIService initialized with streamlined workflow")

    # Implement abstract methods from IAgentService
    async def process_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        message: str,
        context: Optional[Dict[str, Any]] = None,
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
        return await self.generate_bot_response(message, conversation_id, user_id)

    async def get_conversation_state(self, conversation_id: UUID) -> ConversationState:
        """
        Get the current state of a conversation.

        Args:
            conversation_id: ID of the conversation

        Returns:
            ConversationState: Current conversation state
        """
        conversation_key = str(conversation_id)
        context = self.conversation_contexts.get(conversation_key, {})

        # Create a ConversationState from the stored context
        return ConversationState(
            conversation_id=conversation_id,
            user_id=UUID(
                context.get("user_id", "00000000-0000-0000-0000-000000000000")
            ),
            messages=[],  # Could be populated from context if needed
            context=context.get("user_preferences", {}),
            current_agent=None,
            routing_history=context.get("agent_usage", {}),
            session_metadata=context,
        )

    async def reset_conversation(self, conversation_id: UUID) -> None:
        """
        Reset a conversation to its initial state.

        Args:
            conversation_id: ID of the conversation to reset
        """
        conversation_key = str(conversation_id)
        if conversation_key in self.conversation_contexts:
            del self.conversation_contexts[conversation_key]
            logger.info(f"Reset conversation {conversation_key}")

    async def generate_bot_response(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
    ) -> str:
        """
        Generate a bot response using the multi-agent system.

        Args:
            user_message: The user's input message
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID for personalization

        Returns:
            str: Generated response from the multi-agent system
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1

        # Convert to agent message format
        agent_message = AgentMessage(
            role=MessageRole.USER,
            content=user_message,
            message_type=MessageType.TEXT,
            metadata={
                "conversation_id": (str(conversation_id) if conversation_id else None),
                "user_id": str(user_id) if user_id else None,
                "timestamp": start_time.isoformat(),
                "source": "message_service",
            },
        )

        # Get conversation context if available
        conversation_key = str(conversation_id) if conversation_id else "default"
        context = self.conversation_contexts.get(conversation_key, {})

        logger.info(
            f"Processing message for conversation {conversation_key}: {user_message[:100]}..."
        )

        # Execute the multi-agent workflow
        response = await self.workflow.execute(
            message=agent_message,
            conversation_id=str(conversation_id) if conversation_id else None,
            user_id=str(user_id) if user_id else None,
        )

        # Update conversation context
        self._update_conversation_context(conversation_key, agent_message, response)

        # Calculate metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(execution_time, True)

        # Extract response content
        response_content = (
            response.message.content if response else "[Error: No response generated]"
        )

        logger.info(f"Response generated successfully in {execution_time:.2f}s")
        return response_content

    def get_bot_response_sync(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
    ) -> str:
        """
        Synchronous wrapper for generate_bot_response.

        Args:
            user_message: The user's input message
            conversation_id: Optional conversation ID for context
            user_id: Optional user ID for personalization

        Returns:
            str: Generated response from the multi-agent system
        """
        # Run the async method in a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we need to use a different approach
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.generate_bot_response(
                            user_message, conversation_id, user_id
                        ),
                    )
                    return future.result(timeout=self.config.timeout_seconds)
            else:
                return loop.run_until_complete(
                    self.generate_bot_response(user_message, conversation_id, user_id)
                )
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(
                self.generate_bot_response(user_message, conversation_id, user_id)
            )

    def _update_conversation_context(
        self, conversation_key: str, message: AgentMessage, response: AgentResponse
    ) -> None:
        """Update conversation context for future interactions."""
        if conversation_key not in self.conversation_contexts:
            self.conversation_contexts[conversation_key] = {
                "message_count": 0,
                "last_interaction": None,
                "user_preferences": {},
                "conversation_themes": [],
                "agent_usage": {},
            }

        context = self.conversation_contexts[conversation_key]
        context["message_count"] += 1
        context["last_interaction"] = datetime.now().isoformat()

        # Track agent usage
        if response and hasattr(response, "agent_id"):
            agent_id = response.agent_id
            context["agent_usage"][agent_id] = (
                context["agent_usage"].get(agent_id, 0) + 1
            )

        # Limit context size to prevent memory bloat
        if len(self.conversation_contexts) > 1000:
            # Remove oldest conversations
            oldest_key = min(
                self.conversation_contexts.keys(),
                key=lambda k: self.conversation_contexts[k]["last_interaction"],
            )
            del self.conversation_contexts[oldest_key]

    def _update_stats(self, execution_time: float, success: bool) -> None:
        """Update performance statistics."""
        if success:
            self.stats["successful_responses"] += 1
        else:
            self.stats["failed_responses"] += 1

        # Update average response time
        total_successful = self.stats["successful_responses"]
        if total_successful > 0 and success:
            current_avg = self.stats["average_response_time"]
            self.stats["average_response_time"] = (
                current_avg * (total_successful - 1) + execution_time
            ) / total_successful

    def get_conversation_stats(
        self, conversation_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get statistics for a specific conversation or overall."""
        if conversation_id:
            conversation_key = str(conversation_id)
            return self.conversation_contexts.get(conversation_key, {})

        return {
            "service_stats": self.stats,
            "active_conversations": len(self.conversation_contexts),
            "workflow_stats": self.workflow.get_execution_stats(),
        }

    def clear_conversation_context(self, conversation_id: UUID) -> bool:
        """Clear context for a specific conversation."""
        conversation_key = str(conversation_id)
        if conversation_key in self.conversation_contexts:
            del self.conversation_contexts[conversation_key]
            logger.info(f"Cleared context for conversation {conversation_key}")
            return True
        return False

    def get_available_agents(self) -> List[str]:
        """Get list of available agents in the system."""
        return list(self.workflow.agents.keys())

    def get_available_tools(self) -> List[str]:
        """Get list of available tools in the system."""
        return list(self.workflow.tools.keys())
