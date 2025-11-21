import logging
import asyncio
from typing import Optional, List
from uuid import UUID

from ..ai.graph import create_workflow
from ..ai.schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ..repositories.conversation import ConversationRepository
from ..repositories.document import DocumentRepository
from ..utils.text_processing import sanitize_persona

logger = logging.getLogger(__name__)


class AIService:

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        conversation_repository: ConversationRepository,
        document_repository: Optional[DocumentRepository] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        self.checkpointer = checkpointer
        self.conversation_repository = conversation_repository
        self.document_repository = document_repository
        self.workflow = create_workflow(
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            checkpointer=checkpointer,
            document_repository=document_repository,
        )

    def _load_persona(self, conversation_id: UUID) -> Optional[str]:
        """Load persona from conversation"""
        try:
            conversation = self.conversation_repository.get_by_id(conversation_id)
            return conversation.persona_prompt if conversation else None
        except Exception as e:
            logger.error(
                f"Error loading persona for conversation {conversation_id}: {e}"
            )
            return None

    def _build_error_response(self, message: str) -> AgentResponse:
        return AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content=message),
            metadata={"error": True},
            error=message,
        )

    async def process_message(
        self,
        conversation_id: UUID,
        user_id: UUID,
        message: str,
        attachments: Optional[list] = None,
    ) -> AgentResponse:

        thread_id = (
            str(conversation_id) if conversation_id and self.checkpointer else None
        )

        # Load and sanitize persona
        persona = self._load_persona(conversation_id)
        persona = sanitize_persona(persona)

        response = await self.workflow.execute(
            message=message,
            conversation_id=str(conversation_id) if conversation_id else None,
            user_id=str(user_id) if user_id else None,
            thread_id=thread_id,
            persona=persona,
            attachments=attachments,
        )

        if response:
            return response

        return self._build_error_response("Error: No response generated")

    async def generate_bot_response(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        attachments: Optional[list] = None,
    ) -> AgentResponse:

        if conversation_id is None or user_id is None:
            logger.warning("Conversation ID or User ID is None")
            response = await self.workflow.execute(
                message=user_message,
                conversation_id=None,
                user_id=None,
                persona=None,
                attachments=attachments,
            )
            if response:
                return response
            return self._build_error_response("Error: No response generated")

        return await self.process_message(
            conversation_id=conversation_id,
            user_id=user_id,
            message=user_message,
            attachments=attachments,
        )

    async def resume_workflow(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_input: Optional[str] = None,
        rejection_messages: Optional[List] = None,
    ) -> AgentResponse:
        """
        Resume a workflow that was interrupted (e.g. for human approval).

        Args:
            conversation_id: The conversation ID
            user_id: The user ID
            user_input: Optional user input (not currently used)
            rejection_messages: Optional list of ToolMessages indicating tool rejection
        """
        thread_id = (
            str(conversation_id) if conversation_id and self.checkpointer else None
        )

        if not thread_id:
            return self._build_error_response(
                "Cannot resume: Checkpointing not enabled or conversation ID missing"
            )

        try:
            # Check state before resume
            state_info = await self.workflow.get_state(thread_id)
            logger.info(
                f"Resuming workflow - interrupted: {state_info.get('interrupted')}, next: {state_info.get('next')}"
            )

            response = await self.workflow.resume(
                thread_id=thread_id,
                user_input=user_input,
                rejection_messages=rejection_messages,
            )

            if response:
                return response

            # If no response, check state again
            logger.warning("No response after resume, checking state...")
            final_state = await self.workflow.get_state(thread_id)
            logger.error(
                f"Final state after resume - next: {final_state.get('next')}, has response: {final_state.get('values', {}).get('response') is not None}"
            )

            return self._build_error_response(
                "Error: No response generated after resume"
            )

        except Exception as e:
            logger.error(f"Error resuming workflow: {e}", exc_info=True)
            return self._build_error_response(f"Error resuming workflow: {str(e)}")

    async def generate_bot_response_stream(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        attachments: Optional[list] = None,
    ):
        """
        Generate bot response with streaming support.
        Yields incremental token chunks as they arrive from the workflow.
        """
        thread_id = (
            str(conversation_id) if conversation_id and self.checkpointer else None
        )

        # Load and sanitize persona
        persona = None
        if conversation_id:
            persona = self._load_persona(conversation_id)
            persona = sanitize_persona(persona)

        # Track final response
        final_response = None

        try:
            async for event in self.workflow.execute_stream(
                message=user_message,
                conversation_id=str(conversation_id) if conversation_id else None,
                user_id=str(user_id) if user_id else None,
                thread_id=thread_id,
                persona=persona,
                attachments=attachments,
            ):
                event_type = event.get("type")

                if event_type == "node":
                    # Yield node execution notification
                    node_name = event.get("node")
                    yield {"type": "node", "node": node_name}

                elif event_type == "token":
                    # Yield incremental token chunks directly
                    content = event.get("content", "")
                    yield {"type": "token", "content": content}

                elif event_type == "tool_start":
                    # Yield tool start event
                    tool_name = event.get("name", "unknown")
                    yield {"type": "tool", "name": tool_name, "status": "start"}

                elif event_type == "tool_end":
                    # Yield tool end event
                    tool_name = event.get("name", "unknown")
                    yield {"type": "tool", "name": tool_name, "status": "end"}

                elif event_type == "complete":
                    # Store final response
                    final_response = event.get("response")

                elif event_type == "error":
                    # Yield error event
                    error_msg = event.get("error", "Unknown error")
                    yield {"type": "error", "error": error_msg}

                elif event_type == "interrupt":
                    # Yield interrupt event with complete information
                    # This signals the frontend that the workflow is paused for human input
                    next_nodes = event.get("next", [])
                    pending_tool_calls = event.get("pending_tool_calls")
                    logger.info(
                        f"Workflow interrupted before nodes: {next_nodes}, tool_calls: {pending_tool_calls}"
                    )
                    yield {
                        "type": "interrupt",
                        "next": next_nodes,
                        "thread_id": thread_id,
                        "pending_tool_calls": pending_tool_calls,
                        "message": "Workflow paused - awaiting approval for tool execution",
                    }

            # Yield final complete event with full response
            if final_response:
                yield {"type": "complete", "response": final_response}
            else:
                # Build error response if no final response
                error_response = self._build_error_response(
                    "Error: No response generated"
                )
                yield {"type": "complete", "response": error_response}

        except Exception as exc:
            logger.error(
                f"Error in streaming response generation: {exc}", exc_info=True
            )
            error_response = self._build_error_response(f"Error: {str(exc)}")
            yield {"type": "error", "error": str(exc), "response": error_response}

    def get_bot_response_sync(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
    ) -> AgentResponse:
        """Synchronous wrapper for generate_bot_response."""
        try:
            return asyncio.run(
                self.generate_bot_response(user_message, conversation_id, user_id)
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.generate_bot_response(user_message, conversation_id, user_id)
                )
            finally:
                loop.close()
