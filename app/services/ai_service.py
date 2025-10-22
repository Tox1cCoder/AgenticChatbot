import logging
import asyncio
from typing import Optional
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
