import asyncio
from typing import Optional, List, Dict, Any
from uuid import UUID

from ..ai.graph import create_workflow
from ..ai.schemas import (
    AgentMessage,
    AgentResponse,
    AgentType,
    MessageRole,
    InterruptDecision,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ..repositories.conversation import ConversationRepository
from ..repositories.document import DocumentRepository
from ..utils.text_processing import sanitize_persona
from ..ai.utils import make_json_safe
from ..core.config import settings
from ..core.response_constants import (
    ERROR_NO_RESPONSE,
    ERROR_NO_RESPONSE_RESUME,
    ERROR_NO_RESPONSE_DECISIONS,
    UNKNOWN_ERROR,
    WORKFLOW_PAUSED_MESSAGE,
)
from ..ai.prompts import TITLE_GENERATION_PROMPT


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
            return None

    def _build_error_response(self, message: str = ERROR_NO_RESPONSE) -> AgentResponse:
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
        current_task: Optional[Dict[str, Any]] = None,
        all_tasks: Optional[List[Dict[str, Any]]] = None,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
        existing_tasks: Optional[List[Dict[str, Any]]] = None,
        model_request: Optional[Dict[str, Any]] = None,
        persona: Optional[str] = None,
    ) -> AgentResponse:

        thread_id = (
            str(conversation_id) if conversation_id and self.checkpointer else None
        )

        if persona is None:
            persona = self._load_persona(conversation_id)
            persona = sanitize_persona(persona)

        response = await self.workflow.execute(
            message=message,
            conversation_id=str(conversation_id) if conversation_id else None,
            user_id=str(user_id) if user_id else None,
            thread_id=thread_id,
            persona=persona,
            attachments=attachments,
            current_task=current_task,
            all_tasks=all_tasks,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
            existing_tasks=existing_tasks,
            model_request=model_request,
        )

        if response:
            if response.metadata and "interrupt" in response.metadata:
                return response
            return response

        return self._build_error_response()

    async def generate_bot_response(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        attachments: Optional[list] = None,
        current_task: Optional[Dict[str, Any]] = None,
        all_tasks: Optional[List[Dict[str, Any]]] = None,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
        existing_tasks: Optional[List[Dict[str, Any]]] = None,
        model_request: Optional[Dict[str, Any]] = None,
        persona: Optional[str] = None,
    ) -> AgentResponse:

        if conversation_id is None or user_id is None:
            response = await self.workflow.execute(
                message=user_message,
                conversation_id=None,
                user_id=None,
                persona=None,
                attachments=attachments,
                current_task=current_task,
                all_tasks=all_tasks,
                planning_mode_enabled=planning_mode_enabled,
                has_existing_plan=has_existing_plan,
                existing_tasks=existing_tasks,
                model_request=model_request,
            )
            if response:
                return response
            return self._build_error_response()

        return await self.process_message(
            conversation_id=conversation_id,
            user_id=user_id,
            message=user_message,
            attachments=attachments,
            current_task=current_task,
            all_tasks=all_tasks,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
            existing_tasks=existing_tasks,
            model_request=model_request,
            persona=persona,
        )

    async def resume_workflow(
        self,
        conversation_id: UUID,
        user_id: UUID,
        user_input: Optional[str] = None,
    ) -> AgentResponse:
        thread_id = (
            str(conversation_id) if conversation_id and self.checkpointer else None
        )

        if not thread_id:
            return self._build_error_response(
                "Cannot resume: Checkpointing not enabled or conversation ID missing"
            )

        response = await self.workflow.resume(
            thread_id=thread_id,
            user_input=user_input,
        )

        if response:
            return response

        return self._build_error_response(ERROR_NO_RESPONSE_RESUME)

    async def resume_interrupted_execution(
        self,
        thread_id: str,
        decisions: List[InterruptDecision],
    ) -> AgentResponse:
        if not self.checkpointer:
            return self._build_error_response(
                "Cannot resume: Checkpointing not enabled"
            )

        response = await self.workflow.resume_with_decisions(
            thread_id=thread_id,
            decisions=decisions,
        )

        if response:
            return response

        return self._build_error_response(ERROR_NO_RESPONSE_DECISIONS)

    async def generate_bot_response_stream(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
        attachments: Optional[list] = None,
        current_task: Optional[Dict[str, Any]] = None,
        all_tasks: Optional[List[Dict[str, Any]]] = None,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
        existing_tasks: Optional[List[Dict[str, Any]]] = None,
        model_request: Optional[Dict[str, Any]] = None,
        persona: Optional[str] = None,
    ):
        thread_id = (
            str(conversation_id) if conversation_id and self.checkpointer else None
        )

        if persona is None and conversation_id:
            persona = self._load_persona(conversation_id)
            persona = sanitize_persona(persona)

        final_response = None

        async for event in self.workflow.execute_stream(
            message=user_message,
            conversation_id=str(conversation_id) if conversation_id else None,
            user_id=str(user_id) if user_id else None,
            thread_id=thread_id,
            persona=persona,
            attachments=attachments,
            current_task=current_task,
            all_tasks=all_tasks,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
            existing_tasks=existing_tasks,
            model_request=model_request,
        ):
            event_type = event.get("type")

            if event_type == "agent_selected":
                agent_name = event.get("agent", "unknown")
                yield {"type": "agent_selected", "agent": agent_name}

            elif event_type == "node":
                node_name = event.get("node")
                yield {"type": "node", "node": node_name}

            elif event_type == "thinking":
                content = event.get("content", "")
                yield {"type": "thinking", "content": content}

            elif event_type == "token":
                content = event.get("content", "")
                yield {"type": "token", "content": content}

            elif event_type == "tool_start":
                tool_name = event.get("name", "unknown")
                tool_call_id = event.get("tool_call_id")
                tool_args = event.get("args")
                yield {
                    "type": "tool",
                    "name": tool_name,
                    "status": "start",
                    "tool_call_id": tool_call_id,
                    "args": make_json_safe(tool_args),
                }

            elif event_type == "tool_end":
                tool_name = event.get("name", "unknown")
                tool_call_id = event.get("tool_call_id")
                result = event.get("result")
                yield {
                    "type": "tool",
                    "name": tool_name,
                    "status": "end",
                    "tool_call_id": tool_call_id,
                    "result": make_json_safe(result),
                }

            elif event_type == "complete":
                final_response = event.get("response")

            elif event_type == "error":
                error_msg = event.get("error", UNKNOWN_ERROR)
                yield {"type": "error", "error": error_msg}

            elif event_type == "continuation_start":
                # Pass through auto-continue round markers for UX/debugging
                yield event

            elif event_type == "node_complete":
                # Pass through node completion events
                yield event

            elif event_type == "interrupt":
                next_nodes = event.get("next", [])
                pending_tool_calls = event.get("pending_tool_calls")
                yield {
                    "type": "interrupt",
                    "next": next_nodes,
                    "thread_id": thread_id,
                    "pending_tool_calls": pending_tool_calls,
                    "interrupt": event.get("interrupt"),
                    "message": WORKFLOW_PAUSED_MESSAGE,
                }

        if final_response:
            yield {"type": "complete", "response": final_response}
        else:
            error_response = self._build_error_response()
            yield {"type": "complete", "response": error_response}

    def get_bot_response_sync(
        self,
        user_message: str,
        conversation_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
    ) -> AgentResponse:
        return asyncio.run(
            self.generate_bot_response(user_message, conversation_id, user_id)
        )

    async def generate_conversation_title(self, user_message: str) -> str:
        """
        Generate a concise, descriptive title for a conversation based on the first user message.

        Args:
            user_message: The first message from the user

        Returns:
            A short, descriptive title (max 50 characters)
        """
        try:
            from ..ai.agent_config import create_langchain_model

            llm = create_langchain_model(
                agent_type="title_generator",
                include_thinking=False,
            )

            prompt = TITLE_GENERATION_PROMPT.format(user_message=user_message)

            response = await llm.ainvoke(prompt)
            raw_title = response.content
            
            if isinstance(raw_title, list):
                # Handle list content (e.g. from Gemini)
                title_text = ""
                for part in raw_title:
                    if isinstance(part, dict) and part.get("type") == "text":
                        title_text += part.get("text", "")
                    elif isinstance(part, str):
                        title_text += part
                raw_title = title_text

            title = str(raw_title).strip()

            # Clean up the title
            title = title.strip("\"'")  # Remove quotes
            title = title.rstrip(".")  # Remove trailing period

            # Ensure it's not too long
            if len(title) > 50:
                title = title[:47] + "..."

            # Fallback to truncated message if generation fails
            if not title or len(title) < 3:
                title = user_message[:50]
                if len(user_message) > 50:
                    title = title[:47] + "..."

            return title

        except Exception as e:
            # Fallback: use truncated user message
            title = user_message[:50]
            if len(user_message) > 50:
                title = title[:47] + "..."
            return title
