import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING, List
from uuid import UUID

from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .schemas import (
    GraphState,
    AgentMessage,
    AgentResponse,
    MessageRole,
    InterruptDecision,
)
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .memory import get_memory_manager
from .hitl_config import build_interrupt_response
from ..core.config import settings

if TYPE_CHECKING:
    from ..repositories.document import DocumentRepository

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        document_repository: Optional["DocumentRepository"] = None,
    ):
        self.qdrant_client = qdrant_client
        self.router = Router()
        self.chat_agent = ChatAgent()
        self.rag_agent = RAGAgent(
            settings=settings,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            collection_name=settings.qdrant_collection_name,
        )
        self.search_agent = SearchAgent()
        self.image_generator_agent = ImageGeneratorAgent()
        self.agents = {
            "chat_agent": self.chat_agent,
            "rag_agent": self.rag_agent,
            "search_agent": self.search_agent,
            "image_generator_agent": self.image_generator_agent,
        }

        self.checkpointer = checkpointer
        self.document_repository = document_repository

        self.graph = self._build_graph()
        logger.info(
            f"Multi-agent workflow initialized (checkpointing: {'enabled' if checkpointer else 'disabled'})"
        )

        self._cleanup_agents = [
            self.chat_agent,
            self.search_agent,
            self.rag_agent,
            self.image_generator_agent,
        ]

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("search_agent", self._search_node)
        workflow.add_node("image_generator_agent", self._image_generator_node)

        workflow.add_edge(START, "route")

        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {
                "chat_agent": "chat_agent",
                "rag_agent": "rag_agent",
                "search_agent": "search_agent",
                "image_generator_agent": "image_generator_agent",
                "end": END,
            },
        )

        workflow.add_edge("chat_agent", END)
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("search_agent", END)
        workflow.add_edge("image_generator_agent", END)

        # Compile with checkpointer if provided
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

    async def _route_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            logger.error("No messages to route")
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        # Check if conversation has documents available
        conversation_id = state.get("conversation_id")
        has_documents = self._conversation_has_documents(conversation_id)

        persona = state.get("persona")
        agent_msg = AgentMessage(
            role=MessageRole.USER, content=content, metadata={"persona": persona}
        )

        selected_agent = await self.router.route_message(
            agent_msg, list(self.agents.keys()), has_documents=has_documents
        )

        if selected_agent == "rag_agent" and not has_documents:
            selected_agent = "chat_agent"

        state["selected_agent"] = selected_agent
        return state

    def _conversation_has_documents(self, conversation_id: Optional[str]) -> bool:
        if not conversation_id or not self.document_repository:
            return False

        try:
            conversation_uuid = UUID(conversation_id)
        except ValueError:
            logger.warning(
                "Invalid conversation_id '%s' encountered while checking documents",
                conversation_id,
            )
            return False

        try:
            return self.document_repository.count_by_conversation(conversation_uuid) > 0
        except Exception as exc:
            logger.error(
                "Failed to determine document availability for conversation %s: %s",
                conversation_id,
                exc,
            )
            return False

    async def _chat_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        conversation_history = []
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.chat_history_max_messages
                if settings.chat_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments,
        )

        response = await self.chat_agent.process_message(agent_msg, conversation_id)

        state["response"] = response
        state.setdefault("messages", []).append(
            AIMessage(content=response.message.content)
        )

        return state

    async def _rag_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        conversation_history = []
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.rag_history_max_messages
                if settings.rag_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments,
        )

        response = await self.rag_agent.process_message(agent_msg, conversation_id)

        state["response"] = response
        state.setdefault("messages", []).append(
            AIMessage(content=response.message.content)
        )

        return state

    async def _search_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            logger.error("No messages in state for search agent")
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        conversation_history = []
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.search_history_max_messages
                if settings.search_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments,
        )

        response = await self.search_agent.process_message(agent_msg, conversation_id)

        state["response"] = response
        state.setdefault("messages", []).append(
            AIMessage(content=response.message.content)
        )

        return state

    async def _image_generator_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            logger.error("No messages in state for image generator agent")
            return state

        last_message = messages[-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        conversation_history = []
        memory_manager = get_memory_manager()

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )
            history_limit = (
                settings.chat_history_max_messages
                if settings.chat_history_max_messages > 0
                else None
            )
            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        persona = state.get("persona")
        context = state.get("context", {})
        attachments = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments,
        )

        response = await self.image_generator_agent.process_message(
            agent_msg, conversation_id
        )

        state["response"] = response
        state.setdefault("messages", []).append(
            AIMessage(content=response.message.content)
        )

        return state

    def _should_continue(self, state: GraphState) -> str:
        selected_agent = state.get("selected_agent")
        if selected_agent in self.agents:
            return selected_agent
        return "end"

    async def execute(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        persona: Optional[str] = None,
        attachments: Optional[list] = None,
    ) -> Optional[AgentResponse]:

        initial_state: GraphState = {
            "messages": [HumanMessage(content=message)],
            "context": {},
        }

        if conversation_id is not None:
            initial_state["conversation_id"] = conversation_id
        if user_id is not None:
            initial_state["user_id"] = user_id
        initial_state["selected_agent"] = None
        initial_state["response"] = None
        initial_state["persona"] = persona
        initial_state["reasoning_steps"] = None
        initial_state["tool_results"] = None
        initial_state["iteration_count"] = None

        # Store attachments in context for agent access
        if attachments:
            initial_state["context"]["attachments"] = attachments

        config = None
        if self.checkpointer and thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        result = await self.graph.ainvoke(initial_state, config=config)

        agent_response = result.get("response")
        if agent_response and isinstance(agent_response.metadata, dict):
            if "interrupt" in agent_response.metadata:
                logger.info("MultiAgentWorkflow detected interrupt in agent response")
                return agent_response

        return agent_response

    async def resume_execution(
        self,
        thread_id: str,
        resume_value: Any,
    ) -> Optional[AgentResponse]:
        """
        Resume execution after handling interrupts.

        Accepts either a fully-formed LangGraph resume payload or the legacy
        list of InterruptDecision instances for backward compatibility.
        """
        if not self.checkpointer:
            raise RuntimeError("Checkpointing must be enabled for resume_execution")

        config = {"configurable": {"thread_id": thread_id}}
        resume_payload = resume_value

        # Backward compatibility: allow passing InterruptDecision objects directly
        if (
            isinstance(resume_value, list)
            and resume_value
            and all(isinstance(item, InterruptDecision) for item in resume_value)
        ):
            decision_map = {}
            for decision in resume_value:
                task_id = decision.task_id
                if task_id:
                    decision_entry = {
                        "type": decision.type.value,
                    }
                    if decision.args:
                        decision_entry["args"] = decision.args
                    decision_map[task_id] = decision_entry
            resume_payload = decision_map

        command = Command(resume=resume_payload)
        result = await self.graph.ainvoke(command, config=config)

        return result.get("response")

    async def execute_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        persona: Optional[str] = None,
        attachments: Optional[list] = None,
    ):
        """
        Execute the workflow with streaming support.
        Yields token-level events by calling agent streaming methods directly.
        """
        initial_state: GraphState = {
            "messages": [HumanMessage(content=message)],
            "context": {},
        }

        if conversation_id is not None:
            initial_state["conversation_id"] = conversation_id
        if user_id is not None:
            initial_state["user_id"] = user_id
        initial_state["selected_agent"] = None
        initial_state["response"] = None
        initial_state["persona"] = persona
        initial_state["reasoning_steps"] = None
        initial_state["tool_results"] = None
        initial_state["iteration_count"] = None

        # Store attachments in context for agent access
        if attachments:
            initial_state["context"]["attachments"] = attachments

        config = None
        if self.checkpointer and thread_id:
            config = {"configurable": {"thread_id": thread_id}}

        # Route to determine which agent to use
        route_state = await self._route_node(initial_state)
        selected_agent_name = route_state.get("selected_agent")

        if not selected_agent_name:
            yield {
                "type": "error",
                "error": "No agent could be selected for this request",
            }
            return

        # Yield node start event
        yield {"type": "node", "node": selected_agent_name}

        # Get the selected agent and prepare agent message
        agent = self.agents.get(selected_agent_name)
        if not agent:
            yield {"type": "error", "error": f"Agent {selected_agent_name} not found"}
            return

        # Prepare conversation history and agent message
        messages = route_state.get("messages", [])
        last_message = messages[-1] if messages else None
        content = (
            last_message.content
            if last_message and hasattr(last_message, "content")
            else str(last_message) if last_message else message
        )

        conversation_history = []
        memory_manager = get_memory_manager()

        if conversation_id and user_id:
            conv_id_uuid = UUID(conversation_id)
            user_id_uuid = UUID(user_id)

            conv_memory = await memory_manager.get_memory(
                conv_id_uuid, user_id_uuid, force_refresh=True
            )

            # Apply history limits based on agent type
            history_limit = None
            if (
                selected_agent_name == "chat_agent"
                and settings.chat_history_max_messages > 0
            ):
                history_limit = settings.chat_history_max_messages
            elif (
                selected_agent_name == "rag_agent"
                and settings.rag_history_max_messages > 0
            ):
                history_limit = settings.rag_history_max_messages
            elif (
                selected_agent_name == "search_agent"
                and settings.search_history_max_messages > 0
            ):
                history_limit = settings.search_history_max_messages
            elif (
                selected_agent_name == "image_generator_agent"
                and settings.image_history_max_messages > 0
            ):
                history_limit = settings.image_history_max_messages

            conversation_history = conv_memory.get_recent_messages(
                limit=history_limit, exclude_last=1
            )

        context = route_state.get("context", {})
        attachments_from_context = context.get("attachments")

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history, "persona": persona},
            attachments=attachments_from_context,
        )

        # Stream from the agent
        final_response = None
        try:
            async for event in agent.stream_message(agent_msg, conversation_id):
                event_type = event.get("type")

                if event_type == "token":
                    # Forward token events directly
                    yield event
                elif event_type in ["tool_start", "tool_end"]:
                    # Forward tool events
                    yield event
                elif event_type == "complete":
                    # Store final response
                    final_response = event.get("response")
                elif event_type == "interrupt":
                    # Forward interrupt events
                    yield event
                    return  # Stop streaming when interrupted
        except Exception as e:
            logger.error(
                f"Error streaming from {selected_agent_name}: {e}", exc_info=True
            )
            yield {"type": "error", "error": f"Error: {e}"}
            return

        if final_response:
            # Check if response contains interrupt metadata
            if final_response.metadata and "interrupt" in final_response.metadata:
                yield {
                    "type": "interrupt",
                    "interrupt": final_response.metadata["interrupt"],
                }
                return  # Stop streaming, wait for resume

            yield {"type": "complete", "response": final_response}

        if final_response and self.checkpointer and thread_id:

            async def update_checkpoint():
                try:
                    final_state = route_state.copy()
                    final_state["response"] = final_response
                    final_state.setdefault("messages", []).append(
                        AIMessage(content=final_response.message.content)
                    )
                    await self.graph.ainvoke(final_state, config=config)
                except Exception as e:
                    logger.warning(f"Failed to update checkpoint: {e}")

            # Create background task for checkpoint update
            asyncio.create_task(update_checkpoint())

    async def cleanup(self):
        """Cleanup resources from agents that use MCP tools"""
        logger.info("Cleaning up MultiAgentWorkflow resources...")
        for agent in self._cleanup_agents:
            if hasattr(agent, "cleanup"):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.error(
                        f"Error cleaning up agent {agent.__class__.__name__}: {e}"
                    )
        logger.info("MultiAgentWorkflow cleanup completed")


def create_workflow(
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    document_repository: Optional["DocumentRepository"] = None,
) -> MultiAgentWorkflow:
    """
    Create multi-agent workflow with required shared dependencies.
    """
    return MultiAgentWorkflow(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        checkpointer=checkpointer,
        document_repository=document_repository,
    )
