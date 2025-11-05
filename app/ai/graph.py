import logging
from typing import Optional, TYPE_CHECKING
from uuid import UUID

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .schemas import GraphState, AgentMessage, AgentResponse, MessageRole
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent
from .agents.image_generator_agent import ImageGeneratorAgent
from .memory import get_memory_manager
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
        Yields events for each node execution and state update.
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

        # Use astream for streaming execution
        final_response = None
        async for event in self.graph.astream(initial_state, config=config):
            # Event is a dict with node name as key and state as value
            for node_name, node_state in event.items():
                logger.debug(f"Stream event from node: {node_name}")

                # Yield node execution event
                yield {"type": "node", "node": node_name, "state": node_state}

                # Check if we have a response
                if isinstance(node_state, dict) and "response" in node_state:
                    response = node_state.get("response")
                    if response:
                        final_response = response

                        # Yield response content as tokens (for display purposes)
                        if hasattr(response, "message") and hasattr(
                            response.message, "content"
                        ):
                            yield {
                                "type": "content",
                                "content": response.message.content,
                            }

                        # Yield tool execution info if available
                        if hasattr(response, "metadata") and response.metadata:
                            if "tool_artifacts" in response.metadata:
                                yield {
                                    "type": "tool_artifacts",
                                    "artifacts": response.metadata["tool_artifacts"],
                                }

        # Yield final complete event
        yield {"type": "complete", "response": final_response}

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
