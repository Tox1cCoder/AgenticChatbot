import logging
from typing import Optional
from uuid import UUID

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .schemas import GraphState, AgentMessage, AgentResponse, MessageRole, AgentType
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .agents.search_agent import SearchAgent
from .memory import get_memory_manager
from ..core.config import settings

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
    ):
        self.router = Router()
        self.chat_agent = ChatAgent()
        self.rag_agent = RAGAgent(
            settings=settings,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            collection_name=settings.qdrant_collection_name,
        )
        self.search_agent = SearchAgent()
        self.agents = {
            "chat_agent": self.chat_agent,
            "rag_agent": self.rag_agent,
            "search_agent": self.search_agent,
        }

        self.graph = self._build_graph()
        logger.info("Multi-agent workflow initialized")

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GraphState)

        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("search_agent", self._search_node)

        workflow.add_edge(START, "route")

        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {
                "chat_agent": "chat_agent",
                "rag_agent": "rag_agent",
                "search_agent": "search_agent",
                "end": END,
            },
        )

        workflow.add_edge("chat_agent", END)
        workflow.add_edge("rag_agent", END)
        workflow.add_edge("search_agent", END)

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

        agent_msg = AgentMessage(role=MessageRole.USER, content=content)
        selected_agent = await self.router.route_message(
            agent_msg, list(self.agents.keys())
        )

        state["selected_agent"] = selected_agent
        logger.info(f"Routed to: {selected_agent}")
        return state

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
            logger.info(
                f"Loaded {len(conversation_history)} messages from memory for conversation {conversation_id}"
            )

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history},
        )

        response = await self.chat_agent.process_message(
            agent_msg, conversation_id, user_id
        )

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
            logger.info(
                f"Loaded {len(conversation_history)} messages from memory for conversation {conversation_id}"
            )

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history},
        )

        response = await self.rag_agent.process_message(
            agent_msg, conversation_id, user_id
        )

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
            logger.info(
                f"Loaded {len(conversation_history)} messages from memory for conversation {conversation_id}"
            )

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=content,
            metadata={"history": conversation_history},
        )

        response = await self.search_agent.process_message(
            agent_msg, conversation_id, user_id
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

        result = await self.graph.ainvoke(initial_state)
        return result.get("response")


def create_workflow(
    qdrant_client: QdrantClient,
    embedding_model: SentenceTransformer,
) -> MultiAgentWorkflow:
    """
    Create multi-agent workflow with required shared dependencies.
    """
    return MultiAgentWorkflow(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
    )
