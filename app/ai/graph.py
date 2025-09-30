from typing import Any, Dict, List, Optional, TypedDict, Literal
from datetime import datetime
import logging
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from .schemas import AgentMessage, AgentResponse, AgentType, MessageType, WorkflowConfig
from .agents.router import Router
from .agents.chat_agent import ChatAgent
from .agents.rag_agent import RAGAgent
from .memory import get_memory_manager

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for LangGraph workflow."""

    messages: List[AgentMessage]
    current_message: Optional[AgentMessage]
    response: Optional[AgentResponse]
    selected_agent: Optional[str]
    conversation_id: Optional[str]
    user_id: Optional[str]
    iteration_count: int
    start_time: datetime


class Workflow:
    """LangGraph workflow for multi-agent system."""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()

        # Initialize agents
        self.router = Router()
        self.chat_agent = ChatAgent()
        self.rag_agent = RAGAgent()

        # Available agents for routing
        self.agents = {"chat_agent": self.chat_agent, "rag_agent": self.rag_agent}

        # Initialize agents asynchronously
        self._agents_initialized = False

        # Statistics tracking
        self._execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "agent_usage": {"chat_agent": 0, "rag_agent": 0},
        }

        # Build the graph
        self.graph = self._build_graph()

        logger.info("Workflow initialized with streamlined agent system")

    async def _ensure_agents_initialized(self) -> None:
        """Ensure all agents are initialized."""
        if self._agents_initialized:
            return

        await self.chat_agent.initialize()
        await self.rag_agent.initialize()
        self._agents_initialized = True
        logger.info("All agents initialized successfully")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("route", self._route_node)
        workflow.add_node("chat_agent", self._chat_node)
        workflow.add_node("rag_agent", self._rag_node)
        workflow.add_node("finalize", self._finalize_node)

        # Set entry point
        workflow.set_entry_point("route")

        # Add conditional edges from router
        workflow.add_conditional_edges(
            "route",
            self._should_continue,
            {"chat_agent": "chat_agent", "rag_agent": "rag_agent", "end": END},
        )

        # Both agents go to finalize
        workflow.add_edge("chat_agent", "finalize")
        workflow.add_edge("rag_agent", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _route_node(self, state: GraphState) -> GraphState:
        """Route the message to appropriate agent."""

        current_message = state["current_message"]
        if not current_message:
            logger.error("No current message to route")
            return state

        # Route using router
        available_agents = list(self.agents.keys())
        selected_agent = await self.router.route_message(
            current_message, available_agents
        )

        state["selected_agent"] = selected_agent
        logger.info(f"Message routed to: {selected_agent}")

        return state

    async def _chat_node(self, state: GraphState) -> GraphState:
        """Execute chat agent."""

        current_message = state["current_message"]
        if not current_message:
            return state

        response = await self.chat_agent.process_message(
            current_message,
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
        )
        state["response"] = response
        logger.info("Chat agent processing completed")

        return state

    async def _rag_node(self, state: GraphState) -> GraphState:
        """Execute RAG agent."""

        current_message = state["current_message"]
        if not current_message:
            return state

        response = await self.rag_agent.process_message(
            current_message,
            conversation_id=state.get("conversation_id"),
            user_id=state.get("user_id"),
        )
        state["response"] = response
        logger.info("RAG agent processing completed")

        return state

    async def _finalize_node(self, state: GraphState) -> GraphState:
        """Finalize the response."""

        if state["response"]:
            # Add any final processing here if needed
            logger.info("Response finalized successfully")
        else:
            logger.warning("No response generated")

        return state

    def _should_continue(self, state: GraphState) -> str:
        """Determine which agent to route to."""

        selected_agent = state.get("selected_agent")

        if selected_agent in self.agents:
            return selected_agent

        return "end"

    async def execute(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[AgentResponse]:
        """Execute the workflow for a given message."""

        # Ensure agents are initialized
        await self._ensure_agents_initialized()

        # Update statistics
        self._execution_stats["total_executions"] += 1

        # Initialize state
        initial_state = GraphState(
            messages=[message],
            current_message=message,
            response=None,
            selected_agent=None,
            conversation_id=conversation_id,
            user_id=user_id,
            iteration_count=0,
            start_time=datetime.now(),
        )

        # Execute the graph
        result = await self.graph.ainvoke(initial_state)

        # Update statistics
        if result.get("response"):
            self._execution_stats["successful_executions"] += 1
            selected_agent = result.get("selected_agent")
            if selected_agent in self._execution_stats["agent_usage"]:
                self._execution_stats["agent_usage"][selected_agent] += 1
        else:
            self._execution_stats["failed_executions"] += 1

        # Return the response
        return result.get("response")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get workflow execution statistics."""
        return self._execution_stats.copy()

    async def cleanup(self) -> None:
        """Clean up workflow resources."""
        await self.chat_agent.cleanup()
        await self.rag_agent.cleanup()
        logger.info("Workflow cleanup completed")


# Factory function
def create_workflow(config: Optional[WorkflowConfig] = None) -> Workflow:
    """Create a workflow instance."""
    return Workflow(config)
