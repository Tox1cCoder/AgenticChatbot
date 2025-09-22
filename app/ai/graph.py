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

        # Build the graph
        self.graph = self._build_graph()

        logger.info("Workflow initialized with streamlined agent system")

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

        try:
            # Route using router
            available_agents = list(self.agents.keys())
            selected_agent = await self.router.route_message(
                current_message, available_agents
            )

            state["selected_agent"] = selected_agent
            logger.info(f"Message routed to: {selected_agent}")

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            state["selected_agent"] = "chat_agent"  # Default fallback

        return state

    async def _chat_node(self, state: GraphState) -> GraphState:
        """Execute chat agent."""

        current_message = state["current_message"]
        if not current_message:
            return state

        try:
            response = await self.chat_agent.process_message(
                current_message,
                conversation_id=state.get("conversation_id"),
                user_id=state.get("user_id"),
            )
            state["response"] = response
            logger.info("Chat agent processing completed")

        except Exception as e:
            logger.error(f"Chat agent failed: {e}")
            # Create fallback response
            state["response"] = AgentResponse(
                content="I apologize, but I encountered an error. Please try again.",
                agent_id="chat_agent",
                response_type="error",
            )

        return state

    async def _rag_node(self, state: GraphState) -> GraphState:
        """Execute RAG agent."""

        current_message = state["current_message"]
        if not current_message:
            return state

        try:
            response = await self.rag_agent.process_message(
                current_message,
                conversation_id=state.get("conversation_id"),
                user_id=state.get("user_id"),
            )
            state["response"] = response
            logger.info("RAG agent processing completed")

        except Exception as e:
            logger.error(f"RAG agent failed: {e}")
            # Create fallback response
            state["response"] = AgentResponse(
                content="I apologize, but I couldn't find the information you're looking for. Please try rephrasing your question.",
                agent_id="rag_agent",
                response_type="error",
            )

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

        try:
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

            # Return the response
            return result.get("response")

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return AgentResponse(
                content="I apologize, but I encountered an error while processing your request.",
                agent_id="system",
                response_type="error",
            )


# Factory function
def create_workflow(config: Optional[WorkflowConfig] = None) -> Workflow:
    """Create a workflow instance."""
    return Workflow(config)
