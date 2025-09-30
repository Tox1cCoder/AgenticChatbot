from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import re

from ..schemas import AgentMessage, AgentType, MessageType, AgentRequest
from ..interfaces import IAgentRouter, IAgent

logger = logging.getLogger(__name__)


class Router(IAgentRouter):
    """Router that uses pattern matching instead of scoring."""

    # Keyword patterns for routing
    RAG_KEYWORDS = [
        "search",
        "find",
        "lookup",
        "information",
        "data",
        "document",
        "file",
        "knowledge",
        "research",
        "explain",
        "definition",
        "what is",
    ]

    QUESTION_PATTERNS = [
        r"what\s+is",
        r"how\s+to",
        r"where\s+is",
        r"when\s+did",
        r"why\s+does",
        r"who\s+is",
        r"which\s+one",
    ]

    def __init__(self):
        self._logger = logging.getLogger("router")
        self.registered_agents: Dict[str, IAgent] = {}

    async def route_request(self, request: AgentRequest) -> AgentType:
        """Route a request to the most appropriate agent."""
        content = request.message.content.lower().strip() if request.message else ""

        # Check for RAG patterns first
        if self._should_use_rag(content):
            return AgentType.RAG

        # Default to chat agent
        return AgentType.CHAT

    async def get_agent_scores(self, request: AgentRequest) -> Dict[AgentType, float]:
        """Get confidence scores for all available agents."""
        content = request.message.content.lower().strip() if request.message else ""

        scores = {}

        # Score RAG agent
        if self._should_use_rag(content):
            scores[AgentType.RAG] = 0.8
            scores[AgentType.CHAT] = 0.2
        else:
            scores[AgentType.RAG] = 0.2
            scores[AgentType.CHAT] = 0.8

        return scores

    def register_agent(self, agent: IAgent) -> None:
        """Register an agent with the router."""
        self.registered_agents[agent.agent_id] = agent
        self._logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the router."""
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
            self._logger.info(f"Unregistered agent: {agent_id}")

    async def route_message(
        self,
        message: AgentMessage,
        available_agents: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Route message to appropriate agent using simple pattern matching."""

        content = message.content.lower().strip()

        # Check for RAG patterns first
        if self._should_use_rag(content):
            if "rag_agent" in available_agents:
                self._logger.info(
                    f"Routing to RAG agent for: {message.content[:50]}..."
                )
                return "rag_agent"

        # Default to chat agent
        if "chat_agent" in available_agents:
            self._logger.info(f"Routing to Chat agent for: {message.content[:50]}...")
            return "chat_agent"

        # Fallback to first available agent
        if available_agents:
            agent = available_agents[0]
            self._logger.warning(f"Using fallback agent {agent}")
            return agent

        raise ValueError("No available agents for routing")

    def _should_use_rag(self, content: str) -> bool:
        """Heuristic to determine if RAG agent should be used."""

        # Check for RAG keywords
        for keyword in self.RAG_KEYWORDS:
            if keyword in content:
                return True

        # Check for question patterns
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        # Long messages with specific terms
        if len(content.split()) > 15 and any(
            term in content for term in ["explain", "details", "information"]
        ):
            return True

        return False


# Factory function for router
def create_router() -> Router:
    """Create a router instance."""
    return Router()
