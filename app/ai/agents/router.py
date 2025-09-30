import logging
import re
from typing import List

from ..schemas import AgentMessage

logger = logging.getLogger(__name__)


class Router:

    RAG_KEYWORDS = [
        "search",
        "find",
        "lookup",
        "document",
        "file",
        "knowledge",
        "explain",
        "what is",
        "how to",
        "tell me about",
    ]

    QUESTION_PATTERNS = [
        r"what\s+is",
        r"how\s+to",
        r"where\s+is",
        r"when\s+did",
        r"why\s+does",
        r"who\s+is",
        r"tell\s+me\s+about",
    ]

    async def route_message(
        self, message: AgentMessage, available_agents: List[str]
    ) -> str:
        content = message.content.lower().strip()

        if self._should_use_rag(content) and "rag_agent" in available_agents:
            logger.info(f"Routing to RAG agent: {content[:50]}...")
            return "rag_agent"

        if "chat_agent" in available_agents:
            logger.info(f"Routing to Chat agent: {content[:50]}...")
            return "chat_agent"

        return available_agents[0] if available_agents else "chat_agent"

    def _should_use_rag(self, content: str) -> bool:
        for keyword in self.RAG_KEYWORDS:
            if keyword in content:
                return True

        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        if len(content.split()) > 15:
            return any(
                term in content
                for term in ["explain", "details", "information", "document"]
            )

        return False
