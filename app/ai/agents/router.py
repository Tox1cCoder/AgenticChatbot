import logging
from typing import List

from google import genai

from ..schemas import AgentMessage
from ..prompts import ROUTER_SYSTEM_PROMPT
from ...core.config import settings

logger = logging.getLogger(__name__)


class Router:
    """LLM-based router that intelligently selects the appropriate agent."""

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key
        if not api_key:
            logger.error("Gemini API key not configured")
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized for Router")

    async def route_message(
        self, message: AgentMessage, available_agents: List[str]
    ) -> str:
        """Route the message to the appropriate agent using LLM."""
        content = message.content.strip()

        try:
            # Build the routing prompt
            prompt = f"{ROUTER_SYSTEM_PROMPT}\n\nUser message: {content}"

            # Get LLM decision using Gemini
            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            response_text = (
                response.text if hasattr(response, "text") else str(response)
            )
            selected_agent = response_text.strip().lower()

            # Validate the selected agent is available
            if selected_agent in available_agents:
                logger.info(f"LLM routed to {selected_agent}: {content[:50]}...")
                return selected_agent

        except Exception as e:
            logger.error(f"Error in LLM routing: {e}, falling back to chat_agent")
            return (
                "chat_agent"
                if "chat_agent" in available_agents
                else available_agents[0]
            )
