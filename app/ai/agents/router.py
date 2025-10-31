import logging
from typing import List

from google import genai

from ..schemas import AgentMessage
from ..prompts import ROUTER_SYSTEM_PROMPT
from ...core.config import settings

logger = logging.getLogger(__name__)


class Router:
    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)

    async def route_message(
        self, message: AgentMessage, available_agents: List[str], has_documents: bool = False
    ) -> str:
        """Route the message to the appropriate agent"""
        content = message.content.strip()
        persona = message.metadata.get("persona")

        # Build the routing prompt with document context
        prompt_parts = []
        
        if persona is not None and persona.strip():
            prompt_parts.append(f"Custom Persona: {persona}\n")
        
        if has_documents:
            prompt_parts.append("CONTEXT: This conversation has uploaded documents available.\n")
        
        prompt_parts.append(ROUTER_SYSTEM_PROMPT)
        prompt_parts.append(f"\n\nUser message: {content}")
        
        prompt = "\n".join(prompt_parts)

        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=prompt
        )

        response_text = response.text if hasattr(response, "text") else str(response)
        selected_agent = response_text.strip().lower()

        # Validate the selected agent is available
        if selected_agent in available_agents:
            logger.info(f"LLM routed to {selected_agent}: {content[:50]}...")
            return selected_agent

        fallback_agent = (
            "chat_agent" if "chat_agent" in available_agents else available_agents[0]
        )

        return fallback_agent
