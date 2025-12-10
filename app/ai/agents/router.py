import re
from typing import List, Optional

from google import genai
import asyncio

from ..schemas import AgentMessage
from ..prompts import ROUTER_SYSTEM_PROMPT
from ...core.config import settings


class Router:
    def __init__(self):
        self.model_name = "gemini-flash-latest"
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key
        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()
        self.gemini_client = genai.Client(api_key=api_key)

    async def route_message(
        self,
        message: AgentMessage,
        available_agents: List[str],
        has_documents: bool = False,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
    ) -> str:
        content = message.content.strip()
        persona = message.metadata.get("persona")

        prompt_parts = []

        if persona is not None and persona.strip():
            prompt_parts.append(f"Custom Persona: {persona}\n")

        if has_documents:
            prompt_parts.append(
                "CONTEXT: This conversation has uploaded documents available.\n"
            )

        if planning_mode_enabled:
            prompt_parts.append(
                "CONTEXT: Planning mode is active for this conversation.\n"
            )

        if has_existing_plan:
            prompt_parts.append(
                "CONTEXT: This conversation has an existing task plan.\n"
            )

        prompt_parts.append(ROUTER_SYSTEM_PROMPT)
        prompt_parts.append(f"\n\nUser message: {content}")

        prompt = "\n".join(prompt_parts)

        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=self.model_name,
            contents=prompt,
        )

        response_text = response.text if hasattr(response, "text") else str(response)
        selected_agent = self._extract_agent_name(response_text, available_agents)

        if selected_agent:
            return selected_agent

        if has_documents and "rag_agent" in available_agents:
            return "rag_agent"

        if planning_mode_enabled and "planning_agent" in available_agents:
            return "planning_agent"

        return "chat_agent" if "chat_agent" in available_agents else available_agents[0]

    def _extract_agent_name(
        self, response_text: str, available_agents: List[str]
    ) -> Optional[str]:
        if not response_text:
            return None

        normalized_lines = [
            line.strip() for line in response_text.splitlines() if line.strip()
        ]
        for line in normalized_lines:
            cleaned_line = re.sub(r"[^a-z0-9_]+", " ", line.lower())
            tokens = cleaned_line.replace("-", "_").split()
            for token in tokens:
                if token in available_agents:
                    return token

        lower_text = response_text.lower()
        for agent in available_agents:
            if agent in lower_text:
                return agent

        return None
