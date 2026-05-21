import asyncio
import logging

from google import genai

from ...core.config import settings
from ..agent_config import build_gemini_generate_config
from ..prompts import ROUTER_SYSTEM_PROMPT
from ..schemas import AgentMessage
from ..skills_tool import get_available_skill_summaries
from ..text_normalization import tokenize_text
from ..time_context import build_runtime_time_context_block

logger = logging.getLogger(__name__)


class Router:
    def __init__(self):
        self.model_name = "gemini-3-flash-preview"
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key or ""
        if isinstance(api_key, str) and api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        if not api_key:
            logger.warning("Router Gemini client not initialized: missing GEMINI_API_KEY")
            self.gemini_client = None
            return

        try:
            self.gemini_client = genai.Client(api_key=api_key)
        except Exception as exc:
            logger.warning("Router Gemini client initialization failed: %s", exc, exc_info=True)
            self.gemini_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route_message(
        self,
        message: AgentMessage,
        available_agents: list[str],
        has_documents: bool = False,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
    ) -> str:
        """Route a user message to the most appropriate agent via LLM."""
        if not available_agents:
            return "chat_agent"

        content = (message.content or "").strip()

        # Planning mode + existing plan: planning_agent is the supervisor; the
        # router should never wander off to chat_agent just because the user
        # phrased a delegation request casually. The LLM router is unreliable
        # for this, so we short-circuit deterministically.
        if planning_mode_enabled and has_existing_plan and "planning_agent" in available_agents:
            return "planning_agent"

        metadata = message.metadata or {}
        persona = metadata.get("persona")
        request_user_id = metadata.get("user_id")
        request_device_id = metadata.get("device_id")

        prompt = self._build_prompt(
            content=content,
            persona=persona,
            available_agents=available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
            user_id=request_user_id,
            device_id=request_device_id,
        )

        selected_agent = await self._call_llm(prompt, available_agents)
        if selected_agent:
            return selected_agent

        logger.warning("Router LLM returned no recognisable agent name")
        return "chat_agent"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _call_llm(self, prompt: str, available_agents: list[str]) -> str | None:
        """Invoke Gemini and parse the response into an agent name."""
        if self.gemini_client is None:
            logger.warning("Router fallback: Gemini client unavailable, defaulting to chat_agent")
            return None

        try:
            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=build_gemini_generate_config(
                    model_name=self.model_name,
                    include_thinking=False,
                ),
            )
            response_text = response.text if hasattr(response, "text") else str(response)
            return self._extract_agent_name(response_text, available_agents)
        except Exception as exc:
            logger.warning("Router fallback: Gemini routing failed: %s", exc)
            return None

    def _build_prompt(
        self,
        content: str,
        persona: str | None,
        available_agents: list[str],
        has_documents: bool,
        planning_mode_enabled: bool,
        has_existing_plan: bool,
        user_id: str | None,
        device_id: str | None,
    ) -> str:
        prompt_parts: list[str] = []

        if persona is not None and persona.strip():
            prompt_parts.append(f"Custom Persona: {persona}\n")

        if has_documents:
            prompt_parts.append(
                "IMPORTANT: This conversation has uploaded documents. "
                "Unless the user's intent is clearly unrelated to the documents "
                "(e.g. greeting, image generation, planning, building an interactive app), "
                "you MUST route to rag_agent.\n"
            )

        if planning_mode_enabled:
            prompt_parts.append("CONTEXT: Planning mode is active for this conversation.\n")

        if has_existing_plan:
            prompt_parts.append("CONTEXT: This conversation has an existing task plan.\n")

        prompt_parts.append(f"Available agents for this request: {', '.join(available_agents)}")
        prompt_parts.append(build_runtime_time_context_block().strip())
        prompt_parts.append(ROUTER_SYSTEM_PROMPT)

        active_skills = get_available_skill_summaries(user_id=user_id, device_id=device_id)
        if active_skills:
            skills_context = "\n".join(
                (
                    f"- **{skill.get('lookup_name') or skill.get('name')}** "
                    f"[{skill.get('source') or 'server'}]: "
                    f"{skill.get('description') or ''}"
                )
                for skill in active_skills
            )
            prompt_parts.append(f"\nActive skills available for this request:\n{skills_context}")

        prompt_parts.append(f"\n\nUser message: {content}")
        return "\n".join(prompt_parts)

    @staticmethod
    def _extract_agent_name(response_text: str, available_agents: list[str]) -> str | None:
        if not response_text:
            return None

        normalized_lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        for line in normalized_lines:
            tokens = tokenize_text(line.replace("-", "_"), preserve_underscore=True)
            for token in tokens:
                if token in available_agents:
                    return token

        lower_text = response_text.lower()
        for agent in available_agents:
            if agent in lower_text:
                return agent

        return None
