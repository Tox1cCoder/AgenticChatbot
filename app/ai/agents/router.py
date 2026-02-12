import asyncio
import logging
import re
from typing import List, Optional

from google import genai

from ...core.config import settings
from ..agent_config import build_gemini_generate_config
from ..prompts import ROUTER_SYSTEM_PROMPT
from ..schemas import AgentMessage

logger = logging.getLogger(__name__)


class Router:
    _PLANNING_PATTERN = re.compile(
        r"\b(create|make|build|generate|show|list|add|remove|delete|edit|update|modify|reorder|prioritize)\b.*\b(plan|tasks?|todo|to-do|task list|todo list|to-do list|roadmap|milestones?|checklist)\b|\b(plan|task list|todo list|to-do list|roadmap|checklist|next steps)\b",
        re.IGNORECASE,
    )
    _IMAGE_PATTERN = re.compile(
        r"\b(draw|sketch|paint|illustrate|render|design|generate|create|make)\b.*\b(image|picture|photo|illustration|art|artwork|logo|icon|poster|banner)\b|\b(image|picture|photo)\s+of\b",
        re.IGNORECASE,
    )
    _SEARCH_PATTERN = re.compile(
        r"\b(search|look up|lookup|google|web|internet|online|news|headline|weather|forecast|stock price|crypto price|exchange rate|live score|current events)\b",
        re.IGNORECASE,
    )
    _TIME_SENSITIVE_PATTERN = re.compile(
        r"\b(latest|recent|today|this week|this month|this year|right now|at the moment)\b",
        re.IGNORECASE,
    )
    _DOCUMENT_PATTERN = re.compile(
        r"\b(document|documents|doc|file|files|pdf|report|paper|contract|agreement|attachment|attached|upload|uploaded|transcript|slides?)\b",
        re.IGNORECASE,
    )
    _GREETING_PATTERN = re.compile(
        r"^\s*(hi|hello|hey|yo|good morning|good afternoon|good evening)\b",
        re.IGNORECASE,
    )

    def __init__(self):
        self.model_name = "gemini-3-flash-preview"
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key or ""
        if isinstance(api_key, str) and api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        if not api_key:
            logger.warning(
                "Router Gemini client not initialized: missing GEMINI_API_KEY"
            )
            self.gemini_client = None
            return

        try:
            self.gemini_client = genai.Client(api_key=api_key)
        except Exception as exc:
            logger.warning(
                "Router Gemini client initialization failed: %s", exc, exc_info=True
            )
            self.gemini_client = None

    async def route_message(
        self,
        message: AgentMessage,
        available_agents: List[str],
        has_documents: bool = False,
        planning_mode_enabled: bool = False,
        has_existing_plan: bool = False,
    ) -> str:
        if not available_agents:
            return "chat_agent"

        content = (message.content or "").strip()
        metadata = message.metadata or {}
        persona = metadata.get("persona")

        # Deterministic routing first to ensure document-first behavior.
        deterministic_agent = self._deterministic_route(
            content=content,
            available_agents=available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
        )
        if deterministic_agent:
            return deterministic_agent

        if not self.gemini_client:
            return self._fallback_route(
                content=content,
                available_agents=available_agents,
                has_documents=has_documents,
                planning_mode_enabled=planning_mode_enabled,
                has_existing_plan=has_existing_plan,
            )

        prompt = self._build_prompt(
            content=content,
            persona=persona,
            available_agents=available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
        )

        selected_agent: Optional[str] = None
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
            selected_agent = self._extract_agent_name(response_text, available_agents)
        except Exception as exc:
            logger.warning("Router model call failed: %s", exc, exc_info=True)

        if selected_agent:
            if (
                has_documents
                and "rag_agent" in available_agents
                and selected_agent != "rag_agent"
                and self._should_force_rag(content, selected_agent)
            ):
                return "rag_agent"
            return selected_agent

        return self._fallback_route(
            content=content,
            available_agents=available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
        )

    def _build_prompt(
        self,
        content: str,
        persona: Optional[str],
        available_agents: List[str],
        has_documents: bool,
        planning_mode_enabled: bool,
        has_existing_plan: bool,
    ) -> str:
        prompt_parts: List[str] = []

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

        prompt_parts.append(
            f"Available agents for this request: {', '.join(available_agents)}"
        )
        prompt_parts.append(ROUTER_SYSTEM_PROMPT)
        prompt_parts.append(f"\n\nUser message: {content}")
        return "\n".join(prompt_parts)

    def _deterministic_route(
        self,
        content: str,
        available_agents: List[str],
        has_documents: bool,
        planning_mode_enabled: bool,
        has_existing_plan: bool,
    ) -> Optional[str]:
        if not available_agents:
            return None

        has_rag = "rag_agent" in available_agents
        has_planning = "planning_agent" in available_agents
        has_search = "search_agent" in available_agents
        has_image = "image_generator_agent" in available_agents
        has_chat = "chat_agent" in available_agents

        if has_documents and has_rag:
            if self._is_planning_intent(content) and has_planning:
                return "planning_agent"
            if self._is_image_generation_intent(content) and has_image:
                return "image_generator_agent"
            if (
                has_search
                and self._is_explicit_search_intent(content)
                and not self._mentions_documents(content)
            ):
                return "search_agent"
            if self._is_greeting(content) and has_chat:
                return "chat_agent"
            return "rag_agent"

        if (
            planning_mode_enabled
            and has_existing_plan
            and has_planning
            and self._is_planning_intent(content)
        ):
            return "planning_agent"

        if self._is_planning_intent(content) and has_planning:
            return "planning_agent"
        if self._is_image_generation_intent(content) and has_image:
            return "image_generator_agent"
        if self._is_explicit_search_intent(content) and has_search:
            return "search_agent"

        return None

    def _fallback_route(
        self,
        content: str,
        available_agents: List[str],
        has_documents: bool,
        planning_mode_enabled: bool,
        has_existing_plan: bool,
    ) -> str:
        if not available_agents:
            return "chat_agent"

        deterministic = self._deterministic_route(
            content=content,
            available_agents=available_agents,
            has_documents=has_documents,
            planning_mode_enabled=planning_mode_enabled,
            has_existing_plan=has_existing_plan,
        )
        if deterministic:
            return deterministic

        if "chat_agent" in available_agents:
            return "chat_agent"
        return available_agents[0]

    def _is_planning_intent(self, text: str) -> bool:
        return bool(self._PLANNING_PATTERN.search(text or ""))

    def _is_image_generation_intent(self, text: str) -> bool:
        return bool(self._IMAGE_PATTERN.search(text or ""))

    def _is_explicit_search_intent(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        if self._SEARCH_PATTERN.search(lowered):
            return True
        if self._TIME_SENSITIVE_PATTERN.search(lowered):
            return any(
                token in lowered
                for token in (
                    "news",
                    "price",
                    "weather",
                    "score",
                    "market",
                    "headline",
                    "event",
                    "rate",
                )
            )
        return False

    def _mentions_documents(self, text: str) -> bool:
        return bool(self._DOCUMENT_PATTERN.search(text or ""))

    def _is_greeting(self, text: str) -> bool:
        return bool(self._GREETING_PATTERN.search(text or ""))

    def _should_force_rag(self, content: str, selected_agent: str) -> bool:
        if selected_agent == "planning_agent" and self._is_planning_intent(content):
            return False
        if selected_agent == "image_generator_agent" and self._is_image_generation_intent(
            content
        ):
            return False
        if selected_agent == "search_agent":
            if self._is_explicit_search_intent(content) and not self._mentions_documents(
                content
            ):
                return False
        if self._is_greeting(content):
            return False
        return True

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
