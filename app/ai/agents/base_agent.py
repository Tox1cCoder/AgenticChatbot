import logging
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import TOOL_CONTEXT_SUFFIX
from ..utils import coerce_response_text
from ...core.config import settings
from ..mcp_integration import get_global_mcp_manager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents. Child classes must implement: agent_type, agent_id, _get_base_system_prompt()."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.chat_agent_model
        self.gemini_client: Optional[genai.Client] = None
        self.langchain_model: Optional[ChatGoogleGenerativeAI] = None
        self.mcp_manager = None
        self.tools: List[BaseTool] = []

        self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            gemini_api_key = settings.gemini_api_key
            if not gemini_api_key:
                logger.error("GEMINI_API_KEY is not set")
                raise ValueError("GEMINI_API_KEY is not set")

            if gemini_api_key.startswith("GEMINI_API_KEY="):
                api_key = gemini_api_key.split("=", 1)[1].strip()
            else:
                api_key = gemini_api_key

            self.gemini_client = genai.Client(api_key=api_key)

            model_kwargs = {
                "model": self.model_name,
                "google_api_key": api_key,
                "temperature": 1.0,
            }

            if settings.enable_thinking:
                model_kwargs["thinking"] = True

            self.langchain_model = ChatGoogleGenerativeAI(**model_kwargs)

            logger.info(
                f"Initialized Gemini client and LangChain model with model: {self.model_name}"
            )

        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            raise

    async def _init_tools(self) -> None:
        if self.mcp_manager is not None:
            return

        try:
            self.mcp_manager = await get_global_mcp_manager()

            all_tools = await self.mcp_manager.get_tools()

            self.tools = self._deduplicate_tools(all_tools)

            server_status = self.mcp_manager.get_servers_status()
            active_servers = [
                name for name, status in server_status.items() if status.get("enabled")
            ]
            logger.info(
                f"Initialized {len(self.tools)} unique tools from "
                f"{len(active_servers)} active MCP servers: {active_servers}"
            )

        except Exception as e:
            logger.error(f"Error initializing MCP tools: {e}")
            self.tools = []

    def _deduplicate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        unique_tools: Dict[str, BaseTool] = {}
        for tool in tools:
            unique_tools.setdefault(tool.name, tool)
        return list(unique_tools.values())

    def _get_llm_with_tools(self) -> ChatGoogleGenerativeAI:
        if not self.tools:
            return self.langchain_model

        tool_choice = (
            settings.tool_choice_mode
            if hasattr(settings, "tool_choice_mode")
            else "auto"
        )

        if tool_choice.lower() in ["auto", "any", "none"]:
            mode = tool_choice.upper()
        else:
            mode = "AUTO"

        return self.langchain_model.bind_tools(
            self.tools, tool_config={"function_calling_config": {"mode": mode}}
        )

    async def invoke_model_with_history(
        self,
        messages: List[BaseMessage],
        conversation_history: List[Any],
        persona: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        try:
            if self.mcp_manager is None:
                await self._init_tools()

            llm_with_tools = self._get_llm_with_tools()

            has_tool_context = any(
                isinstance(msg, ToolMessage)
                or (hasattr(msg, "tool_calls") and msg.tool_calls)
                or (
                    hasattr(msg, "additional_kwargs")
                    and msg.additional_kwargs.get("tool_calls")
                )
                for msg in messages
            )

            system_prompt = self._build_system_prompt(persona, has_tool_context)

            langchain_messages = [SystemMessage(content=system_prompt)]
            langchain_messages.extend(messages)

            response = await llm_with_tools.ainvoke(langchain_messages)

            tool_calls = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

            thinking = None
            if hasattr(response, "thinking") and response.thinking:
                thinking = response.thinking

            response_text = coerce_response_text(response.content)

            metadata = {
                "model": self.model_name,
                "conversation_id": conversation_id,
                "has_tool_calls": tool_calls is not None,
                "tool_count": len(self.tools),
            }

            if thinking:
                metadata["thinking"] = thinking

            agent_message = AgentMessage(
                role=MessageRole.ASSISTANT, content=response_text, tool_calls=tool_calls
            )

            return AgentResponse(
                agent_type=self.agent_type,
                agent_id=self.agent_id,
                message=agent_message,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error invoking model with history: {e}")
            return self._build_error_response(
                message="I encountered an error processing your request.",
                conversation_id=conversation_id,
                error=str(e),
            )

    def _build_system_prompt(
        self, persona: Optional[str], has_tool_context: bool
    ) -> str:
        system_prompt = self._get_base_system_prompt()

        if has_tool_context:
            system_prompt = f"{system_prompt}\n\n{TOOL_CONTEXT_SUFFIX}"

        if persona:
            system_prompt = (
                f"Custom Persona:\n{persona.strip()}\n\n---\n{system_prompt}"
            )

        return system_prompt

    @abstractmethod
    def _get_base_system_prompt(self) -> str:
        pass

    def _build_error_response(
        self, message: str, conversation_id: Optional[str], error: Optional[str] = None
    ) -> AgentResponse:
        return AgentResponse(
            agent_type=self.agent_type,
            agent_id=self.agent_id,
            message=AgentMessage(
                role=MessageRole.ASSISTANT, content=f"I'm sorry, but {message}"
            ),
            metadata={
                "model": self.model_name,
                "conversation_id": conversation_id,
                "error": error or message,
            },
            error=error or message,
        )

    async def cleanup(self) -> None:
        self.mcp_manager = None
        self.tools = []
        logger.debug(
            f"{self.__class__.__name__} cleanup completed (MCP manager is shared)"
        )

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        pass

    @property
    @abstractmethod
    def agent_id(self) -> str:
        pass
