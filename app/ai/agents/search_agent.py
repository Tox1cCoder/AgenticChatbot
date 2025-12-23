import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt, SEARCH_SYSTEM_PROMPT, TOOL_CONTEXT_SUFFIX
from ..utils import coerce_response_text
from ...core.config import settings
from ..mcp_integration import get_global_mcp_manager

logger = logging.getLogger(__name__)


class SearchAgent:

    def __init__(self):
        # Use configurable model
        self.model_name = settings.search_agent_model
        self.langchain_model = None
        self.mcp_manager = None
        self.tools = []
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        # Build LangChain model (thinking_level not supported in LangChain yet)
        model_kwargs = {
            "model": self.model_name,
            "google_api_key": api_key,
            "temperature": 1.0,
        }
        # Note: thinking_level should be configured in direct genai.Client calls,
        # not in LangChain's ChatGoogleGenerativeAI

        self.langchain_model = ChatGoogleGenerativeAI(**model_kwargs)

    async def _init_mcp(self):
        """Initialize MCP manager and load external tool suites using global singleton"""
        if self.mcp_manager is not None:
            return  # Already initialized

        try:
            self.mcp_manager = await get_global_mcp_manager()
            all_tools = await self.mcp_manager.get_tools()
        except Exception as e:
            logger.error(
                "Failed to get global MCP manager for SearchAgent: %s",
                e,
                exc_info=True,
            )
            self.tools = []
            return

        self.tools = self._deduplicate_tools(all_tools)

        server_status = self.mcp_manager.get_servers_status()
        active_servers = [
            name for name, status in server_status.items() if status.get("enabled")
        ]
        if self.tools:
            logger.info(
                "Loaded %d MCP tools for SearchAgent from %d servers",
                len(self.tools),
                len(active_servers),
            )
        else:
            logger.warning(
                "No MCP tools available for SearchAgent; running without tools"
            )

    def _deduplicate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Ensure the tool list does not contain duplicate names."""
        unique_tools: Dict[str, BaseTool] = {}
        for tool in tools or []:
            unique_tools.setdefault(tool.name, tool)
        return list(unique_tools.values())

    async def invoke_model(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Invoke the model directly, potentially returning tool calls.
        This replaces the internal AgentExecutor loop.
        """
        # Initialize MCP tools if needed
        if self.mcp_manager is None:
            await self._init_mcp()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Check if this invocation includes tool results (post-tool-execution)
        # This happens when the graph routes back after tool execution
        has_tool_results = "Tool results:" in message.content

        # Build prompt
        prompt = build_search_prompt(
            message.content,
            conversation_history,
            persona=persona,
            has_tool_results=has_tool_results,
        )

        # Configure tool calling; allow follow-up tool planning when needed
        llm_with_tools = self._get_llm_with_tools()

        try:
            # Invoke model
            response = await llm_with_tools.ainvoke([HumanMessage(content=prompt)])

            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls

            # Create response metadata
            search_metadata = {
                "model": self.model_name,
                "conversation_id": conversation_id,
                "context_messages": len(conversation_history),
                "tools_available": len(self.tools),
                "agent_type": "tool_calling",
                "persona_used": persona,
            }

            return AgentResponse(
                agent_type=AgentType.SEARCH,
                agent_id="search_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=coerce_response_text(response.content),
                    tool_calls=tool_calls if tool_calls else None,
                ),
                metadata=search_metadata,
            )

        except Exception as e:
            logger.error(f"Error invoking search agent model: {e}", exc_info=True)
            return AgentResponse(
                agent_type=AgentType.SEARCH,
                agent_id="search_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="I encountered an error while processing your request.",
                ),
                error=str(e),
                metadata={"error": str(e)},
            )

    async def invoke_model_with_history(
        self,
        messages: List[BaseMessage],
        conversation_history: List[Any],
        persona: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        if self.mcp_manager is None:
            await self._init_mcp()

        llm_with_tools = self._get_llm_with_tools()

        # Check if there are already tool results in the message history
        has_tool_context = any(
            isinstance(m, ToolMessage) or (hasattr(m, "tool_calls") and m.tool_calls)
            for m in messages
        )

        system_prompt = SEARCH_SYSTEM_PROMPT
        if has_tool_context:
            system_prompt = system_prompt + TOOL_CONTEXT_SUFFIX
        if persona and persona.strip():
            system_prompt = (
                f"Custom Persona:\n{persona.strip()}\n\n---\n{system_prompt}"
            )

        langchain_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        langchain_messages.extend(messages)

        response = await llm_with_tools.ainvoke(langchain_messages)

        tool_calls = (
            response.tool_calls
            if hasattr(response, "tool_calls") and response.tool_calls
            else []
        )

        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=AgentMessage(
                role=MessageRole.ASSISTANT,
                content=coerce_response_text(response.content),
                tool_calls=tool_calls if tool_calls else None,
            ),
            metadata={
                "model": self.model_name,
                "conversation_id": conversation_id,
                "context_messages": len(conversation_history),
                "tools_available": len(self.tools),
                "agent_type": "tool_calling",
                "persona_used": persona,
            },
        )

    def _get_llm_with_tools(self):
        tool_choice = (
            settings.tool_choice_mode
            if hasattr(settings, "tool_choice_mode")
            else "auto"
        )
        return self.langchain_model.bind_tools(
            self.tools,
            tool_config={
                "function_calling_config": {
                    "mode": (
                        tool_choice.upper()
                        if tool_choice in ["auto", "any", "none"]
                        else "AUTO"
                    )
                }
            },
        )

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process a search query.
        """
        return await self.invoke_model(message, conversation_id)

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream message processing.
        """
        # Initialize MCP tools if needed
        if self.mcp_manager is None:
            await self._init_mcp()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Build prompt
        prompt = build_search_prompt(
            message.content, conversation_history, persona=persona
        )

        llm_with_tools = self.langchain_model.bind_tools(self.tools)

        try:
            async for event in llm_with_tools.astream_events(
                [HumanMessage(content=prompt)], version="v1"
            ):
                event_type = event.get("event")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = coerce_response_text(chunk.content)
                        if token:
                            yield {"type": "token", "content": token}

        except Exception as e:
            logger.error(f"Error streaming search agent: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    async def cleanup(self):
        """Cleanup agent resources"""
        self.mcp_manager = None
        self.tools = []
        logger.debug("SearchAgent cleanup completed (MCP manager is shared)")
