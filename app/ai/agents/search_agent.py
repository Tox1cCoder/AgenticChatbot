import logging
import json
from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt
from ..utils import coerce_response_text
from ...core.config import settings
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class SearchAgent:

    def __init__(self):
        self.model_name = "gemini-3-pro-preview" # "gemini-flash-latest"
        self.langchain_model = None
        self.mcp_manager = None
        self.tools = []
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()
        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=api_key, temperature=0.12
        )

    async def _init_mcp(self):
        """Initialize MCP manager and load external tool suites"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

            except Exception as e:
                logger.error(f"Failed to initialize MCP manager: {e}", exc_info=True)
                self.tools = []
                return

        try:
            all_tools = await self.mcp_manager.get_tools()
        except Exception as e:
            logger.error(f"Failed to load MCP tools for SearchAgent: {e}", exc_info=True)
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
            logger.warning("No MCP tools available for SearchAgent; running without tools")

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
        tool_choice = (
            settings.tool_choice_mode
            if hasattr(settings, "tool_choice_mode")
            else "auto"
        )

        llm_with_tools = self.langchain_model.bind_tools(
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

    # Legacy support / wrapper for graph compatibility if needed,
    # but the graph should ideally call invoke_model now.
    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Process a search query.
        NOTE: This now just calls invoke_model. The graph is responsible for handling tool calls.
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
                        token = chunk.content
                        if isinstance(token, list):
                            token = "".join(str(item) for item in token if item)
                        if token:
                            yield {"type": "token", "content": token}

                # We might need to handle tool_call_chunks if we want to stream tool calls too

        except Exception as e:
            logger.error(f"Error streaming search agent: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}

    def _create_agent_executor(self, tools: List[BaseTool], system_prompt: str):
        """Create agent executor with proper tool binding configuration."""
        if not self.langchain_model:
            raise RuntimeError("SearchAgent language model not initialized")

        # Configure tool calling based on settings
        tool_choice = (
            settings.tool_choice_mode
            if hasattr(settings, "tool_choice_mode")
            else "auto"
        )

        # Configure model with tool binding
        llm_with_tools = self.langchain_model.bind_tools(
            tools,
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

        # Configure human-in-the-loop middleware if enabled
        middleware = []
        agent = create_agent(
            model=llm_with_tools,
            tools=tools,
            system_prompt=system_prompt,
            middleware=middleware if middleware else None,
        )

        return agent

    def _extract_images_from_tavily(self, tool_result: Any) -> List[Dict[str, str]]:
        """Extract images from Tavily search results"""
        try:
            # Parse tool result as JSON if it's a string
            if isinstance(tool_result, str):
                result_data = json.loads(tool_result)
            else:
                result_data = tool_result

            # Extract images array from response
            images = result_data.get("images", [])

            # Return images with url and description
            return [
                {"url": img.get("url", ""), "description": img.get("description", "")}
                for img in images
                if img.get("url")
            ]
        except Exception as e:
            logger.error(f"Failed to extract images from Tavily response: {e}")
            return []

    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("Search Agent MCP manager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up Search Agent: {e}")
