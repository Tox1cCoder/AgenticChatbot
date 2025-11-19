import logging
import json
from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt, SEARCH_SYSTEM_PROMPT
from ..utils import (
    coerce_response_text,
    extract_agent_execution_info,
    get_error_recovery_hint,
)
from ..hitl_config import (
    get_hitl_middleware_config,
    should_enable_hitl,
    is_agent_response_interrupted,
    extract_interrupt_data_from_agent_response,
    build_interrupt_response,
)
from ...core.config import settings
from ...core.exceptions.mcp import ServerNotFoundError
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class SearchAgent:

    def __init__(self):
        self.model_name = "gemini-flash-latest"
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

                combined_tools: Dict[str, BaseTool] = {}

                preferred_servers = ["tavily", "time"]

                for server_name in preferred_servers:
                    server_tools = await self.mcp_manager.get_server_tools(server_name)

                    for tool in server_tools:
                        combined_tools[tool.name] = tool

                if not combined_tools:
                    for tool in await self.mcp_manager.get_tools():
                        combined_tools[tool.name] = tool

                self.tools = list(combined_tools.values())

            except Exception as e:
                logger.error(f"Failed to initialize MCP manager: {e}", exc_info=True)
                self.tools = []

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Process a search query with tool calling"""

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

        error_message: Optional[str] = None
        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(
                self.tools, SEARCH_SYSTEM_PROMPT
            )

            # Invoke agent with the user message
            agent_response = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )

            # Check for interrupts from HumanInTheLoopMiddleware
            if is_agent_response_interrupted(agent_response):
                logger.info(
                    "SearchAgent detected interrupt from HumanInTheLoopMiddleware"
                )
                interrupt_data = extract_interrupt_data_from_agent_response(
                    agent_response
                )
                interrupt_response = build_interrupt_response(
                    interrupt_data, conversation_id or "", conversation_id or ""
                )

                # Return minimal AgentResponse with interrupt metadata
                return AgentResponse(
                    agent_type=AgentType.SEARCH,
                    agent_id="search_agent",
                    message=AgentMessage(
                        role=MessageRole.ASSISTANT,
                        content="Tool execution requires approval",
                    ),
                    metadata={"interrupt": interrupt_response},
                )

            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)

            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]

            extracted_images = []
            for artifact in tool_artifacts:
                if artifact.get("tool") == "tavily_search" and artifact.get("output"):
                    images = self._extract_images_from_tavily(artifact["output"])
                    if images:
                        extracted_images.extend(images)

        except Exception as e:
            logger.error(f"Error invoking search agent: {e}", exc_info=True)
            error_message = f"{type(e).__name__}: {e}"
            recovery_hint = get_error_recovery_hint(e, "search_agent", {})
            tools_used = []
            tool_artifacts = [
                {
                    "tool": "search_agent",
                    "args": {},
                    "error": error_message,
                    "hint": recovery_hint,
                }
            ]
            extracted_images = []

        response_text = coerce_response_text(response_text)

        # Create response metadata
        search_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "tools_available": len(self.tools),
            "agent_type": "tool_calling",
            "persona_used": persona,
        }

        # Add tool usage metadata
        if tools_used:
            search_metadata["tools_used"] = tools_used
            search_metadata["tool_calls_count"] = len(tools_used)

        # Add images to metadata if any were extracted
        if extracted_images:
            search_metadata["images"] = extracted_images
        if tool_artifacts:
            search_metadata["tool_artifacts"] = tool_artifacts
            if not error_message:
                error_entries = [
                    art["error"] for art in tool_artifacts if art.get("error")
                ]
                if error_entries:
                    error_message = error_entries[0]
                    search_metadata["error"] = error_message
        elif error_message:
            search_metadata["error"] = error_message

        # Create response message
        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        return AgentResponse(
            agent_type=AgentType.SEARCH,
            agent_id="search_agent",
            message=response_message,
            metadata=search_metadata,
            tool_artifacts=tool_artifacts if tool_artifacts else None,
            error=error_message,
        )

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream message processing with token-by-token generation.
        Yields events as tokens are generated and tools are executed.
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

        accumulated_content = ""
        tools_used: List[str] = []
        tool_artifacts: List[Dict[str, Any]] = []
        error_message: Optional[str] = None
        extracted_images = []

        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(
                self.tools, SEARCH_SYSTEM_PROMPT
            )

            # Stream agent execution
            async for event in agent_executor.astream_events(
                {"messages": [HumanMessage(content=prompt)]}, version="v1"
            ):
                event_type = event.get("event")

                # Extract tokens from LLM events
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Handle both string and list content
                        token = chunk.content
                        if isinstance(token, list):
                            # If it's a list, join the string parts
                            token = "".join(str(item) for item in token if item)
                        if token:  # Only yield non-empty tokens
                            accumulated_content += token
                            yield {"type": "token", "content": token}

                # Track tool execution
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown_tool")
                    yield {"type": "tool_start", "name": tool_name}

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown_tool")
                    yield {"type": "tool_end", "name": tool_name}

            # Get final response for complete extraction
            agent_response = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )

            # Check for interrupts before extracting normal execution info
            if is_agent_response_interrupted(agent_response):
                logger.info("SearchAgent detected interrupt during streaming")
                interrupt_data = extract_interrupt_data_from_agent_response(
                    agent_response
                )
                interrupt_response = build_interrupt_response(
                    interrupt_data, conversation_id or "", conversation_id or ""
                )
                # Yield interrupt event
                yield {
                    "type": "interrupt",
                    "interrupt": interrupt_response,
                }
                # Also yield complete with interrupt in metadata for consistency
                yield {
                    "type": "complete",
                    "response": AgentResponse(
                        agent_type=AgentType.SEARCH,
                        agent_id="search_agent",
                        message=AgentMessage(
                            role=MessageRole.ASSISTANT,
                            content="Tool execution requires approval",
                        ),
                        metadata={"interrupt": interrupt_response},
                    ),
                }
                return

            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)

            accumulated_content = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]

            # Extract images from Tavily results
            for artifact in tool_artifacts:
                if artifact.get("tool") == "tavily_search" and artifact.get("output"):
                    images = self._extract_images_from_tavily(artifact["output"])
                    if images:
                        extracted_images.extend(images)

        except Exception as e:
            logger.error(f"Error streaming search agent: {e}", exc_info=True)
            error_message = f"{type(e).__name__}: {e}"
            recovery_hint = get_error_recovery_hint(e, "search_agent", {})
            tools_used = []
            tool_artifacts = [
                {
                    "tool": "search_agent",
                    "args": {},
                    "error": error_message,
                    "hint": recovery_hint,
                }
            ]
            extracted_images = []

        accumulated_content = coerce_response_text(accumulated_content)

        # Create response metadata
        search_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "tools_available": len(self.tools),
            "agent_type": "tool_calling",
            "persona_used": persona,
        }

        # Add tool usage metadata
        if tools_used:
            search_metadata["tools_used"] = tools_used
            search_metadata["tool_calls_count"] = len(tools_used)

        # Add images to metadata if any were extracted
        if extracted_images:
            search_metadata["images"] = extracted_images
        if tool_artifacts:
            search_metadata["tool_artifacts"] = tool_artifacts
            if not error_message:
                error_entries = [
                    art["error"] for art in tool_artifacts if art.get("error")
                ]
                if error_entries:
                    error_message = error_entries[0]
                    search_metadata["error"] = error_message
        elif error_message:
            search_metadata["error"] = error_message

        # Create response message
        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=accumulated_content
        )

        # Yield complete event
        yield {
            "type": "complete",
            "response": AgentResponse(
                agent_type=AgentType.SEARCH,
                agent_id="search_agent",
                message=response_message,
                metadata=search_metadata,
                tool_artifacts=tool_artifacts if tool_artifacts else None,
                error=error_message,
            ),
        }

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
        if should_enable_hitl():
            tool_names = [tool.name for tool in tools]
            interrupt_config = get_hitl_middleware_config(tool_names)
            if interrupt_config:
                hitl_middleware = HumanInTheLoopMiddleware(
                    interrupt_on=interrupt_config,
                    description_prefix="Search tool execution pending approval",
                )
                middleware.append(hitl_middleware)

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
