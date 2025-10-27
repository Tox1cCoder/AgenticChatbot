import logging
import json
from typing import Optional, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt, SEARCH_SYSTEM_PROMPT
from ..utils import coerce_response_text, format_tool_result, make_json_safe
from ...core.config import settings
from ...core.exceptions.mcp import ServerNotFoundError
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class SearchAgent:

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.langchain_model = None
        self.mcp_manager = None
        self.tools = []
        self._init_gemini()

    def _init_gemini(self):
        api_key = settings.gemini_api_key
        if not api_key:
            logger.error("Gemini API key not configured")
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()
        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=api_key, temperature=0.0
        )

    async def _init_mcp(self):
        """Initialize MCP manager and load external tool suites"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

                combined_tools: Dict[str, BaseTool] = {}

                preferred_servers = ["tavily", "time"]
                missing_servers: List[str] = []

                for server_name in preferred_servers:
                    try:
                        server_tools = await self.mcp_manager.get_server_tools(
                            server_name
                        )
                    except ServerNotFoundError:
                        missing_servers.append(server_name)
                        logger.debug(
                            "Preferred MCP server '%s' not configured for SearchAgent",
                            server_name,
                        )
                        continue

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
        """Process a search query using ReAct pattern with tool calling"""

        if self.mcp_manager is None:
            await self._init_mcp()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Build prompt
        prompt = build_search_prompt(
            message.content, conversation_history, persona=persona
        )

        try:
            # Configure tool calling based on settings
            tool_choice = settings.tool_choice_mode if hasattr(settings, 'tool_choice_mode') else "auto"
            parallel_tool_calls = settings.enable_parallel_tool_calls if hasattr(settings, 'enable_parallel_tool_calls') else True
            max_iterations = settings.react_agent_max_iterations if hasattr(settings, 'react_agent_max_iterations') else 10
            
            # Bind tools to model for this request
            llm_with_tools = self.langchain_model.bind_tools(
                self.tools, 
                tool_config={
                    "function_calling_config": {
                        "mode": tool_choice.upper() if tool_choice in ["auto", "any", "none"] else "AUTO"
                    }
                },
                parallel_tool_calls=parallel_tool_calls
            )

            # Create initial message
            messages = [HumanMessage(content=prompt)]
            
            extracted_images = []
            tools_used: List[str] = []
            tool_artifacts: List[Dict[str, Any]] = []
            reasoning_steps: List[Dict[str, Any]] = []
            iteration_count = 0

            # ReAct Loop: Reasoning → Acting → Observation
            for iteration in range(max_iterations):
                iteration_count = iteration + 1
                
                # Reasoning step: Get agent's thought and potential actions
                ai_message = await llm_with_tools.ainvoke(messages)
                messages.append(ai_message)
                
                # Extract reasoning trace if present
                if hasattr(ai_message, 'content') and ai_message.content:
                    reasoning_steps.append({
                        "iteration": iteration_count,
                        "thought": coerce_response_text(ai_message.content),
                        "type": "reasoning"
                    })

                # Check if agent decided to finish (no tool calls)
                if not ai_message.tool_calls:
                    response_text = coerce_response_text(ai_message.content)
                    break

                # Acting step: Execute tool calls
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    tools_used.append(tool_name)
                    
                    reasoning_steps.append({
                        "iteration": iteration_count,
                        "action": tool_name,
                        "arguments": make_json_safe(tool_args),
                        "type": "action"
                    })

                    # Find and execute the tool
                    tool_result = None
                    tool_found = False
                    for tool in self.tools:
                        if tool.name == tool_name:
                            tool_found = True
                            try:
                                tool_result = await tool.ainvoke(tool_args)
                                tool_output_text = format_tool_result(tool_result)

                                # Extract images from Tavily response
                                if tool_name == "tavily_search":
                                    images = self._extract_images_from_tavily(tool_result)
                                    if images:
                                        extracted_images.extend(images)

                                tool_artifacts.append({
                                    "tool": tool_name,
                                    "arguments": make_json_safe(tool_args),
                                    "output": tool_output_text,
                                })
                                
                                reasoning_steps.append({
                                    "iteration": iteration_count,
                                    "observation": tool_output_text,
                                    "type": "observation"
                                })
                                
                                logger.info("SearchAgent executed tool: %s", tool_name)

                            except Exception as e:
                                tool_output_text = f"Error executing tool: {str(e)}"
                                tool_artifacts.append({
                                    "tool": tool_name,
                                    "arguments": make_json_safe(tool_args),
                                    "error": str(e),
                                })
                                reasoning_steps.append({
                                    "iteration": iteration_count,
                                    "observation": tool_output_text,
                                    "error": str(e),
                                    "type": "observation"
                                })
                                logger.error("Tool execution error: %s", e)
                            break

                    # Handle tool not found
                    if not tool_found:
                        tool_output_text = f"Tool {tool_name} not found"
                        tool_artifacts.append({
                            "tool": tool_name,
                            "arguments": make_json_safe(tool_args),
                            "error": "Tool not found",
                        })
                        reasoning_steps.append({
                            "iteration": iteration_count,
                            "observation": tool_output_text,
                            "error": "Tool not found",
                            "type": "observation"
                        })

                    # Observation step: Add tool result to message history
                    messages.append(
                        ToolMessage(content=tool_output_text, tool_call_id=tool_id)
                    )
            else:
                # Max iterations reached
                response_text = "Search completed but max iterations reached."
                reasoning_steps.append({
                    "iteration": iteration_count,
                    "note": "Maximum iterations reached",
                    "type": "termination"
                })

        except Exception as e:
            logger.error(f"Error invoking search agent: {e}", exc_info=True)
            response_text = "An error occurred while processing your search request."

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
        )

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
