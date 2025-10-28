import logging
import json
import asyncio
from typing import Optional, List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_search_prompt, SEARCH_SYSTEM_PROMPT
from ..utils import format_tool_result, make_json_safe, extract_agent_execution_info, execute_tools_concurrently
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
        """Process a search query using create_agent with tool calling"""

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

        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(self.tools, SEARCH_SYSTEM_PROMPT)
            
            # Invoke agent with the user message
            agent_response = await agent_executor.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            })
            
            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)
            
            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]
            
            # Extract images from Tavily results
            extracted_images = []
            for artifact in tool_artifacts:
                if artifact.get("tool") == "tavily_search" and artifact.get("output"):
                    images = self._extract_images_from_tavily(artifact["output"])
                    if images:
                        extracted_images.extend(images)
            
            # Log parallel tool execution summary
            if tools_used:
                logger.info(f"SearchAgent executed {len(tools_used)} tool(s): {', '.join(tools_used)}")

        except Exception as e:
            logger.error(f"Error invoking search agent: {e}", exc_info=True)
            response_text = "An error occurred while processing your search request."
            tools_used = []
            tool_artifacts = []
            extracted_images = []

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

    def _create_agent_executor(self, tools: List[BaseTool], system_prompt: str):
        """Create agent executor with proper tool binding configuration."""
        # Configure tool calling based on settings
        tool_choice = settings.tool_choice_mode if hasattr(settings, 'tool_choice_mode') else "auto"
        
        # Configure model with tool binding
        llm_with_tools = self.langchain_model.bind_tools(
            tools, 
            tool_config={
                "function_calling_config": {
                    "mode": tool_choice.upper() if tool_choice in ["auto", "any", "none"] else "AUTO"
                }
            }
        )
        
        agent = create_agent(
            model=llm_with_tools,
            tools=tools,
            system_prompt=system_prompt
        )
        
        return agent

    async def _execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
        """Execute multiple tool calls concurrently and extract images from  results."""
        if not tool_calls or not settings.enable_parallel_tool_calls:
            # Fall back to sequential execution
            results = []
            images = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                # Find and execute tool
                tool_found = False
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_found = True
                        try:
                            result = await tool.ainvoke(tool_args)
                            formatted_result = format_tool_result(result)
                            
                            # Extract images
                            if tool_name == "tavily_search":
                                tavily_images = self._extract_images_from_tavily(result)
                                if tavily_images:
                                    images.extend(tavily_images)
                            
                            results.append({
                                "tool": tool_name,
                                "args": make_json_safe(tool_args),
                                "output": formatted_result
                            })
                        except Exception as e:
                            results.append({
                                "tool": tool_name,
                                "args": make_json_safe(tool_args),
                                "error": str(e)
                            })
                        break
                
                if not tool_found:
                    results.append({
                        "tool": tool_name,
                        "args": make_json_safe(tool_args),
                        "error": "Tool not found"
                    })
            
            return results, images
        
        # Execute tools concurrently
        parallel_results = await execute_tools_concurrently(tool_calls, self.tools)
        
        # Convert to tool artifacts format and extract images
        tool_artifacts = []
        extracted_images = []
        for tool_name, tool_args, result, success in parallel_results:
            if success:
                formatted_result = format_tool_result(result)
                
                # Extract images
                if tool_name == "tavily_search":
                    tavily_images = self._extract_images_from_tavily(result)
                    if tavily_images:
                        extracted_images.extend(tavily_images)
                
                tool_artifacts.append({
                    "tool": tool_name,
                    "args": make_json_safe(tool_args),
                    "output": formatted_result
                })
            else:
                tool_artifacts.append({
                    "tool": tool_name,
                    "args": make_json_safe(tool_args),
                    "error": str(result)
                })
        
        logger.info(f"Executed {len(tool_artifacts)} tools in parallel")
        return tool_artifacts, extracted_images

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
