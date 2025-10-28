import logging
import base64
import asyncio
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_chat_prompt
from ..utils import coerce_response_text, format_tool_result, make_json_safe, extract_agent_execution_info, execute_tools_concurrently
from ...core.config import settings
from ...core.exceptions.mcp import ServerNotFoundError
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class ChatAgent:

    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.gemini_client = None
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

        self.gemini_client = genai.Client(api_key=api_key)

        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=api_key, temperature=0.7
        )

    async def _init_tools(self):
        """Initialize MCP manager and load general-purpose tools"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()

                combined_tools: Dict[str, BaseTool] = {}

                preferred_servers = ["calculator", "time"]
                for server_name in preferred_servers:
                    try:
                        server_tools = await self.mcp_manager.get_server_tools(
                            server_name
                        )
                    except ServerNotFoundError:
                        logger.debug(
                            "Preferred MCP server '%s' not configured for ChatAgent",
                            server_name,
                        )
                        continue

                    for tool in server_tools:
                        combined_tools[tool.name] = tool

                for tool in await self.mcp_manager.get_tools():
                    combined_tools.setdefault(tool.name, tool)

                self.tools = list(combined_tools.values())

                server_status = self.mcp_manager.get_servers_status()
                active_servers = [
                    name for name, status in server_status.items() if status.get("enabled")
                ]
                logger.info(
                    "Loaded %d MCP tools for ChatAgent from %d servers",
                    len(self.tools),
                    len(active_servers),
                )

            except Exception as e:
                logger.error(
                    f"Failed to initialize MCP manager for ChatAgent: {e}",
                    exc_info=True,
                )
                self.tools = []

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:

        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        prompt = build_chat_prompt(
            message.content, conversation_history, persona=persona
        )

        attachments = (
            message.attachments
            if hasattr(message, "attachments") and message.attachments
            else None
        )

        if attachments:
            response_text = await self._generate_with_vision(prompt, attachments)
            tools_used: List[str] = []
            tool_artifacts: List[Dict[str, Any]] = []
        else:
            # Initialize tools if not done yet
            if self.mcp_manager is None:
                await self._init_tools()

            # For text-only messages, use tool calling flow if tools are available
            if self.tools:
                response_text, tools_used, tool_artifacts = await self._generate_with_tools(prompt)
            else:
                response_text = await self._generate(prompt)
                tools_used = []
                tool_artifacts = []

        response_text = coerce_response_text(response_text)

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        # Build metadata
        metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "persona_used": persona,
            "has_images": bool(attachments),
            "tools_available": len(self.tools),
        }

        # Add tool usage metadata if tools were used
        if tools_used:
            metadata["tools_used"] = tools_used
            metadata["tool_calls_count"] = len(tools_used)
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts

        return AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=response_message,
            metadata=metadata,
            tool_artifacts=tool_artifacts if tool_artifacts else None,
        )

    async def _generate(self, prompt: str) -> str:
        if not self.gemini_client:
            logger.error("Gemini client not initialized")
            return "Error: Gemini API not configured"

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response: {str(e)}"

    async def _generate_with_tools(self, prompt: str) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """Generate response with tool calling support using create_agent."""
        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(self.tools, prompt)
            
            # Invoke agent with the user message
            agent_response = await agent_executor.ainvoke({
                "messages": [HumanMessage(content=prompt)]
            })
            
            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)
            
            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]
            
            # Log parallel tool execution summary
            if tools_used:
                logger.info(f"ChatAgent executed {len(tools_used)} tool(s): {', '.join(tools_used)}")
            
            return response_text, tools_used, tool_artifacts

        except Exception as exc:
            logger.error("Error in tool calling flow: %s", exc, exc_info=True)
            raise

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
        
        # Create agent using LangChain's create_agent
        agent = create_agent(
            model=llm_with_tools,
            tools=tools,
            system_prompt=system_prompt
        )
        
        return agent

    async def _execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tool calls concurrently using asyncio.gather."""
        if not tool_calls or not settings.enable_parallel_tool_calls:
            # Fall back to sequential execution
            results = []
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
                            results.append({
                                "tool": tool_name,
                                "args": make_json_safe(tool_args),
                                "output": format_tool_result(result)
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
            
            return results
        
        # Execute tools concurrently
        parallel_results = await execute_tools_concurrently(tool_calls, self.tools)
        
        # Convert to tool artifacts format
        tool_artifacts = []
        for tool_name, tool_args, result, success in parallel_results:
            if success:
                tool_artifacts.append({
                    "tool": tool_name,
                    "args": make_json_safe(tool_args),
                    "output": format_tool_result(result)
                })
            else:
                tool_artifacts.append({
                    "tool": tool_name,
                    "args": make_json_safe(tool_args),
                    "error": str(result)
                })
        
        logger.info(f"Executed {len(tool_artifacts)} tools in parallel")
        return tool_artifacts
        
    async def _generate_with_vision(self, prompt: str, attachments: List[dict]) -> str:
        """Generate response with vision support using multimodal content"""
        parts = []

        parts.append(types.Part(text=prompt))

        # Add images from attachments
        for attachment in attachments:
            try:
                # Decode base64 image data
                image_data = base64.b64decode(attachment.get("data", ""))
                mime_type = attachment.get("mime", "image/jpeg")

                # Create image part from bytes
                parts.append(
                    types.Part.from_bytes(data=image_data, mime_type=mime_type)
                )
                logger.info(
                    f"Added image to vision request: {attachment.get('name', 'unknown')}"
                )
            except Exception as img_err:
                logger.error(f"Failed to process image attachment: {img_err}")

        # Generate response with multimodal content
        response = self.gemini_client.models.generate_content(
            model=self.model_name, contents=parts
        )
        return response.text if hasattr(response, "text") else str(response)

    async def cleanup(self):
        """Cleanup MCP resources"""
        if self.mcp_manager:
            try:
                await self.mcp_manager.cleanup()
                logger.info("ChatAgent MCP cleanup completed")
            except Exception as e:
                logger.error(f"Error cleaning up ChatAgent MCP resources: {e}")
