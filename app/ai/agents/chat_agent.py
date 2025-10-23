import logging
import base64
from typing import Optional, List, Dict

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import BaseTool

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_chat_prompt
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

                # Fallback: include all available tools to support dynamic expansion
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
            tools_used = []
        else:
            # Initialize tools if not done yet
            if self.mcp_manager is None:
                await self._init_tools()

            # For text-only messages, use tool calling flow if tools are available
            if self.tools:
                response_text, tools_used = await self._generate_with_tools(prompt)
            else:
                response_text = await self._generate(prompt)
                tools_used = []

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

        return AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=response_message,
            metadata=metadata,
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

    async def _generate_with_tools(self, prompt: str) -> tuple[str, List[str]]:
        """Generate response with tool calling support"""
        try:
            # Bind tools to model for this request
            llm_with_tools = self.langchain_model.bind_tools(
                self.tools, tool_config={"function_calling_config": {"mode": "AUTO"}}
            )

            # Create initial message
            messages = [HumanMessage(content=prompt)]
            tools_used = []

            # Agent loop: model -> tool calls -> model -> response
            max_iterations = 5

            for iteration in range(max_iterations):
                # Invoke model
                ai_message = await llm_with_tools.ainvoke(messages)
                messages.append(ai_message)

                # Check if model wants to use tools
                if not ai_message.tool_calls:
                    response_text = ai_message.content
                    break

                # Execute tool calls
                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    # Track tool usage
                    tools_used.append(tool_name)

                    # Find and execute the tool
                    tool_result = None
                    for tool in self.tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = await tool.ainvoke(tool_args)
                                logger.info(f"ChatAgent executed tool: {tool_name}")
                            except Exception as e:
                                tool_result = f"Error executing tool: {str(e)}"
                                logger.error(f"Tool execution error: {e}")
                            break

                    # Add tool result to messages
                    if tool_result is None:
                        tool_result = f"Tool {tool_name} not found"

                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_id)
                    )
            else:
                # Max iterations reached
                response_text = "Response completed but max iterations reached."

            return response_text, tools_used

        except Exception as e:
            logger.error(f"Error in tool calling flow: {e}", exc_info=True)
            # Fallback to non-tool generation
            return await self._generate(prompt), []

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
