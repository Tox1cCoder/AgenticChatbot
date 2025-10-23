import json
import logging
import base64
from typing import Optional, List, Dict, Any

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

        response_text = self._coerce_response_text(response_text)

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
        """Generate response with tool calling support."""
        try:
            llm_with_tools = self.langchain_model.bind_tools(
                self.tools, tool_config={"function_calling_config": {"mode": "AUTO"}}
            )

            messages = [HumanMessage(content=prompt)]
            tools_used: List[str] = []
            tool_artifacts: List[Dict[str, Any]] = []
            max_iterations = 5

            for _ in range(max_iterations):
                ai_message = await llm_with_tools.ainvoke(messages)
                messages.append(ai_message)

                if not ai_message.tool_calls:
                    response_text = self._coerce_response_text(ai_message.content)
                    break

                for tool_call in ai_message.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    tools_used.append(tool_name)

                    tool_output_text: str
                    for tool in self.tools:
                        if tool.name != tool_name:
                            continue
                        try:
                            response_payload = await tool.ainvoke(tool_args)
                            tool_output_text = self._format_tool_result(response_payload)
                            tool_artifacts.append(
                                {
                                    "tool": tool_name,
                                    "arguments": self._make_json_safe(tool_args),
                                    "output": tool_output_text,
                                }
                            )
                            logger.info("ChatAgent executed tool: %s", tool_name)
                        except Exception as exc:
                            tool_output_text = f"Error executing tool: {exc}"
                            tool_artifacts.append(
                                {
                                    "tool": tool_name,
                                    "arguments": self._make_json_safe(tool_args),
                                    "error": str(exc),
                                }
                            )
                            logger.error("Tool execution error: %s", exc)
                        break
                    else:
                        tool_output_text = f"Tool {tool_name} not found"
                        tool_artifacts.append(
                            {
                                "tool": tool_name,
                                "arguments": self._make_json_safe(tool_args),
                                "error": "Tool not found",
                            }
                        )

                    messages.append(
                        ToolMessage(content=tool_output_text, tool_call_id=tool_id)
                    )

            else:
                response_text = "Response completed but max iterations reached."

            return response_text, tools_used, tool_artifacts

        except Exception as exc:
            logger.error("Error in tool calling flow: %s", exc, exc_info=True)
            fallback = await self._generate(prompt)
            return fallback, [], []

    def _coerce_response_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if text_value:
                        parts.append(str(text_value))
                else:
                    parts.append(str(item))
            return "\n".join(filter(None, parts))
        if content is None:
            return ""
        return str(content)

    def _format_tool_result(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except TypeError:
                return str(value)
        return "" if value is None else str(value)

    def _make_json_safe(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            return self._make_json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return self._make_json_safe(value.dict())
        return str(value)

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
    def _coerce_response_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text_value = item.get("text") or item.get("content")
                    if text_value:
                        parts.append(str(text_value))
                else:
                    parts.append(str(item))
            return "\n".join(filter(None, parts))
        if content is None:
            return ""
        return str(content)

    def _format_tool_result(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except TypeError:
                return str(value)
        return "" if value is None else str(value)

    def _make_json_safe(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            return self._make_json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return self._make_json_safe(value.dict())
        return str(value)

