import logging
import base64
from typing import Optional, List, Dict, Any, AsyncIterator

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_chat_prompt
from ..utils import (
    coerce_response_text,
    extract_agent_execution_info,
    get_error_recovery_hint,
)
from ..hitl_config import build_interrupt_response
from ...core.config import settings
from ...core.exceptions.mcp import ServerNotFoundError
from ..mcp_integration import MCPManager

logger = logging.getLogger(__name__)


class ChatAgent:

    def __init__(self):
        self.model_name = "gemini-flash-latest"
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
            model=self.model_name, google_api_key=api_key, temperature=0.8
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
                    name
                    for name, status in server_status.items()
                    if status.get("enabled")
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
        """Process message and return response with potential tool calls."""
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

        try:
            if attachments:
                # Vision mode - no tools
                response_text = await self._generate_with_vision(prompt, attachments)

                return AgentResponse(
                    agent_type=AgentType.CHAT,
                    agent_id="chat_agent",
                    message=AgentMessage(
                        role=MessageRole.ASSISTANT,
                        content=coerce_response_text(response_text),
                    ),
                    metadata={
                        "model": self.model_name,
                        "conversation_id": conversation_id,
                        "context_messages": len(conversation_history),
                        "persona_used": persona,
                        "has_images": True,
                    },
                )
            else:
                # Initialize tools if not done yet
                if self.mcp_manager is None:
                    await self._init_tools()

                if self.tools and self.langchain_model:
                    # Use invoke_model to return tool calls without executing
                    return await self.invoke_model(message, conversation_id)
                else:
                    # No tools available, generate directly
                    response_text = await self._generate(prompt)

                    return AgentResponse(
                        agent_type=AgentType.CHAT,
                        agent_id="chat_agent",
                        message=AgentMessage(
                            role=MessageRole.ASSISTANT,
                            content=coerce_response_text(response_text),
                        ),
                        metadata={
                            "model": self.model_name,
                            "conversation_id": conversation_id,
                            "context_messages": len(conversation_history),
                            "persona_used": persona,
                        },
                    )
        except Exception as exc:
            logger.error(
                "Error while processing message in ChatAgent: %s", exc, exc_info=True
            )
            response_text = await self._handle_generation_error(prompt, exc)

            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=coerce_response_text(response_text),
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    async def invoke_model(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        # Initialize MCP tools if needed
        if self.mcp_manager is None:
            await self._init_tools()

        # Extract conversation history
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Check if this invocation includes tool results (post-tool-execution)
        has_tool_results = "Tool results:" in message.content

        # Build prompt
        prompt = build_chat_prompt(
            message.content,
            conversation_history,
            persona=persona,
        )

        # Configure tool calling
        # If we already have tool results, force the model to NOT call more tools
        tool_choice = (
            "none"
            if has_tool_results
            else (
                settings.tool_choice_mode
                if hasattr(settings, "tool_choice_mode")
                else "auto"
            )
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
            metadata = {
                "model": self.model_name,
                "conversation_id": conversation_id,
                "context_messages": len(conversation_history),
                "tools_available": len(self.tools),
                "persona_used": persona,
            }

            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=coerce_response_text(response.content),
                    tool_calls=tool_calls if tool_calls else None,
                ),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Error invoking chat agent model: {e}", exc_info=True)
            error_text = await self._handle_generation_error(prompt, e)
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=coerce_response_text(error_text),
                ),
                metadata={
                    "model": self.model_name,
                    "conversation_id": conversation_id,
                    "error": f"{type(e).__name__}: {e}",
                },
            )

    async def _generate(self, prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    async def _handle_generation_error(self, prompt: str, error: Exception) -> str:
        """Ask the base LLM to craft a user-facing reply that acknowledges an internal error."""
        error_message = f"{type(error).__name__}: {error}"
        recovery_hint = get_error_recovery_hint(error, "chat_agent", {})

        fallback_prompt = (
            f"{prompt}\n\n"
            "SYSTEM NOTE FOR ASSISTANT:\n"
            "You attempted to respond to the user but encountered a system error.\n"
            f"Error details: {error_message}\n"
            f"Recovery hint: {recovery_hint}\n\n"
            "Compose a concise, empathetic reply to the user acknowledging the issue and, if possible, suggesting a next step."
        )

        fallback_response = await self._generate(fallback_prompt)
        return coerce_response_text(fallback_response)

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
