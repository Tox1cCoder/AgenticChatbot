import logging
import base64
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_chat_prompt, CHAT_SYSTEM_PROMPT
from ..utils import (
    coerce_response_text,
    get_error_recovery_hint,
)
from ...core.config import settings
from ..mcp_integration import get_global_mcp_manager

logger = logging.getLogger(__name__)


class ChatAgent:

    def __init__(self):
        # Use configurable model (gemini-2.5-flash for thinking support)
        self.model_name = settings.chat_agent_model
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

        # Build LangChain model with optional thinking support
        model_kwargs = {
            "model": self.model_name,
            "google_api_key": api_key,
            "temperature": 0.8,
        }
        if settings.enable_thinking and settings.thinking_budget > 0:
            model_kwargs["thinking_budget"] = settings.thinking_budget

        self.langchain_model = ChatGoogleGenerativeAI(**model_kwargs)

    async def _init_tools(self):
        """Initialize MCP manager and load general-purpose tools using global singleton"""
        if self.mcp_manager is not None:
            return  # Already initialized
        
        try:
            # Use global singleton MCP manager for performance
            self.mcp_manager = await get_global_mcp_manager()
            all_tools = await self.mcp_manager.get_tools()
        except Exception as e:
            logger.error(
                "Failed to get global MCP manager for ChatAgent: %s",
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
                "Loaded %d MCP tools for ChatAgent from %d servers",
                len(self.tools),
                len(active_servers),
            )
        else:
            logger.warning(
                "No MCP tools available for ChatAgent; running without tools"
            )

    def _deduplicate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Ensure the tool list does not contain duplicates by name."""
        unique_tools: Dict[str, BaseTool] = {}
        for tool in tools or []:
            unique_tools.setdefault(tool.name, tool)
        return list(unique_tools.values())

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

        # Configure tool calling based on global setting; allow the model to decide
        llm_with_tools = self._get_llm_with_tools()

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

    async def invoke_model_with_history(
        self,
        messages: List[BaseMessage],
        conversation_history: List[Any],
        persona: Optional[str],
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        if self.mcp_manager is None:
            await self._init_tools()

        llm_with_tools = self._get_llm_with_tools()

        system_prompt = CHAT_SYSTEM_PROMPT
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
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
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
        """Cleanup agent resources (MCP manager is shared and cleaned up globally)"""
        self.mcp_manager = None
        self.tools = []
        logger.debug("ChatAgent cleanup completed (MCP manager is shared)")
