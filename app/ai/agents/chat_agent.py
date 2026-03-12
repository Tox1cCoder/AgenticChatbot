import logging
import base64
from typing import Optional, List

from google.genai import types
from langchain_core.messages import HumanMessage

from .base_agent import BaseAgent
from ..agent_config import build_gemini_generate_config
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_chat_prompt, CHAT_SYSTEM_PROMPT
from ..utils import coerce_response_text

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):

    def __init__(self):
        super().__init__(agent_config_key="chat")

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CHAT

    @property
    def agent_id(self) -> str:
        return "chat_agent"

    def _get_base_system_prompt(self) -> str:
        return CHAT_SYSTEM_PROMPT

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")
        message_content = message.content or ""

        prompt = build_chat_prompt(
            message_content, conversation_history, persona=persona
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
            return self._build_error_response(
                message="I encountered an error processing your request.",
                conversation_id=conversation_id,
                error=f"{type(exc).__name__}: {exc}",
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
        message_content = message.content or ""

        # Build prompt
        prompt = build_chat_prompt(
            message_content,
            conversation_history,
            persona=persona,
        )

        # Configure tool calling based on global setting; allow the model to decide
        llm_with_tools = self._get_llm_with_tools(conversation_id=conversation_id)

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

    async def _generate(self, prompt: str) -> str:
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized")

        try:
            system_prompt = self._get_full_system_prompt()
            generation_config = build_gemini_generate_config(
                model_name=self.model_name,
                include_thinking=True,
                system_instruction=system_prompt,
            )
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=generation_config,
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    async def _generate_with_vision(self, prompt: str, attachments: List[dict]) -> str:
        """Generate response with vision support using multimodal content"""
        parts = []

        parts.append(types.Part(text=prompt))

        # Add images from attachments
        for attachment in attachments:
            try:
                raw_data = attachment.get("data", "")
                if isinstance(raw_data, str) and raw_data.startswith("data:"):
                    # Support data URLs: data:image/png;base64,<payload>
                    header, _, payload = raw_data.partition(",")
                    raw_data = payload or ""
                    if not attachment.get("mime") and ";" in header:
                        inferred_mime = header[5:].split(";", 1)[0].strip()
                        if inferred_mime:
                            attachment["mime"] = inferred_mime

                # Decode base64 image data
                image_data = base64.b64decode(raw_data)
                mime_type = attachment.get("mime", "image/jpeg")

                # Create image part from bytes
                parts.append(
                    types.Part.from_bytes(data=image_data, mime_type=mime_type)
                )
            except Exception as img_err:
                logger.error(f"Failed to process image attachment: {img_err}")

        # Generate response with multimodal content
        system_prompt = self._get_full_system_prompt()
        generation_config = build_gemini_generate_config(
            model_name=self.model_name,
            include_thinking=True,
            system_instruction=system_prompt,
        )
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=generation_config,
        )
        return response.text if hasattr(response, "text") else str(response)

    async def cleanup(self):
        await super().cleanup()
