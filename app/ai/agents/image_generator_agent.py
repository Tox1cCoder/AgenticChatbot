import base64
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

from google.genai import types
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    BaseMessage,
    ToolMessage,
)

from .base_agent import BaseAgent
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import TOOL_CONTEXT_SUFFIX
from ..agent_config import create_langchain_model, create_gemini_client, AGENT_CONFIG
from ...core.config import settings
from ..mcp_integration import get_global_mcp_manager
from ..utils import coerce_response_text

logger = logging.getLogger(__name__)


class ImageGeneratorAgent(BaseAgent):

    def __init__(self):
        self.default_aspect_ratio = settings.image_generator_default_aspect_ratio
        self.max_images = max(1, settings.image_generator_max_images)
        self.enabled = settings.enable_image_generation
        super().__init__(agent_config_key="image_generator")

    def _init_gemini(self) -> None:
        if not self.enabled:
            return
        
        # Use image generator model for gemini_client (for actual image generation)
        self.gemini_client = create_gemini_client()
        
        # Override LangChain model to use flash for tool calling (not image generation)
        self.langchain_model = create_langchain_model(
            agent_type="image_generator",
            model_override=AGENT_CONFIG["image_generator"]["langchain_model"],
            include_thinking=False,
        )

    @property
    def agent_type(self) -> AgentType:
        return AgentType.IMAGE_GENERATOR

    @property
    def agent_id(self) -> str:
        return "image_generator_agent"

    def _get_base_system_prompt(self) -> str:
        return self._get_system_prompt()

    async def invoke_model(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        if not self.enabled:
            return self._build_error_response(
                message="Image generation is currently disabled.",
                conversation_id=conversation_id,
            )

        # Initialize tools if not done yet
        if self.mcp_manager is None:
            await self._init_tools()

        # Check if this invocation includes tool results
        has_tool_results = "Tool results:" in message.content

        if has_tool_results:
            # Extract the enhanced prompt from the content
            enhanced_prompt = message.content

            # Generate the image directly
            images, narrative = await self._generate_images(
                enhanced_prompt, message.content
            )

            response_metadata = {
                "model": self.model_name,
                "conversation_id": conversation_id,
                "images": images,
                "tools_used": True,
            }

            return AgentResponse(
                agent_type=AgentType.IMAGE_GENERATOR,
                agent_id="image_generator_agent",
                message=AgentMessage(role=MessageRole.ASSISTANT, content=narrative),
                metadata=response_metadata,
            )

        # Configure tool calling (only if no tool results yet)
        llm_with_tools = self._get_llm_with_tools()

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": message.content},
        ]

        try:
            response = await llm_with_tools.ainvoke(messages)

            tool_calls = []
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = response.tool_calls
                # Return tool calls to the graph
                return AgentResponse(
                    agent_type=AgentType.IMAGE_GENERATOR,
                    agent_id="image_generator_agent",
                    message=AgentMessage(
                        role=MessageRole.ASSISTANT,
                        content=coerce_response_text(response.content),
                        tool_calls=tool_calls,
                    ),
                    metadata={"tools_available": len(self.tools)},
                )

            # If no tool calls, the content is the Enhanced Prompt.
            enhanced_prompt = coerce_response_text(response.content)

            # Now generate the image
            images, narrative = await self._generate_images(
                enhanced_prompt, message.content
            )

            response_metadata = {
                "model": self.model_name,
                "conversation_id": conversation_id,
                "images": images,
                "tools_available": len(self.tools),
            }

            return AgentResponse(
                agent_type=AgentType.IMAGE_GENERATOR,
                agent_id="image_generator_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=narrative or "Here is the image I created.",
                ),
                metadata=response_metadata,
            )

        except Exception as e:
            logger.error(f"Error in image generator agent: {e}", exc_info=True)
            return self._build_error_response(
                message=str(e), conversation_id=conversation_id
            )

    def _get_system_prompt(self) -> str:
        return """You are an expert image generation prompt engineer.
Your goal is to create a detailed, descriptive prompt for an image generator based on the user's request.
You have access to external tools to fetch real-time context (like weather, time, news) if relevant to the image.
If the user asks for "a picture of the current weather in NY", use the weather tool first.
Once you have sufficient information, output the FINAL detailed prompt for the image generator.
Do not output anything else, just the prompt."""

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        return await self.invoke_model(message, conversation_id)

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        result = await self.invoke_model(message, conversation_id)

        if result.error:
            yield {"type": "error", "error": result.error}
            return

        yield {"type": "complete", "response": result}

    async def _generate_images(
        self, prepared_prompt: str, original_prompt: str
    ) -> tuple[List[dict], str]:
        if not self.gemini_client:
            return [], ""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prepared_prompt)],
            )
        ]

        config_kwargs = {"response_modalities": ["IMAGE", "TEXT"]}
        image_config_cls = getattr(types, "ImageGenerationConfig", None)
        if image_config_cls is not None:
            config_kwargs["image_generation_config"] = image_config_cls(
                number_of_images=self.max_images,
                aspect_ratio=self.default_aspect_ratio,
            )

        generate_config = types.GenerateContentConfig(**config_kwargs)

        images: List[dict] = []
        narrative_parts: List[str] = []

        stream = self.gemini_client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=generate_config,
        )

        for chunk in stream:
            if not getattr(chunk, "candidates", None):
                continue

            candidate = chunk.candidates[0]
            if not candidate or not getattr(candidate, "content", None):
                continue

            for part in getattr(candidate.content, "parts", []) or []:
                inline_data = getattr(part, "inline_data", None)
                if inline_data and getattr(inline_data, "data", None):
                    encoded = self._encode_image(inline_data.data)
                    if encoded:
                        images.append(
                            {
                                "data": encoded,
                                "mime": inline_data.mime_type or "image/png",
                                "prompt": original_prompt,
                                "model": self.model_name,
                                "aspect_ratio": self.default_aspect_ratio,
                            }
                        )
                        if len(images) >= self.max_images:
                            break

                text_segment = getattr(part, "text", None)
                if text_segment:
                    narrative_parts.append(text_segment)

            if len(images) >= self.max_images:
                break

            top_level_text = getattr(candidate, "text", None)
            if top_level_text:
                narrative_parts.append(top_level_text)

        narrative = " ".join(segment.strip() for segment in narrative_parts if segment)
        return images, narrative.strip()

    @staticmethod
    def _encode_image(raw_data) -> Optional[str]:
        if raw_data is None:
            return None

        try:
            if isinstance(raw_data, (bytes, bytearray)):
                return base64.b64encode(raw_data).decode("utf-8")
            if isinstance(raw_data, str):
                return raw_data
            if isinstance(raw_data, memoryview):
                return base64.b64encode(raw_data.tobytes()).decode("utf-8")
            return base64.b64encode(bytes(raw_data)).decode("utf-8")
        except Exception as err:
            logger.error("Failed to encode image data: %s", err)
            return None

    async def cleanup(self):
        await super().cleanup()
