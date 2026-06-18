import base64
import logging
from collections.abc import AsyncIterator
from typing import Any

from google.genai import types
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage as LCHumanMessage

from ...core.config import settings
from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ..agent_config import AGENT_CONFIG, create_gemini_client, create_langchain_model
from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..utils import coerce_response_text
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ImageGeneratorAgent(BaseAgent):
    def __init__(self, runtime_model_resolver: IRuntimeModelResolver | None = None):
        self.default_aspect_ratio = settings.image_generator_default_aspect_ratio
        self.max_images = max(1, settings.image_generator_max_images)
        self.enabled = settings.enable_image_generation
        super().__init__(
            agent_config_key="image_generator",
            runtime_model_resolver=runtime_model_resolver,
        )

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

    def _should_harvest_inline_images(self) -> bool:
        return True

    def _harvest_inline_images(
        self, response: AgentResponse, original_prompt: str
    ) -> list[dict[str, Any]]:
        """Wrap images the base agent surfaced from the model into image records.

        Returns records in the same shape ``_generate_images`` produces so the
        rest of the pipeline (metadata, persistence, rendering) is unchanged.
        """
        raw = (response.metadata or {}).get("response_inline_images") or []
        images: list[dict[str, Any]] = []
        for item in raw:
            data = item.get("data") if isinstance(item, dict) else None
            if not data:
                continue
            images.append(
                {
                    "data": data,
                    "mime": item.get("mime") or "image/png",
                    "prompt": original_prompt,
                    "model": self.model_name,
                    "aspect_ratio": self.default_aspect_ratio,
                }
            )
            if len(images) >= self.max_images:
                break
        return images

    async def invoke_model(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        if not self.enabled:
            return self._build_error_response(
                message="Image generation is currently disabled.",
                conversation_id=conversation_id,
            )

        # Initialize tools if not done yet
        if self.mcp_manager is None:
            await self._init_tools()

        message_content = message.content or ""
        request_user_id = message.metadata.get("user_id")
        request_device_id = message.metadata.get("device_id")

        # Check if this invocation includes tool results
        has_tool_results = "Tool results:" in message_content

        if has_tool_results:
            # Extract the enhanced prompt from the content
            enhanced_prompt = message_content

            # Generate the image directly
            images, narrative = await self._generate_images(enhanced_prompt, message_content)

            # Generate a natural user-facing response instead of showing the enhanced prompt
            user_facing = narrative or await self._generate_user_facing_response(message_content)

            response_metadata = {
                "model": self.model_name,
                "conversation_id": conversation_id,
                "images": images,
                "tools_used": True,
            }

            return AgentResponse(
                agent_type=AgentType.IMAGE_GENERATOR,
                agent_id="image_generator_agent",
                message=AgentMessage(role=MessageRole.ASSISTANT, content=user_facing),
                metadata=response_metadata,
            )

        # Configure tool calling (only if no tool results yet)
        llm_with_tools = self._get_llm_with_tools(
            conversation_id=conversation_id,
            user_id=request_user_id,
            device_id=request_device_id,
        )

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": message_content},
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
                    metadata={
                        "model": self.model_name,
                        "conversation_id": conversation_id,
                    },
                )

            # If no tool calls, the content is the Enhanced Prompt.
            enhanced_prompt = coerce_response_text(response.content)

            # Now generate the image
            images, narrative = await self._generate_images(enhanced_prompt, message_content)

            # Generate a natural user-facing response instead of showing the enhanced prompt
            user_facing = narrative or await self._generate_user_facing_response(message_content)

            response_metadata = {
                "model": self.model_name,
                "conversation_id": conversation_id,
                "images": images,
            }

            return AgentResponse(
                agent_type=AgentType.IMAGE_GENERATOR,
                agent_id="image_generator_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content=user_facing,
                ),
                metadata=response_metadata,
            )

        except Exception as e:
            logger.error(f"Error in image generator agent: {e}", exc_info=True)
            return self._build_error_response(message=str(e), conversation_id=conversation_id)

    def _get_system_prompt(self) -> str:
        return """You are an expert image generation prompt engineer.
Your goal is to create a detailed, descriptive prompt for an image generator based on the user's request.
You have access to external tools to fetch real-time context (like weather, time, news) if relevant to the image.
If the user asks for "a picture of the current weather in NY", use the weather tool first.
Once you have sufficient information, output the FINAL detailed prompt for the image generator.
Do not output anything else, just the prompt."""

    async def invoke_model_with_history(
        self,
        messages: list[BaseMessage],
        conversation_history: list[Any],
        persona: str | None,
        conversation_id: str | None = None,
        user_id: str | None = None,
        device_id: str | None = None,
        model_request: dict[str, Any] | None = None,
        history_summary: str | None = None,
        **system_prompt_kwargs: Any,
    ) -> AgentResponse:
        """Override to add image generation after LLM prompt-engineering step.

        The base class handles LLM invocation with tool calling.  When the LLM
        returns tool calls we pass them through so the graph can execute the
        tools and route back.  When the LLM returns text (the enhanced prompt)
        we feed it into the Gemini image generator and attach the resulting
        images to the response metadata.
        """
        response = await super().invoke_model_with_history(
            messages,
            conversation_history,
            persona,
            conversation_id,
            user_id=user_id,
            device_id=device_id,
            model_request=model_request,
            history_summary=history_summary,
            **system_prompt_kwargs,
        )

        # If the LLM requested tool calls, let the graph handle them first.
        if response.message.tool_calls:
            return response

        # If there's an error, return as-is.
        if response.error:
            return response

        # Derive the original user request from the current turn messages.
        original_prompt = ""
        for msg in reversed(messages):
            if (
                hasattr(msg, "content")
                and isinstance(msg.content, str)
                and msg.content.strip()
                and isinstance(msg, LCHumanMessage)
            ):
                original_prompt = msg.content.strip()
                break

        # The runtime model may itself be an image-capable model that returns the
        # generated images inline (with reasoning but no usable text). Harvest
        # those instead of discarding them and reporting "no response".
        harvested = self._harvest_inline_images(response, original_prompt)
        if harvested:
            response.metadata = response.metadata or {}
            response.metadata.pop("response_inline_images", None)
            response.metadata["images"] = harvested
            text = (response.message.content or "").strip()
            response.message.content = text or await self._generate_user_facing_response(
                original_prompt
            )
            return response

        # Otherwise the LLM produced a text response — treat it as the enhanced
        # prompt and generate images via the dedicated Gemini image client.
        enhanced_prompt = (response.message.content or "").strip()
        if not enhanced_prompt:
            return response

        original_prompt = original_prompt or enhanced_prompt

        try:
            images, narrative = await self._generate_images(enhanced_prompt, original_prompt)
        except Exception as e:
            logger.error("Image generation failed: %s", e, exc_info=True)
            return response

        if images:
            if not response.metadata:
                response.metadata = {}
            response.metadata["images"] = images

        # Replace the enhanced prompt with a natural user-facing message
        if narrative:
            response.message.content = narrative
        elif images:
            response.message.content = await self._generate_user_facing_response(original_prompt)

        return response

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
    ) -> AgentResponse:
        return await self.invoke_model(message, conversation_id)

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        result = await self.invoke_model(message, conversation_id)

        if result.error:
            yield {"type": "error", "error": result.error}
            return

        yield {"type": "complete", "response": result}

    async def _generate_user_facing_response(self, original_request: str) -> str:
        """Ask the LLM to produce a short, friendly message about the generated image."""
        try:
            llm = self.langchain_model
            prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. The user asked you to generate an image and "
                        "it has been created successfully. Write a brief, friendly response "
                        "acknowledging that the image is ready. Mention what was requested so "
                        "the user knows which image was created. Do NOT include the image "
                        "itself or any technical details — just a short conversational message."
                    ),
                },
                {"role": "user", "content": original_request},
            ]
            result = await llm.ainvoke(prompt_messages)
            text = coerce_response_text(result.content).strip()
            return text if text else "Your image has been generated!"
        except Exception as e:
            logger.warning("Failed to generate user-facing response: %s", e)
            return "Your image has been generated!"

    async def _generate_images(
        self, prepared_prompt: str, original_prompt: str
    ) -> tuple[list[dict], str]:
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

        images: list[dict] = []
        narrative_parts: list[str] = []

        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=generate_config,
        )

        for candidate in getattr(response, "candidates", []) or []:
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

            top_level_text = getattr(candidate, "text", None)
            if top_level_text:
                narrative_parts.append(top_level_text)

            if len(images) >= self.max_images:
                break

        narrative = " ".join(segment.strip() for segment in narrative_parts if segment)
        return images, narrative.strip()

    @staticmethod
    def _encode_image(raw_data) -> str | None:
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
