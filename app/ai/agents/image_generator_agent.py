import base64
import logging
from typing import Optional, List

from google import genai
from google.genai import types

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_image_generator_prompt
from ...core.config import settings

logger = logging.getLogger(__name__)


class ImageGeneratorAgent:
    """Agent responsible for generating images."""

    def __init__(self):
        self.model_name = settings.image_generator_model
        self.default_aspect_ratio = settings.image_generator_default_aspect_ratio
        self.max_images = max(1, settings.image_generator_max_images)
        self.enabled = settings.enable_image_generation
        self.gemini_client: Optional[genai.Client] = None
        self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize Gemini client"""
        if not self.enabled:
            logger.info("Image generation disabled via configuration")
            return

        api_key = settings.gemini_api_key

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:
        """Generate an image for the provided prompt."""

        if not self.enabled:
            logger.warning("Image generation requested but disabled")
            return self._build_error_response(
                "Image generation is currently disabled. Please contact the administrator.",
                conversation_id,
            )

        if not self.gemini_client:
            logger.error("Gemini client not initialized for image generation")
            return self._build_error_response(
                "Unable to generate images because the Gemini client is not configured.",
                conversation_id,
            )

        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        prompt = build_image_generator_prompt(
            message.content,
            conversation_history,
            persona=persona,
        )

        try:
            images, narrative = await self._generate_images(prompt, message.content)
        except Exception as err:
            logger.error("Image generation failed: %s", err, exc_info=True)
            return self._build_error_response(
                f"An error occurred while generating the image: {err}",
                conversation_id,
            )

        if not images:
            logger.warning("No images generated for prompt: %s", prompt)
            return self._build_error_response(
                "No image was generated for your request. Please try refining your description.",
                conversation_id,
            )

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=narrative
            or "Here is the image I created based on your description.",
        )

        response_metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "context_messages": len(conversation_history),
            "persona_used": persona,
            "images": images,
        }

        return AgentResponse(
            agent_type=AgentType.IMAGE_GENERATOR,
            agent_id="image_generator_agent",
            message=response_message,
            metadata=response_metadata,
        )

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
            try:
                config_kwargs["image_generation_config"] = image_config_cls(
                    number_of_images=self.max_images,
                    aspect_ratio=self.default_aspect_ratio,
                )
            except Exception as err:
                logger.debug("Unable to configure image generation parameters: %s", err)

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
        """Convert inline image data into base64 string."""
        if raw_data is None:
            return None

        try:
            if isinstance(raw_data, (bytes, bytearray)):
                return base64.b64encode(raw_data).decode("utf-8")
            if isinstance(raw_data, str):
                # Library may already return base64-encoded string
                return raw_data
            if isinstance(raw_data, memoryview):
                return base64.b64encode(raw_data.tobytes()).decode("utf-8")
            # Fallback: attempt to convert to bytes
            return base64.b64encode(bytes(raw_data)).decode("utf-8")
        except Exception as err:
            logger.error("Failed to encode image data: %s", err, exc_info=True)
            return None

    def _build_error_response(
        self, message: str, conversation_id: Optional[str]
    ) -> AgentResponse:
        """Create standardized error responses."""
        response_message = AgentMessage(
            role=MessageRole.ASSISTANT,
            content=message,
        )

        return AgentResponse(
            agent_type=AgentType.IMAGE_GENERATOR,
            agent_id="image_generator_agent",
            message=response_message,
            metadata={
                "model": self.model_name,
                "conversation_id": conversation_id,
                "images": [],
                "error": True,
            },
            error=message,
        )
