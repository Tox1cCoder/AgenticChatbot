import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage as LCHumanMessage

from ...core.config import settings
from ...interfaces.runtime_model_resolver_interface import IRuntimeModelResolver
from ...usage import begin_usage_operation, bind_usage_context, current_usage_context
from ...usage.recorder import classify_error, classify_status
from ..agent_config import AGENT_CONFIG, create_gemini_client, create_langchain_model
from ..image_generation import (
    ImageFinal,
    ImageGenerationRequest,
    ImagePartial,
    ImagePreviewPublisher,
    ImageUsage,
    NarrativeDelta,
    image_provider_family,
    resolve_image_provider,
)
from ..schemas import AgentResponse, AgentType
from ..utils import coerce_response_text, extract_inline_images_from_content
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...usage.recorder import ModelUsageRecorder


class ImageGeneratorAgent(BaseAgent):
    def __init__(
        self,
        runtime_model_resolver: IRuntimeModelResolver | None = None,
        recorder: "ModelUsageRecorder | None" = None,
    ):
        self.default_aspect_ratio = settings.image_generator_default_aspect_ratio
        self.max_images = max(1, settings.image_generator_max_images)
        self.enabled = settings.enable_image_generation
        super().__init__(
            agent_config_key="image_generator",
            runtime_model_resolver=runtime_model_resolver,
            recorder=recorder,
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

    def _get_system_prompt(self) -> str:
        return """You are an expert image generation prompt engineer.
Your goal is to create a detailed, descriptive prompt for an image generator based on
the user's request.
You have access to external tools to fetch real-time context (like weather, time, news)
if relevant to the image.
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
        source_images: list[dict[str, str]] = []
        for msg in reversed(messages):
            if not isinstance(msg, LCHumanMessage) or not hasattr(msg, "content"):
                continue
            candidate_prompt = coerce_response_text(msg.content).strip()
            candidate_images = extract_inline_images_from_content(msg.content)
            if candidate_prompt or candidate_images:
                original_prompt = candidate_prompt
                source_images = candidate_images
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
            images, narrative = await self._generate_images(
                enhanced_prompt,
                original_prompt,
                source_images=source_images,
            )
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

            async def _call() -> Any:
                return await llm.ainvoke(prompt_messages)

            if self.recorder is None:
                result = await _call()
            else:
                provider, model = self._auxiliary_model_dimensions(llm)
                context = current_usage_context().child(
                    operation="image_user_response", agent_id=self.agent_id
                )
                with bind_usage_context(context), begin_usage_operation() as operation:
                    result = await self.recorder.record_one_async_attempt(
                        call=_call,
                        provider=provider,
                        model=model,
                        operation=operation,
                    )
            text = coerce_response_text(result.content).strip()
            return text if text else "Your image has been generated!"
        except Exception as e:
            logger.warning("Failed to generate user-facing response: %s", e)
            return "Your image has been generated!"

    @staticmethod
    def _auxiliary_model_dimensions(llm: Any) -> tuple[str, str]:
        """Read ledger dimensions from the auxiliary LangChain model instance."""

        def _string_attribute(*names: str) -> str:
            for name in names:
                try:
                    value = getattr(llm, name, None)
                except Exception:
                    continue
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        fallback_model = str(AGENT_CONFIG["image_generator"]["langchain_model"])
        model = _string_attribute("model_name", "model") or fallback_model
        provider = _string_attribute("model_provider", "provider").lower()
        if not provider:
            llm_type = _string_attribute("_llm_type").lower()
            if "openai" in llm_type:
                provider = "openai"
            elif any(marker in llm_type for marker in ("google", "genai", "gemini")):
                provider = "gemini"
        return (provider or "gemini")[:32], model[:255]

    async def _generate_images(
        self,
        prepared_prompt: str,
        original_prompt: str,
        *,
        source_images: list[dict[str, str]] | None = None,
    ) -> tuple[list[dict], str]:
        provider = resolve_image_provider(
            self.model_name,
            gemini_client=self.gemini_client,
        )
        if provider is None:
            return [], ""

        request = ImageGenerationRequest(
            prompt=prepared_prompt,
            model=self.model_name,
            max_images=self.max_images,
            aspect_ratio=self.default_aspect_ratio,
            source_images=list(source_images or []),
        )

        if self.recorder is None:
            return await self._consume_image_stream(provider, request, original_prompt, handle=None)

        # Image generation is its own operation, distinct from the
        # prompt-enhancement and acknowledgement calls the base agent records.
        context = current_usage_context().child(
            operation="image_generation", agent_id=self.agent_id
        )
        with bind_usage_context(context), begin_usage_operation() as operation:
            handle = self.recorder.begin_streaming_attempt(
                provider=image_provider_family(self.model_name),
                model=self.model_name,
                operation=operation,
            )
            try:
                result = await self._consume_image_stream(
                    provider, request, original_prompt, handle=handle
                )
            except asyncio.CancelledError:
                await handle.finalize("cancelled")
                raise
            except Exception as exc:
                await handle.finalize(classify_status(exc), error_code=classify_error(exc))
                raise
            await handle.finalize("success")
            return result

    async def _consume_image_stream(
        self,
        provider: Any,
        request: ImageGenerationRequest,
        original_prompt: str,
        *,
        handle: Any | None,
    ) -> tuple[list[dict], str]:
        """Drain the provider stream, publishing previews and feeding the handle.

        Consumes the stream to exhaustion (never breaking at ``max_images``) so
        the terminal ``ImageUsage`` event is always seen; the provider caps
        image emission internally.
        """
        publisher = ImagePreviewPublisher(
            enabled=settings.enable_image_streaming,
            max_b64_chars=settings.image_stream_preview_max_b64_chars,
        )

        images: list[dict] = []
        narrative_parts: list[str] = []

        async for event in provider.stream_generate(request):
            if isinstance(event, ImageFinal):
                images.append(
                    {
                        "data": event.data_b64,
                        "mime": event.mime,
                        "prompt": original_prompt,
                        "model": self.model_name,
                        "aspect_ratio": self.default_aspect_ratio,
                    }
                )
                if handle is not None:
                    handle.note_generated_image()
                publisher.publish(
                    image_index=event.index,
                    status="final",
                    mime=event.mime,
                    data_b64=event.data_b64,
                )
            elif isinstance(event, ImagePartial):
                publisher.publish(
                    image_index=event.index,
                    status="partial",
                    mime=event.mime,
                    data_b64=event.data_b64,
                    seq=event.seq,
                )
            elif isinstance(event, ImageUsage):
                if handle is not None:
                    handle.set_usage(event.usage, provider_request_id=event.provider_request_id)
            elif isinstance(event, NarrativeDelta) and event.text:
                narrative_parts.append(event.text)

        narrative = " ".join(segment.strip() for segment in narrative_parts if segment)
        return images, narrative.strip()

    async def cleanup(self):
        await super().cleanup()
