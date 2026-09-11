"""Gemini image generation provider (google-genai, async streaming).

Gemini image models do not emit progressive pixel previews; each image
arrives as a whole ``inline_data`` part. Streaming still delivers every image
the moment it is generated (ahead of narrative text and turn finalization)
and keeps the event loop unblocked, unlike the previous synchronous
``models.generate_content`` call.
"""

from __future__ import annotations

import base64
import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any

from google.genai import types

from ...usage.normalizers import normalize_provider_usage
from .models import (
    ImageFinal,
    ImageGenerationRequest,
    ImageStreamEvent,
    ImageUsage,
    NarrativeDelta,
)

logger = logging.getLogger(__name__)


def _encode_inline_data(raw_data: Any) -> str | None:
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
        logger.error("Failed to encode Gemini image data: %s", err)
        return None


class GeminiImageProvider:
    def __init__(self, client: Any) -> None:
        self._client = client

    def _build_contents(self, request: ImageGenerationRequest) -> list[types.Content]:
        parts = [types.Part.from_text(text=request.prompt)]
        for source_image in request.source_images:
            encoded = source_image.get("data")
            if not encoded:
                continue
            try:
                parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(encoded, validate=True),
                        mime_type=source_image.get("mime") or "image/png",
                    )
                )
            except (TypeError, ValueError):
                logger.warning("Skipping invalid source image supplied for image editing")
        return [types.Content(role="user", parts=parts)]

    def _build_config(self, request: ImageGenerationRequest) -> types.GenerateContentConfig:
        config_kwargs: dict[str, Any] = {"response_modalities": ["IMAGE", "TEXT"]}
        image_config_cls = getattr(types, "ImageGenerationConfig", None)
        if image_config_cls is not None:
            config_kwargs["image_generation_config"] = image_config_cls(
                number_of_images=request.max_images,
                aspect_ratio=request.aspect_ratio,
            )
        # Image generation binds no tools, and automatic function calling
        # defaults to enabled. Saying so keeps the SDK out of the business of
        # executing anything on this path.
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )
        return types.GenerateContentConfig(**config_kwargs)

    async def stream_generate(
        self, request: ImageGenerationRequest
    ) -> AsyncIterator[ImageStreamEvent]:
        stream = self._client.aio.models.generate_content_stream(
            model=request.model,
            contents=self._build_contents(request),
            config=self._build_config(request),
        )
        if inspect.isawaitable(stream):
            stream = await stream

        image_index = 0
        latest_usage_metadata: Any = None
        latest_response_id: str | None = None
        async for chunk in stream:
            usage_metadata = getattr(chunk, "usage_metadata", None)
            if usage_metadata is not None:
                latest_usage_metadata = usage_metadata
            response_id = getattr(chunk, "response_id", None)
            if response_id:
                latest_response_id = response_id
            for candidate in getattr(chunk, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                if content is None:
                    continue
                for part in getattr(content, "parts", None) or []:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data is not None and getattr(inline_data, "data", None):
                        # Cap emission at max_images but keep consuming the
                        # stream: the terminal usage_metadata chunk arrives
                        # after the images, so returning early would discard it.
                        if image_index < request.max_images:
                            encoded = _encode_inline_data(inline_data.data)
                            if encoded:
                                yield ImageFinal(
                                    index=image_index,
                                    data_b64=encoded,
                                    mime=getattr(inline_data, "mime_type", None) or "image/png",
                                )
                                image_index += 1
                        continue
                    text_segment = getattr(part, "text", None)
                    if text_segment:
                        yield NarrativeDelta(text=text_segment)

        # Exactly one terminal usage event after the stream is exhausted, even
        # when the provider reported no usage_metadata (source="unavailable").
        yield ImageUsage(
            usage=normalize_provider_usage(
                provider="gemini",
                payload={"usage_metadata": latest_usage_metadata},
            ),
            provider_request_id=latest_response_id,
        )
