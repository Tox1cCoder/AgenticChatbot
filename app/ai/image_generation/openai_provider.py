"""OpenAI image generation provider (gpt-image models, streaming).

``gpt-image-1`` streams a small number of progressive previews
(``image_generation.partial_image`` events) followed by the completed image
(``image_generation.completed``). The API generates one image per streamed
request, so ``max_images`` is effectively 1 here.

The API key resolves through the standard SDK chain (``OPENAI_API_KEY`` env
var) unless an explicit key is passed.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ...usage.normalizers import normalize_provider_usage
from .models import (
    ImageFinal,
    ImageGenerationRequest,
    ImagePartial,
    ImageStreamEvent,
    ImageUsage,
)

logger = logging.getLogger(__name__)

# gpt-image models accept a fixed set of sizes; map the configured aspect
# ratio onto the nearest one instead of failing the request.
_SIZE_BY_ASPECT: dict[str, str] = {
    "1:1": "1024x1024",
    "16:9": "1536x1024",
    "4:3": "1536x1024",
    "3:2": "1536x1024",
    "9:16": "1024x1536",
    "3:4": "1024x1536",
    "2:3": "1024x1536",
}

_PARTIAL_PREVIEW_COUNT = 2


class OpenAIImageProvider:
    def __init__(self, api_key: str | None = None, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
        else:
            from openai import AsyncOpenAI

            # Application owns retries; disable SDK-internal retries.
            self._client = (
                AsyncOpenAI(api_key=api_key, max_retries=0)
                if api_key
                else AsyncOpenAI(max_retries=0)
            )

    @staticmethod
    def _size_for_aspect(aspect_ratio: str) -> str:
        return _SIZE_BY_ASPECT.get(str(aspect_ratio or "").strip(), "auto")

    async def _open_stream(self, request: ImageGenerationRequest) -> Any:
        common: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "n": 1,
            "size": self._size_for_aspect(request.aspect_ratio),
            "stream": True,
            "partial_images": _PARTIAL_PREVIEW_COUNT,
        }
        source = next(
            (img for img in request.source_images if img.get("data")),
            None,
        )
        if source is not None:
            import base64
            import io

            image_file = io.BytesIO(base64.b64decode(source["data"]))
            image_file.name = "source.png"
            return await self._client.images.edit(image=image_file, **common)
        return await self._client.images.generate(**common)

    async def stream_generate(
        self, request: ImageGenerationRequest
    ) -> AsyncIterator[ImageStreamEvent]:
        stream = await self._open_stream(request)
        seq = 0
        async for event in stream:
            event_type = getattr(event, "type", None)
            data_b64 = getattr(event, "b64_json", None)
            mime = f"image/{getattr(event, 'output_format', None) or 'png'}"
            if event_type in ("image_generation.partial_image", "image_edit.partial_image"):
                # Partial previews never carry usage accounting.
                if data_b64:
                    seq += 1
                    yield ImagePartial(index=0, data_b64=data_b64, mime=mime, seq=seq)
            elif event_type in ("image_generation.completed", "image_edit.completed"):
                if data_b64:
                    yield ImageFinal(index=0, data_b64=data_b64, mime=mime)
                # The completed event carries the whole request's token usage;
                # emit it as the terminal accounting event (unavailable when the
                # provider omitted it).
                yield ImageUsage(usage=normalize_provider_usage(provider="openai", payload=event))
