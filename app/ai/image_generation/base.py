"""Protocol every image generation provider implements."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from .models import ImageGenerationRequest, ImageStreamEvent


@runtime_checkable
class ImageGenerationProvider(Protocol):
    """A provider streams `ImageStreamEvent`s for one generation request.

    Implementations must yield at most one ``ImageFinal`` per image index and
    may yield any number of ``ImagePartial``/``NarrativeDelta`` events before
    it. Errors raise — the caller owns error handling and fallbacks.
    """

    def stream_generate(
        self, request: ImageGenerationRequest
    ) -> AsyncIterator[ImageStreamEvent]: ...
