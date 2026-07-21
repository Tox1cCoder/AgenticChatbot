"""Model-name → provider resolution.

Adding a provider (e.g. a future Anthropic/Claude image API):
implement ``ImageGenerationProvider`` in a sibling module and add its
model-name prefixes below. Nothing else in the pipeline changes — the agent
and event layers only speak the ``ImageStreamEvent`` vocabulary.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import ImageGenerationProvider
from .gemini import GeminiImageProvider
from .openai_provider import OpenAIImageProvider

logger = logging.getLogger(__name__)

_OPENAI_PREFIXES = ("gpt-image", "dall-e")


def image_provider_family(model: str) -> str:
    """Return the provider family (``"openai"`` or ``"gemini"``) for ``model``.

    Single source of truth for the provider-name → family mapping, shared by
    ``resolve_image_provider`` (routing) and the usage recorder (the
    ``provider`` dimension of an ``image_generation`` ledger event). Any name
    that is not an OpenAI image model resolves to Gemini, matching the default
    routing behavior below.
    """
    normalized = str(model or "").strip().lower()
    return "openai" if normalized.startswith(_OPENAI_PREFIXES) else "gemini"


def resolve_image_provider(
    model: str,
    *,
    gemini_client: Any | None = None,
    openai_api_key: str | None = None,
) -> ImageGenerationProvider | None:
    """Return the provider for ``model``, or None when unavailable.

    Gemini remains the default for any unrecognized model name (behavior
    parity with the pre-provider implementation, which always used the
    configured Gemini client).
    """
    if image_provider_family(model) == "openai":
        try:
            return OpenAIImageProvider(api_key=openai_api_key)
        except Exception as err:
            logger.error("OpenAI image provider unavailable for %s: %s", model, err)
            return None
    if gemini_client is None:
        return None
    return GeminiImageProvider(gemini_client)
