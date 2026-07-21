"""Provider-agnostic request/event model for streaming image generation.

Providers translate their native SDK streams into this vocabulary so the
agent and the event pipeline never see provider-specific shapes:

- ``ImagePartial`` — a progressive preview of an image that is still being
  generated (OpenAI ``gpt-image-1`` emits these; Gemini does not).
- ``ImageFinal`` — a completed image. Exactly one per ``index``.
- ``NarrativeDelta`` — provider-authored text accompanying the generation
  (Gemini image models interleave these with image parts).
- ``ImageUsage`` — the provider's terminal token accounting for the whole
  stream. Emitted at most once, after every image has been delivered, so the
  usage ledger records the image model's real cost without truncating the
  stream at ``max_images``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...usage.types import NormalizedUsage


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    model: str
    max_images: int = 1
    aspect_ratio: str = "1:1"
    # Source images for edit-style requests: [{"data": <b64>, "mime": str}]
    source_images: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ImagePartial:
    index: int
    data_b64: str
    mime: str
    seq: int


@dataclass(frozen=True)
class ImageFinal:
    index: int
    data_b64: str
    mime: str


@dataclass(frozen=True)
class NarrativeDelta:
    text: str


@dataclass(frozen=True)
class ImageUsage:
    usage: NormalizedUsage
    provider_request_id: str | None = None


ImageStreamEvent = ImagePartial | ImageFinal | NarrativeDelta | ImageUsage
