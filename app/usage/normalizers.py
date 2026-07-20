"""Pure normalization of raw provider usage payloads into ``NormalizedUsage``.

``normalize_provider_usage`` never raises on malformed input and never
retains a reference to the input ``payload`` — it extracts plain scalars
into a frozen ``NormalizedUsage`` and returns. It deliberately does NOT
synthesize ``total_tokens`` from ``input_tokens + output_tokens`` when the
provider omitted a total: later consumers (context-ratio accounting) need to
distinguish a provider-reported total from a derived one. Callers that want
that fallback synthesis already have it via
``TokenCounter.extract_reported_usage``, which is unaffected by this module.
"""

from __future__ import annotations

from typing import Any

from app.ai.token_counter import TokenCounter
from app.usage.types import NormalizedUsage

_INPUT_KEYS = ("input_tokens", "prompt_tokens", "prompt_token_count", "input_token_count")
_OUTPUT_KEYS = (
    "output_tokens",
    "completion_tokens",
    "candidates_token_count",
    "output_token_count",
)
_TOTAL_KEYS = ("total_tokens", "total_token_count")
_REASONING_KEYS = ("reasoning_tokens", "thoughts_token_count")
_REASONING_NESTED_PATHS = (
    ("output_token_details", "reasoning"),
    ("output_token_details", "reasoning_tokens"),
    ("completion_tokens_details", "reasoning_tokens"),
    ("output_tokens_details", "reasoning_tokens"),
)
_CACHED_INPUT_KEYS = (
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cached_content_token_count",
)
_CACHED_INPUT_NESTED_PATHS = (("input_tokens_details", "cached_tokens"),)
_INPUT_TEXT_NESTED_PATHS = (("input_tokens_details", "text_tokens"),)
_INPUT_IMAGE_NESTED_PATHS = (("input_tokens_details", "image_tokens"),)
_OUTPUT_TEXT_NESTED_PATHS = (("output_tokens_details", "text_tokens"),)
_ENVELOPE_KEYS = ("usage_metadata", "usage")


def normalize_provider_usage(*, provider: str, payload: Any) -> NormalizedUsage:
    """Normalize a raw provider usage payload into a ``NormalizedUsage``.

    ``payload`` may be a dict or a response-like object; both ``usage`` and
    ``usage_metadata`` envelopes are inspected (``usage_metadata`` first,
    matching ``TokenCounter.extract_reported_usage``'s precedence). Returns
    ``NormalizedUsage(source="unavailable")`` — never ``None`` — when nothing
    usable is found, including for malformed/non-mapping payloads.
    """
    del provider  # Alias-based shape detection covers every supported provider.
    for key in _ENVELOPE_KEYS:
        envelope = TokenCounter._raw_value(payload, key)
        if envelope is None:
            continue
        usage = _normalize_envelope(envelope)
        if usage is not None:
            return usage
    return NormalizedUsage(source="unavailable")


def _normalize_envelope(envelope: Any) -> NormalizedUsage | None:
    """Extract every known field from a single usage envelope.

    Returns ``None`` (not an "unavailable" ``NormalizedUsage``) when the
    envelope yields nothing, so the caller can keep trying other envelopes.
    """
    input_tokens = TokenCounter._first_usage_int(envelope, _INPUT_KEYS)
    output_tokens = TokenCounter._first_usage_int(envelope, _OUTPUT_KEYS)
    total_tokens = TokenCounter._first_usage_int(envelope, _TOTAL_KEYS)

    reasoning_tokens = TokenCounter._first_usage_int(envelope, _REASONING_KEYS)
    if reasoning_tokens is None:
        reasoning_tokens = TokenCounter._first_nested_usage_int(envelope, _REASONING_NESTED_PATHS)

    cached_input_tokens = TokenCounter._sum_usage_ints(envelope, _CACHED_INPUT_KEYS)
    if cached_input_tokens is None:
        cached_input_tokens = TokenCounter._first_nested_usage_int(
            envelope, _CACHED_INPUT_NESTED_PATHS
        )

    input_text_tokens = TokenCounter._first_nested_usage_int(envelope, _INPUT_TEXT_NESTED_PATHS)
    input_image_tokens = TokenCounter._first_nested_usage_int(envelope, _INPUT_IMAGE_NESTED_PATHS)
    output_text_tokens = TokenCounter._first_nested_usage_int(envelope, _OUTPUT_TEXT_NESTED_PATHS)

    prompt_text, prompt_image = _sum_modality_tokens(
        TokenCounter._raw_value(envelope, "prompt_tokens_details")
    )
    if input_text_tokens is None:
        input_text_tokens = prompt_text
    if input_image_tokens is None:
        input_image_tokens = prompt_image

    candidate_text, candidate_image = _sum_modality_tokens(
        TokenCounter._raw_value(envelope, "candidates_tokens_details")
    )
    if output_text_tokens is None:
        output_text_tokens = candidate_text
    output_image_tokens = candidate_image

    fields = (
        input_tokens,
        output_tokens,
        total_tokens,
        reasoning_tokens,
        cached_input_tokens,
        input_text_tokens,
        input_image_tokens,
        output_text_tokens,
        output_image_tokens,
    )
    if all(value is None for value in fields):
        return None

    return NormalizedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=cached_input_tokens,
        input_text_tokens=input_text_tokens,
        input_image_tokens=input_image_tokens,
        output_text_tokens=output_text_tokens,
        output_image_tokens=output_image_tokens,
        source="provider_reported",
    )


def _sum_modality_tokens(details: Any) -> tuple[int | None, int | None]:
    """Sum Gemini modality-detail entries into (text_total, image_total).

    ``details`` is a list of ``{"modality": "TEXT"|"IMAGE", "token_count": N}``
    entries (dicts or attribute-bearing objects). Non-list input (missing,
    malformed) yields ``(None, None)`` rather than raising.
    """
    if not isinstance(details, (list, tuple)):
        return None, None
    text_total: int | None = None
    image_total: int | None = None
    for entry in details:
        token_count = TokenCounter._first_usage_int(entry, ("token_count",))
        if token_count is None:
            continue
        modality = TokenCounter._raw_value(entry, "modality")
        modality_name = str(modality).strip().upper() if modality is not None else ""
        if modality_name == "TEXT":
            text_total = (text_total or 0) + token_count
        elif modality_name == "IMAGE":
            image_total = (image_total or 0) + token_count
    return text_total, image_total


__all__ = ["normalize_provider_usage"]
