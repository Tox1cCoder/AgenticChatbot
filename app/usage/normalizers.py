"""Pure normalization of raw provider usage payloads into ``NormalizedUsage``.

``normalize_provider_usage`` never raises on malformed input and never
retains a reference to the input ``payload`` — it extracts plain scalars
into a frozen ``NormalizedUsage`` and returns. It deliberately does NOT
synthesize ``total_tokens`` from ``input_tokens + output_tokens`` when the
provider omitted a total: later consumers (context-ratio accounting) need to
distinguish a provider-reported total from a derived one. Callers that want
that fallback synthesis already have it via
``TokenCounter.extract_reported_usage``, which is unaffected by this module.

The envelope-search chain and every alias/nested-path tuple used below are
imported from ``app.ai.token_counter`` — the single source of truth shared
with ``TokenCounter.extract_reported_usage`` — so the two functions cannot
silently drift apart on which provider shapes they recognize. Per-function
differences are deliberate and never folded into the shared constants:
no total-synthesis here; this module additionally sums Gemini's raw
modality-detail arrays (a shape LangChain's standardized ``usage_metadata``
never surfaces, so ``extract_reported_usage`` has no use for it); and this
module recognizes one extra reasoning nested path
(``output_tokens_details.reasoning_tokens``, plural — OpenAI's raw shape)
via ``_NORMALIZE_REASONING_NESTED_PATHS`` below, layered on top of the
shared 3-entry base rather than added to it, so
``extract_reported_usage``'s reasoning extraction stays exactly what it was
before Task 4.
"""

from __future__ import annotations

from typing import Any

from app.ai.token_counter import (
    _USAGE_INPUT_ALIASES,
    _USAGE_INPUT_IMAGE_NESTED_PATHS,
    _USAGE_INPUT_TEXT_NESTED_PATHS,
    _USAGE_OUTPUT_ALIASES,
    _USAGE_OUTPUT_IMAGE_NESTED_PATHS,
    _USAGE_OUTPUT_TEXT_NESTED_PATHS,
    _USAGE_REASONING_ALIASES,
    _USAGE_REASONING_NESTED_PATHS,
    _USAGE_TOTAL_ALIASES,
    TokenCounter,
)
from app.usage.types import NormalizedUsage

# Deliberate per-function extension, not shared: extract_reported_usage's
# reasoning nested-path search must stay its original 3 pre-Task-4 entries
# (see the comment on _USAGE_REASONING_NESTED_PATHS in token_counter.py), but
# this normalizer also recognizes OpenAI's raw plural
# "output_tokens_details.reasoning_tokens" shape, which extract_reported_usage
# never has (LangChain's standardized usage_metadata doesn't surface it).
_NORMALIZE_REASONING_NESTED_PATHS = _USAGE_REASONING_NESTED_PATHS + (
    ("output_tokens_details", "reasoning_tokens"),
)


def normalize_provider_usage(*, provider: str, payload: Any) -> NormalizedUsage:
    """Normalize a raw provider usage payload into a ``NormalizedUsage``.

    ``payload`` may be a dict or a response-like object. Searches the same
    ordered envelope chain as ``TokenCounter.extract_reported_usage``:
    ``usage_metadata``, ``usage``, then ``response_metadata.usage``,
    ``response_metadata.token_usage``, and bare ``response_metadata`` —
    covering LangChain response objects whose usage lives only in
    ``response_metadata`` (e.g. as handed to the recorder in Task 5). Returns
    ``NormalizedUsage(source="unavailable")`` — never ``None`` — when nothing
    usable is found, including for malformed/non-mapping payloads.
    """
    del provider  # Alias-based shape detection covers every supported provider.
    for envelope in TokenCounter._iter_usage_envelopes(payload):
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
    input_tokens = TokenCounter._first_usage_int(envelope, _USAGE_INPUT_ALIASES)
    output_tokens = TokenCounter._first_usage_int(envelope, _USAGE_OUTPUT_ALIASES)
    total_tokens = TokenCounter._first_usage_int(envelope, _USAGE_TOTAL_ALIASES)

    reasoning_tokens = TokenCounter._first_usage_int(envelope, _USAGE_REASONING_ALIASES)
    if reasoning_tokens is None:
        reasoning_tokens = TokenCounter._first_nested_usage_int(
            envelope, _NORMALIZE_REASONING_NESTED_PATHS
        )

    cached_input_tokens = TokenCounter._extract_cached_input_tokens(envelope)

    input_text_tokens = TokenCounter._first_nested_usage_int(
        envelope, _USAGE_INPUT_TEXT_NESTED_PATHS
    )
    input_image_tokens = TokenCounter._first_nested_usage_int(
        envelope, _USAGE_INPUT_IMAGE_NESTED_PATHS
    )
    output_text_tokens = TokenCounter._first_nested_usage_int(
        envelope, _USAGE_OUTPUT_TEXT_NESTED_PATHS
    )
    output_image_tokens = TokenCounter._first_nested_usage_int(
        envelope, _USAGE_OUTPUT_IMAGE_NESTED_PATHS
    )

    # Gemini's raw modality-detail arrays are a fallback source only — the
    # OpenAI-style nested paths above win when both happen to be present.
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
    if output_image_tokens is None:
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
    entries (dicts or attribute-bearing objects). Non-list input, non-mapping
    entries, and entries missing "modality" or "token_count" are all skipped
    as unknown rather than raising.
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
