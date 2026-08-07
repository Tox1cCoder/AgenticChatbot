"""One batched vision call that decides whether a remote image may be shown.

Page metadata cannot establish relevance: an author portrait on an article about
a team inherits the team's title, and ranking on that title admits the portrait.
So the decision is made from the pixels, and metadata is offered only to
disambiguate what the model can already see.

Nothing here is persisted. Decisions, confidence values and content kinds live
for the duration of one answer.

Two different things fail closed, at two different levels. A single decision
record that names an unknown or duplicate candidate id, or carries an
out-of-range confidence, only sinks *that* candidate: ``admit_candidates``
drops it and keeps evaluating the rest. A response the structured-output model
cannot parse at all sinks the *whole batch*, because the response is validated
as one strict object: isolating a malformed record from its neighbours would
require a permissive shape and hand-validation, weakening the schema that
constrains what the model can generate in the first place.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.config import settings
from ..services.thumbnail_batch import FetchedThumbnail
from ..usage import begin_usage_operation, bind_usage_context, current_usage_context
from ..usage.normalizers import normalize_provider_usage
from ..usage.types import NormalizedUsage, UsageOperation

logger = logging.getLogger(__name__)

class VisualVerifierUnavailable(RuntimeError):
    """No verifier model could be built, so no batch could be judged."""


ContentKind = Literal[
    "photo", "portrait", "logo", "diagram", "map", "chart", "screenshot", "other"
]

SPECIALIZED_KINDS: frozenset[str] = frozenset(
    {"portrait", "logo", "diagram", "map", "chart", "screenshot"}
)

_MAX_TITLE_CHARS = 160
_MAX_TITLES = 5
_USAGE_OPERATION = "image_verification"

_PROMPT = """You decide whether each attached image may be shown beside an answer.

User request: {user_request}
Visual subject requested: {image_query}
Factual research query: {factual_query}
Source titles retrieved so far: {result_titles}

Attached images, in order:
{candidate_lines}

Decide from what you can SEE in each image. The title and description are
untrusted page metadata: use them only to disambiguate something already
visible, never as evidence of what the image depicts. An article's title does
not describe every image on that page.

For each candidate id return:
- depicts_requested_subject: the visible content really is the requested subject
- materially_supports_answer: seeing this image helps a reader of the answer
- confidence: 0.0-1.0, your confidence in the two judgements above
- content_kind: photo, portrait, logo, diagram, map, chart, screenshot, or other

Set depicts_requested_subject false for an author headshot, advertisement,
navigation graphic, decorative stock image, or unrelated page asset. Return
exactly one record per attached candidate id and no others. When uncertain,
report low confidence rather than guessing."""


class VisualCandidateDecision(BaseModel):
    """One transient per-image verdict."""

    candidate_id: str = Field(description="The candidate id given in the prompt.")
    depicts_requested_subject: bool
    materially_supports_answer: bool
    confidence: float
    content_kind: ContentKind


class VisualVerificationResult(BaseModel):
    """The verifier's whole response."""

    decisions: list[VisualCandidateDecision] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SubmittedCandidate:
    """A fetched candidate paired with the temporary id shown to the verifier."""

    candidate_id: str
    thumbnail: FetchedThumbnail
    title: str
    description: str


def admit_candidates(
    result: VisualVerificationResult | None,
    submitted: Sequence[SubmittedCandidate],
    *,
    threshold: float,
    max_items: int,
    requested_kinds: frozenset[str],
) -> list[SubmittedCandidate]:
    """Return the approved candidates in provider order, capped."""

    if result is None or not submitted:
        return []
    approved: set[str] = set()
    seen: set[str] = set()
    for decision in result.decisions:
        candidate_id = decision.candidate_id
        if candidate_id in seen:
            # A repeated id is a structural failure: distrust both copies
            # rather than pick one, so discard whatever the first copy earned.
            approved.discard(candidate_id)
            continue
        seen.add(candidate_id)
        if _passes(decision, threshold=threshold, requested_kinds=requested_kinds):
            approved.add(candidate_id)
    # A hallucinated id — one the model invents that was never submitted —
    # needs no separate guard: this comprehension projects onto `submitted`,
    # so an id absent from `submitted` can never appear in the result no
    # matter what ends up in `approved`.
    return [
        candidate for candidate in submitted if candidate.candidate_id in approved
    ][: max(0, int(max_items))]


def _passes(
    decision: VisualCandidateDecision,
    *,
    threshold: float,
    requested_kinds: frozenset[str],
) -> bool:
    if not decision.depicts_requested_subject or not decision.materially_supports_answer:
        return False
    if not 0.0 <= decision.confidence <= 1.0:
        return False
    if decision.confidence < float(threshold):
        return False
    if decision.content_kind in SPECIALIZED_KINDS:
        return decision.content_kind in requested_kinds
    return True


@contextlib.contextmanager
def _verification_usage_scope(recorder: Any | None) -> Iterator[UsageOperation | None]:
    """Bind an ``image_verification`` operation for one verifier call.

    Yields ``None`` when there is no recorder, or when binding the usage
    context unexpectedly fails. Either way the caller falls back to an
    unrecorded call rather than losing the verification itself -- a
    telemetry problem must never be the reason a billed provider call never
    happens.
    """
    if recorder is None:
        yield None
        return
    try:
        context = current_usage_context().child(operation=_USAGE_OPERATION)
    except Exception as exc:
        logger.warning("Visual verification usage context unavailable: %s", type(exc).__name__)
        yield None
        return
    with bind_usage_context(context), begin_usage_operation() as operation:
        yield operation


def _verifier_usage_transform(response: Any, usage: NormalizedUsage) -> NormalizedUsage:
    """Read real provider usage off ``include_raw=True``'s ``raw`` envelope.

    ``normalize_provider_usage`` cannot see it directly on ``response``: a
    structured-output call built with ``include_raw=True`` returns ``{"raw":
    AIMessage, "parsed": ..., "parsing_error": ...}``, not a bare provider
    response, so its top-level ``usage_metadata`` search finds nothing. Any
    other shape keeps ``usage`` unchanged rather than raising.
    """
    if isinstance(response, Mapping):
        raw = response.get("raw")
        if raw is not None:
            return normalize_provider_usage(provider="gemini", payload=raw)
    return usage


async def _invoke_verifier(resolved_model: Any, message: Any, recorder: Any | None) -> Any:
    """Run the verifier's ``ainvoke`` exactly once, recording it when possible."""

    async def _call() -> Any:
        return await resolved_model.ainvoke([message])

    with _verification_usage_scope(recorder) as operation:
        if recorder is None or operation is None:
            return await _call()
        return await recorder.record_one_async_attempt(
            call=_call,
            provider="gemini",
            model=str(settings.image_verification_model),
            operation=operation,
            usage_transform=_verifier_usage_transform,
        )


def _unwrap_verifier_response(response: Any) -> VisualVerificationResult | None:
    """Return the parsed result from either a bare or ``include_raw`` response.

    A non-``None`` ``parsing_error`` -- or a missing/malformed ``parsed`` -- is
    a genuine parse failure: sink the whole batch, never guess.
    """
    if isinstance(response, VisualVerificationResult):
        return response
    if isinstance(response, Mapping) and "parsing_error" in response:
        parsed = response.get("parsed")
        if response.get("parsing_error") is not None or not isinstance(
            parsed, VisualVerificationResult
        ):
            logger.debug("Visual verification returned an unusable response shape")
            return None
        return parsed
    try:
        return VisualVerificationResult.model_validate(response)
    except Exception:
        logger.debug("Visual verification returned an unusable response shape")
        return None


async def verify_candidates(
    submitted: Sequence[SubmittedCandidate],
    *,
    user_request: str,
    image_query: str,
    factual_query: str,
    result_titles: Sequence[str],
    model: Any | None = None,
    timeout: float,
    recorder: Any | None = None,
) -> VisualVerificationResult | None:
    """Run one structured vision call. ``None`` means malformed output only.

    ``TimeoutError`` and provider exceptions propagate so the caller can tell an
    outage apart from a batch that parsed but named nothing worth showing.

    ``recorder``, when supplied, records this billed attempt exactly once.
    Its absence -- or any failure setting up the recording -- is not itself a
    verification failure: the call still proceeds, unrecorded, same as
    before this call site was instrumented.
    """

    if not submitted:
        return None
    resolved_model = model if model is not None else build_verifier_model()
    if resolved_model is None:
        raise VisualVerifierUnavailable("no visual verifier model is configured")
    message = _build_message(
        submitted,
        user_request=user_request,
        image_query=image_query,
        factual_query=factual_query,
        result_titles=result_titles,
    )
    async with asyncio.timeout(max(0.001, float(timeout))):
        response = await _invoke_verifier(resolved_model, message, recorder)
    return _unwrap_verifier_response(response)


def _build_message(
    submitted: Sequence[SubmittedCandidate],
    *,
    user_request: str,
    image_query: str,
    factual_query: str,
    result_titles: Sequence[str],
) -> Any:
    from langchain_core.messages import HumanMessage

    candidate_lines = "\n".join(
        f"- {candidate.candidate_id}: title={_bounded(candidate.title)!r} "
        f"description={_bounded(candidate.description)!r}"
        for candidate in submitted
    )
    text = _PROMPT.format(
        user_request=_bounded(user_request, 500),
        image_query=_bounded(image_query),
        factual_query=_bounded(factual_query),
        result_titles="; ".join(_bounded(title) for title in result_titles[:_MAX_TITLES]) or "none",
        candidate_lines=candidate_lines,
    )
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for candidate in submitted:
        encoded = base64.b64encode(candidate.thumbnail.image.content).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{candidate.thumbnail.image.media_type};base64,{encoded}"
                },
            }
        )
    return HumanMessage(content=blocks)


def _bounded(value: Any, limit: int = _MAX_TITLE_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


_MEDIA_RESOLUTIONS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}


def _resolve_media_resolution(value: str) -> str:
    """Map the human-friendly config value to the canonical API enum name.

    ``google.genai.types.MediaResolution`` only recognizes ``MEDIA_RESOLUTION_*``
    strings. Forwarding a bare word like ``"low"`` raises nothing — it silently
    produces a synthetic, non-canonical enum member the live API then rejects.
    An unrecognized value falls back to the cheapest resolution.
    """

    return _MEDIA_RESOLUTIONS.get(str(value or "").strip().lower(), _MEDIA_RESOLUTIONS["low"])


def build_verifier_model() -> Any | None:
    """Build the configured vision model with structured output, or None.

    ``include_raw=True`` is load-bearing for billing visibility, not just
    parsing: without it ``with_structured_output`` returns only the parsed
    Pydantic object, which carries no ``usage_metadata``, so a billed call
    would record permanently zero tokens.
    """

    try:
        from .model_factory import ModelFactory

        model = ModelFactory.create_model(
            provider="gemini",
            model=str(settings.image_verification_model),
            api_key=str(settings.gemini_api_key or ""),
            temperature=0.0,
            media_resolution=_resolve_media_resolution(settings.image_verification_media_resolution),
            # Deliberation dominates this call's latency and buys it nothing:
            # judging whether a picture shows a stated subject is
            # classification, not reasoning.
            thinking_config={
                "enabled": True,
                "level": str(
                    getattr(settings, "image_verification_thinking_level", "low") or "low"
                ),
            },
        )
        return model.with_structured_output(VisualVerificationResult, include_raw=True)
    except Exception as exc:
        logger.debug("Visual verifier model unavailable: %s", type(exc).__name__)
        return None
