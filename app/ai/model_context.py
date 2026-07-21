"""Pure helpers for resolving model context-window metadata.

Used by provider catalog sync, runtime model resolution, agent middleware,
and the Streamlit demo to render a context-window usage indicator.

This module has zero I/O and zero DB dependencies. Everything is a pure
function over plain dicts / dataclasses so it can be unit-tested in
isolation and called from any layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from app.usage import NormalizedUsage

ContextSource = Literal["provider_api", "registry", "heuristic", "unknown"]
DisplayState = Literal["unknown", "ok", "warn", "danger"]
# How a model's limits are shaped. ``shared_context`` models bound a single
# combined window; ``separate_io`` models (e.g. Gemini image) publish distinct
# input and output limits with no combined window; ``unknown`` models (e.g.
# OpenAI GPT Image) publish no usable denominator at all.
LimitType = Literal["shared_context", "separate_io", "unknown"]
UsedTokenSource = Literal[
    "provider_reported_total",
    "provider_reported_split",
    "estimated_total",
    "unknown",
]


@dataclass
class ModelContextWindow:
    """Resolved context-window metadata for a single (provider, model) pair."""

    provider: str
    model: str
    context_window_tokens: int | None
    max_input_tokens: int | None
    max_output_tokens: int | None
    source: ContextSource
    known: bool
    limit_type: LimitType = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly shape defined by the data contract."""
        return {
            "provider": self.provider,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "limit_type": self.limit_type,
            "source": self.source,
            "known": self.known,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Conservative registry of well-known model families. Values are biased toward
# the documented public limits at time of writing; provider API metadata
# (Task 2) should always be preferred when available.
#
# Lookup strategy: exact (lowercased) match first, then longest matching
# family prefix.

_RegistryEntry = dict[str, int]

_OPENAI_REGISTRY: dict[str, _RegistryEntry] = {
    # gpt-5 family
    "gpt-5": {"context_window_tokens": 400000, "max_output_tokens": 128000},
    "gpt-5-mini": {"context_window_tokens": 400000, "max_output_tokens": 128000},
    "gpt-5-pro": {"context_window_tokens": 400000, "max_output_tokens": 128000},
    # gpt-4.1 family
    "gpt-4.1": {"context_window_tokens": 1_047_576, "max_output_tokens": 32768},
    "gpt-4.1-mini": {"context_window_tokens": 1_047_576, "max_output_tokens": 32768},
    "gpt-4.1-nano": {"context_window_tokens": 1_047_576, "max_output_tokens": 32768},
    # gpt-4o family
    "gpt-4o": {"context_window_tokens": 128000, "max_output_tokens": 16384},
    "gpt-4o-mini": {"context_window_tokens": 128000, "max_output_tokens": 16384},
    "chatgpt-4o-latest": {"context_window_tokens": 128000, "max_output_tokens": 16384},
    # o-series
    "o1": {"context_window_tokens": 200000, "max_output_tokens": 100000},
    "o1-mini": {"context_window_tokens": 200000, "max_output_tokens": 100000},
    "o1-pro": {"context_window_tokens": 200000, "max_output_tokens": 100000},
    "o3": {"context_window_tokens": 200000, "max_output_tokens": 100000},
    "o3-mini": {"context_window_tokens": 200000, "max_output_tokens": 100000},
    "o3-pro": {"context_window_tokens": 200000, "max_output_tokens": 100000},
    "o4-mini": {"context_window_tokens": 200000, "max_output_tokens": 100000},
}

_GEMINI_REGISTRY: dict[str, _RegistryEntry] = {
    "gemini-2.5-flash": {"context_window_tokens": 1_048_576, "max_output_tokens": 65536},
    "gemini-2.5-pro": {"context_window_tokens": 1_048_576, "max_output_tokens": 65536},
    "gemini-3-flash-preview": {
        "context_window_tokens": 1_048_576,
        "max_output_tokens": 65536,
    },
    "gemini-3.1-pro-preview": {
        "context_window_tokens": 1_048_576,
        "max_output_tokens": 65536,
    },
    # Image models publish separate input/output limits and no combined
    # window. ``gemini-3-pro-image-preview`` is the deprecated id; its
    # replacement ``gemini-3-pro-image`` shares the documented limits and the
    # alias is kept for persisted-history compatibility.
    "gemini-3-pro-image": {"max_input_tokens": 65536, "max_output_tokens": 32768},
    "gemini-3-pro-image-preview": {"max_input_tokens": 65536, "max_output_tokens": 32768},
    "gemini-3.1-flash-image": {"max_input_tokens": 65536, "max_output_tokens": 32768},
}

_ANTHROPIC_REGISTRY: dict[str, _RegistryEntry] = {
    "claude-opus-4-7": {"context_window_tokens": 200000, "max_output_tokens": 8192},
    "claude-sonnet-4-6": {"context_window_tokens": 200000, "max_output_tokens": 8192},
    "claude-haiku-4-5": {"context_window_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-5-sonnet": {"context_window_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-5-haiku": {"context_window_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-opus": {"context_window_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-sonnet": {"context_window_tokens": 200000, "max_output_tokens": 8192},
    "claude-3-haiku": {"context_window_tokens": 200000, "max_output_tokens": 8192},
}

_REGISTRY: dict[str, dict[str, _RegistryEntry]] = {
    "openai": _OPENAI_REGISTRY,
    "gemini": _GEMINI_REGISTRY,
    "anthropic": _ANTHROPIC_REGISTRY,
}


def _lookup_registry(provider: str, model_id: str) -> _RegistryEntry | None:
    """Look up registry entry by exact (lowercased) ID, then family prefix."""
    provider_key = (provider or "").lower()
    families = _REGISTRY.get(provider_key)
    if not families:
        return None

    model_lc = (model_id or "").lower()
    if not model_lc:
        return None

    # Exact match first.
    entry = families.get(model_lc)
    if entry is not None:
        return entry

    # Longest matching family prefix.
    best: tuple[int, _RegistryEntry] | None = None
    for family_key, family_entry in families.items():
        if model_lc.startswith(family_key) and (best is None or len(family_key) > best[0]):
            best = (len(family_key), family_entry)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Provider API metadata normalization
# ---------------------------------------------------------------------------

# Keys we accept from provider-supplied metadata, mapped to a logical role.
_CONTEXT_KEYS = (
    "context_window_tokens",
    "contextWindowTokens",
    "max_input_tokens",
    "maxInputTokens",
    "input_token_limit",
    "inputTokenLimit",
)
_OUTPUT_KEYS = (
    "max_output_tokens",
    "maxOutputTokens",
    "output_token_limit",
    "outputTokenLimit",
)


def _coerce_positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _extract_first(raw: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key in raw:
            value = _coerce_positive_int(raw[key])
            if value is not None:
                return value
    return None


def normalize_context_window_metadata(
    provider: str,
    model_id: str,
    raw_metadata: dict[str, Any] | None,
) -> ModelContextWindow | None:
    """Parse provider API metadata into a ``ModelContextWindow``.

    Accepts snake_case (``input_token_limit``, ``output_token_limit``) and
    camelCase (``inputTokenLimit``, ``outputTokenLimit``,
    ``contextWindowTokens``, ``maxInputTokens``, ``maxOutputTokens``) keys.
    Also accepts the canonical ``context_window_tokens`` /
    ``max_input_tokens`` / ``max_output_tokens`` shape.

    Returns ``None`` when no useful field is present.
    """
    if not isinstance(raw_metadata, dict) or not raw_metadata:
        return None

    # Collect every present context-style value so we can pick the largest.
    context_candidates: list[int] = []
    for key in _CONTEXT_KEYS:
        if key in raw_metadata:
            value = _coerce_positive_int(raw_metadata[key])
            if value is not None:
                context_candidates.append(value)

    max_output = _extract_first(raw_metadata, _OUTPUT_KEYS)

    if not context_candidates and max_output is None:
        return None

    context_window = max(context_candidates) if context_candidates else None
    # When only context_window_tokens / max_input_tokens is given, they are
    # equivalent for our purposes.
    max_input = context_window

    return ModelContextWindow(
        provider=provider,
        model=model_id,
        context_window_tokens=context_window,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        source="provider_api",
        known=True,
        limit_type="shared_context",
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _catalog_has_window_signal(catalog: dict[str, Any]) -> bool:
    """Return True if ``catalog`` carries any usable context-window signal."""
    if catalog.get("context_window_known") is True:
        return True
    for key in _CONTEXT_KEYS + _OUTPUT_KEYS:
        if key in catalog and _coerce_positive_int(catalog[key]) is not None:
            return True
    return False


def resolve_model_context_window(
    provider: str,
    model_id: str,
    catalog_metadata: dict[str, Any] | None = None,
) -> ModelContextWindow:
    """Resolve context-window metadata for ``(provider, model_id)``.

    Resolution order:
    1. ``catalog_metadata`` if it carries provider-API signal (any known
       context/input/output field or ``context_window_known=True``).
    2. Conservative built-in registry (exact then longest family prefix).
    3. Unknown sentinel result with ``known=False``.
    """
    if isinstance(catalog_metadata, dict) and _catalog_has_window_signal(catalog_metadata):
        normalized = normalize_context_window_metadata(provider, model_id, catalog_metadata)
        if normalized is not None:
            return normalized

    entry = _lookup_registry(provider, model_id)
    if entry is not None:
        context_window = entry.get("context_window_tokens")
        max_output = entry.get("max_output_tokens")
        if context_window is not None:
            # Shared combined window; input limit equals the window.
            return ModelContextWindow(
                provider=provider,
                model=model_id,
                context_window_tokens=context_window,
                max_input_tokens=context_window,
                max_output_tokens=max_output,
                source="registry",
                known=True,
                limit_type="shared_context",
            )
        # Separate input/output limits (image models); no combined window.
        return ModelContextWindow(
            provider=provider,
            model=model_id,
            context_window_tokens=None,
            max_input_tokens=entry.get("max_input_tokens"),
            max_output_tokens=max_output,
            source="registry",
            known=True,
            limit_type="separate_io",
        )

    return ModelContextWindow(
        provider=provider,
        model=model_id,
        context_window_tokens=None,
        max_input_tokens=None,
        max_output_tokens=None,
        source="unknown",
        known=False,
        limit_type="unknown",
    )


# ---------------------------------------------------------------------------
# Usage computation
# ---------------------------------------------------------------------------


def _nonneg_int(value: Any) -> int | None:
    """Return ``value`` when it is a non-boolean, non-negative int, else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _used_tokens_from_usage(usage: NormalizedUsage | None) -> tuple[int | None, UsedTokenSource]:
    """Pick the best available token figure and label how it was derived.

    Precedence: a reported/estimated combined total, then the sum of the known
    input/output split, then unknown. A reported total never overrides a
    reported split — both are preserved separately on the payload; this only
    chooses the single ``used_tokens`` figure for the shared-window ratio.
    """
    if usage is None:
        return None, "unknown"

    source = getattr(usage, "source", "unavailable")
    reported = source == "provider_reported"

    total = _nonneg_int(getattr(usage, "total_tokens", None))
    if total is not None:
        return total, ("provider_reported_total" if reported else "estimated_total")

    input_tokens = _nonneg_int(getattr(usage, "input_tokens", None))
    output_tokens = _nonneg_int(getattr(usage, "output_tokens", None))
    if input_tokens is not None or output_tokens is not None:
        summed = (input_tokens or 0) + (output_tokens or 0)
        return summed, ("provider_reported_split" if reported else "estimated_total")

    return None, "unknown"


def _classify_display_state(usage_ratio: float | None) -> DisplayState:
    if usage_ratio is None:
        return "unknown"
    if usage_ratio < 0.70:
        return "ok"
    if usage_ratio < 0.90:
        return "warn"
    return "danger"


def _ratio(numerator: int | None, denominator: Any) -> float | None:
    if numerator is None or not isinstance(denominator, int) or denominator <= 0:
        return None
    return numerator / denominator


def build_context_window_usage(
    context_window: dict[str, Any] | None,
    usage: NormalizedUsage | None,
) -> dict[str, Any]:
    """Compute the limit-aware context-window usage payload for the indicator.

    ``context_window`` is the dict shape returned by
    :meth:`ModelContextWindow.to_dict` (carrying ``limit_type``). ``usage`` is a
    :class:`app.usage.NormalizedUsage`. Provider-reported input, output, and
    total are preserved independently; a reported total never overwrites a
    reported split. The returned ratio is raw (never clamped) — the drawn gauge
    is capped only at render time.

    ``limit_type`` drives the ratio:
    - ``shared_context``: ``usage_ratio = used_tokens / context_window_tokens``.
    - ``separate_io``: input and output ratios are computed independently
      against their own limits and the larger known ratio wins; a combined
      total is never divided by a single limit.
    - unknown / ``known=False``: the token counts are retained but no ratio is
      produced.
    """
    input_tokens = _nonneg_int(getattr(usage, "input_tokens", None)) if usage else None
    output_tokens = _nonneg_int(getattr(usage, "output_tokens", None)) if usage else None
    total_tokens = _nonneg_int(getattr(usage, "total_tokens", None)) if usage else None
    usage_source = getattr(usage, "source", "unavailable") if usage else "unavailable"

    payload: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_source": usage_source,
        "used_tokens": None,
        "used_token_source": "unknown",
        "input_usage_ratio": None,
        "output_usage_ratio": None,
        "usage_ratio": None,
        "usage_ratio_basis": None,
        "display_state": "unknown",
    }

    # No usable denominator: retain the counts but never invent a ratio.
    if not context_window or not context_window.get("known"):
        return payload

    limit_type = context_window.get("limit_type") or "shared_context"

    if limit_type == "separate_io":
        input_ratio = _ratio(input_tokens, context_window.get("max_input_tokens"))
        output_ratio = _ratio(output_tokens, context_window.get("max_output_tokens"))
        known_ratios = [r for r in (input_ratio, output_ratio) if r is not None]
        usage_ratio = max(known_ratios) if known_ratios else None
        used_tokens, used_source = _used_tokens_from_usage(usage)
        payload.update(
            {
                "used_tokens": used_tokens,
                "used_token_source": used_source,
                "input_usage_ratio": input_ratio,
                "output_usage_ratio": output_ratio,
                "usage_ratio": usage_ratio,
                "usage_ratio_basis": (
                    "most_constrained_io_limit" if usage_ratio is not None else None
                ),
                "display_state": _classify_display_state(usage_ratio),
            }
        )
        return payload

    if limit_type == "unknown":
        return payload

    # shared_context
    denominator = context_window.get("context_window_tokens") or context_window.get(
        "max_input_tokens"
    )
    used_tokens, used_source = _used_tokens_from_usage(usage)
    usage_ratio = _ratio(used_tokens, denominator)
    if usage_ratio is None:
        payload["used_tokens"] = used_tokens
        payload["used_token_source"] = used_source if used_tokens is not None else "unknown"
        return payload

    payload.update(
        {
            "used_tokens": used_tokens,
            "used_token_source": used_source,
            "input_usage_ratio": _ratio(input_tokens, denominator),
            "output_usage_ratio": _ratio(output_tokens, denominator),
            "usage_ratio": usage_ratio,
            "usage_ratio_basis": "shared_context_total",
            "display_state": _classify_display_state(usage_ratio),
        }
    )
    return payload


__all__ = [
    "ModelContextWindow",
    "build_context_window_usage",
    "normalize_context_window_metadata",
    "resolve_model_context_window",
]
