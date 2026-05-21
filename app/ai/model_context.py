"""Pure helpers for resolving model context-window metadata.

Used by provider catalog sync, runtime model resolution, agent middleware,
and the Streamlit demo to render a context-window usage indicator.

This module has zero I/O and zero DB dependencies. Everything is a pure
function over plain dicts / dataclasses so it can be unit-tested in
isolation and called from any layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ContextSource = Literal["provider_api", "registry", "heuristic", "unknown"]
DisplayState = Literal["unknown", "ok", "warn", "danger"]
UsedTokenSource = Literal["actual_input", "estimated_total", "unknown"]


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

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly shape defined by the data contract."""
        return {
            "provider": self.provider,
            "model": self.model,
            "context_window_tokens": self.context_window_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
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
        return ModelContextWindow(
            provider=provider,
            model=model_id,
            context_window_tokens=context_window,
            max_input_tokens=context_window,
            max_output_tokens=max_output,
            source="registry",
            known=True,
        )

    return ModelContextWindow(
        provider=provider,
        model=model_id,
        context_window_tokens=None,
        max_input_tokens=None,
        max_output_tokens=None,
        source="unknown",
        known=False,
    )


# ---------------------------------------------------------------------------
# Usage computation
# ---------------------------------------------------------------------------


def _select_used_tokens(
    token_breakdown: dict[str, Any] | None,
) -> tuple[int | None, UsedTokenSource]:
    """Pick the best available token total from a ``TokenBudgetBreakdown`` dict."""
    if not isinstance(token_breakdown, dict):
        return None, "unknown"

    actual = token_breakdown.get("actual") or {}
    if isinstance(actual, dict):
        actual_input = actual.get("input_tokens")
        if isinstance(actual_input, int) and actual_input >= 0:
            return actual_input, "actual_input"

    estimated = token_breakdown.get("estimated") or {}
    if isinstance(estimated, dict):
        total = estimated.get("total_tokens")
        if isinstance(total, int) and total > 0:
            return total, "estimated_total"

    return None, "unknown"


def _classify_display_state(usage_ratio: float | None) -> DisplayState:
    if usage_ratio is None:
        return "unknown"
    if usage_ratio < 0.70:
        return "ok"
    if usage_ratio < 0.90:
        return "warn"
    return "danger"


def build_context_window_usage(
    context_window: dict[str, Any] | None,
    token_breakdown: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute the context-window usage payload used by the demo indicator.

    ``context_window`` is the dict shape returned by
    :meth:`ModelContextWindow.to_dict`. ``token_breakdown`` is the dict
    returned by ``TokenBudgetBreakdown.to_dict()``.

    Returns a dict with ``used_tokens``, ``used_token_source``,
    ``usage_ratio``, and ``display_state``.
    """
    # Unknown context window: the indicator collapses to a fully unknown
    # payload — we have no denominator to make the token count meaningful.
    if not context_window or not context_window.get("known"):
        return {
            "used_tokens": None,
            "used_token_source": "unknown",
            "usage_ratio": None,
            "display_state": "unknown",
        }

    used_tokens, used_source = _select_used_tokens(token_breakdown)

    if used_tokens is None:
        return {
            "used_tokens": None,
            "used_token_source": "unknown",
            "usage_ratio": None,
            "display_state": "unknown",
        }

    denominator = context_window.get("max_input_tokens") or context_window.get(
        "context_window_tokens"
    )
    if not isinstance(denominator, int) or denominator <= 0:
        return {
            "used_tokens": used_tokens,
            "used_token_source": used_source,
            "usage_ratio": None,
            "display_state": "unknown",
        }

    usage_ratio = used_tokens / denominator

    return {
        "used_tokens": used_tokens,
        "used_token_source": used_source,
        "usage_ratio": usage_ratio,
        "display_state": _classify_display_state(usage_ratio),
    }


__all__ = [
    "ModelContextWindow",
    "build_context_window_usage",
    "normalize_context_window_metadata",
    "resolve_model_context_window",
]
