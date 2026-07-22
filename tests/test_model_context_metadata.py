"""Tests for the pure context-window metadata helpers in ``app.ai.model_context``.

Covers:
- Registry lookups (exact, case-insensitive, family-prefix).
- Unknown model handling.
- ``normalize_context_window_metadata`` parsing snake_case and camelCase keys.
- ``resolve_model_context_window`` precedence (provider API metadata wins).
- ``build_context_window_usage`` token-source choice, ratio math, and display
  state thresholds.
"""

from __future__ import annotations

import pytest

from app.ai.model_context import (
    ModelContextWindow,
    build_context_window_usage,
    normalize_context_window_metadata,
    resolve_model_context_window,
)
from app.usage import NormalizedUsage

# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------


def test_resolve_exact_openai_gpt_4o():
    cw = resolve_model_context_window("openai", "gpt-4o")
    assert isinstance(cw, ModelContextWindow)
    assert cw.known is True
    assert cw.source == "registry"
    assert cw.provider == "openai"
    assert cw.model == "gpt-4o"
    assert cw.context_window_tokens == 128000
    assert cw.max_input_tokens == 128000
    assert cw.max_output_tokens == 16384


def test_resolve_case_insensitive_model_id():
    upper = resolve_model_context_window("openai", "GPT-4O")
    lower = resolve_model_context_window("openai", "gpt-4o")
    assert upper.known is True
    assert upper.context_window_tokens == lower.context_window_tokens
    assert upper.max_output_tokens == lower.max_output_tokens


def test_resolve_family_prefix_match_gpt_4o_mini_dated():
    cw = resolve_model_context_window("openai", "gpt-4o-mini-2024-07-18")
    assert cw.known is True
    assert cw.context_window_tokens == 128000
    assert cw.max_output_tokens == 16384


def test_resolve_family_prefix_match_gpt_5_variant():
    cw = resolve_model_context_window("openai", "gpt-5-mini-some-suffix")
    assert cw.known is True
    assert cw.context_window_tokens == 400000
    assert cw.max_output_tokens == 128000


def test_resolve_family_prefix_match_o1_pro_dated():
    cw = resolve_model_context_window("openai", "o1-pro-2025-01-01")
    assert cw.known is True
    assert cw.context_window_tokens == 200000
    assert cw.max_output_tokens == 100000


def test_resolve_gpt_4_1_family_one_million_context():
    cw = resolve_model_context_window("openai", "gpt-4.1-mini")
    assert cw.known is True
    assert cw.context_window_tokens == 1_047_576
    assert cw.max_output_tokens == 32768


def test_resolve_gemini_25_flash_family():
    cw = resolve_model_context_window("gemini", "gemini-2.5-flash-preview")
    assert cw.known is True
    assert cw.context_window_tokens == 1_048_576
    assert cw.max_output_tokens == 65536


def test_resolve_gemini_3_pro_preview_exact():
    cw = resolve_model_context_window("gemini", "gemini-3.1-pro-preview")
    assert cw.known is True
    assert cw.context_window_tokens == 1_048_576
    assert cw.max_output_tokens == 65536


def test_resolve_anthropic_claude_opus_known():
    cw = resolve_model_context_window("anthropic", "claude-opus-4-7")
    assert cw.known is True
    assert cw.context_window_tokens == 200000
    assert cw.max_output_tokens == 8192


def test_resolve_anthropic_claude_3_5_sonnet_family_prefix():
    cw = resolve_model_context_window("anthropic", "claude-3-5-sonnet-20241022")
    assert cw.known is True
    assert cw.context_window_tokens == 200000


def test_resolve_unknown_model_returns_unknown():
    cw = resolve_model_context_window("openai", "totally-made-up-9999")
    assert cw.known is False
    assert cw.source == "unknown"
    assert cw.context_window_tokens is None
    assert cw.max_input_tokens is None
    assert cw.max_output_tokens is None


def test_resolve_unknown_provider_returns_unknown():
    cw = resolve_model_context_window("mystery-provider", "gpt-4o")
    assert cw.known is False
    assert cw.source == "unknown"


# ---------------------------------------------------------------------------
# normalize_context_window_metadata
# ---------------------------------------------------------------------------


def test_normalize_snake_case_input_output_token_limit():
    out = normalize_context_window_metadata(
        "gemini",
        "gemini-2.5-flash",
        {"input_token_limit": 1_000_000, "output_token_limit": 8192},
    )
    assert out is not None
    assert out.known is True
    assert out.source == "provider_api"
    assert out.context_window_tokens == 1_000_000
    assert out.max_input_tokens == 1_000_000
    assert out.max_output_tokens == 8192


def test_normalize_camel_case_input_output_token_limit():
    out = normalize_context_window_metadata(
        "gemini",
        "gemini-2.5-flash",
        {"inputTokenLimit": 999_999, "outputTokenLimit": 16384},
    )
    assert out is not None
    assert out.known is True
    assert out.source == "provider_api"
    assert out.context_window_tokens == 999_999
    assert out.max_input_tokens == 999_999
    assert out.max_output_tokens == 16384


def test_normalize_accepts_context_window_tokens_directly():
    out = normalize_context_window_metadata(
        "openai",
        "gpt-4o",
        {"context_window_tokens": 128000, "max_output_tokens": 16384},
    )
    assert out is not None
    assert out.context_window_tokens == 128000
    assert out.max_input_tokens == 128000
    assert out.max_output_tokens == 16384
    assert out.source == "provider_api"


def test_normalize_camel_max_input_and_output():
    out = normalize_context_window_metadata(
        "openai",
        "gpt-4o",
        {"maxInputTokens": 200000, "maxOutputTokens": 4096},
    )
    assert out is not None
    assert out.max_input_tokens == 200000
    assert out.context_window_tokens == 200000
    assert out.max_output_tokens == 4096


def test_normalize_returns_none_when_no_useful_fields():
    out = normalize_context_window_metadata("openai", "gpt-4o", {"foo": 1})
    assert out is None


def test_normalize_returns_none_for_empty_or_non_dict():
    assert normalize_context_window_metadata("openai", "gpt-4o", None) is None
    assert normalize_context_window_metadata("openai", "gpt-4o", {}) is None


def test_normalize_picks_larger_when_both_context_and_input_present():
    out = normalize_context_window_metadata(
        "openai",
        "gpt-4o",
        {"context_window_tokens": 100000, "max_input_tokens": 128000},
    )
    assert out is not None
    # Should prefer the larger value to avoid under-reporting.
    assert out.context_window_tokens == 128000
    assert out.max_input_tokens == 128000


def test_normalize_explicit_separate_io_keeps_independent_limits():
    out = normalize_context_window_metadata(
        "gemini",
        "gemini-3-pro-image",
        {
            "context_window_tokens": None,
            "max_input_tokens": 65_536,
            "max_output_tokens": 32_768,
            "limit_type": "separate_io",
        },
    )

    assert out is not None
    assert out.known is True
    assert out.source == "provider_api"
    assert out.limit_type == "separate_io"
    assert out.context_window_tokens is None
    assert out.max_input_tokens == 65_536
    assert out.max_output_tokens == 32_768


def test_normalize_explicit_shared_context_keeps_shared_denominator():
    out = normalize_context_window_metadata(
        "openai",
        "gpt-4o",
        {
            "context_window_tokens": 128_000,
            "max_input_tokens": 120_000,
            "max_output_tokens": 16_384,
            "limit_type": "shared_context",
        },
    )

    assert out is not None
    assert out.limit_type == "shared_context"
    assert out.context_window_tokens == 128_000
    assert out.max_input_tokens == 128_000


# ---------------------------------------------------------------------------
# Catalog metadata precedence
# ---------------------------------------------------------------------------


def test_catalog_metadata_provider_api_takes_precedence_over_registry():
    catalog = {
        "context_window_tokens": 1_000_000,
        "max_output_tokens": 4096,
    }
    cw = resolve_model_context_window("openai", "gpt-4o", catalog_metadata=catalog)
    assert cw.known is True
    assert cw.source == "provider_api"
    assert cw.context_window_tokens == 1_000_000
    assert cw.max_output_tokens == 4096


def test_catalog_metadata_camel_keys_take_precedence():
    catalog = {"inputTokenLimit": 750_000, "outputTokenLimit": 16384}
    cw = resolve_model_context_window("gemini", "gemini-2.5-flash", catalog_metadata=catalog)
    assert cw.source == "provider_api"
    assert cw.context_window_tokens == 750_000
    assert cw.max_output_tokens == 16384


def test_catalog_metadata_context_window_known_flag_respected():
    catalog = {"context_window_known": True, "context_window_tokens": 64000}
    cw = resolve_model_context_window("openai", "gpt-4o", catalog_metadata=catalog)
    assert cw.source == "provider_api"
    assert cw.context_window_tokens == 64000


def test_catalog_metadata_without_useful_fields_falls_back_to_registry():
    catalog = {"unrelated": "value"}
    cw = resolve_model_context_window("openai", "gpt-4o", catalog_metadata=catalog)
    assert cw.source == "registry"
    assert cw.context_window_tokens == 128000


# ---------------------------------------------------------------------------
# ModelContextWindow.to_dict
# ---------------------------------------------------------------------------


def test_model_context_window_to_dict_shape():
    cw = resolve_model_context_window("openai", "gpt-4o")
    d = cw.to_dict()
    assert d == {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": 128000,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "limit_type": "shared_context",
        "source": "registry",
        "known": True,
    }


def test_unknown_window_to_dict_has_none_numeric_fields():
    cw = resolve_model_context_window("openai", "totally-made-up-9999")
    d = cw.to_dict()
    assert d["known"] is False
    assert d["source"] == "unknown"
    assert d["context_window_tokens"] is None
    assert d["max_input_tokens"] is None
    assert d["max_output_tokens"] is None


# ---------------------------------------------------------------------------
# build_context_window_usage
# ---------------------------------------------------------------------------


def _shared_cw(*, context_window_tokens: int = 128_000, max_output_tokens: int = 16_384) -> dict:
    return {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": context_window_tokens,
        "max_input_tokens": context_window_tokens,
        "max_output_tokens": max_output_tokens,
        "limit_type": "shared_context",
        "source": "registry",
        "known": True,
    }


def test_build_usage_shared_context_prefers_reported_total():
    """A provider-reported total is the ratio numerator for a shared window."""
    cw = resolve_model_context_window("openai", "gpt-4o").to_dict()
    usage = build_context_window_usage(
        cw,
        NormalizedUsage(
            input_tokens=12_000,
            output_tokens=2_000,
            total_tokens=15_000,
            reasoning_tokens=1_000,
            source="provider_reported",
        ),
    )
    assert usage["used_tokens"] == 15_000
    assert usage["used_token_source"] == "provider_reported_total"
    assert usage["usage_ratio"] == pytest.approx(15_000 / 128_000)
    assert usage["usage_ratio_basis"] == "shared_context_total"
    assert usage["usage_source"] == "provider_reported"
    assert usage["display_state"] == "ok"


def test_build_usage_shared_context_sums_split_when_total_absent():
    """Without a reported total, the known input/output split is summed."""
    cw = resolve_model_context_window("openai", "gpt-4o").to_dict()
    usage = build_context_window_usage(
        cw,
        NormalizedUsage(input_tokens=12_000, output_tokens=None, source="provider_reported"),
    )
    assert usage["used_tokens"] == 12_000
    assert usage["used_token_source"] == "provider_reported_split"
    assert usage["usage_ratio"] == pytest.approx(12_000 / 128_000)
    assert usage["display_state"] == "ok"


def test_build_usage_falls_back_to_estimated_total():
    cw = resolve_model_context_window("openai", "gpt-4o").to_dict()
    usage = build_context_window_usage(
        cw, NormalizedUsage(total_tokens=20_000, source="locally_estimated")
    )
    assert usage["used_tokens"] == 20_000
    assert usage["used_token_source"] == "estimated_total"
    assert usage["usage_ratio"] == pytest.approx(20_000 / 128_000)
    assert usage["usage_source"] == "locally_estimated"
    assert usage["display_state"] == "ok"


def test_build_usage_unknown_when_no_tokens():
    cw = resolve_model_context_window("openai", "gpt-4o").to_dict()
    usage = build_context_window_usage(cw, NormalizedUsage(source="unavailable"))
    assert usage["used_tokens"] is None
    assert usage["used_token_source"] == "unknown"
    assert usage["usage_ratio"] is None
    assert usage["display_state"] == "unknown"


def test_build_usage_unknown_window_retains_counts_without_ratio():
    """An unknown model keeps its raw token counts but invents no ratio."""
    cw = resolve_model_context_window("openai", "totally-made-up-9999").to_dict()
    usage = build_context_window_usage(
        cw,
        NormalizedUsage(input_tokens=8_000, output_tokens=100, source="provider_reported"),
    )
    assert usage["input_tokens"] == 8_000
    assert usage["output_tokens"] == 100
    assert usage["display_state"] == "unknown"
    assert usage["usage_ratio"] is None
    assert usage["used_tokens"] is None


def test_build_usage_none_context_window_returns_unknown():
    usage = build_context_window_usage(
        None, NormalizedUsage(input_tokens=8_000, source="provider_reported")
    )
    assert usage["display_state"] == "unknown"
    assert usage["usage_ratio"] is None
    assert usage["used_tokens"] is None
    assert usage["used_token_source"] == "unknown"
    # Raw counts are still surfaced even without a denominator.
    assert usage["input_tokens"] == 8_000


def test_build_usage_display_state_ok_below_70_percent():
    usage = build_context_window_usage(
        _shared_cw(context_window_tokens=100),
        NormalizedUsage(total_tokens=69, source="provider_reported"),
    )
    assert usage["usage_ratio"] == pytest.approx(0.69)
    assert usage["display_state"] == "ok"


def test_build_usage_display_state_warn_at_70_percent():
    usage = build_context_window_usage(
        _shared_cw(context_window_tokens=100),
        NormalizedUsage(total_tokens=70, source="provider_reported"),
    )
    assert usage["display_state"] == "warn"


def test_build_usage_display_state_warn_at_89_percent():
    usage = build_context_window_usage(
        _shared_cw(context_window_tokens=100),
        NormalizedUsage(total_tokens=89, source="provider_reported"),
    )
    assert usage["display_state"] == "warn"


def test_build_usage_display_state_danger_at_90_percent():
    usage = build_context_window_usage(
        _shared_cw(context_window_tokens=100),
        NormalizedUsage(total_tokens=90, source="provider_reported"),
    )
    assert usage["display_state"] == "danger"


def test_build_usage_shared_context_uses_context_window_denominator():
    """A shared window measures the used total against context_window_tokens."""
    usage = build_context_window_usage(
        _shared_cw(context_window_tokens=200, max_output_tokens=50),
        NormalizedUsage(
            input_tokens=80, output_tokens=20, total_tokens=120, source="provider_reported"
        ),
    )
    assert usage["used_tokens"] == 120
    assert usage["used_token_source"] == "provider_reported_total"
    assert usage["usage_ratio"] == pytest.approx(0.6)


def test_build_usage_falls_back_to_max_input_when_no_context_window():
    cw = {
        "provider": "openai",
        "model": "fake",
        "context_window_tokens": None,
        "max_input_tokens": 1000,
        "max_output_tokens": None,
        "limit_type": "shared_context",
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(
        cw, NormalizedUsage(input_tokens=100, source="provider_reported")
    )
    assert usage["usage_ratio"] == pytest.approx(0.1)


def test_normalized_usage_rejects_negative_counts():
    """Negative counts are rejected upstream (Task 1), so they never reach the
    gauge — the estimator/normalizer maps them to None before this point."""
    with pytest.raises((ValueError, TypeError)):
        NormalizedUsage(input_tokens=-5)


# ---------------------------------------------------------------------------
# Limit-aware gauge (Task 8): separate-I/O, shared-context, unknown, no clamp
# ---------------------------------------------------------------------------


def test_separate_io_context_uses_most_constrained_limit():
    result = build_context_window_usage(
        {
            "provider": "gemini",
            "model": "gemini-3-pro-image",
            "context_window_tokens": None,
            "max_input_tokens": 65_536,
            "max_output_tokens": 32_768,
            "limit_type": "separate_io",
            "known": True,
            "source": "registry",
        },
        NormalizedUsage(
            input_tokens=20_000,
            output_tokens=5_000,
            total_tokens=25_000,
            source="provider_reported",
        ),
    )
    assert result["input_tokens"] == 20_000
    assert result["output_tokens"] == 5_000
    assert result["used_tokens"] == 25_000
    assert result["input_usage_ratio"] == pytest.approx(20_000 / 65_536)
    assert result["output_usage_ratio"] == pytest.approx(5_000 / 32_768)
    assert result["usage_ratio"] == pytest.approx(20_000 / 65_536)
    assert result["usage_ratio_basis"] == "most_constrained_io_limit"


def test_shared_context_model_uses_reported_total():
    result = build_context_window_usage(
        resolve_model_context_window("gemini", "gemini-2.5-flash").to_dict(),
        NormalizedUsage(
            input_tokens=400, output_tokens=100, total_tokens=500, source="provider_reported"
        ),
    )
    assert result["used_tokens"] == 500
    assert result["used_token_source"] == "provider_reported_total"
    assert result["usage_ratio"] == pytest.approx(500 / 1_048_576)
    assert result["usage_ratio_basis"] == "shared_context_total"


def test_openai_image_usage_has_tokens_but_unknown_window():
    cw = resolve_model_context_window("openai", "gpt-image-2").to_dict()
    assert cw["known"] is False
    assert cw["limit_type"] == "unknown"
    result = build_context_window_usage(
        cw,
        NormalizedUsage(
            input_tokens=100, output_tokens=2_000, total_tokens=2_100, source="provider_reported"
        ),
    )
    # Tokens are retained, but there is no denominator to make a ratio.
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 2_000
    assert result["total_tokens"] == 2_100
    assert result["usage_ratio"] is None
    assert result["display_state"] == "unknown"


def test_gemini_3_pro_image_has_65536_input_and_32768_output_limits():
    cw = resolve_model_context_window("gemini", "gemini-3-pro-image")
    assert cw.known is True
    assert cw.limit_type == "separate_io"
    assert cw.context_window_tokens is None
    assert cw.max_input_tokens == 65_536
    assert cw.max_output_tokens == 32_768
    # The deprecated preview alias resolves identically for persisted history.
    alias = resolve_model_context_window("gemini", "gemini-3-pro-image-preview")
    assert alias.limit_type == "separate_io"
    assert alias.max_input_tokens == 65_536
    assert alias.max_output_tokens == 32_768


def test_ratio_is_not_clamped_in_backend():
    """An over-limit usage yields a ratio > 1.0 — the backend never clamps;
    only the drawn gauge is capped at render time."""
    usage = build_context_window_usage(
        _shared_cw(context_window_tokens=100),
        NormalizedUsage(total_tokens=150, source="provider_reported"),
    )
    assert usage["usage_ratio"] == pytest.approx(1.5)
    assert usage["display_state"] == "danger"
