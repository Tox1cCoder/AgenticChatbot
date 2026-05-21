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


def _breakdown(
    *,
    estimated_total: int = 0,
    actual_input: int | None = None,
    actual_output: int | None = None,
) -> dict:
    return {
        "estimated": {
            "system_prompt_tokens": 0,
            "history_tokens": 0,
            "current_turn_tokens": 0,
            "tool_message_tokens": 0,
            "tool_schema_tokens": 0,
            "total_tokens": estimated_total,
        },
        "actual": {
            "input_tokens": actual_input,
            "output_tokens": actual_output,
        },
        "counts": {"history_messages": 0, "tool_messages": 0, "bound_tools": 0},
        "bound_tool_names": [],
    }


def test_build_usage_prefers_actual_input_over_estimated_total():
    cw = resolve_model_context_window("openai", "gpt-4o").to_dict()
    usage = build_context_window_usage(
        cw,
        _breakdown(estimated_total=20_000, actual_input=12_000),
    )
    assert usage["used_tokens"] == 12_000
    assert usage["used_token_source"] == "actual_input"
    assert usage["usage_ratio"] == pytest.approx(12_000 / 128_000)
    assert usage["display_state"] == "ok"


def test_build_usage_falls_back_to_estimated_total():
    cw = resolve_model_context_window("openai", "gpt-4o").to_dict()
    usage = build_context_window_usage(cw, _breakdown(estimated_total=20_000))
    assert usage["used_tokens"] == 20_000
    assert usage["used_token_source"] == "estimated_total"
    assert usage["usage_ratio"] == pytest.approx(20_000 / 128_000)
    assert usage["display_state"] == "ok"


def test_build_usage_unknown_when_no_tokens():
    cw = resolve_model_context_window("openai", "gpt-4o").to_dict()
    usage = build_context_window_usage(cw, _breakdown())
    assert usage["used_tokens"] is None
    assert usage["used_token_source"] == "unknown"
    assert usage["usage_ratio"] is None
    assert usage["display_state"] == "unknown"


def test_build_usage_unknown_window_returns_unknown_state():
    cw = resolve_model_context_window("openai", "totally-made-up-9999").to_dict()
    usage = build_context_window_usage(
        cw,
        _breakdown(estimated_total=10_000, actual_input=8_000),
    )
    assert usage["display_state"] == "unknown"
    assert usage["usage_ratio"] is None


def test_build_usage_none_context_window_returns_unknown():
    usage = build_context_window_usage(None, _breakdown(actual_input=8_000))
    assert usage["display_state"] == "unknown"
    assert usage["usage_ratio"] is None
    assert usage["used_tokens"] is None
    assert usage["used_token_source"] == "unknown"


def test_build_usage_display_state_ok_below_70_percent():
    # 0.69 -> ok
    cw = {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": 100,
        "max_input_tokens": 100,
        "max_output_tokens": 16,
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(cw, _breakdown(actual_input=69))
    assert usage["usage_ratio"] == pytest.approx(0.69)
    assert usage["display_state"] == "ok"


def test_build_usage_display_state_warn_at_70_percent():
    cw = {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": 100,
        "max_input_tokens": 100,
        "max_output_tokens": 16,
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(cw, _breakdown(actual_input=70))
    assert usage["display_state"] == "warn"


def test_build_usage_display_state_warn_at_89_percent():
    cw = {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": 100,
        "max_input_tokens": 100,
        "max_output_tokens": 16,
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(cw, _breakdown(actual_input=89))
    assert usage["display_state"] == "warn"


def test_build_usage_display_state_danger_at_90_percent():
    cw = {
        "provider": "openai",
        "model": "gpt-4o",
        "context_window_tokens": 100,
        "max_input_tokens": 100,
        "max_output_tokens": 16,
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(cw, _breakdown(actual_input=90))
    assert usage["display_state"] == "danger"


def test_build_usage_uses_max_input_tokens_for_ratio():
    """When max_input_tokens differs from context_window_tokens, prefer max_input_tokens."""
    cw = {
        "provider": "openai",
        "model": "fake",
        "context_window_tokens": 200,
        "max_input_tokens": 100,
        "max_output_tokens": 16,
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(cw, _breakdown(actual_input=50))
    assert usage["usage_ratio"] == pytest.approx(0.5)


def test_build_usage_falls_back_to_context_window_when_no_max_input():
    cw = {
        "provider": "openai",
        "model": "fake",
        "context_window_tokens": 1000,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(cw, _breakdown(actual_input=100))
    assert usage["usage_ratio"] == pytest.approx(0.1)


def test_build_usage_negative_actual_input_falls_back_to_unknown():
    cw = {
        "provider": "openai",
        "model": "fake",
        "context_window_tokens": 100,
        "max_input_tokens": 100,
        "max_output_tokens": 16,
        "source": "registry",
        "known": True,
    }
    usage = build_context_window_usage(cw, _breakdown(actual_input=-5))
    assert usage["used_tokens"] is None
    assert usage["used_token_source"] == "unknown"
    assert usage["display_state"] == "unknown"
