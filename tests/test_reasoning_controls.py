from __future__ import annotations

import pytest

from app.ai.reasoning_controls import (
    gemini_reasoning_kwargs,
    resolve_reasoning_control,
    validate_reasoning_effort,
)


def test_gemini_36_flash_uses_native_levels() -> None:
    control = resolve_reasoning_control("gemini", "gemini-3.6-flash", supports_reasoning=True)
    assert control.parameter_name == "thinking_level"
    assert control.levels == ("minimal", "low", "medium", "high")
    assert control.default_level == "medium"


def test_gemini_31_pro_does_not_offer_minimal() -> None:
    control = resolve_reasoning_control("gemini", "gemini-3.1-pro-preview")
    assert control.levels == ("low", "medium", "high")


@pytest.mark.parametrize(
    ("model", "levels", "parameter"),
    [
        ("gemini-2.5-pro", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-2.5-flash", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-2.5-flash-lite", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-pro-latest", ("low", "medium", "high"), "thinking_level"),
        ("gemini-flash-latest", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-flash-lite-latest", ("low", "medium", "high"), "thinking_budget"),
    ],
)
def test_gemini_compatible_families_expose_native_levels(
    model: str, levels: tuple[str, ...], parameter: str
) -> None:
    control = resolve_reasoning_control("gemini", model, supports_reasoning=True)
    assert control.levels == levels
    assert control.parameter_name == parameter


def test_openai_56_retains_max() -> None:
    control = resolve_reasoning_control("openai", "gpt-5.6-sol")
    assert control.parameter_name == "reasoning.effort"
    assert control.levels == ("none", "low", "medium", "high", "xhigh", "max")


def test_openai_pro_rule_precedes_base_family() -> None:
    control = resolve_reasoning_control("openai", "gpt-5.4-pro")
    assert control.levels == ("medium", "high", "xhigh")


@pytest.mark.parametrize("model", ["o1", "o3", "o3-mini", "o4-mini"])
def test_openai_o_series_exposes_reasoning_effort(model: str) -> None:
    control = resolve_reasoning_control("openai", model, supports_reasoning=True)
    assert control.parameter_name == "reasoning.effort"
    assert control.levels == ("low", "medium", "high")


def test_openai_o1_pro_is_not_broadened_by_o1_family() -> None:
    control = resolve_reasoning_control("openai", "o1-pro", supports_reasoning=True)
    assert control.levels == ("high",)


def test_unknown_reasoning_model_is_provider_default_only() -> None:
    control = resolve_reasoning_control("openai", "gpt-future", supports_reasoning=True)
    assert control.supported is True
    assert control.parameter_name is None
    assert control.levels == ()
    assert validate_reasoning_effort("openai", "gpt-future", None, supports_reasoning=True) is None


def test_unknown_gemini_reasoning_model_is_provider_default_only() -> None:
    control = resolve_reasoning_control("gemini", "gemini-unknown-latest", supports_reasoning=True)
    assert control.supported is True
    assert control.parameter_name is None
    assert control.levels == ()


def test_invalid_native_value_is_not_remapped() -> None:
    with pytest.raises(ValueError, match="Accepted: low, medium, high"):
        validate_reasoning_effort("gemini", "gemini-3.1-pro-preview", "minimal")


def test_native_value_is_normalized_without_translation() -> None:
    assert validate_reasoning_effort("openai", "gpt-5.6", " MAX ") == "max"


def test_gemini_25_named_level_uses_documented_budget() -> None:
    assert gemini_reasoning_kwargs("gemini-2.5-flash", "medium") == {"thinking_budget": 8192}


def test_gemini_3_named_level_is_unchanged() -> None:
    assert gemini_reasoning_kwargs("gemini-3.6-flash", "high") == {"thinking_level": "high"}
