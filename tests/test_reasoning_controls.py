from __future__ import annotations

import pytest

from app.ai.reasoning_controls import resolve_reasoning_control, validate_reasoning_effort


def test_gemini_36_flash_uses_native_levels() -> None:
    control = resolve_reasoning_control("gemini", "gemini-3.6-flash", supports_reasoning=True)
    assert control.parameter_name == "thinking_level"
    assert control.levels == ("minimal", "low", "medium", "high")
    assert control.default_level == "medium"


def test_gemini_31_pro_does_not_offer_minimal() -> None:
    control = resolve_reasoning_control("gemini", "gemini-3.1-pro-preview")
    assert control.levels == ("low", "medium", "high")


def test_openai_56_retains_max() -> None:
    control = resolve_reasoning_control("openai", "gpt-5.6-sol")
    assert control.parameter_name == "reasoning.effort"
    assert control.levels == ("none", "low", "medium", "high", "xhigh", "max")


def test_openai_pro_rule_precedes_base_family() -> None:
    control = resolve_reasoning_control("openai", "gpt-5.4-pro")
    assert control.levels == ("medium", "high", "xhigh")


def test_unknown_reasoning_model_is_provider_default_only() -> None:
    control = resolve_reasoning_control("openai", "gpt-future", supports_reasoning=True)
    assert control.supported is True
    assert control.parameter_name is None
    assert control.levels == ()
    assert (
        validate_reasoning_effort(
            "openai", "gpt-future", None, supports_reasoning=True
        )
        is None
    )


def test_invalid_native_value_is_not_remapped() -> None:
    with pytest.raises(ValueError, match="Accepted: low, medium, high"):
        validate_reasoning_effort("gemini", "gemini-3.1-pro-preview", "minimal")


def test_native_value_is_normalized_without_translation() -> None:
    assert validate_reasoning_effort("openai", "gpt-5.6", " MAX ") == "max"
