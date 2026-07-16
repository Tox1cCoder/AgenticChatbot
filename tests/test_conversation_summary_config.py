from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(**overrides) -> Settings:
    values = {
        "secret_key": "test-secret-key-with-at-least-32-bytes",
        "environment": "development",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_conversation_summary_defaults_are_stable():
    settings = _settings()

    assert settings.conversation_summary_enabled is True
    assert settings.conversation_summary_provider == "gemini"
    assert settings.conversation_summary_model == "gemini-2.5-flash"
    assert settings.conversation_summary_trigger_messages == 60
    assert settings.conversation_summary_trigger_tokens == 18_000
    assert settings.conversation_summary_soft_context_ratio == pytest.approx(0.70)
    assert settings.conversation_summary_hard_context_ratio == pytest.approx(0.85)
    assert settings.conversation_summary_keep_recent_turns == 4
    assert settings.conversation_summary_max_tokens == 1_500
    assert settings.conversation_summary_timeout_seconds == 30
    assert settings.conversation_summary_max_attempts == 5
    assert settings.conversation_summary_lease_seconds == 120
    assert settings.conversation_summary_retry_base_seconds == 5
    assert settings.conversation_summary_retry_max_seconds == 900
    assert settings.conversation_summary_reconcile_seconds == 60
    assert settings.conversation_summary_safety_margin_tokens == 1_024
    assert settings.conversation_summary_default_reserved_output_tokens == 4_096


@pytest.mark.parametrize(
    ("disabled_field", "enabled_field"),
    [
        ("conversation_summary_trigger_messages", "conversation_summary_trigger_tokens"),
        ("conversation_summary_trigger_tokens", "conversation_summary_trigger_messages"),
    ],
)
def test_zero_disables_only_one_background_threshold(disabled_field, enabled_field):
    settings = _settings(**{disabled_field: 0})

    assert getattr(settings, disabled_field) == 0
    assert getattr(settings, enabled_field) > 0


def test_enabled_summary_rejects_both_background_thresholds_disabled():
    with pytest.raises(ValidationError, match="at least one background threshold"):
        _settings(
            conversation_summary_trigger_messages=0,
            conversation_summary_trigger_tokens=0,
        )


def test_disabled_summary_allows_both_background_thresholds_disabled():
    settings = _settings(
        conversation_summary_enabled=False,
        conversation_summary_trigger_messages=0,
        conversation_summary_trigger_tokens=0,
    )

    assert settings.conversation_summary_enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("conversation_summary_trigger_messages", -1),
        ("conversation_summary_trigger_tokens", -1),
        ("conversation_summary_keep_recent_turns", -1),
        ("conversation_summary_safety_margin_tokens", -1),
        ("conversation_summary_default_reserved_output_tokens", -1),
    ],
)
def test_summary_budget_fields_must_be_non_negative(field, value):
    with pytest.raises(ValidationError, match="non-negative"):
        _settings(**{field: value})


@pytest.mark.parametrize("value", [-1, 0])
def test_non_positive_summary_token_cap_is_rejected(value):
    with pytest.raises(ValidationError, match="positive"):
        _settings(conversation_summary_max_tokens=value)


@pytest.mark.parametrize(
    "field",
    [
        "conversation_summary_timeout_seconds",
        "conversation_summary_max_attempts",
        "conversation_summary_lease_seconds",
        "conversation_summary_retry_base_seconds",
        "conversation_summary_retry_max_seconds",
        "conversation_summary_reconcile_seconds",
    ],
)
def test_summary_operational_intervals_must_be_positive(field):
    with pytest.raises(ValidationError, match="positive"):
        _settings(**{field: 0})


@pytest.mark.parametrize(
    ("soft", "hard"),
    [
        (0, 0.85),
        (0.70, 1),
        (0.85, 0.85),
        (0.90, 0.85),
    ],
)
def test_summary_context_ratios_must_be_strictly_ordered(soft, hard):
    with pytest.raises(ValidationError, match="soft context ratio"):
        _settings(
            conversation_summary_soft_context_ratio=soft,
            conversation_summary_hard_context_ratio=hard,
        )


def test_keep_recent_turns_must_be_below_enabled_message_threshold():
    with pytest.raises(ValidationError, match="keep recent turns"):
        _settings(
            conversation_summary_keep_recent_turns=4,
            conversation_summary_trigger_messages=4,
        )


def test_retry_cap_must_not_be_below_base_delay():
    with pytest.raises(ValidationError, match="retry max"):
        _settings(
            conversation_summary_retry_base_seconds=10,
            conversation_summary_retry_max_seconds=5,
        )


@pytest.mark.parametrize(
    ("provider", "model", "message"),
    [
        ("", "gemini-2.5-flash", "provider"),
        ("gemini", "", "model"),
        ("gemini", "gemini-3-flash-preview", "stable"),
    ],
)
def test_production_requires_explicit_stable_compaction_model(provider, model, message):
    with pytest.raises(ValidationError, match=message):
        _settings(
            environment="production",
            conversation_summary_provider=provider,
            conversation_summary_model=model,
        )
