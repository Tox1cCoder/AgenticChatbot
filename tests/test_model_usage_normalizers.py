"""Tests for ``app.usage.normalizers.normalize_provider_usage``.

``normalize_provider_usage`` is a pure function: it converts a raw provider
usage payload (a dict or a response-like object) into a ``NormalizedUsage``
without ever retaining a reference to the input payload, and without
synthesizing a ``total_tokens`` the provider did not report (that precedence
decision belongs to later context-ratio consumers, not this normalizer).
"""

from __future__ import annotations

from types import SimpleNamespace

from app.usage.normalizers import normalize_provider_usage
from app.usage.types import NormalizedUsage


def test_openai_completion_usage_with_modality_details_normalizes():
    """Verbatim example from the task brief."""
    usage = normalize_provider_usage(
        provider="openai",
        payload={
            "usage": {
                "input_tokens": 50,
                "output_tokens": 100,
                "total_tokens": 150,
                "input_tokens_details": {"text_tokens": 10, "image_tokens": 40},
            }
        },
    )
    assert usage == NormalizedUsage(
        input_tokens=50,
        output_tokens=100,
        total_tokens=150,
        input_text_tokens=10,
        input_image_tokens=40,
        source="provider_reported",
    )


def test_openai_usage_maps_cached_output_text_and_reasoning_tokens():
    usage = normalize_provider_usage(
        provider="openai",
        payload={
            "usage": {
                "input_tokens": 80,
                "output_tokens": 200,
                "total_tokens": 280,
                "input_tokens_details": {
                    "text_tokens": 60,
                    "image_tokens": 0,
                    "cached_tokens": 20,
                },
                "output_tokens_details": {"text_tokens": 180, "reasoning_tokens": 20},
            }
        },
    )
    assert usage == NormalizedUsage(
        input_tokens=80,
        output_tokens=200,
        total_tokens=280,
        reasoning_tokens=20,
        cached_input_tokens=20,
        input_text_tokens=60,
        input_image_tokens=0,
        output_text_tokens=180,
        source="provider_reported",
    )


def test_gemini_usage_metadata_normalizes_counts_and_modality_arrays():
    payload = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=500,
            candidates_token_count=120,
            total_token_count=650,
            thoughts_token_count=30,
            cached_content_token_count=80,
            prompt_tokens_details=[
                SimpleNamespace(modality="TEXT", token_count=450),
                SimpleNamespace(modality="IMAGE", token_count=50),
            ],
            candidates_tokens_details=[{"modality": "TEXT", "token_count": 120}],
        )
    )

    usage = normalize_provider_usage(provider="gemini", payload=payload)

    assert usage == NormalizedUsage(
        input_tokens=500,
        output_tokens=120,
        total_tokens=650,
        reasoning_tokens=30,
        cached_input_tokens=80,
        input_text_tokens=450,
        input_image_tokens=50,
        output_text_tokens=120,
        source="provider_reported",
    )


def test_anthropic_style_cache_tokens_combine_into_cached_input_tokens():
    usage = normalize_provider_usage(
        provider="anthropic",
        payload={
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 50,
                "cache_read_input_tokens": 400,
                "cache_creation_input_tokens": 100,
            }
        },
    )
    assert usage == NormalizedUsage(
        input_tokens=1000,
        output_tokens=50,
        cached_input_tokens=500,
        source="provider_reported",
    )


def test_totals_missing_leaves_total_none_when_input_and_output_present():
    """Deliberate: normalize_provider_usage never derives total = input + output.

    Task 8's context-ratio precedence must be able to tell a provider-reported
    total apart from a derived one, so a missing total must stay None here
    (unlike TokenCounter.extract_reported_usage, which does synthesize it).
    """
    usage = normalize_provider_usage(
        provider="anthropic",
        payload={"usage": {"input_tokens": 300, "output_tokens": 40}},
    )
    assert usage == NormalizedUsage(
        input_tokens=300,
        output_tokens=40,
        total_tokens=None,
        source="provider_reported",
    )
    assert usage.total_tokens is None


def test_negative_values_are_treated_as_unknown_not_crashed():
    usage = normalize_provider_usage(
        provider="openai",
        payload={"usage": {"input_tokens": -5, "output_tokens": 20}},
    )
    assert usage.input_tokens is None
    assert usage.output_tokens == 20
    assert usage.source == "provider_reported"


def test_all_negative_values_yield_unavailable_source():
    usage = normalize_provider_usage(
        provider="openai",
        payload={"usage": {"input_tokens": -5, "output_tokens": -1}},
    )
    assert usage == NormalizedUsage(source="unavailable")


def test_boolean_values_are_treated_as_unknown_not_counted():
    usage = normalize_provider_usage(
        provider="openai",
        payload={"usage": {"input_tokens": True, "output_tokens": 10}},
    )
    assert usage.input_tokens is None
    assert usage.output_tokens == 10


def test_malformed_non_mapping_envelope_returns_unavailable():
    usage = normalize_provider_usage(provider="openai", payload={"usage": "not-a-mapping"})
    assert usage == NormalizedUsage(source="unavailable")


def test_fully_absent_usage_returns_unavailable_source():
    assert normalize_provider_usage(provider="openai", payload={}) == NormalizedUsage(
        source="unavailable"
    )
    assert normalize_provider_usage(provider="gemini", payload=SimpleNamespace()) == (
        NormalizedUsage(source="unavailable")
    )


def test_non_mapping_payload_never_crashes():
    assert normalize_provider_usage(provider="openai", payload="garbage") == NormalizedUsage(
        source="unavailable"
    )
    assert normalize_provider_usage(provider="openai", payload=None) == NormalizedUsage(
        source="unavailable"
    )


def test_provider_argument_does_not_gate_shape_detection():
    """Alias-based shape detection is provider-agnostic, mirroring
    TokenCounter.extract_reported_usage's existing design."""
    usage = normalize_provider_usage(
        provider="unspecified",
        payload={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
    )
    assert usage.source == "provider_reported"
    assert usage.input_tokens == 10


def test_generated_images_defaults_to_zero_not_populated_by_normalizer():
    usage = normalize_provider_usage(
        provider="openai",
        payload={"usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}},
    )
    assert usage.generated_images == 0


def test_usage_metadata_envelope_takes_precedence_over_usage_when_both_present():
    payload = {
        "usage_metadata": {"input_tokens": 111, "output_tokens": 22},
        "usage": {"input_tokens": 999, "output_tokens": 888},
    }
    usage = normalize_provider_usage(provider="openai", payload=payload)
    assert usage.input_tokens == 111
    assert usage.output_tokens == 22


def test_normalize_provider_usage_does_not_retain_payload_reference():
    payload = {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
    usage = normalize_provider_usage(provider="openai", payload=payload)

    payload["usage"]["input_tokens"] = 999
    payload["usage"]["total_tokens"] = 999

    assert usage.input_tokens == 10
    assert usage.total_tokens == 15
