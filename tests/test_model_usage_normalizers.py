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


# ---------------------------------------------------------------------------
# Fix round 1 — coverage gaps that let the output-image and cache-fallback
# bugs ship. See task-4-report.md "Fix round 1" for the root-cause writeup.
# ---------------------------------------------------------------------------


def test_openai_output_image_tokens_maps_from_output_tokens_details_image_tokens():
    """output_tokens_details.image_tokens is a required OpenAI shape (per the
    brief) but was previously unreachable — output_image_tokens was only ever
    populated from Gemini's candidates_tokens_details modality array."""
    usage = normalize_provider_usage(
        provider="openai",
        payload={
            "usage": {
                "input_tokens": 40,
                "output_tokens": 300,
                "total_tokens": 340,
                "output_tokens_details": {"text_tokens": 100, "image_tokens": 200},
            }
        },
    )
    assert usage == NormalizedUsage(
        input_tokens=40,
        output_tokens=300,
        total_tokens=340,
        output_text_tokens=100,
        output_image_tokens=200,
        source="provider_reported",
    )


def test_gemini_candidates_modality_image_entry_maps_to_output_image_tokens():
    """The only path that sets output_image_tokens from a Gemini modality
    array was previously entirely unexercised by any test."""
    payload = {
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 300,
            "candidates_tokens_details": [
                {"modality": "TEXT", "token_count": 50},
                {"modality": "IMAGE", "token_count": 250},
            ],
        }
    }
    usage = normalize_provider_usage(provider="gemini", payload=payload)
    assert usage.output_text_tokens == 50
    assert usage.output_image_tokens == 250


def test_cached_input_tokens_falls_back_to_langchain_input_token_details():
    """LangChain's standardized InputTokenDetails shape (cache_read /
    cache_creation nested under input_token_details, singular) must combine
    into cached_input_tokens the same way extract_reported_usage already
    does — the two must share one implementation, not two drifting copies."""
    usage = normalize_provider_usage(
        provider="anthropic",
        payload={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 20,
                "input_token_details": {"cache_read": 120, "cache_creation": 30},
            }
        },
    )
    assert usage.cached_input_tokens == 150


def test_cached_input_tokens_falls_back_to_prompt_tokens_details_cached_tokens():
    """OpenAI chat-completions raw shape (prompt_tokens_details.cached_tokens)
    must also resolve to cached_input_tokens."""
    usage = normalize_provider_usage(
        provider="openai",
        payload={
            "usage": {
                "input_tokens": 700,
                "output_tokens": 25,
                "prompt_tokens_details": {"cached_tokens": 90},
            }
        },
    )
    assert usage.cached_input_tokens == 90


def test_usage_found_only_in_response_metadata_token_usage_normalizes():
    """The recorder (Task 5) hands the normalizer raw LangChain response
    objects whose usage sometimes lives only in response_metadata — the
    normalizer must search the same envelope chain extract_reported_usage
    does, not just top-level usage/usage_metadata."""
    payload = SimpleNamespace(
        response_metadata={"token_usage": {"prompt_tokens": 640, "completion_tokens": 32}}
    )
    usage = normalize_provider_usage(provider="openai", payload=payload)
    assert usage.input_tokens == 640
    assert usage.output_tokens == 32
    assert usage.source == "provider_reported"


def test_malformed_modality_list_entries_do_not_crash():
    """A non-dict list entry and an entry missing the 'modality' key must
    both be skipped as unknown, never raise."""
    payload = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "prompt_tokens_details": ["not-a-mapping", {"token_count": 4}],
        }
    }
    usage = normalize_provider_usage(provider="gemini", payload=payload)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.input_text_tokens is None
    assert usage.input_image_tokens is None
