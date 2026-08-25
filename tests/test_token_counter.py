from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.token_counter import EphemeralTokenCounterStore, ReportedTokenUsage, TokenCounter


def test_ephemeral_counter_store_is_one_shot_bounded_and_expiring(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("app.ai.token_counter.time.monotonic", lambda: clock[0])
    store = EphemeralTokenCounterStore(max_entries=2, ttl_seconds=5)
    first = TokenCounter()
    second = TokenCounter()
    third = TokenCounter()

    first_ref = store.put(first)
    second_ref = store.put(second)
    third_ref = store.put(third)

    assert store.take(first_ref) is None
    assert store.take(second_ref) is second
    assert store.take(second_ref) is None
    assert len(store) == 1
    clock[0] = 106.0
    assert store.take(third_ref) is None
    assert len(store) == 0


def test_openai_uses_model_tokenizer_for_text():
    counter = TokenCounter()
    text = "Conversation compaction counts the actual model tokens."

    result = counter.count_text(provider="openai", model="gpt-4o", text=text)

    expected = len(tiktoken.encoding_for_model("gpt-4o").encode(text))
    assert result.tokens == expected
    assert result.strategy.startswith("openai:tiktoken:")
    assert result.source == "local"


def test_openai_unknown_model_uses_explicit_family_fallback():
    counter = TokenCounter()

    result = counter.count_text(
        provider="openai",
        model="private-gpt-4o-deployment",
        text="fallback encoding",
    )

    assert result.tokens > 0
    assert result.strategy == "openai:tiktoken:o200k_base:fallback"


def test_gemini_local_fallback_uses_utf8_byte_upper_bound():
    counter = TokenCounter()
    text = "การสรุปบทสนทนาต้องนับโทเค็นภาษาไทยอย่างระมัดระวัง"

    result = counter.count_text(provider="gemini", model="gemini-2.5-flash", text=text)

    assert result.tokens == len(text.encode("utf-8"))
    old_ascii_heuristic = math.floor(len(text) / 4)
    assert result.tokens > old_ascii_heuristic
    assert result.strategy == "gemini:utf8_byte_upper_bound"


@pytest.mark.asyncio
async def test_native_text_counter_is_reserved_for_the_exact_final_count() -> None:
    """``count_text`` is the local fit-check path; ``count_text_exact`` is the RPC.

    Evidence packing calls ``count_text`` once per incremental fit test, so a
    provider round trip there multiplies network calls by the candidate count.
    """
    calls: list[tuple[str, str]] = []

    async def native_text(*, model: str, text: str) -> int:
        calls.append((model, text))
        return 7

    counter = TokenCounter(native_text_counters={"gemini": native_text})

    local = counter.count_text(
        provider="gemini",
        model="gemini-2.5-flash",
        text="encoded evidence",
    )

    assert local.source == "local"
    assert local.strategy == "gemini:utf8_byte_upper_bound"
    assert calls == []

    exact = await counter.count_text_exact(
        provider="gemini",
        model="gemini-2.5-flash",
        text="encoded evidence",
    )

    assert exact.tokens == 7
    assert exact.strategy == "gemini:native_text"
    assert exact.source == "provider"
    assert calls == [("gemini-2.5-flash", "encoded evidence")]


@pytest.mark.asyncio
async def test_exact_text_count_falls_back_locally_when_the_provider_call_fails() -> None:
    async def native_text(*, model: str, text: str) -> int:
        del model, text
        raise TimeoutError("provider unavailable")

    counter = TokenCounter(native_text_counters={"gemini": native_text})

    exact = await counter.count_text_exact(
        provider="gemini",
        model="gemini-2.5-flash",
        text="encoded evidence",
    )

    assert exact.source == "local"
    assert exact.strategy == "gemini:utf8_byte_upper_bound"
    assert exact.tokens == len(b"encoded evidence")


def test_anthropic_uses_conservative_local_estimate():
    counter = TokenCounter()

    result = counter.count_text(provider="anthropic", model="claude-sonnet-4", text="hello")

    assert result.tokens == 2
    assert result.strategy == "anthropic:utf8_bytes_div_3"


def test_unknown_provider_uses_utf8_byte_upper_bound():
    counter = TokenCounter()
    text = "ไทย"

    result = counter.count_text(provider="custom", model="private-v1", text=text)

    assert result.tokens == len(text.encode("utf-8"))
    assert result.strategy == "custom:utf8_byte_upper_bound"


def test_tool_schema_serialization_is_canonical():
    counter = TokenCounter()
    first = {
        "name": "lookup",
        "description": "Look up a record",
        "parameters": {"type": "object", "properties": {"b": {}, "a": {}}},
    }
    reordered = {
        "parameters": {"properties": {"a": {}, "b": {}}, "type": "object"},
        "description": "Look up a record",
        "name": "lookup",
    }

    first_count = counter.count_tools(provider="openai", model="gpt-4o", tools=[first])
    second_count = counter.count_tools(provider="openai", model="gpt-4o", tools=[reordered])

    assert first_count.tokens == second_count.tokens
    assert counter.canonical_json(first) == counter.canonical_json(reordered)


def test_message_count_includes_tool_call_and_tool_result_identity():
    counter = TokenCounter()
    plain = [HumanMessage(content="Use the calculator")]
    with_tools = [
        *plain,
        AIMessage(
            content="",
            tool_calls=[{"name": "calculator", "args": {"x": 2}, "id": "call-7"}],
        ),
        ToolMessage(content="4", name="calculator", tool_call_id="call-7"),
    ]

    plain_count = counter.count_messages(provider="openai", model="gpt-4o", messages=plain)
    tool_count = counter.count_messages(
        provider="openai",
        model="gpt-4o",
        messages=with_tools,
    )

    assert tool_count.tokens > plain_count.tokens


def test_openai_image_estimate_uses_dimensions_and_detail():
    counter = TokenCounter()

    low = counter.count_attachments(
        provider="openai",
        model="gpt-4o",
        attachments=[{"mime_type": "image/png", "width": 1024, "height": 1024, "detail": "low"}],
    )
    high = counter.count_attachments(
        provider="openai",
        model="gpt-4o",
        attachments=[{"mime_type": "image/png", "width": 1024, "height": 1024, "detail": "high"}],
    )

    assert low.tokens == 85
    assert high.tokens == 765


def test_image_without_dimensions_uses_conservative_fallback():
    counter = TokenCounter()

    result = counter.count_attachments(
        provider="gemini",
        model="gemini-2.5-flash",
        attachments=[{"mime_type": "image/jpeg", "filename": "photo.jpg"}],
    )

    assert result.tokens >= 1_200


def test_request_breakdown_includes_all_components_and_reservations():
    counter = TokenCounter()

    result = counter.estimate_request(
        provider="openai",
        model="gpt-4o",
        messages=[HumanMessage(content="Inspect the image")],
        tools=[{"name": "inspect", "parameters": {"type": "object"}}],
        attachments=[{"mime_type": "image/png", "width": 512, "height": 512}],
        reserved_output_tokens=1_000,
        safety_margin_tokens=200,
    )

    assert result.message_tokens > 0
    assert result.tool_tokens > 0
    assert result.attachment_tokens > 0
    assert result.input_tokens == (
        result.message_tokens + result.tool_tokens + result.attachment_tokens
    )
    assert result.total_tokens == result.input_tokens + 1_000 + 200
    assert result.content_class == "mixed"


@pytest.mark.asyncio
async def test_authoritative_provider_count_replaces_local_input_count():
    calls = []

    async def native_counter(**request):
        calls.append(request)
        return 321

    counter = TokenCounter(native_counters={"gemini": native_counter})

    result = await counter.count_request(
        provider="gemini",
        model="gemini-2.5-flash",
        messages=[HumanMessage(content="near the boundary")],
        tools=[],
        attachments=[],
        reserved_output_tokens=100,
        safety_margin_tokens=20,
        authoritative=True,
    )

    assert result.input_tokens == 321
    assert result.total_tokens == 441
    assert result.source == "provider"
    assert result.strategy == "gemini:native_count"
    assert calls[0]["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_authoritative_request_without_native_counter_falls_back_locally():
    counter = TokenCounter()

    result = await counter.count_request(
        provider="anthropic",
        model="claude-sonnet-4",
        messages=[HumanMessage(content="fallback")],
        authoritative=True,
    )

    assert result.input_tokens > 0
    assert result.source == "local"


@pytest.mark.parametrize(
    ("provider", "response", "expected"),
    [
        (
            "openai",
            SimpleNamespace(
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    }
                }
            ),
            (100, 20, 120, None),
        ),
        (
            "gemini",
            SimpleNamespace(
                usage_metadata={
                    "prompt_token_count": 200,
                    "candidates_token_count": 30,
                    "total_token_count": 240,
                    "thoughts_token_count": 10,
                }
            ),
            (200, 30, 240, 10),
        ),
        (
            "anthropic",
            SimpleNamespace(usage_metadata={"input_tokens": 300, "output_tokens": 40}),
            (300, 40, 340, None),
        ),
    ],
)
def test_extract_reported_usage_handles_provider_shapes(provider, response, expected):
    usage = TokenCounter().extract_reported_usage(provider=provider, response=response)

    assert usage is not None
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.reasoning_tokens,
    ) == expected
    assert usage.source == "reported"


def test_extract_reported_usage_rejects_empty_or_negative_values():
    response = SimpleNamespace(usage_metadata={"input_tokens": -1, "output_tokens": None})

    assert TokenCounter().extract_reported_usage(provider="anthropic", response=response) is None


def test_extract_reported_usage_preserves_provider_cost_metadata():
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 20,
            "total_cost": "0.0042",
            "currency": "USD",
        }
    )

    usage = TokenCounter().extract_reported_usage(provider="openai", response=response)

    assert usage is not None
    assert usage.cost_amount == pytest.approx(0.0042)
    assert usage.cost_currency == "USD"


def test_extract_reported_usage_populates_cached_and_modality_details_when_present():
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "input_tokens_details": {"text_tokens": 70, "image_tokens": 30, "cached_tokens": 20},
            "output_tokens_details": {"text_tokens": 35},
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
    )

    usage = TokenCounter().extract_reported_usage(provider="openai", response=response)

    assert usage is not None
    assert usage.cached_input_tokens == 20
    assert usage.input_text_tokens == 70
    assert usage.input_image_tokens == 30
    assert usage.output_text_tokens == 35
    assert usage.reasoning_tokens == 5


def test_extract_reported_usage_sums_anthropic_style_cache_tokens():
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 500,
            "output_tokens": 50,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 100,
        }
    )

    usage = TokenCounter().extract_reported_usage(provider="anthropic", response=response)

    assert usage is not None
    assert usage.cached_input_tokens == 400


def test_extract_reported_usage_sums_langchain_input_token_details_cache_fields():
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 500,
            "output_tokens": 50,
            "input_token_details": {"cache_read": 120, "cache_creation": 30},
        }
    )

    usage = TokenCounter().extract_reported_usage(provider="anthropic", response=response)

    assert usage is not None
    assert usage.cached_input_tokens == 150


def test_extract_reported_usage_new_fields_default_to_none_when_absent():
    response = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5})

    usage = TokenCounter().extract_reported_usage(provider="openai", response=response)

    assert usage is not None
    assert usage.cached_input_tokens is None
    assert usage.input_text_tokens is None
    assert usage.input_image_tokens is None
    assert usage.output_text_tokens is None
    assert usage.output_image_tokens is None


def test_extract_reported_usage_ignores_plural_output_tokens_details_reasoning():
    """Regression (Fix round 2): extract_reported_usage's reasoning
    nested-path search must stay exactly the 3 pre-Task-4 paths
    (output_token_details.reasoning, output_token_details.reasoning_tokens,
    completion_tokens_details.reasoning_tokens — all singular "token[_]").
    The plural output_tokens_details.reasoning_tokens shape is
    normalizer-only; extract_reported_usage must report None for it, exactly
    as it did before Task 4, reachable via token_instrumentation.py's
    extract_actual_usage path."""
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 7},
        }
    )

    usage = TokenCounter().extract_reported_usage(provider="openai", response=response)

    assert usage is not None
    assert usage.reasoning_tokens is None


def test_reported_token_usage_positional_construction_stays_backward_compatible():
    """Existing positional construction (input, output, total, reasoning)
    must keep working after the new fields are appended with defaults."""
    usage = ReportedTokenUsage(100, 20, 120, None)

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 20, 120)
    assert usage.reasoning_tokens is None
    assert usage.cached_input_tokens is None
    assert usage.input_text_tokens is None
    assert usage.input_image_tokens is None
    assert usage.output_text_tokens is None
    assert usage.output_image_tokens is None
    assert usage.source == "reported"


# ------------------------------------------------------------------
# Fit bound versus trigger estimate
# ------------------------------------------------------------------

_THAI_SENTENCE = "การสรุปบทสนทนาต้องนับโทเค็นภาษาไทยอย่างระมัดระวัง "


def test_gemini_estimate_bound_is_a_third_of_the_packing_upper_bound():
    """Threshold decisions need a realistic count, not the packing upper bound.

    ``bound="upper"`` answers "can this exceed the provider limit?" and must
    over-count. ``bound="estimate"`` answers "is this conversation large
    enough to act on?" and must not treat every UTF-8 byte as a token.
    """
    counter = TokenCounter()
    text = _THAI_SENTENCE * 20
    byte_length = len(text.encode("utf-8"))

    upper = counter.count_text(provider="gemini", model="gemini-2.5-flash", text=text)
    estimate = counter.count_text(
        provider="gemini", model="gemini-2.5-flash", text=text, bound="estimate"
    )

    assert upper.tokens == byte_length
    assert estimate.tokens == math.ceil(byte_length / 3)
    assert estimate.strategy == "gemini:utf8_bytes_div_3"


def test_upper_bound_remains_the_default_for_gemini():
    """Every existing caller keeps the conservative bound it was written against."""
    counter = TokenCounter()
    text = _THAI_SENTENCE * 5

    default = counter.count_text(provider="gemini", model="gemini-2.5-flash", text=text)
    explicit = counter.count_text(
        provider="gemini", model="gemini-2.5-flash", text=text, bound="upper"
    )

    assert default.tokens == explicit.tokens == len(text.encode("utf-8"))
    assert default.strategy == explicit.strategy == "gemini:utf8_byte_upper_bound"


def test_estimate_bound_does_not_change_providers_with_an_exact_tokenizer():
    """OpenAI counts with tiktoken, which is already exact - nothing to relax."""
    counter = TokenCounter()
    text = "the quick brown fox " * 10

    upper = counter.count_text(provider="openai", model="gpt-4o-mini", text=text)
    estimate = counter.count_text(
        provider="openai", model="gpt-4o-mini", text=text, bound="estimate"
    )

    assert upper.tokens == estimate.tokens
    assert upper.strategy == estimate.strategy


def test_count_messages_threads_the_estimate_bound_through_to_the_text_strategy():
    counter = TokenCounter()
    text = _THAI_SENTENCE * 20
    messages = [{"role": "user", "content": text}]
    envelope = 2 + 4

    upper = counter.count_messages(provider="gemini", model="gemini-2.5-flash", messages=messages)
    estimate = counter.count_messages(
        provider="gemini", model="gemini-2.5-flash", messages=messages, bound="estimate"
    )

    assert upper.tokens == len(text.encode("utf-8")) + envelope
    assert estimate.tokens == math.ceil(len(text.encode("utf-8")) / 3) + envelope
    assert estimate.strategy == "gemini:utf8_bytes_div_3:messages"
