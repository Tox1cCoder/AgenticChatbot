from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.token_counter import ReportedTokenUsage, TokenCounter


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


def test_gemini_thai_estimate_is_conservative_not_four_character_heuristic():
    counter = TokenCounter()
    text = "การสรุปบทสนทนาต้องนับโทเค็นภาษาไทยอย่างระมัดระวัง"

    result = counter.count_text(provider="gemini", model="gemini-2.5-flash", text=text)

    assert result.tokens >= math.ceil(len(text.encode("utf-8")) / 3)
    old_ascii_heuristic = math.floor(len(text) / 4)
    assert result.tokens > old_ascii_heuristic
    assert result.strategy == "gemini:utf8_bytes_div_3"


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
