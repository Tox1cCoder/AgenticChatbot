"""Tests for per-message context-window metadata persisted by BaseAgent.

Covers:
- ``_apply_runtime_metadata`` copies ``runtime_config.context_window`` into
  the message metadata, with deep-copy semantics so later mutations do not
  leak back into the runtime config.
- ``_merge_context_window_usage`` merges token-usage fields (computed from
  ``TokenBudgetBreakdown.to_dict()``) into ``metadata['context_window']``,
  preserving the static fields and adding ``used_tokens``,
  ``used_token_source``, ``usage_ratio``, ``display_state``.
- For unknown / custom models (``known=False``) the merge MUST NOT produce a
  misleading ``usage_ratio``.
- ``_create_fallback_runtime_config`` populates ``context_window`` from the
  static registry so fallback responses still expose context metadata for
  the actual provider/model that answered.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import ToolMessage

from app.ai.token_instrumentation import (
    TokenBudgetBreakdown,
    estimate_message_tokens,
    extract_actual_usage,
)
from app.core.runtime_modeling import (
    ResolvedRuntimeModelConfig,
    RuntimeFallbackConfig,
)


class _FakeAgent:
    """Concrete BaseAgent stand-in that bypasses tool/gemini init."""

    def __init__(self, agent_config_key: str = "chat", model_name: str = "gpt-4o"):
        self.agent_config_key = agent_config_key
        self.model_name = model_name
        self.gemini_client = None
        self.langchain_model = object()


def _runtime_config(
    *,
    provider: str = "openai",
    model: str = "gpt-4o",
    context_window: dict[str, Any] | None = None,
) -> ResolvedRuntimeModelConfig:
    return ResolvedRuntimeModelConfig(
        agent_key="chat",
        provider=provider,
        model=model,
        temperature=0.5,
        api_key="fake-key",
        key_source="env",
        source="request",
        warnings=[],
        context_window=context_window,
    )


def _known_context_window(
    *,
    provider: str = "openai",
    model: str = "gpt-4o",
    context_window_tokens: int = 128000,
    max_output_tokens: int = 16384,
    source: str = "registry",
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "context_window_tokens": context_window_tokens,
        "max_input_tokens": context_window_tokens,
        "max_output_tokens": max_output_tokens,
        "source": source,
        "known": True,
    }


def _unknown_context_window(
    *, provider: str = "openai", model: str = "weird-custom-id"
) -> dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "context_window_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "source": "unknown",
        "known": False,
    }


# ---------------------------------------------------------------------------
# _apply_runtime_metadata: context_window plumbing
# ---------------------------------------------------------------------------


def test_apply_runtime_metadata_includes_context_window_when_present():
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    metadata: dict[str, Any] = {}
    expected = _known_context_window()
    runtime = _runtime_config(context_window=expected)

    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    assert metadata.get("context_window") == expected


def test_apply_runtime_metadata_omits_context_window_when_none():
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    metadata: dict[str, Any] = {}
    runtime = _runtime_config(context_window=None)

    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    assert "context_window" not in metadata


def test_apply_runtime_metadata_copies_context_window():
    """Mutating ``runtime_config.context_window`` after the call must not
    affect ``metadata['context_window']`` — the helper performs a copy."""
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    metadata: dict[str, Any] = {}
    original = _known_context_window()
    runtime = _runtime_config(context_window=original)

    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    runtime.context_window["context_window_tokens"] = 999  # type: ignore[index]
    runtime.context_window["mutated"] = True  # type: ignore[index]

    assert metadata["context_window"]["context_window_tokens"] == 128000
    assert "mutated" not in metadata["context_window"]


# ---------------------------------------------------------------------------
# _merge_context_window_usage: merges token usage onto static window
# ---------------------------------------------------------------------------


def test_context_window_usage_merged_in_invoke():
    """With a known context window and actual_total_tokens set, the merge
    helper must produce all four usage fields and preserve the static fields.
    """
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    runtime = _runtime_config(context_window=_known_context_window())
    metadata: dict[str, Any] = {}
    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    breakdown = TokenBudgetBreakdown(
        system_prompt_tokens=100,
        history_tokens=400,
        current_turn_tokens=500,
        total_tokens=1000,
        actual_input_tokens=12000,
        actual_output_tokens=200,
        actual_total_tokens=12500,
        actual_reasoning_tokens=300,
    )
    base_module.BaseAgent._merge_context_window_usage(
        agent, metadata, breakdown.to_dict()
    )

    cw = metadata["context_window"]
    # Static fields preserved.
    assert cw["provider"] == "openai"
    assert cw["model"] == "gpt-4o"
    assert cw["context_window_tokens"] == 128000
    assert cw["max_input_tokens"] == 128000
    assert cw["max_output_tokens"] == 16384
    assert cw["source"] == "registry"
    assert cw["known"] is True
    # Usage fields populated.
    assert cw["used_tokens"] == 12500
    assert cw["used_token_source"] == "actual_total"
    assert cw["usage_ratio"] == 12500 / 128000
    assert cw["display_state"] == "ok"


def test_context_window_usage_uses_estimated_when_actual_missing():
    """Without ``actual_input_tokens``, the merge helper should fall back to
    the estimated total."""
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    runtime = _runtime_config(context_window=_known_context_window())
    metadata: dict[str, Any] = {}
    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    breakdown = TokenBudgetBreakdown(
        system_prompt_tokens=100,
        history_tokens=400,
        current_turn_tokens=500,
        total_tokens=1000,
    )
    base_module.BaseAgent._merge_context_window_usage(
        agent, metadata, breakdown.to_dict()
    )

    cw = metadata["context_window"]
    assert cw["used_tokens"] == 1000
    assert cw["used_token_source"] == "estimated_total"
    assert cw["display_state"] == "ok"


def test_context_window_usage_unknown_model_no_misleading_ratio():
    """For an unknown / custom model (``known=False``), the merge MUST NOT
    invent a ``usage_ratio``."""
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    runtime = _runtime_config(
        model="weird-custom-id",
        context_window=_unknown_context_window(model="weird-custom-id"),
    )
    metadata: dict[str, Any] = {}
    base_module.BaseAgent._apply_runtime_metadata(agent, metadata, runtime)

    breakdown = TokenBudgetBreakdown(
        total_tokens=1500,
        actual_input_tokens=1500,
    )
    base_module.BaseAgent._merge_context_window_usage(
        agent, metadata, breakdown.to_dict()
    )

    cw = metadata["context_window"]
    assert cw["known"] is False
    assert cw["context_window_tokens"] is None
    assert cw["used_tokens"] is None
    assert cw["used_token_source"] == "unknown"
    assert cw["usage_ratio"] is None
    assert cw["display_state"] == "unknown"


def test_merge_context_window_usage_noop_when_no_context_window():
    """If the metadata has no ``context_window`` (e.g. runtime_config was
    missing it), the merge helper must do nothing — no KeyError, no synth."""
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    metadata: dict[str, Any] = {"provider": "openai"}
    breakdown = TokenBudgetBreakdown(total_tokens=500, actual_input_tokens=500)

    base_module.BaseAgent._merge_context_window_usage(
        agent, metadata, breakdown.to_dict()
    )

    assert "context_window" not in metadata


# ---------------------------------------------------------------------------
# _create_fallback_runtime_config: context_window inherited from registry
# ---------------------------------------------------------------------------


def test_fallback_runtime_config_carries_context_window():
    """Fallback bypasses ModelConfigService; verify it still surfaces a
    registry-resolved context_window for the actual provider/model that will
    answer."""
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    fallback = RuntimeFallbackConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        temperature=0.5,
        api_key="gemini-key",
        key_source="env",
    )

    resolved = base_module.BaseAgent._create_fallback_runtime_config(
        agent,
        fallback,
        reason="provider_unconfigured",
        from_provider="openai",
    )

    assert resolved is not None
    assert resolved.context_window is not None
    assert resolved.context_window["provider"] == "gemini"
    assert resolved.context_window["model"] == "gemini-2.5-flash"
    assert resolved.context_window["known"] is True
    assert resolved.context_window["source"] == "registry"
    assert resolved.context_window["context_window_tokens"] == 1_048_576


def test_fallback_runtime_config_unknown_model_context_window():
    """Fallback to an unknown model id must still produce a context_window
    payload with ``known=False`` (no misleading values)."""
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    fallback = RuntimeFallbackConfig(
        provider="gemini",
        model="completely-made-up-model-x9",
        temperature=0.5,
        api_key="gemini-key",
        key_source="env",
    )

    resolved = base_module.BaseAgent._create_fallback_runtime_config(
        agent,
        fallback,
        reason="provider_unconfigured",
        from_provider="openai",
    )

    assert resolved is not None
    assert resolved.context_window is not None
    assert resolved.context_window["known"] is False
    assert resolved.context_window["source"] == "unknown"
    assert resolved.context_window["context_window_tokens"] is None
    assert resolved.context_window["max_input_tokens"] is None
    assert resolved.context_window["max_output_tokens"] is None


def test_fallback_runtime_config_returns_none_when_fallback_missing():
    """No fallback config supplied -> helper returns None (existing contract)."""
    from app.ai.agents import base_agent as base_module

    agent = _FakeAgent()
    resolved = base_module.BaseAgent._create_fallback_runtime_config(
        agent,
        None,
        reason="provider_unconfigured",
        from_provider="openai",
    )
    assert resolved is None


# ---------------------------------------------------------------------------
# extract_actual_usage: handles both dict and object usage_metadata shapes
# ---------------------------------------------------------------------------


def test_extract_actual_usage_handles_dict_usage_metadata():
    """LangChain >= 0.2 exposes usage_metadata as a UsageMetadata TypedDict
    (a plain dict at runtime). The extractor must read it via subscript."""
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 1234,
            "output_tokens": 56,
            "total_tokens": 1290,
            "output_token_details": {"reasoning": 12},
        }
    )

    result = extract_actual_usage(response)

    assert result == {
        "input_tokens": 1234,
        "output_tokens": 56,
        "total_tokens": 1290,
        "reasoning_tokens": 12,
    }


def test_extract_actual_usage_handles_object_usage_metadata():
    """Older providers may expose usage_metadata as an object with attributes."""
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(input_tokens=987, output_tokens=10)
    )

    result = extract_actual_usage(response)

    assert result == {
        "input_tokens": 987,
        "output_tokens": 10,
        "total_tokens": 997,
        "reasoning_tokens": None,
    }


def test_extract_actual_usage_falls_back_to_response_metadata_usage():
    """If usage_metadata is absent, fall back to response_metadata['usage']
    using OpenAI's prompt_tokens/completion_tokens keys."""
    response = SimpleNamespace(
        response_metadata={
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 20,
                "total_tokens": 550,
                "completion_tokens_details": {"reasoning_tokens": 9},
            }
        }
    )

    result = extract_actual_usage(response)

    assert result == {
        "input_tokens": 500,
        "output_tokens": 20,
        "total_tokens": 550,
        "reasoning_tokens": 9,
    }


def test_extract_actual_usage_handles_token_usage_envelope():
    """Some providers nest the envelope under ``token_usage`` instead."""
    response = SimpleNamespace(
        response_metadata={
            "token_usage": {"prompt_tokens": 600, "completion_tokens": 30}
        }
    )

    result = extract_actual_usage(response)

    assert result == {
        "input_tokens": 600,
        "output_tokens": 30,
        "total_tokens": 630,
        "reasoning_tokens": None,
    }


def test_extract_actual_usage_prefers_usage_metadata_over_response_metadata():
    """When both shapes are present, the standardized usage_metadata wins."""
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 111, "output_tokens": 22},
        response_metadata={"usage": {"prompt_tokens": 999, "completion_tokens": 88}},
    )

    result = extract_actual_usage(response)

    assert result == {
        "input_tokens": 111,
        "output_tokens": 22,
        "total_tokens": 133,
        "reasoning_tokens": None,
    }


def test_extract_actual_usage_returns_nones_for_unknown_shape():
    """A response with neither shape yields Nones — no exception."""
    response = SimpleNamespace()

    result = extract_actual_usage(response)

    assert result == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
    }


def test_extract_actual_usage_returns_nones_for_none_response():
    """Defensive: a None response yields Nones — no AttributeError."""
    result = extract_actual_usage(None)

    assert result == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
    }


def test_extract_actual_usage_handles_dict_with_none_values():
    """If the dict explicitly carries None values, result stays None
    rather than overwriting with None (and definitely does not raise)."""
    response = SimpleNamespace(
        usage_metadata={"input_tokens": None, "output_tokens": None}
    )

    result = extract_actual_usage(response)

    assert result == {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
    }


def test_estimate_message_tokens_counts_tool_result_identity():
    """Fallback estimates should include the tool-result envelope, not only
    the visible result text, because providers receive name/id metadata too."""
    with_identity = ToolMessage(
        content="result body",
        tool_call_id="call-abc-123",
        name="search_documents",
    )
    without_identity = ToolMessage(
        content="result body",
        tool_call_id="",
        name="",
    )

    assert estimate_message_tokens(with_identity) > estimate_message_tokens(without_identity)
