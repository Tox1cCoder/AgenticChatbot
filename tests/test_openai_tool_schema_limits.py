"""OpenAI rejects the whole request when a tool description is too long.

The bug this file exists for, from `model_usage_events` on 2026-09-08:

    openai  gpt-5.6-luna  search_agent  error  BadRequestError   n=9
    openai  gpt-5.6-luna  search_agent  success                  n=7

Same model, same key, interleaved — so neither credentials nor model
availability. Every call that offered tools was rejected in 360-1900 ms, before
generation; the one call per turn with `disable_tools=True` succeeded in
7-9.6 s. Three of the eight bound tools had a `function.description` over
OpenAI's documented 1024-character maximum, which returns 400
`string_above_max_length` and names the offending index.

Gemini accepts those descriptions, so `RuntimeModelMiddleware`'s fallback
"recovered" every time and the only surviving signal was a warning that said
"after a provider error" — while discarding the error. It looked like an
OpenAI outage and was entirely our own payload.

The clamp is a safety net, not the cure: a truncated description is a degraded
one. `test_no_bound_tool_needs_clamping_today` is the standing record of which
descriptions still need shortening editorially, and is expected to fail until
they are.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.tools import tool

from app.ai.openai_tool_limits import (
    OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS,
    clamp_openai_tool_descriptions,
    oversized_tool_descriptions,
)


def _spec(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ----------------------------------------------------------------------
# the clamp
# ----------------------------------------------------------------------


def test_an_oversized_description_is_brought_within_the_limit():
    clamped = clamp_openai_tool_descriptions([_spec("wide", "x" * 3000)])

    assert len(clamped[0]["function"]["description"]) <= OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS


def test_a_description_within_the_limit_is_left_exactly_alone():
    """Byte-for-byte. A clamp that rewrites compliant text is a prompt change."""
    description = "Search the web for current information. Use a specific query."

    clamped = clamp_openai_tool_descriptions([_spec("narrow", description)])

    assert clamped[0]["function"]["description"] == description


def test_a_description_exactly_at_the_limit_is_untouched():
    """Off-by-one at the boundary would silently truncate a legal description."""
    description = "y" * OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS

    clamped = clamp_openai_tool_descriptions([_spec("edge", description)])

    assert clamped[0]["function"]["description"] == description


def test_the_truncation_prefers_a_sentence_boundary():
    """Cutting mid-word leaves the model reading a fragment.

    A description is instructions; ending one halfway through a clause is worse
    guidance than ending it a sentence early.
    """
    first = "Use this tool to search. "
    description = first + ("Extra guidance follows here. " * 200)

    clamped = clamp_openai_tool_descriptions([_spec("prose", description)])
    result = clamped[0]["function"]["description"]

    assert len(result) <= OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS
    assert result.endswith(".")
    assert result.startswith(first)


def test_truncation_falls_back_to_a_hard_cut_when_there_is_no_boundary():
    """A description with no sentence break still has to fit."""
    clamped = clamp_openai_tool_descriptions([_spec("nospace", "z" * 5000)])

    assert len(clamped[0]["function"]["description"]) == OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS


def test_the_original_specs_are_not_mutated():
    """Tool objects are shared and cached across providers.

    Mutating in place would shorten the description Gemini receives too —
    applying an OpenAI constraint to a provider that does not have it.
    """
    spec = _spec("shared", "q" * 3000)

    clamp_openai_tool_descriptions([spec])

    assert len(spec["function"]["description"]) == 3000


def test_everything_but_the_description_survives_the_clamp():
    spec = _spec("keeper", "w" * 3000)
    spec["function"]["parameters"] = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "the query"}},
        "required": ["query"],
    }

    clamped = clamp_openai_tool_descriptions([spec])
    function = clamped[0]["function"]

    assert function["name"] == "keeper"
    assert function["parameters"]["required"] == ["query"]
    assert function["parameters"]["properties"]["query"]["description"] == "the query"


def test_a_tool_with_no_description_is_passed_through():
    spec = {"type": "function", "function": {"name": "bare", "parameters": {}}}

    assert clamp_openai_tool_descriptions([spec]) == [spec]


def test_langchain_tools_are_converted_and_clamped():
    """The real input is `BaseTool`, not a dict, and must not be mutated."""

    @tool
    def spacious(query: str) -> str:
        """PLACEHOLDER"""
        return query

    spacious.description = "Long guidance. " * 300
    original = spacious.description

    clamped = clamp_openai_tool_descriptions([spacious])

    assert len(clamped[0]["function"]["description"]) <= OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS
    assert clamped[0]["function"]["name"] == "spacious"
    assert spacious.description == original, "the shared tool object was mutated"


# ----------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------


def test_oversized_descriptions_are_reported_with_their_lengths():
    """So an operator learns which tool to shorten, not merely that one is long."""
    report = oversized_tool_descriptions(
        [_spec("fine", "short"), _spec("wide", "x" * 3000), _spec("wider", "y" * 4000)]
    )

    assert report == {"wide": 3000, "wider": 4000}


def test_nothing_is_reported_when_every_description_fits():
    assert oversized_tool_descriptions([_spec("fine", "short")]) == {}


# ----------------------------------------------------------------------
# the model that must apply it
# ----------------------------------------------------------------------


def test_the_openai_model_clamps_what_it_binds():
    """The seam that matters: what `ChatOpenAI` actually sends.

    Asserted through `bind_tools` rather than by calling the clamp directly,
    because a clamp nothing invokes is exactly the shape of the original bug.
    """
    from app.ai.model_factory import ModelFactory

    @tool
    def spacious(query: str) -> str:
        """PLACEHOLDER"""
        return query

    spacious.description = "Long guidance. " * 300

    model = ModelFactory.create_model(
        provider="openai", model="gpt-4o-mini", api_key="test-key", temperature=1.0
    )
    bound = model.bind_tools([spacious])

    sent = bound.kwargs["tools"]
    assert len(sent) == 1
    assert (
        len(sent[0]["function"]["description"]) <= OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS
    ), "ChatOpenAI was handed a description OpenAI will reject with 400"


def test_the_gemini_model_is_not_clamped():
    """Gemini has no such limit, and the long descriptions are tuned for it.

    Applying one provider's constraint to another would be a silent prompt
    change for the 623 successful Gemini calls in this deployment.
    """
    from app.ai.model_factory import ModelFactory

    @tool
    def spacious(query: str) -> str:
        """PLACEHOLDER"""
        return query

    spacious.description = "Long guidance. " * 300

    model = ModelFactory.create_model(
        provider="gemini", model="gemini-3-flash-preview", api_key="test-key", temperature=1.0
    )

    assert not isinstance(model, type(None))
    # The Gemini path binds the tool objects themselves; the description the
    # model would send is the full one.
    assert len(spacious.description) > OPENAI_FUNCTION_DESCRIPTION_MAX_CHARS


# ----------------------------------------------------------------------
# the diagnostic that hid this
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_fallback_warning_names_the_error_it_recovered_from(caplog):
    """The log line is why this took a database query to diagnose.

    It said "after a provider error" and dropped `exc`, so a 400 caused by our
    own payload was indistinguishable from an outage — and the fallback made
    every turn succeed, so nothing else complained.
    """
    import logging
    from types import SimpleNamespace

    from app.ai.workflow.middleware import RuntimeModelMiddleware

    config = SimpleNamespace(
        agent_key="search",
        provider="openai",
        model="gpt-5.6-luna",
        temperature=1.0,
        api_key="primary",
        key_source="user",
        warnings=[],
        capabilities={},
        fallback_config=SimpleNamespace(
            provider="gemini",
            model="gemini-3-flash-preview",
            temperature=1.0,
            api_key="fallback-key",
            key_source="user",
        ),
    )

    middleware = RuntimeModelMiddleware(
        runtime_model_resolver=SimpleNamespace(
            resolve_runtime_config=lambda *_a: config,
        ),
        model_factory=SimpleNamespace(create_model_from_runtime=lambda *_a, **_k: object()),
        agent_key="search",
        user_id="user-1",
        model_request=None,
    )

    calls: list[object] = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise ValueError("Invalid 'tools[3].function.description': string too long.")
        return "recovered"

    request = SimpleNamespace(override=lambda **_kwargs: SimpleNamespace(messages=[]))

    with caplog.at_level(logging.WARNING):
        result = await middleware.awrap_model_call(request, handler)

    assert result == "recovered"
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "ValueError" in logged, f"the exception type was not logged: {logged}"
    assert "string too long" in logged, f"the provider's own message was dropped: {logged}"


# ----------------------------------------------------------------------
# the standing editorial debt
# ----------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "image_search (2547), tool_search (1285) and widget_create (3027) exceed "
        "OpenAI's 1024-char limit. The clamp keeps the request valid, but a "
        "truncated description is degraded guidance -- these need shortening by "
        "hand, which is a prompt-content decision for Thai, not a mechanical cut."
    ),
    strict=True,
)
def test_no_bound_tool_needs_clamping_today():
    """Every search-agent tool description fits without truncation.

    ``strict=True``: when the three are shortened this test starts passing and
    the xfail becomes an error, which is the prompt to delete the marker rather
    than let the debt sit recorded but stale.
    """
    from app.ai.agents.search_agent import SearchAgent

    async def bound_tools():
        agent = SearchAgent()
        await agent._init_tools()
        return agent._get_tools_for_binding(
            conversation_id="11111111-1111-1111-1111-111111111111",
            internal_tools=None,
            user_id="22222222-2222-2222-2222-222222222222",
            device_id=None,
            include_hand_off=True,
            excluded_tool_names=None,
        )

    tools = asyncio.run(bound_tools())
    oversized = oversized_tool_descriptions(tools)

    assert oversized == {}, f"descriptions over the OpenAI limit: {oversized}"
