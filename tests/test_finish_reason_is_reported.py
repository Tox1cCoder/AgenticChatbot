"""An empty model response must say why, in the provider's own words.

`finish_reason` was read nowhere in the production path. Gemini returns an
empty candidate for several unrelated reasons -- MAX_TOKENS spent on thinking,
SAFETY, RECITATION, MALFORMED_FUNCTION_CALL -- and every one arrived downstream
as an ordinary response with no text, failing `empty_public_content` two nodes
later with nothing to distinguish it from any other.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.ai.agents.base_agent import _finish_reason


def test_reads_gemini_style_response_metadata():
    message = AIMessage(content="", response_metadata={"finish_reason": "MAX_TOKENS"})

    assert _finish_reason(message) == "MAX_TOKENS"


def test_reads_additional_kwargs_when_metadata_is_silent():
    message = AIMessage(content="", additional_kwargs={"finish_reason": "SAFETY"})

    assert _finish_reason(message) == "SAFETY"


def test_reads_the_anthropic_spelling():
    message = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})

    assert _finish_reason(message) == "max_tokens"


def test_reads_the_camel_case_spelling():
    message = AIMessage(content="", response_metadata={"finishReason": "RECITATION"})

    assert _finish_reason(message) == "RECITATION"


def test_a_response_that_reports_nothing_is_not_an_error():
    assert _finish_reason(AIMessage(content="hello")) is None
    assert _finish_reason(object()) is None


def test_an_empty_string_is_not_a_reason():
    message = AIMessage(content="", response_metadata={"finish_reason": ""})

    assert _finish_reason(message) is None
