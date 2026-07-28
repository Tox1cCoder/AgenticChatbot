"""The terminal thinking summary must not duplicate the reasoning summary.

``accumulated_thinking`` is built from canonical ``reasoning_delta`` events. For
OpenAI those deltas *are* the reasoning summary, which the agent already stored
as ``reasoning_summary`` — copying them into ``thinking_summary`` as well made
the trace panel render the same text twice ("Reasoning Summary" and "Thinking
Summary").
"""

from __future__ import annotations

from app.ai.graph import apply_accumulated_thinking


class _Response:
    def __init__(self, metadata: dict):
        self.metadata = metadata


def test_sets_thinking_summary_when_absent():
    response = _Response({})
    apply_accumulated_thinking(response, "gemini thoughts")
    assert response.metadata["thinking_summary"] == "gemini thoughts"


def test_does_not_overwrite_an_existing_thinking_summary():
    response = _Response({"thinking_summary": "authoritative"})
    apply_accumulated_thinking(response, "streamed")
    assert response.metadata["thinking_summary"] == "authoritative"


def test_skips_when_reasoning_summary_already_holds_the_same_text():
    response = _Response({"reasoning_summary": "the summary"})
    apply_accumulated_thinking(response, "the summary")
    assert "thinking_summary" not in response.metadata


def test_ignores_surrounding_whitespace_when_comparing():
    response = _Response({"reasoning_summary": "the summary"})
    apply_accumulated_thinking(response, "  the summary\n")
    assert "thinking_summary" not in response.metadata


def test_keeps_thinking_summary_when_it_differs_from_reasoning_summary():
    response = _Response({"reasoning_summary": "short summary"})
    apply_accumulated_thinking(response, "a genuinely different thought trace")
    assert response.metadata["thinking_summary"] == "a genuinely different thought trace"


def test_no_op_for_empty_accumulated_thinking():
    response = _Response({})
    apply_accumulated_thinking(response, "")
    assert response.metadata == {}
