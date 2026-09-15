"""Planning could not read its own answer when the model was thinking.

Reported as ``empty_public_content`` with ``agent_id: planning_agent``. The
cause is not a transient provider outcome: ``_last_ai_text`` read the final
message with

    text = content if isinstance(content, str) else ""

so *any* list-shaped content produced ``""``. Gemini returns list-shaped
content whenever it emits thought parts, and Planning has no per-agent thinking
level, so it inherits the global one. The answer was sitting in a ``text``
block the extractor threw away, and the turn failed as if the model had said
nothing.

This is the third hand-rolled copy of "get the text out of a message" in this
codebase. ``app.ai.utils.coerce_response_text`` and ``BaseMessage.text`` both
handle it correctly; ``specialists._final_text`` was the second and now defers
to ``BaseMessage.text``. This one is the last.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.ai.workflow.planning_execution import _last_ai_text

REASONING_THEN_ANSWER = [
    {"type": "reasoning", "reasoning": "working out the plan"},
    {"type": "text", "text": "Here is the plan: first X, then Y."},
]


def test_a_thinking_models_answer_is_read():
    state = {
        "messages": [
            HumanMessage(content="make a plan"),
            AIMessage(content=REASONING_THEN_ANSWER),
        ]
    }

    assert _last_ai_text(state) == "Here is the plan: first X, then Y."


def test_plain_string_content_still_works():
    state = {"messages": [AIMessage(content="Here is the plan.")]}

    assert _last_ai_text(state) == "Here is the plan."


def test_a_thought_part_is_never_published_as_the_plan():
    """A thought block can carry its text under ``text``; that is not an answer."""
    state = {"messages": [AIMessage(content=[{"type": "thinking", "text": "internal"}])]}

    assert _last_ai_text(state) == ""


def test_a_tool_calling_turn_is_skipped_for_the_earlier_answer():
    """Unchanged behaviour: an intermediate tool turn carries no plan."""
    state = {
        "messages": [
            AIMessage(content="The plan is X."),
            AIMessage(content=[{"type": "text", "text": "calling a tool"}],
                      tool_calls=[{"id": "c1", "name": "write_todos", "args": {}}]),
        ]
    }

    assert _last_ai_text(state) == "The plan is X."


def test_no_assistant_text_anywhere_is_still_empty():
    state = {"messages": [HumanMessage(content="make a plan")]}

    assert _last_ai_text(state) == ""


def test_the_planning_outcome_carries_the_extracted_text():
    """End of the path the failure actually took."""
    from app.ai.workflow.planning_execution import build_planning_outcome

    state = {"messages": [AIMessage(content=REASONING_THEN_ANSWER)]}
    outcome = build_planning_outcome(content=_last_ai_text(state), results=[])

    assert outcome.response.message.content == "Here is the plan: first X, then Y."
