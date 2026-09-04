"""The adapter publishes the finalizer's response or nothing at all.

Validation is only a guarantee if there is no way around it. Recovery used to
scan accumulated stream chunks and checkpoint messages for any assistant-looking
text and publish that — which is precisely a path around the finalizer, and the
text it recovered had passed no output policy.

A turn that produced no finalized response is a failed turn. The caller gets a
typed error, not a salvaged draft.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.ai.graph import MultiAgentWorkflow
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


def _workflow() -> MultiAgentWorkflow:
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    workflow._attach_context_outputs = lambda state, response: response
    workflow._get_agent_type = lambda name: AgentType.CHAT
    return workflow


def _finalized(content: str = "validated answer") -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content=content),
        metadata={"validation": {"passed": True, "policy_ids": ["public_content"]}},
    )


# ----------------------------------------------------------------------
# what still works
# ----------------------------------------------------------------------


def test_the_finalizers_response_is_returned_unchanged():
    response = _finalized()
    state = {"response": response, "active_agent_id": "chat_agent", "messages": []}

    recovered = _workflow()._finalized_response(state)

    assert recovered is response
    assert recovered.metadata["validation"]["passed"] is True


def test_a_response_still_carrying_tool_calls_is_not_publishable():
    """A tool-calling turn is mid-flight, not an answer."""
    response = _finalized(content="")
    response.message.tool_calls = [{"id": "c1", "name": "lookup", "args": {}}]
    state = {"response": response, "messages": []}

    assert _workflow()._finalized_response(state) is None


# ----------------------------------------------------------------------
# what must no longer happen
# ----------------------------------------------------------------------


def test_assistant_text_in_messages_is_never_promoted_to_an_answer():
    """Scanning checkpoint messages was the way around the finalizer."""
    state = {
        "response": None,
        "active_agent_id": "chat_agent",
        "messages": [
            HumanMessage(content="what is it?"),
            AIMessage(content="a draft the finalizer never validated", id="private-1"),
        ],
    }

    assert _workflow()._finalized_response(state) is None


def test_recovery_takes_no_text_but_the_finalized_response():
    """There is no second argument to pass streamed text in through."""
    import inspect

    from app.ai.graph import MultiAgentWorkflow

    parameters = inspect.signature(MultiAgentWorkflow._finalized_response).parameters
    assert list(parameters) == ["self", "state"]


def test_accumulated_text_does_not_backfill_an_empty_finalized_response():
    """Deltas must reproduce validated content, so nothing else may fill it.

    An empty finalized response means the finalizer published nothing;
    substituting the raw stream would publish text no policy approved.
    """
    state = {"response": _finalized(content=""), "messages": []}

    assert _workflow()._finalized_response(state) is None


def test_a_turn_with_no_response_recovers_nothing():
    assert _workflow()._finalized_response({"messages": []}) is None
    assert _workflow()._finalized_response(None) is None


def test_recovery_does_not_fabricate_an_agent_identity():
    """A fabricated response also fabricated whose answer it was."""
    state = {
        "response": None,
        "messages": [AIMessage(content="orphan text", id="private-1")],
    }

    assert _workflow()._finalized_response(state) is None
