"""Pausing a validated partial, and resuming it with its evidence intact.

Two properties carry this, and the second is the one the plan originally
missed (R2):

* **The pause happens after validation, never before.** The answer a user is
  shown and offered Continue on must already have passed the same validation
  as a finished one -- otherwise Continue is an invitation to keep an unchecked
  draft.
* **Continue must rehydrate what the previous epoch gathered.** The next
  invocation is assembled from history plus the current-turn slice, and
  everything an epoch produced is sliced off into
  ``outcome.provenance.private_messages``. Resetting the counters and jumping
  back to the specialist therefore hands epoch 2 the original question and no
  evidence -- so it re-runs work the user already paid for, with a budget that
  has just been refilled.

The carried transcript has to stay *valid*, not merely present: a
``ToolMessage`` whose matching tool call was dropped is a provider error, not a
degraded prompt.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.ai.workflow.continuation import (
    ContinuationPausePayload,
    ContinuationResume,
    carry_messages,
    pairs_are_intact,
)


def _tool_turn(call_id: str = "c1", name: str = "web_search") -> list:
    """One complete round: the request, and the result that answers it."""
    return [
        AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {}}]),
        ToolMessage(content="evidence", tool_call_id=call_id, name=name),
    ]


# ----------------------------------------------------------------------
# what gets carried
# ----------------------------------------------------------------------


def test_the_evidence_from_the_previous_epoch_is_carried():
    produced = [*_tool_turn(), AIMessage(content="partial answer")]

    carried = carry_messages(produced)

    assert any(isinstance(message, ToolMessage) for message in carried)


def test_a_tool_call_keeps_the_result_that_answers_it():
    """Order and adjacency, not just membership."""
    produced = [*_tool_turn("c1"), *_tool_turn("c2")]

    carried = carry_messages(produced)

    assert pairs_are_intact(carried) is True


def test_an_orphaned_tool_result_is_dropped_rather_than_carried():
    """A result with no request is a provider error waiting to happen."""
    produced = [ToolMessage(content="evidence", tool_call_id="missing", name="web_search")]

    carried = carry_messages(produced)

    assert carried == []


def test_an_unanswered_tool_call_is_dropped_with_it():
    """Carrying the request without its answer breaks the same rule."""
    produced = [AIMessage(content="", tool_calls=[{"id": "c1", "name": "web_search", "args": {}}])]

    carried = carry_messages(produced)

    assert carried == []


def test_a_partial_round_is_dropped_but_a_complete_one_survives():
    produced = [
        *_tool_turn("c1"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "web_search", "args": {}}]),
    ]

    carried = carry_messages(produced)

    assert pairs_are_intact(carried) is True
    assert [getattr(m, "tool_call_id", None) for m in carried if isinstance(m, ToolMessage)] == [
        "c1"
    ]


def test_the_partial_answer_itself_is_not_carried():
    """It is already persisted and shown. Carrying it would repeat it."""
    produced = [*_tool_turn(), AIMessage(content="partial answer")]

    carried = carry_messages(produced)

    assert not any(
        getattr(message, "content", None) == "partial answer" for message in carried
    )


def test_a_human_message_is_never_carried():
    """The question is already in the current-turn slice."""
    produced = [HumanMessage(content="the question"), *_tool_turn()]

    carried = carry_messages(produced)

    assert not any(isinstance(message, HumanMessage) for message in carried)


def test_carrying_nothing_is_valid():
    assert carry_messages([]) == []
    assert pairs_are_intact([]) is True


def test_an_offloaded_result_is_carried_by_reference_not_rehydrated():
    """A blob reference stays a reference; the point of offloading was size."""
    produced = [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "web_search", "args": {}}]),
        ToolMessage(
            content='{"blob_id": "abc", "preview": "short"}',
            tool_call_id="c1",
            name="web_search",
        ),
    ]

    carried = carry_messages(produced)

    assert '"blob_id": "abc"' in str(carried[-1].content)


# ----------------------------------------------------------------------
# the pause payload
# ----------------------------------------------------------------------


def test_the_pause_payload_names_the_epoch_it_is_pausing():
    payload = ContinuationPausePayload(
        generation_id="11111111-1111-1111-1111-111111111111",
        logical_turn_id="turn-1",
        execution_epoch=0,
        active_agent_id="chat_agent",
        validated_content="partial answer",
        budget={"exhausted_by": "tool_calls"},
    )

    assert payload.type == "execution_budget_exhausted"
    assert payload.execution_epoch == 0


def test_the_pause_payload_is_distinguishable_from_a_tool_approval():
    """The graph must not treat this as a HITL interrupt.

    Both arrive as interrupts. Confusing them would ask a human to approve a
    tool call that does not exist, or resume a budget pause as an approval.
    """
    payload = ContinuationPausePayload(
        generation_id="11111111-1111-1111-1111-111111111111",
        logical_turn_id="turn-1",
        execution_epoch=1,
        active_agent_id="rag_agent",
        validated_content="partial",
        budget={},
    )

    assert payload.model_dump(mode="json")["type"] == "execution_budget_exhausted"


def test_a_resume_says_which_way_the_user_decided():
    assert ContinuationResume(
        action="continue",
        continuation_id="22222222-2222-2222-2222-222222222222",
        expected_epoch=0,
    ).action == "continue"
    assert ContinuationResume(
        action="stop",
        continuation_id="22222222-2222-2222-2222-222222222222",
        expected_epoch=0,
    ).action == "stop"


# ----------------------------------------------------------------------
# delivery into the next epoch
# ----------------------------------------------------------------------


def _request(**overrides):
    from app.ai.workflow.specialists import SpecialistRequest

    payload = {
        "agent_id": "chat_agent",
        "conversation_id": "conversation-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "persona": None,
        "model_request": None,
        "messages": [HumanMessage(content="the question")],
        "history": [AIMessage(content="an older turn")],
    }
    payload.update(overrides)
    return SpecialistRequest(**payload)


def test_the_next_epoch_is_sent_the_carried_evidence():
    """R2, asserted on what the model receives rather than on the checkpoint.

    A runtime probe once passed a checkpoint assertion while sending the model
    only the original human question, so this reads the invocation itself.
    """
    from app.ai.workflow.specialists import SpecialistFactory

    carried = carry_messages([*_tool_turn("c1"), AIMessage(content="partial")])

    messages = SpecialistFactory._invocation_messages(_request(carried_messages=carried))

    assert any(isinstance(message, ToolMessage) for message in messages)


def test_the_carried_evidence_sits_between_history_and_the_question():
    """History, then what this turn already learned, then what was asked."""
    from app.ai.workflow.specialists import SpecialistFactory

    carried = carry_messages(_tool_turn("c1"))

    messages = SpecialistFactory._invocation_messages(_request(carried_messages=carried))
    kinds = [type(message).__name__ for message in messages]

    assert kinds.index("ToolMessage") < kinds.index("HumanMessage")
    assert kinds[0] == "AIMessage"


def test_the_invocation_transcript_stays_provider_valid():
    from app.ai.workflow.specialists import SpecialistFactory

    carried = carry_messages([*_tool_turn("c1"), *_tool_turn("c2")])

    messages = SpecialistFactory._invocation_messages(_request(carried_messages=carried))

    assert pairs_are_intact(messages) is True


def test_a_first_epoch_carries_nothing_and_is_unchanged():
    from app.ai.workflow.specialists import SpecialistFactory

    request = _request()

    messages = SpecialistFactory._invocation_messages(request)

    assert messages == [*request.history, *request.messages]
