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


def test_the_pause_payload_reports_an_undecidable_mutation():
    """The turn has to know, not just the model.

    A ``MutationOutcomeUnknown`` is answered to the model in band as a
    ``ToolMessage``, but the continuation decision is made outside that loop
    and cannot read prose. Without this the offer would be minted for a turn
    whose side effect may or may not have happened.
    """
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
    from app.ai.workflow.continuation import make_continuation_pause_node
    from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome

    outcome = ResponseOutcome(
        agent_id="chat_agent",
        response=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="partial"),
            metadata={"mutation_outcome_unknown": True},
        ),
        provenance=OutcomeProvenance(output_policy_ids=("public_content",)),
    )

    seen: list[dict] = []
    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: seen.append(payload) or {"action": "stop"}
    )

    import asyncio

    asyncio.run(
        node(
            {
                "agent_outcome": outcome,
                "active_agent_id": "chat_agent",
                "execution_epoch": 0,
                "execution_budget": {"exhausted_by": "tool_calls"},
            }
        )
    )

    assert seen[0]["mutation_outcome_unknown"] is True


def test_an_ordinary_pause_reports_no_undecidable_mutation():
    """The flag must not default to blocking every continuation."""
    import asyncio

    from app.ai.workflow.continuation import make_continuation_pause_node

    seen: list[dict] = []
    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: seen.append(payload) or {"action": "stop"}
    )

    asyncio.run(node(_paused_state()))

    assert seen[0]["mutation_outcome_unknown"] is False


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


# ----------------------------------------------------------------------
# the pause node
# ----------------------------------------------------------------------


def _paused_state(**overrides) -> dict:
    from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
    from app.ai.workflow.contracts import OutcomeProvenance, ResponseOutcome

    produced = (*_tool_turn("c1"), AIMessage(content="partial answer"))
    outcome = ResponseOutcome(
        agent_id="chat_agent",
        response=AgentResponse(
            agent_type=AgentType.CHAT,
            agent_id="chat_agent",
            message=AgentMessage(role=MessageRole.ASSISTANT, content="partial answer"),
            metadata={"execution_budget": {"exhausted_by": "tool_calls"}},
        ),
        provenance=OutcomeProvenance(
            output_policy_ids=("public_content",), private_messages=produced
        ),
    )
    state = {
        "agent_outcome": outcome,
        "active_agent_id": "chat_agent",
        "execution_epoch": 0,
        "execution_budget": {"exhausted_by": "tool_calls"},
        "turn_identity": None,
        "conversation_id": "conversation-1",
        "user_id": "user-1",
    }
    state.update(overrides)
    return state


async def test_continuing_advances_the_epoch_and_returns_to_the_same_agent():
    """No routing on resume: the turn already chose its agent."""
    from app.ai.workflow.continuation import make_continuation_pause_node

    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: {"action": "continue", "expected_epoch": 0}
    )

    command = await node(_paused_state())

    assert command.goto == "chat_agent"
    assert command.update["execution_epoch"] == 1
    assert command.update["execution_phase"] == "executing"


async def test_continuing_clears_the_spent_budget():
    from app.ai.workflow.continuation import make_continuation_pause_node

    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: {"action": "continue", "expected_epoch": 0}
    )

    command = await node(_paused_state())

    assert command.update["execution_budget"] is None


async def test_continuing_carries_the_evidence_forward():
    """R2 at the seam that actually writes it."""
    from app.ai.workflow.continuation import make_continuation_pause_node

    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: {"action": "continue", "expected_epoch": 0}
    )

    command = await node(_paused_state())
    carried = command.update["carried_messages"]

    assert pairs_are_intact(carried) is True
    assert any(isinstance(message, ToolMessage) for message in carried)


async def test_stopping_finalizes_without_re_appending_the_answer():
    """The partial was persisted at the pause; finalize must not repeat it."""
    from app.ai.workflow.continuation import make_continuation_pause_node

    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: {"action": "stop", "expected_epoch": 0}
    )

    command = await node(_paused_state())

    assert command.goto == "finalize"
    assert command.update["execution_phase"] == "finalizing"
    assert "messages" not in command.update


async def test_the_payload_offered_to_the_client_describes_this_pause():
    from app.ai.workflow.continuation import make_continuation_pause_node

    seen: list = []

    def interrupt_fn(payload):
        seen.append(payload)
        return {"action": "stop", "expected_epoch": 0}

    node = make_continuation_pause_node(interrupt_fn=interrupt_fn)
    await node(_paused_state())

    assert seen[0]["type"] == "execution_budget_exhausted"
    assert seen[0]["validated_content"] == "partial answer"
    assert seen[0]["active_agent_id"] == "chat_agent"


async def test_a_resume_for_the_wrong_epoch_does_not_run_the_specialist():
    """A stale Continue must not open an epoch against a moved turn."""
    from app.ai.workflow.continuation import make_continuation_pause_node

    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: {"action": "continue", "expected_epoch": 5}
    )

    command = await node(_paused_state())

    assert command.goto == "finalize"
    assert command.update["execution_phase"] == "finalizing"


async def test_an_unreadable_resume_finalizes_rather_than_guessing():
    from app.ai.workflow.continuation import make_continuation_pause_node

    node = make_continuation_pause_node(interrupt_fn=lambda payload: "not a decision")

    command = await node(_paused_state())

    assert command.goto == "finalize"


async def test_an_unresolvable_agent_finalizes_rather_than_jumping_blind():
    from app.ai.workflow.continuation import make_continuation_pause_node

    node = make_continuation_pause_node(
        interrupt_fn=lambda payload: {"action": "continue", "expected_epoch": 0}
    )

    command = await node(_paused_state(active_agent_id=None))

    assert command.goto == "finalize"


# ----------------------------------------------------------------------
# routing into the pause
# ----------------------------------------------------------------------


class _AcceptingValidator:
    async def validate(self, outcome, state):
        return outcome


def _budget_settings(total_epochs: int = 5):
    from types import SimpleNamespace

    return SimpleNamespace(
        generation_soft_model_calls_per_epoch=7,
        generation_hard_model_calls_per_epoch=9,
        generation_soft_tool_calls_per_epoch=12,
        generation_hard_tool_calls_per_epoch=16,
        generation_total_epochs_per_turn=total_epochs,
    )


async def test_an_ordinary_answer_still_goes_straight_to_finalize():
    from app.ai.workflow.finalization import make_validate_output_node

    node = make_validate_output_node(_AcceptingValidator(), settings=_budget_settings())

    command = await node(_paused_state(execution_budget={"exhausted_by": None}))

    assert command.goto == "finalize"


async def test_an_exhausted_answer_pauses_for_a_decision():
    from app.ai.workflow.finalization import make_validate_output_node

    node = make_validate_output_node(_AcceptingValidator(), settings=_budget_settings())

    command = await node(_paused_state())

    assert command.goto == "continuation_pause"


async def test_the_pause_only_follows_a_validated_outcome():
    """Validation runs first, and its failure still ends at finalize.

    Offering Continue on something that failed validation would be offering to
    continue an answer the graph just refused.
    """
    from app.ai.workflow.finalization import (
        OutputValidationError,
        make_validate_output_node,
    )

    class _RejectingValidator:
        async def validate(self, outcome, state):
            raise OutputValidationError("nope")

    node = make_validate_output_node(_RejectingValidator(), settings=_budget_settings())

    command = await node(_paused_state())

    assert command.goto == "finalize"
    assert command.update["execution_phase"] == "failed"


async def test_a_turn_with_no_epochs_left_is_not_offered_a_continue():
    """Offering one that Continue would refuse is worse than not offering."""
    from app.ai.workflow.finalization import make_validate_output_node

    node = make_validate_output_node(_AcceptingValidator(), settings=_budget_settings(2))

    command = await node(
        _paused_state(execution_budget={"exhausted_by": "tool_calls", "epochs_used": 2})
    )

    assert command.goto == "finalize"


async def test_a_hard_limit_is_continuable_too():
    """It is still a partial answer with evidence behind it."""
    from app.ai.workflow.finalization import make_validate_output_node

    node = make_validate_output_node(_AcceptingValidator(), settings=_budget_settings())

    command = await node(_paused_state(execution_budget={"exhausted_by": "hard_limit"}))

    assert command.goto == "continuation_pause"


# ----------------------------------------------------------------------
# reading the pause back off a checkpoint
# ----------------------------------------------------------------------


class _Interrupt:
    def __init__(self, value, interrupt_id="i1"):
        self.value = value
        self.id = interrupt_id


class _Task:
    def __init__(self, interrupts, result=None):
        self.interrupts = interrupts
        self.result = result


class _Snapshot:
    def __init__(self, tasks):
        self.tasks = tasks


def _pause_value(**overrides) -> dict:
    value = {
        "type": "execution_budget_exhausted",
        "generation_id": "11111111-1111-1111-1111-111111111111",
        "logical_turn_id": "turn-1",
        "execution_epoch": 0,
        "active_agent_id": "chat_agent",
        "validated_content": "partial answer",
        "budget": {"exhausted_by": "tool_calls"},
    }
    value.update(overrides)
    return value


def _approval_value() -> dict:
    return {
        "action_requests": [{"action": "web_search", "args": {}, "tool_call_id": "c1"}],
        "metadata": {"interrupt_id": "i1"},
    }


def test_a_live_pause_is_recognised():
    from app.ai.workflow.continuation import pending_continuation_payload

    snapshot = _Snapshot([_Task([_Interrupt(_pause_value())])])

    payload = pending_continuation_payload(snapshot)

    assert payload is not None
    assert payload.validated_content == "partial answer"
    assert payload.execution_epoch == 0


def test_an_answered_pause_is_not_reported_as_live():
    """LangGraph keeps reporting a resolved interrupt; only the task result differs."""
    from app.ai.workflow.continuation import pending_continuation_payload

    snapshot = _Snapshot([_Task([_Interrupt(_pause_value())], result={})])

    assert pending_continuation_payload(snapshot) is None


def test_a_tool_approval_is_not_read_as_a_pause():
    from app.ai.workflow.continuation import pending_continuation_payload

    snapshot = _Snapshot([_Task([_Interrupt(_approval_value())])])

    assert pending_continuation_payload(snapshot) is None


def test_a_pause_is_not_read_as_a_tool_approval():
    """The other direction, which is the one that would ask a human to approve nothing."""
    from app.ai.hitl_config import pending_interrupt_payload

    snapshot = _Snapshot([_Task([_Interrupt(_pause_value())])])

    assert pending_interrupt_payload(snapshot) is None


def test_a_malformed_pause_value_is_ignored_rather_than_raised():
    """A checkpoint is read on every resume; raising there strands the turn."""
    from app.ai.workflow.continuation import pending_continuation_payload

    snapshot = _Snapshot([_Task([_Interrupt({"type": "execution_budget_exhausted"})])])

    assert pending_continuation_payload(snapshot) is None


def test_no_interrupts_at_all_is_not_a_pause():
    from app.ai.workflow.continuation import pending_continuation_payload

    assert pending_continuation_payload(_Snapshot([])) is None
