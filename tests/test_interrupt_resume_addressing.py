"""Addressing human decisions to the interrupt that asked for them.

LangGraph accepts a bare resume value only while one interrupt is pending. Once
two are — which is what Planning workers pausing in parallel produces — it
requires ``Command(resume={interrupt_id: value})`` and raises otherwise. The
client does not know interrupt ids and should not have to: every pending
interrupt carries the tool-call ids it is asking about, so the server can
address the decisions itself.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.utils import address_decisions_to_interrupts, build_interrupt_resume_payload


def _decision(tool_call_id: str, decision_type: str = "approve") -> dict:
    return {
        "task_id": tool_call_id,
        "tool_call_id": tool_call_id,
        "type": decision_type,
        "args": None,
    }


def _interrupt(interrupt_id: str, *tool_call_ids: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=interrupt_id,
        value={
            "action_requests": [
                {"name": "write", "tool_call_id": call_id} for call_id in tool_call_ids
            ]
        },
    )


def test_a_single_interrupt_still_resumes_with_a_flat_payload():
    """One pending interrupt is the common case and must not change shape."""
    decisions = [_decision("call-1"), _decision("call-2")]

    addressed = address_decisions_to_interrupts(decisions, [_interrupt("i1", "call-1", "call-2")])

    assert addressed == decisions


def test_each_decision_reaches_the_interrupt_that_asked_for_it():
    decisions = [_decision("call-a"), _decision("call-b", "reject")]
    interrupts = [_interrupt("i1", "call-a"), _interrupt("i2", "call-b")]

    addressed = address_decisions_to_interrupts(decisions, interrupts)

    assert addressed == {"i1": [decisions[0]], "i2": [decisions[1]]}


def test_a_decision_for_an_unknown_call_is_refused_rather_than_guessed():
    """Sending it to the wrong interrupt would approve something else."""
    interrupts = [_interrupt("i1", "call-a"), _interrupt("i2", "call-b")]

    with pytest.raises(ValueError, match="call-zzz"):
        address_decisions_to_interrupts([_decision("call-zzz")], interrupts)


def test_an_unanswered_interrupt_is_refused_rather_than_left_hanging():
    """Resuming without a decision for every pending interrupt makes LangGraph
    replay that node with no answer, which reads as a rejection nobody made."""
    interrupts = [_interrupt("i1", "call-a"), _interrupt("i2", "call-b")]

    with pytest.raises(ValueError, match="i2"):
        address_decisions_to_interrupts([_decision("call-a")], interrupts)


def test_no_pending_interrupts_leaves_the_payload_alone():
    decisions = [_decision("call-1")]
    assert address_decisions_to_interrupts(decisions, []) == decisions


def test_the_payload_builder_is_unchanged_by_addressing():
    """Addressing wraps the existing payload; it does not rewrite decisions."""
    built = build_interrupt_resume_payload(
        [SimpleNamespace(type="approve", task_id=None, tool_call_id="call-1", args=None)]
    )
    assert built == [
        {"task_id": "call-1", "tool_call_id": "call-1", "type": "approve", "args": None}
    ]
