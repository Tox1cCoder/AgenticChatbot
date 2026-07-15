"""Helpers used by the Streamlit HITL approval UI."""

from app.ui.hitl_decisions import (
    approval_tool_label,
    attach_stream_context,
    build_interrupt_decision,
    interrupt_allowed_decisions,
    interrupt_request_target_ids,
    interrupt_stream_context,
)


def test_build_interrupt_decision_preserves_distinct_task_and_tool_call_ids():
    request = {
        "task_id": "approval-row-1",
        "tool_call_id": "35f2c29f-e553-441b-b2d4-e10b23529040",
        "action": "client__desktop_commander__write_file",
    }

    decision = build_interrupt_decision("approve", request, args=None)

    assert decision == {
        "type": "approve",
        "task_id": "approval-row-1",
        "tool_call_id": "35f2c29f-e553-441b-b2d4-e10b23529040",
        "action": "client__desktop_commander__write_file",
        "args": None,
    }


def test_interrupt_request_target_ids_support_camel_case_and_id_fallback():
    assert interrupt_request_target_ids(
        {"taskId": "approval-row-1", "toolCallId": "tool-call-1"}
    ) == ("approval-row-1", "tool-call-1")
    assert interrupt_request_target_ids({"id": "tool-call-2"}) == (
        "tool-call-2",
        "tool-call-2",
    )


def test_attach_and_read_stream_context_roundtrip():
    interrupt = {"interrupt_id": "i1", "action_requests": []}
    attach_stream_context(interrupt, thinking="  weighing options  ", content="Here is a plan")
    assert interrupt_stream_context(interrupt) == ("weighing options", "Here is a plan")


def test_attach_stream_context_ignores_blank_and_non_dict():
    interrupt = {"interrupt_id": "i1"}
    attach_stream_context(interrupt, thinking="   ", content="")
    assert "agent_thinking" not in interrupt
    assert "agent_partial_content" not in interrupt
    # Non-dict is returned unchanged and reads as empty.
    assert attach_stream_context(None, thinking="x", content="y") is None
    assert interrupt_stream_context(None) == ("", "")


def test_interrupt_stream_context_defaults_to_empty():
    assert interrupt_stream_context({}) == ("", "")


def test_interrupt_allowed_decisions_normalizes_legacy_and_camel_case_payloads():
    assert interrupt_allowed_decisions({}) == frozenset({"approve", "edit", "reject"})
    assert interrupt_allowed_decisions({"allowedDecisions": ["APPROVE", "reject"]}) == (
        frozenset({"approve", "reject"})
    )
    assert interrupt_allowed_decisions({"allowed_decisions": "edit"}) == frozenset({"edit"})


def test_interrupt_allowed_decisions_fails_closed_for_malformed_explicit_value():
    assert interrupt_allowed_decisions({"allowed_decisions": {"approve": True}}) == frozenset()


def test_approval_tool_label_numbers_cumulatively_across_turns():
    # First interrupt of a turn: base 0 → Tool 1.
    assert approval_tool_label(0, 0, "client__dc__read_file") == "Tool 1: `client__dc__read_file`"
    # Second single-tool interrupt: base advanced to 1 → Tool 2 (not Tool 1 again).
    assert approval_tool_label(1, 0, "client__dc__write_file") == "Tool 2: `client__dc__write_file`"
    # Multi-tool interrupt after one resolved: Tool 2 and Tool 3.
    assert approval_tool_label(1, 0, "a") == "Tool 2: `a`"
    assert approval_tool_label(1, 1, "b") == "Tool 3: `b`"


def test_approval_tool_label_tolerates_bad_base():
    assert approval_tool_label(None, 0, "x") == "Tool 1: `x`"
    assert approval_tool_label(-5, 0, "x") == "Tool 1: `x`"
