"""Regression: interrupt action_requests must come from the LIVE interrupt value.

A node that calls ``interrupt()`` suspends *before* its writes to ``state["context"]``
are committed, so ``context["pending_action_requests"]`` is never in the checkpoint at
pause time. After the first approval cycle resolves (the node returns normally) a stale
value gets committed and would then be reused for every later interrupt on the thread.
Reading it back from ``snapshot.values`` therefore yields a STALE set of tool calls
whose ids do not match the resumed node's ``last_message.tool_calls`` — which makes
``apply_hitl_decisions`` raise "Missing HITL decision for tool_call_id ...".

The interrupt's own value IS checkpointed and current, so the payload must be recovered
from the snapshot's pending interrupts, not from committed state.
"""

from types import SimpleNamespace

from app.ai.graph import MultiAgentWorkflow
from app.ai.hitl_config import build_interrupt_response
from app.ai.utils import (
    apply_hitl_decisions,
    build_interrupt_resume_payload,
    normalize_tool_call,
)
from app.ui.hitl_decisions import build_interrupt_decision, interrupt_request_target_ids


def _interrupt(value):
    return SimpleNamespace(value=value, id="int-xyz")


def _snapshot(*, interrupts=(), tasks=(), context=None):
    values = {"context": context or {}}
    return SimpleNamespace(interrupts=tuple(interrupts), tasks=tuple(tasks), values=values)


LIVE_CALL = {
    "name": "client__desktop_commander__write_file",
    "args": {"path": "out.txt"},
    "id": "0927a7ed-59bf-434d-8c06-9d7c225ad75e",
    "tool_call_id": "0927a7ed-59bf-434d-8c06-9d7c225ad75e",
}
STALE_CONTEXT = {
    "pending_action_requests": [
        {"name": "client__time__now", "args": {}, "id": "stale-001", "tool_call_id": "stale-001"}
    ]
}


def test_recovers_action_requests_from_live_interrupt_value():
    snapshot = _snapshot(
        interrupts=[_interrupt({"action_requests": [LIVE_CALL], "metadata": {"device_id": "d1"}})],
        context=STALE_CONTEXT,
    )
    payload = MultiAgentWorkflow._interrupt_payload_from_pending_interrupts(snapshot)
    assert payload is not None
    ids = [r.get("id") or r.get("tool_call_id") for r in payload["action_requests"]]
    assert ids == ["0927a7ed-59bf-434d-8c06-9d7c225ad75e"]
    assert payload["metadata"] == {"device_id": "d1"}


def test_recovers_from_task_interrupts_when_aggregate_empty():
    task = SimpleNamespace(interrupts=(_interrupt({"action_requests": [LIVE_CALL]}),))
    snapshot = _snapshot(interrupts=(), tasks=[task], context=STALE_CONTEXT)
    payload = MultiAgentWorkflow._interrupt_payload_from_pending_interrupts(snapshot)
    assert payload is not None
    assert payload["action_requests"][0]["id"] == "0927a7ed-59bf-434d-8c06-9d7c225ad75e"


def test_returns_none_when_no_pending_interrupts():
    snapshot = _snapshot(interrupts=(), tasks=(), context=STALE_CONTEXT)
    assert MultiAgentWorkflow._interrupt_payload_from_pending_interrupts(snapshot) is None


def test_approve_all_roundtrip_uses_live_ids_not_stale_state():
    """End-to-end: stale committed state + live interrupt -> 'Approve All' must resolve."""
    snapshot = _snapshot(
        interrupts=[_interrupt({"action_requests": [LIVE_CALL]})],
        context=STALE_CONTEXT,
    )
    interrupt_payload = MultiAgentWorkflow._interrupt_payload_from_pending_interrupts(snapshot)

    # Server builds the UI-facing response from the recovered payload.
    interrupt_payload["interrupt_id"] = "int-1"
    response = build_interrupt_response(interrupt_payload, "thread-1", "conv-1")
    action_requests = response["action_requests"]

    # Demo "Approve All": one approve decision per shown action_request.
    decisions = []
    for req in action_requests:
        task_id, _tcid = interrupt_request_target_ids(req)
        decisions.append(build_interrupt_decision("approve", req, args=None))
        assert task_id == "0927a7ed-59bf-434d-8c06-9d7c225ad75e"

    resume_data = build_interrupt_resume_payload(decisions)

    # Graph applies decisions against the ACTUAL pending tool calls (last_message).
    approved, rejected = apply_hitl_decisions([normalize_tool_call(LIVE_CALL)], resume_data)
    assert len(approved) == 1
    assert rejected == {}
