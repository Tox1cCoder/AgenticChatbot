"""Pending checkpoint interrupts are the recovery authority.

Two heuristics used to stand in for "is this turn waiting on a human", and both
break now that Planning workers pause in parallel inside the outer graph:

* a node-name allowlist (``_APPROVAL_INTERRUPT_NODES``) — which never listed
  ``planning_worker``, so a paused worker looked like a crashed turn; and
* "the last message is an ``AIMessage`` with tool calls" — which a worker
  cannot satisfy, because its tool call lives in its own private message list
  and never reaches the parent's.

The replacement reads the checkpoint's own pending interrupts. Two properties
of that data are load-bearing and were established by probing LangGraph 1.2.9
rather than by reading it:

1. ``snapshot.interrupts`` keeps reporting an interrupt **after it has been
   answered**. Only ``task.result is None`` distinguishes a task still waiting
   from one already resolved. Presenting a resolved interrupt again would ask
   the human to re-approve a call that already ran.
2. Answering a subset of pending interrupts leaves the rest pending, and the
   answered worker runs exactly once. Partial resume is therefore legitimate,
   and an omitted id is not a rejection.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.hitl_config import PendingInterruptPayload, pending_interrupt_payload


def _interrupt(interrupt_id: str, *tool_call_ids: str, metadata: dict | None = None):
    value = {
        "action_requests": [
            {"tool_call_id": call_id, "id": call_id, "name": "write_file", "args": {}}
            for call_id in tool_call_ids
        ]
    }
    if metadata is not None:
        value["metadata"] = metadata
    return SimpleNamespace(id=interrupt_id, value=value)


def _task(name: str, *interrupts, result=None):
    return SimpleNamespace(
        name=name, id=f"task-{name}", interrupts=tuple(interrupts), result=result
    )


def _snapshot(*tasks, values=None, next_nodes=None):
    return SimpleNamespace(
        tasks=tuple(tasks),
        interrupts=tuple(item for task in tasks for item in task.interrupts),
        values=values or {},
        next=tuple(next_nodes or (task.name for task in tasks)),
    )


# ----------------------------------------------------------------------
# no parent message, no node allowlist
# ----------------------------------------------------------------------


def test_a_worker_interrupt_needs_no_parent_ai_tool_call():
    """A worker's tool call never reaches the parent's message list."""
    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1")),
        values={"messages": []},
    )

    payload = pending_interrupt_payload(snapshot)

    assert payload is not None
    assert payload.interrupt_ids == ("i1",)
    assert payload.action_requests[0]["tool_call_id"] == "c1"


def test_recovery_does_not_consult_a_node_name_allowlist():
    """Any node that paused is a pause, including one invented tomorrow."""
    snapshot = _snapshot(_task("some_future_node", _interrupt("i1", "c1")))

    payload = pending_interrupt_payload(snapshot)

    assert payload is not None
    assert payload.interrupt_ids == ("i1",)


def test_a_turn_with_no_pending_interrupt_is_not_paused():
    assert pending_interrupt_payload(_snapshot()) is None
    assert pending_interrupt_payload(_snapshot(_task("finalize"))) is None


def test_a_snapshot_without_tasks_falls_back_to_the_aggregate():
    """Some snapshot shapes expose only ``interrupts``."""
    snapshot = SimpleNamespace(
        tasks=(), interrupts=(_interrupt("i1", "c1"),), values={}, next=("worker",)
    )

    payload = pending_interrupt_payload(snapshot)

    assert payload is not None
    assert payload.interrupt_ids == ("i1",)


# ----------------------------------------------------------------------
# resolved interrupts are not pending
# ----------------------------------------------------------------------


def test_an_answered_interrupt_is_not_presented_again():
    """Probed: LangGraph keeps reporting a resolved interrupt on the snapshot.

    ``task.result`` is the only thing that separates it from a live one. Asking
    the human to re-approve a call that already ran is the failure this guards.
    """
    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1"), result={"worker_results": ["done"]}),
        _task("planning_worker", _interrupt("i2", "c2")),
    )

    payload = pending_interrupt_payload(snapshot)

    assert payload is not None
    assert payload.interrupt_ids == ("i2",)
    assert [request["tool_call_id"] for request in payload.action_requests] == ["c2"]


def test_a_task_returning_an_empty_result_still_counts_as_resolved():
    """``result`` is falsy but not None. Truthiness would re-present it."""
    snapshot = _snapshot(_task("planning_worker", _interrupt("i1", "c1"), result={}))

    assert pending_interrupt_payload(snapshot) is None


def test_every_interrupt_resolved_means_the_turn_is_not_paused():
    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1"), result={"a": 1}),
        _task("planning_worker", _interrupt("i2", "c2"), result={"b": 2}),
    )

    assert pending_interrupt_payload(snapshot) is None


# ----------------------------------------------------------------------
# parallel interrupts keep distinct identity and provenance
# ----------------------------------------------------------------------


def test_parallel_metadata_is_keyed_per_tool_call():
    """``metadata.update`` let the second worker's provenance overwrite the first."""
    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1", metadata={"device_id": "d1"})),
        _task("planning_worker", _interrupt("i2", "c2", metadata={"device_id": "d2"})),
    )

    payload = pending_interrupt_payload(snapshot)

    assert payload is not None
    assert set(payload.metadata_by_tool_call_id) == {"c1", "c2"}
    assert payload.metadata_by_tool_call_id["c1"]["device_id"] == "d1"
    assert payload.metadata_by_tool_call_id["c2"]["device_id"] == "d2"


def test_parallel_interrupts_keep_distinct_ids_and_requests():
    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1")),
        _task("planning_worker", _interrupt("i2", "c2")),
    )

    payload = pending_interrupt_payload(snapshot)

    assert payload is not None
    assert payload.interrupt_ids == ("i1", "i2")
    assert [request["tool_call_id"] for request in payload.action_requests] == ["c1", "c2"]


def test_one_interrupt_covering_two_calls_maps_both_to_its_metadata():
    snapshot = _snapshot(
        _task("chat_agent", _interrupt("i1", "c1", "c2", metadata={"device_id": "d1"}))
    )

    payload = pending_interrupt_payload(snapshot)

    assert payload is not None
    assert payload.interrupt_ids == ("i1",)
    assert set(payload.metadata_by_tool_call_id) == {"c1", "c2"}


def test_an_interrupt_with_no_action_requests_is_ignored():
    """A pause with nothing to approve is not a question for the user."""
    bare = SimpleNamespace(id="i1", value={"metadata": {"device_id": "d1"}})
    snapshot = _snapshot(_task("planning_worker", bare))

    assert pending_interrupt_payload(snapshot) is None


def test_the_payload_carries_no_objective_or_worker_content():
    snapshot = _snapshot(_task("planning_worker", _interrupt("i1", "c1")))
    payload = pending_interrupt_payload(snapshot)

    assert isinstance(payload, PendingInterruptPayload)
    assert set(PendingInterruptPayload.model_fields) == {
        "action_requests",
        "interrupt_ids",
        "metadata_by_tool_call_id",
    }


# ----------------------------------------------------------------------
# resume addressing
# ----------------------------------------------------------------------


def _decision(tool_call_id: str, decision_type: str = "approve") -> dict:
    return {"tool_call_id": tool_call_id, "type": decision_type, "args": None}


def test_resume_is_keyed_by_exact_interrupt_id():
    from app.ai.utils import address_decisions_to_interrupts

    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1")),
        _task("planning_worker", _interrupt("i2", "c2")),
    )
    payload = pending_interrupt_payload(snapshot)

    resume = address_decisions_to_interrupts([_decision("c1"), _decision("c2")], payload)

    assert set(resume) == {"i1", "i2"}
    assert resume["i1"][0]["tool_call_id"] == "c1"


def test_an_omitted_interrupt_is_left_pending_not_rejected():
    """Probed: the answered worker runs once and the rest stay pending."""
    from app.ai.utils import address_decisions_to_interrupts

    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1")),
        _task("planning_worker", _interrupt("i2", "c2")),
    )
    payload = pending_interrupt_payload(snapshot)

    resume = address_decisions_to_interrupts([_decision("c1")], payload)

    assert set(resume) == {"i1"}


def test_a_decision_for_an_unknown_call_is_rejected():
    from app.ai.utils import address_decisions_to_interrupts

    snapshot = _snapshot(_task("planning_worker", _interrupt("i1", "c1")))
    payload = pending_interrupt_payload(snapshot)

    with pytest.raises(ValueError, match="ghost"):
        address_decisions_to_interrupts([_decision("ghost")], payload)


def test_two_decisions_for_the_same_call_are_rejected():
    """A duplicate is ambiguous: approve and reject cannot both be applied."""
    from app.ai.utils import address_decisions_to_interrupts

    snapshot = _snapshot(_task("planning_worker", _interrupt("i1", "c1")))
    payload = pending_interrupt_payload(snapshot)

    with pytest.raises(ValueError, match="c1"):
        address_decisions_to_interrupts([_decision("c1"), _decision("c1", "reject")], payload)


def test_a_decision_for_an_already_resolved_interrupt_is_rejected():
    from app.ai.utils import address_decisions_to_interrupts

    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1"), result={"done": True}),
        _task("planning_worker", _interrupt("i2", "c2")),
    )
    payload = pending_interrupt_payload(snapshot)

    with pytest.raises(ValueError, match="c1"):
        address_decisions_to_interrupts([_decision("c1")], payload)


def test_resuming_with_no_decisions_is_rejected():
    from app.ai.utils import address_decisions_to_interrupts

    snapshot = _snapshot(_task("planning_worker", _interrupt("i1", "c1")))
    payload = pending_interrupt_payload(snapshot)

    with pytest.raises(ValueError):
        address_decisions_to_interrupts([], payload)


# ----------------------------------------------------------------------
# provenance survives rendering
# ----------------------------------------------------------------------


def _provenance_interrupt(interrupt_id: str, call_id: str, server: str, device: str):
    return SimpleNamespace(
        id=interrupt_id,
        value={
            "action_requests": [
                {"tool_call_id": call_id, "id": call_id, "name": "write_file", "args": {}}
            ],
            "metadata": {
                "device_id": device,
                "tool_provenance": {call_id: {"server_name": server, "device_id": device}},
            },
        },
    )


def test_both_workers_provenance_reaches_the_rendered_payload():
    """A shallow metadata merge kept only the last worker's tool_provenance.

    The first call then rendered with no server or device attribution, which is
    exactly the information a human needs to judge the approval.
    """
    snapshot = _snapshot(
        _task("planning_worker", _provenance_interrupt("i1", "c1", "files", "d1")),
        _task("planning_worker", _provenance_interrupt("i2", "c2", "email", "d2")),
    )

    rendered = pending_interrupt_payload(snapshot).to_interrupt_payload()

    provenance = rendered["metadata"]["tool_provenance"]
    assert set(provenance) == {"c1", "c2"}
    assert provenance["c1"]["server_name"] == "files"
    assert provenance["c2"]["server_name"] == "email"


def test_a_single_device_is_promoted_but_two_are_not():
    """One top-level device_id cannot represent two devices, so it is omitted."""
    same = _snapshot(
        _task("planning_worker", _provenance_interrupt("i1", "c1", "files", "d1")),
        _task("planning_worker", _provenance_interrupt("i2", "c2", "email", "d1")),
    )
    differing = _snapshot(
        _task("planning_worker", _provenance_interrupt("i1", "c1", "files", "d1")),
        _task("planning_worker", _provenance_interrupt("i2", "c2", "email", "d2")),
    )

    assert pending_interrupt_payload(same).to_interrupt_payload()["metadata"]["device_id"] == "d1"
    assert (
        "device_id" not in pending_interrupt_payload(differing).to_interrupt_payload()["metadata"]
    )


def test_the_rendered_payload_carries_every_interrupt_id():
    snapshot = _snapshot(
        _task("planning_worker", _interrupt("i1", "c1")),
        _task("planning_worker", _interrupt("i2", "c2")),
    )

    rendered = pending_interrupt_payload(snapshot).to_interrupt_payload()

    assert rendered["interrupt_ids"] == ["i1", "i2"]
    assert rendered["interrupt_id"] == "i1"


def test_build_interrupt_response_consumes_the_rendered_payload():
    from app.ai.hitl_config import build_interrupt_response

    snapshot = _snapshot(
        _task("planning_worker", _provenance_interrupt("i1", "c1", "files", "d1")),
        _task("planning_worker", _provenance_interrupt("i2", "c2", "email", "d2")),
    )
    rendered = pending_interrupt_payload(snapshot).to_interrupt_payload()

    response = build_interrupt_response(rendered, "thread-1", "conv-1")

    target_ids = {request["tool_call_id"] for request in response["action_requests"]}
    assert target_ids == {"c1", "c2"}
