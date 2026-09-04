"""Typed Planning worker events reach the public stream.

Task 4 gave the Planning nodes an injected ``StreamWriter``. Probing LangGraph
1.2.9 showed those events reached nobody: ``astream_events(version="v3")``
derives its stream modes from a transformer mux, and none of the four native
transformers subscribes to ``custom`` — so the graph was never asked to emit
that channel and every ``writer(...)`` call was discarded.

The mux is the supported extension point. A transformer declaring
``required_stream_modes = ("custom",)`` makes v3 request the channel, and the
events then arrive on the main iterator as ``method: "custom"`` with
``params.namespace`` and ``params.data``.

What is projected is deliberately narrow: worker identity and lifecycle. An
objective, a worker's answer text, an execution key, a provider receipt, or
document content reaching the public stream would leak private work into the
user-visible trace.
"""

from __future__ import annotations

from typing import Annotated

from typing_extensions import TypedDict

from app.services.event_streaming.langchain_v3 import V3ProtocolTranslator


def _extend_results(existing: list | None, update: list | None) -> list:
    return [*(existing or []), *(update or [])]


class _FanOutState(TypedDict):
    results: Annotated[list, _extend_results]


def _custom(data: dict, namespace: list[str] | None = None) -> dict:
    return {
        "type": "event",
        "method": "custom",
        "params": {"namespace": namespace or [], "timestamp": 0, "data": data},
    }


def _translate(*events: dict):
    translator = V3ProtocolTranslator()
    produced = []
    for event in events:
        produced.extend(translator.translate(event))
    return produced


# ----------------------------------------------------------------------
# worker lifecycle
# ----------------------------------------------------------------------


def test_a_worker_start_becomes_a_running_subagent_event():
    [event] = _translate(
        _custom(
            {
                "type": "planning_worker",
                "phase": "start",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
            }
        )
    )

    assert event.type == "subagent_start"
    assert event.subagent is not None
    assert event.subagent.id == "d1:t1"
    assert event.subagent.name == "search_agent"
    assert event.subagent.status == "running"


def test_a_completed_worker_end_becomes_a_completed_subagent_event():
    [event] = _translate(
        _custom(
            {
                "type": "planning_worker",
                "phase": "end",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
                "status": "completed",
            }
        )
    )

    assert event.type == "subagent_end"
    assert event.subagent.status == "completed"


def test_a_failed_worker_end_carries_the_failure_status_and_code():
    [event] = _translate(
        _custom(
            {
                "type": "planning_worker",
                "phase": "end",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
                "status": "failed",
                "error_code": "worker_timeout",
            }
        )
    )

    assert event.type == "subagent_end"
    assert event.subagent.status == "failed"
    assert event.data["error_code"] == "worker_timeout"


def test_a_paused_worker_is_reported_as_requiring_approval():
    [event] = _translate(
        _custom(
            {
                "type": "planning_worker",
                "phase": "interrupt",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
            }
        )
    )

    assert event.type == "subagent_end"
    assert event.subagent.status == "requires_approval"


def test_parallel_workers_keep_distinct_subagent_ids():
    events = _translate(
        _custom(
            {
                "type": "planning_worker",
                "phase": "start",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
            }
        ),
        _custom(
            {
                "type": "planning_worker",
                "phase": "start",
                "dispatch_id": "d1",
                "task_id": "t2",
                "agent_id": "rag_agent",
            }
        ),
    )

    assert [event.subagent.id for event in events] == ["d1:t1", "d1:t2"]


# ----------------------------------------------------------------------
# worker tool lifecycle
# ----------------------------------------------------------------------


def test_worker_tool_start_and_end_are_attributed_to_the_worker():
    start, end = _translate(
        _custom(
            {
                "type": "planning_worker_tool",
                "phase": "start",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
                "tool_call_id": "call-1",
                "tool_name": "search_documents",
            }
        ),
        _custom(
            {
                "type": "planning_worker_tool",
                "phase": "end",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
                "tool_call_id": "call-1",
                "tool_name": "search_documents",
                "status": "success",
            }
        ),
    )

    assert start.type == "subagent_tool_execution_start"
    assert end.type == "subagent_tool_execution_end"
    assert start.tool_call_id == end.tool_call_id == "call-1"
    assert start.tool_name == "search_documents"
    assert start.subagent.id == "d1:t1"


# ----------------------------------------------------------------------
# dispatch lifecycle
# ----------------------------------------------------------------------


def test_a_validated_dispatch_reports_its_wave_and_task_count():
    [event] = _translate(
        _custom(
            {
                "type": "planning_dispatch",
                "phase": "validated",
                "dispatch_id": "d1",
                "wave": 1,
                "task_count": 3,
            }
        )
    )

    assert event.type == "subagent_start"
    assert event.data == {"dispatch_id": "d1", "wave": 1, "task_count": 3}


def test_a_collected_wave_reports_completion():
    [event] = _translate(
        _custom(
            {
                "type": "planning_dispatch",
                "phase": "collected",
                "dispatch_id": "d1",
                "wave": 1,
                "task_count": 3,
            }
        )
    )

    assert event.type == "subagent_end"
    assert event.data["wave"] == 1


# ----------------------------------------------------------------------
# nothing private leaks
# ----------------------------------------------------------------------


def test_unknown_custom_events_are_not_projected():
    """The channel is shared; only Planning's own event types are public."""
    assert _translate(_custom({"type": "something_else", "secret": "x"})) == []
    assert _translate(_custom({"no_type": True})) == []


def test_only_allowlisted_fields_survive_projection():
    """A writer that grows a field must not silently publish it."""
    [event] = _translate(
        _custom(
            {
                "type": "planning_worker",
                "phase": "end",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
                "status": "completed",
                "objective": "the private objective",
                "content": "the worker's private answer",
                "execution_key": "a" * 64,
                "provider_receipt_id": "prov-1",
            }
        )
    )

    rendered = event.model_dump_json()
    for secret in ("the private objective", "the worker's private answer", "a" * 64, "prov-1"):
        assert secret not in rendered
    assert set(event.data) <= {"dispatch_id", "task_id", "wave", "task_count", "error_code"}


def test_a_nested_namespace_is_reported_but_does_not_become_answer_text():
    """Worker events must never be mistaken for the public answer stream."""
    events = _translate(
        _custom(
            {
                "type": "planning_worker",
                "phase": "start",
                "dispatch_id": "d1",
                "task_id": "t1",
                "agent_id": "search_agent",
            },
            namespace=["planning_worker:abc"],
        )
    )

    assert [event.type for event in events] == ["subagent_start"]
    assert events[0].namespace == ["planning_worker:abc"]


# ----------------------------------------------------------------------
# the channel is actually subscribed
# ----------------------------------------------------------------------


async def test_writer_events_reach_the_iterator_through_a_real_graph():
    """End-to-end: without the custom transformer these events are discarded.

    ``astream_events(version="v3")`` requests exactly the union of
    ``required_stream_modes`` across its transformers. This is the test that
    would have caught Task 4 shipping a writer nobody was listening to.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.config import get_stream_writer
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    from app.services.event_streaming.langchain_v3 import iter_v3_events_from_graph

    async def _worker(payload: dict) -> dict:
        writer = get_stream_writer()
        name = payload["name"]
        for phase, extra in (("start", {}), ("end", {"status": "completed"})):
            writer(
                {
                    "type": "planning_worker",
                    "phase": phase,
                    "dispatch_id": "d1",
                    "task_id": name,
                    "agent_id": "search_agent",
                    **extra,
                }
            )
        return {"results": [name]}

    graph = StateGraph(_FanOutState)
    graph.add_node("planning_dispatch", lambda _state: {})
    graph.add_node("planning_worker", _worker)
    graph.add_node("planning_collect", lambda _state: {})
    graph.add_conditional_edges(
        "planning_dispatch",
        lambda _state: [Send("planning_worker", {"name": n}) for n in ("t1", "t2")],
        ["planning_worker"],
    )
    graph.add_edge("planning_worker", "planning_collect")
    graph.add_edge(START, "planning_dispatch")
    graph.add_edge("planning_collect", END)
    compiled = graph.compile(checkpointer=InMemorySaver())

    seen = [
        event
        async for event in iter_v3_events_from_graph(
            compiled, {"results": []}, config={"configurable": {"thread_id": "t-custom"}}
        )
        if event.type in ("subagent_start", "subagent_end")
    ]

    assert {event.subagent.id for event in seen} == {"d1:t1", "d1:t2"}
    assert {event.type for event in seen} == {"subagent_start", "subagent_end"}
