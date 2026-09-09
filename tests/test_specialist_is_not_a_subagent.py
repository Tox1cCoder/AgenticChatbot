"""A specialist running its own subgraph is not a dispatched subagent.

``_translate_lifecycle`` turns LangGraph subgraph lifecycle events into
``subagent_start``/``subagent_end``. That was written when the only subgraphs in
the run *were* delegated workers. Routing-v2 gives every specialist its own
compiled ``create_agent`` subgraph, so the same lifecycle now fires on every
ordinary turn: sending "Hello" reported a "Chat" subagent nobody dispatched.

Genuine delegated workers are announced from the dispatch-validation and
worker-tool channels, which carry the dispatch and task identity the UI groups
by; they do not depend on this lifecycle path.
"""

from __future__ import annotations

import pytest

from app.services.event_streaming.langchain_v3 import (
    PUBLIC_ANSWER_NODES,
    V3ProtocolTranslator,
)


def _lifecycle(event: str, namespace: list[str], *, seq: int = 1, graph_name: str | None = None):
    data: dict = {"event": event, "namespace": namespace}
    if graph_name is not None:
        data["graph_name"] = graph_name
    return {
        "type": "event",
        "method": "lifecycle",
        "params": {"namespace": [], "timestamp": 0, "data": data},
        "seq": seq,
    }


def _types(events) -> list[str]:
    return [event.type for event in events]


@pytest.mark.parametrize("specialist", sorted(PUBLIC_ANSWER_NODES))
def test_a_specialist_subgraph_raises_no_subagent(specialist):
    translator = V3ProtocolTranslator()
    namespace = [f"{specialist}:run-1"]

    started = list(translator.translate(_lifecycle("started", namespace, seq=1)))
    completed = list(translator.translate(_lifecycle("completed", namespace, seq=2)))

    assert started == [], f"{specialist} announced itself as a subagent"
    assert completed == []


def test_a_genuine_subflow_still_reports_a_subagent():
    """The delegated case this path exists for must keep working."""
    translator = V3ProtocolTranslator()
    namespace = ["subflow:abc"]

    started = list(
        translator.translate(_lifecycle("started", namespace, seq=1, graph_name="subflow"))
    )
    completed = list(translator.translate(_lifecycle("completed", namespace, seq=2)))

    assert _types(started) == ["subagent_start"]
    assert started[0].subagent is not None
    assert started[0].subagent.status == "running"
    assert _types(completed) == ["subagent_end"]
    assert completed[0].subagent.status == "completed"


def test_a_failed_subflow_still_reports_its_status():
    translator = V3ProtocolTranslator()

    failed = list(translator.translate(_lifecycle("failed", ["subflow:abc"], seq=1)))

    assert _types(failed) == ["subagent_end"]
    assert failed[0].subagent.status == "failed"
