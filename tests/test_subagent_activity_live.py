from __future__ import annotations

import pytest

from app.ui.subagent_activity import build_live_subagent_activity_view


def _subagent_event(phase, *, worker_id, agent, status, **extra):
    return {
        "type": "subagent",
        "phase": phase,
        "subagent": {
            "id": worker_id,
            "name": agent,
            "path": ["planning_agent", worker_id],
            "status": status,
        },
        **extra,
    }


def test_live_view_tracks_each_worker_independently():
    view = build_live_subagent_activity_view(
        _subagent_event(
            "start", worker_id="w1", agent="search_agent", status="running", task="Search"
        ),
        previous=None,
    )
    view = build_live_subagent_activity_view(
        _subagent_event("start", worker_id="w2", agent="rag_agent", status="running", task="Read"),
        previous=view,
    )
    assert view["total"] == 2
    assert view["running"] == 2
    assert view["status"] == "running"

    view = build_live_subagent_activity_view(
        _subagent_event(
            "end", worker_id="w1", agent="search_agent", status="completed", summary="found"
        ),
        previous=view,
    )
    assert view["completed"] == 1
    assert view["running"] == 1
    by_id = {row["id"]: row for row in view["results"]}
    assert by_id["w1"]["status"] == "completed"
    assert by_id["w1"]["summary"] == "found"
    assert by_id["w2"]["status"] == "running"

    view = build_live_subagent_activity_view(
        _subagent_event(
            "end", worker_id="w2", agent="rag_agent", status="completed", summary="read"
        ),
        previous=view,
    )
    assert view["status"] == "completed"
    assert view["completed"] == 2


@pytest.mark.parametrize("worker_status", ["failed", "timeout", "requires_approval"])
def test_live_view_does_not_mark_blocked_worker_dispatch_completed(worker_status):
    view = build_live_subagent_activity_view(
        _subagent_event(
            "end",
            worker_id="w1",
            agent="search_agent",
            status=worker_status,
            error=worker_status,
        ),
        previous=None,
    )

    assert view["status"] == "failed"
    assert view["completed"] == 0
    assert view["failed"] == 1


def test_live_view_reports_partial_when_only_some_workers_complete():
    view = build_live_subagent_activity_view(
        _subagent_event("end", worker_id="w1", agent="search_agent", status="completed"),
        previous=None,
    )
    view = build_live_subagent_activity_view(
        _subagent_event("end", worker_id="w2", agent="rag_agent", status="timeout"),
        previous=view,
    )

    assert view["status"] == "partial"
    assert view["completed"] == 1
    assert view["failed"] == 1
