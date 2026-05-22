from __future__ import annotations

import json

from app.ui.subagent_activity import (
    build_live_subagent_activity_view,
    build_subagent_activity_view,
)


def test_build_subagent_activity_view_uses_explicit_metadata():
    metadata = {
        "subagent_dispatches": [
            {
                "rationale": "parallel checks",
                "task_ids": ["w1", "w2"],
                "agents": ["search_agent", "chat_agent"],
                "status": "partial",
            }
        ],
        "subagent_results": [
            {
                "id": "w1",
                "agent": "search_agent",
                "status": "completed",
                "elapsed_ms": 1200,
                "summary": "Found current docs.",
                "related_todo_ids": ["todo-1"],
            },
            {
                "id": "w2",
                "agent": "chat_agent",
                "status": "timeout",
                "elapsed_ms": 5000,
                "summary": "Timed out.",
                "error": "timeout",
                "related_todo_ids": ["todo-2"],
            },
        ],
    }

    view = build_subagent_activity_view(metadata)

    assert view is not None
    assert view["total"] == 2
    assert view["completed"] == 1
    assert view["status"] == "partial"
    assert view["results"][0]["id"] == "w1"
    assert view["results"][1]["status"] == "timeout"


def test_build_subagent_activity_view_falls_back_to_tool_artifact_render_payload():
    metadata = {
        "tool_artifacts": [
            {
                "tool": "dispatch_subagents",
                "render": {
                    "type": "subagent_dispatch",
                    "structured_content": {
                        "status": "completed",
                        "rationale": "split work",
                        "results": [
                            {
                                "id": "worker-a",
                                "agent": "search_agent",
                                "status": "completed",
                                "elapsed_ms": 400,
                                "summary": "done",
                            }
                        ],
                    },
                },
            }
        ]
    }

    view = build_subagent_activity_view(metadata)

    assert view is not None
    assert view["status"] == "completed"
    assert view["rationales"] == ["split work"]
    assert view["results"][0]["agent"] == "search_agent"


def test_build_subagent_activity_view_falls_back_to_raw_json_result():
    metadata = {
        "tool_artifacts": [
            {
                "tool": "dispatch_subagents",
                "result": json.dumps(
                    {
                        "status": "failed",
                        "results": [
                            {
                                "id": "worker-b",
                                "agent": "rag_agent",
                                "status": "failed",
                                "elapsed_ms": 300,
                                "summary": "failed",
                                "error": "boom",
                            }
                        ],
                    }
                ),
            }
        ]
    }

    view = build_subagent_activity_view(metadata)

    assert view is not None
    assert view["status"] == "failed"
    assert view["failed"] == 1
    assert view["results"][0]["error"] == "boom"


def test_build_live_subagent_activity_view_from_dispatch_start():
    event = {
        "type": "tool",
        "name": "dispatch_subagents",
        "phase": "start",
        "args": {
            "rationale": "split independent checks",
            "tasks": [
                {
                    "id": "worker-a",
                    "agent": "search_agent",
                    "task": "Check the current docs.",
                    "related_todo_ids": ["todo-1"],
                },
                {
                    "id": "worker-b",
                    "agent": "chat_agent",
                    "task": "Draft the answer.",
                    "related_todo_ids": ["todo-2"],
                },
            ],
        },
    }

    view = build_live_subagent_activity_view(event)

    assert view is not None
    assert view["status"] == "running"
    assert view["total"] == 2
    assert view["completed"] == 0
    assert view["rationales"] == ["split independent checks"]
    assert view["results"][0]["status"] == "running"
    assert view["results"][0]["summary"] == "Check the current docs."


def test_build_live_subagent_activity_view_from_planning_node_complete():
    event = {
        "type": "node_complete",
        "node": "planning_agent",
        "tool_calls": [
            {
                "id": "dispatch-1",
                "name": "dispatch_subagents",
                "args": {
                    "rationale": "parallelizable work",
                    "tasks": [
                        {
                            "id": "worker-a",
                            "agent": "search_agent",
                            "task": "Find source material.",
                        }
                    ],
                },
            }
        ],
    }

    view = build_live_subagent_activity_view(event)

    assert view is not None
    assert view["status"] == "running"
    assert view["total"] == 1
    assert view["rationales"] == ["parallelizable work"]
    assert view["results"][0]["summary"] == "Find source material."


def test_build_live_subagent_activity_view_updates_from_dispatch_end_render():
    previous = build_live_subagent_activity_view(
        {
            "type": "tool",
            "name": "dispatch_subagents",
            "phase": "start",
            "args": {
                "tasks": [
                    {
                        "id": "worker-a",
                        "agent": "search_agent",
                        "task": "Check the current docs.",
                    }
                ]
            },
        }
    )
    event = {
        "type": "tool",
        "name": "dispatch_subagents",
        "phase": "end",
        "render": {
            "type": "subagent_dispatch",
            "structured_content": {
                "status": "completed",
                "rationale": "split independent checks",
                "results": [
                    {
                        "id": "worker-a",
                        "agent": "search_agent",
                        "status": "completed",
                        "elapsed_ms": 1200,
                        "summary": "Docs checked.",
                    }
                ],
            },
        },
    }

    view = build_live_subagent_activity_view(event, previous=previous)

    assert view is not None
    assert view["status"] == "completed"
    assert view["completed"] == 1
    assert view["results"][0]["summary"] == "Docs checked."


def test_build_live_subagent_activity_view_ignores_other_tools():
    previous = {"status": "running", "total": 1, "results": []}

    view = build_live_subagent_activity_view(
        {"type": "tool", "name": "web_search", "phase": "start"},
        previous=previous,
    )

    assert view is previous


def test_build_live_subagent_activity_view_keeps_previous_when_start_args_are_incomplete():
    previous = {"status": "running", "total": 1, "results": [{"id": "worker-a"}]}

    view = build_live_subagent_activity_view(
        {"type": "tool", "name": "dispatch_subagents", "phase": "start", "args": {}},
        previous=previous,
    )

    assert view is previous


# ---------------------------------------------------------------------------
# Phase 10 follow-up: subagent activity surfaces requested/resolved model
# ---------------------------------------------------------------------------


def test_build_subagent_activity_view_surfaces_requested_and_resolved_model():
    metadata = {
        "subagent_results": [
            {
                "id": "w1",
                "agent": "search_agent",
                "status": "completed",
                "elapsed_ms": 1200,
                "summary": "Found current docs.",
                "requested_model": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "reasoning_effort": "high",
                },
                "resolved_model": {
                    "provider": "openai",
                    "model": "gpt-5.4",
                    "config_source": "request",
                    "reasoning_effort": "high",
                    "context_window": {
                        "used_tokens": 4096,
                        "used_token_source": "actual_total",
                        "display_state": "ok",
                    },
                },
            }
        ]
    }

    view = build_subagent_activity_view(metadata)

    assert view is not None
    result = view["results"][0]
    assert result["requested_model"]["model"] == "gpt-5.4"
    assert result["requested_model"]["reasoning_effort"] == "high"
    assert result["resolved_model"]["model"] == "gpt-5.4"
    assert result["resolved_model"]["config_source"] == "request"
    assert result["resolved_model"]["context_window"]["used_tokens"] == 4096


def test_build_subagent_activity_view_keeps_resolved_model_without_override():
    """If only resolved_model is present (no explicit override), it should still
    surface so the UI can show which model actually answered.
    """
    metadata = {
        "subagent_results": [
            {
                "id": "w1",
                "agent": "chat_agent",
                "status": "completed",
                "elapsed_ms": 800,
                "summary": "ok",
                "resolved_model": {
                    "provider": "gemini",
                    "model": "gemini-3-flash-preview",
                    "config_source": "default",
                },
            }
        ]
    }

    view = build_subagent_activity_view(metadata)

    assert view is not None
    result = view["results"][0]
    assert "requested_model" not in result
    assert result["resolved_model"]["model"] == "gemini-3-flash-preview"


def test_build_live_subagent_activity_view_surfaces_requested_model_for_pending_workers():
    """Live activity should show the requested model for pending workers so the
    user sees which model the supervisor assigned before results arrive.
    """
    event = {
        "type": "tool",
        "name": "dispatch_subagents",
        "phase": "start",
        "args": {
            "tasks": [
                {
                    "id": "worker-a",
                    "agent": "search_agent",
                    "task": "Check the current docs.",
                    "model_override": {
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "reasoning_effort": "high",
                    },
                }
            ]
        },
    }

    view = build_live_subagent_activity_view(event)

    assert view is not None
    result = view["results"][0]
    assert result["requested_model"]["model"] == "gpt-5.4"
    assert result["requested_model"]["reasoning_effort"] == "high"
