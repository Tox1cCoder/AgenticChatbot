from __future__ import annotations

from app.core.response_constants import build_bot_metadata
from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage
from app.services.message_service import _merge_stream_tool_artifacts_into_response
from app.ui.subagent_activity import build_subagent_activity_view


def test_merge_stream_tool_artifacts_preserves_subagent_render_payload():
    response = WorkflowResponse(
        agent_type="planning",
        agent_id="planning_agent",
        message=WorkflowResponseMessage(content="done"),
        metadata={},
    )
    stream_artifacts = [
        {
            "tool_call_id": "dispatch-1",
            "tool": "dispatch_subagents",
            "args": {
                "tasks": [
                    {
                        "id": "worker-a",
                        "agent": "search_agent",
                        "task": "Find source material.",
                    }
                ]
            },
            "output": '{"status":"completed","results":[]}',
            "error": None,
            "status": "success",
            "render": {
                "type": "subagent_dispatch",
                "structured_content": {
                    "status": "completed",
                    "results": [
                        {
                            "id": "worker-a",
                            "agent": "search_agent",
                            "status": "completed",
                            "summary": "Found source material.",
                        }
                    ],
                },
            },
        }
    ]

    _merge_stream_tool_artifacts_into_response(response, stream_artifacts)
    metadata = build_bot_metadata(response)
    view = build_subagent_activity_view(metadata)

    assert view is not None
    assert view["status"] == "completed"
    assert view["results"][0]["summary"] == "Found source material."


def test_merge_stream_tool_artifacts_enriches_existing_artifact_without_render():
    response = WorkflowResponse(
        agent_type="planning",
        agent_id="planning_agent",
        message=WorkflowResponseMessage(content="done"),
        metadata={},
        tool_artifacts=[
            {
                "tool_call_id": "dispatch-1",
                "tool": "dispatch_subagents",
                "args": {},
                "output": '{"status":"completed","results":[]}',
                "error": None,
                "status": "success",
            }
        ],
    )
    stream_artifacts = [
        {
            "tool_call_id": "dispatch-1",
            "tool": "dispatch_subagents",
            "render": {
                "type": "subagent_dispatch",
                "structured_content": {
                    "status": "completed",
                    "results": [
                        {
                            "id": "worker-a",
                            "agent": "search_agent",
                            "status": "completed",
                            "summary": "Found source material.",
                        }
                    ],
                },
            },
        }
    ]

    _merge_stream_tool_artifacts_into_response(response, stream_artifacts)

    assert response.tool_artifacts is not None
    assert response.tool_artifacts[0]["render"]["type"] == "subagent_dispatch"
