from app.ai.agent_metadata import (
    agent_identity,
    attach_agent_metadata,
    normalize_subagent_metadata,
)
from app.core.response_constants import build_bot_metadata
from app.schemas.workflow import WorkflowResponse, WorkflowResponseMessage


def test_agent_identity_resolves_custom_agent_name():
    custom_agents = {
        "custom_agent:abc": {
            "id": "abc",
            "runtime_agent_id": "custom_agent:abc",
            "name": "Data Analyst",
        }
    }

    assert agent_identity("custom_agent:abc", custom_agents) == {
        "id": "custom_agent:abc",
        "kind": "custom",
        "name": "Data Analyst",
        "custom_agent_id": "abc",
    }


def test_agent_identity_resolves_base_agent_name():
    assert agent_identity("search_agent", {}) == {
        "id": "search_agent",
        "kind": "base",
        "name": "Search Agent",
        "custom_agent_id": None,
    }


def test_attach_agent_metadata_prefers_response_agent_id():
    metadata = {}

    attach_agent_metadata(
        metadata,
        response_agent_id="search_agent",
        selected_agent_id="chat_agent",
        custom_agents={},
    )

    assert metadata["agent"]["id"] == "search_agent"
    assert metadata["agent"]["source"] == "response"


def test_attach_agent_metadata_uses_custom_response_compat_fields():
    metadata = {
        "runtime_agent_id": "custom_agent:abc",
        "custom_agent_id": "abc",
        "custom_agent_name": "Data Analyst",
    }

    attach_agent_metadata(
        metadata,
        response_agent_id=None,
        selected_agent_id=None,
        custom_agents={},
    )

    assert metadata["agent"] == {
        "id": "custom_agent:abc",
        "kind": "custom",
        "name": "Data Analyst",
        "custom_agent_id": "abc",
        "source": "response",
    }


def test_normalize_subagent_metadata_adds_display_names():
    custom_agents = {
        "custom_agent:abc": {
            "id": "abc",
            "runtime_agent_id": "custom_agent:abc",
            "name": "Data Analyst",
        }
    }
    raw = [{"id": "w1", "agent": "custom_agent:abc", "status": "completed", "summary": "ok"}]

    normalized = normalize_subagent_metadata(raw, custom_agents)

    assert normalized == [
        {
            "id": "w1",
            "agent": "custom_agent:abc",
            "agent_name": "Data Analyst",
            "agent_kind": "custom",
            "custom_agent_id": "abc",
            "status": "completed",
            "summary": "ok",
        }
    ]


def test_build_bot_metadata_keeps_agent_and_removes_redundant_custom_fields():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="answer"),
        metadata={
            "agent": {
                "id": "custom_agent:abc",
                "kind": "custom",
                "name": "Data Analyst",
                "custom_agent_id": "abc",
                "source": "response",
            },
            "runtime_agent_id": "custom_agent:abc",
            "custom_agent_name": "Data Analyst",
            "custom_agent_id": "abc",
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["agent"]["name"] == "Data Analyst"
    assert "runtime_agent_id" not in metadata
    assert "custom_agent_name" not in metadata
    assert "custom_agent_id" not in metadata


def test_build_bot_metadata_preserves_custom_agent_warnings():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="answer"),
        metadata={
            "agent": {
                "id": "custom_agent:abc",
                "kind": "custom",
                "name": "Data Analyst",
                "custom_agent_id": "abc",
                "source": "response",
            },
            "custom_agent_warnings": ["Selected client tool is unavailable."],
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["custom_agent_warnings"] == ["Selected client tool is unavailable."]


def test_build_bot_metadata_does_not_remove_legacy_fields_when_agent_missing():
    response = WorkflowResponse(
        message=WorkflowResponseMessage(content="answer"),
        metadata={
            "runtime_agent_id": "custom_agent:abc",
            "custom_agent_name": "Data Analyst",
            "custom_agent_id": "abc",
        },
    )

    metadata = build_bot_metadata(response)

    assert metadata["runtime_agent_id"] == "custom_agent:abc"
