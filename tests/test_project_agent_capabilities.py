"""Inherited agent settings must agree with live runtime capabilities."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai.skill_resolver import ResolvedSkill, filter_skills_by_refs
from app.ai.workflow.custom_agents import CustomAgentsMixin
from app.models.custom_agent import CustomAgent
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.services.custom_agent_capability_resolver import resolve_custom_agent_capabilities
from app.services.custom_agent_service import CustomAgentService
from app.services.project_service import ProjectService


def _agent(owner, *, enabled=True):
    return CustomAgent(
        id=uuid4(),
        owner_id=owner,
        name="Inherited",
        slug="inherited",
        prompt="Help",
        provider_type="openai",
        model="gpt-4.1-mini",
        enabled=enabled,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        skill_refs=[{"source": "client", "lookup_name": "calendar", "name": "calendar"}],
        tool_refs=[],
    )


def test_disabled_inherited_agent_is_absent_from_runtime_roster():
    agent = _agent(uuid4(), enabled=False)
    state = CustomAgentService._runtime_state_from_attachments(
        [(SimpleNamespace(agent_order=0), agent)]
    )
    assert state == {}, "disabled saved agents must not be executable"


def test_disabled_agent_is_not_a_handoff_target_or_constructible():
    workflow = CustomAgentsMixin()
    workflow.agents = {"chat_agent": object()}
    agent_id = f"custom_agent:{uuid4()}"
    state = {"custom_agents": {agent_id: {"name": "Disabled", "enabled": False}}}
    assert workflow._handoff_targets(state, "chat_agent") == []
    assert workflow._custom_handoff_target_descriptions(state, "chat_agent") == {}
    assert workflow._build_custom_agent(state, agent_id) is None


@pytest.mark.parametrize("source", ["server", "client"])
def test_legacy_skill_availability_agrees_with_runtime_visibility(source):
    ref = {"source": source, "lookup_name": "calendar", "name": "calendar"}
    skill = ResolvedSkill("calendar", "Calendar", "client", "calendar")
    availability = resolve_custom_agent_capabilities(
        selected_tool_refs=[],
        selected_skill_refs=[ref],
        live_tool_refs=[],
        live_skill_refs=[skill],
        request_device_id="device-a",
        device_available=True,
    )
    assert availability.status == "ready"
    assert filter_skills_by_refs([skill], [ref]) == [skill]


def test_project_agents_report_device_scoped_availability():
    owner, project_id = uuid4(), uuid4()
    agent = _agent(owner)
    project = Project(
        id=project_id,
        owner_id=owner,
        name="Project",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    snapshots = []

    def snapshot(user, device):
        assert user == str(owner)
        snapshots.append(device)
        return {
            "status": "ready",
            "device_id": device,
            "session_id": f"session-{device}",
            "tool_catalog_version": 1,
            "skill_catalog_version": 1,
        }

    custom = CustomAgentService(
        repository=None,
        conversation_validation_utils=None,
        model_config_service=None,
        get_device_snapshot=snapshot,
        list_client_tool_refs=lambda *_: [],
        list_skill_refs=lambda _user, device: (
            [{"source": "client", "lookup_name": "calendar", "name": "calendar"}]
            if device == "a"
            else []
        ),
    )
    projects = ProjectService(
        repository=SimpleNamespace(
            get_live=lambda _: project,
            conversation_counts=lambda _: {},
            list_agents=lambda *_: [agent],
        ),
        custom_agent_repository=None,
        conversation_validation_utils=None,
        custom_agent_service=custom,
    )
    ready = projects.get_project(owner, project_id, include_agents=True, device_id="a")
    missing = projects.get_project(owner, project_id, include_agents=True, device_id="b")
    assert ready.custom_agents[0].availability.status == "ready"
    assert missing.custom_agents[0].availability.status == "degraded"
    assert missing.custom_agents[0].availability.missing_skills[0].name == "calendar"
    assert ready.custom_agents[0].id == missing.custom_agents[0].id == agent.id
    assert ready.custom_agents[0].skill_refs == missing.custom_agents[0].skill_refs
    assert snapshots == ["a", "a", "b", "b"]


@pytest.mark.parametrize("schema", [ProjectCreate, ProjectUpdate])
@pytest.mark.parametrize("name", ["   ", "\t\n"])
def test_project_name_rejects_whitespace(schema, name):
    with pytest.raises(ValidationError):
        schema(name=name)


@pytest.mark.parametrize("schema", [ProjectCreate, ProjectUpdate])
def test_project_name_is_trimmed(schema):
    assert schema(name="  Roadmap  ").name == "Roadmap"
