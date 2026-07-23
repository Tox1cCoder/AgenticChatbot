"""Service- and schema-level tests for custom agents.

Schema tests are pure. Service tests run against the local Postgres database
(custom_agents uses JSONB + a partial unique index, which SQLite cannot model),
with fakes injected for the model/tool/skill validation collaborators. Every
test cleans up the rows it creates for its throwaway users.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import delete

from app.core.config import settings
from app.core.exceptions import (
    CustomAgentForbiddenError,
    CustomAgentInUseError,
    CustomAgentNotFoundError,
    CustomAgentValidationError,
)
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.user import User
from app.repositories.custom_agent import CustomAgentRepository
from app.schemas.custom_agent import (
    ConversationCustomAgentsUpdate,
    CustomAgentAvailability,
    CustomAgentCreate,
    CustomAgentOptions,
    CustomAgentUpdate,
    DeviceCatalogSnapshot,
)
from app.services.custom_agent_service import CustomAgentService
from app.utils.validation.conversation_validation import ConversationValidationUtils

# --------------------------------------------------------------------------- #
# Schema tests (pure, no DB)
# --------------------------------------------------------------------------- #


def test_custom_agent_schema_rejects_invalid_tool_refs():
    base = dict(
        name="Data Analyst",
        prompt="You are a precise data analyst.",
        provider_type="openai",
        model="gpt-4.1-mini",
    )
    with pytest.raises(ValidationError):
        CustomAgentCreate(
            **base,
            tool_refs=[{"type": "bogus", "qualified_tool_id": "calculator::calculate"}],
        )
    with pytest.raises(ValidationError):
        CustomAgentCreate(
            **base,
            tool_refs=[{"type": "client", "qualified_tool_id": "client__csv__profile"}],
        )
    ok = CustomAgentCreate(
        **base,
        tool_refs=[
            {
                "type": "server_mcp",
                "server_name": "calculator",
                "tool_name": "calculate",
                "qualified_tool_id": "calculator::calculate",
            },
            {
                "type": "client",
                "device_id": "desktop-1",
                "session_id": "session-1",
                "catalog_version": "v1",
                "tool_instance_id": "csv-profile-instance",
                "server_name": "csv",
                "qualified_tool_id": "client__csv__profile",
                "tool_name": "profile",
            },
        ],
    )
    assert len(ok.tool_refs) == 2


def test_custom_agent_schema_accepts_server_mcp_tool_refs():
    agent = CustomAgentCreate(
        name="Calculator",
        prompt="Use selected tools when arithmetic is required.",
        provider_type="openai",
        model="gpt-4.1-mini",
        tool_refs=[
            {
                "type": "server_mcp",
                "server_name": "calculator",
                "tool_name": "calculate",
                "qualified_tool_id": "calculator::calculate",
            }
        ],
    )

    assert agent.tool_refs[0].model_dump()["qualified_tool_id"] == "calculator::calculate"


def test_custom_agent_schema_normalizes_numeric_client_catalog_version():
    agent = CustomAgentCreate(
        name="Desktop",
        prompt="Use selected desktop tools.",
        provider_type="openai",
        model="gpt-4.1-mini",
        tool_refs=[
            {
                "type": "client",
                "device_id": "desktop-1",
                "session_id": "session-1",
                "catalog_version": 3,
                "tool_instance_id": "csv-profile-instance",
                "server_name": "csv",
                "qualified_tool_id": "client__csv__profile",
                "tool_name": "profile",
            }
        ],
    )

    assert agent.tool_refs[0].model_dump()["catalog_version"] == "3"


def test_custom_agent_schema_rejects_empty_name_and_prompt():
    with pytest.raises(ValidationError):
        CustomAgentCreate(
            name="   ", prompt="valid prompt", provider_type="openai", model="gpt-4.1-mini"
        )
    with pytest.raises(ValidationError):
        CustomAgentCreate(name="Valid", prompt="   ", provider_type="openai", model="gpt-4.1-mini")


def test_attachment_schema_rejects_duplicate_ids():
    dup = uuid4()
    with pytest.raises(ValidationError):
        ConversationCustomAgentsUpdate(custom_agent_ids=[dup, dup])


def test_custom_agent_availability_serializes_camel_case():
    value = CustomAgentAvailability(
        status="degraded",
        device_id="device-b",
        session_id="session-b",
        missing_tools=[
            {
                "server_name": "desktop-commander",
                "qualified_tool_id": "desktop-commander::read_file",
                "tool_name": "read_file",
            }
        ],
        missing_skills=[{"lookup_name": "kobo-library", "name": "kobo-library"}],
        warnings=["missing"],
    )

    payload = value.model_dump(mode="json", by_alias=True)

    assert payload["deviceId"] == "device-b"
    assert payload["missingTools"][0]["qualifiedToolId"] == ("desktop-commander::read_file")
    assert payload["missingSkills"][0]["lookupName"] == "kobo-library"


def test_custom_agent_options_requires_explicit_device_snapshot():
    options = CustomAgentOptions(device_snapshot=DeviceCatalogSnapshot(status="unavailable"))

    payload = options.model_dump(mode="json", by_alias=True)

    assert payload["deviceSnapshot"] == {
        "deviceId": None,
        "sessionId": None,
        "toolCatalogVersion": None,
        "skillCatalogVersion": None,
        "status": "unavailable",
    }


# --------------------------------------------------------------------------- #
# Service tests (real DB)
# --------------------------------------------------------------------------- #


class _FakeModelConfig:
    VALID = {("openai", "gpt-4.1-mini"), ("gemini", "gemini-3-flash-preview")}

    def validate_provider_model(self, user_id, provider_type, model, *, allow_custom_model=False):
        if (provider_type, model) not in self.VALID:
            raise ValueError(f"invalid model {provider_type}/{model}")


def _fake_client_tools(user_id, device_id):
    if device_id != "desktop-1":
        return []
    return [
        {
            "type": "client",
            "device_id": "desktop-1",
            "session_id": "session-1",
            "catalog_version": "v1",
            "tool_instance_id": "csv-profile-instance",
            "server_name": "csv",
            "qualified_tool_id": "csv::profile",
            "tool_name": "profile",
        }
    ]


def _fake_skills(user_id, device_id):
    if device_id != "desktop-1":
        return []
    return [{"source": "client", "lookup_name": "data-analysis", "name": "data-analysis"}]


def _fake_device_snapshot(user_id, device_id):
    if device_id == "syncing-desktop":
        return {
            "device_id": device_id,
            "session_id": "syncing-session",
            "tool_catalog_version": 0,
            "skill_catalog_version": 0,
            "status": "unavailable",
        }
    if device_id != "desktop-1":
        return {
            "device_id": device_id,
            "session_id": None,
            "tool_catalog_version": None,
            "skill_catalog_version": None,
            "status": "unavailable",
        }
    return {
        "device_id": "desktop-1",
        "session_id": "session-1",
        "tool_catalog_version": 1,
        "skill_catalog_version": 2,
        "status": "ready",
    }


def _fake_server_tools():
    return [
        {
            "type": "server_mcp",
            "server_name": "calculator",
            "tool_name": "calculate",
            "qualified_tool_id": "calculator::calculate",
            "description": "Evaluate arithmetic expressions.",
        },
        {
            "type": "server_mcp",
            "server_name": "spreadsheet",
            "tool_name": "read_sheet",
            "qualified_tool_id": "spreadsheet::read_sheet",
            "description": "Read spreadsheet data.",
        },
    ]


class _Env:
    def __init__(self):
        self.db = Database(settings.database_url)
        self.session_factory = self.db.session
        self.owner_id = self._make_user()
        self.other_owner_id = self._make_user()
        self.conversation_id = self._make_conversation(self.owner_id)
        repo = CustomAgentRepository(self.session_factory)
        conv_validation = ConversationValidationUtils(self.session_factory)
        self.service = CustomAgentService(
            repo,
            conv_validation,
            _FakeModelConfig(),
            list_client_tool_refs=_fake_client_tools,
            list_server_tool_refs=_fake_server_tools,
            list_skill_refs=_fake_skills,
            get_device_snapshot=_fake_device_snapshot,
        )

    def _make_user(self):
        uid = uuid4()
        with self.session_factory() as s:
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
            s.commit()
        return uid

    def _make_conversation(self, owner_id):
        cid = uuid4()
        with self.session_factory() as s:
            s.add(Conversation(id=cid, owner_id=owner_id, title="t"))
            s.commit()
        return cid

    def cleanup(self):
        with self.session_factory() as s:
            for owner in (self.owner_id, self.other_owner_id):
                s.execute(
                    delete(ConversationCustomAgent).where(ConversationCustomAgent.owner_id == owner)
                )
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == owner))
                s.execute(delete(Conversation).where(Conversation.owner_id == owner))
                s.execute(delete(User).where(User.id == owner))
            s.commit()


@pytest.fixture
def env():
    e = _Env()
    try:
        yield e
    finally:
        e.cleanup()


def _payload(name="Data Analyst", **overrides):
    base = dict(
        name=name,
        prompt="You are a precise data analyst.",
        provider_type="openai",
        model="gpt-4.1-mini",
    )
    base.update(overrides)
    return CustomAgentCreate(**base)


def test_create_list_get_custom_agent(env):
    created = env.service.create_agent(env.owner_id, _payload())
    assert created.slug == "data-analyst"
    assert created.runtime_agent_id == f"custom_agent:{created.id}"

    listed = env.service.list_agents(env.owner_id)
    assert [a.id for a in listed] == [created.id]

    fetched = env.service.get_agent(env.owner_id, created.id)
    assert fetched.name == "Data Analyst"


def test_ownership_isolation(env):
    created = env.service.create_agent(env.owner_id, _payload())
    assert env.service.list_agents(env.other_owner_id) == []
    with pytest.raises(CustomAgentForbiddenError):
        env.service.get_agent(env.other_owner_id, created.id)


def test_get_missing_raises_not_found(env):
    with pytest.raises(CustomAgentNotFoundError):
        env.service.get_agent(env.owner_id, uuid4())


def test_update_applies_live_config(env):
    created = env.service.create_agent(env.owner_id, _payload())
    updated = env.service.update_agent(
        env.owner_id,
        created.id,
        CustomAgentUpdate(prompt="Updated prompt", model="gpt-4.1-mini"),
    )
    assert updated.prompt == "Updated prompt"
    # Re-fetch confirms persistence.
    assert env.service.get_agent(env.owner_id, created.id).prompt == "Updated prompt"


def test_update_rejects_invalid_model(env):
    created = env.service.create_agent(env.owner_id, _payload())
    with pytest.raises(CustomAgentValidationError):
        env.service.update_agent(
            env.owner_id, created.id, CustomAgentUpdate(model="nonexistent-model")
        )


def test_duplicate_name_rejected_then_freed_by_soft_delete(env):
    first = env.service.create_agent(env.owner_id, _payload(name="Analyst"))
    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(env.owner_id, _payload(name="Analyst"))
    # Soft-deleting frees the slug (partial unique index excludes deleted rows).
    env.service.delete_agent(env.owner_id, first.id)
    recreated = env.service.create_agent(env.owner_id, _payload(name="Analyst"))
    assert recreated.slug == "analyst"


def test_create_rejects_invalid_model(env):
    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(env.owner_id, _payload(provider_type="openai", model="bad"))


def test_create_validates_tool_and_skill_refs(env):
    server_mcp_tool = {
        "type": "server_mcp",
        "server_name": "calculator",
        "tool_name": "calculate",
        "qualified_tool_id": "calculator::calculate",
    }
    client_tool = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "session-1",
        "catalog_version": "v1",
        "tool_instance_id": "csv-profile-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }
    skill = {"source": "client", "lookup_name": "data-analysis", "name": "data-analysis"}

    ok = env.service.create_agent(
        env.owner_id,
        _payload(tool_refs=[server_mcp_tool, client_tool], skill_refs=[skill]),
        device_id="desktop-1",
    )
    assert len(ok.tool_refs) == 2

    # Unknown backend MCP tool.
    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(
            env.owner_id,
            _payload(
                name="Other",
                tool_refs=[
                    {
                        "type": "server_mcp",
                        "server_name": "fs",
                        "tool_name": "delete",
                        "qualified_tool_id": "fs::delete",
                    }
                ],
            ),
        )
    # Client tool from a device with no active catalog.
    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(
            env.owner_id,
            _payload(name="Other2", tool_refs=[client_tool]),
            device_id="other-device",
        )
    # Unknown skill.
    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(
            env.owner_id,
            _payload(
                name="Other3",
                skill_refs=[{"source": "server", "lookup_name": "unknown", "name": "unknown"}],
            ),
        )


def test_update_validates_and_persists_tool_and_skill_refs(env):
    created = env.service.create_agent(env.owner_id, _payload())
    server_mcp_tool = {
        "type": "server_mcp",
        "server_name": "calculator",
        "tool_name": "calculate",
        "qualified_tool_id": "calculator::calculate",
    }
    client_tool = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "session-1",
        "catalog_version": "v1",
        "tool_instance_id": "csv-profile-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }
    skill = {"source": "client", "lookup_name": "data-analysis", "name": "data-analysis"}

    updated = env.service.update_agent(
        env.owner_id,
        created.id,
        CustomAgentUpdate(
            tool_refs=[server_mcp_tool, client_tool],
            skill_refs=[skill],
        ),
        device_id="desktop-1",
    )

    assert [ref["qualified_tool_id"] for ref in updated.tool_refs] == [
        "calculator::calculate",
        "csv::profile",
    ]
    assert [(ref["source"], ref["lookup_name"], ref["name"]) for ref in updated.skill_refs] == [
        ("client", "data-analysis", "data-analysis")
    ]

    cleared = env.service.update_agent(
        env.owner_id,
        created.id,
        CustomAgentUpdate(tool_refs=[], skill_refs=[]),
        device_id="desktop-1",
    )

    assert cleared.tool_refs == []
    assert cleared.skill_refs == []


def test_contextual_read_reports_missing_local_capabilities(env):
    selected = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "session-1",
        "catalog_version": "v1",
        "tool_instance_id": "csv-profile-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }
    created = env.service.create_agent(
        env.owner_id,
        _payload(tool_refs=[selected]),
        device_id="desktop-1",
    )
    env.service._list_client_tool_refs = lambda _user_id, _device_id: []

    read = env.service.get_agent(env.owner_id, created.id, device_id="desktop-1")

    assert read.availability is not None
    assert read.availability.status == "degraded"
    assert read.availability.missing_tools[0].qualified_tool_id == ("csv::profile")


@pytest.mark.asyncio
async def test_options_echo_complete_device_snapshot(env):
    options = await env.service.get_options(env.owner_id, device_id="desktop-1")

    assert options.device_snapshot.status == "ready"
    assert options.device_snapshot.device_id == "desktop-1"
    assert options.device_snapshot.session_id == "session-1"
    assert options.device_snapshot.tool_catalog_version == 1
    assert options.device_snapshot.skill_catalog_version == 2


@pytest.mark.asyncio
async def test_options_never_label_unsynced_empty_catalogs_ready(env):
    options = await env.service.get_options(env.owner_id, device_id="syncing-desktop")

    assert options.device_snapshot.status == "unavailable"
    assert options.client_tools == []
    assert options.skills == []


def test_context_retries_when_snapshot_rotates_during_catalog_read(env):
    snapshots = iter(
        [
            {
                "device_id": "desktop-1",
                "session_id": "s1",
                "tool_catalog_version": 1,
                "skill_catalog_version": 1,
                "status": "ready",
            },
            {
                "device_id": "desktop-1",
                "session_id": "s2",
                "tool_catalog_version": 1,
                "skill_catalog_version": 1,
                "status": "ready",
            },
            {
                "device_id": "desktop-1",
                "session_id": "s2",
                "tool_catalog_version": 1,
                "skill_catalog_version": 1,
                "status": "ready",
            },
        ]
    )
    tool_reads = iter([[{"session_id": "s1"}], [{"session_id": "s2"}]])
    env.service._get_device_snapshot = lambda _u, _d: next(snapshots)
    env.service._list_client_tool_refs = lambda _u, _d: next(tool_reads)
    env.service._list_skill_refs = lambda _u, _d: []

    snapshot, tools, _skills = env.service._load_device_context(env.owner_id, "desktop-1")

    assert snapshot["session_id"] == "s2"
    assert tools == [{"session_id": "s2"}]


def test_update_preserves_existing_unavailable_ref_but_rejects_new_fabricated_ref(env):
    selected = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "session-1",
        "catalog_version": "v1",
        "tool_instance_id": "csv-profile-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }
    created = env.service.create_agent(
        env.owner_id,
        _payload(tool_refs=[selected]),
        device_id="desktop-1",
    )

    preserved = env.service.update_agent(
        env.owner_id,
        created.id,
        CustomAgentUpdate(prompt="Changed", tool_refs=[selected]),
        device_id="not-connected",
    )
    # Stored form carries the schema default display_metadata=None.
    assert preserved.tool_refs == [{**selected, "display_metadata": None}]

    fabricated = dict(selected, qualified_tool_id="csv::fabricated", tool_name="fabricated")
    with pytest.raises(CustomAgentValidationError):
        env.service.update_agent(
            env.owner_id,
            created.id,
            CustomAgentUpdate(tool_refs=[selected, fabricated]),
            device_id="not-connected",
        )


def test_new_ref_requires_full_current_binding_not_only_live_logical_key(env):
    forged = {
        "type": "client",
        "device_id": "desktop-1",
        "session_id": "forged-session",
        "catalog_version": "v1",
        "tool_instance_id": "forged-instance",
        "server_name": "csv",
        "qualified_tool_id": "csv::profile",
        "tool_name": "profile",
    }

    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(
            env.owner_id,
            _payload(name="Forged", tool_refs=[forged]),
            device_id="desktop-1",
        )


def test_client_ref_dedupe_uses_server_and_qualified_id_not_device_instance():
    a = {
        "type": "client",
        "server_name": "desktop-commander",
        "qualified_tool_id": "desktop-commander::read_file",
        "device_id": "device-a",
        "session_id": "session-a",
        "tool_instance_id": "instance-a",
    }
    b = dict(
        a,
        device_id="device-b",
        session_id="session-b",
        tool_instance_id="instance-b",
    )

    assert CustomAgentService._dedupe_tool_refs([a, b]) == [a]


def test_existing_skill_survives_absence_but_new_and_legacy_source_do_not(env):
    selected = {
        "source": "client",
        "lookup_name": "data-analysis",
        "name": "data-analysis",
    }
    created = env.service.create_agent(
        env.owner_id,
        _payload(name="Skill Agent", skill_refs=[selected]),
        device_id="desktop-1",
    )
    env.service._list_skill_refs = lambda _user_id, _device_id: []

    preserved = env.service.update_agent(
        env.owner_id,
        created.id,
        CustomAgentUpdate(prompt="Changed", skill_refs=[selected]),
        device_id="not-connected",
    )
    assert preserved.skill_refs == [{**selected, "display_metadata": None}]

    new_ref = {"source": "client", "lookup_name": "never-installed", "name": "never-installed"}
    with pytest.raises(CustomAgentValidationError):
        env.service.update_agent(
            env.owner_id,
            created.id,
            CustomAgentUpdate(skill_refs=[selected, new_ref]),
            device_id="not-connected",
        )

    env.service._list_skill_refs = _fake_skills
    legacy_new = dict(selected, source="server")
    with pytest.raises(CustomAgentValidationError):
        env.service.create_agent(
            env.owner_id,
            _payload(name="Legacy New", skill_refs=[legacy_new]),
            device_id="desktop-1",
        )


def test_skill_ref_dedupe_normalizes_legacy_server_source():
    legacy = {"source": "server", "lookup_name": "kobo-library", "name": "kobo-library"}
    current = dict(legacy, source="client")

    assert CustomAgentService._dedupe_skill_refs([legacy, current]) == [legacy]


def test_soft_delete_detaches_from_conversations(env):
    agent = env.service.create_agent(env.owner_id, _payload())
    env.service.set_conversation_agents(
        env.owner_id,
        env.conversation_id,
        ConversationCustomAgentsUpdate(custom_agent_ids=[agent.id]),
    )
    assert len(env.service.list_conversation_agents(env.owner_id, env.conversation_id)) == 1

    env.service.delete_agent(env.owner_id, agent.id)

    assert env.service.list_conversation_agents(env.owner_id, env.conversation_id) == []
    assert env.service.list_agents(env.owner_id) == []
    with pytest.raises(CustomAgentNotFoundError):
        env.service.get_agent(env.owner_id, agent.id)


def test_set_and_replace_conversation_agents_preserves_order(env):
    a = env.service.create_agent(env.owner_id, _payload(name="Alpha"))
    b = env.service.create_agent(env.owner_id, _payload(name="Beta"))

    attached = env.service.set_conversation_agents(
        env.owner_id,
        env.conversation_id,
        ConversationCustomAgentsUpdate(custom_agent_ids=[b.id, a.id]),
    )
    assert [x.id for x in attached] == [b.id, a.id]

    replaced = env.service.set_conversation_agents(
        env.owner_id,
        env.conversation_id,
        ConversationCustomAgentsUpdate(custom_agent_ids=[a.id]),
    )
    assert [x.id for x in replaced] == [a.id]


def test_attach_rejects_unowned_agent(env):
    other_agent = env.service.create_agent(env.other_owner_id, _payload(name="Foreign"))
    with pytest.raises(CustomAgentForbiddenError):
        env.service.set_conversation_agents(
            env.owner_id,
            env.conversation_id,
            ConversationCustomAgentsUpdate(custom_agent_ids=[other_agent.id]),
        )


def _service_with_registry(env, registry):
    return CustomAgentService(
        env.service.repository,
        env.service.conversation_validation_utils,
        _FakeModelConfig(),
        generation_registry=registry,
        list_client_tool_refs=_fake_client_tools,
        list_server_tool_refs=_fake_server_tools,
        list_skill_refs=_fake_skills,
    )


def test_update_blocks_when_custom_agent_active(env):
    from app.services.generation_registry import GenerationRegistry

    agent = env.service.create_agent(env.owner_id, _payload())
    registry = GenerationRegistry()
    registry.register(
        uuid4(),
        env.conversation_id,
        env.owner_id,
        selected_agent=f"custom_agent:{agent.id}",
    )
    locked = _service_with_registry(env, registry)

    with pytest.raises(CustomAgentInUseError):
        locked.update_agent(env.owner_id, agent.id, CustomAgentUpdate(prompt="x"))
    with pytest.raises(CustomAgentInUseError):
        locked.delete_agent(env.owner_id, agent.id)


def test_conservative_lock_blocks_detach_when_selected_agent_unknown(env):
    from app.services.generation_registry import GenerationRegistry

    agent = env.service.create_agent(env.owner_id, _payload())
    env.service.set_conversation_agents(
        env.owner_id,
        env.conversation_id,
        ConversationCustomAgentsUpdate(custom_agent_ids=[agent.id]),
    )
    registry = GenerationRegistry()
    # Active generation in the conversation, selected agent not yet resolved.
    registry.register(uuid4(), env.conversation_id, env.owner_id, selected_agent=None)
    locked = _service_with_registry(env, registry)

    # Detaching the agent (replacing attachments with empty) must be blocked.
    with pytest.raises(CustomAgentInUseError):
        locked.set_conversation_agents(
            env.owner_id,
            env.conversation_id,
            ConversationCustomAgentsUpdate(custom_agent_ids=[]),
        )
    # Edit/delete also blocked via the attached-conversation fallback.
    with pytest.raises(CustomAgentInUseError):
        locked.delete_agent(env.owner_id, agent.id)


def test_custom_agent_model_uses_provider_catalog_validation():
    """Custom provider/model validation reuses the base-agent catalog/credential checks."""
    from uuid import uuid4

    from app.services.model_config_service import (
        ModelConfigService,
        _normalize_runtime_agent_key,
    )

    class _ProviderSvc:
        def __init__(self, *, configured=True, models=None):
            self._configured = configured
            self._models = models or []

        def get_cached_provider_status(self, user_id, provider_type):
            return {"configured": self._configured, "models": self._models}

    # "custom" is now an accepted runtime key (but not a persisted agent key).
    assert _normalize_runtime_agent_key("custom") == "custom"

    svc = ModelConfigService(
        repository=None,
        provider_service=_ProviderSvc(models=[{"id": "gpt-4.1-mini"}]),
    )
    # Valid (provider configured + model in catalog).
    svc.validate_provider_model(uuid4(), "openai", "gpt-4.1-mini")
    # Model absent from the catalog.
    with pytest.raises(ValueError):
        svc.validate_provider_model(uuid4(), "openai", "gpt-9-ultra")
    # Unsupported provider.
    with pytest.raises(ValueError):
        svc.validate_provider_model(uuid4(), "anthropic", "claude")

    # Provider not configured (missing credentials).
    unconfigured = ModelConfigService(
        repository=None, provider_service=_ProviderSvc(configured=False)
    )
    with pytest.raises(ValueError):
        unconfigured.validate_provider_model(uuid4(), "openai", "gpt-4.1-mini")


def test_build_runtime_state(env):
    agent = env.service.create_agent(env.owner_id, _payload(temperature=0.2))
    env.service.set_conversation_agents(
        env.owner_id,
        env.conversation_id,
        ConversationCustomAgentsUpdate(custom_agent_ids=[agent.id]),
    )
    state = env.service.build_runtime_state(env.owner_id, env.conversation_id)
    runtime_id = f"custom_agent:{agent.id}"
    assert runtime_id in state
    entry = state[runtime_id]
    assert entry["model_agent_key"] == "custom"
    assert entry["model_request"]["provider_type"] == "openai"
    assert entry["model_request"]["temperature"] == 0.2
    assert entry["agent_order"] == 0


@pytest.mark.asyncio
async def test_options_include_all_backend_server_mcp_tools(env):
    options = await env.service.get_options(env.owner_id, device_id="desktop-1")

    qualified_ids = {t["qualified_tool_id"] for t in options.server_default_tools}
    assert qualified_ids == {"calculator::calculate", "spreadsheet::read_sheet"}
    assert all(t.get("type") == "server_mcp" for t in options.server_default_tools)
    assert options.server_tools == []


@pytest.mark.asyncio
async def test_options_group_client_tools_into_servers(env):
    """Sidecar MCP servers surface as server-level entries (parallel to the
    backend ``server_default_tools``) so a user can attach a whole sidecar MCP
    server, not just hunt through individual client tool rows."""
    options = await env.service.get_options(env.owner_id, device_id="desktop-1")

    assert options.client_servers == [
        {"server_name": "csv", "device_id": "desktop-1", "tool_count": 1}
    ]


@pytest.mark.asyncio
async def test_options_client_servers_empty_without_active_device(env):
    options = await env.service.get_options(env.owner_id, device_id="not-connected")
    assert options.client_servers == []


def test_group_client_servers_counts_and_scopes_by_device():
    """Grouping is keyed by (server_name, device_id): tools collapse into one
    entry per server per device, counted, with server_name-less rows dropped."""
    tools = [
        {"server_name": "csv", "device_id": "d1", "qualified_tool_id": "client__csv__a"},
        {"server_name": "csv", "device_id": "d1", "qualified_tool_id": "client__csv__b"},
        {"server_name": "fs", "device_id": "d1", "qualified_tool_id": "client__fs__x"},
        {"server_name": "csv", "device_id": "d2", "qualified_tool_id": "client__csv__a"},
        {"device_id": "d1", "qualified_tool_id": "no_server_name"},
    ]

    assert CustomAgentService._group_client_servers(tools) == [
        {"server_name": "csv", "device_id": "d1", "tool_count": 2},
        {"server_name": "csv", "device_id": "d2", "tool_count": 1},
        {"server_name": "fs", "device_id": "d1", "tool_count": 1},
    ]


def test_dedupe_tool_refs_collapses_group_and_individual_duplicates():
    """A tool chosen both via its whole-server group and individually persists
    once; client tools differing only by instance id stay distinct; order kept."""
    refs = [
        {"type": "server_mcp", "qualified_tool_id": "calc::add", "tool_name": "add"},
        {"type": "server_mcp", "qualified_tool_id": "calc::add", "tool_name": "add"},
        {"type": "server_mcp", "qualified_tool_id": "calc::sub", "tool_name": "sub"},
        {
            "type": "client",
            "qualified_tool_id": "csv::p",
            "device_id": "d1",
            "session_id": "s1",
            "tool_instance_id": "i1",
        },
        {
            "type": "client",
            "qualified_tool_id": "csv::p",
            "device_id": "d1",
            "session_id": "s1",
            "tool_instance_id": "i1",
        },
        {
            "type": "client",
            "qualified_tool_id": "csv::p",
            "device_id": "d1",
            "session_id": "s1",
            "tool_instance_id": "i2",
        },
    ]

    result = CustomAgentService._dedupe_tool_refs(refs)

    assert [(r["type"], r.get("qualified_tool_id"), r.get("tool_instance_id")) for r in result] == [
        ("server_mcp", "calc::add", None),
        ("server_mcp", "calc::sub", None),
        ("client", "csv::p", "i1"),
        ("client", "csv::p", "i2"),
    ]
