"""HITL approval flow for skill-capability mutations (Task 9).

Covers:
- the mutation gate: ``identity_requires_approval`` / ``any_call_requires_approval``
  auto-require approval for a ``mutation: True`` identity/tool unless an
  explicit policy entry already decided it (tool override still wins).
- redaction: ``redact_sensitive_args`` and its use inside
  ``_build_tool_interrupt_request`` so a sensitive-looking argument key never
  reaches an approval prompt.
- propagation: ``mutation`` flows end-to-end from the manifest capability
  through ``capability_catalog_entries`` -> ``_parse_tool_specs`` ->
  ``_build_tool``'s StructuredTool metadata.
- dispatch: ``RuntimeBridgeService._execute_tool_request`` builds the
  sidecar's ``SkillExecutionEngine`` with a mutation-allowing policy only
  when ``request.mutation_approved`` is True, and the existing
  session/catalog re-validation still rejects a stale/mismatched request
  regardless of that flag.
"""

from types import SimpleNamespace

import pytest

from app.ai.client_runtime_tools import _build_tool, _parse_tool_specs
from app.ai.hitl_config import (
    CallIdentity,
    _build_tool_interrupt_request,
    any_call_requires_approval,
    identity_requires_approval,
    redact_sensitive_args,
)
from app.ai.utils import apply_hitl_decisions
from client_backend.schemas.runtime import ToolDispatchRequest
from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService
from client_backend.services.skill_runtime.manager import SkillReadiness, SkillRuntimeManager
from client_backend.services.skill_runtime.permissions import SkillPermissionPolicy
from shared.skills.manifest import load_manifest


def _policy(master=True, servers=None, tools=None, global_tools=None):
    return {
        "master_enabled": master,
        "servers": servers or {},
        "tools": tools or {},
        "global_tools": global_tools or [],
    }


def _skill_manifest_dict(name: str = "demo", *, mutation: bool) -> dict:
    """A minimal, provider-neutral manifest dict with one capability."""
    return {
        "schema_version": "1.0",
        "name": name,
        "description": "A skill used to exercise the HITL mutation gate.",
        "runtime": {"type": "python_module", "module": "skills.demo.cli"},
        "dependencies": {"python": [], "node": [], "system": []},
        "secrets": [],
        "permissions": [],
        "capabilities": [
            {
                "name": "mutate",
                "description": "Mutates something.",
                "input_schema": {"type": "object", "properties": {}},
                "execution": {"argv": ["mutate"]},
                "permissions": [],
                "secrets": [],
                "mutation": mutation,
            }
        ],
    }


class _ServerClientStub:
    """Bare server-client stub: just enough for RuntimeBridgeService.__init__."""

    base_url = "http://server.test"

    def is_authenticated(self) -> bool:
        return True


def _make_bridge() -> RuntimeBridgeService:
    return RuntimeBridgeService(server_client=_ServerClientStub())


# ---------------------------------------------------------------------------
# Gate: mutation triggers approval
# ---------------------------------------------------------------------------


class TestMutationGate:
    def test_mutation_identity_requires_approval_by_default(self):
        identity = CallIdentity(
            name="client__skill_demo__mutate",
            server_name="skill_demo",
            qualified_tool_id="skill::demo::mutate",
            origin="client_skill",
            mutation=True,
        )
        assert identity_requires_approval(identity, _policy()) is True

    def test_non_mutation_identity_does_not_require_approval(self):
        identity = CallIdentity(
            name="client__skill_demo__list",
            server_name="skill_demo",
            qualified_tool_id="skill::demo::list",
            origin="client_skill",
            mutation=False,
        )
        assert identity_requires_approval(identity, _policy()) is False

    def test_policy_can_pre_approve_a_mutation_via_explicit_tool_override(self):
        identity = CallIdentity(
            name="client__skill_demo__mutate",
            server_name="skill_demo",
            qualified_tool_id="skill::demo::mutate",
            origin="client_skill",
            mutation=True,
        )
        policy = _policy(tools={"skill::demo::mutate": False})
        assert identity_requires_approval(identity, policy) is False

    def test_any_call_requires_approval_gates_a_mutation_tool_via_metadata(self):
        tool = SimpleNamespace(
            name="client__skill_demo__mutate",
            metadata={
                "server_name": "skill_demo",
                "qualified_tool_id": "skill::demo::mutate",
                "tool_origin": "client_skill",
                "mutation": True,
            },
        )
        calls = [{"name": tool.name, "args": {}, "id": "c1"}]
        assert (
            any_call_requires_approval(calls, policy=_policy(), tool_map={tool.name: tool})
            is True
        )

    def test_any_call_requires_approval_skips_a_non_mutation_tool(self):
        tool = SimpleNamespace(
            name="client__skill_demo__list",
            metadata={
                "server_name": "skill_demo",
                "qualified_tool_id": "skill::demo::list",
                "tool_origin": "client_skill",
                "mutation": False,
            },
        )
        calls = [{"name": tool.name, "args": {}, "id": "c1"}]
        assert (
            any_call_requires_approval(calls, policy=_policy(), tool_map={tool.name: tool})
            is False
        )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_redact_sensitive_args_masks_known_key_markers(self):
        redacted = redact_sensitive_args({"calendar_id": "primary", "api_token": "SECRET"})
        assert redacted["calendar_id"] == "primary"
        assert redacted["api_token"] == "<redacted>"

    def test_redact_sensitive_args_is_case_insensitive_and_substring_matched(self):
        redacted = redact_sensitive_args(
            {"Authorization": "Bearer xyz", "user_password": "hunter2", "safe_field": "ok"}
        )
        assert redacted["Authorization"] == "<redacted>"
        assert redacted["user_password"] == "<redacted>"
        assert redacted["safe_field"] == "ok"

    def test_redact_sensitive_args_non_dict_input_returns_empty_dict(self):
        assert redact_sensitive_args(None) == {}
        assert redact_sensitive_args("not-a-dict") == {}

    def test_build_tool_interrupt_request_redacts_sensitive_task_args(self):
        task = {
            "tool_call_id": "call-1",
            "action": "skill::demo::mutate",
            "args": {"calendar_id": "primary", "api_token": "SECRET"},
        }
        request = _build_tool_interrupt_request(task, 0, "task", {})
        assert request.args["calendar_id"] == "primary"
        assert request.args["api_token"] == "<redacted>"


# ---------------------------------------------------------------------------
# Propagation: mutation flag from manifest -> catalog -> spec -> tool metadata
# ---------------------------------------------------------------------------


class TestMutationPropagation:
    def test_capability_catalog_entries_include_mutation_true(self):
        manifest = load_manifest(_skill_manifest_dict("demo", mutation=True))
        manager = SkillRuntimeManager()
        readiness = SkillReadiness(status="ready")

        entries = manager.capability_catalog_entries("demo", manifest, readiness)

        assert len(entries) == 1
        assert entries[0]["mutation"] is True

    def test_capability_catalog_entries_include_mutation_false(self):
        manifest = load_manifest(_skill_manifest_dict("demo", mutation=False))
        manager = SkillRuntimeManager()
        readiness = SkillReadiness(status="ready")

        entries = manager.capability_catalog_entries("demo", manifest, readiness)

        assert entries[0]["mutation"] is False

    def test_mutation_declared_via_permissions_token_propagates(self):
        # A capability may declare mutation via the literal "mutation"
        # permission token instead of the flag; the catalog signal must still
        # be True (same definition the sidecar permission evaluator uses), so
        # such a capability is gated and can be approved rather than becoming
        # permanently unexecutable.
        data = _skill_manifest_dict("demo", mutation=False)
        data["capabilities"][0]["permissions"] = ["mutation"]
        manifest = load_manifest(data)
        readiness = SkillReadiness(status="ready")

        entries = SkillRuntimeManager().capability_catalog_entries("demo", manifest, readiness)

        assert entries[0]["mutation"] is True
        assert manifest.capabilities[0].is_mutation() is True

    def _catalog_entry(self, *, mutation: bool) -> dict:
        return {
            "name": "mutate",
            "description": "Mutates something.",
            "origin": "skill",
            "server_name": "skill_demo",
            "qualified_id": "skill::demo::mutate",
            "input_schema": {"type": "object", "properties": {}},
            "mutation": mutation,
        }

    def test_parse_tool_specs_carries_mutation_true(self):
        catalog = {"tools": [self._catalog_entry(mutation=True)]}
        specs = _parse_tool_specs(catalog)
        assert len(specs) == 1
        assert specs[0].mutation is True

    def test_parse_tool_specs_carries_mutation_false_by_default(self):
        catalog = {"tools": [self._catalog_entry(mutation=False)]}
        specs = _parse_tool_specs(catalog)
        assert specs[0].mutation is False

    def test_build_tool_metadata_carries_mutation_true(self):
        catalog = {"tools": [self._catalog_entry(mutation=True)]}
        spec = _parse_tool_specs(catalog)[0]

        tool = _build_tool(
            spec=spec,
            bound_user_id="user-1",
            bound_device_id="11111111-1111-1111-1111-111111111111",
            bound_session_id="session-1",
            bound_catalog_version=1,
        )

        assert tool.metadata["mutation"] is True

    def test_build_tool_metadata_carries_mutation_false(self):
        catalog = {"tools": [self._catalog_entry(mutation=False)]}
        spec = _parse_tool_specs(catalog)[0]

        tool = _build_tool(
            spec=spec,
            bound_user_id="user-1",
            bound_device_id="11111111-1111-1111-1111-111111111111",
            bound_session_id="session-1",
            bound_catalog_version=1,
        )

        assert tool.metadata["mutation"] is False


# ---------------------------------------------------------------------------
# Dispatch: sidecar only allows a mutation when the server marks it approved
# ---------------------------------------------------------------------------


class _RecordingSkillExecutionEngine:
    """Stub replacing SkillExecutionEngine: records the policy it was built with.

    Mirrors the real engine's own default (``SkillPermissionPolicy()``, i.e.
    ``allow_mutation=False``) when constructed with no policy, so assertions
    on ``allow_mutation`` reflect what runtime_bridge actually decided to
    pass, not an artifact of the stub.
    """

    last_instances: list["_RecordingSkillExecutionEngine"] = []

    def __init__(self, *, permission_policy: SkillPermissionPolicy | None = None, **_kwargs):
        self.permission_policy = (
            permission_policy if permission_policy is not None else SkillPermissionPolicy()
        )
        _RecordingSkillExecutionEngine.last_instances.append(self)

    async def execute(self, qualified_tool_id, arguments, context):
        return {"ok": True, "result": {}}


@pytest.fixture(autouse=True)
def _reset_recording_engine():
    _RecordingSkillExecutionEngine.last_instances = []
    yield
    _RecordingSkillExecutionEngine.last_instances = []


class TestDispatchApprovalSignal:
    @pytest.mark.asyncio
    async def test_mutation_approved_true_builds_engine_with_mutation_allowed(self, monkeypatch):
        monkeypatch.setattr(
            runtime_bridge_module, "SkillExecutionEngine", _RecordingSkillExecutionEngine
        )
        bridge = _make_bridge()

        result = await bridge._execute_tool_request(
            ToolDispatchRequest(
                request_id="r1",
                tool_name="mutate",
                qualified_tool_id="skill::demo::mutate",
                arguments={},
                timeout_seconds=10,
                mutation_approved=True,
            )
        )

        assert result == {}
        assert len(_RecordingSkillExecutionEngine.last_instances) == 1
        policy = _RecordingSkillExecutionEngine.last_instances[0].permission_policy
        assert policy.allow_mutation is True

    @pytest.mark.asyncio
    async def test_mutation_approved_false_builds_engine_with_mutation_blocked(self, monkeypatch):
        monkeypatch.setattr(
            runtime_bridge_module, "SkillExecutionEngine", _RecordingSkillExecutionEngine
        )
        bridge = _make_bridge()

        await bridge._execute_tool_request(
            ToolDispatchRequest(
                request_id="r2",
                tool_name="mutate",
                qualified_tool_id="skill::demo::mutate",
                arguments={},
                timeout_seconds=10,
            )
        )

        assert len(_RecordingSkillExecutionEngine.last_instances) == 1
        policy = _RecordingSkillExecutionEngine.last_instances[0].permission_policy
        assert policy.allow_mutation is False


class TestApprovalDoesNotBypassRevalidation:
    @pytest.mark.asyncio
    async def test_stale_session_rejected_even_when_mutation_approved(self, monkeypatch):
        monkeypatch.setattr(
            runtime_bridge_module, "SkillExecutionEngine", _RecordingSkillExecutionEngine
        )
        bridge = _make_bridge()
        bridge._current_tool_catalog = {
            "skill::demo::mutate": {"qualified_id": "skill::demo::mutate", "name": "mutate"}
        }
        bridge._session_id = "current"

        sent_payloads: list = []

        async def _capture(payload) -> None:
            sent_payloads.append(payload)

        monkeypatch.setattr(bridge, "_send_runtime_message", _capture)

        await bridge._handle_tool_request(
            ToolDispatchRequest(
                request_id="r3",
                tool_name="mutate",
                qualified_tool_id="skill::demo::mutate",
                arguments={},
                timeout_seconds=10,
                expected_session_id="OLD",
                mutation_approved=True,
            )
        )

        assert _RecordingSkillExecutionEngine.last_instances == []
        assert sent_payloads[0].success is False

    @pytest.mark.asyncio
    async def test_catalog_version_mismatch_rejected_even_when_mutation_approved(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            runtime_bridge_module, "SkillExecutionEngine", _RecordingSkillExecutionEngine
        )
        bridge = _make_bridge()
        bridge._current_tool_catalog = {
            "skill::demo::mutate": {"qualified_id": "skill::demo::mutate", "name": "mutate"}
        }
        bridge._tool_catalog_version = 3

        sent_payloads: list = []

        async def _capture(payload) -> None:
            sent_payloads.append(payload)

        monkeypatch.setattr(bridge, "_send_runtime_message", _capture)

        await bridge._handle_tool_request(
            ToolDispatchRequest(
                request_id="r4",
                tool_name="mutate",
                qualified_tool_id="skill::demo::mutate",
                arguments={},
                timeout_seconds=10,
                expected_catalog_version=2,
                mutation_approved=True,
            )
        )

        assert _RecordingSkillExecutionEngine.last_instances == []
        assert sent_payloads[0].success is False

    @pytest.mark.asyncio
    async def test_unknown_capability_rejected_even_when_mutation_approved(self, monkeypatch):
        monkeypatch.setattr(
            runtime_bridge_module, "SkillExecutionEngine", _RecordingSkillExecutionEngine
        )
        bridge = _make_bridge()
        bridge._current_tool_catalog = {}

        sent_payloads: list = []

        async def _capture(payload) -> None:
            sent_payloads.append(payload)

        monkeypatch.setattr(bridge, "_send_runtime_message", _capture)

        await bridge._handle_tool_request(
            ToolDispatchRequest(
                request_id="r5",
                tool_name="mutate",
                qualified_tool_id="skill::demo::mutate",
                arguments={},
                timeout_seconds=10,
                mutation_approved=True,
            )
        )

        assert _RecordingSkillExecutionEngine.last_instances == []
        assert sent_payloads[0].success is False


class TestDeniedMutationApproval:
    def test_rejected_mutation_is_not_in_the_approved_set(self):
        # A human REJECTing a mutation tool call must keep it out of the
        # approved (dispatchable) set entirely — it never reaches dispatch, so
        # mutation_approved=True is never sent for it.
        tool_calls = [
            {
                "id": "tc-1",
                "name": "client__skill_demo__mutate",
                "args": {"target": "x"},
            }
        ]

        approved, rejected_feedback = apply_hitl_decisions(
            tool_calls, {"type": "reject", "tool_call_id": "tc-1"}
        )

        assert approved == []
        assert "tc-1" in rejected_feedback

    def test_approved_mutation_stays_in_the_approved_set(self):
        tool_calls = [
            {"id": "tc-1", "name": "client__skill_demo__mutate", "args": {"target": "x"}}
        ]

        approved, rejected_feedback = apply_hitl_decisions(
            tool_calls, {"type": "approve", "tool_call_id": "tc-1"}
        )

        assert [tc["id"] for tc in approved] == ["tc-1"]
        assert rejected_feedback == {}
