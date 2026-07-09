"""Tests for client_backend.services.skill_runtime.permissions."""

import json

from client_backend.services.skill_runtime.permissions import (
    PermissionDecision,
    SkillPermissionEvaluator,
    SkillPermissionPolicy,
    redact_arguments,
)
from shared.skills.errors import PERMISSION_DENIED, PERMISSION_REQUIRED
from shared.skills.manifest import SkillCapabilitySpec, SkillRuntimeSpec


def _capability(
    *,
    permissions: list[str] | None = None,
    mutation: bool = False,
) -> SkillCapabilitySpec:
    """Build a minimal, provider-neutral capability for permission tests."""
    return SkillCapabilitySpec(
        name="do_thing",
        description="Does a thing.",
        input_schema={"type": "object", "properties": {}},
        execution={"argv": []},
        permissions=permissions or [],
        secrets=[],
        mutation=mutation,
    )


def _runtime(runtime_type: str = "python_module") -> SkillRuntimeSpec:
    return SkillRuntimeSpec(type=runtime_type, module="skills.example.cli")


class TestAllowedReadCapability:
    def test_read_permission_granted_by_exact_prefix_is_allowed(self):
        capability = _capability(permissions=["filesystem:read:/data"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:/data"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True
        assert decision.code is None
        assert decision.missing == []

    def test_read_permission_granted_by_family_wildcard_is_allowed(self):
        capability = _capability(permissions=["filesystem:read:/data"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:*"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True
        assert decision.code is None


class TestPathPrefixBoundary:
    def test_sibling_path_with_shared_prefix_is_not_covered(self):
        # "filesystem:read:/data" must NOT cover "/database" -- that would be
        # a naive string-prefix match without a "/" boundary check.
        capability = _capability(permissions=["filesystem:read:/database"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:/data"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is False
        assert decision.code == PERMISSION_REQUIRED
        assert "filesystem:read:/database" in decision.missing

    def test_nested_subpath_is_covered(self):
        capability = _capability(permissions=["filesystem:read:/data/sub"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:/data"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True
        assert decision.code is None

    def test_adjacent_prefix_with_hyphen_is_not_covered(self):
        # "/data" must not cover "/data-backup" (no "/" boundary between them).
        capability = _capability(permissions=["filesystem:read:/data-backup"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:/data"}))

        decision = SkillPermissionEvaluator(policy).evaluate(
            capability=capability, runtime=_runtime()
        )

        assert decision.allowed is False
        assert "filesystem:read:/data-backup" in decision.missing

    def test_empty_path_prefix_grant_does_not_open_whole_filesystem(self):
        # A blank/mis-serialized "filesystem:read:" grant must grant NOTHING --
        # NOT universal filesystem read. Use "filesystem:read:*" for that.
        capability = _capability(permissions=["filesystem:read:/etc/passwd"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:"}))

        decision = SkillPermissionEvaluator(policy).evaluate(
            capability=capability, runtime=_runtime()
        )

        assert decision.allowed is False
        assert decision.code == PERMISSION_REQUIRED

    def test_root_only_path_prefix_grant_does_not_open_whole_filesystem(self):
        capability = _capability(permissions=["filesystem:read:/etc/passwd"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:/"}))

        decision = SkillPermissionEvaluator(policy).evaluate(
            capability=capability, runtime=_runtime()
        )

        assert decision.allowed is False

    def test_family_wildcard_still_grants_everything(self):
        # The intended "grant everything" channel remains the explicit wildcard.
        capability = _capability(permissions=["filesystem:read:/etc/passwd"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:*"}))

        decision = SkillPermissionEvaluator(policy).evaluate(
            capability=capability, runtime=_runtime()
        )

        assert decision.allowed is True


class TestBlockedWriteCapability:
    def test_write_permission_not_covered_by_read_grant_is_blocked(self):
        capability = _capability(permissions=["filesystem:write:/data"])
        policy = SkillPermissionPolicy(granted=frozenset({"filesystem:read:/data"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is False
        assert decision.code == PERMISSION_REQUIRED
        assert "filesystem:write:/data" in decision.missing


class TestBlockedProcessSpawn:
    def test_process_spawn_without_grant_is_blocked(self):
        capability = _capability(permissions=["process:spawn"])
        evaluator = SkillPermissionEvaluator(SkillPermissionPolicy.deny_all())

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is False
        assert decision.code == PERMISSION_REQUIRED
        assert "process:spawn" in decision.missing

    def test_process_spawn_is_exact_only_no_wildcard_beyond_star(self):
        capability = _capability(permissions=["process:spawn"])
        policy = SkillPermissionPolicy(granted=frozenset({"process:*"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is False
        assert "process:spawn" in decision.missing

    def test_process_spawn_exact_grant_is_allowed(self):
        capability = _capability(permissions=["process:spawn"])
        policy = SkillPermissionPolicy(granted=frozenset({"process:spawn"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True


class TestBlockedMutationCapability:
    def test_mutation_without_allow_mutation_is_blocked_pending_approval(self):
        capability = _capability(mutation=True)
        policy = SkillPermissionPolicy(allow_mutation=False)
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is False
        assert decision.code == PERMISSION_REQUIRED
        assert decision.requires_approval is True
        assert "mutation" in decision.missing

    def test_mutation_with_allow_mutation_true_is_allowed(self):
        capability = _capability(mutation=True)
        policy = SkillPermissionPolicy(allow_mutation=True)
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True
        assert decision.requires_approval is False

    def test_mutation_with_allow_all_policy_is_allowed(self):
        capability = _capability(mutation=True)
        evaluator = SkillPermissionEvaluator(SkillPermissionPolicy.allow_all())

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True

    def test_non_mutating_capability_never_requires_approval(self):
        capability = _capability(mutation=False)
        evaluator = SkillPermissionEvaluator(SkillPermissionPolicy.deny_all())

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True
        assert decision.requires_approval is False

    def test_blanket_resource_wildcard_does_not_unlock_mutation(self):
        # granted={"*"} covers network/filesystem/process resources, but must
        # NOT silently allow a mutation without the explicit allow_mutation opt-in.
        capability = _capability(mutation=True)
        policy = SkillPermissionPolicy(granted=frozenset({"*"}), allow_mutation=False)

        decision = SkillPermissionEvaluator(policy).evaluate(
            capability=capability, runtime=_runtime()
        )

        assert decision.allowed is False
        assert decision.requires_approval is True
        assert "mutation" in decision.missing

    def test_literal_mutation_token_grant_does_not_unlock_mutation(self):
        # Putting the literal "mutation" token in the resource grant set must
        # NOT satisfy the structural mutation gate.
        capability = _capability(mutation=True)
        policy = SkillPermissionPolicy(granted=frozenset({"mutation"}), allow_mutation=False)

        decision = SkillPermissionEvaluator(policy).evaluate(
            capability=capability, runtime=_runtime()
        )

        assert decision.allowed is False
        assert decision.requires_approval is True


class TestShellRuntimeReserved:
    def test_shell_runtime_is_hard_denied_regardless_of_policy(self):
        capability = _capability()
        # "shell" is a reserved runtime type that load_manifest() would
        # reject at validation time; model_construct bypasses that
        # validation so the evaluator's own defensive guard can be tested.
        shell_runtime = SkillRuntimeSpec.model_construct(type="shell")
        evaluator = SkillPermissionEvaluator(SkillPermissionPolicy.allow_all())

        decision = evaluator.evaluate(capability=capability, runtime=shell_runtime)

        assert decision.allowed is False
        assert decision.code == PERMISSION_DENIED
        assert decision.requires_approval is False


class TestRedactArguments:
    def test_none_and_empty_dict_yield_empty_dict(self):
        assert redact_arguments(None) == {}
        assert redact_arguments({}) == {}

    def test_values_are_replaced_keys_are_kept(self):
        redacted = redact_arguments({"token": "SUPER_SECRET_VALUE", "path": "/x"})

        assert set(redacted.keys()) == {"token", "path"}
        assert "SUPER_SECRET_VALUE" not in redacted.values()
        assert "/x" not in redacted.values()
        # Every value is the explicit placeholder — not "" or the key, which
        # would still not leak but would carry wrong repair semantics.
        assert set(redacted.values()) == {"<redacted>"}


class TestRedactedPermissionErrorPayload:
    def test_error_payload_never_leaks_argument_values(self):
        decision = PermissionDecision(
            allowed=False,
            code=PERMISSION_REQUIRED,
            message="capability requires permissions that are not granted: process:spawn",
            missing=["process:spawn"],
            requires_approval=False,
        )

        payload = decision.to_error_payload({"token": "SUPER_SECRET_VALUE", "path": "/x"})

        serialized = json.dumps(payload)
        assert "SUPER_SECRET_VALUE" not in serialized
        assert "/x" not in serialized

        assert payload["code"] == PERMISSION_REQUIRED
        assert payload["message"] == decision.message
        assert payload["repair"]["missing"] == ["process:spawn"]
        assert payload["repair"]["requires_approval"] is False
        assert set(payload["repair"]["arguments_redacted"].keys()) == {"token", "path"}

    def test_error_payload_with_no_arguments_is_empty_redacted_dict(self):
        decision = PermissionDecision(
            allowed=False,
            code=PERMISSION_DENIED,
            message="shell runtime is reserved and cannot be executed",
        )

        payload = decision.to_error_payload(None)

        assert payload["repair"]["arguments_redacted"] == {}


class TestAllowAllPolicy:
    def test_mutation_write_and_process_capability_is_allowed(self):
        capability = _capability(
            permissions=["filesystem:write:/data", "process:spawn"],
            mutation=True,
        )
        evaluator = SkillPermissionEvaluator(SkillPermissionPolicy.allow_all())

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True
        assert decision.code is None
        assert decision.missing == []


class TestNetworkWildcard:
    def test_network_host_allowed_under_network_wildcard(self):
        capability = _capability(permissions=["network:api.example.com"])
        policy = SkillPermissionPolicy(granted=frozenset({"network:*"}))
        evaluator = SkillPermissionEvaluator(policy)

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is True

    def test_network_host_blocked_under_deny_all(self):
        capability = _capability(permissions=["network:api.example.com"])
        evaluator = SkillPermissionEvaluator(SkillPermissionPolicy.deny_all())

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is False
        assert decision.code == PERMISSION_REQUIRED
        assert "network:api.example.com" in decision.missing


class TestDefaultPolicyIsDenyAll:
    def test_evaluator_with_no_policy_defaults_to_deny_all(self):
        capability = _capability(permissions=["process:spawn"])
        evaluator = SkillPermissionEvaluator()

        decision = evaluator.evaluate(capability=capability, runtime=_runtime())

        assert decision.allowed is False
        assert decision.code == PERMISSION_REQUIRED
