from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.ai.tool_execution_policy import (
    AmbiguousToolExecutionPolicyError,
    ToolExecutionPolicyValidationError,
    ToolIdentity,
    get_current_tool_policy,
    matching_policy_rules,
    policy_match_specificity,
    resolve_tool_execution_policy,
    resolve_tool_identity,
    tool_policy_context,
)
from app.core.config import (
    Settings,
    ToolExecutionPolicyMatch,
    ToolExecutionPolicyOverride,
    settings,
)


def _settings(**overrides) -> Settings:
    values = {
        "secret_key": "test-secret-key-with-at-least-32-bytes",
        "environment": "development",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _identity(**overrides) -> ToolIdentity:
    values = {
        "tool_origin": "client_mcp",
        "qualified_tool_id": "desktop_commander::start_process",
        "exposed_tool_name": "client__desktop_commander__start_process",
        "source_tool_name": "start_process",
        "server_name": "desktop_commander",
    }
    values.update(overrides)
    return ToolIdentity(**values)


# --- ToolIdentity ------------------------------------------------------------


def test_exact_policy_identity_includes_origin():
    server = ToolIdentity(
        tool_origin="server_mcp",
        qualified_tool_id="desktop_commander::start_process",
        exposed_tool_name="start_process",
        source_tool_name="start_process",
        server_name="desktop_commander",
    )
    client = ToolIdentity(
        tool_origin="client_mcp",
        qualified_tool_id="desktop_commander::start_process",
        exposed_tool_name="client__desktop_commander__start_process",
        source_tool_name="start_process",
        server_name="desktop_commander",
    )
    assert (server.tool_origin, server.qualified_tool_id) != (
        client.tool_origin,
        client.qualified_tool_id,
    )


# --- resolve_tool_identity -----------------------------------------------------


def test_resolve_tool_identity_defaults_untagged_tool_to_internal():
    tool = SimpleNamespace(name="write_todos", metadata={})

    identity = resolve_tool_identity(tool, exposed_tool_name="write_todos")

    assert identity.tool_origin == "internal"
    assert identity.qualified_tool_id == "internal::write_todos"
    assert identity.source_tool_name == "write_todos"
    assert identity.exposed_tool_name == "write_todos"
    assert identity.server_name is None


def test_resolve_tool_identity_missing_metadata_attribute_falls_back_to_internal():
    tool = SimpleNamespace(name="hand_off")

    identity = resolve_tool_identity(tool, exposed_tool_name="hand_off")

    assert identity.tool_origin == "internal"
    assert identity.qualified_tool_id == "internal::hand_off"
    assert identity.server_name is None


def test_resolve_tool_identity_reads_server_mcp_metadata():
    tool = SimpleNamespace(
        name="start_process",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
        },
    )

    identity = resolve_tool_identity(tool, exposed_tool_name="start_process")

    assert identity.tool_origin == "server_mcp"
    assert identity.qualified_tool_id == "desktop_commander::start_process"
    assert identity.source_tool_name == "start_process"
    assert identity.server_name == "desktop_commander"


def test_resolve_tool_identity_reads_client_runtime_metadata():
    tool = SimpleNamespace(
        name="client__desktop_commander__start_process",
        metadata={
            "tool_origin": "client_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
            "source_tool_name": "start_process",
        },
    )

    identity = resolve_tool_identity(
        tool, exposed_tool_name="client__desktop_commander__start_process"
    )

    assert identity.tool_origin == "client_mcp"
    assert identity.source_tool_name == "start_process"
    assert identity.exposed_tool_name == "client__desktop_commander__start_process"
    assert identity.qualified_tool_id == "desktop_commander::start_process"


def test_resolve_tool_identity_alias_only_changes_exposed_name():
    tool = SimpleNamespace(
        name="start_process",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
        },
    )

    identity = resolve_tool_identity(tool, exposed_tool_name="aliased_start_process")

    assert identity.exposed_tool_name == "aliased_start_process"
    assert identity.source_tool_name == "start_process"
    assert identity.qualified_tool_id == "desktop_commander::start_process"


# --- ToolExecutionPolicyMatch shape validation ---------------------------------


def test_match_accepts_origin_only():
    ToolExecutionPolicyMatch(tool_origin="client_mcp")


def test_match_accepts_origin_and_exposed_name():
    ToolExecutionPolicyMatch(tool_origin="client_mcp", exposed_tool_name="start_process")


def test_match_accepts_origin_server_and_source():
    ToolExecutionPolicyMatch(
        tool_origin="client_mcp",
        server_name="desktop_commander",
        source_tool_name="start_process",
    )


def test_match_accepts_origin_and_qualified_id():
    ToolExecutionPolicyMatch(
        tool_origin="client_mcp",
        qualified_tool_id="desktop_commander::start_process",
    )


def test_match_rejects_missing_tool_origin():
    with pytest.raises(ValidationError):
        ToolExecutionPolicyMatch(qualified_tool_id="desktop_commander::start_process")


def test_match_rejects_unknown_tool_origin():
    with pytest.raises(ValidationError):
        ToolExecutionPolicyMatch(tool_origin="not_a_real_origin")


@pytest.mark.parametrize(
    "partial_kwargs",
    [
        {"server_name": "desktop_commander"},
        {"source_tool_name": "start_process"},
    ],
)
def test_match_rejects_partial_server_source_pair(partial_kwargs):
    with pytest.raises(ValidationError):
        ToolExecutionPolicyMatch(tool_origin="client_mcp", **partial_kwargs)


@pytest.mark.parametrize(
    "mixed_kwargs",
    [
        {
            "exposed_tool_name": "start_process",
            "qualified_tool_id": "desktop_commander::start_process",
        },
        {
            "exposed_tool_name": "start_process",
            "server_name": "desktop_commander",
            "source_tool_name": "start_process",
        },
        {
            "qualified_tool_id": "desktop_commander::start_process",
            "server_name": "desktop_commander",
            "source_tool_name": "start_process",
        },
    ],
)
def test_match_rejects_mixed_shapes(mixed_kwargs):
    with pytest.raises(ValidationError):
        ToolExecutionPolicyMatch(tool_origin="client_mcp", **mixed_kwargs)


def test_match_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ToolExecutionPolicyMatch(tool_origin="client_mcp", unexpected_field="x")


# --- ToolExecutionPolicyOverride ------------------------------------------------


def test_bare_qualified_id_rule_is_rejected():
    with pytest.raises(ValidationError):
        ToolExecutionPolicyOverride(
            match={"qualified_tool_id": "desktop_commander::start_process"},
            timeout_seconds=45,
        )


def test_override_accepts_exact_qualified_id_rule():
    override = ToolExecutionPolicyOverride(
        match={
            "tool_origin": "client_mcp",
            "qualified_tool_id": "desktop_commander::start_process",
        },
        timeout_seconds=45,
        hard_timeout_seconds=47,
        total_timeout_seconds=47,
        max_attempts=1,
        retry_safe=False,
        idempotent=False,
        trust_mcp_metadata=False,
        timeout_hint="The process did not finish in the interactive budget.",
    )

    assert override.timeout_seconds == 45
    assert override.match.tool_origin == "client_mcp"


def test_trust_mcp_metadata_requires_exact_qualified_id_match():
    with pytest.raises(ValidationError):
        ToolExecutionPolicyOverride(
            match={"tool_origin": "client_mcp"},
            trust_mcp_metadata=True,
        )


def test_trust_mcp_metadata_allowed_on_exact_match():
    override = ToolExecutionPolicyOverride(
        match={
            "tool_origin": "client_mcp",
            "qualified_tool_id": "desktop_commander::start_process",
        },
        trust_mcp_metadata=True,
    )

    assert override.trust_mcp_metadata is True


@pytest.mark.parametrize(
    "field",
    ["timeout_seconds", "hard_timeout_seconds", "total_timeout_seconds", "max_timeout_seconds"],
)
def test_override_rejects_non_positive_timeouts(field):
    with pytest.raises(ValidationError):
        ToolExecutionPolicyOverride(match={"tool_origin": "internal"}, **{field: 0})


@pytest.mark.parametrize("value", [0, 6])
def test_override_rejects_max_attempts_outside_bounds(value):
    with pytest.raises(ValidationError):
        ToolExecutionPolicyOverride(match={"tool_origin": "internal"}, max_attempts=value)


def test_override_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ToolExecutionPolicyOverride(
            match={"tool_origin": "internal"},
            not_a_real_field=True,
        )


# --- Settings integration --------------------------------------------------------


def test_settings_tool_execution_policy_defaults():
    settings = _settings()

    assert settings.tool_execution_policies == {}
    assert settings.tool_execution_max_interactive_timeout_seconds == 120.0
    assert settings.tool_execution_cancellation_grace_seconds == 2.0
    assert settings.tool_execution_client_execution_grace_seconds == 2.0
    assert settings.tool_execution_client_response_grace_seconds == 1.0


def test_settings_accepts_configured_tool_execution_policies():
    settings = _settings(
        tool_execution_policies={
            "client-mcp-default-cap": {
                "match": {"tool_origin": "client_mcp"},
                "max_timeout_seconds": 60,
            },
            "desktop-start-process": {
                "match": {
                    "tool_origin": "client_mcp",
                    "qualified_tool_id": "desktop_commander::start_process",
                },
                "timeout_seconds": 45,
                "hard_timeout_seconds": 47,
                "total_timeout_seconds": 47,
                "max_attempts": 1,
                "retry_safe": False,
                "idempotent": False,
                "trust_mcp_metadata": False,
                "timeout_hint": (
                    "The process did not finish in the interactive budget. "
                    "Ask before repeating it."
                ),
            },
        }
    )

    assert set(settings.tool_execution_policies) == {
        "client-mcp-default-cap",
        "desktop-start-process",
    }
    assert settings.tool_execution_policies["client-mcp-default-cap"].max_timeout_seconds == 60
    assert settings.tool_execution_policies["desktop-start-process"].timeout_seconds == 45


def test_settings_rejects_invalid_tool_execution_policy_match_shape():
    with pytest.raises(ValidationError):
        _settings(
            tool_execution_policies={
                "bad-rule": {
                    "match": {"qualified_tool_id": "desktop_commander::start_process"},
                    "timeout_seconds": 45,
                },
            }
        )


# --- policy_match_specificity ----------------------------------------------------


@pytest.mark.parametrize(
    ("match_kwargs", "expected_specificity"),
    [
        ({"tool_origin": "client_mcp"}, 1),
        ({"tool_origin": "client_mcp", "exposed_tool_name": "start_process"}, 2),
        (
            {
                "tool_origin": "client_mcp",
                "server_name": "desktop_commander",
                "source_tool_name": "start_process",
            },
            3,
        ),
        (
            {
                "tool_origin": "client_mcp",
                "qualified_tool_id": "desktop_commander::start_process",
            },
            4,
        ),
    ],
)
def test_policy_match_specificity_orders_shapes_least_to_most_specific(
    match_kwargs, expected_specificity
):
    match = ToolExecutionPolicyMatch(**match_kwargs)

    assert policy_match_specificity(match) == expected_specificity


# --- matching_policy_rules ---------------------------------------------------------


def test_matching_policy_rules_returns_empty_for_no_matches():
    identity = _identity()
    rules = {
        "server-only": ToolExecutionPolicyOverride(
            match={"tool_origin": "server_mcp"},
            timeout_seconds=10,
        ),
    }

    assert matching_policy_rules(identity, rules) == []


def test_matching_policy_rules_orders_least_to_most_specific():
    identity = _identity()
    rules = {
        "exact": ToolExecutionPolicyOverride(
            match={
                "tool_origin": "client_mcp",
                "qualified_tool_id": "desktop_commander::start_process",
            },
            timeout_seconds=45,
        ),
        "origin-default": ToolExecutionPolicyOverride(
            match={"tool_origin": "client_mcp"},
            max_timeout_seconds=60,
        ),
        "server-source": ToolExecutionPolicyOverride(
            match={
                "tool_origin": "client_mcp",
                "server_name": "desktop_commander",
                "source_tool_name": "start_process",
            },
            retry_safe=False,
        ),
        "exposed-name": ToolExecutionPolicyOverride(
            match={
                "tool_origin": "client_mcp",
                "exposed_tool_name": "client__desktop_commander__start_process",
            },
            idempotent=False,
        ),
        "not-applicable-origin": ToolExecutionPolicyOverride(
            match={"tool_origin": "internal"},
            timeout_seconds=5,
        ),
    }

    ordered = matching_policy_rules(identity, rules)

    assert [config_key for config_key, _ in ordered] == [
        "origin-default",
        "exposed-name",
        "server-source",
        "exact",
    ]


def test_matching_policy_rules_rejects_same_specificity_ambiguity():
    identity = _identity()
    rules = {
        "exact-a": ToolExecutionPolicyOverride(
            match={
                "tool_origin": "client_mcp",
                "qualified_tool_id": "desktop_commander::start_process",
            },
            timeout_seconds=45,
        ),
        "exact-b": ToolExecutionPolicyOverride(
            match={
                "tool_origin": "client_mcp",
                "qualified_tool_id": "desktop_commander::start_process",
            },
            timeout_seconds=90,
        ),
    }

    with pytest.raises(AmbiguousToolExecutionPolicyError) as exc_info:
        matching_policy_rules(identity, rules)

    message = str(exc_info.value)
    assert "exact-a" in message
    assert "exact-b" in message


def test_matching_policy_rules_ambiguity_only_applies_within_same_specificity():
    identity = _identity()
    rules = {
        "origin-default": ToolExecutionPolicyOverride(
            match={"tool_origin": "client_mcp"},
            max_timeout_seconds=60,
        ),
        "exact": ToolExecutionPolicyOverride(
            match={
                "tool_origin": "client_mcp",
                "qualified_tool_id": "desktop_commander::start_process",
            },
            timeout_seconds=45,
        ),
    }

    ordered = matching_policy_rules(identity, rules)

    assert [config_key for config_key, _ in ordered] == ["origin-default", "exact"]


# --- resolve_tool_execution_policy ------------------------------------------------


@pytest.fixture(autouse=True)
def _baseline_tool_execution_settings(monkeypatch):
    """Deterministic policy settings so resolver tests never depend on
    whatever the real environment's .env happens to override."""
    monkeypatch.setattr(settings, "tool_execution_timeout", 30)
    monkeypatch.setattr(settings, "tool_execution_policies", {})
    monkeypatch.setattr(settings, "tool_execution_max_interactive_timeout_seconds", 120.0)
    monkeypatch.setattr(settings, "tool_execution_cancellation_grace_seconds", 2.0)
    monkeypatch.setattr(settings, "tool_execution_client_execution_grace_seconds", 2.0)
    monkeypatch.setattr(settings, "tool_execution_client_response_grace_seconds", 1.0)


def test_unknown_tool_uses_single_attempt_bounded_default():
    tool = SimpleNamespace(name="write_todos", metadata={})

    policy = resolve_tool_execution_policy(
        tool, exposed_tool_name="write_todos", invocation_kind="native_async"
    )

    assert policy.timeout_seconds == 30.0
    assert policy.hard_timeout_seconds == 32.0
    assert policy.total_timeout_seconds == 32.0
    assert policy.max_attempts == 1
    assert policy.metadata_trusted is False
    assert policy.policy_config_keys == ()
    assert policy.outer_timeout_disabled is False
    assert policy.policy_source == "default"
    assert policy.cancellation == "cooperative"


def test_untrusted_mcp_meta_cannot_raise_timeout_or_enable_retry():
    tool = SimpleNamespace(
        name="start_process",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
            "_meta": {"execution_timeout_seconds": None, "retry_safe": True},
        },
    )

    policy = resolve_tool_execution_policy(
        tool, exposed_tool_name="start_process", invocation_kind="sync_thread"
    )

    assert policy.timeout_seconds == 30.0
    assert policy.hard_timeout_seconds == 32.0
    assert policy.total_timeout_seconds == 32.0
    assert policy.max_attempts == 1
    assert policy.retry_safe is False
    assert policy.metadata_trusted is False
    assert policy.policy_config_keys == ()
    assert policy.outer_timeout_disabled is False
    assert policy.policy_source == "default"


def test_exact_rule_may_trust_allowlisted_mcp_fields(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "desktop-start-process": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "server_mcp",
                    "qualified_tool_id": "desktop_commander::start_process",
                },
                trust_mcp_metadata=True,
            ),
        },
    )
    tool = SimpleNamespace(
        name="start_process",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
            "idempotentHint": True,
            "_meta": {
                "execution_timeout_seconds": 45,
                "retry_safe": True,
                "timeout_hint": "x" * 300,
                "execution_mode": "background",
            },
        },
    )

    policy = resolve_tool_execution_policy(
        tool, exposed_tool_name="start_process", invocation_kind="native_async"
    )

    assert policy.timeout_seconds == 45.0
    assert policy.hard_timeout_seconds == 47.0
    assert policy.total_timeout_seconds == 47.0
    assert policy.max_attempts == 1
    assert policy.retry_safe is True
    assert policy.idempotent is True
    assert len(policy.timeout_hint) == 240
    assert policy.metadata_trusted is True
    assert policy.policy_config_keys == ("desktop-start-process",)
    assert policy.outer_timeout_disabled is False
    assert policy.policy_source == "config+metadata"


def test_remote_meta_cannot_disable_outer_timeout(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "desktop-start-process": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "server_mcp",
                    "qualified_tool_id": "desktop_commander::start_process",
                },
                trust_mcp_metadata=True,
            ),
        },
    )
    tool = SimpleNamespace(
        name="start_process",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
            "_meta": {
                "disable_outer_timeout": True,
                "hard_timeout_seconds": 999,
                "total_timeout_seconds": 999,
                "max_attempts": 5,
                "cancellation": "none",
            },
        },
    )

    policy = resolve_tool_execution_policy(
        tool, exposed_tool_name="start_process", invocation_kind="native_async"
    )

    assert policy.outer_timeout_disabled is False
    assert policy.hard_timeout_seconds == 32.0
    assert policy.total_timeout_seconds == 32.0
    assert policy.max_attempts == 1
    assert policy.cancellation == "cooperative"


def test_broad_cap_still_limits_more_specific_override(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "server-mcp-cap": ToolExecutionPolicyOverride(
                match={"tool_origin": "server_mcp"},
                max_timeout_seconds=60,
            ),
            "desktop-start-process": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "server_mcp",
                    "qualified_tool_id": "desktop_commander::start_process",
                },
                timeout_seconds=90,
            ),
        },
    )
    tool = SimpleNamespace(
        name="start_process",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
        },
    )

    policy = resolve_tool_execution_policy(
        tool, exposed_tool_name="start_process", invocation_kind="native_async"
    )

    assert policy.total_timeout_seconds == 60.0
    assert policy.hard_timeout_seconds == 60.0
    assert policy.timeout_seconds == 58.0
    assert policy.policy_config_keys == ("server-mcp-cap", "desktop-start-process")
    assert policy.metadata_trusted is False
    assert policy.outer_timeout_disabled is False
    assert policy.policy_source == "config"


def test_retry_attempts_require_retry_safe_or_idempotent(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "unsafe-retries": ToolExecutionPolicyOverride(
                match={"tool_origin": "internal"},
                max_attempts=2,
                retry_safe=False,
                idempotent=False,
            ),
        },
    )
    tool = SimpleNamespace(name="write_todos", metadata={})

    with pytest.raises(ToolExecutionPolicyValidationError):
        resolve_tool_execution_policy(
            tool, exposed_tool_name="write_todos", invocation_kind="native_async"
        )


def test_only_dispatch_subagents_can_disable_outer_timeout(monkeypatch):
    accepted_tool = SimpleNamespace(
        name="dispatch_subagents",
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::dispatch_subagents",
            "application_execution_policy": {"disable_outer_timeout": True},
        },
    )
    policy = resolve_tool_execution_policy(
        accepted_tool, exposed_tool_name="dispatch_subagents", invocation_kind="native_async"
    )
    assert policy.outer_timeout_disabled is True

    rejected_internal_tool = SimpleNamespace(
        name="some_other_tool",
        metadata={
            "tool_origin": "internal",
            "qualified_tool_id": "internal::some_other_tool",
            "application_execution_policy": {"disable_outer_timeout": True},
        },
    )
    with pytest.raises(ToolExecutionPolicyValidationError):
        resolve_tool_execution_policy(
            rejected_internal_tool,
            exposed_tool_name="some_other_tool",
            invocation_kind="native_async",
        )

    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "server-disable": ToolExecutionPolicyOverride(
                match={"tool_origin": "server_mcp", "qualified_tool_id": "srv::tool"},
                disable_outer_timeout=True,
            ),
        },
    )
    rejected_server_tool = SimpleNamespace(
        name="tool",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "srv",
            "qualified_tool_id": "srv::tool",
        },
    )
    with pytest.raises(ToolExecutionPolicyValidationError):
        resolve_tool_execution_policy(
            rejected_server_tool, exposed_tool_name="tool", invocation_kind="native_async"
        )

    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "client-disable": ToolExecutionPolicyOverride(
                match={"tool_origin": "client_mcp", "qualified_tool_id": "srv::tool"},
                disable_outer_timeout=True,
            ),
        },
    )
    rejected_client_tool = SimpleNamespace(
        name="client__srv__tool",
        metadata={
            "tool_origin": "client_mcp",
            "server_name": "srv",
            "qualified_tool_id": "srv::tool",
        },
    )
    with pytest.raises(ToolExecutionPolicyValidationError):
        resolve_tool_execution_policy(
            rejected_client_tool,
            exposed_tool_name="client__srv__tool",
            invocation_kind="client_runtime",
        )


def test_same_specificity_matches_fail_closed(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "exact-a": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "server_mcp",
                    "qualified_tool_id": "desktop_commander::start_process",
                },
                timeout_seconds=45,
            ),
            "exact-b": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "server_mcp",
                    "qualified_tool_id": "desktop_commander::start_process",
                },
                timeout_seconds=90,
            ),
        },
    )
    tool = SimpleNamespace(
        name="start_process",
        metadata={
            "tool_origin": "server_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
        },
    )

    with pytest.raises(AmbiguousToolExecutionPolicyError) as exc_info:
        resolve_tool_execution_policy(
            tool, exposed_tool_name="start_process", invocation_kind="native_async"
        )

    message = str(exc_info.value)
    assert "exact-a" in message
    assert "exact-b" in message


# --- resolve_tool_execution_policy: invocation kind, client deadlines -------------


def test_resolve_tool_execution_policy_rejects_unknown_invocation_kind():
    tool = SimpleNamespace(name="write_todos", metadata={})

    with pytest.raises(ValueError):
        resolve_tool_execution_policy(
            tool, exposed_tool_name="write_todos", invocation_kind="background"
        )


def test_non_client_tool_has_no_client_deadlines():
    tool = SimpleNamespace(name="write_todos", metadata={})

    policy = resolve_tool_execution_policy(
        tool, exposed_tool_name="write_todos", invocation_kind="native_async"
    )

    assert policy.client_execution_timeout_seconds is None
    assert policy.client_response_timeout_seconds is None
    assert policy.cancellation == "cooperative"


def test_client_tool_derives_client_deadlines_from_soft_timeout():
    tool = SimpleNamespace(
        name="client__desktop_commander__start_process",
        metadata={
            "tool_origin": "client_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
        },
    )

    policy = resolve_tool_execution_policy(
        tool,
        exposed_tool_name="client__desktop_commander__start_process",
        invocation_kind="client_runtime",
    )

    assert policy.timeout_seconds == 30.0
    assert policy.client_execution_timeout_seconds == 28.0
    assert policy.client_response_timeout_seconds == 29.0
    assert (
        policy.client_execution_timeout_seconds
        < policy.client_response_timeout_seconds
        < policy.timeout_seconds
    )
    assert policy.cancellation == "abandon_only"


def test_client_tool_too_short_for_strict_ordering_fails_validation(monkeypatch):
    monkeypatch.setattr(
        settings,
        "tool_execution_policies",
        {
            "tight-timeout": ToolExecutionPolicyOverride(
                match={
                    "tool_origin": "client_mcp",
                    "qualified_tool_id": "desktop_commander::start_process",
                },
                timeout_seconds=1,
            ),
        },
    )
    tool = SimpleNamespace(
        name="client__desktop_commander__start_process",
        metadata={
            "tool_origin": "client_mcp",
            "server_name": "desktop_commander",
            "qualified_tool_id": "desktop_commander::start_process",
        },
    )

    with pytest.raises(ToolExecutionPolicyValidationError):
        resolve_tool_execution_policy(
            tool,
            exposed_tool_name="client__desktop_commander__start_process",
            invocation_kind="client_runtime",
        )


# --- tool_policy_context / get_current_tool_policy --------------------------------


def test_tool_policy_context_scopes_and_resets():
    assert get_current_tool_policy() is None
    policy = resolve_tool_execution_policy(
        SimpleNamespace(name="write_todos", metadata={}),
        exposed_tool_name="write_todos",
        invocation_kind="native_async",
    )

    with tool_policy_context(policy) as active:
        assert active is policy
        assert get_current_tool_policy() is policy

    assert get_current_tool_policy() is None


def test_tool_policy_context_nesting_restores_previous_policy():
    outer = resolve_tool_execution_policy(
        SimpleNamespace(name="outer_tool", metadata={}),
        exposed_tool_name="outer_tool",
        invocation_kind="native_async",
    )
    inner = resolve_tool_execution_policy(
        SimpleNamespace(name="inner_tool", metadata={}),
        exposed_tool_name="inner_tool",
        invocation_kind="native_async",
    )

    with tool_policy_context(outer):
        assert get_current_tool_policy() is outer
        with tool_policy_context(inner):
            assert get_current_tool_policy() is inner
        assert get_current_tool_policy() is outer

    assert get_current_tool_policy() is None
