from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.ai.tool_execution_policy import (
    AmbiguousToolExecutionPolicyError,
    ToolIdentity,
    matching_policy_rules,
    policy_match_specificity,
    resolve_tool_identity,
)
from app.core.config import Settings, ToolExecutionPolicyMatch, ToolExecutionPolicyOverride


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
