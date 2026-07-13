from types import SimpleNamespace

from app.ai.client_runtime_tools import _parse_tool_specs, make_tool_instance_id
from app.ai.hitl_config import (
    CallIdentity,
    _build_tool_interrupt_request,
    any_call_requires_approval,
    identity_requires_approval,
    redact_sensitive_args,
)


def _policy(tools=None):
    return {
        "master_enabled": True,
        "servers": {},
        "tools": tools or {},
        "global_tools": [],
    }


def _command_entry():
    return {
        "name": "run_skill_command",
        "description": "Run a bundled command",
        "origin": "skill",
        "server_name": "skill_demo",
        "qualified_id": "skill::demo::run_skill_command",
        "input_schema": {
            "type": "object",
            "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
            "required": ["argv"],
        },
        "mutation": True,
        "source_hash": "a" * 64,
    }


def test_fixed_skill_command_requires_approval_by_default():
    identity = CallIdentity(
        name="client__skill_demo__run_skill_command",
        server_name="skill_demo",
        qualified_tool_id="skill::demo::run_skill_command",
        origin="client_skill",
        mutation=True,
    )

    assert identity_requires_approval(identity, _policy()) is True


def test_explicit_policy_can_preapprove_exact_command_tool():
    identity = CallIdentity(
        name="client__skill_demo__run_skill_command",
        server_name="skill_demo",
        qualified_tool_id="skill::demo::run_skill_command",
        origin="client_skill",
        mutation=True,
    )

    assert (
        identity_requires_approval(
            identity,
            _policy(tools={"skill::demo::run_skill_command": False}),
        )
        is False
    )


def test_catalog_parsing_preserves_mandatory_mutation_metadata():
    specs = _parse_tool_specs({"tools": [_command_entry()]})

    assert len(specs) == 1
    assert specs[0].qualified_tool_id == "skill::demo::run_skill_command"
    assert specs[0].mutation is True
    assert specs[0].source_hash == "a" * 64


def test_source_update_rotates_tool_instance_even_before_catalog_version_changes():
    first = make_tool_instance_id(
        "device", "session", "skill::demo::run_skill_command", 4, "a" * 64
    )
    second = make_tool_instance_id(
        "device", "session", "skill::demo::run_skill_command", 4, "b" * 64
    )

    assert first != second


def test_any_call_gate_reads_skill_command_mutation_metadata():
    tool = SimpleNamespace(
        name="client__skill_demo__run_skill_command",
        metadata={
            "server_name": "skill_demo",
            "qualified_tool_id": "skill::demo::run_skill_command",
            "tool_origin": "client_skill",
            "mutation": True,
        },
    )
    calls = [{"name": tool.name, "args": {"argv": ["demo-cli"]}, "id": "call-1"}]

    assert any_call_requires_approval(calls, policy=_policy(), tool_map={tool.name: tool})


def test_approval_prompt_redacts_sensitive_command_arguments():
    assert redact_sensitive_args({"argv": ["demo-cli"], "api_token": "SECRET"}) == {
        "argv": ["demo-cli"],
        "api_token": "<redacted>",
    }
    request = _build_tool_interrupt_request(
        {
            "tool_call_id": "call-1",
            "action": "skill::demo::run_skill_command",
            "args": {"argv": ["demo-cli"], "api_token": "SECRET"},
        },
        0,
        "task",
        {},
    )
    assert request.args["api_token"] == "<redacted>"
