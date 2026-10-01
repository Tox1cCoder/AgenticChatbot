"""HITL fires correctly for client (sidecar) tools and deferred (search-loaded) tools."""

from types import SimpleNamespace

from app.ai.hitl_config import (
    any_call_requires_approval,
    build_tool_interrupt_payload,
    resolve_call_identity,
)


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def test_client_server_rule_gates_a_sidecar_tool_by_name_alone():
    # Sidecar tool, NOT yet in any tool_map (e.g. resolved purely from the call name).
    policy = {
        "master_enabled": True,
        "client_rules": {
            "client_mcp": {"servers": {"desktop_commander": True}, "tools": {}},
            "client_skill": {"servers": {}, "tools": {}},
        },
        "global_tools": [],
    }
    calls = [{"name": "client__desktop_commander__start_process", "args": {}, "id": "c1"}]
    assert any_call_requires_approval(calls, policy=policy) is True


def test_deferred_server_tool_gated_by_read_only_global_policy_after_autoload():
    # A server tool discovered + autoloaded via tool_search this turn: bare name, no
    # metadata, server resolved through the MCP manager (the deferred-binding path).
    loaded_tool = SimpleNamespace(name="run_query", metadata={})
    tool_map = {"run_query": loaded_tool}
    manager = _FakeManager({id(loaded_tool): "postgres"})
    policy = {
        "master_enabled": True,
        "client_rules": {
            "client_mcp": {"servers": {}, "tools": {}},
            "client_skill": {"servers": {}, "tools": {}},
        },
        "global_tools": ["run_query"],
    }

    identity = resolve_call_identity({"name": "run_query"}, tool_map=tool_map, mcp_manager=manager)
    assert identity.server_name == "postgres"

    calls = [{"name": "run_query", "args": {}, "id": "c1"}]
    assert (
        any_call_requires_approval(calls, policy=policy, tool_map=tool_map, mcp_manager=manager)
        is True
    )


def test_interrupt_payload_carries_client_provenance():
    client_tool = SimpleNamespace(
        name="client__excel__delete_sheet",
        metadata={
            "server_name": "excel",
            "qualified_tool_id": "excel::delete_sheet",
            "tool_origin": "client_mcp",
        },
    )

    payload = build_tool_interrupt_payload(
        [{"name": "client__excel__delete_sheet", "args": {}, "id": "c1"}],
        tool_map={client_tool.name: client_tool},
        device_id="dev-1",
    )
    prov = payload["metadata"]["tool_provenance"]
    entry = next(iter(prov.values()))
    assert entry["server_name"] == "excel"
    assert entry["qualified_tool_id"] == "excel::delete_sheet"
    assert entry["tool_origin"] == "client_mcp"


def test_interrupt_payload_redacts_sensitive_args_in_prompt():
    # The approval prompt (action_requests) must not surface a sensitive-keyed
    # argument value, but must keep normal args visible for the approver — and
    # must NOT mutate the original tool call that executes on approval.
    tool_call = {
        "name": "client__skill_demo__mutate",
        "args": {"calendar_id": "primary", "api_token": "SUPER-SECRET"},
        "id": "c1",
    }
    original_args = tool_call["args"]

    payload = build_tool_interrupt_payload([tool_call], tool_map={}, device_id="dev-1")

    prompt_args = payload["action_requests"][0]["args"]
    assert prompt_args["calendar_id"] == "primary"
    assert prompt_args["api_token"] == "<redacted>"
    # The real tool call is untouched, so execution on approval uses real args.
    assert original_args["api_token"] == "SUPER-SECRET"


def test_sensitive_argument_redaction_recurses_through_nested_objects_and_lists():
    from app.ai.hitl_config import redact_sensitive_args

    redacted = redact_sensitive_args(
        {
            "config": {
                "api_token": "SECRET",
                "items": [{"password": "HIDDEN"}, {"name": "safe"}],
            }
        }
    )

    assert redacted == {
        "config": {
            "api_token": "<redacted>",
            "items": [{"password": "<redacted>"}, {"name": "safe"}],
        }
    }
