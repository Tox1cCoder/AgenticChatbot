"""Static guard: the demo MCP panel manages HITL approval via the sidecar API."""

from pathlib import Path


def _demo_source() -> str:
    return Path("demo.py").read_text(encoding="utf-8")


def test_demo_defines_hitl_helpers_and_calls_settings_endpoint():
    src = _demo_source()
    assert "def get_hitl_settings(" in src
    assert "def set_hitl_setting(" in src
    assert "def clear_hitl_setting(" in src
    assert 'make_api_request("GET", "/hitl/settings")' in src
    assert 'make_api_request("POST", "/hitl/settings"' in src


def test_demo_renders_per_server_and_per_tool_controls():
    src = _demo_source()
    assert "Approval: ON" in src              # per-server toggle label
    assert "hitl_tool_mode_" in src           # per-tool tri-state widget key prefix
    assert "qualified_tool_options" in src    # duplicate tool names select by server::tool


def test_demo_renders_per_skill_command_hitl_controls():
    src = _demo_source()
    assert 'f"skill::{skill_name}::run_skill_command"' in src
    assert 'skill.get("commandCapable", False)' in src
    assert 'key=f"hitl_skill_mode_{skill_name}"' in src
    assert 'modes = ["Inherit", "Require", "Skip"]' in src
    assert 'clear_hitl_setting("tool", skill_qualified_id)' in src
    assert (
        'set_hitl_setting("tool", skill_qualified_id, chosen == "Require")'
        in src
    )
    assert "approval rules below are inactive until it is enabled" in src


def test_demo_uses_shared_hitl_decision_builder():
    src = _demo_source()
    assert "from app.ui.hitl_decisions import" in src
    assert "build_interrupt_decision(" in src
    assert "interrupt_request_target_ids(" in src
