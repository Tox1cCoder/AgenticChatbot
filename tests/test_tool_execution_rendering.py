from __future__ import annotations

import pytest

from app.ai.tool_execution import execute_tool_calls


class _FakeTool:
    name = "canva_create_presentation"

    async def ainvoke(self, args):
        return {
            "content": [{"type": "text", "text": "Created presentation"}],
            "structuredContent": {"presentation_id": "deck_123"},
            "_meta": {"openai/outputTemplate": "ui://canva/presentation-viewer.html"},
        }


class _FailingTool:
    name = "failing_tool"

    async def ainvoke(self, args):
        raise ValueError("permission denied")


@pytest.mark.asyncio
async def test_execute_tool_calls_returns_model_text_and_render_artifact():
    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "tool-call-1",
                "name": "canva_create_presentation",
                "args": {"prompt": "roadmap"},
            }
        ],
        tool_map={"canva_create_presentation": _FakeTool()},
    )

    assert outputs == [
        {
            "tool_call_id": "tool-call-1",
            "name": "canva_create_presentation",
            "content": "Created presentation",
            "render": artifacts[0]["render"],
        }
    ]
    assert artifacts[0]["output"] == "Created presentation"
    assert artifacts[0]["render"]["type"] == "mcp_app"
    assert artifacts[0]["render"]["template_uri"] == "ui://canva/presentation-viewer.html"
    assert artifacts[0]["render"]["structured_content"]["presentation_id"] == "deck_123"
    assert images == []


class _MixedContentTool:
    """Returns text + image content blocks — the MCP default shape."""

    name = "mixed_content_tool"

    async def ainvoke(self, args):
        return {
            "content": [
                {"type": "text", "text": "Here is the chart."},
                {"type": "image", "mimeType": "image/png", "data": "iVBORw0KGgo="},
            ]
        }


class _MissingNameTool:
    name = ""

    async def ainvoke(self, args):
        return "never-called"


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_mcp_image_content_blocks():
    """Tool execution must forward MCP content blocks to the normalizer intact.
    Without this, mixed text+image results degrade to render.type == 'json'."""
    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "tc-mixed", "name": "mixed_content_tool", "args": {}}],
        tool_map={"mixed_content_tool": _MixedContentTool()},
    )

    render = outputs[0]["render"]
    assert render["type"] == "image"
    assert render["content"][0]["type"] == "text"
    assert render["content"][1]["type"] == "image"
    assert outputs[0]["content"] == "Here is the chart."
    assert artifacts[0]["render"]["type"] == "image"


@pytest.mark.asyncio
async def test_execute_tool_calls_missing_tool_does_not_double_prefix_error():
    """Tool-not-found passes an already-prefixed error_msg; render must not double-prefix."""
    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "tc-missing", "name": "nope_tool", "args": {}}],
        tool_map={},
    )

    assert outputs[0]["content"].startswith("Error: Tool nope_tool not found")
    assert not outputs[0]["content"].startswith("Error: Error:")
    assert outputs[0]["render"]["type"] == "error"
    assert not outputs[0]["render"]["error"].lower().startswith("error:")
    assert artifacts[0]["status"] == "error"


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_error_render_artifact():
    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[{"id": "tool-call-err", "name": "failing_tool", "args": {}}],
        tool_map={"failing_tool": _FailingTool()},
    )

    assert outputs[0]["content"] == "Error: permission denied"
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["output"] == "Error: permission denied"
    assert artifacts[0]["render"]["type"] == "error"
    assert artifacts[0]["render"]["error"] == "permission denied"
    assert images == []


from app.ai.graph import MultiAgentWorkflow


def test_lookup_tool_render_payload_from_state_context():
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    state_values = {
        "context": {
            "tool_render_results": {
                "tool-call-1": {
                    "version": 1,
                    "type": "mcp_app",
                    "template_uri": "ui://canva/presentation-viewer.html",
                }
            }
        }
    }

    render = workflow._lookup_tool_render_payload(state_values, "tool-call-1")

    assert render["type"] == "mcp_app"
    assert render["template_uri"] == "ui://canva/presentation-viewer.html"


from app.services.stream_events import build_canonical_tool_event


def test_canonical_tool_event_preserves_render_payload():
    render = {
        "version": 1,
        "type": "mcp_app",
        "template_uri": "ui://canva/presentation-viewer.html",
    }

    event = build_canonical_tool_event(
        phase="end",
        name="canva_create_presentation",
        tool_call_id="tool-call-1",
        result="Created presentation",
        render=render,
    )

    assert event["result"] == "Created presentation"
    assert event["render"] == render
