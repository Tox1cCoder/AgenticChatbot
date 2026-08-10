from __future__ import annotations

import hashlib
import json

import pytest

from app.ai.graph import MultiAgentWorkflow
from app.ai.tool_execution import (
    _group_image_candidates,
    build_image_candidates_from_tool_result,
    build_tool_artifact,
    execute_tool_calls,
)


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


class _StructuredErrorTool:
    name = "structured_error_tool"

    async def ainvoke(self, args):
        return {"error": "image search is not configured"}


class _DirectBraveTool:
    name = "brave_image_search"

    def __init__(self, *, with_proxy: bool = True):
        self.original_url = "https://private-origin.example/full/team.jpg?token=secret"
        self.thumbnail_url = "https://imgs.search.brave.com/proxy/team.jpg"
        display_url = self.thumbnail_url if with_proxy else self.original_url
        image = {
            "url": display_url,
            "original_image_url": self.original_url,
            "provider": "brave_image_search",
            "confidence": "high",
            "result_rank": 1,
            "mime_type": "image/jpeg",
            "width": 1200,
            "height": 800,
        }
        if with_proxy:
            image["thumbnail_url"] = self.thumbnail_url
        self.payload = json.dumps(
            {
                "provider": "brave_image_search",
                "query": "T1 team photo",
                "images": [image],
            }
        )

    async def ainvoke(self, args):
        return [{"type": "text", "text": self.payload}]


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_mcp_image_content_blocks():
    """Tool execution must forward MCP content blocks to the normalizer intact.
    Without this, mixed text+image results degrade to render.type == 'json'."""
    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[{"id": "tc-mixed", "name": "mixed_content_tool", "args": {}}],
        tool_map={"mixed_content_tool": _MixedContentTool()},
    )

    render = outputs[0]["render"]
    assert render["type"] == "image"
    assert render["content"][0]["type"] == "text"
    assert render["content"][1]["type"] == "image"
    assert outputs[0]["content"] == "Here is the chart."
    assert artifacts[0]["render"]["type"] == "image"
    assert images == [
        {
            "mime": "image/png",
            "description": "",
            "data": "iVBORw0KGgo=",
        }
    ]
    candidates = artifacts[0]["_rich_item_candidates"]
    image_candidate = next(item for item in candidates if item["type"] == "image")
    assert image_candidate["payload"] == {
        "mime_type": "image/png",
        "data": "iVBORw0KGgo=",
    }


@pytest.mark.asyncio
async def test_execute_tool_calls_missing_tool_uses_compact_error_payload():
    import json

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "tc-missing", "name": "nope_tool", "args": {}}],
        tool_map={},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "not_found"
    assert not outputs[0]["content"].startswith("Error:")
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error_type"] == "not_found"


@pytest.mark.asyncio
async def test_execute_tool_calls_preserves_error_render_artifact():
    import json

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[{"id": "tool-call-err", "name": "failing_tool", "args": {}}],
        tool_map={"failing_tool": _FailingTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert payload["error_type"] == "validation"
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["output"] == outputs[0]["content"]
    assert artifacts[0]["render"]["type"] == "error"
    assert artifacts[0]["error_type"] == "validation"
    assert images == []


@pytest.mark.asyncio
async def test_execute_tool_calls_marks_structured_error_results_as_errors():
    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[{"id": "structured-error", "name": "structured_error_tool", "args": {}}],
        tool_map={"structured_error_tool": _StructuredErrorTool()},
    )

    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["error"] == "image search is not configured"
    assert images == []


@pytest.mark.asyncio
async def test_direct_brave_execution_keeps_original_url_private_but_preserves_digest():
    tool = _DirectBraveTool()

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "direct-brave",
                "name": "brave_image_search",
                "args": {"query": "T1 team photo"},
            }
        ],
        tool_map={"brave_image_search": tool},
    )

    public_serialization = json.dumps(
        {"outputs": outputs, "artifacts": artifacts}, ensure_ascii=False
    )
    assert "original_image_url" not in public_serialization
    assert tool.original_url not in public_serialization
    assert tool.thumbnail_url in public_serialization
    assert images == []

    [candidate] = artifacts[0]["_rich_item_candidates"]
    assert candidate["provenance"]["original_image_digests"] == {
        tool.thumbnail_url: hashlib.sha256(tool.original_url.encode()).hexdigest()
    }
    assert tool.original_url in tool.payload


@pytest.mark.asyncio
async def test_direct_only_brave_execution_drops_the_private_display_url():
    tool = _DirectBraveTool(with_proxy=False)

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "id": "direct-only-brave",
                "name": "brave_image_search",
                "args": {"query": "T1 team photo"},
            }
        ],
        tool_map={"brave_image_search": tool},
    )

    public_serialization = json.dumps(
        {"outputs": outputs, "artifacts": artifacts}, ensure_ascii=False
    )
    assert "original_image_url" not in public_serialization
    assert tool.original_url not in public_serialization
    assert "_rich_item_candidates" not in artifacts[0]
    assert images == []
    assert tool.original_url in tool.payload


@pytest.mark.asyncio
async def test_aliased_brave_execution_uses_canonical_identity_for_privacy():
    alias = "brave_image_search__brave_image_search"
    tool = _DirectBraveTool()
    tool.name = alias
    tool.metadata = {
        "tool_origin": "server_mcp",
        "qualified_tool_id": "brave_image_search::brave_image_search",
        "source_tool_name": "brave_image_search",
        "server_name": "brave_image_search",
    }

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[{"id": "aliased-brave", "name": alias, "args": {"query": "T1"}}],
        tool_map={alias: tool},
    )

    public_serialization = json.dumps(
        {"outputs": outputs, "artifacts": artifacts}, ensure_ascii=False
    )
    assert "original_image_url" not in public_serialization
    assert tool.original_url not in public_serialization
    assert images == []
    [candidate] = artifacts[0]["_rich_item_candidates"]
    assert candidate["provenance"]["original_image_digests"] == {
        tool.thumbnail_url: hashlib.sha256(tool.original_url.encode()).hexdigest()
    }


def test_tool_artifact_preserves_full_output_by_default():
    output = "x" * 1001

    artifact = build_tool_artifact(
        tool_call_id="call-1",
        tool_name="long_tool",
        tool_args={},
        output_text=output,
        error=None,
    )

    assert artifact["output"] == output


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


@pytest.mark.asyncio
async def test_execute_tool_calls_structured_error_render_stays_compact(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "tool_execution_timeout", 1)

    class _BrokenTool:
        name = "broken_tool"
        metadata = {}

        async def ainvoke(self, args):
            raise PermissionError("permission denied for a very sensitive path")

    outputs, artifacts, _ = await execute_tool_calls(
        tool_calls=[{"id": "call-1", "name": "broken_tool", "args": {}}],
        tool_map={"broken_tool": _BrokenTool()},
    )

    payload = json.loads(outputs[0]["content"])
    assert payload["status"] == "error"
    assert len(outputs[0]["content"]) < 500
    assert outputs[0]["render"]["type"] == "error"
    assert artifacts[0]["status"] == "error"
    assert artifacts[0]["render"]["type"] == "error"


def _brave_result(count: int) -> str:
    return json.dumps(
        {
            "query": "T1 roster",
            "provider": "brave_image_search",
            "images": [
                {
                    "url": f"https://cdn.example/team-{index}.jpg",
                    "provider": "brave_image_search",
                    "mime_type": "image/jpeg",
                    "title": f"T1 roster {index}",
                    "description": f"T1 roster {index}",
                    "width": 995,
                    "height": 565,
                    "source_url": "https://sheepesports.example/t1",
                }
                for index in range(count)
            ],
            "total_results": count,
        }
    )


def test_group_images_false_returns_individual_candidates_not_a_grid():
    """Visual verification must see each Brave candidate on its own pixels;
    grouping before verification would hide images from the verifier."""

    candidates = build_image_candidates_from_tool_result(
        _brave_result(4),
        tool_call_id="brave-call",
        tool_name="brave_image_search",
        group_images=False,
    )

    assert len(candidates) == 4
    assert all(candidate["type"] == "image" for candidate in candidates)


def test_group_images_default_still_collapses_a_multi_result_brave_payload():
    """Regression guard: every existing caller must keep grouping by default."""

    candidates = build_image_candidates_from_tool_result(
        _brave_result(4),
        tool_call_id="brave-call",
        tool_name="brave_image_search",
    )

    assert len(candidates) == 1
    assert candidates[0]["type"] == "image_group"


def test_group_image_candidates_max_items_overrides_the_legacy_setting():
    """A verified gallery's cap (rich_image_gallery_max_items, up to 8) must be
    reachable even though the legacy rich_image_group_max_items caps at 3."""

    candidates = build_image_candidates_from_tool_result(
        _brave_result(5),
        tool_call_id="brave-call",
        tool_name="brave_image_search",
        group_images=False,
    )

    group = _group_image_candidates(
        candidates,
        tool_call_id="brave-call",
        query="T1 roster",
        metric_provider="brave",
        max_items=5,
    )

    assert group["type"] == "image_group"
    assert len(group["payload"]["items"]) == 5
