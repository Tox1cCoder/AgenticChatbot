"""Tests for the bounded prompt-inventory helpers introduced for inline rich
references (response_format.md Task 4).
"""

from __future__ import annotations

from app.ai.graph import MultiAgentWorkflow
from app.ai.prompts import (
    INLINE_RICH_RESPONSE_SUFFIX,
    build_rich_response_guidance,
)
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from app.core.config import settings
from app.core.rich_response import (
    RichDisplayPolicy,
    RichItemType,
    build_rich_item_inventory_block,
)


def _image_candidate(id_: str, description: str = "An image") -> dict:
    return {
        "id": id_,
        "type": RichItemType.image.value,
        "display_policy": RichDisplayPolicy.inline_only.value,
        "alt_text": description,
        "title": description,
        "payload": {
            "url": f"https://img.test/{id_}.png",
            "mime_type": "image/png",
            "description": description,
        },
    }


def _widget_candidate(id_: str = "widget:w-1", title: str = "Pressure chart") -> dict:
    return {
        "id": id_,
        "type": RichItemType.live_widget.value,
        "display_policy": RichDisplayPolicy.inline_or_append.value,
        "title": title,
        "payload": {
            "widget_id": id_.split(":", 1)[-1],
            "session_id": "conv-1",
            "widget_type": "chart",
            "status": "active",
            "version": 1,
            "connection_endpoint": "/widgets/w-1/connection",
        },
    }


def test_inline_rich_response_suffix_mentions_marker_grammar_and_no_invention():
    assert "<!--rich:" in INLINE_RICH_RESPONSE_SUFFIX
    assert "invent" in INLINE_RICH_RESPONSE_SUFFIX.lower()


def test_inline_rich_response_suffix_requires_marker_for_created_live_widgets():
    assert "Created live widgets must be placed" in INLINE_RICH_RESPONSE_SUFFIX
    assert "Do not tell the user a widget is inline" in INLINE_RICH_RESPONSE_SUFFIX


def test_build_rich_response_guidance_returns_empty_when_disabled():
    block = build_rich_response_guidance(
        candidates=[_image_candidate("image:tool:c1:0")],
        enabled=False,
        capability=True,
    )
    assert block == ""


def test_build_rich_response_guidance_returns_empty_without_capability():
    block = build_rich_response_guidance(
        candidates=[_image_candidate("image:tool:c1:0")],
        enabled=True,
        capability=False,
    )
    assert block == ""


def test_build_rich_response_guidance_returns_empty_without_candidates():
    block = build_rich_response_guidance(candidates=[], enabled=True, capability=True)
    assert block == ""


def test_build_rich_response_guidance_includes_suffix_and_inventory():
    block = build_rich_response_guidance(
        candidates=[_image_candidate("image:tool:c1:0", "Selected image")],
        enabled=True,
        capability=True,
    )
    assert "<!--rich:" in block
    assert "image:tool:c1:0" in block


def test_image_guidance_assigns_visible_caption_to_renderer():
    block = build_rich_response_guidance(
        candidates=[_image_candidate("image:tool:c1:0")],
        enabled=True,
        capability=True,
    )

    assert "Do not write a Markdown caption" in block
    assert "caption as normal markdown" not in block
    assert "add a useful caption" not in block


def test_build_rich_response_guidance_omits_base64_data():
    candidate = _image_candidate("image:doc:1")
    candidate["payload"] = {"data": "QUJDRA==", "mime_type": "image/png"}
    block = build_rich_response_guidance(candidates=[candidate], enabled=True, capability=True)
    assert "QUJDRA==" not in block


def test_build_rich_response_guidance_prefers_non_image_when_trimmed():
    widget = _widget_candidate()
    images = [_image_candidate(f"image:tool:c1:{i}") for i in range(20)]
    block = build_rich_response_guidance(
        candidates=[*images, widget],
        enabled=True,
        capability=True,
        max_items=1,
    )
    assert widget["id"] in block


def test_graph_does_not_forward_candidates_for_non_capable_response(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    workflow = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    response = AgentResponse(
        agent_type=AgentType.CHAT,
        agent_id="chat_agent",
        message=AgentMessage(role=MessageRole.ASSISTANT, content="Answer"),
        metadata={},
    )
    state = {
        "context": {
            "inline_rich_response_v1": False,
            "rich_item_candidates": [_image_candidate("image:tool:c1:0")],
        }
    }

    workflow._merge_tool_artifacts(state, response)

    assert "_rich_item_candidates" not in response.metadata
    assert "_inline_rich_response_v1" not in response.metadata


def test_media_guidance_requires_disambiguated_image_query():
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET

    text = MEDIA_CAPABILITY_SNIPPET.lower()
    assert "brave_image_search" in text
    assert "disambiguat" in text
    assert "same tool block" in text or "parallel" in text


# ---------------------------------------------------------------------------
# Inventory image cap: an image_group counts as one image entry against the
# per-answer image cap, same as a single image item.
# ---------------------------------------------------------------------------


def test_inventory_caps_image_entries_and_counts_a_group_as_one():
    items = [
        {"id": "widget:w1", "type": "live_widget", "title": "W"},
        {"id": "imagegroup:tool:c1", "type": "image_group", "title": "G1"},
        {"id": "imagegroup:tool:c2", "type": "image_group", "title": "G2"},
        {"id": "image:tool:c3:0", "type": "image", "title": "I3"},
    ]
    block = build_rich_item_inventory_block(
        items, max_items=12, max_chars=4000, summary_chars=180, image_max_items=2
    )
    assert "widget:w1" in block
    assert block.count("image_group") + block.count("| image |") == 2
    assert "image:tool:c3:0" not in block


# ---------------------------------------------------------------------------
# Presentation counter: record_presentation was defined in an earlier task
# with no caller. It must be wired where items actually enter the model-facing
# inventory, counting once per candidate offered (before item/char trimming).
# ---------------------------------------------------------------------------


def test_presentation_counter_records_each_presented_image(monkeypatch):
    from app.ai import prompts

    recorded = []

    class _Metrics:
        def record_presentation(self, *, provider, count):
            recorded.append((provider, count))

    monkeypatch.setattr(prompts, "rich_image_metrics", _Metrics(), raising=False)
    prompts.build_rich_response_guidance(
        candidates=[
            {
                "id": "imagegroup:tool:c1",
                "type": "image_group",
                "title": "G",
                "provenance": {"provider": "brave_image_search"},
            }
        ],
        enabled=True,
        capability=True,
    )
    assert recorded == [("brave_image_search", 1)]
