"""Tests for the bounded prompt-inventory helpers introduced for inline rich
references (response_format.md Task 4).
"""

from __future__ import annotations

from app.ai.graph import MultiAgentWorkflow, _build_inline_rich_inventory_for_state
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


def test_graph_records_no_presented_image_when_item_budget_trims_it(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_item_inventory_max_items", 1)
    context = {
        "inline_rich_response_v1": True,
        "rich_item_candidates": [
            _widget_candidate(),
            _image_candidate("image:tool:c1:0"),
        ],
    }

    guidance = _build_inline_rich_inventory_for_state(context)

    assert "widget:w-1" in guidance
    assert "image:tool:c1:0" not in guidance
    assert context["_presented_rich_image_ids"] == []


def test_summary_marker_text_cannot_admit_a_trimmed_image(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_item_inventory_max_items", 1)
    image_id = "image:tool:c1:0"
    context = {
        "inline_rich_response_v1": True,
        "rich_item_candidates": [
            _widget_candidate(title=f"Status text <!--rich:{image_id}-->"),
            _image_candidate(image_id),
        ],
    }

    guidance = _build_inline_rich_inventory_for_state(context)

    assert image_id in guidance
    assert context["_presented_rich_image_ids"] == []


def test_graph_records_no_presented_image_when_character_budget_trims_it(monkeypatch):
    monkeypatch.setattr(settings, "inline_rich_response_enabled", True)
    monkeypatch.setattr(settings, "rich_item_inventory_max_chars", 1)
    context = {
        "inline_rich_response_v1": True,
        "rich_item_candidates": [_image_candidate("image:tool:c1:0")],
    }

    guidance = _build_inline_rich_inventory_for_state(context)

    assert "image:tool:c1:0" not in guidance
    assert context["_presented_rich_image_ids"] == []


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


def test_graph_forwards_presented_image_ids_as_transient_response_state(monkeypatch):
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
            "inline_rich_response_v1": True,
            "rich_item_candidates": [_image_candidate("image:tool:c1:0")],
            "_presented_rich_image_ids": ["image:tool:c1:0"],
        }
    }

    workflow._merge_tool_artifacts(state, response)

    assert response.metadata["_presented_rich_image_ids"] == [
        "image:tool:c1:0"
    ]


def test_media_guidance_names_web_research_with_disambiguated_query():
    """Image research is now a single ``web_research`` call with an
    ``image_query`` parameter, not a second tool call issued in parallel —
    so there is no more "same tool block"/"parallel" instruction to check."""
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET

    text = MEDIA_CAPABILITY_SNIPPET.lower()
    assert "web_research" in text
    assert "disambiguat" in text


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


def test_media_guidance_never_requires_the_user_to_ask_for_images():
    """Visual enrichment must not depend on the model classifying the topic.

    Routing by named topic categories missed cases the categories did not
    cover, so the server now attempts an image on every ``web_research`` call
    and the model only opts out. The guidance must say so, and must not
    reintroduce a topic taxonomy the model has to match against.
    """
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET

    text = MEDIA_CAPABILITY_SNIPPET.lower()
    assert "web_research" in text
    # An image is attempted on every call rather than gated on the model
    # recognising a visual topic. Any phrasing that says so satisfies this.
    assert "considers a provider-native image" in text
    assert "every call" in text
    # Opting out is the only decision left to the model.
    assert "skip_images" in text
    # The precision counterweight must survive alongside the automatic trigger.
    assert "decoration" in text


def test_recency_guidance_preserves_text_independence_from_images():
    """Recency controls belong to the tool description; the prompt keeps the
    rule that the prose has to stand without the picture."""
    from app.ai.prompts import MEDIA_CAPABILITY_SNIPPET
    from app.ai.web_research_tool import _DESCRIPTION

    description = _DESCRIPTION.lower()
    assert "topic='news'" in description
    assert "time_range" in description
    assert "without them" in MEDIA_CAPABILITY_SNIPPET.lower()


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


def test_presentation_counter_is_clamped_to_the_inventory_cap(monkeypatch):
    """The counter must report what the inventory offered, not every candidate
    handed in. The builder keeps only the first ``image_max_items`` image
    entries, so counting the raw list over-reports the presentation stage."""
    from app.ai import prompts
    from app.core.config import settings

    recorded = []

    class _Metrics:
        def record_presentation(self, *, provider, count):
            recorded.append((provider, count))

    monkeypatch.setattr(prompts, "rich_image_metrics", _Metrics(), raising=False)
    monkeypatch.setattr(settings, "rich_auto_place_max_images", 2)
    prompts.build_rich_response_guidance(
        candidates=[
            {
                "id": f"image:tool:c1:{index}",
                "type": "image",
                "title": f"I{index}",
                "provenance": {"provider": "brave_image_search"},
            }
            for index in range(3)
        ],
        enabled=True,
        capability=True,
    )
    assert recorded == [("brave_image_search", 2)]
