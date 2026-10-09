"""The snapshot policy that turns a source message into inert forked history.

A forked conversation copies the source transcript into new rows. Those rows
must look the same to a reader -- text, images, canvas, citations, reasoning --
while carrying nothing that could resume, approve, continue, or reconnect to
execution that belongs to the source conversation.

The fixtures are built with the real producers (``build_bot_metadata``'s
widget/canvas helpers, ``validate_public_rich_item``, the canvas agent's
``canvas_update`` shape, ``MessageService``'s continuation/interrupt keys) so a
change in what the producers emit shows up here rather than in production.
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

import pytest

from app.core.response_constants import (
    _canvas_rich_item_from_artifact,
    _widget_rich_item_from_live_widget,
    extract_live_widgets_from_artifacts,
)
from app.core.rich_response import (
    parse_inline_rich_references,
    validate_public_rich_item,
    validate_rich_references,
)
from app.services.conversation_fork_metadata import snapshot_fork_message

SOURCE_CONVERSATION_ID = "827faf55-1041-4357-9e1a-0f7d031fa546"
SOURCE_USER_MESSAGE_ID = "11111111-1111-4111-8111-111111111111"
CHILD_USER_MESSAGE_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
WIDGET_ID = "w-5f2c9e"
WEB_IMAGE_REFERENCE = "/web-images/6c1f0b5e-3d6a-4bb3-9a43-0b1f6c2d9e11"
CHAT_IMAGE_ID = "0d4e7a52-8b6f-4c1e-9f0a-5b2c3d4e5f60"
MESSAGE_ID_MAP = {SOURCE_USER_MESSAGE_ID: CHILD_USER_MESSAGE_ID}


def _public(item: dict[str, Any]) -> dict[str, Any]:
    """Finalize a rich item exactly the way ``_finalize_rich_items`` does."""
    return validate_public_rich_item(item).model_dump(mode="json", exclude_none=True)


def _widget_create_artifact() -> dict[str, Any]:
    return {
        "tool": "widget_create",
        "tool_call_id": "call_widget_1",
        "args": {"title": "Budget planner"},
        "output": json.dumps(
            {
                "widget_id": WIDGET_ID,
                "session_id": SOURCE_CONVERSATION_ID,
                "title": "Budget planner",
                "status": "active",
                "version": 1,
            }
        ),
        "status": "success",
    }


def _search_artifact() -> dict[str, Any]:
    return {
        "tool": "web_search",
        "tool_call_id": "call_search_1",
        "args": {"query": "household budget chart"},
        "output": "3 results",
        "status": "success",
    }


def _canvas_artifact() -> dict[str, Any]:
    return {
        "artifact_id": "canvas:main",
        "revision": 2,
        "operation": "update",
        "content": "<!doctype html><html><body><h1>Budget</h1></body></html>",
        "language": "html",
        "title": "Budget dashboard",
    }


def _image_item() -> dict[str, Any]:
    return _public(
        {
            "id": "img:web:1",
            "type": "image",
            "source": "web_search",
            "display_policy": "inline_only",
            "alt_text": "Monthly spending chart",
            "payload": {
                "url": WEB_IMAGE_REFERENCE,
                "mime_type": "image/png",
                "source_url": "https://example.com/budget",
                "width": 640,
                "height": 480,
            },
            "provenance": {"provider": "brave"},
        }
    )


def _rich_answer_content() -> str:
    return (
        "Here is your planner:\n"
        "\n"
        f"<!--rich:widget:{WIDGET_ID}-->\n"
        "\n"
        "And the spending chart:\n"
        "\n"
        "<!--rich:img:web:1-->\n"
        "\n"
        "<!--rich:canvas:main-->\n"
        "\n"
        "Let me know what to change."
    )


def _rich_answer_metadata() -> dict[str, Any]:
    """A completed v1 answer with a widget, an image, a canvas, and a reply link."""
    tool_artifacts = [_widget_create_artifact(), _search_artifact()]
    live_widgets = extract_live_widgets_from_artifacts(tool_artifacts)
    widget_item = _public(_widget_rich_item_from_live_widget(live_widgets[0]))
    canvas_item = _public(_canvas_rich_item_from_artifact(_canvas_artifact()))
    rich_items = [widget_item, canvas_item, _image_item()]
    return {
        "persona_used": "Be concise.",
        "provider": "google",
        "model": "gemini-3-pro",
        "key_source": "db",
        "config_source": "conversation",
        "reasoning_effort": "high",
        "context_window": {"input_tokens": 1200, "output_tokens": 300, "usage_source": "provider"},
        "usage_operation_id": "op-123",
        "usage_user_hash": "hash-abc",
        "conversation_id": SOURCE_CONVERSATION_ID,
        "thinking_summary": "Considered the user's categories.",
        "reasoning_summary": "Built a planner and a chart.",
        "finish_reason": "stop",
        "agent": {
            "id": "canvas_agent",
            "kind": "base",
            "name": "Canvas Agent",
            "custom_agent_id": None,
            "source": "response",
        },
        "custom_agent_warnings": ["Client tool 'fs_read' is unavailable on this device."],
        "workflow": {"graph_version": "v2", "execution_phase": "completed", "transitions": []},
        "tool_artifacts": tool_artifacts,
        "live_widgets": live_widgets,
        "canvas_artifact": _canvas_artifact(),
        "canvas_update": {
            "status": "updated",
            "artifact_id": "canvas:main",
            "base_revision": 1,
            "revision": 2,
        },
        "images": [
            {
                "image_id": CHAT_IMAGE_ID,
                "url": f"/chat-images/{CHAT_IMAGE_ID}",
                "mime": "image/png",
                "name": "generated.png",
            }
        ],
        "web_sources": [
            {"source_id": "src-1", "url": "https://example.com/budget", "title": "Budget guide"}
        ],
        "web_sources_version": 1,
        "rich_items_version": 1,
        "rich_items": rich_items,
        "rich_reference_warnings": [],
        "reply_to_user_message_id": SOURCE_USER_MESSAGE_ID,
        "suggested_questions": ["Can you add a savings goal?"],
        "next_task": {"id": "t-1", "description": "Add savings goal", "order": 1},
        "execution_paused": True,
        "execution_pause_reason": "max_tasks_reached",
        "execution_pause_message": "Completed a planning iteration. Send a message to continue.",
        "todos_synced": True,
        "some_future_key": {"anything": "unknown"},
    }


def _continued_reply_metadata() -> dict[str, Any]:
    """``MessageService._continuation_metadata`` plus the shared reply fields.

    ``partial`` stays true across a whole continuation chain, including the
    epoch that completed the answer.
    """
    return {
        "partial": True,
        "continued": True,
        "generation_id": "33333333-3333-3333-3333-333333333333",
        "logical_turn_id": SOURCE_USER_MESSAGE_ID,
        "execution_epoch": 2,
        "persona_used": None,
        "reply_to_user_message_id": SOURCE_USER_MESSAGE_ID,
    }


def _continuation_pause_metadata() -> dict[str, Any]:
    """``MessageService._apublish_continuation_pause`` output for a paused partial."""
    return {
        "partial": True,
        "continuable": True,
        "stop_reason": "execution_budget_exhausted",
        "generation_id": "33333333-3333-3333-3333-333333333333",
        "logical_turn_id": SOURCE_USER_MESSAGE_ID,
        "execution_epoch": 1,
        "persona_used": None,
        "execution_budget": {"tool_calls_used": 12, "tool_calls_limit": 12},
        "mutation_outcome_unknown": True,
    }


def _approval_metadata() -> dict[str, Any]:
    """``MessageService._interrupt_message_metadata`` for a pending approval."""
    tool_artifacts = [_widget_create_artifact()]
    return {
        "interrupt": {
            "interrupt_id": "int-1",
            "action_requests": [
                {"name": "send_email", "args": {"to": "a@example.com"}, "description": "Send"}
            ],
            "thread_id": f"{SOURCE_CONVERSATION_ID}:{SOURCE_USER_MESSAGE_ID}",
            "conversation_id": SOURCE_CONVERSATION_ID,
            "metadata": {
                "device_id": "55555555-5555-4555-8555-555555555555",
                "tool_provenance": {"call_1": {"device_id": "dev"}},
            },
        },
        "paused": True,
        "pause_reason": "tool_approval_required",
        "persona_used": "Be concise.",
        "thread_id": f"{SOURCE_CONVERSATION_ID}:{SOURCE_USER_MESSAGE_ID}",
        "next": ["tools"],
        "pending_tool_calls": [{"id": "call_1", "name": "send_email", "args": {}}],
        "tool_artifacts": tool_artifacts,
        "live_widgets": extract_live_widgets_from_artifacts(tool_artifacts),
    }


# ---------------------------------------------------------------------------
# Reply links
# ---------------------------------------------------------------------------


def test_reply_link_is_remapped_to_the_child_user_message():
    _, metadata = snapshot_fork_message(
        content=_rich_answer_content(),
        metadata=_rich_answer_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )

    assert metadata["reply_to_user_message_id"] == CHILD_USER_MESSAGE_ID


def test_reply_link_to_a_message_outside_the_copy_is_dropped():
    _, metadata = snapshot_fork_message(
        content="Answer",
        metadata={"reply_to_user_message_id": SOURCE_USER_MESSAGE_ID},
        message_id_map={},
    )

    assert "reply_to_user_message_id" not in metadata


def test_reply_link_matches_a_differently_cased_uuid():
    _, metadata = snapshot_fork_message(
        content="Answer",
        metadata={"reply_to_user_message_id": SOURCE_USER_MESSAGE_ID.upper()},
        message_id_map=MESSAGE_ID_MAP,
    )

    assert metadata["reply_to_user_message_id"] == CHILD_USER_MESSAGE_ID


# ---------------------------------------------------------------------------
# Display data survives unchanged
# ---------------------------------------------------------------------------


def test_canvas_image_and_static_rich_items_are_unchanged():
    source = _rich_answer_metadata()
    content, metadata = snapshot_fork_message(
        content=_rich_answer_content(),
        metadata=source,
        message_id_map=MESSAGE_ID_MAP,
    )

    assert metadata["canvas_artifact"] == source["canvas_artifact"]
    assert metadata["canvas_update"] == source["canvas_update"]
    assert metadata["images"] == source["images"]
    static_items = [item for item in source["rich_items"] if item["type"] != "live_widget"]
    assert metadata["rich_items"] == static_items
    assert metadata["rich_items_version"] == 1
    assert "<!--rich:img:web:1-->" in content
    assert "<!--rich:canvas:main-->" in content
    assert parse_inline_rich_references(content) == ["img:web:1", "canvas:main"]
    assert metadata["rich_reference_warnings"] == []


def test_display_attribution_is_preserved():
    source = _rich_answer_metadata()
    _, metadata = snapshot_fork_message(
        content=_rich_answer_content(), metadata=source, message_id_map=MESSAGE_ID_MAP
    )

    for key in (
        "persona_used",
        "provider",
        "model",
        "reasoning_effort",
        "context_window",
        "thinking_summary",
        "reasoning_summary",
        "agent",
        "web_sources",
        "web_sources_version",
        "suggested_questions",
    ):
        assert metadata[key] == source[key], key


def test_user_attachments_keep_their_image_references():
    attachments = [
        {
            "image_id": CHAT_IMAGE_ID,
            "url": f"/chat-images/{CHAT_IMAGE_ID}",
            "mime": "image/png",
            "name": "photo.png",
        }
    ]
    content, metadata = snapshot_fork_message(
        content="What is in this photo?",
        metadata={"attachments": attachments, "model_request": {"model": "gemini-3-pro"}},
        message_id_map=MESSAGE_ID_MAP,
    )

    assert content == "What is in this photo?"
    assert metadata["attachments"] == attachments
    assert "model_request" not in metadata


# ---------------------------------------------------------------------------
# Live widgets: descriptor and marker leave together
# ---------------------------------------------------------------------------


def test_live_widget_descriptor_and_its_marker_are_removed_together():
    content, metadata = snapshot_fork_message(
        content=_rich_answer_content(),
        metadata=_rich_answer_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )

    assert "live_widgets" not in metadata
    assert all(item["type"] != "live_widget" for item in metadata["rich_items"])
    assert f"widget:{WIDGET_ID}" not in parse_inline_rich_references(content)
    assert "<!--rich:widget:" not in content
    assert validate_rich_references(content, metadata["rich_items"]) == []
    # The marker's position now holds readable text naming what was there.
    label_line = content.split("\n")[2]
    assert "Budget planner" in label_line
    assert "<!--" not in label_line


def test_widget_tool_artifacts_are_removed_but_other_tool_activity_stays():
    source = _rich_answer_metadata()
    _, metadata = snapshot_fork_message(
        content=_rich_answer_content(), metadata=source, message_id_map=MESSAGE_ID_MAP
    )

    assert metadata["tool_artifacts"] == [_search_artifact()]
    assert WIDGET_ID not in json.dumps(metadata)


def test_a_malformed_widget_marker_is_still_removed():
    metadata = _rich_answer_metadata()
    content = f"Planner:\n<!--rich: widget:{WIDGET_ID} -->\nDone."

    new_content, _ = snapshot_fork_message(
        content=content, metadata=metadata, message_id_map=MESSAGE_ID_MAP
    )

    assert "<!--rich:" not in new_content
    assert "Budget planner" in new_content
    assert new_content.startswith("Planner:\n")
    assert new_content.endswith("\nDone.")


def test_a_marker_example_inside_a_code_fence_is_left_alone():
    metadata = _rich_answer_metadata()
    example = f"```\n<!--rich:widget:{WIDGET_ID}-->\n```"
    content = f"{example}\n\n<!--rich:widget:{WIDGET_ID}-->"

    new_content, _ = snapshot_fork_message(
        content=content, metadata=metadata, message_id_map=MESSAGE_ID_MAP
    )

    assert new_content.startswith(example)
    assert new_content.count(f"<!--rich:widget:{WIDGET_ID}-->") == 1


def test_an_appended_widget_without_a_marker_leaves_a_label():
    """``inline_or_append`` widgets render after the body when unreferenced."""
    metadata = _rich_answer_metadata()
    content = "Here is your planner."

    new_content, snapshot = snapshot_fork_message(
        content=content, metadata=metadata, message_id_map=MESSAGE_ID_MAP
    )

    assert new_content.startswith("Here is your planner.\n\n")
    assert new_content.count("Budget planner") == 1
    assert "live_widgets" not in snapshot


def test_legacy_live_widgets_metadata_is_removed_with_a_label():
    tool_artifacts = [_widget_create_artifact()]
    legacy = {
        "persona_used": None,
        "tool_artifacts": tool_artifacts,
        "live_widgets": extract_live_widgets_from_artifacts(tool_artifacts),
    }

    new_content, snapshot = snapshot_fork_message(
        content="Opened a planner.", metadata=legacy, message_id_map={}
    )

    assert "live_widgets" not in snapshot
    assert "tool_artifacts" not in snapshot
    assert new_content.startswith("Opened a planner.\n\n")
    assert "Budget planner" in new_content


# ---------------------------------------------------------------------------
# Runtime capabilities are removed
# ---------------------------------------------------------------------------


def test_runtime_and_task_controls_are_removed():
    _, metadata = snapshot_fork_message(
        content=_rich_answer_content(),
        metadata=_rich_answer_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )

    for key in (
        "next_task",
        "execution_paused",
        "execution_pause_reason",
        "execution_pause_message",
        "todos_synced",
        "usage_operation_id",
        "usage_user_hash",
        "conversation_id",
        "custom_agent_warnings",
        "key_source",
        "config_source",
        "workflow",
        "finish_reason",
    ):
        assert key not in metadata, key


def test_unknown_keys_are_not_copied():
    _, metadata = snapshot_fork_message(
        content="Answer",
        metadata={"some_future_key": {"anything": "unknown"}, "_internal": 1},
        message_id_map={},
    )

    assert metadata == {"fork_history": True}


def test_a_paused_continuation_keeps_its_text_but_cannot_be_continued():
    content, metadata = snapshot_fork_message(
        content="The first half of the answer.",
        metadata=_continuation_pause_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )

    assert content == "The first half of the answer."
    assert metadata["partial"] is True
    assert metadata["stop_reason"] == "execution_budget_exhausted"
    for key in (
        "continuable",
        "generation_id",
        "logical_turn_id",
        "execution_epoch",
        "execution_budget",
        "mutation_outcome_unknown",
    ):
        assert key not in metadata, key


def test_a_completed_continued_reply_keeps_partial_true():
    """Regression: completed continuation chains keep ``partial=True``.

    The snapshot must neither drop the display flag nor read completion into
    it; deciding eligibility is the repository's job.
    """
    content, metadata = snapshot_fork_message(
        content="The whole continued answer.",
        metadata=_continued_reply_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )

    assert content == "The whole continued answer."
    assert metadata["partial"] is True
    assert metadata["continued"] is True
    assert metadata["reply_to_user_message_id"] == CHILD_USER_MESSAGE_ID
    assert "fork_cutoff_eligible" not in metadata
    for key in ("generation_id", "logical_turn_id", "execution_epoch"):
        assert key not in metadata, key


def test_a_stopped_reply_keeps_its_stop_attribution():
    _, metadata = snapshot_fork_message(
        content="Partial answer",
        metadata={
            "stopped": True,
            "partial": True,
            "stop_reason": "user_requested",
            "reply_to_user_message_id": SOURCE_USER_MESSAGE_ID,
        },
        message_id_map=MESSAGE_ID_MAP,
    )

    assert metadata == {
        "stopped": True,
        "partial": True,
        "stop_reason": "user_requested",
        "reply_to_user_message_id": CHILD_USER_MESSAGE_ID,
        "fork_history": True,
    }


def test_an_approval_row_loses_every_resume_handle():
    content, metadata = snapshot_fork_message(
        content="I need approval to send the email.",
        metadata=_approval_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )

    for key in ("interrupt", "thread_id", "next", "pending_tool_calls", "live_widgets"):
        assert key not in metadata, key
    # Still hidden exactly as in the source; nothing left to approve.
    assert metadata["paused"] is True
    assert metadata["pause_reason"] == "tool_approval_required"
    assert "tool_artifacts" not in metadata
    # A hidden row gains no label text: that would turn an empty placeholder
    # into a visible transcript turn.
    assert content == "I need approval to send the email."


def test_an_empty_approval_placeholder_stays_empty():
    content, metadata = snapshot_fork_message(
        content="", metadata=_approval_metadata(), message_id_map=MESSAGE_ID_MAP
    )

    assert content == ""
    assert metadata["paused"] is True


# ---------------------------------------------------------------------------
# Fork markers
# ---------------------------------------------------------------------------


def test_every_snapshot_is_marked_as_fork_history():
    _, metadata = snapshot_fork_message(content="Hi", metadata=None, message_id_map={})

    assert metadata == {"fork_history": True}


def test_cutoff_eligibility_from_an_earlier_fork_is_preserved():
    _, metadata = snapshot_fork_message(
        content="Answer",
        metadata={"fork_history": True, "fork_cutoff_eligible": False, "partial": True},
        message_id_map={},
    )

    assert metadata["fork_cutoff_eligible"] is False
    assert metadata["fork_history"] is True


@pytest.mark.parametrize("value", ["true", 1, None])
def test_a_non_boolean_cutoff_eligibility_is_not_copied(value):
    _, metadata = snapshot_fork_message(
        content="Answer", metadata={"fork_cutoff_eligible": value}, message_id_map={}
    )

    assert "fork_cutoff_eligible" not in metadata


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_inputs_are_never_mutated():
    source = _rich_answer_metadata()
    id_map = dict(MESSAGE_ID_MAP)
    frozen_source = copy.deepcopy(source)
    content = _rich_answer_content()

    snapshot_fork_message(content=content, metadata=source, message_id_map=id_map)

    assert source == frozen_source
    assert id_map == MESSAGE_ID_MAP


def test_output_shares_no_mutable_state_with_the_input():
    source = _rich_answer_metadata()
    _, metadata = snapshot_fork_message(
        content=_rich_answer_content(), metadata=source, message_id_map=MESSAGE_ID_MAP
    )

    metadata["canvas_artifact"]["content"] = "changed"
    metadata["rich_items"][0]["payload"]["content"] = "changed"
    metadata["images"][0]["url"] = "changed"

    assert source["canvas_artifact"]["content"] != "changed"
    assert all(item["payload"].get("content") != "changed" for item in source["rich_items"])
    assert source["images"][0]["url"] != "changed"


def test_the_snapshot_is_deterministic():
    first = snapshot_fork_message(
        content=_rich_answer_content(),
        metadata=_rich_answer_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )
    second = snapshot_fork_message(
        content=_rich_answer_content(),
        metadata=_rich_answer_metadata(),
        message_id_map=MESSAGE_ID_MAP,
    )

    assert first == second
    json.dumps(first[1])  # still JSON-serializable for the JSONB column


def test_snapshot_ids_come_only_from_the_supplied_map():
    other_child = str(uuid.uuid4())
    _, metadata = snapshot_fork_message(
        content="Answer",
        metadata={"reply_to_user_message_id": SOURCE_USER_MESSAGE_ID},
        message_id_map={SOURCE_USER_MESSAGE_ID: other_child},
    )

    assert metadata["reply_to_user_message_id"] == other_child


# ---------------------------------------------------------------------------
# Renderer guards: copied rows never offer controls
# ---------------------------------------------------------------------------


def test_rich_view_never_offers_a_live_widget_on_fork_history():
    from app.ui.rich_response import build_rich_response_view

    source = _rich_answer_metadata()
    # Deliberately unstripped: the guard must hold even if a widget descriptor
    # reached a copied row by some other path.
    metadata = {**source, "fork_history": True}

    view = build_rich_response_view(_rich_answer_content(), metadata)

    rendered = [segment.item for segment in view.segments if segment.kind == "rich"]
    assert all(item["type"] != "live_widget" for item in rendered)
    assert all(item["type"] != "live_widget" for item in view.append_items)
    unavailable = [segment.item_id for segment in view.segments if segment.kind == "unavailable"]
    assert unavailable == [f"widget:{WIDGET_ID}"]


def test_rich_view_still_offers_live_widgets_on_ordinary_rows():
    from app.ui.rich_response import build_rich_response_view

    view = build_rich_response_view(_rich_answer_content(), _rich_answer_metadata())

    rendered = [segment.item for segment in view.segments if segment.kind == "rich"]
    assert any(item["type"] == "live_widget" for item in rendered)


def test_interrupt_recovery_ignores_fork_history():
    from app.ui.hitl_recovery import recoverable_interrupt

    metadata = {**_approval_metadata(), "fork_history": True}

    assert recoverable_interrupt(metadata) is None


def test_interrupt_recovery_restores_an_ordinary_paused_approval():
    from app.ui.hitl_recovery import recoverable_interrupt

    metadata = _approval_metadata()
    metadata["interrupt"].pop("thread_id")

    recovered = recoverable_interrupt(metadata)

    assert recovered is not None
    assert recovered["interrupt_id"] == "int-1"
    # thread_id stored beside the interrupt is folded back in.
    assert recovered["thread_id"] == metadata["thread_id"]
    assert "thread_id" not in metadata["interrupt"]


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"paused": True, "pause_reason": "max_tasks_reached", "interrupt": {"interrupt_id": "x"}},
        {"paused": True, "pause_reason": "tool_approval_required", "interrupt": "raw"},
        {"pause_reason": "tool_approval_required", "interrupt": {"interrupt_id": "x"}},
    ],
)
def test_interrupt_recovery_requires_a_paused_approval(metadata):
    from app.ui.hitl_recovery import recoverable_interrupt

    assert recoverable_interrupt(metadata) is None


def test_demo_does_not_mount_live_widgets_on_fork_history(monkeypatch):
    from tests.test_demo_generation_controls import _import_demo

    demo, streamlit_stub = _import_demo(monkeypatch)
    streamlit_stub.session_state.live_widget_mounts = {}
    mounted: list[Any] = []
    monkeypatch.setattr(demo._stc, "html", lambda *args, **kwargs: mounted.append(args))
    tool_artifacts = [_widget_create_artifact()]
    metadata = {
        "live_widgets": extract_live_widgets_from_artifacts(tool_artifacts),
        "fork_history": True,
    }

    demo.render_live_widgets(metadata, message_key="m1", auto_mount=True, inline=True)

    assert mounted == []
    assert streamlit_stub.session_state.live_widget_mounts == {}
    assert streamlit_stub.buttons == []


def test_demo_still_mounts_live_widgets_on_ordinary_rows(monkeypatch):
    from tests.test_demo_generation_controls import _import_demo

    demo, streamlit_stub = _import_demo(monkeypatch)
    streamlit_stub.session_state.live_widget_mounts = {}
    mounted: list[Any] = []
    monkeypatch.setattr(demo._stc, "html", lambda *args, **kwargs: mounted.append(args))
    tool_artifacts = [_widget_create_artifact()]
    metadata = {"live_widgets": extract_live_widgets_from_artifacts(tool_artifacts)}

    demo.render_live_widgets(metadata, message_key="m1", auto_mount=True, inline=True)

    assert len(mounted) == 1
