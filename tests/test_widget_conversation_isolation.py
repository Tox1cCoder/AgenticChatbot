"""A widget belongs to the conversation that created it.

The HTTP API revalidates that with ``user_owns_conversation``. The model-facing
MCP tools address widgets by bare ``widget_id`` and, until 2026-09-22, did not
revalidate anything: ``widget_get_state`` returned another conversation's full
HTML, and ``widget_update``/``widget_close`` rewrote and closed it.
"""

from __future__ import annotations

import pytest

from app.ai.tool_execution import _widget_access_denied
from app.services import widget_runtime
from app.services.widget_runtime import InMemoryWidgetStore

CONVERSATION_A = "11111111-1111-4111-8111-111111111111"
CONVERSATION_B = "22222222-2222-4222-8222-222222222222"

SECRET_HTML = "<!DOCTYPE html><html><body><h1>A's private widget</h1></body></html>"

WIDGET_ID_TOOLS = ("widget_get_state", "widget_update", "widget_close")


@pytest.fixture
def store(monkeypatch) -> InMemoryWidgetStore:
    created = InMemoryWidgetStore()
    monkeypatch.setattr(widget_runtime, "_widget_store", created, raising=False)
    return created


async def _widget_in_a(store: InMemoryWidgetStore) -> str:
    record = await store.create(
        session_id=CONVERSATION_A,
        initial_state={"html": SECRET_HTML, "height": 400},
        title="A's widget",
    )
    return record.widget_id


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", WIDGET_ID_TOOLS)
async def test_another_conversations_widget_is_refused(store, tool_name: str) -> None:
    widget_id = await _widget_in_a(store)

    denied = await _widget_access_denied(
        tool_name, {"widget_id": widget_id}, CONVERSATION_B
    )

    assert denied is not None
    assert widget_id in denied
    assert SECRET_HTML not in denied


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", WIDGET_ID_TOOLS)
async def test_the_owning_conversation_still_reaches_its_own_widget(store, tool_name: str) -> None:
    widget_id = await _widget_in_a(store)

    assert await _widget_access_denied(tool_name, {"widget_id": widget_id}, CONVERSATION_A) is None


@pytest.mark.asyncio
async def test_a_missing_widget_keeps_the_tool_s_own_not_found_error(store) -> None:
    """Absent is not foreign. The tool reports "not found" as it always did."""

    assert (
        await _widget_access_denied(
            "widget_get_state",
            {"widget_id": "33333333-3333-4333-8333-333333333333"},
            CONVERSATION_B,
        )
        is None
    )


@pytest.mark.asyncio
async def test_tools_not_addressed_by_widget_id_are_untouched(store) -> None:
    widget_id = await _widget_in_a(store)

    for tool_name in ("widget_create", "session_list_widgets", "web_search"):
        denied = await _widget_access_denied(
            tool_name, {"widget_id": widget_id}, CONVERSATION_B
        )
        assert denied is None


@pytest.mark.asyncio
async def test_no_conversation_context_does_not_invent_an_allow(store) -> None:
    """Without a conversation there is nothing to compare against.

    The call is left alone rather than denied: refusing here would break
    in-process callers that never carry a conversation, and the tools were
    already reachable in that shape before this check existed.
    """

    widget_id = await _widget_in_a(store)

    assert await _widget_access_denied("widget_get_state", {"widget_id": widget_id}, None) is None


# ---------------------------------------------------------------------------
# Re-displaying a widget the conversation already owns
# ---------------------------------------------------------------------------
# Reading a widget produced no rich-item candidate, so there was no
# `<!--rich:widget:...-->` marker to place and the model answered by pasting the
# HTML it had just read. That happened inside one conversation too, not only
# across them, because both tool-name sets listed create/update and nothing else.


def _widget_record_json(widget_id: str = "w-abc") -> str:
    import json

    return json.dumps(
        {
            "widget_id": widget_id,
            "session_id": CONVERSATION_A,
            "title": "A's widget",
            "state": {"html": SECRET_HTML, "height": 400},
            "status": "active",
            "version": 2,
        }
    )


def test_reading_a_widget_yields_a_placeable_rich_item() -> None:
    from app.ai.tool_execution import build_live_widget_candidate_from_tool_result

    candidate = build_live_widget_candidate_from_tool_result(
        _widget_record_json(), tool_name="widget_get_state"
    )

    assert candidate is not None
    assert candidate["id"] == "widget:w-abc"
    assert candidate["type"] == "live_widget"
    assert candidate["payload"]["widget_id"] == "w-abc"
    # The mount record is public metadata; widget state never travels in it.
    assert SECRET_HTML not in str(candidate)


def test_reading_a_widget_yields_frontend_mount_metadata() -> None:
    from app.core.response_constants import extract_live_widgets_from_artifacts

    widgets = extract_live_widgets_from_artifacts(
        [
            {
                "tool_call_id": "tc-1",
                "tool": "widget_get_state",
                "args": {},
                "output": _widget_record_json(),
                "error": None,
                "status": "success",
            }
        ]
    )

    assert [widget["widget_id"] for widget in widgets] == ["w-abc"]
    assert widgets[0]["connection_endpoint"] == "/widgets/w-abc/connection"


def test_a_missing_widget_read_yields_no_rich_item() -> None:
    import json

    from app.ai.tool_execution import build_live_widget_candidate_from_tool_result

    assert (
        build_live_widget_candidate_from_tool_result(
            json.dumps({"error": "Widget w-gone not found"}), tool_name="widget_get_state"
        )
        is None
    )
