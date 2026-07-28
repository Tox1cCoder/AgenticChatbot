from __future__ import annotations

import inspect

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


def test_refresh_conversations_list_has_no_unused_fallback_parameter(monkeypatch) -> None:
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)

    assert tuple(inspect.signature(demo.refresh_conversations_list).parameters) == ()


def test_fetch_all_conversations_preserves_page_order_and_normalizes_meta(monkeypatch) -> None:
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    responses = iter(
        [
            {
                "success": True,
                "message": "page one",
                "data": {
                    "items": [{"id": "one"}, {"id": "two"}],
                    "meta": {
                        "total": 3,
                        "perPage": 2,
                        "currentPage": 1,
                        "lastPage": 2,
                    },
                },
            },
            {
                "success": True,
                "message": "page two",
                "data": {
                    "items": [{"id": "three"}],
                    "meta": {
                        "total": 3,
                        "perPage": 2,
                        "currentPage": 2,
                        "lastPage": 2,
                    },
                },
            },
        ]
    )
    endpoints: list[str] = []

    def request(_method: str, endpoint: str):
        endpoints.append(endpoint)
        return next(responses)

    monkeypatch.setattr(demo, "make_api_request", request)

    result = demo.get_conversations(limit=2, fetch_all_pages=True)

    assert endpoints == [
        "/conversations/?page=1&limit=2",
        "/conversations/?page=2&limit=2",
    ]
    assert result["data"]["items"] == [
        {"id": "one"},
        {"id": "two"},
        {"id": "three"},
    ]
    assert result["data"]["meta"] == {
        "total": 3,
        "perPage": 3,
        "currentPage": 1,
        "lastPage": 1,
    }


def test_persisted_custom_agent_label_compatibility_remains_visible(monkeypatch) -> None:
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)

    assert demo.get_message_agent_label({"custom_agent_name": "Historical Agent"}) == (
        "Historical Agent"
    )
