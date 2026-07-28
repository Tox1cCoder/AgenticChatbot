from __future__ import annotations

from typing import Any

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


def _conversation_response(
    items: list[dict[str, Any]],
    *,
    page: int = 1,
    last_page: int = 1,
    total: int | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "message": "ok",
        "data": {
            "items": items,
            "meta": {
                "total": len(items) if total is None else total,
                "perPage": 100,
                "currentPage": page,
                "lastPage": last_page,
            },
        },
    }


def test_get_conversations_encodes_search_and_preview_params(monkeypatch) -> None:
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    endpoints: list[str] = []
    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda _method, endpoint, *_args: (
            endpoints.append(endpoint) or _conversation_response([{"id": "matched"}])
        ),
    )

    demo.get_conversations(
        page=1,
        limit=100,
        include_messages=True,
        latest_messages=3,
        search="Q3 roadmap & budget",
    )

    assert endpoints == [
        "/conversations/?page=1&limit=100&include=messages&latestMessages=3&search=Q3+roadmap+%26+budget"
    ]


def test_get_conversations_preserves_valid_empty_search_page(monkeypatch) -> None:
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)
    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda *_args, **_kwargs: _conversation_response([], total=0),
    )

    response = demo.get_conversations(search="no matches")

    assert response["success"] is True
    assert response["data"]["items"] == []
    assert response["data"]["meta"]["total"] == 0


def test_manager_search_state_is_separate_from_unfiltered_cache(monkeypatch) -> None:
    demo, streamlit = _import_demo_with_ui_stubs(monkeypatch)
    streamlit.session_state.manager_conversations = [{"id": "cached"}]
    demo._reset_manager_search_state("roadmap")
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        demo,
        "get_conversations",
        lambda **kwargs: calls.append(kwargs) or _conversation_response([{"id": "matched"}]),
    )

    demo._load_manager_page(1, search="roadmap")

    assert calls == [
        {
            "page": 1,
            "limit": 100,
            "include_messages": True,
            "latest_messages": 3,
            "fetch_all_pages": False,
            "search": "roadmap",
        }
    ]
    assert streamlit.session_state.manager_conversations == [{"id": "cached"}]
    assert streamlit.session_state.manager_search_conversations == [{"id": "matched"}]


def test_failed_search_keeps_unfiltered_cache_and_remains_retryable(monkeypatch) -> None:
    demo, streamlit = _import_demo_with_ui_stubs(monkeypatch)
    streamlit.session_state.manager_conversations = [{"id": "cached"}]
    demo._reset_manager_search_state("roadmap")
    monkeypatch.setattr(demo, "get_conversations", lambda **_kwargs: {})

    demo._load_manager_page(1, search="roadmap")

    assert streamlit.session_state.manager_conversations == [{"id": "cached"}]
    assert streamlit.session_state.manager_search_page == 0
    assert streamlit.session_state.manager_search_has_more is False


def test_reset_manager_search_state_changes_only_search_cache(monkeypatch) -> None:
    demo, streamlit = _import_demo_with_ui_stubs(monkeypatch)
    streamlit.session_state.manager_conversations = [{"id": "cached"}]
    streamlit.session_state.manager_conv_page = 4

    demo._reset_manager_search_state("beta")

    assert streamlit.session_state.manager_conversations == [{"id": "cached"}]
    assert streamlit.session_state.manager_conv_page == 4
    assert streamlit.session_state.manager_search_query == "beta"
    assert streamlit.session_state.manager_search_conversations == []
    assert streamlit.session_state.manager_search_page == 0
    assert streamlit.session_state.manager_search_total == 0


def test_deduplicate_conversations_keeps_first_seen_order(monkeypatch) -> None:
    demo, _streamlit = _import_demo_with_ui_stubs(monkeypatch)

    result = demo._deduplicate_conversations(
        [
            {"id": "one", "title": "first"},
            {"id": "two", "title": "second"},
            {"id": "one", "title": "replacement"},
            {"title": "missing id"},
        ]
    )

    assert result == [
        {"id": "one", "title": "first"},
        {"id": "two", "title": "second"},
    ]


def test_later_search_pages_merge_without_duplicate_rows(monkeypatch) -> None:
    demo, streamlit = _import_demo_with_ui_stubs(monkeypatch)
    demo._reset_manager_search_state("roadmap")
    streamlit.session_state.manager_search_conversations = [{"id": "one", "title": "first"}]
    monkeypatch.setattr(
        demo,
        "get_conversations",
        lambda **_kwargs: _conversation_response(
            [
                {"id": "one", "title": "duplicate"},
                {"id": "two", "title": "second"},
            ],
            page=2,
            last_page=2,
            total=2,
        ),
    )

    demo._load_manager_page(2, search="roadmap")

    assert streamlit.session_state.manager_search_conversations == [
        {"id": "one", "title": "first"},
        {"id": "two", "title": "second"},
    ]
