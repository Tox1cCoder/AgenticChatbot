from __future__ import annotations

import importlib
import re
import sys
import types
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pytest
from packaging.version import Version


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _CacheDecorator:
    def __init__(self) -> None:
        self.decorator_kwargs: list[dict[str, Any]] = []
        self.clear_calls = 0

    def __call__(self, *args: Any, **kwargs: Any):
        self.decorator_kwargs.append(kwargs)

        def decorate(func):
            if kwargs.get("ttl") != 30:
                func.clear = self.clear
                return func

            cache: dict[tuple[Any, ...], Any] = {}

            @wraps(func)
            def cached(*func_args: Any, **func_kwargs: Any):
                key = (*func_args, *sorted(func_kwargs.items()))
                if key not in cache:
                    cache[key] = func(*func_args, **func_kwargs)
                return cache[key]

            def clear() -> None:
                cache.clear()
                self.clear()

            cached.clear = clear
            return cached

        return decorate

    def clear(self) -> None:
        self.clear_calls += 1


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


class _TabContext(_Context):
    def __init__(self, *, is_open: bool) -> None:
        self.open = is_open


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.query_params: dict[str, str] = {}
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()
        self.sidebar = _Context()
        self.markdown_calls: list[tuple[str, dict[str, Any]]] = []
        self.tabs_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.toast_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, body: str, *args: Any, **kwargs: Any) -> None:
        self.markdown_calls.append((body, kwargs))

    def toast(self, *args: Any, **kwargs: Any) -> None:
        self.toast_calls.append((args, kwargs))

    def expander(self, *args: Any, **kwargs: Any) -> _Context:
        return _Context()

    def caption(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def tabs(self, labels: list[str], *args: Any, **kwargs: Any) -> list[_TabContext]:
        self.tabs_calls.append((labels, kwargs))
        key = kwargs.get("key")
        selected = self.session_state.get(key, labels[0]) if key else labels[0]
        return [_TabContext(is_open=label == selected) for label in labels]

    def __getattr__(self, _name: str):
        def _noop(*_args: Any, **_kwargs: Any):
            return None

        return _noop


def _import_demo(monkeypatch: pytest.MonkeyPatch):
    streamlit_stub = _StreamlitStub()
    components_module = types.ModuleType("streamlit.components")
    components_v1_module = types.ModuleType("streamlit.components.v1")
    components_v1_module.html = lambda *args, **kwargs: None
    components_v1_module.declare_component = lambda *args, **kwargs: (
        lambda **component_kwargs: component_kwargs.get("default")
    )
    components_module.v1 = components_v1_module
    streamlit_stub.components = components_module
    markdown_stub = types.ModuleType("markdown")
    markdown_stub.markdown = lambda text, **_kwargs: text

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "streamlit.components", components_module)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", components_v1_module)
    monkeypatch.setitem(sys.modules, "markdown", markdown_stub)
    sys.modules.pop("demo", None)
    return importlib.import_module("demo"), streamlit_stub


def test_day_boundaries_use_local_midnight_and_exclusive_next_day(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    zone = ZoneInfo("America/New_York")

    start = demo.align_usage_boundary(date(2026, 3, 8), bucket="day", zone=zone)
    end = demo.align_usage_boundary(date(2026, 3, 8), bucket="day", zone=zone, exclusive_end=True)

    assert start.isoformat() == "2026-03-08T00:00:00-05:00"
    assert end.isoformat() == "2026-03-09T00:00:00-04:00"


def test_hour_boundary_truncates_minutes_in_requested_zone(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    zone = ZoneInfo("Asia/Kathmandu")
    value = datetime(2026, 7, 21, 9, 47, 31, tzinfo=ZoneInfo("UTC"))

    aligned = demo.align_usage_boundary(value, bucket="hour", zone=zone)

    assert aligned.isoformat() == "2026-07-21T15:00:00+05:45"


def test_hour_bucket_date_picker_end_includes_the_selected_local_day(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    zone = ZoneInfo("Australia/Lord_Howe")

    end = demo.align_usage_boundary(date(2026, 10, 4), bucket="hour", zone=zone, exclusive_end=True)

    assert end.isoformat() == "2026-10-05T00:00:00+11:00"


@pytest.mark.parametrize(
    ("zone_name", "selected_date", "expected"),
    [
        ("America/New_York", date(2026, 3, 8), "2026-03-09T00:00:00-04:00"),
        ("Australia/Lord_Howe", date(2026, 10, 4), "2026-10-05T00:00:00+11:00"),
    ],
)
def test_render_hour_boundaries_advance_inclusive_midnight_end_across_dst(
    monkeypatch, zone_name, selected_date, expected
):
    demo, _ = _import_demo(monkeypatch)

    _, end = demo._build_usage_query_boundaries(
        start_date=selected_date,
        end_date=selected_date,
        bucket="hour",
        zone=ZoneInfo(zone_name),
        start_hour=datetime.min.time(),
        end_hour=datetime.min.time(),
    )

    assert end.isoformat() == expected


def test_render_hour_boundaries_do_not_advance_non_midnight_explicit_end(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    zone = ZoneInfo("America/New_York")

    _, end = demo._build_usage_query_boundaries(
        start_date=date(2026, 3, 8),
        end_date=date(2026, 3, 8),
        bucket="hour",
        zone=zone,
        start_hour=datetime.min.time(),
        end_hour=datetime(2026, 1, 1, 23).time(),
    )

    assert end.isoformat() == "2026-03-08T23:00:00-04:00"


def test_nonexistent_new_york_hour_has_no_candidate(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    candidates = demo._local_usage_hour_candidates(
        date(2026, 3, 8), datetime(2026, 1, 1, 2).time(), ZoneInfo("America/New_York")
    )

    assert candidates == ()


@pytest.mark.parametrize(
    "value",
    [datetime(2026, 3, 8, 2), datetime(2026, 11, 1, 1)],
)
def test_align_rejects_unvalidated_naive_gap_or_fold(monkeypatch, value):
    demo, _ = _import_demo(monkeypatch)

    with pytest.raises(ValueError, match="explicit"):
        demo.align_usage_boundary(value, bucket="hour", zone=ZoneInfo("America/New_York"))


def test_ambiguous_new_york_hour_exposes_distinct_fold_instants(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    zone = ZoneInfo("America/New_York")

    first = demo._build_usage_query_boundaries(
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 1),
        bucket="hour",
        zone=zone,
        start_hour=datetime(2026, 1, 1, 1).time(),
        end_hour=datetime(2026, 1, 1, 2).time(),
        start_occurrence="first",
    )[0]
    second = demo._build_usage_query_boundaries(
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 1),
        bucket="hour",
        zone=zone,
        start_hour=datetime(2026, 1, 1, 1).time(),
        end_hour=datetime(2026, 1, 1, 2).time(),
        start_occurrence="second",
    )[0]

    assert first.isoformat() == "2026-11-01T01:00:00-04:00"
    assert second.isoformat() == "2026-11-01T01:00:00-05:00"
    assert first.astimezone(ZoneInfo("UTC")) != second.astimezone(ZoneInfo("UTC"))


def test_lord_howe_half_hour_fold_candidates_are_distinct(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    candidates = demo._local_usage_hour_candidates(
        date(2026, 4, 5), datetime(2026, 1, 1, 1, 30).time(), ZoneInfo("Australia/Lord_Howe")
    )

    assert [candidate.isoformat() for candidate in candidates] == [
        "2026-04-05T01:30:00+11:00",
        "2026-04-05T01:30:00+10:30",
    ]


def test_gap_validation_prevents_dashboard_request(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(demo, "get_usage_dashboard", lambda **kwargs: calls.append(kwargs))

    data, error = demo._request_usage_dashboard_for_filters(
        start_date=date(2026, 3, 8),
        end_date=date(2026, 3, 8),
        bucket="hour",
        zone=ZoneInfo("America/New_York"),
        start_hour=datetime(2026, 1, 1, 2).time(),
        end_hour=datetime(2026, 1, 1, 3).time(),
        start_occurrence="first",
        end_occurrence="first",
        conversation_id=None,
    )

    assert data is None
    assert "does not exist" in error
    assert calls == []


def test_dashboard_query_is_encoded_and_has_no_user_identity(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.auth_token = "secret-token"
    stub.session_state.current_user_id = "user-a"
    calls: list[tuple[str, str, bool]] = []

    def fake_usage_get(endpoint: str, *, auth_identity: str, cache_version: int):
        calls.append((endpoint, auth_identity, bool(cache_version)))
        return {
            "status_code": 200,
            "payload": {"success": True, "data": {"totals": {}}},
        }

    monkeypatch.setattr(demo, "_cached_usage_get_request", fake_usage_get)
    response = demo.get_usage_dashboard(
        start=date(2026, 7, 1),
        end=date(2026, 7, 2),
        bucket="day",
        timezone_name="Asia/Kathmandu",
        conversation_id="8b93ef00-9b1c-4c25-a605-e588d70f8ae0",
    )

    endpoint, auth_identity, _ = calls[0]
    query = parse_qs(urlsplit(endpoint).query)
    assert urlsplit(endpoint).path == "/usage/dashboard"
    assert query == {
        "from": ["2026-07-01T00:00:00+05:45"],
        "to": ["2026-07-03T00:00:00+05:45"],
        "bucket": ["day"],
        "timezone": ["Asia/Kathmandu"],
        "conversationId": ["8b93ef00-9b1c-4c25-a605-e588d70f8ae0"],
    }
    assert "user" not in endpoint.lower()
    assert auth_identity.startswith("user-a:")
    assert "secret-token" not in auth_identity
    assert response["totals"] == demo._empty_usage_totals()


def test_usage_cache_is_dedicated_30_second_authenticated_cache(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    usage_decorator = next(
        kwargs for kwargs in stub.cache_data.decorator_kwargs if kwargs.get("ttl") == 30
    )
    assert usage_decorator["max_entries"] > 0

    stub.session_state.auth_token = "tenant-a-token"
    stub.session_state.current_user_id = "tenant-a"
    first = demo._usage_cache_identity()
    stub.session_state.auth_token = "tenant-b-token"
    stub.session_state.current_user_id = "tenant-b"
    second = demo._usage_cache_identity()

    assert first != second
    assert "token" not in first and "token" not in second


def test_usage_cache_miss_sends_authenticated_get_and_returns_response_envelope(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.auth_token = "active-auth-token"
    calls: list[tuple[str, dict[str, Any]]] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"success": True, "data": {}}

    class Session:
        @staticmethod
        def get(url: str, **kwargs: Any):
            calls.append((url, kwargs))
            return Response()

    monkeypatch.setattr(demo, "get_http_session", lambda: Session())
    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda *_args, **_kwargs: pytest.fail("cached usage reads must not emit UI elements"),
    )
    response = demo._cached_usage_get_request(
        "/usage/dashboard", auth_identity="safe-partition", cache_version=0
    )

    assert response == {
        "status_code": 200,
        "payload": {"success": True, "data": {}},
    }
    assert calls == [
        (
            f"{demo.API_BASE_URL}/usage/dashboard",
            {
                "headers": {"Authorization": "Bearer active-auth-token"},
                "timeout": demo.REQUEST_TIMEOUT,
            },
        )
    ]
    assert stub.toast_calls == []


def test_usage_cache_hit_reuses_silent_transport_envelope(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.auth_token = "active-auth-token"
    stub.session_state.current_user_id = "user-a"
    calls: list[str] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"success": True, "data": {"enabled": True}}

    class Session:
        @staticmethod
        def get(url: str, **_kwargs: Any):
            calls.append(url)
            return Response()

    monkeypatch.setattr(demo, "get_http_session", lambda: Session())

    first = demo._usage_get("/usage/capabilities")
    second = demo._usage_get("/usage/capabilities")

    assert first == second == {"success": True, "data": {"enabled": True}}
    assert calls == [f"{demo.API_BASE_URL}/usage/capabilities"]
    assert stub.toast_calls == []


def test_usage_cache_partitions_same_tenant_when_token_changes(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.current_user_id = "user-a"
    calls: list[str] = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"success": True, "data": {}}

    class Session:
        @staticmethod
        def get(url: str, **_kwargs: Any):
            calls.append(url)
            return Response()

    monkeypatch.setattr(demo, "get_http_session", lambda: Session())

    stub.session_state.auth_token = "first-token"
    demo._usage_get("/usage/dashboard")
    stub.session_state.auth_token = "second-token"
    demo._usage_get("/usage/dashboard")

    assert calls == [
        f"{demo.API_BASE_URL}/usage/dashboard",
        f"{demo.API_BASE_URL}/usage/dashboard",
    ]


def test_successful_usage_read_clears_previous_api_error(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state["_last_api_error_message"] = "stale failure"
    monkeypatch.setattr(
        demo,
        "_cached_usage_get_request",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": {"success": True, "data": {}},
        },
    )

    response = demo._usage_get("/usage/dashboard")

    assert response == {"success": True, "data": {}}
    assert stub.session_state["_last_api_error_message"] is None


def test_usage_http_error_is_parsed_and_toasted_outside_cached_function(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.auth_token = "active-auth-token"
    stub.session_state.current_user_id = "user-a"
    calls: list[str] = []

    class Response:
        status_code = 503

        @staticmethod
        def json():
            return {"detail": "Usage service unavailable"}

    class Session:
        @staticmethod
        def get(url: str, **_kwargs: Any):
            calls.append(url)
            return Response()

    monkeypatch.setattr(demo, "get_http_session", lambda: Session())
    identity = demo._usage_cache_identity()

    cached = demo._cached_usage_get_request(
        "/usage/dashboard", auth_identity=identity, cache_version=0
    )
    assert cached == {
        "status_code": 503,
        "payload": {"detail": "Usage service unavailable"},
    }
    assert stub.toast_calls == []

    response = demo._usage_get("/usage/dashboard")

    assert response == {}
    assert calls == [f"{demo.API_BASE_URL}/usage/dashboard"]
    assert stub.session_state["_last_api_error_message"] == "Usage service unavailable"
    assert stub.toast_calls == [(("Usage service unavailable",), {"icon": ":material/cancel:"})]


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (401, {"detail": "Token expired"}),
        (200, {"success": False, "code": "unauthenticated", "message": "Token expired"}),
    ],
)
def test_usage_auth_expiry_transitions_to_login_outside_cache(monkeypatch, status_code, payload):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.auth_token = "expired-token"
    stub.session_state.current_user_id = "user-a"
    stub.session_state.current_user_profile = {"id": "user-a"}
    stub.session_state.show_login = False
    monkeypatch.setattr(
        demo,
        "_cached_usage_get_request",
        lambda *_args, **_kwargs: {"status_code": status_code, "payload": payload},
    )

    response = demo._usage_get("/usage/dashboard")

    assert response == {}
    assert stub.session_state.auth_token is None
    assert stub.session_state.current_user_id is None
    assert stub.session_state.current_user_profile is None
    assert stub.session_state.show_login is True
    assert stub.toast_calls == [(("Please log in",), {"icon": ":material/lock:"})]


def test_usage_malformed_success_response_reports_unexpected_response(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    monkeypatch.setattr(
        demo,
        "_cached_usage_get_request",
        lambda *_args, **_kwargs: {"status_code": 200, "payload": {}},
    )

    response = demo._usage_get("/usage/dashboard")

    assert response == {}
    assert stub.session_state["_last_api_error_message"] == "Unexpected response from API"
    assert stub.toast_calls == [(("Unexpected response from API",), {"icon": ":material/cancel:"})]


def test_usage_capability_defaults_to_hidden_on_false_or_unavailable(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    responses = iter(
        [
            {"success": True, "data": {"enabled": True}},
            {"success": True, "data": {"enabled": False}},
            {},
        ]
    )
    monkeypatch.setattr(demo, "_usage_get", lambda _endpoint: next(responses))

    assert demo.get_usage_capability_enabled() is True
    assert demo.get_usage_capability_enabled() is False
    assert demo.get_usage_capability_enabled() is False


def test_usage_response_normalization_accepts_empty_and_ignores_unknown_fields(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    assert demo._normalize_usage_dashboard({}) is None
    normalized = demo._normalize_usage_dashboard(
        {
            "success": True,
            "data": {
                "totals": {"requestCount": 0},
                "series": [],
                "futureField": {"safe": True},
            },
        }
    )

    assert normalized is not None
    assert normalized["totals"] == demo._empty_usage_totals()
    assert normalized["series"] == []
    assert "futureField" not in normalized


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "?"), (0, "0"), (999, "999"), (2_000, "2k"), (2_200, "2.2k"), (2_000_000, "2M")],
)
def test_compact_token_formatting(monkeypatch, value, expected):
    demo, _ = _import_demo(monkeypatch)
    assert demo._format_tokens(value) == expected


def test_shared_context_tooltip_and_uncapped_ratio(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    presentation = demo._context_window_presentation(
        {
            "known": True,
            "limit_type": "shared_context",
            "context_window_tokens": 65_536,
            "input_tokens": 120,
            "output_tokens": 2_000,
            "used_tokens": 2_200,
            "usage_ratio": 1.25,
            "usage_source": "provider_reported",
            "display_state": "danger",
        }
    )

    assert presentation["tooltip"] == (
        "2.2k / 65.5k (125%) · input 120 · output 2.0k · provider reported"
    )
    assert presentation["raw_ratio"] == 1.25
    assert presentation["visual_ratio"] == 1.0
    assert presentation["source_badge"] == "Provider reported"


def test_shared_context_plan_example_uses_total_over_shared_window(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    presentation = demo._context_window_presentation(
        {
            "known": True,
            "limit_type": "shared_context",
            "context_window_tokens": 65_536,
            "input_tokens": 120,
            "output_tokens": 2_000,
            "used_tokens": 2_200,
            "usage_ratio": 2_200 / 65_536,
            "usage_source": "provider_reported",
        }
    )

    assert presentation["tooltip"] == (
        "2.2k / 65.5k (3%) · input 120 · output 2.0k · provider reported"
    )


def test_separate_io_tooltip_uses_most_constrained_limit(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    presentation = demo._context_window_presentation(
        {
            "known": True,
            "limit_type": "separate_io",
            "max_input_tokens": 65_536,
            "max_output_tokens": 32_768,
            "input_tokens": 120,
            "output_tokens": 2_000,
            "input_usage_ratio": 120 / 65_536,
            "output_usage_ratio": 2_000 / 32_768,
            "usage_ratio": 2_000 / 32_768,
            "usage_source": "provider_reported",
            "display_state": "ok",
        }
    )

    assert presentation["tooltip"] == (
        "6% limiting · input 120 / 65.5k (0.2%) · output 2.0k / 32.8k (6%) · provider reported"
    )
    assert presentation["raw_ratio"] == pytest.approx(2_000 / 32_768)


def test_separate_io_ignores_inconsistent_supplied_combined_ratio(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    presentation = demo._context_window_presentation(
        {
            "limit_type": "separate_io",
            "max_input_tokens": 100,
            "max_output_tokens": 100,
            "input_tokens": 20,
            "output_tokens": 80,
            "input_usage_ratio": 0.2,
            "output_usage_ratio": 0.8,
            "usage_ratio": 0.01,
            "usage_source": "provider_reported",
            "display_state": "ok",
        }
    )

    assert presentation["raw_ratio"] == 0.8
    assert presentation["tooltip"].startswith("80% limiting")
    assert presentation["state"] == "warn"


def test_separate_io_preserves_valid_backend_state_when_ratio_is_not_corrected(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    presentation = demo._context_window_presentation(
        {
            "limit_type": "separate_io",
            "input_usage_ratio": 0.2,
            "output_usage_ratio": 0.8,
            "usage_ratio": 0.8,
            "display_state": "danger",
        }
    )

    assert presentation["raw_ratio"] == 0.8
    assert presentation["state"] == "danger"


def test_malformed_context_numbers_never_raise(monkeypatch):
    demo, _ = _import_demo(monkeypatch)

    presentation = demo._context_window_presentation(
        {
            "limit_type": "shared_context",
            "context_window_tokens": "not-a-number",
            "used_tokens": {"bad": True},
            "usage_ratio": float("nan"),
        }
    )

    assert presentation["state"] == "unknown"
    assert presentation["raw_ratio"] is None


@pytest.mark.parametrize(
    "hostile_value",
    [10**10_000, -(10**10_000), float("nan"), float("inf"), True],
    ids=["huge-positive-int", "huge-negative-int", "nan", "infinity", "boolean"],
)
def test_hostile_unbounded_context_numbers_never_raise(monkeypatch, hostile_value):
    demo, _ = _import_demo(monkeypatch)
    payload = {
        "limit_type": "separate_io",
        "input_tokens": hostile_value,
        "output_tokens": hostile_value,
        "total_tokens": hostile_value,
        "used_tokens": hostile_value,
        "max_input_tokens": hostile_value,
        "max_output_tokens": hostile_value,
        "input_usage_ratio": hostile_value,
        "output_usage_ratio": hostile_value,
        "usage_ratio": hostile_value,
    }

    presentation = demo._context_window_presentation(payload)

    assert presentation["raw_ratio"] is None
    assert presentation["visual_ratio"] == 0.0
    assert presentation["state"] == "unknown"
    assert demo._format_tokens(hostile_value) == "?"
    assert demo._format_context_tokens(hostile_value) == "?"


def test_unknown_denominator_shows_counts_without_percentage(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    presentation = demo._context_window_presentation(
        {
            "known": False,
            "limit_type": "unknown",
            "input_tokens": 120,
            "output_tokens": 2_000,
            "total_tokens": 2_120,
            "usage_source": "locally_estimated",
        }
    )

    assert presentation["tooltip"] == (
        "2.1k total · input 120 · output 2.0k · limit unknown · locally estimated"
    )
    assert presentation["raw_ratio"] is None
    assert presentation["visual_ratio"] == 0.0
    assert presentation["state"] == "unknown"


def test_context_indicator_escapes_accessible_attributes_and_sets_fill(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    demo._render_context_window_indicator(
        'provider"<',
        "model&",
        {
            "known": True,
            "limit_type": "shared_context",
            "context_window_tokens": 100,
            "used_tokens": 250,
            "usage_ratio": 2.5,
            "usage_source": "mixed_reported_estimated",
            "display_state": "danger",
        },
    )

    rendered = stub.markdown_calls[-1][0]
    assert 'style="--ctx-fill:100.0%"' in rendered
    assert 'data-usage-ratio="2.5"' in rendered
    assert "provider&amp;quot;" not in rendered
    assert "provider&amp;" not in rendered
    assert "provider&quot;&lt;:model&amp;" in rendered
    assert "aria-label=" in rendered and "title=" in rendered
    assert 'provider"<' not in rendered


def test_chart_frames_are_bounded_to_api_rows_and_handle_empty(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    assert demo._build_usage_trend_frame([]) == []
    assert demo._build_usage_outcome_frame([], total_requests=0) == []

    series = [
        {
            "start": "2026-07-01T00:00:00Z",
            "totals": {"inputTokens": 1200, "outputTokens": 300},
        }
    ]
    outcomes = [{"key": "success", "totals": {"requestCount": 3}}]
    assert demo._build_usage_trend_frame(series) == [
        {"start": "2026-07-01T00:00:00Z", "tokenType": "Input", "tokens": 1200},
        {"start": "2026-07-01T00:00:00Z", "tokenType": "Output", "tokens": 300},
    ]
    assert demo._build_usage_outcome_frame(outcomes, total_requests=4) == [
        {"outcome": "success", "requests": 3, "rate": 0.75}
    ]


def test_breakdown_chart_frame_uses_only_bounded_api_items(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    items = [
        {
            "key": "provider-a",
            "totals": {"totalTokens": 1500, "requestCount": 3},
            "unknownFutureField": "ignored",
        }
    ]

    assert demo._build_usage_breakdown_frame(items) == [
        {"name": "provider-a", "totalTokens": 1500, "requests": 3}
    ]


@pytest.mark.parametrize(
    ("usage_enabled", "selected_label", "expected_renderer", "has_usage"),
    [
        (True, ":material/monitoring: Usage", "render_usage_view", True),
        (False, ":material/chat: Chat", "render_chat_view", False),
    ],
)
def test_main_tabs_are_keyed_lazy_and_capability_aware(
    monkeypatch, usage_enabled, selected_label, expected_renderer, has_usage
):
    demo, stub = _import_demo(monkeypatch)
    rendered: list[str] = []
    renderer_names = [
        "render_chat_view",
        "render_planning_tab",
        "render_documents_tab",
        "render_settings_view",
        "render_models_view",
        "render_usage_view",
        "render_custom_agents_view",
        "render_tools_tab",
        "render_skills_tab",
    ]
    for name in renderer_names:
        monkeypatch.setattr(demo, name, lambda name=name: rendered.append(name))
    stub.session_state.main_workspace_tab = selected_label

    demo._render_main_workspace_tabs(usage_enabled=usage_enabled)

    labels, kwargs = stub.tabs_calls[-1]
    assert (":material/monitoring: Usage" in labels) is has_usage
    if has_usage:
        assert (
            labels.index(":material/monitoring: Usage")
            == labels.index(":material/smart_toy: Models") + 1
        )
    assert kwargs == {"key": "main_workspace_tab", "on_change": "rerun"}
    assert rendered == [expected_renderer]


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [("complete", 1), ("error", 0), ("interrupt", 0), ("token", 0)],
)
def test_usage_cache_invalidation_is_terminal_completion_only(monkeypatch, event_type, expected):
    demo, _ = _import_demo(monkeypatch)
    calls: list[None] = []
    monkeypatch.setattr(demo, "_clear_usage_cache_after_completed_turn", lambda: calls.append(None))

    demo._handle_usage_stream_event(event_type)

    assert len(calls) == expected


def test_conversation_usage_panel_does_not_fetch_during_generation(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.stream_inflight = True
    calls: list[str] = []
    monkeypatch.setattr(
        demo,
        "get_conversation_usage",
        lambda conversation_id: calls.append(conversation_id),
    )

    demo._render_conversation_usage_panel("conversation-a")

    assert calls == []


@pytest.mark.parametrize("enabled", [True, False])
def test_conversation_panel_respects_usage_capability(monkeypatch, enabled):
    demo, _ = _import_demo(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        demo,
        "_render_conversation_usage_panel",
        lambda conversation_id: calls.append(conversation_id),
    )

    demo._render_conversation_usage_panel_if_enabled("conversation-a", enabled=enabled)

    assert calls == (["conversation-a"] if enabled else [])


def test_timezone_options_are_sorted_and_cached(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    calls: list[None] = []
    demo._usage_timezone_options.cache_clear()
    monkeypatch.setattr(
        demo,
        "available_timezones",
        lambda: calls.append(None) or {"UTC", "America/New_York", "Asia/Kathmandu"},
    )

    first = demo._usage_timezone_options()
    second = demo._usage_timezone_options()

    assert first == ("America/New_York", "Asia/Kathmandu", "UTC")
    assert second is first
    assert len(calls) == 1


def test_streamlit_dependency_supports_keyed_lazy_tabs():
    """Every dependency file must allow a streamlit new enough for keyed lazy tabs.

    Asserts the floor rather than an exact pin so that routine upgrades do not
    fail this test, while a downgrade below the feature's minimum still does.
    """

    minimum = Version("1.55.0")
    root = Path(__file__).resolve().parents[1]
    sources = {
        "requirements.txt": r"streamlit==([0-9][^\s]*)",
        "demo_requirements.txt": r"streamlit>=([0-9][^\s]*)",
        "environment.yml": r"streamlit==([0-9][^\s]*)",
    }

    for filename, pattern in sources.items():
        text = (root / filename).read_text(encoding="utf-8")
        match = re.search(pattern, text)
        assert match is not None, f"{filename} does not declare streamlit"
        assert Version(match.group(1)) >= minimum, (
            f"{filename} pins streamlit {match.group(1)}, below the {minimum} "
            "required for keyed lazy tabs"
        )


def test_local_timezone_fallback_is_portable_and_valid(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(demo, "_discover_system_zone_name", lambda: None)

    assert demo.get_local_timezone_name() == "UTC"
