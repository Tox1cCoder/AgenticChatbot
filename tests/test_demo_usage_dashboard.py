from __future__ import annotations

import importlib
import sys
import types
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pytest


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
            func.clear = self.clear
            return func

        return decorate

    def clear(self) -> None:
        self.clear_calls += 1


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.query_params: dict[str, str] = {}
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()
        self.sidebar = _Context()
        self.markdown_calls: list[tuple[str, dict[str, Any]]] = []

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, body: str, *args: Any, **kwargs: Any) -> None:
        self.markdown_calls.append((body, kwargs))

    def toast(self, *args: Any, **kwargs: Any) -> None:
        return None

    def expander(self, *args: Any, **kwargs: Any) -> _Context:
        return _Context()

    def caption(self, *args: Any, **kwargs: Any) -> None:
        return None

    def button(self, *args: Any, **kwargs: Any) -> bool:
        return False

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


def test_dashboard_query_is_encoded_and_has_no_user_identity(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.auth_token = "secret-token"
    stub.session_state.current_user_id = "user-a"
    calls: list[tuple[str, str, bool]] = []

    def fake_usage_get(endpoint: str, *, auth_identity: str, cache_version: int):
        calls.append((endpoint, auth_identity, bool(cache_version)))
        return {"success": True, "data": {"totals": {}}}

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


def test_usage_cache_miss_uses_authenticated_response_envelope_helper(monkeypatch):
    demo, stub = _import_demo(monkeypatch)
    stub.session_state.auth_token = "active-auth-token"
    calls: list[tuple[str, str, bool, str]] = []

    def fake_request(method: str, endpoint: str, *, use_cache: bool):
        calls.append((method, endpoint, use_cache, stub.session_state.auth_token))
        return {"success": True, "data": {}}

    monkeypatch.setattr(demo, "make_api_request", fake_request)
    response = demo._cached_usage_get_request(
        "/usage/dashboard", auth_identity="safe-partition", cache_version=0
    )

    assert response == {"success": True, "data": {}}
    assert calls == [("GET", "/usage/dashboard", False, "active-auth-token")]


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


def test_usage_tab_is_immediately_after_models_and_cache_clears_only_on_completion():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")
    models_index = source.index('":material/smart_toy: Models"')
    usage_index = source.index('":material/monitoring: Usage"')
    custom_agents_index = source.index('":material/robot_2: Custom Agents"')

    assert models_index < usage_index < custom_agents_index
    assert "_clear_usage_cache_after_completed_turn()" in source
    assert source.count("_clear_usage_cache_after_completed_turn()") == 2  # definition + complete
    complete_block = source[source.index('elif event_type == "complete":') :]
    assert complete_block.index("_clear_usage_cache_after_completed_turn()") < complete_block.index(
        'elif event_type == "error":'
    )


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


def test_local_timezone_fallback_is_portable_and_valid(monkeypatch):
    demo, _ = _import_demo(monkeypatch)
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(demo, "_discover_system_zone_name", lambda: None)

    assert demo.get_local_timezone_name() == "UTC"
