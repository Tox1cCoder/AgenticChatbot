"""Unit tests for user-scoped model usage analytics."""

from dataclasses import fields
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.core.exceptions import ResourceNotFoundException, ValidationException
from app.repositories.model_usage import (
    DimensionUsageTotals,
    ModelUsageRepository,
)
from app.repositories.model_usage import (
    UsageTotals as RepositoryUsageTotals,
)
from app.schemas.model_usage import (
    ContextWindowMetadata,
    ConversationUsage,
    ConversationUsageItem,
    ConversationUsageQuery,
    UsageBreakdownItem,
    UsageCoverage,
    UsageDashboard,
    UsageDashboardQuery,
    UsageRange,
    UsageSeriesPoint,
    UsageTotals,
)
from app.services.model_usage_service import ModelUsageService


def test_usage_schemas_use_camel_case_aliases_and_zero_defaults() -> None:
    totals = UsageTotals()
    assert totals.model_dump(by_alias=True) == {
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "reasoningTokens": 0,
        "cachedInputTokens": 0,
        "generatedImages": 0,
        "requestCount": 0,
    }

    usage_range = UsageRange(
        from_=datetime(2026, 7, 1, tzinfo=timezone.utc),
        to=datetime(2026, 7, 2, tzinfo=timezone.utc),
        bucket="day",
        timezone="UTC",
    )
    assert usage_range.model_dump(by_alias=True)["from"] == datetime(
        2026, 7, 1, tzinfo=timezone.utc
    )


def test_usage_dashboard_query_accepts_public_aliases() -> None:
    query = UsageDashboardQuery.model_validate(
        {
            "from": "2026-07-01T00:00:00+07:00",
            "to": "2026-08-01T00:00:00+07:00",
            "bucket": "day",
            "timezone": "Asia/Bangkok",
        }
    )

    assert query.from_ == datetime.fromisoformat("2026-07-01T00:00:00+07:00")
    assert query.timezone == "Asia/Bangkok"


@pytest.mark.asyncio
async def test_usage_query_params_expose_public_from_and_return_route_validation_422() -> None:
    import httpx
    from fastapi import FastAPI

    from app.schemas.model_usage import UsageDashboardQueryParams

    app = FastAPI()

    @app.get("/usage")
    def get_usage(query: UsageDashboardQueryParams) -> dict:
        return query.model_dump(by_alias=True, mode="json")

    parameters = app.openapi()["paths"]["/usage"]["get"]["parameters"]
    assert [parameter["name"] for parameter in parameters] == [
        "from",
        "to",
        "bucket",
        "timezone",
        "conversationId",
    ]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        valid = await client.get(
            "/usage",
            params={
                "from": "2026-07-01T00:00:00+07:00",
                "to": "2026-07-02T00:00:00+07:00",
                "timezone": "Asia/Bangkok",
            },
        )
        unpaired = await client.get("/usage", params={"to": "2026-07-02T00:00:00+07:00"})
        non_numeric = await client.get(
            "/usage",
            params={
                "from": "2026-07-01T00:00:00Z",
                "to": "2026-07-02T00:00:00Z",
            },
        )

    assert valid.status_code == 200
    assert valid.json()["from"] == "2026-07-01T00:00:00+07:00"
    assert unpaired.status_code == 422
    assert non_numeric.status_code == 422


def test_usage_response_models_share_typed_nested_shapes() -> None:
    generated_at = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
    usage_range = UsageRange(
        from_=datetime(2026, 7, 1, tzinfo=timezone.utc),
        to=datetime(2026, 7, 2, tzinfo=timezone.utc),
        bucket="day",
        timezone="UTC",
    )
    totals = UsageTotals(total_tokens=7, request_count=1)
    breakdown = UsageBreakdownItem(key="openai", totals=totals)
    coverage = UsageCoverage(
        provider_reported_requests=1,
        requests_with_known_total=1,
        total_requests=1,
        known_total_ratio=1.0,
    )
    point = UsageSeriesPoint(start=usage_range.from_, end=usage_range.to, totals=totals)
    conversation = ConversationUsageItem(
        conversation_id="00000000-0000-0000-0000-000000000001",
        title="One",
        totals=totals,
    )

    dashboard = UsageDashboard(
        totals=totals,
        outcomes=[breakdown],
        series=[point],
        by_provider=[breakdown],
        by_model=[],
        by_operation=[],
        by_agent=[],
        top_conversations=[conversation],
        coverage=coverage,
        range=usage_range,
        generated_at=generated_at,
    )
    detail = ConversationUsage(
        totals=totals,
        by_provider=[breakdown],
        by_model=[],
        coverage=coverage,
        latest_context_window=ContextWindowMetadata(
            provider="custom",
            model="unknown-model",
            context_window_tokens=None,
            max_input_tokens=None,
            max_output_tokens=None,
            limit_type="unknown",
            source="unknown",
            known=False,
        ),
        range=usage_range,
        generated_at=generated_at,
    )

    assert dashboard.model_dump(by_alias=True)["topConversations"][0]["conversationId"]
    assert detail.model_dump(by_alias=True, exclude_none=True)["latestContextWindow"] == {
        "provider": "custom",
        "model": "unknown-model",
        "limit_type": "unknown",
        "source": "unknown",
        "known": False,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"from": "2026-07-01T00:00:00+00:00"},
        {"to": "2026-07-02T00:00:00+00:00"},
        {"from": "2026-07-01T00:00:00", "to": "2026-07-02T00:00:00"},
        {"from": "2026-07-01T00:00:00Z", "to": "2026-07-02T00:00:00Z"},
        {
            "from": "2026-07-01T00:00:01+00:00",
            "to": "2026-07-02T00:00:00+00:00",
        },
        {
            "from": "2026-07-01T00:00:00.000001+00:00",
            "to": "2026-07-02T00:00:00+00:00",
        },
    ],
)
def test_usage_query_rejects_unpaired_or_non_numeric_minute_bounds(payload: dict) -> None:
    with pytest.raises(ValidationError):
        UsageDashboardQuery.model_validate(payload)


class EmptyUsageRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get_summary_totals(self, **kwargs):
        self.calls.append(("summary", kwargs))
        return repository_totals()

    def get_minute_series(self, **kwargs):
        self.calls.append(("series", kwargs))
        return []

    def get_bucket_series(self, **kwargs):
        self.calls.append(("bucket_series", kwargs))
        return []

    def get_dimension_breakdown(self, **kwargs):
        self.calls.append(("dimension", kwargs))
        return []

    def get_top_conversations(self, **kwargs):
        self.calls.append(("top", kwargs))
        return []

    def get_latest_conversation_context_window(self, **kwargs):
        self.calls.append(("latest_context", kwargs))
        return None


class OwningConversationRepository:
    def user_owns_conversation(self, owner_id, conversation_id) -> bool:
        return True


class ForeignConversationRepository:
    def user_owns_conversation(self, owner_id, conversation_id) -> bool:
        return False


class BoundedSeriesSpyRepository(EmptyUsageRepository):
    def get_minute_series(self, **kwargs):
        raise AssertionError("dashboard must not load raw dimension-level minute rows")

    def get_bucket_series(self, **kwargs):
        self.calls.append(("bucket_series", kwargs))
        return []


def repository_totals(**values: int) -> RepositoryUsageTotals:
    return RepositoryUsageTotals(
        **{field.name: values.get(field.name, 0) for field in fields(RepositoryUsageTotals)}
    )


class PopulatedUsageRepository(EmptyUsageRepository):
    def __init__(self) -> None:
        super().__init__()
        self.summary = repository_totals()
        self.minute_rows: list[SimpleNamespace] = []
        self.dimensions: dict[str, list[DimensionUsageTotals]] = {}
        self.top_rows: list[SimpleNamespace] = []
        self.latest_context_window = None

    def get_summary_totals(self, **kwargs):
        self.calls.append(("summary", kwargs))
        return self.summary

    def get_minute_series(self, **kwargs):
        self.calls.append(("series", kwargs))
        return self.minute_rows

    def get_bucket_series(self, *, bucket_intervals, **kwargs):
        self.calls.append(("bucket_series", {"bucket_intervals": bucket_intervals, **kwargs}))
        rows = []
        for start, end in bucket_intervals:
            matching = [row for row in self.minute_rows if start <= row.bucket_start_utc < end]
            if not matching:
                continue
            summed = {
                field.name: sum(getattr(row, field.name) for row in matching)
                for field in fields(RepositoryUsageTotals)
            }
            rows.append(
                SimpleNamespace(
                    bucket_start_utc=start,
                    bucket_end_utc=end,
                    totals=repository_totals(**summed),
                )
            )
        return rows

    def get_dimension_breakdown(self, *, dimension, **kwargs):
        self.calls.append(("dimension", {"dimension": dimension, **kwargs}))
        return self.dimensions.get(dimension, [])

    def get_top_conversations(self, **kwargs):
        self.calls.append(("top", kwargs))
        return self.top_rows

    def get_latest_conversation_context_window(self, **kwargs):
        self.calls.append(("latest_context", kwargs))
        return self.latest_context_window


def make_empty_service(
    *, now: datetime = datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc)
) -> tuple[ModelUsageService, EmptyUsageRepository]:
    repository = EmptyUsageRepository()
    return (
        ModelUsageService(
            repository=repository,
            conversation_repository=OwningConversationRepository(),
            clock=lambda: now,
        ),
        repository,
    )


def test_dashboard_defaults_to_last_30_local_days_and_returns_empty_buckets() -> None:
    repository = EmptyUsageRepository()
    now = datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc)
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
        clock=lambda: now,
    )

    result = service.get_dashboard(user_id=uuid4(), query=UsageDashboardQuery())

    assert result.range.from_ == datetime(2026, 6, 22, tzinfo=timezone.utc)
    assert result.range.to == datetime(2026, 7, 22, tzinfo=timezone.utc)
    assert len(result.series) == 30
    assert result.totals == UsageTotals()
    assert result.coverage == UsageCoverage()


def test_dashboard_uses_one_bounded_bucket_query_and_never_raw_minute_rows() -> None:
    repository = BoundedSeriesSpyRepository()
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
        clock=lambda: datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc),
    )

    result = service.get_dashboard(user_id=uuid4(), query=UsageDashboardQuery())

    calls = [kwargs for name, kwargs in repository.calls if name == "bucket_series"]
    assert len(calls) == 1
    assert len(calls[0]["bucket_intervals"]) == 30
    assert len(result.series) == 30


def test_conversation_defaults_to_all_730_retained_days() -> None:
    repository = EmptyUsageRepository()
    now = datetime(2026, 7, 21, 12, 34, tzinfo=timezone.utc)
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
        clock=lambda: now,
    )

    result = service.get_conversation_usage(
        user_id=uuid4(),
        conversation_id=uuid4(),
        query=ConversationUsageQuery(),
    )

    assert result.range.from_ == datetime(2024, 7, 22, tzinfo=timezone.utc)
    assert result.range.to == datetime(2026, 7, 22, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (
            UsageDashboardQuery(
                from_=datetime(2026, 7, 2, tzinfo=timezone.utc),
                to=datetime(2026, 7, 1, tzinfo=timezone.utc),
            ),
            "after",
        ),
        (
            UsageDashboardQuery(
                from_=datetime(2024, 7, 1, tzinfo=timezone.utc),
                to=datetime(2026, 7, 2, tzinfo=timezone.utc),
            ),
            "two years",
        ),
        (
            UsageDashboardQuery(
                from_=datetime(2026, 1, 1, tzinfo=timezone.utc),
                to=datetime(2026, 2, 2, tzinfo=timezone.utc),
                bucket="hour",
            ),
            "31 days",
        ),
        (
            UsageDashboardQuery(
                from_=datetime(2026, 7, 1, 0, 30, tzinfo=timezone.utc),
                to=datetime(2026, 7, 1, 1, 30, tzinfo=timezone.utc),
                bucket="hour",
            ),
            "top of hour",
        ),
        (
            UsageDashboardQuery(
                from_=datetime(2026, 7, 1, 1, 0, tzinfo=timezone.utc),
                to=datetime(2026, 7, 2, 1, 0, tzinfo=timezone.utc),
                bucket="day",
            ),
            "midnight",
        ),
    ],
)
def test_usage_range_rejects_reversed_oversized_or_misaligned_bounds(
    query: UsageDashboardQuery, message: str
) -> None:
    service, _ = make_empty_service()

    with pytest.raises(ValidationException, match=message) as exc_info:
        service.get_dashboard(user_id=uuid4(), query=query)

    assert exc_info.value.status_code == 422


def test_usage_range_rejects_invalid_iana_timezone_as_domain_validation() -> None:
    service, _ = make_empty_service()

    with pytest.raises(ValidationException, match="timezone"):
        service.get_dashboard(user_id=uuid4(), query=UsageDashboardQuery(timezone="Mars/Olympus"))


def test_kathmandu_day_boundaries_query_quarter_hour_utc_instants() -> None:
    service, repository = make_empty_service()
    query = UsageDashboardQuery.model_validate(
        {
            "from": "2026-07-01T00:00:00+05:45",
            "to": "2026-07-02T00:00:00+05:45",
            "timezone": "Asia/Kathmandu",
        }
    )

    service.get_dashboard(user_id=uuid4(), query=query)

    summary_scope = next(kwargs for name, kwargs in repository.calls if name == "summary")
    assert summary_scope["start_inclusive"] == datetime(2026, 6, 30, 18, 15, tzinfo=timezone.utc)
    assert summary_scope["end_exclusive"] == datetime(2026, 7, 1, 18, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-03-10T02:00:00-05:00", "2024-03-10T03:00:00-04:00"),
        ("2024-11-03T01:00:00-06:00", "2024-11-03T02:00:00-05:00"),
    ],
)
def test_new_york_rejects_nonexistent_or_unmatched_fold_offsets(start: str, end: str) -> None:
    service, _ = make_empty_service()
    query = UsageDashboardQuery.model_validate(
        {
            "from": start,
            "to": end,
            "bucket": "hour",
            "timezone": "America/New_York",
        }
    )

    with pytest.raises(ValidationException, match="boundary"):
        service.get_dashboard(user_id=uuid4(), query=query)


def test_new_york_fall_fold_numeric_offsets_select_both_repeated_hours() -> None:
    service, _ = make_empty_service()
    query = UsageDashboardQuery.model_validate(
        {
            "from": "2024-11-03T01:00:00-04:00",
            "to": "2024-11-03T02:00:00-05:00",
            "bucket": "hour",
            "timezone": "America/New_York",
        }
    )

    result = service.get_dashboard(user_id=uuid4(), query=query)

    assert len(result.series) == 2
    assert [point.start.utcoffset() for point in result.series] == [
        timedelta(hours=-4),
        timedelta(hours=-5),
    ]


def test_historical_sub_minute_zone_boundary_is_rejected() -> None:
    service, _ = make_empty_service()
    monrovia_offset = timezone(-timedelta(minutes=43, seconds=8))
    query = UsageDashboardQuery(
        from_=datetime(1900, 1, 1, tzinfo=monrovia_offset),
        to=datetime(1900, 1, 2, tzinfo=monrovia_offset),
        timezone="Africa/Monrovia",
    )

    with pytest.raises(ValidationException, match="minute-aligned"):
        service.get_dashboard(user_id=uuid4(), query=query)


def test_default_historical_sub_minute_zone_boundary_is_rejected() -> None:
    service, _ = make_empty_service(now=datetime(1900, 1, 20, 12, tzinfo=timezone.utc))

    with pytest.raises(ValidationException, match="minute-aligned"):
        service.get_dashboard(
            user_id=uuid4(), query=UsageDashboardQuery(timezone="Africa/Monrovia")
        )


def test_dashboard_uses_only_known_total_sums_and_reports_mutually_exclusive_coverage() -> None:
    repository = PopulatedUsageRepository()
    repository.summary = repository_totals(
        request_count=4,
        input_tokens_sum=123,
        output_tokens_sum=45,
        total_tokens_sum=140,
        total_tokens_known_count=2,
        generated_images_sum=1,
    )
    repository.dimensions["usage_source"] = [
        DimensionUsageTotals(
            "provider_reported",
            repository_totals(request_count=1, total_tokens_sum=100, total_tokens_known_count=1),
        ),
        DimensionUsageTotals(
            "mixed_reported_estimated",
            repository_totals(request_count=1, total_tokens_sum=40, total_tokens_known_count=1),
        ),
        DimensionUsageTotals("locally_estimated", repository_totals(request_count=1)),
        DimensionUsageTotals("unavailable", repository_totals(request_count=1)),
    ]
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
        clock=lambda: datetime(2026, 7, 21, tzinfo=timezone.utc),
    )

    result = service.get_dashboard(user_id=uuid4(), query=UsageDashboardQuery())

    assert result.totals.total_tokens == 140
    assert result.totals.request_count == 4
    assert result.coverage.model_dump() == {
        "provider_reported_requests": 1,
        "mixed_requests": 1,
        "locally_estimated_requests": 1,
        "unavailable_requests": 1,
        "requests_with_known_total": 2,
        "total_requests": 4,
        "known_total_ratio": 0.5,
    }


def test_breakdowns_are_limited_and_sorted_by_known_total_then_key() -> None:
    repository = PopulatedUsageRepository()
    repository.dimensions["provider"] = [
        DimensionUsageTotals(f"provider-{index:02}", repository_totals(total_tokens_sum=index % 3))
        for index in range(25)
    ]
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )

    result = service.get_dashboard(user_id=uuid4(), query=UsageDashboardQuery())

    assert len(result.by_provider) == 20
    assert [(item.totals.total_tokens, item.key) for item in result.by_provider] == sorted(
        [(item.totals.total_tokens, item.key) for item in result.by_provider],
        key=lambda item: (-item[0], item[1]),
    )


def minute_row(bucket_start_utc: datetime, **values: int) -> SimpleNamespace:
    totals = repository_totals(**values)
    return SimpleNamespace(bucket_start_utc=bucket_start_utc, **totals.__dict__)


def test_bangkok_minutes_regroup_into_local_days_and_missing_days_are_zero_filled() -> None:
    repository = PopulatedUsageRepository()
    repository.minute_rows = [
        minute_row(
            datetime(2026, 6, 30, 17, 0, tzinfo=timezone.utc),
            request_count=1,
            total_tokens_sum=7,
            total_tokens_known_count=1,
        ),
        minute_row(
            datetime(2026, 7, 1, 16, 59, tzinfo=timezone.utc),
            request_count=2,
            total_tokens_sum=9,
            total_tokens_known_count=2,
        ),
    ]
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )
    query = UsageDashboardQuery.model_validate(
        {
            "from": "2026-07-01T00:00:00+07:00",
            "to": "2026-07-03T00:00:00+07:00",
            "timezone": "Asia/Bangkok",
        }
    )

    result = service.get_dashboard(user_id=uuid4(), query=query)

    assert [point.totals.total_tokens for point in result.series] == [16, 0]
    assert [point.totals.request_count for point in result.series] == [3, 0]
    assert result.series[0].start.isoformat() == "2026-07-01T00:00:00+07:00"
    assert result.series[0].end.isoformat() == "2026-07-02T00:00:00+07:00"


@pytest.mark.parametrize(
    ("start", "end", "expected_hours"),
    [
        ("2024-03-10T00:00:00-05:00", "2024-03-11T00:00:00-04:00", 23),
        ("2024-11-03T00:00:00-04:00", "2024-11-04T00:00:00-05:00", 25),
    ],
)
def test_new_york_local_day_series_has_dst_aware_duration(
    start: str, end: str, expected_hours: int
) -> None:
    service, _ = make_empty_service()
    query = UsageDashboardQuery.model_validate(
        {"from": start, "to": end, "bucket": "day", "timezone": "America/New_York"}
    )

    result = service.get_dashboard(user_id=uuid4(), query=query)

    assert len(result.series) == 1
    elapsed = result.series[0].end.astimezone(timezone.utc) - result.series[0].start.astimezone(
        timezone.utc
    )
    assert elapsed == timedelta(hours=expected_hours)


@pytest.mark.parametrize(
    ("start", "end", "expected_starts", "expected_durations"),
    [
        (
            "2024-04-07T00:00:00+11:00",
            "2024-04-07T04:00:00+10:30",
            [
                "2024-04-07T00:00:00+11:00",
                "2024-04-07T01:00:00+11:00",
                "2024-04-07T02:00:00+10:30",
                "2024-04-07T03:00:00+10:30",
            ],
            [60, 90, 60, 60],
        ),
        (
            "2024-10-06T00:00:00+10:30",
            "2024-10-06T04:00:00+11:00",
            [
                "2024-10-06T00:00:00+10:30",
                "2024-10-06T01:00:00+10:30",
                "2024-10-06T03:00:00+11:00",
            ],
            [60, 90, 60],
        ),
    ],
)
def test_lord_howe_hour_series_uses_local_wall_hour_boundaries(
    start: str,
    end: str,
    expected_starts: list[str],
    expected_durations: list[int],
) -> None:
    service, _ = make_empty_service()
    query = UsageDashboardQuery.model_validate(
        {
            "from": start,
            "to": end,
            "bucket": "hour",
            "timezone": "Australia/Lord_Howe",
        }
    )

    result = service.get_dashboard(user_id=uuid4(), query=query)

    assert [point.start.isoformat() for point in result.series] == expected_starts
    assert all(point.start.minute == 0 and point.end.minute == 0 for point in result.series)
    assert [
        int(
            (
                point.end.astimezone(timezone.utc) - point.start.astimezone(timezone.utc)
            ).total_seconds()
            // 60
        )
        for point in result.series
    ] == expected_durations


def test_ambiguous_local_midnight_fold_keeps_daily_usage_in_selected_interval() -> None:
    repository = PopulatedUsageRepository()
    repository.minute_rows = [
        minute_row(
            datetime(2024, 11, 3, 5, 30, tzinfo=timezone.utc),
            request_count=1,
            total_tokens_sum=11,
            total_tokens_known_count=1,
        )
    ]
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )
    query = UsageDashboardQuery.model_validate(
        {
            "from": "2024-11-03T00:00:00-05:00",
            "to": "2024-11-04T00:00:00-05:00",
            "timezone": "America/Havana",
        }
    )

    result = service.get_dashboard(user_id=uuid4(), query=query)

    assert len(result.series) == 1
    assert result.series[0].start.fold == 1
    assert result.series[0].totals.total_tokens == 11


@pytest.mark.parametrize("endpoint", ["dashboard", "conversation"])
def test_foreign_conversation_is_ownership_safe_not_found_before_usage_query(endpoint: str) -> None:
    repository = EmptyUsageRepository()
    service = ModelUsageService(
        repository=repository,
        conversation_repository=ForeignConversationRepository(),
    )
    conversation_id = uuid4()

    with pytest.raises(ResourceNotFoundException) as exc_info:
        if endpoint == "dashboard":
            service.get_dashboard(
                user_id=uuid4(),
                query=UsageDashboardQuery(conversation_id=conversation_id),
            )
        else:
            service.get_conversation_usage(
                user_id=uuid4(),
                conversation_id=conversation_id,
                query=ConversationUsageQuery(),
            )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Conversation not found"
    assert repository.calls == []


def test_top_conversations_are_bounded_and_deterministic() -> None:
    repository = PopulatedUsageRepository()
    conversation_ids = [uuid4() for _ in range(25)]
    repository.top_rows = [
        SimpleNamespace(
            conversation_id=conversation_id,
            title=f"Conversation {index}",
            totals=repository_totals(total_tokens_sum=index % 4),
        )
        for index, conversation_id in enumerate(conversation_ids)
    ]
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )

    result = service.get_dashboard(user_id=uuid4(), query=UsageDashboardQuery())

    actual = []
    for item in result.top_conversations:
        actual.append((item.totals.total_tokens, str(item.conversation_id)))
    assert len(actual) == 20
    assert actual == sorted(actual, key=lambda item: (-item[0], item[1]))


def test_conversation_uses_latest_assistant_message_context_window() -> None:
    repository = PopulatedUsageRepository()
    repository.latest_context_window = {
        "provider": "gemini",
        "model": "gemini-3-pro-image",
        "context_window_tokens": None,
        "max_input_tokens": 65_536,
        "max_output_tokens": 32_768,
        "limit_type": "separate_io",
        "source": "registry",
        "known": True,
        "input_tokens": 100,
        "output_tokens": 200,
        "total_tokens": 300,
        "usage_source": "provider_reported",
        "used_tokens": 300,
        "used_token_source": "provider_reported_total",
        "input_usage_ratio": 100 / 65_536,
        "output_usage_ratio": 200 / 32_768,
        "usage_ratio": 200 / 32_768,
        "usage_ratio_basis": "most_constrained_io_limit",
        "display_state": "ok",
    }
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )
    user_id = uuid4()
    conversation_id = uuid4()

    result = service.get_conversation_usage(
        user_id=user_id,
        conversation_id=conversation_id,
        query=ConversationUsageQuery(),
    )

    assert result.latest_context_window is not None
    assert result.latest_context_window.model == "gemini-3-pro-image"
    assert result.latest_context_window.limit_type == "separate_io"
    assert result.latest_context_window.usage_ratio == 200 / 32_768
    assert ("latest_context", {"user_id": user_id, "conversation_id": conversation_id}) in (
        repository.calls
    )


def test_conversation_ignores_invalid_assistant_context_window() -> None:
    repository = PopulatedUsageRepository()
    repository.latest_context_window = {"provider": "openai"}
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )

    result = service.get_conversation_usage(
        user_id=uuid4(),
        conversation_id=uuid4(),
        query=ConversationUsageQuery(),
    )

    assert result.latest_context_window is None


class EmptyMappingResult:
    def mappings(self):
        return self

    def all(self) -> list:
        return []


class CapturingSession:
    def __init__(self, statements: list) -> None:
        self.statements = statements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, statement):
        self.statements.append(statement)
        return EmptyMappingResult()


class EmptyScalarResult:
    def scalar_one_or_none(self):
        return None


class CapturingLatestSession(CapturingSession):
    def execute(self, statement):
        self.statements.append(statement)
        return EmptyScalarResult()


def compile_postgres(statement) -> str:
    return str(
        statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


def test_dimension_breakdown_applies_server_side_limit_and_deterministic_rank() -> None:
    statements: list = []
    repository = ModelUsageRepository(lambda: CapturingSession(statements))

    repository.get_dimension_breakdown(
        user_id=uuid4(),
        start_inclusive=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2026, 7, 2, tzinfo=timezone.utc),
        dimension="provider",
        limit=20,
    )

    sql = compile_postgres(statements[0])
    assert "ORDER BY" in sql
    assert "total_tokens_sum DESC" in sql
    assert "coalesce(model_usage_minute.provider, 'unknown') ASC" in sql
    assert "LIMIT 20" in sql


def test_nullable_agent_breakdown_ranks_by_public_unknown_key_before_limit() -> None:
    statements: list = []
    repository = ModelUsageRepository(lambda: CapturingSession(statements))

    repository.get_dimension_breakdown(
        user_id=uuid4(),
        start_inclusive=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2026, 7, 2, tzinfo=timezone.utc),
        dimension="agent_id",
        limit=20,
    )

    sql = compile_postgres(statements[0])
    assert "coalesce(model_usage_minute.agent_id, 'unknown') AS dimension_value" in sql
    assert (
        "ORDER BY total_tokens_sum DESC, coalesce(model_usage_minute.agent_id, 'unknown') ASC"
        in sql
    )


def test_top_conversations_is_tenant_scoped_half_open_and_bounded_in_sql() -> None:
    statements: list = []
    repository = ModelUsageRepository(lambda: CapturingSession(statements))
    user_id = uuid4()

    repository.get_top_conversations(
        user_id=user_id,
        start_inclusive=datetime(2026, 7, 1, tzinfo=timezone.utc),
        end_exclusive=datetime(2026, 7, 2, tzinfo=timezone.utc),
        limit=20,
    )

    sql = compile_postgres(statements[0])
    assert "JOIN conversations" in sql
    assert f"model_usage_minute.user_id = '{user_id}'" in sql
    assert "model_usage_minute.bucket_start_utc >=" in sql
    assert "model_usage_minute.bucket_start_utc <" in sql
    assert "conversations.deleted_at IS NULL" in sql
    assert "total_tokens_sum DESC" in sql
    assert "model_usage_minute.conversation_id ASC" in sql
    assert "LIMIT 20" in sql


def test_latest_conversation_context_is_owner_scoped_visible_assistant_and_bounded() -> None:
    statements: list = []
    repository = ModelUsageRepository(lambda: CapturingLatestSession(statements))
    user_id = uuid4()
    conversation_id = uuid4()

    assert (
        repository.get_latest_conversation_context_window(
            user_id=user_id, conversation_id=conversation_id
        )
        is None
    )

    sql = compile_postgres(statements[0])
    assert "JOIN conversations" in sql
    assert f"conversations.owner_id = '{user_id}'" in sql
    assert f"conversations.id = '{conversation_id}'" in sql
    assert "conversations.deleted_at IS NULL" in sql
    assert "messages.deleted_at IS NULL" in sql
    assert "messages.sender = 2" in sql
    assert "messages.message_metadata ? 'context_window'" in sql
    assert "ORDER BY messages.sequence DESC, messages.id DESC" in sql
    assert "LIMIT 1" in sql


def test_model_usage_service_is_a_factory_and_injectable_by_interface() -> None:
    from dependency_injector import providers

    from app.core.container import Container
    from app.core.dependency_injection import AppAutoInjector, AppContainerInjector
    from app.interfaces import IModelUsageService

    assert isinstance(Container.model_usage_service, providers.Factory)
    assert AppAutoInjector.wiring_map[IModelUsageService] is Container.model_usage_service
    assert AppContainerInjector.wiring_map[IModelUsageService] is Container.model_usage_service
    assert AppContainerInjector.wiring_map[ModelUsageRepository] is Container.model_usage_repository
