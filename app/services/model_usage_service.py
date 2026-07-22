"""User-scoped assembly and local-time bucketing for model-usage analytics."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, time, timedelta, timezone
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ValidationError

from app.core.exceptions import ResourceNotFoundException, ValidationException
from app.interfaces.model_usage_service_interface import IModelUsageService
from app.repositories.model_usage import DimensionUsageTotals
from app.repositories.model_usage import UsageTotals as RepositoryUsageTotals
from app.schemas.model_usage import (
    ContextWindowMetadata,
    ConversationUsage,
    ConversationUsageItem,
    ConversationUsageQuery,
    UsageBreakdownItem,
    UsageCoverage,
    UsageDashboard,
    UsageDashboardQuery,
    UsageQuery,
    UsageRange,
    UsageSeriesPoint,
    UsageTotals,
)

logger = logging.getLogger(__name__)

_UTC = timezone.utc
USAGE_DASHBOARD_DEFAULT_DAYS = 30
USAGE_CONVERSATION_DEFAULT_DAYS = 730
USAGE_MAX_RANGE_DAYS = 730
USAGE_MAX_HOURLY_RANGE_DAYS = 31


class ModelUsageService(IModelUsageService):
    """Read bounded rollups and expose only one authenticated user's data."""

    def __init__(
        self,
        repository: Any,
        conversation_repository: Any,
        settings: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.conversation_repository = conversation_repository
        self.settings = settings
        self.clock = clock or (lambda: datetime.now(_UTC))

    def get_dashboard(self, *, user_id: UUID, query: UsageDashboardQuery) -> UsageDashboard:
        if query.conversation_id is not None:
            self._require_owned_conversation(user_id, query.conversation_id)
        usage_range, start_utc, end_utc = self._resolve_range(
            query, default_days=USAGE_DASHBOARD_DEFAULT_DAYS
        )
        scope = self._query_scope(user_id, start_utc, end_utc, query.conversation_id)
        totals = self._public_totals(self.repository.get_summary_totals(**scope))
        intervals = self._build_bucket_intervals(usage_range, start_utc, end_utc)
        series = self._build_series(
            self.repository.get_bucket_series(
                user_id=user_id,
                bucket_intervals=intervals,
                conversation_id=query.conversation_id,
            ),
            usage_range,
            intervals,
        )
        coverage = self._coverage(scope)
        return UsageDashboard(
            totals=totals,
            outcomes=self._breakdown(scope, "status"),
            series=series,
            by_provider=self._breakdown(scope, "provider"),
            by_model=self._breakdown(scope, "model"),
            by_operation=self._breakdown(scope, "operation"),
            by_agent=self._breakdown(scope, "agent_id"),
            top_conversations=self._top_conversations(scope),
            coverage=coverage,
            range=usage_range,
            generated_at=self._now(),
        )

    def get_conversation_usage(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        query: ConversationUsageQuery,
    ) -> ConversationUsage:
        self._require_owned_conversation(user_id, conversation_id)
        usage_range, start_utc, end_utc = self._resolve_range(
            query, default_days=USAGE_CONVERSATION_DEFAULT_DAYS
        )
        scope = self._query_scope(user_id, start_utc, end_utc, conversation_id)
        latest = self.repository.get_latest_conversation_context_window(
            user_id=user_id, conversation_id=conversation_id
        )
        return ConversationUsage(
            totals=self._public_totals(self.repository.get_summary_totals(**scope)),
            by_provider=self._breakdown(scope, "provider"),
            by_model=self._breakdown(scope, "model"),
            coverage=self._coverage(scope),
            latest_context_window=self._latest_context_window(latest),
            range=usage_range,
            generated_at=self._now(),
        )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ModelUsageService clock must return an aware datetime")
        return value.astimezone(_UTC)

    def _resolve_range(
        self, query: UsageQuery, *, default_days: int
    ) -> tuple[UsageRange, datetime, datetime]:
        try:
            zone = ZoneInfo(query.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValidationException(
                detail=f"Invalid IANA timezone: {query.timezone}",
                error_code="INVALID_USAGE_TIMEZONE",
            ) from exc
        if query.from_ is None:
            today = self._now().astimezone(zone).date()
            end_local = datetime.combine(today + timedelta(days=1), time.min, zone)
            start_local = datetime.combine(today + timedelta(days=1 - default_days), time.min, zone)
            start_utc = start_local.astimezone(_UTC)
            end_utc = end_local.astimezone(_UTC)
        else:
            start_local, start_utc = self._resolve_explicit_boundary(query.from_, zone, "from")
            end_local, end_utc = self._resolve_explicit_boundary(
                query.to,
                zone,
                "to",  # type: ignore[arg-type]
            )
        self._validate_range(query, start_local, end_local, start_utc, end_utc)
        usage_range = UsageRange(
            from_=start_local,
            to=end_local,
            bucket=query.bucket,
            timezone=query.timezone,
        )
        return usage_range, start_utc, end_utc

    @staticmethod
    def _resolve_explicit_boundary(
        supplied: datetime, zone: ZoneInfo, field_name: str
    ) -> tuple[datetime, datetime]:
        wall = supplied.replace(tzinfo=None)
        supplied_offset = supplied.utcoffset()
        matches: dict[datetime, datetime] = {}
        for fold in (0, 1):
            local = wall.replace(tzinfo=zone, fold=fold)
            utc_value = local.astimezone(_UTC)
            round_trip = utc_value.astimezone(zone)
            if round_trip.replace(tzinfo=None) == wall and local.utcoffset() == supplied_offset:
                matches[utc_value] = round_trip
        if len(matches) != 1:
            raise ValidationException(
                detail=(
                    f"Invalid {field_name} boundary for {zone.key}: the wall time is "
                    "nonexistent, ambiguous without a matching numeric offset, or uses "
                    "an offset that does not apply"
                ),
                error_code="INVALID_USAGE_BOUNDARY",
            )
        utc_value, local = next(iter(matches.items()))
        if utc_value.second != 0 or utc_value.microsecond != 0:
            raise ValidationException(
                detail=f"{field_name} boundary must resolve to a minute-aligned UTC instant",
                error_code="INVALID_USAGE_BOUNDARY",
            )
        return local, utc_value

    @staticmethod
    def _validate_range(
        query: UsageQuery,
        start_local: datetime,
        end_local: datetime,
        start_utc: datetime,
        end_utc: datetime,
    ) -> None:
        for field_name, value in (("from", start_utc), ("to", end_utc)):
            if value.second != 0 or value.microsecond != 0:
                raise ValidationException(
                    detail=f"{field_name} boundary must resolve to a minute-aligned UTC instant",
                    error_code="INVALID_USAGE_BOUNDARY",
                )
        if end_utc <= start_utc:
            raise ValidationException(
                detail="Usage range to must be after from (from is inclusive; to is exclusive)",
                error_code="INVALID_USAGE_RANGE",
            )
        wall_duration = end_local.replace(tzinfo=None) - start_local.replace(tzinfo=None)
        if wall_duration > timedelta(days=USAGE_MAX_RANGE_DAYS):
            raise ValidationException(
                detail=(f"Usage range cannot exceed two years ({USAGE_MAX_RANGE_DAYS} days)"),
                error_code="USAGE_RANGE_TOO_LARGE",
            )
        if query.bucket == "hour":
            if wall_duration > timedelta(days=USAGE_MAX_HOURLY_RANGE_DAYS):
                raise ValidationException(
                    detail=(
                        f"Hourly usage ranges cannot exceed {USAGE_MAX_HOURLY_RANGE_DAYS} days"
                    ),
                    error_code="USAGE_HOURLY_RANGE_TOO_LARGE",
                )
            if any(
                value.minute != 0 or value.second != 0 or value.microsecond != 0
                for value in (start_local, end_local)
            ):
                raise ValidationException(
                    detail="Hourly usage boundaries must be aligned to the local top of hour",
                    error_code="INVALID_USAGE_ALIGNMENT",
                )
        elif any(
            value.time().replace(tzinfo=None) != time.min for value in (start_local, end_local)
        ):
            raise ValidationException(
                detail="Daily usage boundaries must be aligned to local midnight",
                error_code="INVALID_USAGE_ALIGNMENT",
            )

    @staticmethod
    def _query_scope(
        user_id: UUID,
        start_utc: datetime,
        end_utc: datetime,
        conversation_id: UUID | None,
    ) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "start_inclusive": start_utc,
            "end_exclusive": end_utc,
            "conversation_id": conversation_id,
        }

    def _require_owned_conversation(self, user_id: UUID, conversation_id: UUID) -> None:
        if not self.conversation_repository.user_owns_conversation(user_id, conversation_id):
            raise ResourceNotFoundException(
                detail="Conversation not found", error_code="CONVERSATION_NOT_FOUND"
            )

    @staticmethod
    def _public_totals(totals: RepositoryUsageTotals) -> UsageTotals:
        return UsageTotals(
            input_tokens=totals.input_tokens_sum,
            output_tokens=totals.output_tokens_sum,
            total_tokens=totals.total_tokens_sum,
            reasoning_tokens=totals.reasoning_tokens_sum,
            cached_input_tokens=totals.cached_input_tokens_sum,
            generated_images=totals.generated_images_sum,
            request_count=totals.request_count,
        )

    def _breakdown(self, scope: dict[str, Any], dimension: str) -> list[UsageBreakdownItem]:
        rows: list[DimensionUsageTotals] = self.repository.get_dimension_breakdown(
            **scope, dimension=dimension, limit=20
        )
        items = [
            UsageBreakdownItem(
                key=row.dimension_value or "unknown",
                totals=self._public_totals(row.totals),
            )
            for row in rows
        ]
        return sorted(items, key=lambda item: (-item.totals.total_tokens, item.key))[:20]

    def _coverage(self, scope: dict[str, Any]) -> UsageCoverage:
        rows: list[DimensionUsageTotals] = self.repository.get_dimension_breakdown(
            **scope, dimension="usage_source", limit=20
        )
        counts = {row.dimension_value: row.totals.request_count for row in rows}
        total = sum(counts.values())
        known = sum(row.totals.total_tokens_known_count for row in rows)
        return UsageCoverage(
            provider_reported_requests=counts.get("provider_reported", 0),
            mixed_requests=counts.get("mixed_reported_estimated", 0),
            locally_estimated_requests=counts.get("locally_estimated", 0),
            unavailable_requests=counts.get("unavailable", 0),
            requests_with_known_total=known,
            total_requests=total,
            known_total_ratio=known / total if total else 0.0,
        )

    def _top_conversations(self, scope: dict[str, Any]) -> list[ConversationUsageItem]:
        rows = self.repository.get_top_conversations(**scope, limit=20)
        items = [
            ConversationUsageItem(
                conversation_id=row.conversation_id,
                title=row.title,
                totals=self._public_totals(row.totals),
            )
            for row in rows
        ]
        return sorted(
            items,
            key=lambda item: (-item.totals.total_tokens, str(item.conversation_id)),
        )[:20]

    @staticmethod
    def _build_bucket_intervals(
        usage_range: UsageRange,
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[tuple[datetime, datetime]]:
        zone = ZoneInfo(usage_range.timezone)
        if usage_range.bucket == "hour":
            start_wall = start_utc.astimezone(zone).replace(tzinfo=None)
            end_wall = end_utc.astimezone(zone).replace(tzinfo=None)
            boundaries = {start_utc, end_utc}
            wall_cursor = start_wall
            while wall_cursor <= end_wall:
                for candidate in ModelUsageService._valid_wall_instants(wall_cursor, zone):
                    if start_utc <= candidate <= end_utc:
                        if candidate.second != 0 or candidate.microsecond != 0:
                            raise ValidationException(
                                detail=(
                                    "Hourly boundary must resolve to a minute-aligned UTC instant"
                                ),
                                error_code="INVALID_USAGE_BOUNDARY",
                            )
                        boundaries.add(candidate)
                wall_cursor += timedelta(hours=1)
            ordered = sorted(boundaries)
            return list(zip(ordered, ordered[1:], strict=False))

        intervals: list[tuple[datetime, datetime]] = []
        cursor = start_utc
        while cursor < end_utc:
            local = cursor.astimezone(zone)
            next_local = datetime.combine(local.date() + timedelta(days=1), time.min, zone)
            next_cursor = min(next_local.astimezone(_UTC), end_utc)
            intervals.append((cursor, next_cursor))
            cursor = next_cursor
        return intervals

    @staticmethod
    def _valid_wall_instants(wall: datetime, zone: ZoneInfo) -> list[datetime]:
        """Resolve a naive wall time to every real UTC fold, skipping gaps."""
        candidates: set[datetime] = set()
        for fold in (0, 1):
            local = wall.replace(tzinfo=zone, fold=fold)
            utc_value = local.astimezone(_UTC)
            if utc_value.astimezone(zone).replace(tzinfo=None) == wall:
                candidates.add(utc_value)
        return sorted(candidates)

    def _build_series(
        self,
        rows: list[Any],
        usage_range: UsageRange,
        intervals: list[tuple[datetime, datetime]],
    ) -> list[UsageSeriesPoint]:
        zone = ZoneInfo(usage_range.timezone)
        bucket_totals = {
            row.bucket_start_utc.astimezone(_UTC): self._public_totals(row.totals) for row in rows
        }

        return [
            UsageSeriesPoint(
                start=bucket_start.astimezone(zone),
                end=bucket_end.astimezone(zone),
                totals=bucket_totals.get(bucket_start, UsageTotals()),
            )
            for bucket_start, bucket_end in intervals
        ]

    @staticmethod
    def _latest_context_window(raw: Any | None) -> ContextWindowMetadata | None:
        if raw is None:
            return None
        try:
            return ContextWindowMetadata.model_validate(raw)
        except ValidationError:
            logger.warning("Ignoring invalid persisted context-window metadata")
            return None


__all__ = [
    "ModelUsageService",
    "USAGE_CONVERSATION_DEFAULT_DAYS",
    "USAGE_DASHBOARD_DEFAULT_DAYS",
    "USAGE_MAX_HOURLY_RANGE_DAYS",
    "USAGE_MAX_RANGE_DAYS",
]
