"""Typed query and response models for user-scoped usage analytics."""

import re
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.case_conversion import to_camel_case

UsageBucket = Literal["hour", "day"]
ContextSource = Literal["provider_api", "registry", "heuristic", "unknown"]
ContextLimitType = Literal["shared_context", "separate_io", "unknown"]
ContextUsageSource = Literal[
    "provider_reported",
    "mixed_reported_estimated",
    "locally_estimated",
    "unavailable",
]
ContextUsedTokenSource = Literal[
    "provider_reported_total",
    "provider_reported_split",
    "estimated_total",
    "unknown",
]
ContextUsageRatioBasis = Literal["shared_context_total", "most_constrained_io_limit"]
ContextDisplayState = Literal["unknown", "ok", "warn", "danger"]


class UsageModel(BaseModel):
    """Base model using the API's camelCase serialization convention."""

    model_config = ConfigDict(alias_generator=to_camel_case, populate_by_name=True)


class ContextWindowMetadata(BaseModel):
    """Static model limits plus optional usage added after a model call.

    Keys intentionally remain snake_case because this shape is persisted in
    AI SDK message metadata. Unknown future keys are ignored by typed readers.
    """

    provider: str
    model: str
    context_window_tokens: int | None = Field(ge=1)
    max_input_tokens: int | None = Field(ge=1)
    max_output_tokens: int | None = Field(ge=1)
    limit_type: ContextLimitType
    source: ContextSource
    known: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    usage_source: ContextUsageSource | None = None
    used_tokens: int | None = Field(default=None, ge=0)
    used_token_source: ContextUsedTokenSource | None = None
    input_usage_ratio: float | None = Field(default=None, ge=0)
    output_usage_ratio: float | None = Field(default=None, ge=0)
    usage_ratio: float | None = Field(default=None, ge=0)
    usage_ratio_basis: ContextUsageRatioBasis | None = None
    display_state: ContextDisplayState | None = None

    model_config = ConfigDict(extra="ignore")


class UsageTotals(UsageModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    generated_images: int = 0
    request_count: int = 0


class UsageBreakdownItem(UsageModel):
    key: str
    totals: UsageTotals


class UsageSeriesPoint(UsageModel):
    start: datetime
    end: datetime
    totals: UsageTotals


class UsageCoverage(UsageModel):
    provider_reported_requests: int = 0
    mixed_requests: int = 0
    locally_estimated_requests: int = 0
    unavailable_requests: int = 0
    requests_with_known_total: int = 0
    total_requests: int = 0
    known_total_ratio: float = 0.0


class ConversationUsageItem(UsageModel):
    conversation_id: UUID
    title: str | None = None
    totals: UsageTotals


class UsageRange(UsageModel):
    from_: datetime = Field(alias="from")
    to: datetime
    bucket: UsageBucket
    timezone: str


class UsageDashboard(UsageModel):
    totals: UsageTotals
    outcomes: list[UsageBreakdownItem]
    series: list[UsageSeriesPoint]
    by_provider: list[UsageBreakdownItem]
    by_model: list[UsageBreakdownItem]
    by_operation: list[UsageBreakdownItem]
    by_agent: list[UsageBreakdownItem]
    top_conversations: list[ConversationUsageItem]
    coverage: UsageCoverage
    range: UsageRange
    generated_at: datetime


class ConversationUsage(UsageModel):
    totals: UsageTotals
    by_provider: list[UsageBreakdownItem]
    by_model: list[UsageBreakdownItem]
    coverage: UsageCoverage
    latest_context_window: ContextWindowMetadata | None
    range: UsageRange
    generated_at: datetime


ConversationUsageResponse = ConversationUsage


class UsageQuery(UsageModel):
    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None
    bucket: UsageBucket = "day"
    timezone: str = "UTC"

    @field_validator("from_", "to", mode="before")
    @classmethod
    def _require_numeric_offset(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, str) and re.search(r"[+-]\d{2}:\d{2}$", value) is None:
            raise ValueError("usage range boundaries must include a numeric UTC offset")
        return value

    @model_validator(mode="after")
    def _validate_explicit_bounds(self) -> "UsageQuery":
        if (self.from_ is None) != (self.to is None):
            raise ValueError("from and to must be supplied together")
        for value in (self.from_, self.to):
            if value is None:
                continue
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("usage range boundaries must include a numeric UTC offset")
            if value.second != 0 or value.microsecond != 0:
                raise ValueError("usage range boundaries must have zero seconds and microseconds")
        return self


class UsageDashboardQuery(UsageQuery):
    conversation_id: UUID | None = None


class ConversationUsageQuery(UsageQuery):
    pass


UsageDashboardQueryParams = Annotated[UsageDashboardQuery, Query()]
ConversationUsageQueryParams = Annotated[ConversationUsageQuery, Query()]


__all__ = [
    "ContextWindowMetadata",
    "ConversationUsage",
    "ConversationUsageItem",
    "ConversationUsageQuery",
    "ConversationUsageQueryParams",
    "ConversationUsageResponse",
    "UsageBreakdownItem",
    "UsageBucket",
    "UsageCoverage",
    "UsageDashboard",
    "UsageDashboardQuery",
    "UsageDashboardQueryParams",
    "UsageModel",
    "UsageQuery",
    "UsageRange",
    "UsageSeriesPoint",
    "UsageTotals",
]
