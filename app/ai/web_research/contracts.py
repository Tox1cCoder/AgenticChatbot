"""Small, provider-neutral contracts for one turn's web evidence."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.ai.web_query_contract import Freshness

ResearchMode = Literal["none", "quick", "agentic"]
VisualIntent = Literal["none", "figure", "comparison", "gallery"]
SourceStatus = Literal["search_result", "opened", "snippet_only"]
EvidenceStatus = Literal["success", "partial", "empty", "failed"]

_IMAGE_MIME_PATTERN = r"^image/(png|jpeg|webp|gif)$"
_DELIVERY_PATTERN = (
    r"^/web-images/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchScope(FrozenModel):
    conversation_id: str = Field(min_length=1, max_length=160)
    user_id: str = Field(min_length=1, max_length=160)
    logical_turn_id: str = Field(min_length=1, max_length=160)
    device_id: str | None = Field(default=None, max_length=160)


class ResearchRequest(FrozenModel):
    query: str = Field(min_length=3, max_length=400)
    objective: str = Field(min_length=3, max_length=500)
    mode: ResearchMode = "quick"
    freshness: Freshness = "timeless"
    start_date: date | None = None
    end_date: date | None = None
    locale: str | None = Field(default=None, max_length=32)
    include_domains: tuple[str, ...] = ()
    visual_intent: VisualIntent = "none"
    image_query: str | None = Field(default=None, max_length=300)


class ProviderSource(FrozenModel):
    provider: str = Field(min_length=1, max_length=64)
    url: str = Field(min_length=1, max_length=4096)
    title: str | None = Field(default=None, max_length=500)
    snippet: str | None = Field(default=None, max_length=4000)
    rank: int = Field(ge=1)
    query_index: int = Field(ge=1)
    published_at: datetime | None = None


class SourceRecord(FrozenModel):
    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    url: HttpUrl
    title: str | None = Field(default=None, max_length=500)
    snippet: str | None = Field(default=None, max_length=4000)
    status: SourceStatus
    published_at: datetime | None = None
    provider: str = Field(min_length=1, max_length=64)
    query_index: int = Field(ge=1)


class ProviderImageCandidate(FrozenModel):
    provider: str = Field(min_length=1, max_length=64)
    image_url: str = Field(min_length=1, max_length=4096)
    source_url: str = Field(min_length=1, max_length=4096)
    title: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=1000)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    rank: int = Field(ge=1)
    published_at: datetime | None = None


class ImageCandidateRecord(FrozenModel):
    candidate_id: str = Field(pattern=r"^I[1-9][0-9]*$")
    source_id: str = Field(pattern=r"^S[1-9][0-9]*$")
    delivery_url: str = Field(pattern=_DELIVERY_PATTERN)
    mime_type: str = Field(pattern=_IMAGE_MIME_PATTERN)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    byte_size: int = Field(ge=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str | None = Field(default=None, max_length=500)
    description: str | None = Field(default=None, max_length=1000)
    provider: str = Field(min_length=1, max_length=64)


class ResearchFailure(FrozenModel):
    operation: Literal["search", "open", "image_search", "image_fetch"]
    provider: str = Field(min_length=1, max_length=64)
    code: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    retryable: bool = False
    source_id: str | None = Field(default=None, pattern=r"^S[1-9][0-9]*$")


class WebEvidenceBundle(FrozenModel):
    status: EvidenceStatus
    mode: ResearchMode
    visual_intent: VisualIntent
    operation_index: int = Field(ge=1)
    sources: tuple[SourceRecord, ...] = ()
    images: tuple[ImageCandidateRecord, ...] = ()
    failures: tuple[ResearchFailure, ...] = ()
    providers_used: tuple[str, ...] = ()
    reused: bool = False
    omitted_source_count: int = Field(default=0, ge=0)
    omitted_image_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_membership_and_limits(self) -> WebEvidenceBundle:
        source_ids = [source.source_id for source in self.sources]
        image_ids = [image.candidate_id for image in self.images]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("duplicate source ID")
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("duplicate image candidate ID")
        known_sources = set(source_ids)
        for image in self.images:
            if image.source_id not in known_sources:
                raise ValueError(f"image {image.candidate_id} has unknown source {image.source_id}")

        source_limit = {"none": 0, "quick": 5, "agentic": 8}[self.mode]
        image_limit = 6 if self.visual_intent == "gallery" else 4
        if self.visual_intent == "none":
            image_limit = 0
        if len(self.sources) > source_limit:
            raise ValueError(f"source limit for {self.mode} is {source_limit}")
        if len(self.images) > image_limit:
            raise ValueError(f"image limit for {self.visual_intent} is {image_limit}")
        return self


__all__ = [
    "EvidenceStatus",
    "ImageCandidateRecord",
    "ProviderImageCandidate",
    "ProviderSource",
    "ResearchFailure",
    "ResearchMode",
    "ResearchRequest",
    "ResearchScope",
    "SourceRecord",
    "SourceStatus",
    "VisualIntent",
    "WebEvidenceBundle",
]
