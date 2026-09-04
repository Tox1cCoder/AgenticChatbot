"""Typed contracts for the routing evaluation dataset, report, and review.

Three separable artifacts, deliberately not one:

* the **dataset** (`golden_v1.jsonl`) is labelled data and changes rarely;
* the **review** (`golden_v1.review.json`) is a human's statement about a
  specific dataset, bound to it by content hash;
* the **report** is one measured run against one exact provider/model/inventory
  tuple, and is stale the moment any of those move.

Binding the review to `dataset_sha256` rather than to a filename is what stops
an edited dataset from inheriting yesterday's approval. Binding the report to
the tuple is what stops a good result on one model from releasing another.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

__all__ = [
    "REQUIRED_CATEGORIES",
    "REQUIRED_LANGUAGES",
    "RoutingCategory",
    "RoutingDatasetReview",
    "RoutingEvalCase",
    "RoutingEvaluationReport",
    "RoutingLanguage",
    "RoutingPrediction",
    "RoutingReleaseDecision",
    "dataset_sha256",
    "load_dataset",
    "load_review",
]

RoutingLanguage = Literal["en", "th", "vi", "zh", "ja", "ar", "mixed"]

#: Every language the gate requires coverage for. `mixed` is its own class
#: rather than a variant of the others: code-switched input is where a router
#: trained on monolingual prompts actually breaks.
REQUIRED_LANGUAGES: tuple[RoutingLanguage, ...] = ("en", "th", "vi", "zh", "ja", "ar", "mixed")

RoutingCategory = Literal[
    "general_chat",
    "current_information",
    "document_qa",
    "planning",
    "canvas",
    "image_generation",
    "custom_agent",
    "ambiguous_followup",
]

#: The intent categories the gate requires coverage for. A closed set, and the
#: field is typed against it, because a free-text category lets a typo split
#: one category into two thin ones that each fail coverage for no real reason
#: — and lets an omitted category vanish from the check entirely.
REQUIRED_CATEGORIES: tuple[RoutingCategory, ...] = (
    "general_chat",
    "current_information",
    "document_qa",
    "planning",
    "canvas",
    "image_generation",
    "custom_agent",
    "ambiguous_followup",
)


class RoutingEvalCase(BaseModel):
    """One labelled routing case.

    ``acceptable_agent_ids`` is not a hedge. Some turns genuinely have more
    than one defensible target — a question about an attached document that
    could also be answered from the web — and scoring those as failures would
    reward a router that guesses the dataset author's habits rather than one
    that behaves sensibly. The canonical label is still exactly one agent, and
    macro-F1 is computed against it alone.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=64)
    language: RoutingLanguage
    category: RoutingCategory
    message: str = Field(min_length=1, max_length=4000)
    context: dict[str, JsonValue] = Field(default_factory=dict)
    primary_agent_id: str = Field(min_length=1, max_length=160)
    acceptable_agent_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def _message_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @model_validator(mode="after")
    def _primary_is_acceptable(self) -> RoutingEvalCase:
        if len(set(self.acceptable_agent_ids)) != len(self.acceptable_agent_ids):
            raise ValueError("acceptable_agent_ids must not repeat an agent")
        if self.primary_agent_id not in self.acceptable_agent_ids:
            raise ValueError(
                f"primary_agent_id {self.primary_agent_id!r} is not in its own acceptable set"
            )
        return self


class RoutingPrediction(BaseModel):
    """What the router actually did for one case.

    ``predicted_agent_id`` is ``None`` when routing failed outright. That is a
    different thing from routing to the wrong agent, and the metrics keep them
    apart: a failure is scored wrong, but it is not a *substitution*.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=64)
    predicted_agent_id: str | None = None
    attempts: int = Field(ge=1, le=2)
    structured_success: bool
    error_code: str | None = None


class RoutingDatasetReview(BaseModel):
    """A human's approval of one exact dataset.

    Exactly five fields, and no free-text notes field: anything else invites
    an approval that is really a caveat. If the labels need discussion, that
    belongs in review, not in the manifest the gate reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    approved: bool
    reviewing_team: str = Field(min_length=1, max_length=120)
    reviewed_at: datetime
    label_guideline_version: str = Field(min_length=1, max_length=40)


class RoutingReleaseDecision(BaseModel):
    """Whether this report may release, and every reason it may not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    reason_codes: tuple[str, ...] = ()


class RoutingEvaluationReport(BaseModel):
    """One measured run, pinned to the tuple it was measured against."""

    model_config = ConfigDict(extra="forbid")

    dataset_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=160)
    inventory_version: str = Field(min_length=1, max_length=128)
    generated_at: datetime
    case_count: int = Field(ge=0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    accuracy_by_language: dict[str, float] = Field(default_factory=dict)
    first_attempt_structured_success: float = Field(ge=0.0, le=1.0)
    after_retry_structured_success: float = Field(ge=0.0, le=1.0)
    silent_chat_substitutions: int = Field(ge=0)
    finalizer_bypasses: int = Field(ge=0)
    unknown_published_evidence_ids: int = Field(ge=0)
    predictions: tuple[RoutingPrediction, ...] = ()

    @field_validator("generated_at")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        # Freshness is a comparison against `now`. A naive stamp would compare
        # against whatever the checking host's clock happens to mean.
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value


def dataset_sha256(path: str | Path) -> str:
    """Content hash of the dataset file exactly as it sits on disk.

    Hashing the bytes, not the parsed cases, is intentional: a reordering or a
    whitespace edit is still a different artifact than the one a human read.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_dataset_lines(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if line:
                yield number, line


def load_dataset(path: str | Path) -> tuple[RoutingEvalCase, ...]:
    """Parse and validate the whole JSONL dataset, or refuse it.

    Errors name the line, because a 210-case file is not something anyone
    debugs from a bare ``ValidationError``.
    """
    dataset_path = Path(path)
    cases: list[RoutingEvalCase] = []
    seen: set[str] = set()

    for number, line in _iter_dataset_lines(dataset_path):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{dataset_path}: line {number} is not valid JSON: {exc}") from exc
        try:
            case = RoutingEvalCase.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"{dataset_path}: line {number} is not a valid case: {exc}") from exc
        if case.case_id in seen:
            raise ValueError(
                f"{dataset_path}: line {number} has a duplicate case_id {case.case_id!r}"
            )
        seen.add(case.case_id)
        cases.append(case)

    return tuple(cases)


def load_review(path: str | Path) -> RoutingDatasetReview:
    """Parse the review manifest. A malformed manifest is not an approval."""
    review_path = Path(path)
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    return RoutingDatasetReview.model_validate(payload)
