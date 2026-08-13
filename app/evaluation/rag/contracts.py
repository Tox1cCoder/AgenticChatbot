"""Stable value objects exchanged between RAG targets and evaluators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGEvaluationInput:
    question: str
    user_id: str
    conversation_id: str


@dataclass(frozen=True)
class RelevantSpan:
    document_id: str
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class RAGEvaluationReference:
    answer: str | None = None
    relevant_document_ids: frozenset[str] = frozenset()
    relevant_chunk_ids: frozenset[str] = frozenset()
    relevant_spans: tuple[RelevantSpan, ...] = ()
    should_abstain: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "relevant_document_ids", frozenset(self.relevant_document_ids))
        object.__setattr__(self, "relevant_chunk_ids", frozenset(self.relevant_chunk_ids))
        object.__setattr__(self, "relevant_spans", tuple(self.relevant_spans))


@dataclass(frozen=True)
class RetrievalTrace:
    document_id: str
    chunk_id: str | None
    rank: int
    score: float | None


@dataclass(frozen=True)
class EvidenceTrace:
    evidence_id: str
    document_id: str
    chunk_id: str | None
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class ClaimTrace:
    text: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


@dataclass(frozen=True)
class RAGEvaluationOutput:
    answer: str
    abstained: bool
    candidates: tuple[RetrievalTrace, ...]
    evidence: tuple[EvidenceTrace, ...]
    claims: tuple[ClaimTrace, ...]
    citations_valid: bool
    tool_trajectory: tuple[str, ...]
    stage_ms: Mapping[str, float]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "tool_trajectory", tuple(self.tool_trajectory))
        object.__setattr__(self, "stage_ms", dict(self.stage_ms))
