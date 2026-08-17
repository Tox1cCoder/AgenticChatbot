"""Immutable, model-budgeted evidence assembled from authorized retrieval rows."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal
from uuid import UUID

from app.ai.token_counter import TokenCounter
from app.services.rag_retrieval import RetrievalCandidate, RetrievalScope

_ATOMIC_KINDS = frozenset({"table", "image", "equation"})
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    document_id: UUID
    chunk_id: UUID | None
    image_id: UUID | None
    filename: str
    page_start: int | None
    page_end: int | None
    section_path: tuple[str, ...]
    modality: Literal["text", "image"]
    content: str
    trace_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class EvidencePack:
    records: tuple[EvidenceRecord, ...]
    token_count: int
    omitted_count: int
    truncated_count: int = 0
    count_strategy: str = "unknown"

    @property
    def evidence_ids(self) -> frozenset[str]:
        return frozenset(record.evidence_id for record in self.records)

    def to_tool_text(self) -> str:
        return _serialize_records(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [
                {
                    "evidence_id": record.evidence_id,
                    "document_id": str(record.document_id),
                    "chunk_id": str(record.chunk_id) if record.chunk_id else None,
                    "image_id": str(record.image_id) if record.image_id else None,
                    "filename": record.filename,
                    "page_start": record.page_start,
                    "page_end": record.page_end,
                    "section_path": list(record.section_path),
                    "modality": record.modality,
                    "content": record.content,
                    "trace_metadata": dict(record.trace_metadata),
                }
                for record in self.records
            ],
            "evidence_ids": [record.evidence_id for record in self.records],
            "token_count": self.token_count,
            "omitted_count": self.omitted_count,
            "truncated_count": self.truncated_count,
            "count_strategy": self.count_strategy,
            "tool_text": self.to_tool_text(),
        }


class EvidenceAssembler:
    """Pack canonical candidates using one counter over the exact rendered text."""

    def __init__(
        self,
        *,
        token_counter: Any | None = None,
        provider: str,
        model: str,
        repository: Any | None = None,
        overlap_threshold: float = 0.85,
        max_neighbors: int = 2,
    ) -> None:
        self.token_counter = token_counter or TokenCounter()
        self.provider = provider
        self.model = model
        self.repository = repository
        self.overlap_threshold = min(1.0, max(0.0, float(overlap_threshold)))
        self.max_neighbors = max(0, int(max_neighbors))

    def assemble(
        self,
        question: str,
        candidates: Sequence[RetrievalCandidate | Mapping[str, Any] | Any],
        *,
        max_tokens: int,
        subquestions: Sequence[str] = (),
        scope: RetrievalScope | None = None,
    ) -> EvidencePack:
        del question
        allowance = max(0, int(max_tokens))
        canonical, duplicate_count = self._deduplicate(candidates)
        ordered = self._coverage_order(canonical, subquestions)
        selected: list[EvidenceRecord] = []
        omitted = duplicate_count
        truncated = 0
        selected, newly_omitted, newly_truncated = self._pack_candidates(
            selected,
            ordered,
            allowance,
        )
        omitted += newly_omitted
        truncated += newly_truncated

        if self.repository is not None and scope is not None and self.max_neighbors:
            for seed in tuple(selected):
                if not self._minimum_complete_record_fits(selected, seed, allowance):
                    break
                if seed.chunk_id is None:
                    continue
                expanded = self.repository.get_context_expansion_for_scope(
                    seed.chunk_id,
                    document_id=seed.document_id,
                    user_id=scope.user_id,
                    conversation_id=scope.conversation_id,
                    max_neighbors=self.max_neighbors,
                )
                expansion_candidates, expansion_duplicates = self._deduplicate(
                    expanded,
                    existing_records=selected,
                )
                omitted += expansion_duplicates
                selected, newly_omitted, newly_truncated = self._pack_candidates(
                    selected,
                    expansion_candidates,
                    allowance,
                )
                omitted += newly_omitted
                truncated += newly_truncated

        rendered = _serialize_records(selected)
        token_count, strategy = self._count(rendered)
        return EvidencePack(
            records=tuple(selected),
            token_count=token_count,
            omitted_count=omitted,
            truncated_count=truncated,
            count_strategy=strategy,
        )

    def _pack_candidates(
        self,
        selected: list[EvidenceRecord],
        candidates: Sequence[RetrievalCandidate],
        allowance: int,
    ) -> tuple[list[EvidenceRecord], int, int]:
        """Pack complete records for coverage before spending remainder on truncation."""
        deferred: list[RetrievalCandidate] = []
        omitted = 0
        truncated = 0
        for candidate in candidates:
            record = self._record(candidate, len(selected) + 1)
            if self._count(_serialize_records((*selected, record)))[0] <= allowance:
                selected.append(record)
            else:
                deferred.append(candidate)

        for candidate in deferred:
            record = self._record(candidate, len(selected) + 1)
            accepted, was_truncated = self._fit_record(selected, record, allowance)
            if accepted is None:
                omitted += 1
                continue
            selected.append(accepted)
            truncated += int(was_truncated)
        return selected, omitted, truncated

    def _minimum_complete_record_fits(
        self,
        selected: Sequence[EvidenceRecord],
        seed: EvidenceRecord,
        allowance: int,
    ) -> bool:
        minimum = replace(
            seed,
            evidence_id=f"E{len(selected) + 1}",
            content="x",
        )
        return self._count(_serialize_records((*selected, minimum)))[0] <= allowance

    def _fit_record(
        self,
        selected: Sequence[EvidenceRecord],
        record: EvidenceRecord,
        allowance: int,
    ) -> tuple[EvidenceRecord | None, bool]:
        if self._count(_serialize_records((*selected, record)))[0] <= allowance:
            return record, False
        kind = str(record.trace_metadata.get("atomic_kind") or "").casefold()
        if kind in _ATOMIC_KINDS:
            return None, False

        words = record.content.split()
        low, high = 0, len(words)
        fitted: EvidenceRecord | None = None
        while low <= high:
            midpoint = (low + high) // 2
            content = " ".join(words[:midpoint]).strip()
            if content:
                content = f"{content} [TRUNCATED]"
            trial = replace(record, content=content)
            if content and self._count(_serialize_records((*selected, trial)))[0] <= allowance:
                fitted = trial
                low = midpoint + 1
            else:
                high = midpoint - 1
        return fitted, fitted is not None

    def _deduplicate(
        self,
        candidates: Sequence[RetrievalCandidate | Mapping[str, Any] | Any],
        *,
        existing_records: Sequence[EvidenceRecord] = (),
    ) -> tuple[list[RetrievalCandidate], int]:
        kept: list[RetrievalCandidate] = []
        seen_ids = {
            (record.modality, str(record.image_id or record.chunk_id))
            for record in existing_records
            if record.image_id or record.chunk_id
        }
        seen_hashes = {_content_hash(record.content) for record in existing_records}
        seen_tokens = [_normalized_tokens(record.content) for record in existing_records]
        omitted = 0
        for raw_candidate in candidates:
            candidate = _coerce_candidate(raw_candidate)
            if candidate is None or not candidate.content.strip():
                omitted += 1
                continue
            canonical_id = (
                candidate.image_id if candidate.modality == "image" else candidate.chunk_id
            )
            id_key = (candidate.modality, str(canonical_id)) if canonical_id else None
            tokens = _normalized_tokens(candidate.content)
            content_hash = _content_hash(candidate.content)
            if (
                (id_key is not None and id_key in seen_ids)
                or content_hash in seen_hashes
                or any(
                _content_overlaps(tokens, prior, self.overlap_threshold) for prior in seen_tokens
                )
            ):
                omitted += 1
                continue
            if id_key is not None:
                seen_ids.add(id_key)
            seen_hashes.add(content_hash)
            seen_tokens.append(tokens)
            kept.append(candidate)
        return kept, omitted

    @staticmethod
    def _coverage_order(
        candidates: Sequence[RetrievalCandidate], subquestions: Sequence[str]
    ) -> list[RetrievalCandidate]:
        remaining = list(candidates)
        ordered: list[RetrievalCandidate] = []
        normalized_subquestions = tuple(
            normalized
            for subquestion in subquestions
            if (normalized := str(subquestion).strip().casefold())
        )
        covered_subquestions: set[str] = set()

        # Seed relevance with one subquestion, then cover documents before adding
        # more same-document subquestion matches. This prevents a tight pack from
        # spending every slot on one source while retaining deterministic order.
        for normalized in normalized_subquestions:
            match = next(
                (
                    candidate
                    for candidate in remaining
                    if _candidate_matches_subquestion(candidate, normalized)
                ),
                None,
            )
            if match is not None:
                ordered.append(match)
                remaining.remove(match)
                covered_subquestions.update(
                    subquestion
                    for subquestion in normalized_subquestions
                    if _candidate_matches_subquestion(match, subquestion)
                )
                break

        covered_documents = {candidate.document_id for candidate in ordered}
        for candidate in tuple(remaining):
            if candidate.document_id not in covered_documents:
                ordered.append(candidate)
                remaining.remove(candidate)
                covered_documents.add(candidate.document_id)
                covered_subquestions.update(
                    subquestion
                    for subquestion in normalized_subquestions
                    if _candidate_matches_subquestion(candidate, subquestion)
                )

        for normalized in normalized_subquestions:
            if normalized in covered_subquestions:
                continue
            match = next(
                (
                    candidate
                    for candidate in remaining
                    if _candidate_matches_subquestion(candidate, normalized)
                ),
                None,
            )
            if match is not None:
                ordered.append(match)
                remaining.remove(match)
                covered_subquestions.add(normalized)
        ordered.extend(remaining)
        return ordered

    @staticmethod
    def _record(candidate: RetrievalCandidate, ordinal: int) -> EvidenceRecord:
        metadata = dict(candidate.metadata or {})
        atomic_kind = _atomic_kind(candidate.modality, metadata)
        trace = {
            "dense_rank": candidate.dense_rank,
            "dense_score": candidate.dense_score,
            "lexical_rank": candidate.lexical_rank,
            "lexical_score": candidate.lexical_score,
            "fused_score": candidate.fused_score,
            "rerank_score": candidate.rerank_score,
            "chunk_index": candidate.chunk_index,
            "content_sha256": hashlib.sha256(
                _normalize_content(candidate.content).encode("utf-8")
            ).hexdigest(),
            "atomic_kind": atomic_kind,
            **{
                key: metadata[key]
                for key in ("kind", "expansion_kind", "parent_chunk_id")
                if key in metadata
            },
        }
        return EvidenceRecord(
            evidence_id=f"E{ordinal}",
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            image_id=candidate.image_id,
            filename=candidate.filename,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            section_path=tuple(candidate.section_path),
            modality=candidate.modality,
            content=candidate.content.strip(),
            trace_metadata=MappingProxyType(trace),
        )

    def _count(self, text: str) -> tuple[int, str]:
        result = self.token_counter.count_text(
            provider=self.provider,
            model=self.model,
            text=text,
        )
        return int(result.tokens), str(result.strategy)


def _serialize_records(records: Sequence[EvidenceRecord]) -> str:
    parts: list[str] = []
    for record in records:
        metadata_json = json.dumps(
            {
                "modality": record.modality,
                "pages": [record.page_start, record.page_end],
                "section_path": list(record.section_path),
                "source": record.filename,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        content_json = json.dumps(
            record.content,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        parts.extend(
            [
                f"BEGIN UNTRUSTED EVIDENCE {record.evidence_id}",
                f"metadata_json={metadata_json}",
                f"content_json={content_json}",
                f"END UNTRUSTED EVIDENCE {record.evidence_id}",
            ]
        )
    return "\n".join(parts)


def _normalize_content(content: str) -> str:
    return " ".join(str(content or "").casefold().split())


def _content_hash(content: str) -> str:
    return hashlib.sha256(_normalize_content(content).encode("utf-8")).hexdigest()


def _normalized_tokens(content: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(_normalize_content(content)))


def _content_overlaps(
    left: tuple[str, ...], right: tuple[str, ...], threshold: float
) -> bool:
    if not left or not right:
        return False
    shorter = min(len(left), len(right))
    minimum_overlap = max(3, math.ceil(shorter * threshold))
    for size in range(shorter, minimum_overlap - 1, -1):
        if left[-size:] == right[:size] or right[-size:] == left[:size]:
            return True
    return False


def _atomic_kind(modality: str, metadata: Mapping[str, Any]) -> str | None:
    if modality == "image":
        return "image"
    if metadata.get("has_tables") or metadata.get("contains_table"):
        return "table"

    def _kind_from(mapping: Mapping[str, Any]) -> str | None:
        for key in ("kind", "block_type", "element_type", "content_type", "type"):
            value = str(mapping.get(key) or "").strip().casefold()
            if value in _ATOMIC_KINDS:
                return value
        return None

    direct = _kind_from(metadata)
    if direct:
        return direct
    provenance = metadata.get("provenance")
    if isinstance(provenance, Mapping):
        return _kind_from(provenance)
    return None


def _candidate_subquestions(candidate: RetrievalCandidate) -> frozenset[str]:
    raw = (candidate.metadata or {}).get("subquestions") or ()
    return frozenset(str(item).strip().casefold() for item in raw if str(item).strip())


def _candidate_matches_subquestion(
    candidate: RetrievalCandidate,
    normalized_subquestion: str,
) -> bool:
    return (
        normalized_subquestion in _candidate_subquestions(candidate)
        or normalized_subquestion in candidate.content.casefold()
    )


def _optional_uuid(value: Any) -> UUID | None:
    if value is None or value == "":
        return None
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _coerce_candidate(raw: Any) -> RetrievalCandidate | None:
    if isinstance(raw, RetrievalCandidate):
        return raw
    getter = (
        raw.get
        if isinstance(raw, Mapping)
        else lambda key, default=None: getattr(raw, key, default)
    )
    document_id = _optional_uuid(getter("document_id"))
    if document_id is None:
        return None
    document = getter("document")
    metadata = dict(getter("metadata") or getter("chunk_metadata") or {})
    return RetrievalCandidate(
        document_id=document_id,
        chunk_id=_optional_uuid(getter("chunk_id") or getter("id")),
        image_id=_optional_uuid(getter("image_id")),
        modality="image" if getter("modality") == "image" else "text",
        content=str(getter("content") or ""),
        filename=str(
            getter("filename")
            or getter("source")
            or getattr(document, "filename", None)
            or "unknown"
        ),
        page_start=getter("page_start") or getter("page_number"),
        page_end=getter("page_end") or getter("page_number"),
        section_path=tuple(getter("section_path") or ()),
        dense_rank=getter("dense_rank"),
        dense_score=getter("dense_score"),
        lexical_rank=getter("lexical_rank"),
        lexical_score=getter("lexical_score"),
        fused_score=float(getter("fused_score") or 0.0),
        rerank_score=getter("rerank_score"),
        chunk_index=getter("chunk_index"),
        metadata=metadata,
    )
