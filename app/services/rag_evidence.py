"""Immutable, model-budgeted evidence assembled from authorized retrieval rows."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal
from uuid import UUID

from app.ai.token_counter import TokenCounter
from app.services.rag_retrieval import RetrievalCandidate, RetrievalScope

logger = logging.getLogger(__name__)

_ATOMIC_KINDS = frozenset({"table", "image", "equation"})


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
        metrics: Any | None = None,
    ) -> None:
        self.token_counter = token_counter or TokenCounter()
        self.provider = provider
        self.model = model
        self.repository = repository
        self.overlap_threshold = min(1.0, max(0.0, float(overlap_threshold)))
        self.max_neighbors = max(0, int(max_neighbors))
        self.metrics = metrics

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
        started_at = time.monotonic()
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
        self._record_stage(time.monotonic() - started_at)
        self._record_evidence_tokens(token_count)
        return EvidencePack(
            records=tuple(selected),
            token_count=token_count,
            omitted_count=omitted,
            truncated_count=truncated,
            count_strategy=strategy,
        )

    async def assemble_exact(
        self,
        question: str,
        candidates: Sequence[RetrievalCandidate | Mapping[str, Any] | Any],
        *,
        max_tokens: int,
        subquestions: Sequence[str] = (),
        scope: RetrievalScope | None = None,
    ) -> EvidencePack:
        """Assemble locally, then reconcile the final text with one provider call.

        Selection and every incremental fit test use the conservative local
        strategy, so assembly issues no provider round trips and the pack cannot
        overshoot the allowance. Exactly one native call then replaces the
        conservative bound with the provider's exact count of the emitted text,
        which is what the caller charges against the cumulative allowance.
        """
        pack = self.assemble(
            question,
            candidates,
            max_tokens=max_tokens,
            subquestions=subquestions,
            scope=scope,
        )
        count_exact = getattr(self.token_counter, "count_text_exact", None)
        if not pack.records or not callable(count_exact):
            # Counters are injected duck-typed (restart fallbacks, per-provider
            # adapters). One without an exact entry point keeps its local count.
            return pack
        exact = await count_exact(
            provider=self.provider,
            model=self.model,
            text=pack.to_tool_text(),
        )
        return replace(
            pack,
            token_count=int(exact.tokens),
            count_strategy=str(exact.strategy),
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
            content_hash = _content_hash(candidate.content)
            if (
                (id_key is not None and id_key in seen_ids)
                or content_hash in seen_hashes
            ):
                omitted += 1
                continue
            if id_key is not None:
                seen_ids.add(id_key)
            seen_hashes.add(content_hash)
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
        covered_documents: set[UUID] = set()

        # Greedily maximize joint marginal coverage. A candidate that adds both
        # a missing subquestion and a missing document wins over one that adds
        # only either axis. Equal one-axis gains prefer document coverage, then
        # the original retrieval order, giving deterministic ties without letting
        # an early run of unseen documents consume every tight-budget slot.
        while remaining:
            best_index = 0
            best_key = (-1, -1, -1)
            for index, candidate in enumerate(remaining):
                matched = {
                    subquestion
                    for subquestion in normalized_subquestions
                    if _candidate_matches_subquestion(candidate, subquestion)
                }
                new_subquestions = len(matched - covered_subquestions)
                new_document = int(candidate.document_id not in covered_documents)
                key = (new_subquestions + new_document, new_document, -index)
                if key > best_key:
                    best_index = index
                    best_key = key

            selected = remaining.pop(best_index)
            ordered.append(selected)
            covered_documents.add(selected.document_id)
            covered_subquestions.update(
                subquestion
                for subquestion in normalized_subquestions
                if _candidate_matches_subquestion(selected, subquestion)
            )
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

    def _record_stage(self, elapsed_seconds: float) -> None:
        recorder = getattr(self.metrics, "stage", None)
        if not callable(recorder):
            return
        try:
            recorder(
                "evidence_assembly",
                elapsed_seconds=elapsed_seconds,
                labels={"provider": self.provider, "model": self.model},
            )
        except Exception:
            logger.exception("Failed to record evidence-assembly stage metric")

    def _record_evidence_tokens(self, token_count: int) -> None:
        recorder = getattr(self.metrics, "evidence_tokens", None)
        if not callable(recorder):
            return
        try:
            recorder(token_count)
        except Exception:
            logger.exception("Failed to record evidence-pack token metric")


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


def _atomic_kind(modality: str, metadata: Mapping[str, Any]) -> str | None:
    if modality == "image":
        return "image"
    if metadata.get("has_tables") or metadata.get("contains_table"):
        return "table"
    if metadata.get("has_images") or metadata.get("contains_image"):
        return "image"

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
        nested = _kind_from(provenance)
        if nested:
            return nested
    block_provenance = metadata.get("block_provenance")
    if isinstance(block_provenance, Sequence) and not isinstance(
        block_provenance,
        (str, bytes),
    ):
        for block in block_provenance:
            if isinstance(block, Mapping) and (nested := _kind_from(block)):
                return nested
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
    block_provenance = getter("block_provenance")
    if block_provenance:
        metadata["block_provenance"] = list(block_provenance)
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
