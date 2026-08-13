"""Pure deterministic metrics for RAG evaluation records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .contracts import RAGEvaluationOutput, RAGEvaluationReference, RelevantSpan, RetrievalTrace


def _ordered(candidates: Iterable[RetrievalTrace]) -> list[RetrievalTrace]:
    return sorted(candidates, key=lambda trace: trace.rank)


def _retrieval_level_metrics(
    candidates: Sequence[RetrievalTrace],
    relevant_ids: frozenset[str],
    attribute: str,
    ks: Sequence[int],
) -> dict[str, float]:
    prefix = attribute.replace("_id", "")
    metrics: dict[str, float] = {}
    seen_ids: set[str | None] = set()
    unique_candidates: list[RetrievalTrace] = []
    for candidate in candidates:
        identifier = getattr(candidate, attribute)
        if identifier in seen_ids:
            continue
        seen_ids.add(identifier)
        unique_candidates.append(candidate)
    first_rank: int | None = None
    for position, candidate in enumerate(unique_candidates, start=1):
        identifier = getattr(candidate, attribute)
        if identifier in relevant_ids:
            first_rank = position
            break
    metrics[f"{prefix}_reciprocal_rank"] = 1.0 / first_rank if first_rank else 0.0
    metrics[f"{prefix}_mrr"] = metrics[f"{prefix}_reciprocal_rank"]
    for k in ks:
        if k < 1:
            raise ValueError("k values must be positive")
        returned = {getattr(candidate, attribute) for candidate in unique_candidates[:k]}
        hits = len(returned & relevant_ids)
        metrics[f"{prefix}_recall_at_{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0
        metrics[f"{prefix}_hit_rate_at_{k}"] = 1.0 if hits else 0.0
        dcg = sum(
            1.0 / math.log2(position + 1)
            for position, candidate in enumerate(unique_candidates[:k], start=1)
            if getattr(candidate, attribute) in relevant_ids
        )
        ideal_positions = range(1, min(k, len(relevant_ids)) + 1)
        ideal = sum(1.0 / math.log2(position + 1) for position in ideal_positions)
        metrics[f"{prefix}_ndcg_at_{k}"] = dcg / ideal if ideal else 0.0
    return metrics


def retrieval_metrics(
    candidates: Iterable[RetrievalTrace],
    reference: RAGEvaluationReference,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Return document and chunk retrieval metrics keyed by stable IDs."""
    ordered = _ordered(candidates)
    return {
        **_retrieval_level_metrics(ordered, reference.relevant_document_ids, "document_id", ks),
        **_retrieval_level_metrics(ordered, reference.relevant_chunk_ids, "chunk_id", ks),
    }


def citation_metrics(
    output: RAGEvaluationOutput, reference: RAGEvaluationReference | None = None
) -> dict[str, float]:
    """Score machine-resolvable evidence references without an LLM judge."""
    known_ids = {evidence.evidence_id for evidence in output.evidence}
    cited_ids = [evidence_id for claim in output.claims for evidence_id in claim.evidence_ids]
    valid_citations = sum(evidence_id in known_ids for evidence_id in cited_ids)
    cited_claims = sum(bool(claim.evidence_ids) for claim in output.claims)
    valid_claims = sum(
        bool(claim.evidence_ids)
        and all(evidence_id in known_ids for evidence_id in claim.evidence_ids)
        for claim in output.claims
    )
    evidence_document_ids = {evidence.document_id for evidence in output.evidence}
    if reference and reference.relevant_document_ids:
        citation_recall = len(evidence_document_ids & reference.relevant_document_ids) / len(
            reference.relevant_document_ids
        )
    else:
        citation_recall = 1.0 if not output.evidence or known_ids else 0.0
    return {
        "citation_validity": valid_citations / len(cited_ids) if cited_ids else 1.0,
        "citation_precision": valid_citations / len(cited_ids) if cited_ids else 1.0,
        "citation_recall": citation_recall,
        "claim_citation_coverage": cited_claims / len(output.claims) if output.claims else 1.0,
        "claim_citation_validity": valid_claims / len(output.claims) if output.claims else 1.0,
    }


def abstention_metrics(
    output: RAGEvaluationOutput, reference: RAGEvaluationReference
) -> dict[str, float]:
    """Return additive confusion-matrix contributions for abstention."""
    true_positive = float(output.abstained and reference.should_abstain)
    false_positive = float(output.abstained and not reference.should_abstain)
    false_negative = float(not output.abstained and reference.should_abstain)
    return {
        "abstention_true_positive": true_positive,
        "abstention_false_positive": false_positive,
        "abstention_false_negative": false_negative,
    }


def abstention_summary_metrics(scores: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Compute aggregate abstention precision/recall from evaluator contributions."""
    materialized = list(scores)
    true_positive = sum(score.get("abstention_true_positive", 0.0) for score in materialized)
    false_positive = sum(score.get("abstention_false_positive", 0.0) for score in materialized)
    false_negative = sum(score.get("abstention_false_negative", 0.0) for score in materialized)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "abstention_precision": true_positive / precision_denominator
        if precision_denominator
        else 1.0,
        "abstention_recall": true_positive / recall_denominator if recall_denominator else 1.0,
    }


def operational_metrics(output: RAGEvaluationOutput) -> dict[str, float]:
    """Return content-free operational signals supplied by the target."""
    input_tokens = float(output.input_tokens)
    output_tokens = float(output.output_tokens)
    return {
        "tool_count": float(len(output.tool_trajectory)),
        "latency_ms": float(sum(output.stage_ms.values())),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": float(output.cost_usd or 0.0),
    }


def evaluate_output(
    output: RAGEvaluationOutput, reference: RAGEvaluationReference
) -> dict[str, float]:
    """Compose every deterministic metric for one target result."""
    return {
        **retrieval_metrics(output.candidates, reference),
        **citation_metrics(output, reference),
        **abstention_metrics(output, reference),
        **operational_metrics(output),
    }


def deterministic_evaluators() -> list[Any]:
    """Build LangSmith-compatible evaluators without importing optional evaluators."""

    def evaluate(run: Any, example: Any, **_: Any) -> dict[str, list[dict[str, float | str]]]:
        run_outputs = getattr(run, "outputs", None)
        example_outputs = getattr(example, "outputs", None)
        output = output_from_mapping(
            run_outputs if run_outputs is not None else run.get("outputs", run)
        )
        reference = reference_from_mapping(
            example_outputs
            if example_outputs is not None
            else example.get("outputs", example.get("reference", {}))
        )
        return {
            "results": [
                {"key": metric, "score": score}
                for metric, score in evaluate_output(output, reference).items()
            ]
        }

    return [evaluate]


def deterministic_summary_evaluators() -> list[Any]:
    """Build LangSmith-compatible aggregate evaluators for non-additive metrics."""

    def summarize(
        runs: Sequence[Any], examples: Sequence[Any]
    ) -> dict[str, list[dict[str, float | str]]]:
        contributions = [
            abstention_metrics(
                output_from_mapping(getattr(run, "outputs", {}) or {}),
                reference_from_mapping(getattr(example, "outputs", {}) or {}),
            )
            for run, example in zip(runs, examples, strict=True)
        ]
        return {
            "results": [
                {"key": metric, "score": score}
                for metric, score in abstention_summary_metrics(contributions).items()
            ]
        }

    return [summarize]


def reference_from_mapping(value: Mapping[str, Any]) -> RAGEvaluationReference:
    spans = tuple(RelevantSpan(**span) for span in value.get("relevant_spans", ()))
    return RAGEvaluationReference(
        answer=value.get("answer"),
        relevant_document_ids=frozenset(value.get("relevant_document_ids", ())),
        relevant_chunk_ids=frozenset(value.get("relevant_chunk_ids", ())),
        relevant_spans=spans,
        should_abstain=bool(value.get("should_abstain", False)),
    )


def output_from_mapping(value: Mapping[str, Any]) -> RAGEvaluationOutput:
    from .contracts import ClaimTrace, EvidenceTrace

    return RAGEvaluationOutput(
        answer=str(value.get("answer", "")),
        abstained=bool(value.get("abstained", False)),
        candidates=tuple(RetrievalTrace(**item) for item in value.get("candidates", ())),
        evidence=tuple(EvidenceTrace(**item) for item in value.get("evidence", ())),
        claims=tuple(
            ClaimTrace(text=item["text"], evidence_ids=tuple(item.get("evidence_ids", ())))
            for item in value.get("claims", ())
        ),
        citations_valid=bool(value.get("citations_valid", False)),
        tool_trajectory=tuple(value.get("tool_trajectory", ())),
        stage_ms=dict(value.get("stage_ms", {})),
        input_tokens=int(value.get("input_tokens", 0)),
        output_tokens=int(value.get("output_tokens", 0)),
        cost_usd=value.get("cost_usd"),
    )
