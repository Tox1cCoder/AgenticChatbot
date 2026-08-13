"""Consumer-visible deterministic RAG evaluation metric contracts."""

from __future__ import annotations

from app.evaluation.rag.contracts import (
    ClaimTrace,
    EvidenceTrace,
    RAGEvaluationOutput,
    RAGEvaluationReference,
    RetrievalTrace,
)
from app.evaluation.rag.metrics import (
    abstention_metrics,
    citation_metrics,
    deterministic_evaluators,
    operational_metrics,
    retrieval_metrics,
)


def test_recall_at_k_uses_stable_document_ids():
    actual = [
        RetrievalTrace(document_id="doc-b", chunk_id="chunk-b", rank=1, score=0.8),
        RetrievalTrace(document_id="doc-a", chunk_id="chunk-a", rank=2, score=0.7),
    ]
    reference = RAGEvaluationReference(relevant_document_ids={"doc-a"})

    scores = retrieval_metrics(actual, reference, ks=(1, 2))

    assert scores["document_recall_at_1"] == 0.0
    assert scores["document_recall_at_2"] == 1.0
    assert scores["document_hit_rate_at_1"] == 0.0
    assert scores["document_reciprocal_rank"] == 0.5


def test_citation_metrics_reject_unknown_evidence_ids():
    output = RAGEvaluationOutput(
        answer="Revenue increased [E1] and margin improved [E9].",
        abstained=False,
        candidates=(),
        evidence=(EvidenceTrace("E1", "doc-a", "chunk-a", 1, 1),),
        claims=(
            ClaimTrace("Revenue increased.", ("E1",)),
            ClaimTrace("Margin improved.", ("E9",)),
        ),
        citations_valid=False,
        tool_trajectory=("search_chunks",),
        stage_ms={},
        input_tokens=100,
        output_tokens=20,
        cost_usd=None,
    )

    scores = citation_metrics(output)

    assert scores["citation_validity"] == 0.5
    assert scores["citation_precision"] == 0.5
    assert scores["claim_citation_coverage"] == 1.0


def test_citation_precision_and_recall_require_gold_support_not_only_valid_ids():
    output = RAGEvaluationOutput(
        answer="Supported [E1], irrelevant [E2].",
        abstained=False,
        candidates=(),
        evidence=(
            EvidenceTrace("E1", "doc-a", "chunk-a", 1, 1),
            EvidenceTrace("E2", "doc-b", "chunk-b", 1, 1),
        ),
        claims=(ClaimTrace("Supported", ("E1",)), ClaimTrace("Irrelevant", ("E2",))),
        citations_valid=True,
        tool_trajectory=(),
        stage_ms={},
        input_tokens=0,
        output_tokens=0,
        cost_usd=None,
    )
    reference = RAGEvaluationReference(relevant_document_ids={"doc-a", "doc-c"})

    scores = citation_metrics(output, reference)

    assert scores["citation_validity"] == 1.0
    assert scores["citation_precision"] == 0.5
    assert scores["citation_recall"] == 0.5


def test_operational_and_abstention_metrics_are_reported_without_network_calls():
    output = RAGEvaluationOutput(
        answer="I do not have enough evidence.",
        abstained=True,
        candidates=(),
        evidence=(),
        claims=(),
        citations_valid=True,
        tool_trajectory=("search_chunks", "read_chunk"),
        stage_ms={"retrieval": 13.5, "generation": 40.0},
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.002,
    )

    assert abstention_metrics(output, RAGEvaluationReference(should_abstain=True)) == {
        "abstention_true_positive": 1.0,
        "abstention_false_positive": 0.0,
        "abstention_false_negative": 0.0,
    }
    assert operational_metrics(output) == {
        "tool_count": 2.0,
        "latency_ms": 53.5,
        "input_tokens": 10.0,
        "output_tokens": 5.0,
        "total_tokens": 15.0,
        "cost_usd": 0.002,
    }


def test_langsmith_style_run_and_example_objects_are_evaluated_at_the_boundary():
    class Run:
        outputs = {
            "answer": "The warranty period is two years.",
            "abstained": False,
            "candidates": [
                {"document_id": "doc-a", "chunk_id": "chunk-a", "rank": 1, "score": 0.9}
            ],
            "evidence": [],
            "claims": [],
            "citations_valid": True,
            "tool_trajectory": [],
            "stage_ms": {},
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_usd": None,
        }

    class Example:
        outputs = {"relevant_document_ids": ["doc-a"]}

    result = deterministic_evaluators()[0](Run(), Example())

    assert {item["key"]: item["score"] for item in result["results"]}["document_recall_at_1"] == 1.0


def test_abstention_summary_distinguishes_false_positives_and_false_negatives():
    from app.evaluation.rag.metrics import abstention_summary_metrics

    scores = abstention_summary_metrics(
        [
            {
                "abstention_true_positive": 1.0,
                "abstention_false_positive": 0.0,
                "abstention_false_negative": 0.0,
            },
            {
                "abstention_true_positive": 0.0,
                "abstention_false_positive": 1.0,
                "abstention_false_negative": 0.0,
            },
            {
                "abstention_true_positive": 0.0,
                "abstention_false_positive": 0.0,
                "abstention_false_negative": 1.0,
            },
        ]
    )

    assert scores == {"abstention_precision": 0.5, "abstention_recall": 0.5}
