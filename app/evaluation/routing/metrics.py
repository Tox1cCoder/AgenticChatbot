"""Deterministic scoring for one routing evaluation run.

Two accuracies live here and they answer different questions. Keeping them
distinct is the point:

* **canonical** — every case counts once under ``primary_agent_id``. This
  feeds precision/recall/F1 and the confusion matrix. An acceptable
  alternative is never a second true label, because scoring a hit under both
  would let a dataset raise its own macro-F1 just by widening acceptable sets.
* **acceptable-set** — a prediction is correct when it lands anywhere in the
  case's acceptable set. This feeds per-language accuracy, where the question
  is whether a language is served sensibly at all.

Macro-F1 averages over *labels*, not cases, so an agent the router picks ten
times a day cannot drown out one it picks rarely and always gets wrong.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from app.evaluation.routing.contracts import (
    RoutingEvalCase,
    RoutingEvaluationReport,
    RoutingPrediction,
)

__all__ = [
    "CHAT_AGENT_ID",
    "acceptable_accuracy_by_language",
    "build_report",
    "canonical_accuracy",
    "canonical_confusion_matrix",
    "count_silent_chat_substitutions",
    "macro_f1",
    "per_label_f1",
    "structured_success_rates",
]

CHAT_AGENT_ID = "chat_agent"

#: The label recorded when the router produced no decision at all. It is a
#: real cell in the confusion matrix rather than a dropped row, so a run that
#: fails half its cases cannot score well on the half that answered.
NO_PREDICTION = "<none>"


def _prediction_index(
    predictions: Iterable[RoutingPrediction],
) -> dict[str, RoutingPrediction]:
    return {prediction.case_id: prediction for prediction in predictions}


def _predicted_label(prediction: RoutingPrediction | None) -> str:
    if prediction is None or not prediction.predicted_agent_id:
        return NO_PREDICTION
    return prediction.predicted_agent_id


def canonical_confusion_matrix(
    cases: Sequence[RoutingEvalCase], predictions: Iterable[RoutingPrediction]
) -> dict[tuple[str, str], int]:
    """``(true primary, predicted)`` counts, one entry per case."""
    index = _prediction_index(predictions)
    matrix: Counter[tuple[str, str]] = Counter()
    for case in cases:
        matrix[(case.primary_agent_id, _predicted_label(index.get(case.case_id)))] += 1
    return dict(matrix)


def per_label_f1(
    cases: Sequence[RoutingEvalCase], predictions: Iterable[RoutingPrediction]
) -> dict[str, float]:
    """F1 for every label that appears as a *true* label in the dataset.

    Predicted-only labels (including ``<none>``) are not given their own F1 —
    they have no support — but they still count against precision for the
    label they were predicted instead of.
    """
    index = _prediction_index(predictions)
    true_positive: Counter[str] = Counter()
    predicted: Counter[str] = Counter()
    actual: Counter[str] = Counter()

    for case in cases:
        prediction = _predicted_label(index.get(case.case_id))
        actual[case.primary_agent_id] += 1
        predicted[prediction] += 1
        if prediction == case.primary_agent_id:
            true_positive[case.primary_agent_id] += 1

    scores: dict[str, float] = {}
    for label, support in actual.items():
        hits = true_positive[label]
        precision = hits / predicted[label] if predicted[label] else 0.0
        recall = hits / support if support else 0.0
        scores[label] = (
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return scores


def macro_f1(cases: Sequence[RoutingEvalCase], predictions: Iterable[RoutingPrediction]) -> float:
    """Unweighted mean F1 over every true label present in the dataset."""
    scores = per_label_f1(cases, predictions)
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def canonical_accuracy(
    cases: Sequence[RoutingEvalCase], predictions: Iterable[RoutingPrediction]
) -> float:
    """Share of cases routed to their exact primary agent."""
    if not cases:
        return 0.0
    index = _prediction_index(predictions)
    hits = sum(
        1 for case in cases if _predicted_label(index.get(case.case_id)) == case.primary_agent_id
    )
    return hits / len(cases)


def acceptable_accuracy_by_language(
    cases: Sequence[RoutingEvalCase], predictions: Iterable[RoutingPrediction]
) -> dict[str, float]:
    """Per-language share of cases routed anywhere in their acceptable set.

    A language with no cases is absent from the result rather than reported as
    zero: missing coverage is a dataset composition failure, and reporting it
    as a quality failure would blame the router for a gap nobody measured.
    """
    index = _prediction_index(predictions)
    totals: Counter[str] = Counter()
    hits: Counter[str] = Counter()

    for case in cases:
        totals[case.language] += 1
        if _predicted_label(index.get(case.case_id)) in case.acceptable_agent_ids:
            hits[case.language] += 1

    return {language: hits[language] / total for language, total in totals.items()}


def count_silent_chat_substitutions(
    cases: Sequence[RoutingEvalCase], predictions: Iterable[RoutingPrediction]
) -> int:
    """Cases answered by chat when chat was not an acceptable answer.

    This is the failure the routing rewrite exists to prevent: the pre-v2
    router defaulted to chat on any difficulty, which reads to a user as a
    confident wrong answer rather than as a failure. A routing *error* is not
    counted here — only a substitution that produced a plausible-looking chat
    reply instead of the specialist the turn needed.
    """
    index = _prediction_index(predictions)
    return sum(
        1
        for case in cases
        if _predicted_label(index.get(case.case_id)) == CHAT_AGENT_ID
        and CHAT_AGENT_ID not in case.acceptable_agent_ids
    )


def structured_success_rates(
    predictions: Sequence[RoutingPrediction],
) -> tuple[float, float]:
    """``(first attempt, after retry)`` structured-output success shares.

    First-attempt success requires ``attempts == 1``; the retry rate counts
    every case that ended up structured, however many attempts it took. The
    gate holds them to different bars because a router that needs a second
    call routinely is one provider hiccup away from failing.
    """
    total = len(predictions)
    if not total:
        return 0.0, 0.0
    first = sum(1 for p in predictions if p.structured_success and p.attempts == 1)
    eventual = sum(1 for p in predictions if p.structured_success)
    return first / total, eventual / total


def build_report(
    *,
    cases: Sequence[RoutingEvalCase],
    predictions: Sequence[RoutingPrediction],
    dataset_sha256: str,
    provider: str,
    model: str,
    inventory_version: str,
    generated_at: datetime,
    finalizer_bypasses: int = 0,
    unknown_published_evidence_ids: int = 0,
) -> RoutingEvaluationReport:
    """Assemble one report, refusing predictions that do not match the dataset.

    A report whose predictions name cases the dataset does not contain, or
    names one case twice, is unscoreable — and would silently produce a
    plausible number rather than an error.
    """
    known = {case.case_id for case in cases}
    seen: set[str] = set()
    for prediction in predictions:
        if prediction.case_id not in known:
            raise ValueError(f"prediction for unknown case_id {prediction.case_id!r}")
        if prediction.case_id in seen:
            raise ValueError(f"duplicate case_id in predictions: {prediction.case_id!r}")
        seen.add(prediction.case_id)

    first_attempt, after_retry = structured_success_rates(predictions)

    return RoutingEvaluationReport(
        dataset_sha256=dataset_sha256,
        provider=provider,
        model=model,
        inventory_version=inventory_version,
        generated_at=generated_at,
        case_count=len(cases),
        macro_f1=macro_f1(cases, predictions),
        accuracy_by_language=acceptable_accuracy_by_language(cases, predictions),
        first_attempt_structured_success=first_attempt,
        after_retry_structured_success=after_retry,
        silent_chat_substitutions=count_silent_chat_substitutions(cases, predictions),
        finalizer_bypasses=int(finalizer_bypasses),
        unknown_published_evidence_ids=int(unknown_published_evidence_ids),
        predictions=tuple(predictions),
    )


def composition(cases: Sequence[RoutingEvalCase]) -> Mapping[str, Counter[str]]:
    """Language and category counts, for the release gate to check."""
    return {
        "language": Counter(case.language for case in cases),
        "category": Counter(case.category for case in cases),
    }
