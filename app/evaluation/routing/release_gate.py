"""The fail-closed release decision for measured routing quality.

Every check answers "why might this report be reassuring but wrong?" — because
the failure mode of a release gate is not a false alarm, it is a green light
derived from a report nobody should have trusted. A stale run, a run against a
different model, an approval inherited from an edited dataset, a language
nobody wrote cases for: each produces numbers that look exactly like success.

The decision reports *every* failure rather than the first, so one run tells
you everything blocking release instead of one thing at a time.

The reviewer requirement is deliberately not automatable. ``approved`` is set
by a human who did not generate the dataset. Until that happens the gate
refuses, and that refusal is the correct state — not an obstacle to route
around by defaulting it to true.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from app.evaluation.routing.contracts import (
    REQUIRED_CATEGORIES,
    REQUIRED_LANGUAGES,
    RoutingDatasetReview,
    RoutingEvalCase,
    RoutingEvaluationReport,
    RoutingReleaseDecision,
)

__all__ = [
    "MAX_REPORT_AGE",
    "MIN_AFTER_RETRY_STRUCTURED_SUCCESS",
    "MIN_CASES_PER_CATEGORY",
    "MIN_CASES_PER_LANGUAGE",
    "MIN_FIRST_ATTEMPT_STRUCTURED_SUCCESS",
    "MIN_LANGUAGE_ACCURACY",
    "MIN_MACRO_F1",
    "MIN_TOTAL_CASES",
    "MAX_ENGLISH_ACCURACY_GAP",
    "check_routing_release",
]

#: Dataset composition. Below these a metric is an anecdote: a category with
#: three cases moves macro-F1 by a third of a label on one lucky answer.
MIN_TOTAL_CASES = 210
MIN_CASES_PER_LANGUAGE = 30
MIN_CASES_PER_CATEGORY = 10

#: Quality thresholds. `>=` throughout: a stated bar must be reachable.
MIN_MACRO_F1 = 0.90
MIN_LANGUAGE_ACCURACY = 0.85
MAX_ENGLISH_ACCURACY_GAP = 0.05
MIN_FIRST_ATTEMPT_STRUCTURED_SUCCESS = 0.99
MIN_AFTER_RETRY_STRUCTURED_SUCCESS = 0.999

#: A report older than this describes a system that has since been deployed
#: over. Routing sits behind a prompt, an inventory, and a provider, and all
#: three move faster than a week.
MAX_REPORT_AGE = timedelta(days=7)

#: Counters, not rates. One occurrence blocks release.
_VIOLATION_COUNTERS = (
    "silent_chat_substitutions",
    "finalizer_bypasses",
    "unknown_published_evidence_ids",
)


def check_routing_release(
    *,
    report: RoutingEvaluationReport,
    review: RoutingDatasetReview | None,
    expected_provider: str,
    expected_model: str,
    expected_inventory_version: str,
    cases: Sequence[RoutingEvalCase] = (),
    now: datetime | None = None,
) -> RoutingReleaseDecision:
    """Decide whether this report may release, and say why if not.

    ``cases`` defaults to empty rather than optional-and-skipped. Omitting the
    dataset therefore *fails* composition instead of silently passing it — a
    caller who forgets the argument gets a refusal, not an unchecked release.
    """
    moment = now or datetime.now(timezone.utc)
    reasons: set[str] = set()

    reasons |= _review_reasons(report, review)
    reasons |= _tuple_reasons(
        report, expected_provider, expected_model, expected_inventory_version
    )
    reasons |= _freshness_reasons(report, moment)
    reasons |= _composition_reasons(report, cases)
    reasons |= _threshold_reasons(report)

    return RoutingReleaseDecision(passed=not reasons, reason_codes=tuple(sorted(reasons)))


def _review_reasons(
    report: RoutingEvaluationReport, review: RoutingDatasetReview | None
) -> set[str]:
    if review is None:
        return {"missing_review"}
    reasons: set[str] = set()
    if not review.approved:
        reasons.add("review_not_approved")
    if review.dataset_sha256 != report.dataset_sha256:
        # The reviewed labels are not these labels.
        reasons.add("dataset_hash_mismatch")
    return reasons


def _tuple_reasons(
    report: RoutingEvaluationReport,
    expected_provider: str,
    expected_model: str,
    expected_inventory_version: str,
) -> set[str]:
    reasons: set[str] = set()
    if report.provider != expected_provider:
        reasons.add("provider_mismatch")
    if report.model != expected_model:
        reasons.add("model_mismatch")
    if report.inventory_version != expected_inventory_version:
        reasons.add("inventory_version_mismatch")
    return reasons


def _freshness_reasons(report: RoutingEvaluationReport, now: datetime) -> set[str]:
    if report.generated_at > now:
        # Clock skew or a hand-edited stamp must not buy freshness.
        return {"report_not_yet_generated"}
    if now - report.generated_at > MAX_REPORT_AGE:
        return {"report_stale"}
    return set()


def _composition_reasons(
    report: RoutingEvaluationReport, cases: Sequence[RoutingEvalCase]
) -> set[str]:
    reasons: set[str] = set()

    if len(cases) < MIN_TOTAL_CASES:
        reasons.add("case_count_below_minimum")
    if report.case_count != len(cases):
        # The report was scored against a different dataset than the one being
        # checked for composition, so neither answer describes the other.
        reasons.add("case_count_mismatch")

    by_language = Counter(case.language for case in cases)
    for language in REQUIRED_LANGUAGES:
        if by_language[language] < MIN_CASES_PER_LANGUAGE:
            reasons.add(f"language_coverage_below_minimum:{language}")

    by_category = Counter(case.category for case in cases)
    for category in REQUIRED_CATEGORIES:
        # Iterating the required set rather than what the dataset happens to
        # contain: a category with zero cases must fail coverage, not vanish.
        if by_category[category] < MIN_CASES_PER_CATEGORY:
            reasons.add(f"category_coverage_below_minimum:{category}")

    return reasons


def _threshold_reasons(report: RoutingEvaluationReport) -> set[str]:
    reasons: set[str] = set()

    if report.macro_f1 < MIN_MACRO_F1:
        reasons.add("macro_f1_below_threshold")

    accuracy = report.accuracy_by_language
    for language in REQUIRED_LANGUAGES:
        if language not in accuracy:
            reasons.add(f"language_accuracy_missing:{language}")
        elif accuracy[language] < MIN_LANGUAGE_ACCURACY:
            reasons.add(f"language_accuracy_below_threshold:{language}")

    english = accuracy.get("en")
    if english is not None:
        for language in REQUIRED_LANGUAGES:
            if language == "en" or language not in accuracy:
                continue
            # Only a language *trailing* English is a gap. One that does better
            # is not a defect to report.
            if english - accuracy[language] > MAX_ENGLISH_ACCURACY_GAP:
                reasons.add(f"language_accuracy_gap_to_english:{language}")

    if report.first_attempt_structured_success < MIN_FIRST_ATTEMPT_STRUCTURED_SUCCESS:
        reasons.add("first_attempt_structured_success_below_threshold")
    if report.after_retry_structured_success < MIN_AFTER_RETRY_STRUCTURED_SUCCESS:
        reasons.add("after_retry_structured_success_below_threshold")

    for counter in _VIOLATION_COUNTERS:
        if getattr(report, counter) > 0:
            reasons.add(counter)

    return reasons
