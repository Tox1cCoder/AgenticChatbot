"""The release gate that decides whether measured routing may ship.

Every check here fails closed. That is the whole design: a gate whose default
answer is "yes" measures nothing, and each of these codes exists because the
corresponding mistake produces a report that *looks* fine — a stale run, a
report from a different model, an unreviewed dataset, a language nobody wrote
cases for.

The reviewer requirement is deliberately not automatable. `approved` is set by
a human who did not generate the dataset; until then the gate refuses, and
that refusal is the correct state rather than a bug to work around.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.routing.contracts import (
    REQUIRED_CATEGORIES,
    REQUIRED_LANGUAGES,
    RoutingDatasetReview,
    RoutingEvalCase,
    RoutingEvaluationReport,
    dataset_sha256,
    load_dataset,
    load_review,
)
from app.evaluation.routing.harness import build_evaluation_inventory
from app.evaluation.routing.release_gate import (
    MAX_REPORT_AGE,
    MIN_CASES_PER_CATEGORY,
    MIN_CASES_PER_LANGUAGE,
    MIN_TOTAL_CASES,
    check_routing_release,
)

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
DATASET_SHA = "c" * 64
CATEGORIES = REQUIRED_CATEGORIES
PRIMARY_BY_CATEGORY = {
    "general_chat": "chat_agent",
    "current_information": "search_agent",
    "document_qa": "rag_agent",
    "planning": "planning_agent",
    "canvas": "canvas_agent",
    "image_generation": "image_generator_agent",
    "custom_agent": "custom_agent:analyst",
    "ambiguous_followup": "chat_agent",
}


def _passing_dataset() -> list[RoutingEvalCase]:
    """A minimal dataset that satisfies every composition rule exactly."""
    cases: list[RoutingEvalCase] = []
    for language in REQUIRED_LANGUAGES:
        for index in range(MIN_CASES_PER_LANGUAGE):
            category = CATEGORIES[index % len(CATEGORIES)]
            primary = PRIMARY_BY_CATEGORY[category]
            cases.append(
                RoutingEvalCase(
                    case_id=f"{language}-{index:03d}",
                    language=language,
                    category=category,
                    message=f"{language} sample {index}",
                    primary_agent_id=primary,
                    acceptable_agent_ids=(primary,),
                )
            )
    return cases


def _report(**overrides) -> RoutingEvaluationReport:
    payload = {
        "dataset_sha256": DATASET_SHA,
        "provider": "gemini",
        "model": "gemini-3-pro",
        "inventory_version": "inventory-v1",
        "generated_at": NOW - timedelta(hours=1),
        "case_count": MIN_TOTAL_CASES,
        "macro_f1": 0.95,
        "accuracy_by_language": dict.fromkeys(REQUIRED_LANGUAGES, 0.95),
        "first_attempt_structured_success": 1.0,
        "after_retry_structured_success": 1.0,
        "silent_chat_substitutions": 0,
        "finalizer_bypasses": 0,
        "unknown_published_evidence_ids": 0,
        "predictions": (),
    }
    payload.update(overrides)
    return RoutingEvaluationReport(**payload)


def _review(**overrides) -> RoutingDatasetReview:
    payload = {
        "dataset_sha256": DATASET_SHA,
        "approved": True,
        "reviewing_team": "routing-quality",
        "reviewed_at": NOW - timedelta(days=1),
        "label_guideline_version": "v1",
    }
    payload.update(overrides)
    return RoutingDatasetReview(**payload)


def _check(**overrides):
    kwargs = {
        "report": _report(),
        "review": _review(),
        "cases": _passing_dataset(),
        "expected_provider": "gemini",
        "expected_model": "gemini-3-pro",
        "expected_inventory_version": "inventory-v1",
        "now": NOW,
    }
    kwargs.update(overrides)
    return check_routing_release(**kwargs)


# ----------------------------------------------------------------------
# the passing case, so every failure below means something
# ----------------------------------------------------------------------


def test_a_complete_approved_and_fresh_run_passes():
    decision = _check()
    assert decision.passed is True, decision.reason_codes
    assert decision.reason_codes == ()


def test_the_dataset_fixture_actually_meets_every_composition_rule():
    cases = _passing_dataset()
    assert len(cases) >= MIN_TOTAL_CASES
    for language in REQUIRED_LANGUAGES:
        assert sum(1 for case in cases if case.language == language) >= MIN_CASES_PER_LANGUAGE
    for category in CATEGORIES:
        assert sum(1 for case in cases if case.category == category) >= MIN_CASES_PER_CATEGORY


# ----------------------------------------------------------------------
# review
# ----------------------------------------------------------------------


def test_release_refuses_a_missing_review():
    decision = _check(review=None)
    assert decision.passed is False
    assert "missing_review" in decision.reason_codes


def test_release_refuses_an_unapproved_review():
    """The state a freshly generated dataset ships in."""
    decision = _check(review=_review(approved=False))
    assert decision.passed is False
    assert "review_not_approved" in decision.reason_codes


def test_release_refuses_a_review_of_a_different_dataset():
    """A hash mismatch means the reviewed labels are not these labels."""
    decision = _check(review=_review(dataset_sha256="d" * 64))
    assert decision.passed is False
    assert "dataset_hash_mismatch" in decision.reason_codes


# ----------------------------------------------------------------------
# the exact tuple the report was produced against
# ----------------------------------------------------------------------


def test_release_rejects_stale_or_mismatched_tuple():
    decision = _check(report=_report(provider="different-provider"))
    assert decision.passed is False
    assert "provider_mismatch" in decision.reason_codes


def test_release_rejects_a_report_from_another_model():
    decision = _check(report=_report(model="gemini-2-flash"))
    assert decision.passed is False
    assert "model_mismatch" in decision.reason_codes


def test_release_rejects_a_report_from_another_inventory():
    """A different inventory is a different routing problem."""
    decision = _check(report=_report(inventory_version="inventory-v2"))
    assert decision.passed is False
    assert "inventory_version_mismatch" in decision.reason_codes


def test_release_rejects_a_stale_report():
    decision = _check(report=_report(generated_at=NOW - MAX_REPORT_AGE - timedelta(minutes=1)))
    assert decision.passed is False
    assert "report_stale" in decision.reason_codes


def test_release_rejects_a_report_from_the_future():
    """A clock-skewed or hand-edited timestamp must not buy freshness."""
    decision = _check(report=_report(generated_at=NOW + timedelta(hours=1)))
    assert decision.passed is False
    assert "report_not_yet_generated" in decision.reason_codes


# ----------------------------------------------------------------------
# dataset composition
# ----------------------------------------------------------------------


def test_release_rejects_fewer_than_the_minimum_cases():
    short = _passing_dataset()[: MIN_TOTAL_CASES - 1]
    decision = _check(cases=short, report=_report(case_count=len(short)))
    assert decision.passed is False
    assert "case_count_below_minimum" in decision.reason_codes


def test_release_rejects_a_thin_language():
    cases = [case for case in _passing_dataset() if case.language != "ar"]
    cases.extend(
        RoutingEvalCase(
            case_id=f"ar-{index:03d}",
            language="ar",
            category="general_chat",
            message=f"عينة {index}",
            primary_agent_id="chat_agent",
            acceptable_agent_ids=("chat_agent",),
        )
        for index in range(MIN_CASES_PER_LANGUAGE - 1)
    )
    decision = _check(cases=cases, report=_report(case_count=len(cases)))

    assert decision.passed is False
    assert "language_coverage_below_minimum:ar" in decision.reason_codes


def test_release_rejects_a_missing_language_outright():
    cases = [case for case in _passing_dataset() if case.language != "vi"]
    decision = _check(cases=cases, report=_report(case_count=len(cases)))

    assert decision.passed is False
    assert "language_coverage_below_minimum:vi" in decision.reason_codes


def test_release_rejects_a_category_the_dataset_omits_entirely():
    """A zero-case category must fail coverage rather than vanish from it.

    Counting only the categories a dataset happens to contain would let the
    dataset decide which rules apply to it.
    """
    cases = [case for case in _passing_dataset() if case.category != "canvas"]
    decision = _check(cases=cases, report=_report(case_count=len(cases)))

    assert decision.passed is False
    assert "category_coverage_below_minimum:canvas" in decision.reason_codes


def test_release_rejects_a_thin_category():
    """Ten cases per category, or the category is anecdote rather than signal."""
    cases = [case for case in _passing_dataset() if case.category != "canvas"]
    cases.extend(
        RoutingEvalCase(
            case_id=f"en-canvas-{index:03d}",
            language="en",
            category="canvas",
            message=f"open the canvas and draft section {index}",
            primary_agent_id="canvas_agent",
            acceptable_agent_ids=("canvas_agent",),
        )
        for index in range(MIN_CASES_PER_CATEGORY - 1)
    )
    decision = _check(cases=cases, report=_report(case_count=len(cases)))

    assert decision.passed is False
    assert "category_coverage_below_minimum:canvas" in decision.reason_codes


def test_release_rejects_a_report_whose_case_count_disagrees_with_the_dataset():
    """Otherwise a report could be gated against a dataset it never ran on."""
    decision = _check(report=_report(case_count=MIN_TOTAL_CASES + 5))
    assert decision.passed is False
    assert "case_count_mismatch" in decision.reason_codes


def test_release_refuses_an_empty_dataset_when_cases_are_not_supplied():
    """Omitting the dataset must not silently skip composition checks."""
    decision = check_routing_release(
        report=_report(),
        review=_review(),
        expected_provider="gemini",
        expected_model="gemini-3-pro",
        expected_inventory_version="inventory-v1",
        now=NOW,
    )
    assert decision.passed is False
    assert "case_count_below_minimum" in decision.reason_codes


# ----------------------------------------------------------------------
# thresholds
# ----------------------------------------------------------------------


def test_release_rejects_macro_f1_below_the_threshold():
    decision = _check(report=_report(macro_f1=0.899))
    assert decision.passed is False
    assert "macro_f1_below_threshold" in decision.reason_codes


def test_macro_f1_exactly_at_the_threshold_passes():
    """`>=`, not `>` — a stated bar must be reachable."""
    decision = _check(report=_report(macro_f1=0.90))
    assert decision.passed is True, decision.reason_codes


def test_release_rejects_a_language_below_its_own_floor():
    accuracy = dict.fromkeys(REQUIRED_LANGUAGES, 0.95)
    accuracy["th"] = 0.84
    decision = _check(report=_report(accuracy_by_language=accuracy))

    assert decision.passed is False
    assert "language_accuracy_below_threshold:th" in decision.reason_codes


def test_release_rejects_a_language_that_trails_english_too_far():
    """A uniformly high bar still hides a language that is measurably worse."""
    accuracy = dict.fromkeys(REQUIRED_LANGUAGES, 0.99)
    accuracy["ar"] = 0.93
    decision = _check(report=_report(accuracy_by_language=accuracy))

    assert decision.passed is False
    assert "language_accuracy_gap_to_english:ar" in decision.reason_codes
    assert "language_accuracy_below_threshold:ar" not in decision.reason_codes


def test_a_language_above_english_is_not_a_gap_failure():
    accuracy = dict.fromkeys(REQUIRED_LANGUAGES, 0.90)
    accuracy["en"] = 0.90
    accuracy["ja"] = 0.99
    decision = _check(report=_report(accuracy_by_language=accuracy))

    assert decision.passed is True, decision.reason_codes


def test_release_rejects_a_report_missing_a_required_language_accuracy():
    accuracy = dict.fromkeys(REQUIRED_LANGUAGES, 0.95)
    accuracy.pop("mixed")
    decision = _check(report=_report(accuracy_by_language=accuracy))

    assert decision.passed is False
    assert "language_accuracy_missing:mixed" in decision.reason_codes


def test_release_rejects_missing_english_because_every_gap_is_measured_from_it():
    accuracy = dict.fromkeys(REQUIRED_LANGUAGES, 0.95)
    accuracy.pop("en")
    decision = _check(report=_report(accuracy_by_language=accuracy))

    assert decision.passed is False
    assert "language_accuracy_missing:en" in decision.reason_codes


def test_release_rejects_first_attempt_structured_success_below_threshold():
    decision = _check(report=_report(first_attempt_structured_success=0.98))
    assert decision.passed is False
    assert "first_attempt_structured_success_below_threshold" in decision.reason_codes


def test_release_rejects_after_retry_structured_success_below_threshold():
    decision = _check(report=_report(after_retry_structured_success=0.998))
    assert decision.passed is False
    assert "after_retry_structured_success_below_threshold" in decision.reason_codes


@pytest.mark.parametrize(
    "field",
    ["silent_chat_substitutions", "finalizer_bypasses", "unknown_published_evidence_ids"],
)
def test_release_rejects_any_nonzero_violation_counter(field):
    """These are not rates. One occurrence is a release blocker."""
    decision = _check(report=_report(**{field: 1}))
    assert decision.passed is False
    assert field in decision.reason_codes


# ----------------------------------------------------------------------
# the decision itself
# ----------------------------------------------------------------------


def test_every_failure_is_reported_rather_than_the_first_one():
    """One run should surface everything wrong, not one thing at a time."""
    decision = _check(
        review=_review(approved=False),
        report=_report(provider="other", macro_f1=0.1, silent_chat_substitutions=3),
    )

    assert decision.passed is False
    for code in (
        "review_not_approved",
        "provider_mismatch",
        "macro_f1_below_threshold",
        "silent_chat_substitutions",
    ):
        assert code in decision.reason_codes


def test_reason_codes_are_unique_and_ordered_deterministically():
    decision = _check(report=_report(macro_f1=0.1, first_attempt_structured_success=0.0))
    assert len(set(decision.reason_codes)) == len(decision.reason_codes)
    assert list(decision.reason_codes) == sorted(decision.reason_codes)


def test_a_passing_decision_is_frozen():
    """A decision that can be edited after the fact is not a decision."""
    decision = _check()
    with pytest.raises(ValidationError):
        decision.passed = False


# ----------------------------------------------------------------------
# the dataset this repository actually ships
# ----------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parent.parent / "eval" / "routing"
GOLDEN = DATASET_DIR / "golden_v1.jsonl"
GOLDEN_REVIEW = DATASET_DIR / "golden_v1.review.json"


def test_the_shipped_dataset_parses_and_meets_composition():
    """The gate is only as good as the artifact it reads.

    Checked here as well as at release time so an edit that drops a language
    is caught by the suite rather than by a blocked deploy.
    """
    cases = load_dataset(GOLDEN)

    assert len(cases) >= MIN_TOTAL_CASES
    for language in REQUIRED_LANGUAGES:
        count = sum(1 for case in cases if case.language == language)
        assert count >= MIN_CASES_PER_LANGUAGE, f"{language} has {count}"
    for category in REQUIRED_CATEGORIES:
        count = sum(1 for case in cases if case.category == category)
        assert count >= MIN_CASES_PER_CATEGORY, f"{category} has {count}"


def test_the_shipped_review_is_bound_to_the_shipped_dataset():
    """An edited dataset must not inherit the previous approval.

    This deliberately does not assert ``approved is False``. A test that fails
    the moment a reviewer approves would only train people to delete it.
    """
    review = load_review(GOLDEN_REVIEW)

    assert review.dataset_sha256 == dataset_sha256(GOLDEN), (
        "the review manifest names a different dataset than the one shipped; "
        "the labels must be re-reviewed after any edit"
    )


def test_most_specialist_cases_do_not_accept_chat_as_an_answer():
    """Otherwise ``silent_chat_substitutions`` could never rise above zero.

    A dataset that quietly lists chat_agent as acceptable everywhere makes the
    counter vacuous while the gate still reports it as clean.
    """
    cases = load_dataset(GOLDEN)
    specialist = [case for case in cases if case.primary_agent_id != "chat_agent"]
    lenient = [case for case in specialist if "chat_agent" in case.acceptable_agent_ids]

    assert len(lenient) / len(specialist) < 0.15, (
        f"{len(lenient)} of {len(specialist)} specialist cases accept chat_agent"
    )


def test_every_dataset_target_exists_in_the_evaluation_inventory():
    """A case routed to an agent the run cannot offer is unscoreable."""
    routable = set(build_evaluation_inventory().routable_ids())
    targets = {agent for case in load_dataset(GOLDEN) for agent in case.acceptable_agent_ids}

    assert targets <= routable, (
        f"dataset targets nothing can route to: {sorted(targets - routable)}"
    )
