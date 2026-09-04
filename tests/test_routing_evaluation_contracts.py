"""Contracts and metrics for the versioned routing evaluation.

The dataset is a *labelled* artifact, so the interesting failures here are not
crashes — they are labels that quietly disagree with themselves. A case whose
primary agent is not in its own acceptable set is unscoreable in both
directions at once, and a report that counts one case twice because it had two
acceptable answers inflates exactly the number the release gate reads.

Metric choice, stated once so the report is not ambiguous:

* ``macro_f1`` is **canonical**. Every case counts once under
  ``primary_agent_id``; acceptable alternatives are never a second true label.
* ``accuracy_by_language`` is **acceptable-set**. A prediction is correct when
  it lands anywhere in the case's acceptable set.

The two answer different questions on purpose. Macro-F1 pins per-agent quality
including the classes the router rarely picks; per-language accuracy asks
whether a language is served acceptably at all.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.evaluation.routing.contracts import (
    REQUIRED_CATEGORIES,
    REQUIRED_LANGUAGES,
    RoutingDatasetReview,
    RoutingEvalCase,
    RoutingEvaluationReport,
    RoutingPrediction,
    dataset_sha256,
    load_dataset,
)
from app.evaluation.routing.metrics import (
    acceptable_accuracy_by_language,
    build_report,
    canonical_confusion_matrix,
    count_silent_chat_substitutions,
    macro_f1,
    structured_success_rates,
)


def _case(**overrides) -> RoutingEvalCase:
    payload = {
        "case_id": "en-001",
        "language": "en",
        "category": "general_chat",
        "message": "how are you today",
        "primary_agent_id": "chat_agent",
        "acceptable_agent_ids": ("chat_agent",),
    }
    payload.update(overrides)
    return RoutingEvalCase(**payload)


def _prediction(**overrides) -> RoutingPrediction:
    payload = {
        "case_id": "en-001",
        "predicted_agent_id": "chat_agent",
        "attempts": 1,
        "structured_success": True,
    }
    payload.update(overrides)
    return RoutingPrediction(**payload)


# ----------------------------------------------------------------------
# the case
# ----------------------------------------------------------------------


def test_dataset_requires_primary_inside_acceptable_set():
    with pytest.raises(ValidationError):
        RoutingEvalCase(
            case_id="thai-001",
            language="th",
            category="planning",
            message="ช่วยวางแผนงาน",
            primary_agent_id="planning_agent",
            acceptable_agent_ids=("chat_agent",),
        )


def test_a_case_is_frozen_and_refuses_unknown_fields():
    case = _case()
    with pytest.raises(ValidationError):
        RoutingEvalCase(**{**case.model_dump(), "notes": "why this label"})
    with pytest.raises(ValidationError):
        case.case_id = "changed"


def test_an_unsupported_language_is_rejected():
    with pytest.raises(ValidationError):
        _case(language="de")


def test_the_acceptable_set_must_not_repeat_an_agent():
    """A duplicate would weight one case more heavily than its siblings."""
    with pytest.raises(ValidationError):
        _case(acceptable_agent_ids=("chat_agent", "chat_agent"))


def test_an_empty_message_is_rejected():
    with pytest.raises(ValidationError):
        _case(message="   ")


def test_required_languages_are_the_seven_the_gate_checks():
    assert REQUIRED_LANGUAGES == ("en", "th", "vi", "zh", "ja", "ar", "mixed")


def test_the_category_vocabulary_is_closed():
    """A free-text category lets one typo split a category into two thin ones."""
    with pytest.raises(ValidationError):
        _case(category="planing")


def test_every_declared_category_is_actually_usable():
    for category in REQUIRED_CATEGORIES:
        assert _case(category=category).category == category


# ----------------------------------------------------------------------
# the dataset file
# ----------------------------------------------------------------------


def test_a_dataset_hash_is_stable_and_content_addressed(tmp_path):
    path = tmp_path / "golden.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    first = dataset_sha256(path)

    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    assert dataset_sha256(path) == first

    path.write_text('{"a": 1}\n{"b": 3}\n', encoding="utf-8")
    assert dataset_sha256(path) != first


def test_loading_refuses_a_duplicate_case_id(tmp_path):
    path = tmp_path / "golden.jsonl"
    line = _case().model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_dataset(path)


def test_loading_reports_the_line_number_of_a_bad_case(tmp_path):
    path = tmp_path / "golden.jsonl"
    good = _case().model_dump_json()
    path.write_text(f"{good}\n" + '{"case_id": "broken"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_dataset(path)


def test_loading_skips_blank_lines_but_keeps_order(tmp_path):
    path = tmp_path / "golden.jsonl"
    first = _case(case_id="en-001").model_dump_json()
    second = _case(case_id="en-002").model_dump_json()
    path.write_text(f"{first}\n\n{second}\n", encoding="utf-8")

    assert [case.case_id for case in load_dataset(path)] == ["en-001", "en-002"]


# ----------------------------------------------------------------------
# the prediction and the report
# ----------------------------------------------------------------------


def test_a_prediction_may_not_claim_more_attempts_than_the_router_allows():
    with pytest.raises(ValidationError):
        _prediction(attempts=3)
    with pytest.raises(ValidationError):
        _prediction(attempts=0)


def test_a_failed_route_records_no_predicted_agent():
    prediction = _prediction(
        predicted_agent_id=None, structured_success=False, error_code="routing_timeout"
    )
    assert prediction.predicted_agent_id is None


def test_a_review_manifest_carries_exactly_five_fields():
    review = RoutingDatasetReview(
        dataset_sha256="a" * 64,
        approved=False,
        reviewing_team="unreviewed",
        reviewed_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        label_guideline_version="v1",
    )
    assert set(review.model_dump()) == {
        "dataset_sha256",
        "approved",
        "reviewing_team",
        "reviewed_at",
        "label_guideline_version",
    }
    with pytest.raises(ValidationError):
        RoutingDatasetReview(**{**review.model_dump(), "reviewer_notes": "looks fine"})


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------


def _pair(case_id, language, primary, predicted, *, acceptable=None, category="general_chat"):
    case = _case(
        case_id=case_id,
        language=language,
        category=category,
        primary_agent_id=primary,
        acceptable_agent_ids=tuple(acceptable or (primary,)),
    )
    prediction = _prediction(case_id=case_id, predicted_agent_id=predicted)
    return case, prediction


def test_macro_f1_counts_every_case_once_under_its_primary_label():
    """An acceptable alternative is not a second true label.

    Scoring a hit under both would let a dataset raise macro-F1 just by
    widening its acceptable sets, which is the opposite of what the gate is for.
    """
    pairs = [
        _pair("c1", "en", "chat_agent", "chat_agent", acceptable=("chat_agent", "search_agent")),
        _pair("c2", "en", "search_agent", "chat_agent", acceptable=("search_agent", "chat_agent")),
    ]
    cases = [case for case, _ in pairs]
    predictions = [prediction for _, prediction in pairs]

    matrix = canonical_confusion_matrix(cases, predictions)

    assert matrix[("chat_agent", "chat_agent")] == 1
    assert matrix[("search_agent", "chat_agent")] == 1
    assert sum(matrix.values()) == len(cases)

    # search_agent: no true positives -> F1 0. chat_agent: recall 1, precision
    # 0.5 -> F1 2/3. Macro averages over both labels, not over cases.
    assert macro_f1(cases, predictions) == pytest.approx((2 / 3 + 0.0) / 2)


def test_macro_f1_is_one_when_every_canonical_label_is_hit():
    pairs = [
        _pair("c1", "en", "chat_agent", "chat_agent"),
        _pair("c2", "en", "search_agent", "search_agent"),
    ]
    cases = [case for case, _ in pairs]
    predictions = [prediction for _, prediction in pairs]
    assert macro_f1(cases, predictions) == pytest.approx(1.0)


def test_macro_f1_averages_over_labels_not_cases():
    """Otherwise a rare agent's failures vanish under a common agent's volume."""
    pairs = [_pair(f"c{i}", "en", "chat_agent", "chat_agent") for i in range(9)]
    pairs.append(_pair("c9", "en", "canvas_agent", "chat_agent"))
    cases = [case for case, _ in pairs]
    predictions = [prediction for _, prediction in pairs]

    assert macro_f1(cases, predictions) < 0.6


def test_acceptable_accuracy_accepts_any_agent_in_the_set():
    pairs = [
        _pair("c1", "th", "rag_agent", "search_agent", acceptable=("rag_agent", "search_agent")),
        _pair("c2", "th", "rag_agent", "chat_agent", acceptable=("rag_agent", "search_agent")),
    ]
    cases = [case for case, _ in pairs]
    predictions = [prediction for _, prediction in pairs]

    assert acceptable_accuracy_by_language(cases, predictions) == {"th": pytest.approx(0.5)}


def test_a_language_with_no_cases_is_absent_rather_than_zero():
    """A missing language is a composition failure, not a quality failure."""
    case, prediction = _pair("c1", "en", "chat_agent", "chat_agent")
    assert set(acceptable_accuracy_by_language([case], [prediction])) == {"en"}


def test_a_case_with_no_prediction_counts_as_wrong():
    """Silence is not neutrality: the router failed to answer for that case."""
    case, _ = _pair("c1", "en", "chat_agent", "chat_agent")
    assert acceptable_accuracy_by_language([case], []) == {"en": pytest.approx(0.0)}


def test_a_silent_chat_substitution_is_counted_only_where_chat_is_unacceptable():
    pairs = [
        _pair("c1", "en", "rag_agent", "chat_agent"),
        _pair("c2", "en", "chat_agent", "chat_agent"),
        _pair(
            "c3", "en", "search_agent", "chat_agent", acceptable=("search_agent", "chat_agent")
        ),
    ]
    cases = [case for case, _ in pairs]
    predictions = [prediction for _, prediction in pairs]

    assert count_silent_chat_substitutions(cases, predictions) == 1


def test_structured_success_rates_separate_first_attempt_from_after_retry():
    predictions = [
        _prediction(case_id="c1", attempts=1, structured_success=True),
        _prediction(case_id="c2", attempts=2, structured_success=True),
        _prediction(case_id="c3", attempts=2, structured_success=False),
    ]
    first, after_retry = structured_success_rates(predictions)

    assert first == pytest.approx(1 / 3)
    assert after_retry == pytest.approx(2 / 3)


def test_structured_success_rates_are_zero_for_no_predictions():
    assert structured_success_rates([]) == (0.0, 0.0)


# ----------------------------------------------------------------------
# the assembled report
# ----------------------------------------------------------------------


def test_build_report_records_the_exact_tuple_it_was_run_against():
    pairs = [_pair("c1", "en", "chat_agent", "chat_agent")]
    cases = [case for case, _ in pairs]
    predictions = [prediction for _, prediction in pairs]

    report = build_report(
        cases=cases,
        predictions=predictions,
        dataset_sha256="b" * 64,
        provider="gemini",
        model="gemini-3-pro",
        inventory_version="inventory-v1",
        generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
    )

    assert isinstance(report, RoutingEvaluationReport)
    assert report.provider == "gemini"
    assert report.model == "gemini-3-pro"
    assert report.inventory_version == "inventory-v1"
    assert report.case_count == 1
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.accuracy_by_language == {"en": pytest.approx(1.0)}
    assert report.predictions == tuple(predictions)


def test_build_report_refuses_a_prediction_for_an_unknown_case():
    """A report whose predictions do not match its dataset is unscoreable."""
    case, _ = _pair("c1", "en", "chat_agent", "chat_agent")
    stray = _prediction(case_id="c-unknown")

    with pytest.raises(ValueError, match="unknown case_id"):
        build_report(
            cases=[case],
            predictions=[stray],
            dataset_sha256="b" * 64,
            provider="gemini",
            model="gemini-3-pro",
            inventory_version="inventory-v1",
            generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        )


def test_build_report_refuses_two_predictions_for_one_case():
    case, prediction = _pair("c1", "en", "chat_agent", "chat_agent")

    with pytest.raises(ValueError, match="duplicate case_id"):
        build_report(
            cases=[case],
            predictions=[prediction, prediction],
            dataset_sha256="b" * 64,
            provider="gemini",
            model="gemini-3-pro",
            inventory_version="inventory-v1",
            generated_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        )


def test_a_report_requires_a_timezone_aware_timestamp():
    """Freshness is compared against `now`; a naive stamp compares wrongly."""
    case, prediction = _pair("c1", "en", "chat_agent", "chat_agent")

    with pytest.raises((ValidationError, ValueError)):
        build_report(
            cases=[case],
            predictions=[prediction],
            dataset_sha256="b" * 64,
            provider="gemini",
            model="gemini-3-pro",
            inventory_version="inventory-v1",
            generated_at=datetime(2026, 9, 4),
        )
