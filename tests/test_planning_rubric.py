import json

from app.ai.planning_rubric import (
    FALLBACK_PLANNING_RUBRIC,
    PlanningRubricAttempt,
    PlanningRubricContract,
    PlanningRubricEvaluation,
    build_planning_rubric_feedback,
    parse_planning_rubric_evaluation,
)


def test_fallback_rubric_is_minimal_and_invariant_only():
    assert "valid for the current planning phase" in FALLBACK_PLANNING_RUBRIC
    assert "not silently destroyed" in FALLBACK_PLANNING_RUBRIC
    assert "20 chars" not in FALLBACK_PLANNING_RUBRIC


def test_planning_rubric_contract_accepts_generated_source():
    contract = PlanningRubricContract(
        rubric="- request_fit: Tasks fit this user's actual request.",
        source="generated",
        rationale="The user's request is small and does not need a large checklist.",
    )

    assert contract.source == "generated"
    assert "request_fit" in contract.rubric


def test_parse_planning_rubric_evaluation_from_json_text():
    payload = {
        "result": "needs_revision",
        "explanation": "Two tasks are vague.",
        "criteria": [
            {
                "name": "concrete_backend_scope",
                "passed": False,
                "gap": "Replace 'fix backend' with concrete files and behavior.",
            },
            {"name": "preserve_existing_behavior", "passed": True},
        ],
    }

    parsed = parse_planning_rubric_evaluation(json.dumps(payload), iteration=2)

    assert parsed.iteration == 2
    assert parsed.result == "needs_revision"
    assert parsed.criteria[0].name == "concrete_backend_scope"
    assert parsed.criteria[0].gap == "Replace 'fix backend' with concrete files and behavior."


def test_parse_planning_rubric_evaluation_strips_markdown_fence():
    raw = (
        "```json\n"
        '{"result":"satisfied","explanation":"ok","criteria":[{"name":"request_fit","passed":true}]}'
        "\n```"
    )

    parsed = parse_planning_rubric_evaluation(raw, iteration=0)

    assert parsed.result == "satisfied"
    assert parsed.criteria[0].passed is True


def test_build_planning_rubric_feedback_includes_failed_criteria_only():
    evaluation = PlanningRubricEvaluation(
        iteration=0,
        result="needs_revision",
        explanation="Needs work.",
        criteria=[
            {"name": "concrete_backend_scope", "passed": False, "gap": "Make task 1 concrete."},
            {"name": "preserve_existing_behavior", "passed": True},
        ],
    )

    feedback = build_planning_rubric_feedback(evaluation)

    assert "Needs work." in feedback
    assert "concrete_backend_scope" in feedback
    assert "Make task 1 concrete." in feedback
    assert "preserve_existing_behavior" not in feedback


def test_attempt_metadata_excludes_none_fields():
    attempt = PlanningRubricAttempt(
        status="satisfied",
        grading_run_id="run-1",
        iterations=1,
        source="generate_plan",
        rubric=FALLBACK_PLANNING_RUBRIC,
        evaluations=[
            PlanningRubricEvaluation(
                iteration=0,
                result="satisfied",
                explanation="All criteria passed.",
                criteria=[{"name": "request_fit", "passed": True}],
            )
        ],
    )

    metadata = attempt.metadata()

    assert metadata["status"] == "satisfied"
    assert metadata["evaluations"][0]["criteria"][0]["name"] == "request_fit"
    assert "error" not in metadata
