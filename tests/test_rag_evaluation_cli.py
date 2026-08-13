"""Failure-closed execution contracts for the evaluation CLI."""

from __future__ import annotations

import importlib.util


def _script():
    spec = importlib.util.spec_from_file_location("evaluate_rag_cli", "scripts/evaluate_rag.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_online_run_without_credentials_fails_closed(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    assert _script().main([]) == 1


def test_quota_failure_returns_nonzero_for_requested_online_experiment(monkeypatch):
    script = _script()
    monkeypatch.setenv("LANGSMITH_API_KEY", "configured")
    monkeypatch.setattr(
        script, "run_online", lambda _: (_ for _ in ()).throw(RuntimeError("HTTP 429 quota"))
    )

    assert script.main(["--compare-baseline", "baseline"]) == 1


def test_offline_runner_executes_the_target_and_evaluators():
    script = _script()
    calls: list[dict[str, object]] = []

    def target(inputs):
        calls.append(inputs)
        return {"answer": "", "abstained": True}

    summary = script.run_offline(script.parse_args(["--offline"]), target=target)

    assert 100 <= len(calls) <= 300
    assert summary["mode"] == "offline"
    assert "abstention_precision" in summary["metrics"]
    assert "document_recall_at_1" in summary["metrics"]


def test_offline_target_receives_only_inputs_not_gold_reference_outputs():
    script = _script()

    def target(inputs):
        assert set(inputs) == {"question", "user_id", "conversation_id"}
        return {"answer": "", "abstained": True}

    script.run_offline(script.parse_args(["--offline"]), target=target)


def test_offline_cli_fails_without_an_explicit_local_target():
    assert _script().main(["--offline"]) == 1


def test_compare_baseline_rejects_pending_human_review(monkeypatch):
    script = _script()
    monkeypatch.setenv("LANGSMITH_API_KEY", "configured")

    class Client:
        def list_examples(self, **_):
            return []

        def evaluate(self, *_args, **_kwargs):
            return type("Results", (), {"experiment_name": "candidate"})()

    monkeypatch.setattr(
        script,
        "run_online",
        lambda _args: (Client(), type("Results", (), {"experiment_name": "candidate"})()),
    )

    assert script.main(["--compare-baseline", "baseline"]) == 1


def test_online_runner_validates_remote_example_references_before_evaluation():
    script = _script()

    class Example:
        inputs = {"question": "q", "user_id": "u", "conversation_id": "c"}
        outputs = {"relevant_document_ids": ["missing-document"], "relevant_spans": []}

    class Client:
        def list_examples(self, **_):
            return [Example()]

        def evaluate(self, *_args, **_kwargs):
            raise AssertionError("remote labels must be validated before evaluation")

    try:
        script.run_online(
            script.parse_args([]),
            client_factory=Client,
            target_factory=lambda: lambda payload: payload,
            prepare_scope=lambda: None,
        )
    except ValueError as error:
        assert "unknown documents" in str(error)
    else:
        raise AssertionError("stale remote label was accepted")


def test_experiment_metrics_include_summary_evaluator_scores():
    script = _script()

    class Frame:
        columns = ["feedback.document_recall_at_5"]

        def __getitem__(self, _):
            return type("Series", (), {"mean": lambda self: 0.8})()

    class Project:
        feedback_stats = {"abstention_precision": {"avg": 0.7}}
        session_feedback_stats = {"abstention_recall": {"avg": 0.6}}

    class Client:
        def get_test_results(self, **_):
            return Frame()

        def read_project(self, **_):
            return Project()

    assert script.experiment_metrics(Client(), "experiment") == {
        "document_recall_at_5": 0.8,
        "abstention_precision": 0.7,
        "abstention_recall": 0.6,
    }
