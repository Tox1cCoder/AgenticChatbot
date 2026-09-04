"""Failure-closed execution contracts for the evaluation CLI."""

from __future__ import annotations

import importlib.util
import json


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


def _reviewed_dataset(monkeypatch, script):
    """Clear the pending-human-review guard so the comparison branch runs."""
    monkeypatch.setattr(
        script,
        "load_golden_dataset",
        lambda _path: [{"metadata": {"label_review_status": "reviewed"}}],
    )


def test_compare_baseline_awaits_the_smithdb_comparison_query(monkeypatch, capsys):
    script = _script()
    monkeypatch.setenv("LANGSMITH_API_KEY", "configured")
    _reviewed_dataset(monkeypatch, script)
    client = object()
    monkeypatch.setattr(
        script,
        "run_online",
        lambda _args: (client, type("Results", (), {"experiment_name": "candidate"})()),
    )
    recorded: list[tuple[object, str, str]] = []

    async def comparison_metrics(client_argument, candidate_name, baseline_name):
        recorded.append((client_argument, candidate_name, baseline_name))
        return (
            {"document_recall_at_5": 0.9, "abstention_recall": 0.8},
            {"document_recall_at_5": 0.9, "abstention_recall": 0.8},
        )

    monkeypatch.setattr(script, "comparison_metrics", comparison_metrics)

    exit_code = script.main(["--compare-baseline", "baseline"])

    assert exit_code == 0
    assert recorded == [(client, "candidate", "baseline")]
    verdicts = json.loads(capsys.readouterr().out.splitlines()[0])
    assert {verdict["metric"] for verdict in verdicts} >= {"document_recall_at_5"}


def test_compare_baseline_reports_a_failed_smithdb_query(monkeypatch):
    script = _script()
    monkeypatch.setenv("LANGSMITH_API_KEY", "configured")
    _reviewed_dataset(monkeypatch, script)
    monkeypatch.setattr(
        script,
        "run_online",
        lambda _args: (object(), type("Results", (), {"experiment_name": "candidate"})()),
    )

    async def comparison_metrics(*_args):
        raise ValueError("baseline experiment has no deterministic feedback: baseline")

    monkeypatch.setattr(script, "comparison_metrics", comparison_metrics)

    assert script.main(["--compare-baseline", "baseline"]) == 1


def test_compare_baseline_requires_the_candidate_experiment_name(monkeypatch):
    script = _script()
    monkeypatch.setenv("LANGSMITH_API_KEY", "configured")
    _reviewed_dataset(monkeypatch, script)
    monkeypatch.setattr(
        script, "run_online", lambda _args: (object(), type("Results", (), {})())
    )

    async def comparison_metrics(*_args):
        raise AssertionError("comparison must not run without a candidate experiment name")

    monkeypatch.setattr(script, "comparison_metrics", comparison_metrics)

    assert script.main(["--compare-baseline", "baseline"]) == 1
