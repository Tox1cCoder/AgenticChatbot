"""Contracts for the embedding-dimension / provider-Batch comparison harness.

Comparing 768 vs 1536 vs 3072 dimensions, and synchronous vs Gemini Batch
indexing, requires real embedding-provider calls this environment has no
budget or credentials for. These tests verify the CLI, the report schema,
and the orchestration logic (dependency-injected, no network) fail closed
and report "unexecuted" honestly when no ``--experiment-hook`` is wired,
and assemble a truthful, schema-valid report when one is.
"""

from __future__ import annotations

import importlib.util
import json


def _script():
    spec = importlib.util.spec_from_file_location(
        "experiment_embedding_dimensions_cli", "scripts/experiment_embedding_dimensions.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI defaults
# ---------------------------------------------------------------------------


def test_defaults_compare_the_required_dimension_matrix():
    script = _script()

    args = script.parse_args([])

    assert args.dimensions == [768, 1536, 3072]
    assert args.output.endswith(".json")
    assert args.include_provider_batch is False


def test_output_must_end_with_json():
    script = _script()

    try:
        script.parse_args(["--output", "artifacts/matrix.txt"])
    except SystemExit as exit_info:
        assert exit_info.code != 0
    else:
        raise AssertionError("non-.json --output was accepted")


def test_dimensions_and_provider_batch_flag_round_trip():
    script = _script()

    args = script.parse_args(
        ["--dimensions", "768", "1536", "--include-provider-batch", "--dataset-tag", "v2"]
    )

    assert args.dimensions == [768, 1536]
    assert args.include_provider_batch is True
    assert args.dataset_tag == "v2"


# ---------------------------------------------------------------------------
# validate_report -- the anti-fabrication guard
# ---------------------------------------------------------------------------


def _unexecuted_fixture(script) -> dict:
    args = script.parse_args([])
    return script.run_experiment(args)


def test_report_has_the_required_sections():
    script = _script()
    report = _unexecuted_fixture(script)

    script.validate_report(report)

    assert report.keys() >= {"corpus", "dataset", "dimensions", "comparison", "provenance"}


def test_report_is_unexecuted_without_an_experiment_hook():
    script = _script()
    report = _unexecuted_fixture(script)

    assert report["status"] == "unexecuted"
    assert all(cell["measured"] is False for cell in report["dimensions"].values())
    assert report["comparison"]["measured"] is False
    assert len(report["provenance"]["prerequisites"]) >= 1


def test_validate_report_rejects_a_number_reported_next_to_measured_false():
    script = _script()
    report = _unexecuted_fixture(script)
    report["dimensions"]["768"]["indexing"]["synchronous"]["cost_usd"] = 12.5

    try:
        script.validate_report(report)
    except ValueError as error:
        assert "cost_usd" in str(error)
    else:
        raise AssertionError("a fabricated number next to measured=False was accepted")


def test_validate_report_rejects_missing_sections():
    script = _script()
    report = _unexecuted_fixture(script)
    del report["comparison"]

    try:
        script.validate_report(report)
    except ValueError as error:
        assert "comparison" in str(error)
    else:
        raise AssertionError("missing section was accepted")


def test_validate_report_rejects_unexecuted_status_without_prerequisites():
    script = _script()
    report = _unexecuted_fixture(script)
    report["provenance"]["prerequisites"] = []

    try:
        script.validate_report(report)
    except ValueError as error:
        assert "prerequisites" in str(error)
    else:
        raise AssertionError("unexecuted status without a listed prerequisite was accepted")


# ---------------------------------------------------------------------------
# run_experiment orchestration -- dependency-injected, no network calls
# ---------------------------------------------------------------------------


def _fake_hook_factory(*, fail_dimension: int | None = None):
    calls: list[tuple[int, str]] = []

    def hook(dimension: int, mode: str) -> dict:
        calls.append((dimension, mode))
        if dimension == fail_dimension:
            raise RuntimeError(f"provider outage at dimension {dimension}")
        cell = {
            "documents_indexed": 11,
            "wall_time_s": 10.0 if mode == "synchronous" else 4.0,
            "cost_usd": 0.05 if mode == "synchronous" else 0.03,
            "failures": 0,
            "operational_notes": f"{mode} indexing completed",
            "quality": {"document_recall_at_5": 0.9 - dimension / 100000}
            if mode == "synchronous"
            else None,
        }
        return cell

    hook.calls = calls
    return hook


def test_run_experiment_executes_synchronous_only_by_default():
    script = _script()
    args = script.parse_args(["--dimensions", "768", "1536"])
    hook = _fake_hook_factory()

    report = script.run_experiment(args, experiment_hook=hook)

    script.validate_report(report)
    assert report["status"] == "executed"
    assert {dimension for dimension, _mode in hook.calls} == {768, 1536}
    assert all(mode == "synchronous" for _dimension, mode in hook.calls)
    assert report["dimensions"]["768"]["indexing"]["provider_batch"]["measured"] is False
    assert report["dimensions"]["768"]["quality"]["document_recall_at_5"] > 0


def test_run_experiment_includes_provider_batch_when_requested():
    script = _script()
    args = script.parse_args(["--dimensions", "768", "--include-provider-batch"])
    hook = _fake_hook_factory()

    report = script.run_experiment(args, experiment_hook=hook)

    assert sorted(hook.calls) == [(768, "provider_batch"), (768, "synchronous")]
    batch_cell = report["dimensions"]["768"]["indexing"]["provider_batch"]
    sync_cell = report["dimensions"]["768"]["indexing"]["synchronous"]
    assert batch_cell["measured"] is True
    assert batch_cell["wall_time_s"] < sync_cell["wall_time_s"]


def test_run_experiment_marks_a_failed_dimension_as_partial():
    script = _script()
    args = script.parse_args(["--dimensions", "768", "1536"])
    hook = _fake_hook_factory(fail_dimension=1536)

    report = script.run_experiment(args, experiment_hook=hook)

    assert report["status"] == "partial"
    assert report["dimensions"]["768"]["measured"] is True
    assert report["dimensions"]["1536"]["measured"] is False
    assert "provider outage" in report["dimensions"]["1536"]["reason"]


def test_run_experiment_never_fabricates_a_recommendation_without_operator_tolerance():
    script = _script()
    args = script.parse_args(["--dimensions", "768", "1536", "3072"])
    hook = _fake_hook_factory()

    report = script.run_experiment(args, experiment_hook=hook)

    assert report["comparison"]["recommendation"]["selected_dimension"] is None
    assert report["comparison"]["recommendation"]["rationale"] is None


def test_run_experiment_computes_a_recommendation_only_from_measured_deltas():
    script = _script()
    args = script.parse_args(
        ["--dimensions", "768", "1536", "3072", "--recall-parity-tolerance", "0.01"]
    )
    hook = _fake_hook_factory()

    report = script.run_experiment(args, experiment_hook=hook)

    recommendation = report["comparison"]["recommendation"]
    assert recommendation["selected_dimension"] in {768, 1536, 3072}
    assert "document_recall_at_5" in recommendation["rationale"]


def test_run_experiment_records_git_sha_and_config_hash():
    script = _script()
    args = script.parse_args([])

    report = script.run_experiment(args)

    assert report["git_sha"]
    assert report["config_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# main() wiring
# ---------------------------------------------------------------------------


def test_main_writes_unexecuted_report_and_returns_nonzero(tmp_path):
    script = _script()
    output_path = tmp_path / "matrix.json"

    exit_code = script.main(["--output", str(output_path)])

    assert exit_code != 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "unexecuted"


def test_main_returns_zero_when_run_experiment_executes_fully(tmp_path, monkeypatch):
    script = _script()
    output_path = tmp_path / "matrix.json"
    fake_report = _unexecuted_fixture(script)
    fake_report["status"] = "executed"
    monkeypatch.setattr(script, "run_experiment", lambda args, **_deps: fake_report)

    exit_code = script.main(["--output", str(output_path)])

    assert exit_code == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "executed"


def test_resolve_callable_loads_a_real_module_attribute():
    script = _script()

    resolved = script._resolve_callable("json:dumps")

    assert resolved is not None
    assert resolved({"a": 1}) == '{"a": 1}'


def test_resolve_callable_returns_none_for_empty_reference():
    script = _script()

    assert script._resolve_callable(None) is None
    assert script._resolve_callable("") is None


def test_resolve_callable_rejects_a_malformed_reference():
    script = _script()

    try:
        script._resolve_callable("not-a-module-colon-callable")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed MODULE:CALLABLE reference was accepted")
