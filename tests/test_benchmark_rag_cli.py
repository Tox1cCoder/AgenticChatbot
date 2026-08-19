"""Contracts for the thousand-document RAG scale-benchmark harness.

This harness cannot produce real numbers in a checkout with no live
PostgreSQL/Qdrant/Redis and no 1,000-document corpus (see
``eval/rag/corpus_manifest.jsonl``, which holds 11 fixtures). These tests
verify the harness fails closed and reports "unexecuted" honestly rather
than fabricating a result, and that its orchestration wires real
measurements through when every prerequisite (corpus scale, an ingestion
target, a query target, a metrics source) is actually supplied.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _script():
    spec = importlib.util.spec_from_file_location("benchmark_rag_cli", "scripts/benchmark_rag.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_report_fixture() -> dict:
    path = Path("tests/fixtures/benchmark_report_schema_example.json")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Step 1 (brief): CLI defaults and report-schema validation
# ---------------------------------------------------------------------------


def test_benchmark_defaults_to_required_scale():
    script = _script()
    args = script.parse_args([])

    assert args.documents == 1000
    assert args.output.endswith(".json")


def test_report_contains_quality_latency_capacity_and_recovery_sections():
    script = _script()
    report = benchmark_report_fixture()

    script.validate_report(report)

    assert report.keys() >= {"corpus", "quality", "latency", "capacity", "failures"}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def test_output_must_end_with_json(capsys):
    script = _script()

    try:
        script.parse_args(["--output", "artifacts/rag-scale.txt"])
    except SystemExit as exit_info:
        assert exit_info.code != 0
    else:
        raise AssertionError("non-.json --output was accepted")


def test_documents_and_concurrency_flags_round_trip():
    script = _script()

    args = script.parse_args(
        [
            "--documents",
            "50",
            "--ingest-concurrency",
            "4",
            "--query-concurrency",
            "16",
            "--failure-injection",
        ]
    )

    assert args.documents == 50
    assert args.ingest_concurrency == 4
    assert args.query_concurrency == 16
    assert args.failure_injection is True


def test_failure_injection_defaults_to_disabled():
    script = _script()

    args = script.parse_args([])

    assert args.failure_injection is False


# ---------------------------------------------------------------------------
# Config hash and git SHA -- every report must be traceable to its inputs
# ---------------------------------------------------------------------------


def test_config_hash_is_stable_for_identical_args_and_redacts_secrets():
    script = _script()
    args_with_secret = script.parse_args(["--auth-token", "super-secret-token"])
    args_without_secret = script.parse_args([])

    hash_with_secret = script.compute_config_hash(args_with_secret)
    hash_again = script.compute_config_hash(args_with_secret)

    assert hash_with_secret == hash_again
    assert hash_with_secret != script.compute_config_hash(args_without_secret)
    assert "super-secret-token" not in json.dumps(script.redacted_config(args_with_secret))


def test_current_git_sha_returns_a_real_sha_in_this_repository():
    script = _script()

    sha = script.current_git_sha()

    assert isinstance(sha, str)
    assert len(sha) == 40 or sha == "unknown"


# ---------------------------------------------------------------------------
# Corpus sufficiency gate -- this is the honest core of the harness
# ---------------------------------------------------------------------------


def test_assess_corpus_reports_the_real_fixture_count_as_insufficient():
    script = _script()

    corpus = script.assess_corpus(Path("eval/rag/corpus_manifest.jsonl"), documents_requested=1000)

    assert corpus["documents_available"] == 11
    assert corpus["documents_requested"] == 1000
    assert corpus["sufficient"] is False


def test_assess_corpus_is_sufficient_when_manifest_meets_the_request():
    script = _script()

    corpus = script.assess_corpus(Path("eval/rag/corpus_manifest.jsonl"), documents_requested=5)

    assert corpus["sufficient"] is True


# ---------------------------------------------------------------------------
# Prometheus histogram parsing -- real math over real exported samples
# ---------------------------------------------------------------------------

_SAMPLE_METRICS_TEXT = """
# HELP rag_stage_duration_seconds Wall-clock duration of one RAG pipeline stage.
# TYPE rag_stage_duration_seconds histogram
rag_stage_duration_seconds_bucket{stage="dense_retrieval",le="0.005"} 0
rag_stage_duration_seconds_bucket{stage="dense_retrieval",le="0.01"} 2
rag_stage_duration_seconds_bucket{stage="dense_retrieval",le="0.025"} 8
rag_stage_duration_seconds_bucket{stage="dense_retrieval",le="0.05"} 10
rag_stage_duration_seconds_bucket{stage="dense_retrieval",le="+Inf"} 10
rag_stage_duration_seconds_count{stage="dense_retrieval"} 10
rag_stage_duration_seconds_sum{stage="dense_retrieval"} 0.2
# HELP rag_cache_operations_total Exact-cache lookups.
# TYPE rag_cache_operations_total counter
rag_cache_operations_total{cache="query_embedding",result="hit"} 30
rag_cache_operations_total{cache="query_embedding",result="miss"} 10
""".strip()


def test_stage_latency_from_metrics_text_computes_percentiles_from_real_buckets():
    # Hand-computed from _SAMPLE_METRICS_TEXT's cumulative buckets
    # (le=0.005:0, le=0.01:2, le=0.025:8, le=0.05:10, le=+Inf:10; total=10)
    # via the same linear interpolation PromQL's histogram_quantile() uses:
    #   p50: target=5.0 -> between (0.01, 2) and (0.025, 8):
    #        0.01 + (5-2)/(8-2)*(0.025-0.01) = 0.0175s
    #   p95: target=9.5 -> between (0.025, 8) and (0.05, 10):
    #        0.025 + (9.5-8)/(10-8)*(0.05-0.025) = 0.04375s
    #   p99: target=9.9 -> between (0.025, 8) and (0.05, 10):
    #        0.025 + (9.9-8)/(10-8)*(0.05-0.025) = 0.04875s
    # An interpolation or off-by-one bug that preserved ordering would not be
    # caught by asserting p50 <= p95 <= p99 alone, so these are exact values.
    script = _script()

    latency = script.stage_latency_from_metrics_text(_SAMPLE_METRICS_TEXT)

    assert "dense_retrieval" in latency
    stage = latency["dense_retrieval"]
    assert stage["sample_count"] == 10
    assert stage["p50_ms"] == 17.5
    assert stage["p95_ms"] == 43.75
    assert stage["p99_ms"] == 48.75


def test_stage_latency_from_metrics_text_ignores_unrelated_families():
    script = _script()

    latency = script.stage_latency_from_metrics_text("# no matching families\n")

    assert latency == {}


def test_cache_ratios_from_metrics_text_computes_hit_ratio():
    script = _script()

    ratios = script.cache_ratios_from_metrics_text(_SAMPLE_METRICS_TEXT)

    assert ratios["query_embedding"]["hits"] == 30
    assert ratios["query_embedding"]["misses"] == 10
    assert ratios["query_embedding"]["hit_ratio"] == 0.75


def test_histogram_quantile_returns_none_for_empty_total():
    script = _script()

    assert script._histogram_quantile([1.0, float("inf")], [0.0, 0.0], 0.0, 0.5) is None


# ---------------------------------------------------------------------------
# validate_report -- the anti-fabrication guard
# ---------------------------------------------------------------------------


def test_validate_report_rejects_missing_sections():
    script = _script()
    report = benchmark_report_fixture()
    del report["capacity"]

    try:
        script.validate_report(report)
    except ValueError as error:
        assert "capacity" in str(error)
    else:
        raise AssertionError("missing section was accepted")


def test_validate_report_rejects_a_number_reported_alongside_measured_false():
    script = _script()
    report = benchmark_report_fixture()
    report["capacity"]["total_chunks"] = 42

    try:
        script.validate_report(report)
    except ValueError as error:
        assert "total_chunks" in str(error)
    else:
        raise AssertionError("a fabricated number next to measured=False was accepted")


def test_validate_report_rejects_a_deferred_metric_reported_as_zero():
    script = _script()
    report = benchmark_report_fixture()
    report["capacity"]["measured"] = True
    report["capacity"]["cost_per_document_usd"] = 0.0

    try:
        script.validate_report(report)
    except ValueError as error:
        assert "cost_per_document_usd" in str(error)
    else:
        raise AssertionError("a deferred metric reported as a computed zero was accepted")


def test_validate_report_rejects_unexecuted_status_without_prerequisites():
    script = _script()
    report = benchmark_report_fixture()
    report["provenance"]["prerequisites"] = []

    try:
        script.validate_report(report)
    except ValueError as error:
        assert "prerequisites" in str(error)
    else:
        raise AssertionError("unexecuted status without a listed prerequisite was accepted")


def test_validate_report_accepts_a_fully_measured_report():
    script = _script()
    report = benchmark_report_fixture()
    report["status"] = "executed"
    report["corpus"] = {
        "manifest_path": "eval/rag/corpus_manifest.jsonl",
        "documents_requested": 5,
        "documents_available": 11,
        "sufficient": True,
    }
    report["quality"] = {
        "measured": True,
        "rows_evaluated": 290,
        "aggregate": {"document_recall_at_5": 0.9},
        "distractor": {"document_recall_at_5": 0.8},
        "abstention_precision": 0.95,
        "abstention_recall": 0.9,
        "tool_iteration_metrics": None,
        "reason": None,
    }
    report["latency"] = {
        "measured": True,
        "tenant_filter_ms": {"p50": 100.0, "p95": 200.0, "p99": 250.0, "n": 290},
        "stages": {"dense_retrieval": {"p50_ms": 10.0, "p95_ms": 20.0, "p99_ms": 25.0}},
        "source": "http://localhost:8000/metrics/rag",
        "reason": None,
    }
    report["capacity"] = {
        "measured": True,
        "total_chunks": 500,
        "indexing_lag_seconds": {"p50_s": 1.0, "p95_s": 2.0, "p99_s": 3.0},
        "vector_memory": {"points_count": 500},
        "queue_saturation": None,
        "cache_ratios": {"query_embedding": {"hit_ratio": 0.5}},
        "cost_usd": 1.23,
        "cost_per_document_usd": None,
        "cost_per_question_usd": None,
        "cached_input_token_ratio": None,
        "reason": None,
    }
    report["failures"] = {
        "measured": True,
        "requested": True,
        "kinds": ["provider", "qdrant", "redis", "worker"],
        "results": [{"kind": "provider", "injected": True, "recovered": True}],
        "reason": None,
    }
    report["provenance"] = {"prerequisites": [], "notes": []}

    script.validate_report(report)  # must not raise


# ---------------------------------------------------------------------------
# run_benchmark orchestration -- dependency-injected, no real network calls
# ---------------------------------------------------------------------------


def test_run_benchmark_reports_unexecuted_when_corpus_is_insufficient():
    script = _script()
    args = script.parse_args(["--documents", "1000"])

    report = script.run_benchmark(args)

    script.validate_report(report)
    assert report["status"] == "unexecuted"
    assert report["corpus"]["sufficient"] is False
    assert report["quality"]["measured"] is False
    assert report["latency"]["measured"] is False
    assert report["capacity"]["measured"] is False
    assert report["failures"]["measured"] is False
    assert len(report["provenance"]["prerequisites"]) >= 1


def test_run_benchmark_reports_unexecuted_when_corpus_sufficient_but_no_targets_wired():
    script = _script()
    args = script.parse_args(["--documents", "5"])

    report = script.run_benchmark(args)

    assert report["status"] == "unexecuted"
    assert report["corpus"]["sufficient"] is True
    assert any("ingestion target" in item for item in report["provenance"]["prerequisites"])
    assert any("query target" in item for item in report["provenance"]["prerequisites"])


def test_run_benchmark_executes_fully_when_every_dependency_is_injected():
    script = _script()
    args = script.parse_args(
        ["--documents", "5", "--allow-partial-corpus", "--failure-injection"]
    )

    def fake_ingest(corpus_entries, *, concurrency):
        return [
            {
                "document_id": entry["document_id"],
                "status": "READY",
                "elapsed_s": 1.5,
                "chunk_count": 3,
            }
            for entry in corpus_entries
        ]

    def fake_query_target(inputs):
        return {"answer": "The warranty period is two years.", "abstained": False}

    def fake_metrics_fetcher():
        return _SAMPLE_METRICS_TEXT

    def fake_failure_injector(kind):
        return {"injected": True, "recovered": True, "recovery_seconds": 4.2}

    report = script.run_benchmark(
        args,
        ingest_fn=fake_ingest,
        query_target=fake_query_target,
        metrics_fetcher=fake_metrics_fetcher,
        failure_injector=fake_failure_injector,
    )

    script.validate_report(report)
    # 11 manifest fixtures satisfy the requested 5, so this is a real "executed" run,
    # not merely allowed through by --allow-partial-corpus.
    assert report["status"] == "executed"
    assert report["quality"]["measured"] is True
    assert report["quality"]["rows_evaluated"] > 0
    assert report["latency"]["measured"] is True
    assert "dense_retrieval" in report["latency"]["stages"]
    assert report["capacity"]["measured"] is True
    assert report["capacity"]["total_chunks"] == 11 * 3
    assert report["failures"]["measured"] is True
    assert len(report["failures"]["results"]) == 4


def test_run_benchmark_reports_partial_when_scale_requirement_is_overridden():
    script = _script()
    args = script.parse_args(["--documents", "1000", "--allow-partial-corpus"])

    report = script.run_benchmark(
        args,
        ingest_fn=lambda entries, *, concurrency: [
            {"document_id": e["document_id"], "status": "READY", "elapsed_s": 1.0, "chunk_count": 1}
            for e in entries
        ],
        query_target=lambda inputs: {"answer": "x", "abstained": True},
        metrics_fetcher=lambda: _SAMPLE_METRICS_TEXT,
    )

    script.validate_report(report)
    assert report["status"] == "partial"
    assert report["corpus"]["sufficient"] is False
    assert any("11" in note or "1000" in note for note in report["provenance"]["notes"])


def test_run_benchmark_never_populates_deferred_metrics_even_when_executed():
    script = _script()
    args = script.parse_args(["--documents", "5", "--allow-partial-corpus"])

    def fake_ingest(corpus_entries, *, concurrency):
        return [
            {
                "document_id": e["document_id"],
                "status": "READY",
                "elapsed_s": 1.0,
                "chunk_count": None,
            }
            for e in corpus_entries
        ]

    report = script.run_benchmark(
        args,
        ingest_fn=fake_ingest,
        query_target=lambda inputs: {"answer": "x", "abstained": True},
        metrics_fetcher=lambda: _SAMPLE_METRICS_TEXT,
    )

    assert report["capacity"]["cost_per_document_usd"] is None
    assert report["capacity"]["cost_per_question_usd"] is None
    assert report["capacity"]["cached_input_token_ratio"] is None
    assert report["quality"]["tool_iteration_metrics"] is None


def test_run_benchmark_records_git_sha_and_config_hash():
    script = _script()
    args = script.parse_args(["--documents", "1000"])

    report = script.run_benchmark(args)

    assert report["git_sha"]
    assert report["config_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# write_report / main() wiring
# ---------------------------------------------------------------------------


def test_write_report_creates_parent_directories(tmp_path):
    script = _script()
    report = benchmark_report_fixture()
    output_path = tmp_path / "nested" / "rag-scale.json"

    script.write_report(report, output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def test_main_writes_unexecuted_report_and_returns_nonzero_when_prerequisites_missing(
    tmp_path, monkeypatch
):
    script = _script()
    output_path = tmp_path / "rag-scale.json"

    exit_code = script.main(["--documents", "1000", "--output", str(output_path)])

    assert exit_code != 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "unexecuted"


def test_main_returns_zero_when_run_benchmark_executes_fully(tmp_path, monkeypatch):
    script = _script()
    output_path = tmp_path / "rag-scale.json"

    fake_report = {
        "schema_version": "1.0",
        "status": "executed",
        "generated_at": "2026-08-19T00:00:00+00:00",
        "git_sha": "deadbeef",
        "config_hash": "sha256:abc",
        "config": {},
        "corpus": {
            "manifest_path": "x",
            "documents_requested": 5,
            "documents_available": 11,
            "sufficient": True,
        },
        "quality": {
            "measured": True,
            "rows_evaluated": 1,
            "aggregate": {},
            "distractor": None,
            "abstention_precision": None,
            "abstention_recall": None,
            "tool_iteration_metrics": None,
            "reason": None,
        },
        "latency": {
            "measured": True,
            "tenant_filter_ms": {"p50": 1.0, "p95": 1.0, "p99": 1.0, "n": 1},
            "stages": {},
            "source": "x",
            "reason": None,
        },
        "capacity": {
            "measured": True,
            "total_chunks": 1,
            "indexing_lag_seconds": None,
            "vector_memory": None,
            "queue_saturation": None,
            "cache_ratios": None,
            "cost_usd": None,
            "cost_per_document_usd": None,
            "cost_per_question_usd": None,
            "cached_input_token_ratio": None,
            "reason": None,
        },
        "failures": {
            "measured": False,
            "requested": False,
            "kinds": ["provider", "qdrant", "redis", "worker"],
            "results": [],
            "reason": "not requested",
        },
        "provenance": {"prerequisites": [], "notes": []},
    }
    monkeypatch.setattr(script, "run_benchmark", lambda args, **_deps: fake_report)

    exit_code = script.main(["--documents", "5", "--output", str(output_path)])

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "executed"
