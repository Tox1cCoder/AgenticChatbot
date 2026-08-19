"""Qualify RAG behavior at representative corpus scale.

This harness answers one question honestly: does the RAG pipeline still
meet its quality, latency, capacity, and recovery bar once a representative
corpus (by default 1,000 documents) is loaded and under concurrent traffic?

It never estimates, extrapolates, or hand-writes a result. Every section of
the produced report starts ``measured: false`` / ``null`` and is only filled
in from a real measurement taken during this run:

- ``corpus`` -- checked directly against the manifest on disk. No network
  required, always populated.
- ``quality`` -- deterministic RAG metrics (recall, citation validity,
  abstention, plus a "distractor" breakdown) computed by replaying
  ``eval/rag/golden_v1.jsonl`` through a live query target while the scale
  corpus sits loaded, using the exact evaluator functions
  ``scripts/evaluate_rag.py`` uses (``app.evaluation.rag.metrics``).
- ``latency`` -- p50/p95/p99 stage latency read from the real
  ``rag_stage_duration_seconds`` histogram exported at ``/metrics/rag``
  (``app/observability/rag.py``, Task 12), plus client-observed
  tenant-filter latency timed around the query phase.
- ``capacity`` -- chunk counts and indexing lag observed from the ingestion
  phase, plus optional vector-memory / queue-saturation / cost figures from
  operator-supplied hooks (see below). ``cost_per_document_usd``,
  ``cost_per_question_usd``, and ``cached_input_token_ratio`` are deferred
  pending a design decision (Task 12 round-1 review) and always stay
  ``null`` -- never a computed zero.
- ``failures`` -- provider/Qdrant/Redis/worker outage-recovery outcomes from
  an operator-supplied failure-injection hook.

If a prerequisite is missing -- an insufficient corpus, no reachable
ingestion/query target, no metrics source -- the run stops before writing
any number into these sections. The report is still written, with
``status: "unexecuted"`` and every missing prerequisite listed under
``provenance.prerequisites``, so "nothing ran" is a first-class, inspectable
outcome rather than a silent failure.

Live-infrastructure hooks (all optional; each becomes ``null``/``measured:
false`` when omitted):

    --qdrant-stats-hook MODULE:CALLABLE     () -> dict of Qdrant collection stats
    --queue-inspector-hook MODULE:CALLABLE  () -> dict of queue-depth/worker stats
    --cost-fetcher-hook MODULE:CALLABLE     () -> float total cost in USD
    --failure-injector-hook MODULE:CALLABLE (kind: str) -> dict recovery outcome

Usage::

    python scripts/benchmark_rag.py \\
        --documents 1000 \\
        --conversation-id <uuid-to-load-the-scale-corpus-into> \\
        --auth-token <bearer-token> \\
        --ingest-concurrency 8 \\
        --query-concurrency 32 \\
        --failure-injection \\
        --output artifacts/rag-scale-1000.json

Against this repository today (11 fixtures in
``eval/rag/corpus_manifest.jsonl``, no live PostgreSQL/Qdrant/Redis), this
always produces a ``status: "unexecuted"`` report -- that is the correct,
honest result here, not a bug.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.rag.corpus import load_corpus_manifest, load_golden_dataset
from app.evaluation.rag.metrics import (
    abstention_metrics,
    abstention_summary_metrics,
    evaluate_output,
    output_from_mapping,
    reference_from_mapping,
)
from app.evaluation.rag.target import build_http_target, load_local_target
from scripts.benchmark_ingestion import (
    _get_auth_headers,
    _login,
    _percentile,
    _process_document,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_MANIFEST = ROOT / "eval" / "rag" / "corpus_manifest.jsonl"
DEFAULT_GOLDEN_DATASET = ROOT / "eval" / "rag" / "golden_v1.jsonl"
DEFAULT_OUTPUT = "artifacts/rag-scale-report.json"
SCHEMA_VERSION = "1.0"

REQUIRED_REPORT_SECTIONS = {"corpus", "quality", "latency", "capacity", "failures"}
_VALID_STATUSES = {"executed", "partial", "unexecuted"}
_FAILURE_KINDS = ("provider", "qdrant", "redis", "worker")
_SECRET_ARG_NAMES = {"auth_token", "password"}

# Sections gated by a "measured" flag; keys allowed to stay non-null even
# when measured is False (bookkeeping, not a measurement).
_ALWAYS_ALLOWED_NON_NULL = {
    "quality": {"measured", "reason"},
    "latency": {"measured", "reason"},
    "capacity": {"measured", "reason"},
    "failures": {"measured", "reason", "requested", "kinds"},
}
_DEFERRED_NULL_ONLY_FIELDS = (
    ("capacity", "cost_per_document_usd"),
    ("capacity", "cost_per_question_usd"),
    ("capacity", "cached_input_token_ratio"),
    ("quality", "tool_iteration_metrics"),
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=1000,
        help="Number of representative documents the corpus must supply (default: 1000).",
    )
    parser.add_argument("--corpus-manifest", default=str(DEFAULT_CORPUS_MANIFEST))
    parser.add_argument(
        "--allow-partial-corpus",
        action="store_true",
        help="Run against fewer documents than --documents requests (dev smoke test). "
        "The report is marked status='partial', never 'executed'.",
    )
    parser.add_argument("--golden-dataset", default=str(DEFAULT_GOLDEN_DATASET))
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--metrics-url", default=None, help="Defaults to '<base-url>/metrics/rag'."
    )
    parser.add_argument("--conversation-id", default=None)
    parser.add_argument("--auth-token", default=None)
    parser.add_argument("--email", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--target", metavar="MODULE:FUNCTION", default=None)
    parser.add_argument("--ingest-concurrency", type=int, default=8)
    parser.add_argument("--query-concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=300, help="Per-document ingest timeout.")
    parser.add_argument("--failure-injection", action="store_true")
    parser.add_argument("--qdrant-stats-hook", metavar="MODULE:CALLABLE", default=None)
    parser.add_argument("--queue-inspector-hook", metavar="MODULE:CALLABLE", default=None)
    parser.add_argument("--cost-fetcher-hook", metavar="MODULE:CALLABLE", default=None)
    parser.add_argument("--failure-injector-hook", metavar="MODULE:CALLABLE", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.output.endswith(".json"):
        parser.error("--output must end with .json")
    return args


def redacted_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {}
    for key, value in vars(args).items():
        config[key] = "<redacted>" if key in _SECRET_ARG_NAMES and value else value
    return config


def compute_config_hash(args: argparse.Namespace) -> str:
    payload = json.dumps(redacted_config(args), sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_git_sha(cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _resolve_callable(reference: str | None) -> Callable[..., Any] | None:
    if not reference:
        return None
    module_name, separator, attr_name = reference.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"'{reference}' must be MODULE:CALLABLE")
    attr = getattr(importlib.import_module(module_name), attr_name)
    if not callable(attr):
        raise ValueError(f"'{reference}' does not resolve to a callable")
    return attr


# ---------------------------------------------------------------------------
# Corpus sufficiency -- no network required, always real
# ---------------------------------------------------------------------------


def _corpus_summary(
    manifest_path: Path, entries: list[dict[str, Any]], documents_requested: int
) -> dict[str, Any]:
    documents_available = len(entries)
    return {
        "manifest_path": str(manifest_path),
        "documents_requested": documents_requested,
        "documents_available": documents_available,
        "sufficient": documents_available >= documents_requested,
    }


def assess_corpus(
    manifest_path: Path,
    documents_requested: int,
    *,
    manifest_loader: Callable[[Path], list[dict[str, Any]]] = load_corpus_manifest,
) -> dict[str, Any]:
    return _corpus_summary(manifest_path, manifest_loader(manifest_path), documents_requested)


# ---------------------------------------------------------------------------
# Real Prometheus histogram / counter parsing (Task 12 telemetry)
# ---------------------------------------------------------------------------


def _histogram_quantile(
    sorted_bounds: list[float], cumulative_counts: list[float], total: float, quantile: float
) -> float | None:
    """Linear-interpolate a quantile from cumulative histogram buckets.

    Same approach as PromQL's ``histogram_quantile()``: it reads real
    exported bucket counts, it does not re-time or estimate anything.
    """
    if total <= 0:
        return None
    target = quantile * total
    previous_bound, previous_count = 0.0, 0.0
    for bound, count in zip(sorted_bounds, cumulative_counts, strict=True):
        if count >= target:
            if bound == float("inf"):
                return previous_bound
            if count == previous_count:
                return bound
            fraction = (target - previous_count) / (count - previous_count)
            return previous_bound + fraction * (bound - previous_bound)
        previous_bound, previous_count = bound, count
    return sorted_bounds[-1] if sorted_bounds else None


def stage_latency_from_metrics_text(text: str) -> dict[str, dict[str, float | None]]:
    """Compute p50/p95/p99 per RAG stage from the real exported histogram."""
    from prometheus_client.parser import text_string_to_metric_families

    bucket_counts: dict[str, dict[float, float]] = {}
    for family in text_string_to_metric_families(text):
        if family.name != "rag_stage_duration_seconds":
            continue
        for sample in family.samples:
            if not sample.name.endswith("_bucket"):
                continue
            stage = sample.labels.get("stage", "other")
            le = sample.labels.get("le")
            if le is None:
                continue
            upper = float("inf") if le == "+Inf" else float(le)
            stage_buckets = bucket_counts.setdefault(stage, {})
            stage_buckets[upper] = stage_buckets.get(upper, 0.0) + sample.value

    result: dict[str, dict[str, float | None]] = {}
    for stage, buckets in bucket_counts.items():
        sorted_bounds = sorted(buckets)
        cumulative_counts = [buckets[bound] for bound in sorted_bounds]
        total = cumulative_counts[-1] if cumulative_counts else 0.0
        result[stage] = {
            "p50_ms": _quantile_ms(sorted_bounds, cumulative_counts, total, 0.50),
            "p95_ms": _quantile_ms(sorted_bounds, cumulative_counts, total, 0.95),
            "p99_ms": _quantile_ms(sorted_bounds, cumulative_counts, total, 0.99),
            "sample_count": total,
        }
    return result


def _quantile_ms(
    sorted_bounds: list[float], cumulative_counts: list[float], total: float, quantile: float
) -> float | None:
    seconds = _histogram_quantile(sorted_bounds, cumulative_counts, total, quantile)
    return None if seconds is None else round(seconds * 1000.0, 3)


def cache_ratios_from_metrics_text(text: str) -> dict[str, dict[str, float | None]]:
    """Compute hit ratios per named cache from the real exported counter."""
    from prometheus_client.parser import text_string_to_metric_families

    counts: dict[str, dict[str, float]] = {}
    for family in text_string_to_metric_families(text):
        # The client library normalizes a counter family's name by stripping
        # its "_total" suffix; the individual samples keep the full name.
        if family.name not in {"rag_cache_operations", "rag_cache_operations_total"}:
            continue
        for sample in family.samples:
            if sample.name != "rag_cache_operations_total":
                continue
            cache = sample.labels.get("cache", "other")
            outcome = sample.labels.get("result", "other")
            bucket = counts.setdefault(cache, {})
            bucket[outcome] = bucket.get(outcome, 0.0) + sample.value

    ratios: dict[str, dict[str, float | None]] = {}
    for cache, outcomes in counts.items():
        hits = outcomes.get("hit", 0.0)
        misses = outcomes.get("miss", 0.0)
        denominator = hits + misses
        ratios[cache] = {
            "hit_ratio": (hits / denominator) if denominator > 0 else None,
            "hits": hits,
            "misses": misses,
            "disabled": outcomes.get("disabled", 0.0),
        }
    return ratios


# ---------------------------------------------------------------------------
# Report schema validation -- the anti-fabrication guard
# ---------------------------------------------------------------------------


def _assert_null_when_unmeasured(section_name: str, section: Mapping[str, Any]) -> None:
    if section.get("measured") is not False:
        return
    allowed = _ALWAYS_ALLOWED_NON_NULL.get(section_name, {"measured", "reason"})
    for key, value in section.items():
        if key in allowed:
            continue
        if value not in (None, [], {}):
            raise ValueError(
                f"{section_name}.{key} must be null when {section_name}.measured is False "
                f"(got {value!r}); do not report an unmeasured value"
            )


def validate_report(report: Mapping[str, Any]) -> None:
    required_top_level = REQUIRED_REPORT_SECTIONS | {
        "schema_version",
        "status",
        "generated_at",
        "git_sha",
        "config_hash",
        "config",
        "provenance",
    }
    missing = required_top_level - report.keys()
    if missing:
        raise ValueError(f"benchmark report missing sections: {', '.join(sorted(missing))}")

    if report["status"] not in _VALID_STATUSES:
        raise ValueError(f"benchmark report status must be one of {sorted(_VALID_STATUSES)}")

    corpus = report["corpus"]
    for key in ("documents_requested", "documents_available", "sufficient"):
        if key not in corpus:
            raise ValueError(f"corpus section missing '{key}'")

    for section_name in ("quality", "latency", "capacity", "failures"):
        section = report[section_name]
        if "measured" not in section:
            raise ValueError(f"{section_name} section missing 'measured' flag")
        if not isinstance(section["measured"], bool):
            raise ValueError(f"{section_name}.measured must be a boolean")
        _assert_null_when_unmeasured(section_name, section)

    for section_name, field in _DEFERRED_NULL_ONLY_FIELDS:
        section = report[section_name]
        if field not in section:
            raise ValueError(f"{section_name} section missing deferred field '{field}'")
        if section[field] is not None:
            raise ValueError(
                f"{section_name}.{field} is deferred pending a design decision and must "
                f"stay null, not a computed value"
            )

    if report["status"] == "unexecuted":
        for section_name in REQUIRED_REPORT_SECTIONS - {"corpus"}:
            if report[section_name]["measured"] is not False:
                raise ValueError("status='unexecuted' requires every section's measured=False")
        if not report["provenance"].get("prerequisites"):
            raise ValueError("status='unexecuted' requires provenance.prerequisites")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _empty_quality(reason: str) -> dict[str, Any]:
    return {
        "measured": False,
        "rows_evaluated": None,
        "aggregate": None,
        "distractor": None,
        "abstention_precision": None,
        "abstention_recall": None,
        "tool_iteration_metrics": None,
        "reason": reason,
    }


def _empty_latency(reason: str) -> dict[str, Any]:
    return {
        "measured": False,
        "tenant_filter_ms": None,
        "stages": None,
        "source": None,
        "reason": reason,
    }


def _empty_capacity(reason: str) -> dict[str, Any]:
    return {
        "measured": False,
        "total_chunks": None,
        "indexing_lag_seconds": None,
        "vector_memory": None,
        "queue_saturation": None,
        "cache_ratios": None,
        "cost_usd": None,
        "cost_per_document_usd": None,
        "cost_per_question_usd": None,
        "cached_input_token_ratio": None,
        "reason": reason,
    }


def _empty_failures(*, requested: bool, reason: str) -> dict[str, Any]:
    return {
        "measured": False,
        "requested": requested,
        "kinds": list(_FAILURE_KINDS),
        "results": [],
        "reason": reason,
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    totals: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            totals.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in totals.items()}


def _run_quality_phase(
    query_target: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    golden_rows: Sequence[Mapping[str, Any]],
    concurrency: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the golden dataset through a live target; time every call.

    Returns (quality_section, tenant_filter_latency_ms).
    """
    per_row_scores: list[dict[str, float]] = []
    distractor_scores: list[dict[str, float]] = []
    abstention_inputs: list[dict[str, float]] = []
    latencies_ms: list[float] = []

    def _evaluate_row(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, float], float]:
        start = time.perf_counter()
        raw_output = query_target(dict(row["inputs"]))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        output = output_from_mapping(raw_output)
        reference = reference_from_mapping(row["reference"])
        return row, evaluate_output(output, reference), elapsed_ms, abstention_metrics(
            output, reference
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(_evaluate_row, row) for row in golden_rows]
        for future in as_completed(futures):
            row, scores, elapsed_ms, abstention = future.result()
            per_row_scores.append(scores)
            latencies_ms.append(elapsed_ms)
            abstention_inputs.append(abstention)
            if row.get("metadata", {}).get("category") == "distractor":
                distractor_scores.append(scores)

    summary = abstention_summary_metrics(abstention_inputs) if abstention_inputs else {}
    quality = {
        "measured": True,
        "rows_evaluated": len(per_row_scores),
        "aggregate": _mean_metrics(per_row_scores),
        "distractor": _mean_metrics(distractor_scores),
        "abstention_precision": summary.get("abstention_precision"),
        "abstention_recall": summary.get("abstention_recall"),
        "tool_iteration_metrics": None,
        "reason": None,
    }
    tenant_filter_ms = {
        "p50": _percentile(latencies_ms, 50) if latencies_ms else None,
        "p95": _percentile(latencies_ms, 95) if latencies_ms else None,
        "p99": _percentile(latencies_ms, 99) if latencies_ms else None,
        "n": len(latencies_ms),
    }
    return quality, tenant_filter_ms


def _build_capacity_section(
    ingest_results: list[dict[str, Any]],
    *,
    cache_ratios: Mapping[str, Any] | None,
    vector_memory: Mapping[str, Any] | None,
    queue_saturation: Mapping[str, Any] | None,
    cost_usd: float | None,
) -> dict[str, Any]:
    ready_elapsed = [r["elapsed_s"] for r in ingest_results if r.get("status") == "READY"]
    chunk_counts = [r["chunk_count"] for r in ingest_results if r.get("chunk_count") is not None]
    return {
        "measured": True,
        "total_chunks": sum(chunk_counts) if chunk_counts else None,
        "indexing_lag_seconds": (
            {
                "p50_s": _percentile(ready_elapsed, 50),
                "p95_s": _percentile(ready_elapsed, 95),
                "p99_s": _percentile(ready_elapsed, 99),
            }
            if ready_elapsed
            else None
        ),
        "vector_memory": dict(vector_memory) if vector_memory else None,
        "queue_saturation": dict(queue_saturation) if queue_saturation else None,
        "cache_ratios": dict(cache_ratios) if cache_ratios else None,
        "cost_usd": cost_usd,
        "cost_per_document_usd": None,
        "cost_per_question_usd": None,
        "cached_input_token_ratio": None,
        "reason": None,
    }


def _run_failures(
    *, requested: bool, injector: Callable[[str], Mapping[str, Any]] | None
) -> dict[str, Any]:
    if not requested:
        return _empty_failures(
            requested=False, reason="failure injection not requested (pass --failure-injection)"
        )
    if injector is None:
        return _empty_failures(
            requested=True, reason="no --failure-injector-hook configured"
        )
    results = []
    for kind in _FAILURE_KINDS:
        outcome = dict(injector(kind))
        outcome.setdefault("kind", kind)
        results.append(outcome)
    return {
        "measured": True,
        "requested": True,
        "kinds": list(_FAILURE_KINDS),
        "results": results,
        "reason": None,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_benchmark(
    args: argparse.Namespace,
    *,
    manifest_loader: Callable[[Path], list[dict[str, Any]]] = load_corpus_manifest,
    golden_rows_loader: Callable[[Path], list[dict[str, Any]]] = load_golden_dataset,
    git_sha_fn: Callable[[], str] = current_git_sha,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ingest_fn: Callable[..., list[dict[str, Any]]] | None = None,
    query_target: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    metrics_fetcher: Callable[[], str] | None = None,
    qdrant_stats_fetcher: Callable[[], Mapping[str, Any]] | None = None,
    queue_inspector: Callable[[], Mapping[str, Any]] | None = None,
    cost_fetcher: Callable[[], float] | None = None,
    failure_injector: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    corpus_entries = manifest_loader(Path(args.corpus_manifest))
    corpus = _corpus_summary(Path(args.corpus_manifest), corpus_entries, args.documents)

    prerequisites: list[str] = []
    notes: list[str] = []
    if not corpus["sufficient"]:
        message = (
            f"corpus manifest at {args.corpus_manifest} has {corpus['documents_available']} "
            f"documents; {args.documents} are required for this run"
        )
        if args.allow_partial_corpus:
            notes.append(
                f"scale requirement not met: ran against {corpus['documents_available']} of "
                f"{args.documents} requested documents (--allow-partial-corpus)"
            )
        else:
            prerequisites.append(message)
    if ingest_fn is None:
        prerequisites.append(
            "no ingestion target configured (pass --conversation-id with --auth-token or "
            "--email/--password against a reachable --base-url)"
        )
    if query_target is None:
        prerequisites.append(
            "no query target configured (pass --target MODULE:FUNCTION or set "
            "RAG_EVALUATION_TARGET_URL/RAG_EVALUATION_BEARER_TOKEN)"
        )
    if metrics_fetcher is None:
        prerequisites.append(
            "no metrics source configured (pass a reachable --metrics-url for a live "
            "/metrics/rag endpoint)"
        )

    corpus_ready = corpus["sufficient"] or args.allow_partial_corpus
    can_execute = corpus_ready and ingest_fn is not None and query_target is not None

    if not can_execute:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "unexecuted",
            "generated_at": now_fn().isoformat(),
            "git_sha": git_sha_fn(),
            "config_hash": compute_config_hash(args),
            "config": redacted_config(args),
            "corpus": corpus,
            "quality": _empty_quality("prerequisites not met; see provenance.prerequisites"),
            "latency": _empty_latency("prerequisites not met; see provenance.prerequisites"),
            "capacity": _empty_capacity("prerequisites not met; see provenance.prerequisites"),
            "failures": _empty_failures(
                requested=args.failure_injection,
                reason="prerequisites not met; see provenance.prerequisites",
            ),
            "provenance": {"prerequisites": prerequisites, "notes": notes},
        }
        return report

    golden_rows = golden_rows_loader(Path(args.golden_dataset))
    quality, tenant_filter_ms = _run_quality_phase(
        query_target, golden_rows, args.query_concurrency
    )

    ingest_results = ingest_fn(corpus_entries, concurrency=args.ingest_concurrency)

    stages: dict[str, Any] = {}
    cache_ratios: dict[str, Any] | None = None
    if metrics_fetcher is not None:
        metrics_text = metrics_fetcher()
        stages = stage_latency_from_metrics_text(metrics_text)
        cache_ratios = cache_ratios_from_metrics_text(metrics_text)
    latency = {
        "measured": True,
        "tenant_filter_ms": tenant_filter_ms,
        "stages": stages,
        "source": args.metrics_url or f"{args.base_url.rstrip('/')}/metrics/rag",
        "reason": None,
    }

    vector_memory = qdrant_stats_fetcher() if qdrant_stats_fetcher is not None else None
    queue_saturation = queue_inspector() if queue_inspector is not None else None
    cost_usd = cost_fetcher() if cost_fetcher is not None else None
    capacity = _build_capacity_section(
        ingest_results,
        cache_ratios=cache_ratios,
        vector_memory=vector_memory,
        queue_saturation=queue_saturation,
        cost_usd=cost_usd,
    )

    failures = _run_failures(requested=args.failure_injection, injector=failure_injector)

    status = "executed" if corpus["sufficient"] else "partial"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": now_fn().isoformat(),
        "git_sha": git_sha_fn(),
        "config_hash": compute_config_hash(args),
        "config": redacted_config(args),
        "corpus": corpus,
        "quality": quality,
        "latency": latency,
        "capacity": capacity,
        "failures": failures,
        "provenance": {"prerequisites": prerequisites, "notes": notes},
    }


def write_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# main() -- wires real dependencies from CLI flags/environment
# ---------------------------------------------------------------------------


def _build_ingest_fn(args: argparse.Namespace) -> Callable[..., list[dict[str, Any]]] | None:
    if not args.conversation_id:
        return None
    try:
        import requests
    except ImportError:
        return None

    auth_token = args.auth_token
    if not auth_token and args.email and args.password:
        try:
            auth_token = _login(requests, args.base_url.rstrip("/"), args.email, args.password)
        except Exception:
            return None
    if not auth_token:
        return None

    def ingest(
        corpus_entries: Sequence[Mapping[str, Any]], *, concurrency: int
    ) -> list[dict[str, Any]]:
        auth_headers = _get_auth_headers(auth_token)
        base_url = args.base_url.rstrip("/")
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = {
                executor.submit(
                    _process_document,
                    requests,
                    base_url,
                    args.conversation_id,
                    ROOT / entry["path"],
                    args.timeout,
                    auth_headers,
                ): entry
                for entry in corpus_entries
            }
            for future in as_completed(futures):
                record = future.result()
                record.setdefault("chunk_count", None)
                results.append(record)
        return results

    return ingest


def _build_query_target(
    args: argparse.Namespace,
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]] | None:
    if args.target:
        try:
            return load_local_target(args.target)
        except Exception:
            return None
    try:
        return build_http_target()
    except RuntimeError:
        return None


def _build_metrics_fetcher(args: argparse.Namespace) -> Callable[[], str] | None:
    metrics_url = args.metrics_url or f"{args.base_url.rstrip('/')}/metrics/rag"

    def fetch() -> str:
        import requests

        response = requests.get(metrics_url, timeout=10)
        response.raise_for_status()
        return response.text

    try:
        fetch()
    except Exception:
        return None
    return fetch


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_benchmark(
            args,
            ingest_fn=_build_ingest_fn(args),
            query_target=_build_query_target(args),
            metrics_fetcher=_build_metrics_fetcher(args),
            qdrant_stats_fetcher=_resolve_callable(args.qdrant_stats_hook),
            queue_inspector=_resolve_callable(args.queue_inspector_hook),
            cost_fetcher=_resolve_callable(args.cost_fetcher_hook),
            failure_injector=_resolve_callable(args.failure_injector_hook),
        )
    except Exception as error:
        print(f"benchmark run failed: {error}", file=sys.stderr)
        return 1

    validate_report(report)
    write_report(report, args.output)
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))

    if report["status"] == "unexecuted":
        for prerequisite in report["provenance"]["prerequisites"]:
            print(f"prerequisite not met: {prerequisite}", file=sys.stderr)
        return 3
    if report["status"] == "partial":
        for note in report["provenance"]["notes"]:
            print(f"partial run: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
