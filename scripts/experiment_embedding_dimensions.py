"""Compare embedding dimensions and provider indexing modes on real evidence.

Task 13, Step 4 of the RAG production-hardening plan asks: does raising the
Gemini embedding dimension from 768 to 1536 to 3072 buy enough retrieval
quality to justify its extra Qdrant vector-storage cost, and does Gemini's
asynchronous embedding Batch API beat synchronous request batches for
non-interactive (re)indexing? Answering either question needs real
embedding-provider calls this repository has no budget or credentials for
in CI, so this script never estimates, extrapolates, or hand-writes a
result -- it is the harness that produces a trustworthy answer once someone
runs it with real infrastructure.

For each requested dimension it calls an operator-supplied
``--experiment-hook MODULE:FUNCTION`` once per indexing mode. The hook owns
every deployment-specific concern this repository cannot verify here: which
Gemini API calls to make, how to stand up a same-dimension Qdrant
collection, how to submit and poll a Batch job. Its contract::

    def hook(dimension: int, mode: str) -> dict:
        '''mode is "synchronous" or "provider_batch".'''
        return {
            "documents_indexed": int,
            "wall_time_s": float,
            "cost_usd": float,
            "failures": int,
            "operational_notes": str,
            # Only meaningful once per dimension (indexing mode does not
            # change embedding content) -- populate on the "synchronous"
            # call, leave None on "provider_batch":
            "quality": dict[str, float] | None,
        }

``quality`` should carry the same deterministic metric keys
``scripts/evaluate_rag.py`` reports (``app.evaluation.rag.metrics``), e.g.
``document_recall_at_5``, ``citation_validity``, so dimensions can be
compared on identical terms against the unchanged golden dataset
(``eval/rag/golden_v1.jsonl``).

Without ``--experiment-hook``, the script still writes a schema-valid
report with ``status: "unexecuted"`` and the missing prerequisite listed,
rather than silently doing nothing or fabricating numbers.

Any dimension recommendation is computed, never hard-coded: pass
``--recall-parity-tolerance`` (an operator-chosen number, not a value this
script invents) to have it pick the smallest dimension whose measured
``document_recall_at_5`` is within that tolerance of the largest
dimension's. Omit it and ``comparison.recommendation`` stays null -- the
plan's Global Constraints forbid selecting a threshold that was not chosen
from evaluation results.

Usage::

    python scripts/experiment_embedding_dimensions.py \\
        --dimensions 768 1536 3072 \\
        --dataset rag-golden-v1 \\
        --include-provider-batch \\
        --experiment-hook my_ops_module:run_dimension_cell \\
        --output artifacts/rag-dimension-matrix.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_MANIFEST = ROOT / "eval" / "rag" / "corpus_manifest.jsonl"
DEFAULT_GOLDEN_DATASET = ROOT / "eval" / "rag" / "golden_v1.jsonl"
DEFAULT_OUTPUT = "artifacts/rag-dimension-matrix.json"
DEFAULT_DIMENSIONS = (768, 1536, 3072)
SCHEMA_VERSION = "1.0"

_MODES = ("synchronous", "provider_batch")
_VALID_STATUSES = {"executed", "partial", "unexecuted"}
_RECALL_METRIC = "document_recall_at_5"

_REQUIRED_TOP_LEVEL = {
    "schema_version",
    "status",
    "generated_at",
    "git_sha",
    "config_hash",
    "config",
    "corpus",
    "dataset",
    "dimensions",
    "comparison",
    "provenance",
}
_CELL_ALWAYS_ALLOWED_NON_NULL = {"measured", "reason"}
_DIMENSION_ALWAYS_ALLOWED_NON_NULL = {"measured", "reason", "indexing"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIMENSIONS),
        help="Embedding dimensions to compare (default: 768 1536 3072).",
    )
    parser.add_argument("--dataset", default="rag-golden-v1")
    parser.add_argument("--dataset-tag", default="v1")
    parser.add_argument("--corpus-manifest", default=str(DEFAULT_CORPUS_MANIFEST))
    parser.add_argument("--golden-dataset", default=str(DEFAULT_GOLDEN_DATASET))
    parser.add_argument(
        "--include-provider-batch",
        action="store_true",
        help="Also run each dimension through Gemini's asynchronous embedding Batch API.",
    )
    parser.add_argument(
        "--baseline-experiment",
        metavar="EXPERIMENT",
        default=None,
        help="Recorded for provenance only; this script does not fetch baseline metrics.",
    )
    parser.add_argument("--experiment-hook", metavar="MODULE:FUNCTION", default=None)
    parser.add_argument(
        "--recall-parity-tolerance",
        type=float,
        default=None,
        help=f"Operator-chosen tolerance on {_RECALL_METRIC} for an automatic "
        "dimension recommendation. Omit to leave the recommendation null.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.output.endswith(".json"):
        parser.error("--output must end with .json")
    return args


def redacted_config(args: argparse.Namespace) -> dict[str, Any]:
    return dict(vars(args))


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
# Report schema validation -- the anti-fabrication guard
# ---------------------------------------------------------------------------


def _assert_cell_null_when_unmeasured(path: str, cell: dict[str, Any]) -> None:
    if cell.get("measured") is not False:
        return
    for key, value in cell.items():
        if key in _CELL_ALWAYS_ALLOWED_NON_NULL:
            continue
        if value not in (None, [], {}):
            raise ValueError(
                f"{path}.{key} must be null when {path}.measured is False "
                f"(got {value!r}); do not report an unmeasured value"
            )


def validate_report(report: dict[str, Any]) -> None:
    missing = _REQUIRED_TOP_LEVEL - report.keys()
    if missing:
        raise ValueError(f"dimension report missing sections: {', '.join(sorted(missing))}")
    if report["status"] not in _VALID_STATUSES:
        raise ValueError(f"dimension report status must be one of {sorted(_VALID_STATUSES)}")

    for dimension_key, entry in report["dimensions"].items():
        if "measured" not in entry:
            raise ValueError(f"dimensions.{dimension_key} missing 'measured' flag")
        if entry["measured"] is False and entry.get("quality") not in (None, {}):
            raise ValueError(
                f"dimensions.{dimension_key}.quality must be null when measured is False"
            )
        indexing = entry.get("indexing", {})
        for mode in _MODES:
            if mode not in indexing:
                raise ValueError(f"dimensions.{dimension_key}.indexing missing '{mode}'")
            _assert_cell_null_when_unmeasured(
                f"dimensions.{dimension_key}.indexing.{mode}", indexing[mode]
            )

    comparison = report["comparison"]
    if "measured" not in comparison:
        raise ValueError("comparison section missing 'measured' flag")
    if comparison["measured"] is False:
        for key, value in comparison.items():
            if key in {"measured", "reason"}:
                continue
            if key == "recommendation":
                populated = value.get("selected_dimension") is not None
                populated = populated or value.get("rationale") is not None
                if populated:
                    raise ValueError(
                        "comparison.recommendation must be null when comparison.measured is False"
                    )
                continue
            if value not in (None, [], {}):
                raise ValueError(
                    f"comparison.{key} must be null when comparison.measured is False "
                    f"(got {value!r})"
                )

    if report["status"] == "unexecuted":
        if any(entry["measured"] is not False for entry in report["dimensions"].values()):
            raise ValueError("status='unexecuted' requires every dimension's measured=False")
        if not report["provenance"].get("prerequisites"):
            raise ValueError("status='unexecuted' requires provenance.prerequisites")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _empty_indexing_cell(reason: str) -> dict[str, Any]:
    return {
        "measured": False,
        "documents_indexed": None,
        "wall_time_s": None,
        "cost_usd": None,
        "failures": None,
        "operational_notes": None,
        "reason": reason,
    }


def _empty_dimension_entry(reason: str) -> dict[str, Any]:
    return {
        "measured": False,
        "quality": None,
        "indexing": {
            "synchronous": _empty_indexing_cell(reason),
            "provider_batch": _empty_indexing_cell(reason),
        },
        "reason": reason,
    }


def _run_one_dimension(
    dimension: int,
    *,
    hook: Callable[[int, str], dict[str, Any]],
    include_provider_batch: bool,
) -> dict[str, Any]:
    try:
        synchronous_cell = dict(hook(dimension, "synchronous"))
    except Exception as error:
        return _empty_dimension_entry(str(error))
    synchronous_cell["measured"] = True
    synchronous_cell.setdefault("reason", None)
    quality = synchronous_cell.pop("quality", None)

    if include_provider_batch:
        try:
            batch_cell = dict(hook(dimension, "provider_batch"))
            batch_cell["measured"] = True
            batch_cell.setdefault("reason", None)
            batch_cell.pop("quality", None)
        except Exception as error:
            batch_cell = _empty_indexing_cell(str(error))
    else:
        batch_cell = _empty_indexing_cell("not requested (pass --include-provider-batch)")

    return {
        "measured": True,
        "quality": quality,
        "indexing": {"synchronous": synchronous_cell, "provider_batch": batch_cell},
        "reason": None,
    }


def _build_comparison(
    dimensions_report: dict[str, dict[str, Any]],
    *,
    include_provider_batch: bool,
    recall_parity_tolerance: float | None,
) -> dict[str, Any]:
    measured_dimensions = {
        int(key): entry for key, entry in dimensions_report.items() if entry["measured"]
    }
    if not measured_dimensions:
        return {
            "measured": False,
            "quality_deltas_vs_largest_dimension": None,
            "cost_deltas_vs_largest_dimension": None,
            "synchronous_vs_provider_batch": None,
            "recommendation": {
                "selected_dimension": None,
                "rationale": None,
                "tolerance_used": None,
            },
            "reason": "no dimension completed successfully",
        }

    largest = max(measured_dimensions)
    largest_quality = measured_dimensions[largest].get("quality") or {}
    quality_deltas: dict[str, dict[str, float]] = {}
    cost_deltas: dict[str, float] = {}
    for dimension, entry in measured_dimensions.items():
        quality = entry.get("quality") or {}
        quality_deltas[str(dimension)] = {
            metric: quality[metric] - largest_quality[metric]
            for metric in quality
            if metric in largest_quality
        }
        sync_cost = entry["indexing"]["synchronous"].get("cost_usd")
        largest_sync_cost = measured_dimensions[largest]["indexing"]["synchronous"].get("cost_usd")
        if sync_cost is not None and largest_sync_cost is not None:
            cost_deltas[str(dimension)] = sync_cost - largest_sync_cost

    batch_comparison = None
    if include_provider_batch:
        batch_comparison = {}
        for dimension, entry in measured_dimensions.items():
            sync_cell = entry["indexing"]["synchronous"]
            batch_cell = entry["indexing"]["provider_batch"]
            if not batch_cell["measured"]:
                continue
            batch_comparison[str(dimension)] = {
                "wall_time_delta_s": batch_cell["wall_time_s"] - sync_cell["wall_time_s"],
                "cost_delta_usd": batch_cell["cost_usd"] - sync_cell["cost_usd"],
                "failures_delta": batch_cell["failures"] - sync_cell["failures"],
            }

    recommendation: dict[str, Any] = {
        "selected_dimension": None,
        "rationale": None,
        "tolerance_used": recall_parity_tolerance,
    }
    if recall_parity_tolerance is not None:
        candidates = [
            dimension
            for dimension, entry in measured_dimensions.items()
            if _RECALL_METRIC in (entry.get("quality") or {})
        ]
        if candidates:
            best_recall = max(measured_dimensions[d]["quality"][_RECALL_METRIC] for d in candidates)
            within_tolerance = [
                dimension
                for dimension in candidates
                if best_recall - measured_dimensions[dimension]["quality"][_RECALL_METRIC]
                <= recall_parity_tolerance
            ]
            selected = min(within_tolerance)
            recommendation["selected_dimension"] = selected
            recommendation["rationale"] = (
                f"{_RECALL_METRIC}={measured_dimensions[selected]['quality'][_RECALL_METRIC]:.4f} "
                f"at dimension {selected} is within {recall_parity_tolerance} of the best "
                f"measured {_RECALL_METRIC}={best_recall:.4f} (dimension {largest}); "
                f"selecting the smallest dimension at parity to minimize Qdrant vector storage."
            )

    return {
        "measured": True,
        "quality_deltas_vs_largest_dimension": quality_deltas,
        "cost_deltas_vs_largest_dimension": cost_deltas or None,
        "synchronous_vs_provider_batch": batch_comparison,
        "recommendation": recommendation,
        "reason": None,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(
    args: argparse.Namespace,
    *,
    experiment_hook: Callable[[int, str], dict[str, Any]] | None = None,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    git_sha_fn: Callable[[], str] = current_git_sha,
) -> dict[str, Any]:
    corpus_manifest_path = Path(args.corpus_manifest)
    corpus = {
        "manifest_path": str(corpus_manifest_path),
        "exists": corpus_manifest_path.exists(),
    }
    dataset = {
        "name": args.dataset,
        "tag": args.dataset_tag,
        "golden_dataset_path": str(args.golden_dataset),
        "baseline_experiment": args.baseline_experiment,
    }

    prerequisites: list[str] = []
    if experiment_hook is None:
        prerequisites.append(
            "no --experiment-hook configured; comparing embedding dimensions and "
            "provider-Batch indexing requires real Gemini embedding-provider calls "
            "this environment has no credentials or budget for"
        )

    dimensions_report: dict[str, dict[str, Any]] = {}
    if experiment_hook is None:
        for dimension in args.dimensions:
            dimensions_report[str(dimension)] = _empty_dimension_entry(
                "prerequisites not met; see provenance.prerequisites"
            )
    else:
        for dimension in args.dimensions:
            dimensions_report[str(dimension)] = _run_one_dimension(
                dimension,
                hook=experiment_hook,
                include_provider_batch=args.include_provider_batch,
            )

    all_measured = all(entry["measured"] for entry in dimensions_report.values())
    any_measured = any(entry["measured"] for entry in dimensions_report.values())

    comparison = _build_comparison(
        dimensions_report,
        include_provider_batch=args.include_provider_batch,
        recall_parity_tolerance=args.recall_parity_tolerance,
    )

    if experiment_hook is None:
        status = "unexecuted"
    elif all_measured:
        status = "executed"
    elif any_measured:
        status = "partial"
    else:
        status = "unexecuted"
        prerequisites.append("every requested dimension failed; see dimensions.*.reason")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": now_fn().isoformat(),
        "git_sha": git_sha_fn(),
        "config_hash": compute_config_hash(args),
        "config": redacted_config(args),
        "corpus": corpus,
        "dataset": dataset,
        "dimensions": dimensions_report,
        "comparison": comparison,
        "provenance": {
            "prerequisites": prerequisites,
            "notes": [
                "quality is measured once per dimension (on the synchronous indexing "
                "call) because indexing mode does not change embedding content.",
                "comparison.recommendation stays null unless --recall-parity-tolerance "
                "is explicitly supplied; this script never hard-codes a quality bar.",
            ],
        },
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        experiment_hook = _resolve_callable(args.experiment_hook)
        report = run_experiment(args, experiment_hook=experiment_hook)
    except Exception as error:
        print(f"dimension experiment failed: {error}", file=sys.stderr)
        return 1

    validate_report(report)
    write_report(report, args.output)
    print(json.dumps({"status": report["status"], "output": args.output}, sort_keys=True))

    if report["status"] == "unexecuted":
        for prerequisite in report["provenance"]["prerequisites"]:
            print(f"prerequisite not met: {prerequisite}", file=sys.stderr)
        return 3
    if report["status"] == "partial":
        for dimension, entry in report["dimensions"].items():
            if not entry["measured"]:
                print(f"dimension {dimension} did not complete: {entry['reason']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
