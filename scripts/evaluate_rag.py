"""Run the versioned RAG evaluation dataset locally or through LangSmith."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.evaluation.rag.corpus import (
    load_corpus_manifest,
    load_golden_dataset,
    prepare_evaluation_scope_from_environment,
    validate_golden_dataset,
    validate_reference_rows,
)
from app.evaluation.rag.langsmith_queries import comparison_metrics
from app.evaluation.rag.metrics import (
    deterministic_evaluators,
    deterministic_summary_evaluators,
    output_from_mapping,
)
from app.evaluation.rag.ragas_metrics import ragas_evaluators
from app.evaluation.rag.release_gates import compare_release_gates, load_release_gates
from app.evaluation.rag.target import build_http_target, load_local_target, scoped_target

DEFAULT_DATASET = "rag-golden-v1"
DEFAULT_DATASET_TAG = "v1"
DEFAULT_EXPERIMENT_PREFIX = "pre-hardening-baseline"
ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG)
    parser.add_argument("--experiment-prefix", default=DEFAULT_EXPERIMENT_PREFIX)
    parser.add_argument(
        "--offline", action="store_true", help="Validate the local immutable corpus only"
    )
    parser.add_argument(
        "--with-ragas", action="store_true", help="Enable optional lazy RAGAS evaluators"
    )
    parser.add_argument("--compare-baseline", metavar="EXPERIMENT")
    parser.add_argument("--ragas-factory", metavar="MODULE:CALLABLE")
    parser.add_argument("--target", metavar="MODULE:FUNCTION")
    return parser.parse_args(argv)


def offline_summary(
    dataset: str = DEFAULT_DATASET, dataset_tag: str = DEFAULT_DATASET_TAG
) -> dict[str, Any]:
    if (dataset, dataset_tag) != (DEFAULT_DATASET, DEFAULT_DATASET_TAG):
        raise ValueError("only the local rag-golden-v1 dataset at tag v1 is available offline")
    rows = load_golden_dataset(ROOT / "eval" / "rag" / "golden_v1.jsonl")
    validate_golden_dataset(
        rows, load_corpus_manifest(ROOT / "eval" / "rag" / "corpus_manifest.jsonl")
    )
    return {
        "mode": "offline",
        "dataset": dataset,
        "dataset_tag": dataset_tag,
        "rows_validated": len(rows),
        "network_calls": 0,
    }


def _examples_to_rows(examples: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(getattr(example, "id", "<remote>")),
            "inputs": dict(getattr(example, "inputs", {}) or {}),
            "reference": dict(getattr(example, "outputs", {}) or {}),
        }
        for example in examples
    ]


def _ragas_dependencies(factory_path: str | None) -> dict[str, Any]:
    if not factory_path:
        raise ValueError("--with-ragas requires --ragas-factory MODULE:CALLABLE")
    module_name, separator, callable_name = factory_path.partition(":")
    if not separator or not module_name or not callable_name:
        raise ValueError("--ragas-factory must be MODULE:CALLABLE")
    dependencies = getattr(importlib.import_module(module_name), callable_name)()
    if not isinstance(dependencies, dict):
        raise ValueError("RAGAS dependency factory must return a dict")
    return dependencies


def run_offline(args: argparse.Namespace, *, target: Any | None = None) -> dict[str, Any]:
    """Execute every local example through deterministic evaluators without upload."""
    summary = offline_summary(args.dataset, args.dataset_tag)
    rows = load_golden_dataset(ROOT / "eval" / "rag" / "golden_v1.jsonl")
    target = target or load_local_target(getattr(args, "target", None))
    dependencies = (
        _ragas_dependencies(getattr(args, "ragas_factory", None)) if args.with_ragas else {}
    )
    evaluators = deterministic_evaluators() + ragas_evaluators(args.with_ragas, **dependencies)
    run_scores: list[dict[str, float]] = []
    runs: list[Any] = []
    for row in rows:
        result = target(dict(row["inputs"]))
        output_from_mapping(result)

        class Run:
            outputs = result

        runs.append(Run())

        class Example:
            inputs = row["inputs"]
            outputs = row["reference"]

        for evaluator in evaluators:
            evaluation = evaluator(runs[-1], Example())
            if "results" in evaluation:
                run_scores.append(
                    {item["key"]: float(item["score"]) for item in evaluation["results"]}
                )
            else:
                run_scores.append({evaluation["key"]: float(evaluation["score"])})
    summary_evaluator = deterministic_summary_evaluators()[0]
    summary_result = summary_evaluator(
        runs,
        [type("Example", (), {"outputs": row["reference"]})() for row in rows],
    )
    totals: dict[str, list[float]] = {}
    for scores in run_scores:
        for key, value in scores.items():
            totals.setdefault(key, []).append(value)
    summary["metrics"] = {key: sum(values) / len(values) for key, values in totals.items()}
    summary["metrics"].update(
        {item["key"]: float(item["score"]) for item in summary_result["results"]}
    )
    summary["evaluations_executed"] = len(run_scores)
    return summary


def run_online(
    args: argparse.Namespace,
    *,
    client_factory: Any | None = None,
    target_factory: Any = build_http_target,
    prepare_scope: Any = prepare_evaluation_scope_from_environment,
) -> tuple[Any, Any]:
    if client_factory is None:
        from langsmith import Client

        client_factory = Client
    client = client_factory()
    dependencies = (
        _ragas_dependencies(getattr(args, "ragas_factory", None)) if args.with_ragas else {}
    )
    evaluators = deterministic_evaluators() + ragas_evaluators(args.with_ragas, **dependencies)
    examples = list(client.list_examples(dataset_name=args.dataset, as_of=args.dataset_tag))
    validate_reference_rows(
        _examples_to_rows(examples),
        load_corpus_manifest(ROOT / "eval" / "rag" / "corpus_manifest.jsonl"),
    )
    target = scoped_target(target_factory(), prepare_scope())
    local_rows = load_golden_dataset(ROOT / "eval" / "rag" / "golden_v1.jsonl")
    pending_review = any(
        row["metadata"].get("label_review_status") == "pending_human_review" for row in local_rows
    )
    results = client.evaluate(
        target,
        data=examples,
        evaluators=evaluators,
        summary_evaluators=deterministic_summary_evaluators(),
        experiment_prefix=args.experiment_prefix,
        metadata={
            "evaluation_label_review": "pending" if pending_review else "reviewed",
            "binding": not pending_review,
        },
        upload_results=not args.offline,
    )
    return client, results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offline:
        try:
            print(json.dumps(run_offline(args), sort_keys=True))
            return 0
        except Exception as error:
            print(f"offline evaluation failed: {error}", file=sys.stderr)
            return 1
    if not os.getenv("LANGSMITH_API_KEY"):
        print(
            "LangSmith is not configured; run with --offline for local validation.", file=sys.stderr
        )
        return 1
    try:
        client, results = run_online(args)
    except Exception as error:
        message = str(error)
        print(f"online evaluation was not recorded: {message}", file=sys.stderr)
        return 1
    if args.compare_baseline:
        local_rows = load_golden_dataset(ROOT / "eval" / "rag" / "golden_v1.jsonl")
        if any(
            row["metadata"].get("label_review_status") == "pending_human_review"
            for row in local_rows
        ):
            print(
                "release-gate comparison blocked: dataset labels are pending human review",
                file=sys.stderr,
            )
            return 1
        try:
            gates = load_release_gates(ROOT / "eval" / "rag" / "release_gates.json")
            candidate_name = getattr(results, "experiment_name", None)
            if not candidate_name:
                raise ValueError("LangSmith did not return the candidate experiment name")
            candidate_metrics, baseline_metrics = asyncio.run(
                comparison_metrics(client, candidate_name, args.compare_baseline)
            )
            verdicts = compare_release_gates(baseline_metrics, candidate_metrics, gates)
        except Exception as error:
            print(f"release-gate comparison failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps([verdict.__dict__ for verdict in verdicts], sort_keys=True))
        # Unmeasured gates (status="unmeasured" in release_gates.json) have no
        # evidence-based threshold yet and carry binding=False; they must never
        # count as a pass or a failure of the release decision.
        if not all(verdict.passed for verdict in verdicts if verdict.binding):
            return 2
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
