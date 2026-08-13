"""Run the versioned RAG evaluation dataset locally or through LangSmith."""

from __future__ import annotations

import argparse
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
)
from app.evaluation.rag.metrics import deterministic_evaluators, deterministic_summary_evaluators
from app.evaluation.rag.ragas_metrics import ragas_evaluators
from app.evaluation.rag.release_gates import compare_release_gates, load_release_gates
from app.evaluation.rag.target import build_http_target, scoped_target

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
    target = scoped_target(target_factory(), prepare_scope())
    evaluators = deterministic_evaluators() + ragas_evaluators(args.with_ragas)
    results = client.evaluate(
        target,
        data=client.list_examples(dataset_name=args.dataset, as_of=args.dataset_tag),
        evaluators=evaluators,
        summary_evaluators=deterministic_summary_evaluators(),
        experiment_prefix=args.experiment_prefix,
        upload_results=not args.offline,
    )
    return client, results


def experiment_metrics(client: Any, experiment_name: str) -> dict[str, float]:
    """Aggregate recorded deterministic feedback for an immutable experiment."""
    frame = client.get_test_results(project_name=experiment_name)
    metrics = {
        column.removeprefix("feedback."): float(frame[column].mean())
        for column in frame.columns
        if column.startswith("feedback.")
    }
    if not metrics:
        raise ValueError(f"baseline experiment has no deterministic feedback: {experiment_name}")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offline:
        print(json.dumps(offline_summary(args.dataset, args.dataset_tag), sort_keys=True))
        return 0
    if not os.getenv("LANGSMITH_API_KEY"):
        print(
            "LangSmith is not configured; run with --offline for local validation.", file=sys.stderr
        )
        return 0
    try:
        client, results = run_online(args)
    except Exception as error:
        # External quota/network failures must not invalidate offline evidence.
        message = str(error)
        if "429" in message or "quota" in message.lower():
            print(f"LangSmith recording unavailable (quota): {message}", file=sys.stderr)
            return 0
        raise
    if args.compare_baseline:
        gates = load_release_gates(ROOT / "eval" / "rag" / "release_gates.json")
        candidate_name = getattr(results, "experiment_name", None)
        if not candidate_name:
            raise ValueError("LangSmith did not return the candidate experiment name")
        candidate_metrics = experiment_metrics(client=client, experiment_name=candidate_name)
        baseline_metrics = experiment_metrics(client=client, experiment_name=args.compare_baseline)
        verdicts = compare_release_gates(baseline_metrics, candidate_metrics, gates)
        print(json.dumps([verdict.__dict__ for verdict in verdicts], sort_keys=True))
        if not all(verdict.passed for verdict in verdicts):
            return 2
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
