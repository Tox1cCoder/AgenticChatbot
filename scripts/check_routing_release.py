"""Decide whether a routing evaluation report may release. Exit 1 if not.

Deterministic and offline: it re-derives the decision from a stored report, so
a disputed release is re-checkable without spending another provider run, and
CI can gate on it without credentials.

The expected provider/model/inventory default to **this machine's
configuration**, not to whatever the report claims. That is the point of the
tuple check: it asks "does this report describe the system about to ship?" A
flag the operator sets to match the report would answer "yes" by construction,
so the overrides exist only for checking a report against a deployment target
other than the local one.

Usage::

    .\\.venv\\Scripts\\python.exe scripts/check_routing_release.py \\
        --report eval/routing/report.json

Exit codes: ``0`` released, ``1`` blocked, ``2`` the inputs could not be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings  # noqa: E402
from app.evaluation.routing.contracts import (  # noqa: E402
    RoutingEvaluationReport,
    dataset_sha256,
    load_dataset,
    load_review,
)
from app.evaluation.routing.harness import build_evaluation_inventory  # noqa: E402
from app.evaluation.routing.release_gate import check_routing_release  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dataset", default="eval/routing/golden_v1.jsonl")
    parser.add_argument("--review", default="eval/routing/golden_v1.review.json")
    parser.add_argument("--expect-provider", default=None)
    parser.add_argument("--expect-model", default=None)
    parser.add_argument("--expect-inventory-version", default=None)
    parser.add_argument("--json", action="store_true", help="print the decision as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        report = RoutingEvaluationReport.model_validate_json(
            Path(args.report).read_text(encoding="utf-8")
        )
        cases = load_dataset(args.dataset)
    except Exception as exc:  # noqa: BLE001 - unreadable inputs are an operator error
        print(f"could not read the report or dataset: {exc}", file=sys.stderr)
        return 2

    # A missing manifest is a *decision input*, not a crash: "nobody has
    # reviewed this" is exactly what `missing_review` says.
    review_path = Path(args.review)
    review = load_review(review_path) if review_path.exists() else None

    decision = check_routing_release(
        report=report,
        review=review,
        cases=cases,
        expected_provider=args.expect_provider or settings.router_provider,
        expected_model=args.expect_model or settings.router_model,
        expected_inventory_version=(
            args.expect_inventory_version or build_evaluation_inventory().version
        ),
    )

    if args.json:
        print(json.dumps(json.loads(decision.model_dump_json()), indent=2, sort_keys=True))
    else:
        print("RELEASE: PASS" if decision.passed else "RELEASE: BLOCKED")
        for code in decision.reason_codes:
            print(f"  - {code}")
        if review is None:
            print(f"  (no review manifest at {review_path})")
        elif not review.approved:
            print(
                "  a reviewer other than the dataset author must verify the labels and set "
                f'"approved": true in {review_path}'
            )
        if decision.passed:
            digest = dataset_sha256(args.dataset)[:16]
            print(f"  dataset {digest}... reviewed and within thresholds")

    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
