"""Measure the production router against the golden dataset.

This is the only place in the routing evaluation that talks to a provider.
Everything downstream - metrics, thresholds, the release decision - is
deterministic and re-derivable from the report this writes, so a disputed
release can be re-checked without spending another run.

It calls the real ``RoutingService.route``. Not a copy of its prompt, not a
reimplementation of its schema handling: a harness that approximates the thing
it is grading will pass while production fails. Strict runtime resolution
means there is no provider fallback - a misconfigured run fails loudly instead
of quietly measuring a different model.

Usage::

    $env:LANGSMITH_TRACING='false'
    .\\.venv\\Scripts\\python.exe scripts/evaluate_routing.py \\
        --dataset eval/routing/golden_v1.jsonl \\
        --user-id <uuid of a user with router credentials> \\
        --output eval/routing/report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ai.workflow.contracts import WorkflowRoutingException  # noqa: E402
from app.ai.workflow.routing import (  # noqa: E402
    RoutingContextBuilder,
    RoutingService,
)
from app.core.config import settings  # noqa: E402
from app.core.container import get_container  # noqa: E402
from app.evaluation.routing.contracts import (  # noqa: E402
    RoutingEvalCase,
    RoutingPrediction,
    dataset_sha256,
    load_dataset,
)
from app.evaluation.routing.harness import (  # noqa: E402
    EvaluationDocumentRepository,
    EvaluationHistoryProvider,
    build_context_request,
    build_evaluation_inventory,
)
from app.evaluation.routing.metrics import build_report  # noqa: E402
from app.observability.routing import RoutingMetricsRecorder  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="eval/routing/golden_v1.jsonl")
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        required=True,
        help="where to write the JSON report",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="user whose configured router credentials the run resolves strictly",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="evaluate only the first N cases (smoke runs)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="cases in flight against the provider"
    )
    return parser.parse_args(argv)


async def _predict(
    service: RoutingService,
    builder: RoutingContextBuilder,
    case: RoutingEvalCase,
    inventory,
    *,
    user_id: str,
) -> RoutingPrediction:
    """Route one case and record what the router actually did.

    Attempts and structured success are read from a *per-case* recorder rather
    than inferred. `route` deliberately does not report them to its caller —
    they are observability, not part of the decision — and guessing "it must
    have taken one attempt because it succeeded" would hide exactly the
    degradation the 0.99 first-attempt bar exists to catch.
    """
    recorder = RoutingMetricsRecorder()
    service._metrics = recorder  # noqa: SLF001 - the recorder is the documented seam

    builder.history_provider = EvaluationHistoryProvider(
        str(case.context.get("previous_agent_id") or "") or None
    )
    builder.document_repository = EvaluationDocumentRepository(
        bool(case.context.get("has_documents"))
    )

    request = build_context_request(case, inventory, user_id=user_id)
    context = await builder.build(request)

    predicted: str | None = None
    error_code: str | None = None
    try:
        decision = await service.route(
            context,
            inventory,
            user_id=user_id,
            model_request=None,
            request_id=f"eval-{case.case_id}",
        )
        predicted = decision.agent_id
    except WorkflowRoutingException as exc:
        error_code = str(getattr(getattr(exc, "error", None), "code", "") or "routing_failed")
    except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
        error_code = f"unexpected:{type(exc).__name__}"

    counters = recorder.counters
    attempts = 2 if counters.get("routing.attempts.2") else 1
    structured_success = bool(counters.get("routing.schema.ok"))

    return RoutingPrediction(
        case_id=case.case_id,
        predicted_agent_id=predicted,
        attempts=attempts,
        structured_success=structured_success,
        error_code=error_code,
    )


async def _run(args: argparse.Namespace) -> int:
    cases = load_dataset(args.dataset)
    if args.limit > 0:
        cases = cases[: args.limit]
    inventory = build_evaluation_inventory()

    # The container's RoutingService, not a hand-built one: the model resolver
    # is a database-backed service, and a script that constructed its own
    # would be measuring a router the server does not run.
    service: RoutingService = get_container().routing_service()
    service.validate_static_configuration()

    # One builder per worker: it holds the per-case history/document stubs, so
    # sharing one across concurrent cases would let them read each other's.
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    builders = [
        RoutingContextBuilder(history_provider=None, document_repository=None, settings=settings)
        for _ in range(max(1, args.concurrency))
    ]
    builder_pool: asyncio.Queue = asyncio.Queue()
    for builder in builders:
        builder_pool.put_nowait(builder)

    async def _one(case: RoutingEvalCase) -> RoutingPrediction:
        async with semaphore:
            builder = await builder_pool.get()
            try:
                return await _predict(service, builder, case, inventory, user_id=args.user_id)
            finally:
                builder_pool.put_nowait(builder)

    predictions = list(await asyncio.gather(*(_one(case) for case in cases)))

    provider, model = _resolved_tuple(service, args.user_id)
    report = build_report(
        cases=cases,
        predictions=predictions,
        dataset_sha256=dataset_sha256(args.dataset),
        provider=provider,
        model=model,
        inventory_version=inventory.version,
        generated_at=datetime.now(timezone.utc),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json.loads(report.model_dump_json()), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"cases            {report.case_count}")
    print(f"macro F1         {report.macro_f1:.4f}")
    for language, accuracy in sorted(report.accuracy_by_language.items()):
        print(f"  {language:<6}         {accuracy:.4f}")
    print(f"structured 1st   {report.first_attempt_structured_success:.4f}")
    print(f"structured final {report.after_retry_structured_success:.4f}")
    print(f"chat subs        {report.silent_chat_substitutions}")
    print(f"report           {out}")
    return 0


def _resolved_tuple(service: RoutingService, user_id: str) -> tuple[str, str]:
    """The exact provider/model the run used, not the configured intent.

    A report that records what the settings *asked for* rather than what was
    resolved would pin the release gate to a tuple the run never executed.
    """
    configured = service._resolve_strictly(user_id, None)  # noqa: SLF001
    return configured.provider, configured.model


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
