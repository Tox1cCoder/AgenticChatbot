"""SmithDB-backed LangSmith reads for RAG experiment comparison.

LangSmith marks the v1 run-query and run-retrieve endpoint families with
``Deprecation: true`` and a 2027-01-31 sunset. ``Client.get_test_results()``
reaches them through ``Client.list_runs()`` / ``POST /api/v1/runs/query``, so
experiment feedback is aggregated here through the SmithDB v2 resource
(``Client.runs.query()`` -> ``POST /api/v2/runs/query``) instead.

Two properties of the v2 resource shape this module:

* ``runs.query()`` returns only the last 24 hours unless ``min_start_time`` is
  supplied, so every query passes the experiment project's own ``start_time``.
  Omitting it silently truncates an older experiment to zero runs, which would
  read as "no feedback" rather than as an error -- and because ``start_time``
  is optional on the project schema, a missing one is refused here rather than
  passed through into that same silent truncation.
* ``selects`` is an explicit allowlist. Only ``ID`` and ``FEEDBACK_STATS`` are
  requested; run inputs and outputs stay on the server, so no prompt, document
  body, or user/conversation identifier is transferred to compute a metric.

Project- and session-level statistics come from ``aread_project`` and override
the per-run averages, because summary evaluators record their score once on the
experiment rather than on each root run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

ROOT_RUN_SELECTS = ["ID", "FEEDBACK_STATS"]

#: Largest page the v2 endpoint accepts (1-1000; it returns 100 when omitted).
#: The async iterator pages on its own, so this only decides how many round
#: trips an experiment costs -- the 210-case golden dataset is three at the
#: default and one here.
ROOT_RUN_PAGE_SIZE = 1_000


def _average(statistics: Any) -> float | None:
    """Read one entry's mean, whichever shape the SDK handed back.

    ``Client.runs.query()`` returns SmithDB ``Run`` models whose
    ``feedback_stats`` values are ``FeedbackStats`` instances, while
    ``aread_project`` returns project statistics as plain dictionaries. Reading
    only mappings drops every per-root-run metric silently: the comparison then
    reports whatever the project summary happens to hold, or fails closed as
    though the experiment recorded no feedback at all.
    """
    value = statistics.get("avg") if isinstance(statistics, Mapping) else getattr(
        statistics, "avg", None
    )
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _averages(source: Any) -> dict[str, float]:
    """Return only the feedback entries that carry a numeric average."""
    entries = source if isinstance(source, Mapping) else getattr(source, "__dict__", None)
    if not isinstance(entries, Mapping):
        return {}
    averages: dict[str, float] = {}
    for key, statistics in entries.items():
        average = _average(statistics)
        if average is not None:
            averages[str(key)] = average
    return averages


async def experiment_metrics(client: Any, experiment_name: str) -> dict[str, float]:
    """Aggregate recorded deterministic feedback for an immutable experiment."""
    project = await client.aread_project(
        project_name=experiment_name,
        include_stats=True,
    )
    started = getattr(project, "start_time", None)
    if started is None:
        # Refused rather than defaulted. start_time is optional on the project
        # schema, and an absent min_start_time silently means "the last 24
        # hours": an older experiment would come back empty and be reported as
        # having no deterministic feedback instead of as a truncated read.
        raise ValueError(
            f"experiment has no start_time, so its runs cannot be bounded: {experiment_name}"
        )

    totals: dict[str, list[float]] = {}
    async for run in client.runs.query(
        project_ids=[str(project.id)],
        is_root=True,
        min_start_time=started,
        selects=list(ROOT_RUN_SELECTS),
        page_size=ROOT_RUN_PAGE_SIZE,
    ):
        for key, average in _averages(getattr(run, "feedback_stats", None)).items():
            totals.setdefault(key, []).append(average)

    metrics = {key: sum(values) / len(values) for key, values in totals.items()}
    metrics.update(_averages(getattr(project, "feedback_stats", None)))
    metrics.update(_averages(getattr(project, "session_feedback_stats", None)))
    if not metrics:
        raise ValueError(f"baseline experiment has no deterministic feedback: {experiment_name}")
    return metrics


async def comparison_metrics(
    client: Any,
    candidate_name: str,
    baseline_name: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Read both sides of a release-gate comparison concurrently."""
    candidate, baseline = await asyncio.gather(
        experiment_metrics(client, candidate_name),
        experiment_metrics(client, baseline_name),
    )
    return candidate, baseline
