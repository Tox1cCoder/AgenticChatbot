"""SmithDB v2 read contracts for the LangSmith evaluation queries.

The v1 run-query, run-retrieve, legacy experiment-run, sharing, and annotation
queue endpoint families carry ``Deprecation: true`` with a 2027-01-31 sunset.
These tests pin the replacement call shape and fail closed if any production
module reaches for a legacy query method again.
"""

from __future__ import annotations

import ast
import inspect
import io
import re
import tokenize
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from langsmith import AsyncClient, Client
from langsmith._internal._beta_decorator import deprecated
from langsmith._openapi_client.types.run import Run
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from app.evaluation.rag.langsmith_queries import comparison_metrics, experiment_metrics

ROOT = Path(__file__).resolve().parents[1]
SMITHDB_FLOOR = Version("0.10.15")
#: Raw v1 paths, for anyone who hand-rolls a request instead of using the SDK.
#: Deliberately anchored to ``/api/v1/``: the replacement for the first of these
#: is ``/api/v2/runs/query``, so an unanchored ``/runs/query`` would ban the fix
#: along with the defect.
LEGACY_ENDPOINT_LITERALS = (
    "/api/v1/runs/query",
    "/api/v1/runs/",
    "/api/v1/public/runs",
    "/api/v1/annotation-queues/",
)
#: ``POST /api/v1/datasets/{dataset_id}/runs``, the legacy experiment-run write.
LEGACY_DATASET_RUNS_PATH = re.compile(r"/api/v1/datasets/[^\s\"']*?/runs")
PRODUCTION_ROOTS = ("app", "scripts")
CLIENT_CLASSES = (Client, AsyncClient)
DECLARATION_FILES = ("pyproject.toml", "environment.yml", "requirements.txt")
PRE_SMITHDB_RELEASES = ("0.3.45", "0.8.11", "0.9.8", "0.10.9", "0.10.14")
SMITHDB_RELEASES = ("0.10.15", "0.10.18", "0.11.2")
_SPECIFIER = re.compile(r"langsmith\s*((?:[<>=!~]=?[^,\s\"']+)(?:\s*,\s*[<>=!~]=?[^,\s\"']+)*)")


class FakeRuns:
    """Async runs resource double that records every v2 query it receives."""

    def __init__(self, rows=None, rows_by_project=None):
        self.rows = list(rows or [])
        self.rows_by_project = dict(rows_by_project or {})
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        project_ids = kwargs.get("project_ids") or []
        key = project_ids[0] if project_ids else None
        rows = self.rows_by_project.get(key, self.rows)

        async def iterate():
            for row in rows:
                yield row

        return iterate()


class FakeClient:
    def __init__(self, *, project=None, projects=None, runs=None):
        self._project = project
        self._projects = dict(projects or {})
        self.runs = runs if runs is not None else FakeRuns()
        self.project_calls = []

    async def aread_project(self, **kwargs):
        self.project_calls.append(kwargs)
        name = kwargs.get("project_name")
        if name in self._projects:
            return self._projects[name]
        if self._project is None:
            raise ValueError(f"unknown experiment: {name}")
        return self._project


def _project(*, feedback=None, session_feedback=None, started=None):
    return SimpleNamespace(
        id=uuid4(),
        start_time=started or datetime(2025, 1, 2, tzinfo=timezone.utc),
        feedback_stats=feedback,
        session_feedback_stats=session_feedback,
    )


def _run(**feedback_stats):
    """A root run as the installed SDK actually returns it.

    ``Run`` coerces each entry of ``feedback_stats`` into a ``FeedbackStats``
    model, so an aggregator that only understands mappings reads a real
    experiment as having no feedback at all. Dictionaries here would hide that.
    """
    return Run(feedback_stats={key: {"avg": value} for key, value in feedback_stats.items()})


async def test_experiment_metrics_use_smithdb_v2_with_full_time_window():
    started = datetime(2025, 1, 2, tzinfo=timezone.utc)
    project = _project(
        feedback={"groundedness": {"avg": 0.8}},
        session_feedback={"abstention_recall": {"avg": 0.7}},
        started=started,
    )
    runs = FakeRuns([_run(document_recall_at_5=1.0), _run(document_recall_at_5=0.5)])
    client = FakeClient(project=project, runs=runs)

    metrics = await experiment_metrics(client, "baseline")

    assert metrics == {
        "document_recall_at_5": 0.75,
        "groundedness": 0.8,
        "abstention_recall": 0.7,
    }
    assert runs.calls == [
        {
            "project_ids": [str(project.id)],
            "is_root": True,
            "min_start_time": started,
            "selects": ["ID", "FEEDBACK_STATS"],
            "page_size": 1_000,
        }
    ]


async def test_experiment_metrics_request_project_statistics_by_name():
    project = _project(feedback={"groundedness": {"avg": 0.9}})
    client = FakeClient(project=project, runs=FakeRuns([]))

    await experiment_metrics(client, "candidate")

    assert client.project_calls == [{"project_name": "candidate", "include_stats": True}]


async def test_experiment_metrics_ignore_feedback_entries_without_an_average():
    project = _project(feedback={"unscored": {"avg": None}, "groundedness": {"avg": 0.4}})
    runs = FakeRuns(
        [
            Run(feedback_stats={"partial": {"n": 3}}),
            Run(feedback_stats=None),
            SimpleNamespace(feedback_stats={"ignored_without_avg": {"n": 1}}),
            _run(document_recall_at_5=0.25),
        ]
    )

    metrics = await experiment_metrics(FakeClient(project=project, runs=runs), "candidate")

    assert metrics == {"document_recall_at_5": 0.25, "groundedness": 0.4}


async def test_project_statistics_override_root_run_averages():
    project = _project(
        feedback={"document_recall_at_5": {"avg": 0.1}},
        session_feedback={"document_recall_at_5": {"avg": 0.2}},
    )
    runs = FakeRuns([_run(document_recall_at_5=1.0)])

    metrics = await experiment_metrics(FakeClient(project=project, runs=runs), "candidate")

    assert metrics == {"document_recall_at_5": 0.2}


async def test_experiment_without_any_feedback_fails_closed():
    client = FakeClient(project=_project(), runs=FakeRuns([]))

    with pytest.raises(ValueError, match="no deterministic feedback"):
        await experiment_metrics(client, "empty-experiment")


async def test_comparison_metrics_return_candidate_then_baseline():
    candidate_project = _project(feedback={"groundedness": {"avg": 0.9}})
    baseline_project = _project(feedback={"groundedness": {"avg": 0.6}})
    runs = FakeRuns(
        rows_by_project={
            str(candidate_project.id): [_run(document_recall_at_5=1.0)],
            str(baseline_project.id): [_run(document_recall_at_5=0.5)],
        }
    )
    client = FakeClient(
        projects={"candidate": candidate_project, "baseline": baseline_project},
        runs=runs,
    )

    candidate, baseline = await comparison_metrics(client, "candidate", "baseline")

    assert candidate == {"document_recall_at_5": 1.0, "groundedness": 0.9}
    assert baseline == {"document_recall_at_5": 0.5, "groundedness": 0.6}
    assert len(runs.calls) == 2


async def test_comparison_metrics_propagate_a_missing_baseline():
    client = FakeClient(
        projects={"candidate": _project(feedback={"groundedness": {"avg": 0.9}})},
        runs=FakeRuns([]),
    )

    with pytest.raises(ValueError, match="unknown experiment: baseline"):
        await comparison_metrics(client, "candidate", "baseline")


async def test_a_project_without_a_start_time_is_refused_before_it_is_queried():
    """A missing lower bound silently means "the last 24 hours".

    ``start_time`` is optional on the project schema, and ``min_start_time``
    defaults to one day ago when absent. Passing it through unchecked turns an
    older experiment into zero runs, which reads as "no deterministic feedback"
    rather than as the truncation it is.
    """
    project = SimpleNamespace(
        id=uuid4(),
        start_time=None,
        feedback_stats={"groundedness": {"avg": 0.8}},
        session_feedback_stats=None,
    )
    runs = FakeRuns([_run(document_recall_at_5=1.0)])

    with pytest.raises(ValueError, match="start_time"):
        await experiment_metrics(FakeClient(project=project, runs=runs), "candidate")

    assert runs.calls == []


async def test_the_query_asks_for_the_largest_supported_page():
    """The default page is 100 runs; a golden-dataset experiment is larger."""
    project = _project(feedback={"groundedness": {"avg": 0.9}})
    runs = FakeRuns([])

    await experiment_metrics(FakeClient(project=project, runs=runs), "candidate")

    assert runs.calls[0]["page_size"] == 1_000


def test_installed_langsmith_supports_smithdb_queries():
    assert Version(version("langsmith")) >= SMITHDB_FLOOR


def _declared_specifier(filename: str) -> str:
    matches = _SPECIFIER.findall((ROOT / filename).read_text(encoding="utf-8"))
    assert len(matches) == 1, f"{filename} must declare langsmith exactly once, found {matches}"
    return matches[0]


@pytest.mark.parametrize("filename", DECLARATION_FILES)
def test_declared_langsmith_floor_excludes_pre_smithdb_releases(filename: str):
    permitted = SpecifierSet(_declared_specifier(filename))

    assert not [release for release in PRE_SMITHDB_RELEASES if permitted.contains(release)]
    assert [release for release in SMITHDB_RELEASES if permitted.contains(release)]


def _executable_source(source: str) -> str:
    """Blank comments and docstrings, preserving every other character position.

    Prose that names a deprecated method is documentation, not a call site --
    this module's own docstring explains the endpoints it replaces. Ordinary
    string literals stay intact so a hardcoded ``/api/v1/runs/query`` URL is
    still caught.
    """
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    characters = list(source)

    def blank(start: tuple[int, int], end: tuple[int, int]) -> None:
        first = offsets[start[0] - 1] + start[1]
        last = min(offsets[end[0] - 1] + end[1], len(characters))
        for index in range(first, last):
            if characters[index] != "\n":
                characters[index] = " "

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            blank(token.start, token.end)

    documented = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, documented) or not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
            and first.end_col_offset is not None
        ):
            blank((first.lineno, first.col_offset), (first.end_lineno, first.end_col_offset))

    return "".join(characters)


def _deprecation_code_marks() -> tuple[Any, Any]:
    """The code objects langsmith's own ``deprecated`` decorator produces.

    Every decoration shares one code object, so identity against a probe is an
    exact test for "the installed SDK marks this method deprecated" -- with no
    hand-maintained list to fall behind the next SDK bump. That matters here:
    the four markers this replaced named three of the seventeen methods the
    installed SDK actually marks.
    """

    def _probe() -> None: ...

    async def _aprobe() -> None: ...

    return deprecated("probe")(_probe).__code__, deprecated("probe")(_aprobe).__code__


def _deprecated_client_methods() -> dict[str, str]:
    """Every deprecated client method, mapped to the SDK's own guidance."""
    sync_mark, async_mark = _deprecation_code_marks()
    found: dict[str, str] = {}
    for client_class in CLIENT_CLASSES:
        for name in dir(client_class):
            if name.startswith("_"):
                continue
            attribute = getattr(client_class, name, None)
            if getattr(attribute, "__code__", None) not in (sync_mark, async_mark):
                continue
            found[name] = next(
                (
                    cell.cell_contents
                    for cell in (attribute.__closure__ or ())
                    if isinstance(cell.cell_contents, str)
                ),
                "deprecated by the installed SDK",
            )
    return found


def _methods_reaching_v1_behind_a_suppressed_warning() -> dict[str, str]:
    """Supported methods that call a deprecated one with the warning silenced.

    These are the dangerous ones. They are not themselves deprecated and they
    emit nothing at runtime, so the only way to know is to read the SDK --
    which is exactly how ``get_test_results()``, the call this migration
    started from, looked clean while reaching ``POST /api/v1/runs/query``.
    """
    found: dict[str, str] = {}
    for client_class in CLIENT_CLASSES:
        for name in dir(client_class):
            if name.startswith("_"):
                continue
            attribute = getattr(client_class, name, None)
            if not callable(attribute):
                continue
            try:
                source = inspect.getsource(attribute)
            except (OSError, TypeError):
                continue
            if "suppress_deprecation_warning" in source:
                found[name] = (
                    "reaches a v1 endpoint behind a suppressed deprecation "
                    "warning; call the v2 resource directly"
                )
    return found


def banned_client_methods() -> dict[str, str]:
    return {
        **_deprecated_client_methods(),
        **_methods_reaching_v1_behind_a_suppressed_warning(),
    }


def test_the_ban_list_is_derived_from_the_installed_sdk():
    """Guard the guard: an empty or stale inventory would pass everything."""
    deprecated_methods = _deprecated_client_methods()

    assert {"list_runs", "read_run", "get_experiment_results"} <= set(deprecated_methods)
    assert all("2027" in message for message in deprecated_methods.values())
    assert "get_test_results" in _methods_reaching_v1_behind_a_suppressed_warning()


def test_production_python_calls_no_deprecated_or_v1_falling_back_client_method():
    """Ban by method name, accepting that a name can belong to something else.

    ``.evaluate_run(`` is also the LangSmith *evaluator* protocol method, so a
    custom evaluator would trip this. That is the right way round to be wrong:
    a false positive is one loud line naming the SDK's own guidance, and the
    developer can see in a second that it is not a client call. A miss is what
    this whole migration is cleaning up.
    """
    banned = banned_client_methods()
    offenders: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            code = _executable_source(path.read_text(encoding="utf-8"))
            offenders.extend(
                f"{path.relative_to(ROOT).as_posix()}: .{name}() -- {banned[name]}"
                for name in sorted(banned)
                if f".{name}(" in code
            )

    assert offenders == []


def test_production_python_contains_no_legacy_langsmith_endpoint():
    offenders: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            code = _executable_source(path.read_text(encoding="utf-8"))
            offenders.extend(
                f"{path.relative_to(ROOT).as_posix()}: {literal}"
                for literal in LEGACY_ENDPOINT_LITERALS
                if literal in code
            )
            if LEGACY_DATASET_RUNS_PATH.search(code):
                offenders.append(
                    f"{path.relative_to(ROOT).as_posix()}: legacy experiment-run path"
                )

    assert offenders == []


def test_the_inventory_still_detects_a_real_call_site():
    """Guard the guard: blanking documentation must not blind the scan."""
    code = _executable_source(
        '"""Mentions .list_runs( and /api/v1/runs/query in prose."""\n'
        "# and .get_test_results( in a comment\n"
        "def read(client):\n"
        '    """Docstring naming .get_experiment_results( too."""\n'
        "    client.list_runs(project_name='p')\n"
        '    return client.session.post("https://host/api/v1/runs/query")\n'
    )

    assert [name for name in banned_client_methods() if f".{name}(" in code] == ["list_runs"]
    # The retrieve prefix is contained in the query path, so one hardcoded URL
    # trips both. Overlap costs a duplicated offender line and nothing else.
    assert {literal for literal in LEGACY_ENDPOINT_LITERALS if literal in code} == {
        "/api/v1/runs/query",
        "/api/v1/runs/",
    }


def test_the_v2_replacement_path_is_not_itself_banned():
    """``/api/v2/runs/query`` is the fix; an unanchored literal would ban it."""
    code = 'REPLACEMENT = "https://host/api/v2/runs/query"'

    assert [literal for literal in LEGACY_ENDPOINT_LITERALS if literal in code] == []


def test_the_run_double_carries_the_sdk_feedback_type_not_a_dictionary():
    """Guard the guard: this fixture only proves anything while it stays typed."""
    statistics = _run(document_recall_at_5=0.75).feedback_stats["document_recall_at_5"]

    assert not isinstance(statistics, dict)
    assert statistics.avg == 0.75
