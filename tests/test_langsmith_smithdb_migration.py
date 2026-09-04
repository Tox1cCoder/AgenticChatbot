"""SmithDB v2 read contracts for the LangSmith evaluation queries.

The v1 run-query, run-retrieve, legacy experiment-run, sharing, and annotation
queue endpoint families carry ``Deprecation: true`` with a 2027-01-31 sunset.
These tests pin the replacement call shape and fail closed if any production
module reaches for a legacy query method again.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from app.evaluation.rag.langsmith_queries import comparison_metrics, experiment_metrics

ROOT = Path(__file__).resolve().parents[1]
SMITHDB_FLOOR = Version("0.10.15")
LEGACY_QUERY_MARKERS = (
    ".list_runs(",
    ".get_test_results(",
    ".get_experiment_results(",
    "/api/v1/runs/query",
)
PRODUCTION_ROOTS = ("app", "scripts")
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
    return SimpleNamespace(
        feedback_stats={key: {"avg": value} for key, value in feedback_stats.items()}
    )


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
            SimpleNamespace(feedback_stats={"partial": {"n": 3}}),
            SimpleNamespace(feedback_stats=None),
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


def test_production_python_contains_no_legacy_langsmith_run_queries():
    offenders: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            code = _executable_source(path.read_text(encoding="utf-8"))
            offenders.extend(
                f"{path.relative_to(ROOT).as_posix()}: {marker}"
                for marker in LEGACY_QUERY_MARKERS
                if marker in code
            )

    assert offenders == []


def test_the_legacy_query_inventory_still_detects_a_real_call_site():
    """Guard the guard: blanking documentation must not blind the scan."""
    code = _executable_source(
        '"""Mentions .list_runs( and /api/v1/runs/query in prose."""\n'
        "# and .get_test_results( in a comment\n"
        "def read(client):\n"
        '    """Docstring naming .get_experiment_results( too."""\n'
        "    client.list_runs(project_name='p')\n"
        '    return client.session.post("https://host/api/v1/runs/query")\n'
    )

    assert [marker for marker in LEGACY_QUERY_MARKERS if marker in code] == [
        ".list_runs(",
        "/api/v1/runs/query",
    ]
