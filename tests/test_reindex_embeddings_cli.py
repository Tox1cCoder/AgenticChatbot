"""Guards for the embedding reindex CLI used during collection migrations."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from uuid import UUID

import pytest

from scripts import reindex_embeddings


def test_cli_accepts_selectors_dry_run_and_continue_on_error():
    assert reindex_embeddings._parse_args(["--document-id", "abc"]).document_id == "abc"
    assert (
        reindex_embeddings._parse_args(["--conversation-id", "conv-1"]).conversation_id == "conv-1"
    )
    assert reindex_embeddings._parse_args(["--all"]).all is True
    assert reindex_embeddings._parse_args(["--dry-run"]).dry_run is True
    assert (
        reindex_embeddings._parse_args(["--all", "--continue-on-error"]).continue_on_error is True
    )
    with pytest.raises(SystemExit):
        reindex_embeddings._parse_args(["--document-id", "d", "--conversation-id", "c"])


def test_dry_run_only_resolves_database_targets_without_provider_initialization(
    monkeypatch, capsys
):
    document_id = UUID("31ed2dbc-3312-4490-b8fc-2b721f290fab")

    class _Session:
        def execute(self, _statement, _params):
            return SimpleNamespace(all=lambda: [(document_id,)])

    class _DB:
        @contextmanager
        def session(self):
            yield _Session()

    provider_calls: list[str] = []

    def _unexpected_provider(name: str):
        provider_calls.append(name)
        raise AssertionError(f"dry-run initialized {name}")

    container = SimpleNamespace(
        db=lambda: _DB(),
        document_index_service=lambda: _unexpected_provider("document index"),
        rag_embedding_service=lambda: _unexpected_provider("embedding provider"),
    )
    container_module = ModuleType("app.core.container")
    container_module.get_container = lambda: container
    monkeypatch.setitem(sys.modules, "app.core.container", container_module)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    result = reindex_embeddings._run(
        reindex_embeddings._parse_args(["--document-id", str(document_id), "--dry-run"])
    )

    assert result == 0
    assert provider_calls == []
    assert str(document_id) in capsys.readouterr().out


def test_missing_selector_is_rejected_before_container_initialization(monkeypatch):
    container_module = ModuleType("app.core.container")
    container_module.get_container = lambda: (_ for _ in ()).throw(
        AssertionError("selector validation initialized the container")
    )
    monkeypatch.setitem(sys.modules, "app.core.container", container_module)

    assert reindex_embeddings._run(reindex_embeddings._parse_args([])) == 2


def test_mark_needs_reindex_casts_document_ids_to_uuid_array():
    executed = {}

    class _Session:
        def execute(self, statement, params):
            executed["statement"] = str(statement)
            executed["params"] = params

        def commit(self):
            executed["committed"] = True

    class _DB:
        @contextmanager
        def session(self):
            yield _Session()

    reindex_embeddings._mark_needs_reindex(
        _DB(),
        [UUID("31ed2dbc-3312-4490-b8fc-2b721f290fab")],
    )

    assert "ANY(CAST(:ids AS uuid[]))" in executed["statement"]
    assert executed["params"]["ids"] == ["31ed2dbc-3312-4490-b8fc-2b721f290fab"]
    assert executed["committed"] is True
