"""Guards for the embedding reindex CLI used during collection migrations."""

from __future__ import annotations

from contextlib import contextmanager
from uuid import UUID

from scripts import reindex_embeddings


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
