"""Phase 3 guards: DocumentParseArtifactRepository contract.

The repository must match the sync, session-factory style used elsewhere on
the server — not the async/AsyncSession shape that previously existed. No
test here hits a real database; we verify the repository calls the session
factory correctly and persists the expected model shape.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.models.document_parse_artifact import DocumentParseArtifact
from app.repositories.document_parse_artifact import DocumentParseArtifactRepository


class _FakeQuery:
    def __init__(self, session):
        self._session = session

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._session._query_results

    def first(self):
        return self._session._query_results[0] if self._session._query_results else None

    def delete(self, synchronize_session=False):
        self._session._deleted_query_ran = True
        return len(self._session._query_results)


class _FakeSession:
    def __init__(self):
        self.added: list = []
        self.deleted: list = []
        self.committed = 0
        self.refreshed: list = []
        self.closed = False
        self._query_results: list = []
        self._deleted_query_ran = False

    def add(self, obj):
        self.added.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed += 1

    def refresh(self, obj):
        self.refreshed.append(obj)

    def query(self, _model):
        return _FakeQuery(self)

    def execute(self, _stmt):
        # Not used in this repo, but supply a default for compatibility.
        result = MagicMock()
        result.scalars.return_value.all.return_value = self._query_results
        result.scalar_one_or_none.return_value = (
            self._query_results[0] if self._query_results else None
        )
        return result


def _fake_session_factory(session: _FakeSession):
    @contextmanager
    def factory():
        try:
            yield session
        finally:
            session.closed = True

    return factory


def test_repository_accepts_session_factory():
    sig = inspect.signature(DocumentParseArtifactRepository.__init__)
    assert "session_factory" in sig.parameters, (
        f"Constructor must accept session_factory; got {list(sig.parameters)}"
    )


def test_repository_methods_are_synchronous():
    for name, member in inspect.getmembers(
        DocumentParseArtifactRepository, predicate=inspect.isfunction
    ):
        if name.startswith("_"):
            continue
        assert not inspect.iscoroutinefunction(member), (
            f"{name} must be sync; the rest of the DB layer is sync"
        )


def test_create_builds_artifact_row_and_commits():
    session = _FakeSession()
    repo = DocumentParseArtifactRepository(session_factory=_fake_session_factory(session))

    document_id = uuid4()
    artifact = repo.create(
        document_id=document_id,
        artifact_type="mineru_markdown",
        storage_path="/tmp/output.md",
        mime_type="text/markdown",
        size_bytes=1024,
        checksum_sha256="abc123",
        artifact_metadata={"pages": 4},
    )

    assert isinstance(artifact, DocumentParseArtifact)
    assert artifact.document_id == document_id
    assert artifact.artifact_type == "mineru_markdown"
    assert artifact.storage_path == "/tmp/output.md"
    assert artifact.checksum_sha256 == "abc123"
    assert artifact.artifact_metadata == {"pages": 4}

    assert len(session.added) == 1
    assert session.committed == 1


def test_list_by_document_queries_by_document_id():
    session = _FakeSession()
    marker = DocumentParseArtifact(
        id=uuid4(),
        document_id=uuid4(),
        artifact_type="mineru_markdown",
        storage_path="/tmp/x.md",
    )
    session._query_results = [marker]

    repo = DocumentParseArtifactRepository(session_factory=_fake_session_factory(session))

    results = repo.list_by_document(document_id=marker.document_id)
    assert results == [marker]


def test_replace_for_document_deletes_then_adds():
    session = _FakeSession()
    existing = [
        DocumentParseArtifact(
            id=uuid4(),
            document_id=uuid4(),
            artifact_type="mineru_markdown",
            storage_path="/tmp/old.md",
        )
    ]
    session._query_results = existing

    repo = DocumentParseArtifactRepository(session_factory=_fake_session_factory(session))
    document_id = existing[0].document_id

    created = repo.replace_for_document(
        document_id=document_id,
        artifacts=[
            {
                "artifact_type": "mineru_markdown",
                "storage_path": "/tmp/new.md",
                "mime_type": "text/markdown",
                "size_bytes": 1,
                "checksum_sha256": "sha",
                "artifact_metadata": {},
            }
        ],
    )

    assert session._deleted_query_ran, "Old artifacts must be cleared before insert"
    assert len(session.added) == 1
    assert isinstance(created[0], DocumentParseArtifact)
    assert created[0].document_id == document_id


def test_container_exposes_parse_artifact_and_chunk_repos():
    from app.core.container import container

    assert hasattr(container, "document_parse_artifact_repository"), (
        "container must expose document_parse_artifact_repository"
    )
    assert hasattr(container, "document_chunk_repository"), (
        "container must expose document_chunk_repository"
    )

    repo_pa = container.document_parse_artifact_repository()
    assert isinstance(repo_pa, DocumentParseArtifactRepository)

    from app.repositories.document_chunk import DocumentChunkRepository

    repo_ch = container.document_chunk_repository()
    assert isinstance(repo_ch, DocumentChunkRepository)
