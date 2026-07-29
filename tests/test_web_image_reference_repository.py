"""Async repository behavior for selected web-image references."""

from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

import pytest

from app.repositories.web_image_reference import WebImageReferenceRepository


class _ScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self

    def first(self):
        return self.rows[0] if self.rows else None


class _SyncSession:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.added = []
        self.commits = 0
        self.refreshed = []
        self.statements = []

    def add(self, value):
        self.added.append(value)

    def commit(self):
        self.commits += 1

    def refresh(self, value):
        self.refreshed.append(value)

    def execute(self, statement):
        self.statements.append(statement)
        return _ScalarResult(self.rows)


class _AsyncSession:
    def __init__(self, sync_session):
        self.sync_session = sync_session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def run_sync(self, work):
        return work(self.sync_session)


def _sync_factory_that_must_not_run():
    @contextmanager
    def factory():
        raise AssertionError("sync DB transport must not run")
        yield

    return factory


def _async_factory(session):
    return lambda: _AsyncSession(session)


@pytest.mark.asyncio
async def test_acreate_uses_async_transport_and_commits():
    session = _SyncSession()
    repository = WebImageReferenceRepository(
        _sync_factory_that_must_not_run(), _async_factory(session)
    )
    data = {
        "id": uuid4(),
        "conversation_id": uuid4(),
        "user_id": uuid4(),
        "upstream_url": "https://img.example/image.jpg",
        "expected_mime": "image/jpeg",
        "provider": "brave",
    }

    record = await repository.acreate(data)

    assert record in session.added
    assert record.upstream_url == data["upstream_url"]
    assert session.commits == 1
    assert session.refreshed == [record]


@pytest.mark.asyncio
async def test_aget_for_user_filters_id_owner_and_soft_delete():
    marker = object()
    session = _SyncSession(rows=[marker])
    repository = WebImageReferenceRepository(
        _sync_factory_that_must_not_run(), _async_factory(session)
    )
    image_id = uuid4()
    user_id = uuid4()

    assert await repository.aget_for_user(image_id, user_id) is marker

    compiled = str(session.statements[0].compile(compile_kwargs={"literal_binds": True}))
    assert "web_image_references.id" in compiled
    assert image_id.hex in compiled
    assert "web_image_references.user_id" in compiled
    assert user_id.hex in compiled
    assert "web_image_references.deleted_at IS NULL" in compiled
