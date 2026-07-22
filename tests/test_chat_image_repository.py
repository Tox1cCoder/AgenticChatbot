from contextlib import contextmanager
from uuid import uuid4

from app.repositories.chat_image import ChatImageRepository


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, rows=None):
        self.added = []
        self.committed = 0
        self.refreshed = []
        self._rows = rows or []

    def add(self, obj):
        obj.id = obj.id or uuid4()
        self.added.append(obj)

    def commit(self):
        self.committed += 1

    def refresh(self, obj):
        self.refreshed.append(obj)

    def execute(self, _stmt):
        return _FakeQuery(self._rows)


def _factory(session):
    @contextmanager
    def _f():
        yield session

    return _f


def test_create_persists_and_commits():
    session = _FakeSession()
    repo = ChatImageRepository(_factory(session))
    row = repo.create(
        {
            "id": uuid4(),
            "conversation_id": uuid4(),
            "user_id": uuid4(),
            "sha256": "a" * 64,
            "size_bytes": 10,
            "content_type": "image/png",
            "storage_path": "ab/abc.png",
        }
    )
    assert session.committed == 1
    assert row in session.added


def test_get_for_user_returns_row():
    marker = object()
    repo = ChatImageRepository(_factory(_FakeSession(rows=[marker])))
    assert repo.get_for_user(uuid4(), uuid4()) is marker


def test_find_active_by_sha_for_user_returns_row():
    marker = object()
    repo = ChatImageRepository(_factory(_FakeSession(rows=[marker])))
    assert repo.find_active_by_sha_for_user("a" * 64, uuid4()) is marker
