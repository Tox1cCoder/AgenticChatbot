"""ToolApprovalSettingRepository uses the sync session-factory style (no real DB)."""

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from app.repositories.tool_approval_setting import ToolApprovalSettingRepository


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.added = []
        self.deleted = []
        self.committed = 0

    def execute(self, _stmt):
        rows = self._rows

        class _Result:
            def scalars(self_inner):
                return SimpleNamespace(all=lambda: list(rows))

            def scalar_one_or_none(self_inner):
                return rows[0] if rows else None

        return _Result()

    def add(self, obj):
        self.added.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed += 1

    def refresh(self, _obj):
        pass

    def expunge(self, _obj):
        pass


def _factory(session):
    @contextmanager
    def _cm():
        yield session

    return _cm


def test_set_creates_when_missing():
    session = _FakeSession(rows=[])
    repo = ToolApprovalSettingRepository(_factory(session))
    user_id = uuid4()

    setting = repo.set(user_id, "server", "desktop_commander", True)

    assert session.added and session.committed >= 1
    assert setting.scope_type == "server"
    assert setting.scope_value == "desktop_commander"
    assert setting.require_approval is True


def test_set_updates_when_present():
    existing = SimpleNamespace(
        scope_type="tool", scope_value="excel::delete_sheet", require_approval=False
    )
    session = _FakeSession(rows=[existing])
    repo = ToolApprovalSettingRepository(_factory(session))

    setting = repo.set(uuid4(), "tool", "excel::delete_sheet", True)

    assert setting is existing
    assert setting.require_approval is True
    assert not session.added  # updated in place, not inserted


def test_build_policy_groups_by_scope():
    rows = [
        SimpleNamespace(
            scope_type="server", scope_value="desktop_commander", require_approval=True
        ),
        SimpleNamespace(
            scope_type="tool", scope_value="desktop_commander::list_files", require_approval=False
        ),
        SimpleNamespace(scope_type="garbage", scope_value="ignored", require_approval=True),
    ]
    repo = ToolApprovalSettingRepository(_factory(_FakeSession(rows=rows)))

    policy = repo.build_policy(uuid4())

    assert policy == {
        "servers": {"desktop_commander": True},
        "tools": {"desktop_commander::list_files": False},
    }


def test_rejects_invalid_scope_type():
    repo = ToolApprovalSettingRepository(_factory(_FakeSession(rows=[])))
    try:
        repo.set(uuid4(), "garbage", "ignored", True)
    except ValueError as exc:
        assert "scope_type" in str(exc)
    else:
        raise AssertionError("invalid scope_type should fail")
