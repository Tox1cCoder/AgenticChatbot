"""DB-free coverage for ToolResultBlobRepository.get_for_user_and_conversation's WHERE clause.

tests/integration/test_tool_result_blob_repository_postgres.py proves real database
semantics, but it skips whenever TEST_DATABASE_URL is unset -- which is every bare
`pytest` invocation in this repo, since nothing sets that variable by default. This
file captures the actual statement the repository hands to `execute()` and inspects
its predicates directly, so a dropped or swapped predicate fails a default run
instead of silently passing behind a skipped test.
"""

from __future__ import annotations

import operator
from contextlib import contextmanager
from uuid import UUID, uuid4

from sqlalchemy.sql import operators as sa_operators

from app.repositories.tool_result_blob import ToolResultBlobRepository


class _FakeResult:
    def scalars(self):
        return self

    def first(self):
        return None


class _FakeSession:
    def __init__(self):
        self.captured_statement = None

    def execute(self, statement):
        self.captured_statement = statement
        return _FakeResult()


def _fake_session_factory(session: _FakeSession):
    @contextmanager
    def factory():
        yield session

    return factory


def _clauses(statement) -> list:
    whereclause = statement.whereclause
    return list(getattr(whereclause, "clauses", [whereclause]))


def _equality_predicates(statement) -> dict[str, UUID]:
    """Map column name -> bound value for every `column == value` predicate."""
    return {
        clause.left.name: clause.right.value
        for clause in _clauses(statement)
        if clause.operator is operator.eq
    }


def _is_null_predicate_columns(statement) -> list[str]:
    return [
        clause.left.name for clause in _clauses(statement) if clause.operator is sa_operators.is_
    ]


def test_statement_binds_id_user_and_conversation_to_distinct_columns():
    """Three distinct values catch a swap, not just a missing predicate.

    A test using the same UUID for user_id and conversation_id would still pass if
    the repository swapped those two columns; using distinct values and asserting
    the exact mapping makes a swap produce a mismatched dict, not a coincidental match.
    """
    session = _FakeSession()
    repository = ToolResultBlobRepository(session_factory=_fake_session_factory(session))
    blob_id, user_id, conversation_id = uuid4(), uuid4(), uuid4()

    repository.get_for_user_and_conversation(blob_id, user_id, conversation_id)

    assert session.captured_statement is not None
    assert _equality_predicates(session.captured_statement) == {
        "id": blob_id,
        "user_id": user_id,
        "conversation_id": conversation_id,
    }


def test_statement_excludes_soft_deleted_rows():
    session = _FakeSession()
    repository = ToolResultBlobRepository(session_factory=_fake_session_factory(session))

    repository.get_for_user_and_conversation(uuid4(), uuid4(), uuid4())

    assert _is_null_predicate_columns(session.captured_statement) == ["deleted_at"]
