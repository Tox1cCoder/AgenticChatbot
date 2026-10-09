"""PostgreSQL checks for the conversation fork-lineage revision (skips without a URL)."""

from __future__ import annotations

import os
import subprocess
import sys
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from tests.test_alembic_full_chain_postgres import (
    _PROJECT_ROOT,
    _postgres_test_url,
    _run_alembic,
    _scratch_database,
    _seed_owner,
)

_PARENT = "9b132832d676"
_REVISION = "c4f1a7b9d2e6"
_NEW_COLUMNS = {
    "forked_from_conversation_id",
    "forked_from_message_id",
    "forked_through_sequence",
    "fork_request_id",
    "fork_request_fingerprint",
}
_NEW_CONSTRAINTS = {
    "fk_conversations_forked_from_conversation_id",
    "fk_conversations_forked_from_message_id",
    "uq_conversations_owner_fork_request",
    "ck_conversations_forked_through_sequence_positive",
    "ck_conversations_fork_request_fields_together",
}


def _constraint_names(connection) -> set[str]:
    rows = connection.execute(
        text("SELECT conname FROM pg_constraint WHERE conrelid = 'conversations'::regclass")
    )
    return {row[0] for row in rows}


def _columns(connection) -> set[str]:
    return {col["name"] for col in inspect(connection).get_columns("conversations")}


def _insert_fork(connection, owner_id, **values) -> None:
    columns = {"id": uuid4(), "owner_id": owner_id, **values}
    names = ", ".join(columns)
    binds = ", ".join(f":{name}" for name in columns)
    connection.execute(
        text(
            f"INSERT INTO conversations ({names}, title, created_at, updated_at, "
            f"planning_mode_enabled, next_message_sequence) "
            f"VALUES ({binds}, 'fork', now(), now(), false, 1)"
        ),
        columns,
    )


def test_upgrade_keeps_existing_rows_and_downgrade_round_trips() -> None:
    owner_id, conversation_id = uuid4(), uuid4()
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _PARENT)
        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                _seed_owner(connection, owner_id, conversation_id, name="fork-existing")

            _run_alembic(scratch_url, "upgrade", _REVISION)
            with engine.connect() as connection:
                assert _columns(connection) >= _NEW_COLUMNS
                assert _constraint_names(connection) >= _NEW_CONSTRAINTS
                row = connection.execute(
                    text(
                        "SELECT forked_from_conversation_id, forked_from_message_id, "
                        "forked_through_sequence, fork_request_id, fork_request_fingerprint "
                        "FROM conversations WHERE id = :id"
                    ),
                    {"id": conversation_id},
                ).one()
                assert tuple(row) == (None,) * 5

            _run_alembic(scratch_url, "downgrade", _PARENT)
            with engine.connect() as connection:
                assert not _NEW_COLUMNS & _columns(connection)
                assert not _NEW_CONSTRAINTS & _constraint_names(connection)
                assert (
                    connection.scalar(
                        text("SELECT count(*) FROM conversations WHERE id = :id"),
                        {"id": conversation_id},
                    )
                    == 1
                )

            _run_alembic(scratch_url, "upgrade", "head")
            with engine.connect() as connection:
                assert _columns(connection) >= _NEW_COLUMNS
        finally:
            engine.dispose()


def test_constraints_and_set_null_behaviour_at_head() -> None:
    owner_id, other_id, source_id = uuid4(), uuid4(), uuid4()
    message_id = uuid4()
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", "head")
        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                _seed_owner(connection, owner_id, source_id, name="fork-owner")
                _seed_owner(connection, other_id, name="fork-other")
                connection.execute(
                    text(
                        "INSERT INTO messages (id, conversation_id, sender, content, sequence, "
                        "created_at, updated_at) VALUES (:id, :cid, 1, 'hi', 1, now(), now())"
                    ),
                    {"id": message_id, "cid": source_id},
                )

            request_id = uuid4()
            child_id = uuid4()
            with engine.begin() as connection:
                _insert_fork(
                    connection,
                    owner_id,
                    id=child_id,
                    forked_from_conversation_id=source_id,
                    forked_from_message_id=message_id,
                    forked_through_sequence=1,
                    fork_request_id=request_id,
                    fork_request_fingerprint="a" * 64,
                )
                # Same request ID under another owner is a different key.
                _insert_fork(
                    connection,
                    other_id,
                    fork_request_id=request_id,
                    fork_request_fingerprint="a" * 64,
                )

            rejected = {
                "duplicate owner/request": dict(
                    fork_request_id=request_id, fork_request_fingerprint="b" * 64
                ),
                "zero sequence": dict(forked_through_sequence=0),
                "request id without fingerprint": dict(fork_request_id=uuid4()),
                "fingerprint without request id": dict(fork_request_fingerprint="c" * 64),
            }
            for values in rejected.values():
                with pytest.raises(IntegrityError), engine.begin() as connection:
                    _insert_fork(connection, owner_id, **values)

            # The unique constraint must cover soft-deleted children.
            with engine.begin() as connection:
                connection.execute(
                    text("UPDATE conversations SET deleted_at = now() WHERE id = :id"),
                    {"id": child_id},
                )
            with pytest.raises(IntegrityError), engine.begin() as connection:
                _insert_fork(
                    connection,
                    owner_id,
                    fork_request_id=request_id,
                    fork_request_fingerprint="a" * 64,
                )

            # Hard-deleting the message, then the source, nulls the lineage and
            # keeps the cutoff; the child survives both.
            with engine.begin() as connection:
                connection.execute(text("DELETE FROM messages WHERE id = :id"), {"id": message_id})
                connection.execute(
                    text("DELETE FROM conversations WHERE id = :id"), {"id": source_id}
                )
                row = connection.execute(
                    text(
                        "SELECT forked_from_conversation_id, forked_from_message_id, "
                        "forked_through_sequence FROM conversations WHERE id = :id"
                    ),
                    {"id": child_id},
                ).one()
            assert tuple(row) == (None, None, 1)
        finally:
            engine.dispose()


def test_autogenerate_reports_no_conversation_drift() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", "head")
        env = dict(os.environ)
        env["DATABASE_URL"] = scratch_url.render_as_string(hide_password=False)
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "check"],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    output = f"{result.stdout}\n{result.stderr}"
    assert "conversations" not in output, "conversation drift detected"


def test_metadata_create_all_and_drop_all_handle_the_conversation_message_cycle() -> None:
    from app.models.base import Base

    with _scratch_database(_postgres_test_url()) as scratch_url:
        engine = create_engine(scratch_url)
        try:
            Base.metadata.create_all(engine)
            assert "conversations" in inspect(engine).get_table_names()
            Base.metadata.drop_all(engine)
            assert inspect(engine).get_table_names() == []
        finally:
            engine.dispose()
