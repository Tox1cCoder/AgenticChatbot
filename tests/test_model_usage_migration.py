from __future__ import annotations

import importlib
import inspect
import os
import re

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy import inspect as sa_inspect

from app.models.base import Base
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.message import Message
from app.models.user import User

# Dedicated, dropped-and-recreated schema so this test never depends on (or
# mutates) whatever alembic_version/table state the shared TEST_DATABASE_URL
# database happens to be in — it was found to have no alembic_version row at
# all (tables seeded ad hoc by other integration tests), so pretending it
# sits at any particular revision would be false. Only the four tables our
# migration's foreign keys target are seeded here.
_SCRATCH_SCHEMA = "model_usage_migration_test"


def _migration_module():
    return importlib.import_module("app.alembic.versions.y2z3a4b5c6d7_add_model_usage_ledger")


def test_model_usage_ledger_migration_is_the_single_new_head():
    migration = _migration_module()

    assert migration.revision == "y2z3a4b5c6d7"
    assert migration.down_revision == "x1y2z3a4b5c6"


def test_upgrade_creates_both_tables_with_matching_ondelete_clauses():
    source = inspect.getsource(_migration_module().upgrade)
    lowered = source.lower()

    assert '"model_usage_events"' in lowered
    assert '"model_usage_minute"' in lowered
    assert "uq_model_usage_events_event_key" in lowered
    assert "uq_model_usage_events_operation_attempt" in lowered
    assert "pk_model_usage_minute" in lowered
    assert "insert into" not in lowered


# Matches a single `sa.ForeignKeyConstraint([...], [...], name="...",
# ondelete="...")` call and captures its source column(s), name, and
# ondelete value as one bounded unit — unlike a bare `'ondelete="..."' in
# source` substring check, this can't pass if CASCADE/SET NULL were swapped
# between two columns, because each match is scoped to one FK's own block.
_FK_BLOCK_RE = re.compile(
    r"sa\.ForeignKeyConstraint\(\s*"
    r"\[(?P<source_cols>[^\]]*)\],\s*"
    r"\[(?P<target_cols>[^\]]*)\],\s*"
    r'name="(?P<name>[^"]+)",\s*'
    r'ondelete="(?P<ondelete>[^"]+)",?\s*'
    r"\)",
    re.DOTALL,
)


def _parsed_foreign_keys(source: str) -> dict[str, tuple[tuple[str, ...], str]]:
    parsed = {}
    for match in _FK_BLOCK_RE.finditer(source):
        source_cols = tuple(
            column.strip().strip('"')
            for column in match.group("source_cols").split(",")
            if column.strip()
        )
        parsed[match.group("name")] = (source_cols, match.group("ondelete"))
    return parsed


def test_upgrade_binds_each_foreign_keys_ondelete_to_the_correct_column():
    source = inspect.getsource(_migration_module().upgrade)
    foreign_keys = _parsed_foreign_keys(source)

    assert foreign_keys == {
        "fk_model_usage_events_user": (("user_id",), "CASCADE"),
        "fk_model_usage_events_conversation": (("conversation_id",), "SET NULL"),
        "fk_model_usage_events_request_message": (("request_message_id",), "SET NULL"),
        "fk_model_usage_events_document": (("document_id",), "SET NULL"),
        "fk_model_usage_minute_user": (("user_id",), "CASCADE"),
        "fk_model_usage_minute_conversation": (("conversation_id",), "CASCADE"),
    }


def test_upgrade_creates_indexes_after_tables():
    source = inspect.getsource(_migration_module().upgrade)

    events_table_pos = source.index('"model_usage_events"')
    minute_table_pos = source.index('"model_usage_minute"')
    first_index_pos = source.index("op.create_index")

    assert first_index_pos > events_table_pos
    assert first_index_pos > minute_table_pos


def test_downgrade_drops_rollups_before_events():
    source = inspect.getsource(_migration_module().downgrade)

    minute_drop_pos = source.index('drop_table("model_usage_minute")')
    events_drop_pos = source.index('drop_table("model_usage_events")')

    assert minute_drop_pos < events_drop_pos


def _drop_scratch_schema(connection) -> None:
    connection.execute(text(f'DROP SCHEMA IF EXISTS "{_SCRATCH_SCHEMA}" CASCADE'))
    connection.commit()


def test_migration_upgrades_and_downgrades_against_postgres():
    """Actually run this migration's upgrade/downgrade against real PostgreSQL.

    A pure-Python schema-contract/source-inspection test (the rest of this
    file) cannot catch DDL-emission failures such as PostgreSQL's 63-char
    identifier limit — that class of bug only surfaces when Postgres itself
    parses the generated DDL. This test executes the migration's actual
    `upgrade()`/`downgrade()` functions against a real connection via
    `alembic.operations.Operations`, so identifier-length and
    constraint-syntax errors surface here instead of only against the
    shared dev database.

    Requires TEST_DATABASE_URL; skipped otherwise, mirroring
    tests/integration/test_conversation_compaction_postgres.py.

    Runs in a dedicated, dropped-and-recreated Postgres *schema*
    (`_SCRATCH_SCHEMA`) rather than through Alembic's `command.upgrade`/
    `alembic_version` bookkeeping: the shared TEST_DATABASE_URL database was
    found to have no alembic_version row at all (its tables are seeded ad
    hoc by other integration tests, e.g.
    tests/integration/test_conversation_compaction_postgres.py), so treating
    it as sitting at any particular revision would be false, and replaying
    the entire migration history just for this test would make the test
    depend on ~38 unrelated migrations. The scratch schema is seeded with
    only the four tables this migration's foreign keys target (users,
    conversations, messages, documents), then dropped again at the end —
    idempotent across reruns because it is unconditionally dropped both
    before and after.
    """
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL migration execution tests")

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            _drop_scratch_schema(connection)
            connection.execute(text(f'CREATE SCHEMA "{_SCRATCH_SCHEMA}"'))
            connection.execute(text(f'SET search_path TO "{_SCRATCH_SCHEMA}"'))
            connection.commit()

            try:
                Base.metadata.create_all(
                    connection,
                    tables=[
                        User.__table__,
                        Conversation.__table__,
                        Message.__table__,
                        Document.__table__,
                    ],
                )
                connection.commit()

                migration_context = MigrationContext.configure(connection)
                with Operations.context(migration_context):
                    _migration_module().upgrade()
                connection.commit()

                inspector = sa_inspect(connection)
                tables = set(inspector.get_table_names(schema=_SCRATCH_SCHEMA))
                assert "model_usage_events" in tables
                assert "model_usage_minute" in tables

                with Operations.context(migration_context):
                    _migration_module().downgrade()
                connection.commit()

                inspector = sa_inspect(connection)
                tables_after = set(inspector.get_table_names(schema=_SCRATCH_SCHEMA))
                assert "model_usage_events" not in tables_after
                assert "model_usage_minute" not in tables_after
            finally:
                # Restore Postgres's default search_path before returning this
                # connection to the pool — it may be a pooled physical
                # connection reused by a later checkout, and SET (without
                # LOCAL) persists at the session level past our commits.
                connection.execute(text('SET search_path TO "$user", public'))
                connection.commit()
                _drop_scratch_schema(connection)
    finally:
        engine.dispose()
