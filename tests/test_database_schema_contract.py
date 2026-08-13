from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from app.core.config import settings
from app.models import *  # noqa: F401,F403
from app.models.base import Base

EXTERNAL_TABLES = {
    "alembic_version",
    "checkpoint_blobs",
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
}


def _engine():
    if not settings.database_url.startswith("postgresql"):
        pytest.skip("schema contract requires PostgreSQL")
    return create_engine(settings.database_url)


def test_live_database_has_no_unmodeled_app_tables():
    inspector = inspect(_engine())
    live_tables = set(inspector.get_table_names(schema="public"))
    model_tables = set(Base.metadata.tables)

    unmodeled = live_tables - model_tables - EXTERNAL_TABLES

    assert unmodeled == set()


def test_conversation_device_bindings_is_not_present():
    inspector = inspect(_engine())
    assert "conversation_device_bindings" not in inspector.get_table_names(schema="public")


def test_no_redundant_primary_key_indexes_on_app_tables():
    inspector = inspect(_engine())
    offenders: dict[str, list[str]] = {}
    for table_name in sorted(set(Base.metadata.tables)):
        pk_cols = set(
            inspector.get_pk_constraint(table_name, schema="public").get("constrained_columns")
            or []
        )
        redundant = []
        for index in inspector.get_indexes(table_name, schema="public"):
            cols = index.get("column_names") or []
            if len(cols) == 1 and cols[0] in pk_cols:
                redundant.append(str(index.get("name")))
        if redundant:
            offenders[table_name] = redundant

    assert offenders == {}


def test_checkpoint_tables_are_langgraph_owned_not_model_owned():
    model_tables = set(Base.metadata.tables)
    assert model_tables.isdisjoint(
        {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
    )


def test_document_index_generation_schema_enforces_one_active_generation():
    from app.models.document_chunk import DocumentChunk
    from app.models.document_index_generation import DocumentIndexGeneration

    generation_columns = DocumentIndexGeneration.__table__.columns
    assert {
        "id",
        "document_id",
        "status",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "chunking_version",
        "created_at",
        "activated_at",
        "retired_at",
        "failed_at",
        "failure_code",
    }.issubset(generation_columns.keys())
    active_index = next(
        index
        for index in DocumentIndexGeneration.__table__.indexes
        if index.name == "uq_document_index_generation_active"
    )
    assert active_index.unique is True
    assert "status = 'active'" in str(active_index.dialect_options["postgresql"]["where"])

    chunk_columns = DocumentChunk.__table__.columns
    assert chunk_columns.index_generation_id.nullable is False
    assert chunk_columns.index_generation_id.foreign_keys
    unique_column_sets = {
        tuple(column.name for column in constraint.columns)
        for constraint in DocumentChunk.__table__.constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    }
    assert ("document_id", "index_generation_id", "chunk_index") in unique_column_sets


def test_generation_migration_builds_partial_index_and_backfills_before_not_null(
    monkeypatch,
):
    from app.alembic.versions import c3d4e5f6a7b8_add_document_index_generations as migration

    calls: list[tuple[str, tuple, dict]] = []

    class _Result:
        def fetchall(self):
            return []

    class _Bind:
        def execute(self, *_args, **_kwargs):
            return _Result()

    for name in (
        "create_table",
        "create_index",
        "add_column",
        "alter_column",
        "create_foreign_key",
        "drop_constraint",
        "create_unique_constraint",
    ):
        monkeypatch.setattr(
            migration.op,
            name,
            lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)),
        )
    monkeypatch.setattr(migration.op, "get_bind", _Bind)

    migration.upgrade()

    partial_index = next(
        call
        for call in calls
        if call[0] == "create_index"
        and call[1][0] == "uq_document_index_generation_active"
    )
    assert partial_index[2]["unique"] is True
    assert str(partial_index[2]["postgresql_where"]) == "status = 'active'"
    add_position = next(
        index
        for index, call in enumerate(calls)
        if call[0] == "add_column" and call[1][0] == "document_chunks"
    )
    not_null_position = next(
        index
        for index, call in enumerate(calls)
        if call[0] == "alter_column" and call[1][1] == "index_generation_id"
    )
    assert add_position < not_null_position
    assert any(
        call[0] == "create_unique_constraint"
        and call[1][0] == "uq_document_chunk_generation_index"
        and call[1][2] == ["document_id", "index_generation_id", "chunk_index"]
        for call in calls
    )


def test_no_unexpired_pending_hitl_interrupts_before_legacy_checkpoint_cleanup():
    with _engine().connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM hitl_interrupts "
                "WHERE status = 'pending' AND expires_at > now()"
            )
        ).scalar_one()
    assert count == 0
