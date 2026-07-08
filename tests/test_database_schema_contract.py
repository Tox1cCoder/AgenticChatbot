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


def test_no_unexpired_pending_hitl_interrupts_before_legacy_checkpoint_cleanup():
    with _engine().connect() as conn:
        count = conn.execute(
            text(
                "SELECT COUNT(*) FROM hitl_interrupts "
                "WHERE status = 'pending' AND expires_at > now()"
            )
        ).scalar_one()
    assert count == 0
