from __future__ import annotations

import importlib
import inspect
import os

import pytest
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy import inspect as sa_inspect

from app.models.model_usage import ModelUsageEvent, ModelUsageMinute


def migration_module():
    return importlib.import_module(
        "app.alembic.versions.a4b5c6d7e8f9_add_model_usage_timestamp_indexes"
    )


def test_timestamp_indexes_are_standalone_leading_indexes_in_orm():
    event_index = next(
        index
        for index in ModelUsageEvent.__table__.indexes
        if index.name == "ix_model_usage_events_started_at"
    )
    minute_index = next(
        index
        for index in ModelUsageMinute.__table__.indexes
        if index.name == "ix_model_usage_minute_bucket_start_utc"
    )

    assert [column.name for column in event_index.columns] == ["started_at"]
    assert [column.name for column in minute_index.columns] == ["bucket_start_utc"]
    assert len(event_index.name) <= 63
    assert len(minute_index.name) <= 63


def test_timestamp_index_migration_is_forward_only_and_concurrent():
    migration = migration_module()
    upgrade = inspect.getsource(migration.upgrade)
    downgrade = inspect.getsource(migration.downgrade)

    assert migration.down_revision == "z3a4b5c6d7e8"
    assert "autocommit_block" in upgrade
    assert "postgresql_concurrently=True" in upgrade
    assert "autocommit_block" in downgrade
    assert "postgresql_concurrently=True" in downgrade
    assert "ix_model_usage_events_started_at" in upgrade
    assert "ix_model_usage_minute_bucket_start_utc" in upgrade


def test_timestamp_index_migration_upgrade_downgrade_on_postgres():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL migration tests")
    schema = "model_usage_timestamp_index_migration_test"
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            connection.execute(text(f'SET search_path TO "{schema}"'))
            connection.execute(
                text(
                    "CREATE TABLE model_usage_events "
                    "(id uuid PRIMARY KEY, started_at timestamptz NOT NULL)"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE model_usage_minute "
                    "(rollup_key varchar(64) PRIMARY KEY, bucket_start_utc timestamptz NOT NULL)"
                )
            )
            connection.commit()
            context = MigrationContext.configure(connection)
            try:
                with Operations.context(context):
                    migration_module().upgrade()
                indexes = {
                    index["name"]
                    for table in ("model_usage_events", "model_usage_minute")
                    for index in sa_inspect(connection).get_indexes(table, schema=schema)
                }
                assert "ix_model_usage_events_started_at" in indexes
                assert "ix_model_usage_minute_bucket_start_utc" in indexes
                connection.execute(text("SET LOCAL enable_seqscan = off"))
                event_plan = "\n".join(
                    row[0]
                    for row in connection.execute(
                        text(
                            "EXPLAIN SELECT id FROM model_usage_events "
                            "WHERE started_at >= now() - interval '1 day'"
                        )
                    )
                )
                minute_plan = "\n".join(
                    row[0]
                    for row in connection.execute(
                        text(
                            "EXPLAIN SELECT rollup_key FROM model_usage_minute "
                            "WHERE bucket_start_utc >= now() - interval '1 day'"
                        )
                    )
                )
                assert "ix_model_usage_events_started_at" in event_plan
                assert "ix_model_usage_minute_bucket_start_utc" in minute_plan
                # Inspector queries autobegin a SQLAlchemy transaction; end it
                # before Alembic enters the next PostgreSQL autocommit block.
                connection.commit()

                with Operations.context(context):
                    migration_module().downgrade()
                remaining = {
                    index["name"]
                    for table in ("model_usage_events", "model_usage_minute")
                    for index in sa_inspect(connection).get_indexes(table, schema=schema)
                }
                assert "ix_model_usage_events_started_at" not in remaining
                assert "ix_model_usage_minute_bucket_start_utc" not in remaining
            finally:
                connection.execute(text('SET search_path TO "$user", public'))
                connection.commit()
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                connection.commit()
    finally:
        engine.dispose()
