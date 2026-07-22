"""Live smoke test for the complete Alembic history on empty PostgreSQL."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL, make_url

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRATCH_DATABASE_PREFIX = "chatbot_migration_smoke_"
_SCRATCH_DATABASE_RE = re.compile(r"chatbot_migration_smoke_[0-9a-f]{32}")
_HEAD = "a4b5c6d7e8f9"
_PREVIOUS_HEAD = "z3a4b5c6d7e8"
_PRE_RECONCILIATION_HEAD = "1ce64a959f7d"
_RECONCILIATION_REVISION = "6c6598a9eb26"
_TIMESTAMP_INDEXES = {
    "ix_model_usage_events_started_at",
    "ix_model_usage_minute_bucket_start_utc",
}


def _require_valid_scratch_database_name(name: str) -> str:
    if not _SCRATCH_DATABASE_RE.fullmatch(name):
        raise ValueError("refusing to operate on an invalid scratch database name")
    return name


def _validated_scratch_database_name() -> str:
    return _require_valid_scratch_database_name(f"{_SCRATCH_DATABASE_PREFIX}{uuid4().hex}")


def _database_url(base_url: URL, database: str) -> URL:
    return base_url.set(database=database)


def _postgres_test_url() -> URL:
    test_database_url = os.getenv("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL migration tests")
    base_url = make_url(test_database_url)
    if not base_url.drivername.startswith("postgresql"):
        pytest.skip("PostgreSQL is required for migration tests")
    return base_url


@contextmanager
def _scratch_database(base_url: URL) -> Iterator[URL]:
    scratch_database = _validated_scratch_database_name()
    if scratch_database == base_url.database:
        raise ValueError("scratch database must differ from TEST_DATABASE_URL")
    scratch_url = _database_url(base_url, scratch_database)
    admin_engine = create_engine(
        _database_url(base_url, "postgres"),
        isolation_level="AUTOCOMMIT",
    )
    created = False
    try:
        with admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{scratch_database}"'))
            created = True
        yield scratch_url
    finally:
        try:
            if created:
                _require_valid_scratch_database_name(scratch_database)
                with admin_engine.connect() as connection:
                    connection.execute(
                        text(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                            "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                        ),
                        {"database_name": scratch_database},
                    )
                    connection.execute(text(f'DROP DATABASE IF EXISTS "{scratch_database}"'))
        finally:
            admin_engine.dispose()


def _run_alembic(scratch_url: URL, *args: str) -> None:
    __tracebackhide__ = True
    env = dict(os.environ)
    env["DATABASE_URL"] = scratch_url.render_as_string(hide_password=False)
    try:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "alembic", *args],
                cwd=_PROJECT_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = _redact_database_credentials(
                f"{exc.stdout or ''}\n{exc.stderr or ''}", scratch_url
            )
            pytest.fail(f"Alembic {' '.join(args)} timed out:\n{output}")
    finally:
        env["DATABASE_URL"] = "<redacted>"
    if result.returncode:
        output = _redact_database_credentials(f"{result.stdout}\n{result.stderr}", scratch_url)
        pytest.fail(f"Alembic {' '.join(args)} failed:\n{output}")


def _redact_database_credentials(output: str, database_url: URL) -> str:
    __tracebackhide__ = True
    redacted = output.replace(
        database_url.render_as_string(hide_password=False),
        "<scratch-database-url>",
    )
    if database_url.password:
        for secret in {database_url.password, quote(database_url.password, safe="")}:
            redacted = redacted.replace(secret, "<database-password>")
    return redacted


def _assert_head_schema(scratch_url: URL) -> None:
    engine = create_engine(scratch_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalars().all() == [_HEAD]

            schema = inspect(connection)
            tables = set(schema.get_table_names())
            assert {
                "client_devices",
                "tool_approval_settings",
                "model_usage_events",
                "model_usage_minute",
            } <= tables

            settings_columns = {
                column["name"] for column in schema.get_columns("tool_approval_settings")
            }
            assert {"device_id", "tool_origin"} <= settings_columns

            settings_indexes = {
                index["name"] for index in schema.get_indexes("tool_approval_settings")
            }
            assert {
                "ix_tool_approval_settings_device_id",
                "ix_tool_approval_settings_tool_origin",
            } <= settings_indexes

            event_indexes = {
                index["name"]: index["column_names"]
                for index in schema.get_indexes("model_usage_events")
            }
            minute_indexes = {
                index["name"]: index["column_names"]
                for index in schema.get_indexes("model_usage_minute")
            }
            assert event_indexes["ix_model_usage_events_started_at"] == ["started_at"]
            assert minute_indexes["ix_model_usage_minute_bucket_start_utc"] == ["bucket_start_utc"]

            validity = dict(
                connection.execute(
                    text(
                        "SELECT c.relname, i.indisvalid "
                        "FROM pg_index i "
                        "JOIN pg_class c ON c.oid = i.indexrelid "
                        "JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = current_schema() "
                        "AND c.relname = ANY(:index_names)"
                    ),
                    {"index_names": sorted(_TIMESTAMP_INDEXES)},
                ).all()
            )
            assert validity == {name: True for name in _TIMESTAMP_INDEXES}
    finally:
        engine.dispose()


def _assert_previous_head_schema(scratch_url: URL) -> None:
    engine = create_engine(scratch_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalars().all() == [_PREVIOUS_HEAD]
            schema = inspect(connection)
            tables = set(schema.get_table_names())
            assert {
                "client_devices",
                "tool_approval_settings",
                "model_usage_events",
                "model_usage_minute",
            } <= tables
            assert {"device_id", "tool_origin"} <= {
                column["name"] for column in schema.get_columns("tool_approval_settings")
            }
            assert {
                "ix_tool_approval_settings_device_id",
                "ix_tool_approval_settings_tool_origin",
            } <= {index["name"] for index in schema.get_indexes("tool_approval_settings")}
            all_indexes = {
                index["name"]
                for table in ("model_usage_events", "model_usage_minute")
                for index in schema.get_indexes(table)
            }
            assert _TIMESTAMP_INDEXES.isdisjoint(all_indexes)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "invalid_name",
    [
        "chatbot_test",
        "chatbot_migration_smoke_",
        "chatbot_migration_smoke_not_hex",
        "chatbot_migration_smoke_" + "a" * 32 + '"',
    ],
)
def test_scratch_database_name_validation_rejects_non_generated_names(
    invalid_name: str,
) -> None:
    with pytest.raises(ValueError, match="invalid scratch database name"):
        _require_valid_scratch_database_name(invalid_name)


def test_database_output_redaction_removes_url_and_password_forms() -> None:
    database_url = make_url(
        "postgresql://migration_user:p%40ss@example.invalid/"
        "chatbot_migration_smoke_0123456789abcdef0123456789abcdef"
    )
    output = f"url={database_url.render_as_string(hide_password=False)} raw=p@ss encoded=p%40ss"

    redacted = _redact_database_credentials(output, database_url)

    assert "migration_user" not in redacted
    assert "p@ss" not in redacted
    assert "p%40ss" not in redacted


def test_full_alembic_chain_from_empty_postgres_database() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", "head")
        _assert_head_schema(scratch_url)

        _run_alembic(scratch_url, "downgrade", _PREVIOUS_HEAD)
        _assert_previous_head_schema(scratch_url)

        _run_alembic(scratch_url, "upgrade", "head")
        _assert_head_schema(scratch_url)


def test_reconciliation_migration_round_trips_the_canonical_schema() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _PRE_RECONCILIATION_HEAD)
        external_tables = {
            "checkpoints",
            "checkpoint_writes",
            "checkpoint_blobs",
            "mcp_oauth_tokens",
            "checkpoint_migrations",
        }
        seed_engine = create_engine(scratch_url)
        try:
            with seed_engine.begin() as connection:
                for table_name in external_tables:
                    connection.execute(
                        text(f'CREATE TABLE "{table_name}" (sentinel integer PRIMARY KEY)')
                    )
                    connection.execute(text(f'INSERT INTO "{table_name}" VALUES (1)'))
        finally:
            seed_engine.dispose()

        _run_alembic(scratch_url, "upgrade", _RECONCILIATION_REVISION)
        _run_alembic(scratch_url, "downgrade", _PRE_RECONCILIATION_HEAD)

        engine = create_engine(scratch_url)
        try:
            with engine.connect() as connection:
                assert connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalars().all() == [_PRE_RECONCILIATION_HEAD]
                schema = inspect(connection)
                tables = set(schema.get_table_names())
                assert "tool_approvals" in tables
                assert external_tables <= tables
                for table_name in external_tables:
                    assert connection.scalar(text(f'SELECT sentinel FROM "{table_name}"')) == 1
                assert "file_size" not in {
                    column["name"] for column in schema.get_columns("documents")
                }
                assert {"extra_metadata", "citations"} <= {
                    column["name"] for column in schema.get_columns("messages")
                }
                assert {
                    "estimated_duration_minutes",
                    "started_at",
                    "completion_confidence",
                    "retry_count",
                    "actual_duration_minutes",
                } <= {column["name"] for column in schema.get_columns("task_plans")}

                indexes = {
                    table: {index["name"] for index in schema.get_indexes(table)}
                    for table in (
                        "agent_model_configs",
                        "document_images",
                        "hitl_interrupts",
                        "model_providers",
                        "task_plans",
                    )
                }
                assert "ix_agent_model_configs_id" not in indexes["agent_model_configs"]
                assert "ix_agent_model_configs_user_id" not in indexes["agent_model_configs"]
                assert "idx_document_images_document_id" in indexes["document_images"]
                assert "ix_document_images_document_id" not in indexes["document_images"]
                assert {
                    "ix_hitl_interrupts_assistant_message_id",
                    "ix_hitl_interrupts_status_expires_at",
                } <= indexes["hitl_interrupts"]
                assert "ix_model_providers_id" not in indexes["model_providers"]
                assert "ix_model_providers_user_id" not in indexes["model_providers"]
                assert {
                    "ix_task_plans_conversation_order",
                    "ix_task_plans_status",
                } <= indexes["task_plans"]
                assert "idx_task_plan_conversation_order" not in indexes["task_plans"]

                assert any(
                    foreign_key["constrained_columns"] == ["assistant_message_id"]
                    and foreign_key["referred_table"] == "messages"
                    and foreign_key["referred_columns"] == ["id"]
                    and foreign_key["referred_schema"] in (None, "public")
                    for foreign_key in schema.get_foreign_keys("hitl_interrupts")
                )
        finally:
            engine.dispose()
