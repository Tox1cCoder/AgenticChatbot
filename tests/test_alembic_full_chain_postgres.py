"""Live smoke test for the complete Alembic history on empty PostgreSQL."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL, make_url

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRATCH_DATABASE_PREFIX = "chatbot_migration_smoke_"
_SCRATCH_DATABASE_RE = re.compile(r"chatbot_migration_smoke_[0-9a-f]{32}")
_OLD_HEAD = "a4b5c6d7e8f9"
_HEAD = "b8c9d0e1f2a3"
_PREVIOUS_HEAD = "b2c3d4e5f6a7"
_PRE_RECONCILIATION_HEAD = "1ce64a959f7d"
_PARALLEL_ALLOW_CUSTOM_MODEL_HEAD = "0f1e2d3c4b5a"
_RECONCILIATION_REVISION = "6c6598a9eb26"
_TIMESTAMP_INDEXES = {
    "ix_model_usage_events_started_at",
    "ix_model_usage_minute_bucket_start_utc",
}


def _normalized_metadata(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _normalized_metadata(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_metadata(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_normalized_metadata(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _table_schema_snapshot(connection, table_name: str) -> dict:
    schema = inspect(connection)
    indexes = []
    for index in schema.get_indexes(table_name, schema="public"):
        validity = connection.scalar(
            text(
                "SELECT i.indisvalid FROM pg_index i "
                "WHERE i.indexrelid = to_regclass(:qualified_name)"
            ),
            {"qualified_name": f"public.{index['name']}"},
        )
        indexes.append(
            {
                "name": index["name"],
                "columns": tuple(index.get("column_names") or ()),
                "expressions": tuple(str(item) for item in index.get("expressions") or ()),
                "unique": bool(index.get("unique")),
                "dialect_options": _normalized_metadata(index.get("dialect_options") or {}),
                "valid": validity,
            }
        )

    return {
        "comment": _normalized_metadata(
            schema.get_table_comment(table_name, schema="public").get("text")
        ),
        "columns": tuple(
            (
                column["name"],
                str(column["type"]),
                column["nullable"],
                _normalized_metadata(column.get("default")),
                _normalized_metadata(column.get("identity")),
                _normalized_metadata(column.get("computed")),
                _normalized_metadata(column.get("comment")),
            )
            for column in schema.get_columns(table_name, schema="public")
        ),
        "primary_key": _normalized_metadata(schema.get_pk_constraint(table_name, schema="public")),
        "foreign_keys": tuple(
            sorted(
                _normalized_metadata(item)
                for item in schema.get_foreign_keys(table_name, schema="public")
            )
        ),
        "unique_constraints": tuple(
            sorted(
                _normalized_metadata(item)
                for item in schema.get_unique_constraints(table_name, schema="public")
            )
        ),
        "check_constraints": tuple(
            sorted(
                _normalized_metadata(item)
                for item in schema.get_check_constraints(table_name, schema="public")
            )
        ),
        "indexes": tuple(sorted(indexes, key=lambda item: item["name"])),
    }


def _public_table_schema_snapshot(connection, excluded_tables: set[str] | None = None) -> dict:
    excluded_tables = excluded_tables or set()
    return {
        table_name: _table_schema_snapshot(connection, table_name)
        for table_name in sorted(inspect(connection).get_table_names(schema="public"))
        if table_name not in excluded_tables
    }


def _without_indexes(schema_snapshot: dict, index_names: set[str]) -> dict:
    expected = deepcopy(schema_snapshot)
    for table in expected.values():
        table["indexes"] = tuple(
            index for index in table["indexes"] if index["name"] not in index_names
        )
    return expected


def _without_tables(schema_snapshot: dict, table_names: set[str]) -> dict:
    return {
        table_name: snapshot
        for table_name, snapshot in schema_snapshot.items()
        if table_name not in table_names
    }


def _decision_type_labels(connection) -> list[str]:
    return (
        connection.execute(
            text(
                "SELECT e.enumlabel FROM pg_enum e "
                "JOIN pg_type t ON t.oid = e.enumtypid "
                "JOIN pg_namespace n ON n.oid = t.typnamespace "
                "WHERE n.nspname = 'public' AND t.typname = 'decision_type' "
                "ORDER BY e.enumsortorder"
            )
        )
        .scalars()
        .all()
    )


def _uppercase_decision_type(connection, include_respond: bool) -> None:
    labels = ["accept", "edit", "reject"]
    if include_respond:
        labels.append("respond")
    for label in labels:
        connection.execute(
            text(f"ALTER TYPE decision_type RENAME VALUE '{label}' TO '{label.upper()}'")
        )


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
    __tracebackhide__ = True
    scratch_database = _validated_scratch_database_name()
    if scratch_database == base_url.database:
        raise ValueError("scratch database must differ from TEST_DATABASE_URL")
    scratch_url = _database_url(base_url, scratch_database)
    try:
        admin_engine = create_engine(
            _database_url(base_url, "postgres"),
            isolation_level="AUTOCOMMIT",
        )
    except Exception as exc:
        raise _redacted_database_error(
            "creating the PostgreSQL admin engine", exc, base_url
        ) from None
    created = False
    try:
        try:
            with admin_engine.connect() as connection:
                connection.execute(text(f'CREATE DATABASE "{scratch_database}"'))
                created = True
        except Exception as exc:
            raise _redacted_database_error("creating the scratch database", exc, base_url) from None
        yield scratch_url
    finally:
        try:
            if created:
                _require_valid_scratch_database_name(scratch_database)
                try:
                    with admin_engine.connect() as connection:
                        connection.execute(
                            text(
                                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                                "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                            ),
                            {"database_name": scratch_database},
                        )
                        connection.execute(text(f'DROP DATABASE IF EXISTS "{scratch_database}"'))
                except Exception as exc:
                    raise _redacted_database_error(
                        "dropping the scratch database", exc, base_url
                    ) from None
        finally:
            try:
                admin_engine.dispose()
            except Exception as exc:
                raise _redacted_database_error(
                    "disposing the PostgreSQL admin engine", exc, base_url
                ) from None


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
    redacted = re.sub(
        r"postgresql(?:\+[A-Za-z0-9_.-]+)?://[^\s]+",
        "<scratch-database-url>",
        output,
        flags=re.IGNORECASE,
    )
    rendered_urls = {
        database_url.render_as_string(hide_password=False),
        database_url.render_as_string(hide_password=True),
        str(database_url),
    }
    for rendered_url in sorted(rendered_urls, key=len, reverse=True):
        redacted = redacted.replace(rendered_url, "<scratch-database-url>")
    credential_variants: list[tuple[str, str]] = []
    for credential, replacement in (
        (database_url.password, "<database-password>"),
        (database_url.username, "<database-username>"),
    ):
        if credential:
            credential_variants.extend(
                (secret, replacement) for secret in {credential, quote(credential, safe="")}
            )
    for secret, replacement in sorted(
        credential_variants,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        redacted = redacted.replace(secret, replacement)
    return redacted


def _redacted_database_error(action: str, error: Exception, database_url: URL) -> RuntimeError:
    """Build an exception whose message cannot expose database credentials."""
    message = _redact_database_credentials(str(error), database_url)
    return RuntimeError(f"{action} failed: {message}")


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
                "web_image_references",
                "document_index_generations",
            } <= tables
            assert "index_generation_id" in {
                column["name"] for column in schema.get_columns("document_chunks")
            }
            generation_indexes = {
                index["name"]: index for index in schema.get_indexes("document_index_generations")
            }
            assert generation_indexes["uq_document_index_generation_active"]["unique"] is True
            chunk_indexes = {
                index["name"] for index in schema.get_indexes("document_chunks")
            }
            assert "idx_document_chunks_content_simple_fts" in chunk_indexes

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

            event_foreign_keys = {
                foreign_key["name"]: foreign_key
                for foreign_key in schema.get_foreign_keys("model_usage_events")
            }
            assert (
                event_foreign_keys["fk_model_usage_events_conversation"]["options"]["ondelete"]
                == "CASCADE"
            )

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
                "web_image_references",
            } <= tables
            assert "document_index_generations" not in tables
            assert "index_generation_id" not in {
                column["name"] for column in schema.get_columns("document_chunks")
            }
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
            # a4b5c6d7e8f9 creates these and is an ancestor of _PREVIOUS_HEAD, so a
            # downgrade to that revision keeps them.
            assert all_indexes >= _TIMESTAMP_INDEXES
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
        "postgresql://migration%40user:p%40ss@example.invalid/"
        "chatbot_migration_smoke_0123456789abcdef0123456789abcdef"
    )
    output = (
        f"url={database_url.render_as_string(hide_password=False)} "
        f"masked={database_url.render_as_string(hide_password=True)} "
        "raw_user=migration@user encoded_user=migration%40user "
        "raw_password=p@ss encoded_password=p%40ss"
    )

    redacted = _redact_database_credentials(output, database_url)

    assert "migration@user" not in redacted
    assert "migration%40user" not in redacted
    assert "p@ss" not in redacted
    assert "p%40ss" not in redacted
    assert "***" not in redacted


def test_alembic_failure_redacts_stdout_stderr_and_environment(monkeypatch) -> None:
    database_url = make_url(
        "postgresql://migration%40user:p%40ss@example.invalid/"
        "chatbot_migration_smoke_0123456789abcdef0123456789abcdef"
    )
    captured_env = None

    def failed_run(*_args, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        return subprocess.CompletedProcess(
            args=["alembic"],
            returncode=1,
            stdout="raw migration@user p@ss",
            stderr=(
                f"masked {database_url.render_as_string(hide_password=True)} "
                "encoded migration%40user p%40ss"
            ),
        )

    monkeypatch.setattr(subprocess, "run", failed_run)
    with pytest.raises(pytest.fail.Exception) as failure:
        _run_alembic(database_url, "upgrade", "head")

    message = str(failure.value)
    for secret in ("migration@user", "migration%40user", "p@ss", "p%40ss", "***"):
        assert secret not in message
    assert captured_env["DATABASE_URL"] == "<redacted>"


def test_alembic_timeout_redacts_captured_output(monkeypatch) -> None:
    database_url = make_url(
        "postgresql://migration%40user:p%40ss@example.invalid/"
        "chatbot_migration_smoke_0123456789abcdef0123456789abcdef"
    )

    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["alembic"],
            timeout=180,
            output="raw migration@user p@ss",
            stderr="encoded migration%40user p%40ss",
        )

    monkeypatch.setattr(subprocess, "run", timed_out)
    with pytest.raises(pytest.fail.Exception) as failure:
        _run_alembic(database_url, "upgrade", "head")

    message = str(failure.value)
    for secret in ("migration@user", "migration%40user", "p@ss", "p%40ss"):
        assert secret not in message


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("token", "prefix-token-suffix"),
        ("prefix-token-suffix", "token"),
    ],
)
def test_redaction_replaces_longer_credential_before_contained_credential(
    username: str, password: str
) -> None:
    database_url = URL.create(
        "postgresql",
        username=username,
        password=password,
        host="example.invalid",
        database="chatbot_migration_smoke_0123456789abcdef0123456789abcdef",
    )

    redacted = _redact_database_credentials(
        f"raw_password={password} raw_username={username}",
        database_url,
    )

    assert redacted == ("raw_password=<database-password> raw_username=<database-username>")


def test_scratch_database_operation_errors_are_credential_safe() -> None:
    database_url = make_url(
        "postgresql://token:prefix-token-suffix@example.invalid/"
        "chatbot_migration_smoke_0123456789abcdef0123456789abcdef"
    )
    unsafe_error = RuntimeError(
        "connection postgresql://token:***@example.invalid failed for token prefix-token-suffix"
    )

    safe_error = _redacted_database_error("creating scratch database", unsafe_error, database_url)

    assert str(safe_error) == (
        "creating scratch database failed: connection <scratch-database-url> "
        "failed for <database-username> <database-password>"
    )


@pytest.mark.parametrize(
    ("failure_point", "expected_action"),
    [
        ("engine", "creating the PostgreSQL admin engine"),
        ("connect", "creating the scratch database"),
        ("drop", "dropping the scratch database"),
    ],
)
def test_scratch_database_exception_paths_are_redacted(
    monkeypatch, failure_point: str, expected_action: str
) -> None:
    database_url = make_url(
        "postgresql://token:prefix-token-suffix@example.invalid/source_database"
    )
    unsafe_error = RuntimeError(
        "postgresql://token:prefix-token-suffix@example.invalid/source_database "
        "token prefix-token-suffix"
    )

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, *_args, **_kwargs):
            return None

    class FakeEngine:
        connect_calls = 0

        def connect(self):
            self.connect_calls += 1
            if failure_point == "connect" or (failure_point == "drop" and self.connect_calls == 2):
                raise unsafe_error
            return FakeConnection()

        def dispose(self):
            return None

    def fake_create_engine(*_args, **_kwargs):
        if failure_point == "engine":
            raise unsafe_error
        return FakeEngine()

    monkeypatch.setattr(sys.modules[__name__], "create_engine", fake_create_engine)

    with pytest.raises(RuntimeError) as failure, _scratch_database(database_url):
        pass

    message = str(failure.value)
    assert expected_action in message
    assert failure.value.__suppress_context__
    for secret in ("token", "prefix-token-suffix", "postgresql://"):
        assert secret not in message


@pytest.mark.parametrize(
    "module_name",
    [
        "app.alembic.versions.6c6598a9eb26_create_missing_tool_approvals_table",
        "app.alembic.versions.b5c6d7e8f9a0_repair_tool_approvals_schema",
        "app.alembic.versions.a7b8c9d0e1f2_repair_document_index_generation_timestamps",
    ],
)
def test_inspection_migrations_fail_clearly_offline(monkeypatch, module_name: str) -> None:
    migration = import_module(module_name)
    monkeypatch.setattr(migration.op, "get_context", lambda: SimpleNamespace(as_sql=True))

    with pytest.raises(RuntimeError, match="online PostgreSQL connection"):
        migration.upgrade()
    if migration.revision == _RECONCILIATION_REVISION:
        with pytest.raises(RuntimeError, match="online PostgreSQL connection"):
            migration.downgrade()
    else:
        assert migration.downgrade() is None


def test_head_constant_matches_the_real_alembic_head() -> None:
    """``_HEAD`` and the README both hardcode the head revision.

    Without this check they drift together silently: the README assertion below
    compares documentation against the constant, not against Alembic, and the
    ``alembic_version`` assertion that would catch it only runs when a live
    PostgreSQL is configured.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script_directory = ScriptDirectory.from_config(Config(str(_PROJECT_ROOT / "alembic.ini")))

    assert script_directory.get_heads() == [_HEAD]


def test_readme_tracks_migration_head_and_current_graph_contract() -> None:
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_lower = readme.lower()

    assert f"currently `{_HEAD}`" in readme
    assert "legacy `summarize` node is kept" not in readme
    for warning in (
        "original `6c6598a9eb26`",
        "cannot reconstruct deleted checkpoint or MCP OAuth data",
        "restore the affected tables from a pre-upgrade backup",
        "users must reauthenticate affected MCP servers",
        "inspect these tables before upgrading",
    ):
        assert warning.lower() in readme_lower


def test_full_alembic_chain_from_empty_postgres_database() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", "head")
        _assert_head_schema(scratch_url)
        engine = create_engine(scratch_url)
        try:
            with engine.connect() as connection:
                head_snapshot = _public_table_schema_snapshot(connection)
        finally:
            engine.dispose()
        _run_alembic(scratch_url, "downgrade", _PREVIOUS_HEAD)
        _assert_previous_head_schema(scratch_url)
        engine = create_engine(scratch_url)
        try:
            with engine.connect() as connection:
                previous_snapshot = _public_table_schema_snapshot(connection)
        finally:
            engine.dispose()
        assert "document_index_generations" not in previous_snapshot

        _run_alembic(scratch_url, "upgrade", "head")
        _assert_head_schema(scratch_url)
        engine = create_engine(scratch_url)
        try:
            with engine.connect() as connection:
                assert _public_table_schema_snapshot(connection) == head_snapshot
        finally:
            engine.dispose()


def test_tool_approval_orm_round_trips_migrated_lowercase_enum() -> None:
    from sqlalchemy.orm import Session

    from app.models import Conversation, DecisionType, ToolApproval, User

    assert ToolApproval.__table__.c.decision.type.enums == [
        decision.value for decision in DecisionType
    ]

    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", "head")
        engine = create_engine(scratch_url)
        approval_id = uuid4()
        try:
            with Session(engine) as session:
                user = User(
                    username="orm-approval-user",
                    email="orm-approval@example.invalid",
                    password_hash="not-a-real-password",
                )
                session.add(user)
                session.flush()
                conversation = Conversation(owner_id=user.id, title="ORM approval round trip")
                session.add(conversation)
                session.flush()
                session.add(
                    ToolApproval(
                        id=approval_id,
                        conversation_id=conversation.id,
                        user_id=user.id,
                        interrupt_id="orm-round-trip",
                        tool_name="test_tool",
                        tool_call_id="orm-call",
                        original_args={"source": "orm"},
                        decision=DecisionType.RESPOND,
                    )
                )
                session.commit()
                session.expire_all()

                approval = session.get(ToolApproval, approval_id)
                assert approval is not None
                assert approval.decision is DecisionType.RESPOND

            with engine.connect() as connection:
                assert (
                    connection.scalar(
                        text("SELECT decision::text FROM tool_approvals WHERE id = :id"),
                        {"id": approval_id},
                    )
                    == "respond"
                )
        finally:
            engine.dispose()


def test_head_round_trips_below_reconciliation_with_schema_and_data_intact() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", "head")
        engine = create_engine(scratch_url)
        user_id = uuid4()
        conversation_id = uuid4()
        approval_id = uuid4()
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO users "
                        "(id, username, email, password_hash, created_at, updated_at) "
                        "VALUES (:id, 'deep-user', 'deep@example.invalid', "
                        "'not-a-real-password', now(), now())"
                    ),
                    {"id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO conversations "
                        "(id, owner_id, title, created_at, updated_at, planning_mode_enabled, "
                        "next_message_sequence) VALUES "
                        "(:id, :owner_id, 'deep', now(), now(), false, 1)"
                    ),
                    {"id": conversation_id, "owner_id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO tool_approvals "
                        "(id, conversation_id, user_id, interrupt_id, tool_name, tool_call_id, "
                        "original_args, decision) VALUES "
                        "(:id, :conversation_id, :user_id, 'deep', 'tool', 'call', "
                        "'{\"depth\": 1}'::jsonb, 'respond')"
                    ),
                    {
                        "id": approval_id,
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                    },
                )
                head_snapshot = _public_table_schema_snapshot(connection)

            _run_alembic(scratch_url, "downgrade", _PRE_RECONCILIATION_HEAD)
            with engine.connect() as connection:
                assert set(
                    connection.execute(text("SELECT version_num FROM alembic_version")).scalars()
                ) == {_PRE_RECONCILIATION_HEAD, _PARALLEL_ALLOW_CUSTOM_MODEL_HEAD}
                assert _decision_type_labels(connection) == [
                    "accept",
                    "edit",
                    "reject",
                    "respond",
                ]
                assert (
                    connection.scalar(
                        text("SELECT decision::text FROM tool_approvals WHERE id = :id"),
                        {"id": approval_id},
                    )
                    == "respond"
                )

            _run_alembic(scratch_url, "upgrade", "head")
            with engine.connect() as connection:
                assert _public_table_schema_snapshot(connection) == head_snapshot
                assert connection.execute(
                    text("SELECT decision::text, original_args FROM tool_approvals WHERE id = :id"),
                    {"id": approval_id},
                ).one() == ("respond", {"depth": 1})
        finally:
            engine.dispose()


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
        user_id = uuid4()
        conversation_id = uuid4()
        task_id = uuid4()
        seed_engine = create_engine(scratch_url)
        try:
            with seed_engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO users "
                        "(id, username, email, password_hash, created_at, updated_at) "
                        "VALUES (:id, 'migration-user', 'migration@example.invalid', "
                        "'not-a-real-password', now(), now())"
                    ),
                    {"id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO conversations "
                        "(id, owner_id, title, created_at, updated_at, planning_mode_enabled) "
                        "VALUES (:id, :owner_id, 'migration', now(), now(), false)"
                    ),
                    {"id": conversation_id, "owner_id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO task_plans "
                        "(id, conversation_id, task_order, description, status, "
                        "created_at, updated_at, estimated_duration_minutes, "
                        "actual_duration_minutes, retry_count, completion_confidence, started_at) "
                        "VALUES (:id, :conversation_id, 1, 'migration', 'pending', "
                        "now(), now(), 21, 13, 4, 0.75, now())"
                    ),
                    {"id": task_id, "conversation_id": conversation_id},
                )
                for table_name in external_tables:
                    connection.execute(
                        text(f'CREATE TABLE "{table_name}" (sentinel integer PRIMARY KEY)')
                    )
                    connection.execute(text(f'INSERT INTO "{table_name}" VALUES (1)'))
                predecessor_snapshot = _public_table_schema_snapshot(
                    connection, excluded_tables=external_tables
                )
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
                assert (
                    _public_table_schema_snapshot(connection, excluded_tables=external_tables)
                    == predecessor_snapshot
                )
                for table_name in external_tables:
                    assert connection.scalar(text(f'SELECT sentinel FROM "{table_name}"')) == 1
                restored_metrics = connection.execute(
                    text(
                        "SELECT estimated_duration_minutes, actual_duration_minutes, "
                        "retry_count, completion_confidence, started_at "
                        "FROM task_plans WHERE id = :id"
                    ),
                    {"id": task_id},
                ).one()
                # 6c intentionally retired these columns and their data. Its
                # downgrade restores the exact predecessor schema, but cannot
                # reconstruct values deleted by the forward migration.
                assert restored_metrics == (
                    None,
                    None,
                    0,
                    None,
                    None,
                )
        finally:
            engine.dispose()


@pytest.mark.parametrize("enum_state", ["canonical", "missing", "uppercase"])
def test_reconciliation_repairs_missing_tool_approvals_exactly(enum_state: str) -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _PRE_RECONCILIATION_HEAD)
        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                canonical_table = _table_schema_snapshot(connection, "tool_approvals")
                enum_labels = connection.execute(
                    text(
                        "SELECT e.enumlabel FROM pg_enum e "
                        "JOIN pg_type t ON t.oid = e.enumtypid "
                        "WHERE t.typname = 'decision_type' ORDER BY e.enumsortorder"
                    )
                ).scalars()
                assert enum_labels.all() == ["accept", "edit", "reject"]
                connection.execute(text("DROP TABLE tool_approvals"))
                if enum_state == "missing":
                    connection.execute(text("DROP TYPE decision_type"))
                elif enum_state == "uppercase":
                    _uppercase_decision_type(connection, include_respond=False)

            _run_alembic(scratch_url, "upgrade", _RECONCILIATION_REVISION)

            with engine.connect() as connection:
                assert _table_schema_snapshot(connection, "tool_approvals") == canonical_table
                repaired_labels = connection.execute(
                    text(
                        "SELECT e.enumlabel FROM pg_enum e "
                        "JOIN pg_type t ON t.oid = e.enumtypid "
                        "WHERE t.typname = 'decision_type' ORDER BY e.enumsortorder"
                    )
                ).scalars()
                assert repaired_labels.all() == ["accept", "edit", "reject"]
        finally:
            engine.dispose()


def test_reconciliation_canonicalizes_legacy_enum_across_downgrade_reupgrade() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _PRE_RECONCILIATION_HEAD)
        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                _uppercase_decision_type(connection, include_respond=False)

            _run_alembic(scratch_url, "upgrade", _RECONCILIATION_REVISION)
            with engine.connect() as connection:
                assert _decision_type_labels(connection) == ["accept", "edit", "reject"]

            _run_alembic(scratch_url, "downgrade", _PRE_RECONCILIATION_HEAD)
            _run_alembic(scratch_url, "upgrade", _RECONCILIATION_REVISION)
            with engine.connect() as connection:
                assert _decision_type_labels(connection) == ["accept", "edit", "reject"]
        finally:
            engine.dispose()


@pytest.mark.parametrize("drift", ["unknown-enum", "unsupported-dependent", "missing-column"])
def test_reconciliation_rejects_unsafe_historical_drift(drift: str) -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _PRE_RECONCILIATION_HEAD)
        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                if drift == "unknown-enum":
                    connection.execute(
                        text("ALTER TYPE decision_type RENAME VALUE 'accept' TO 'maybe'")
                    )
                elif drift == "unsupported-dependent":
                    connection.execute(
                        text("CREATE TABLE rogue_decisions (decision decision_type)")
                    )
                else:
                    connection.execute(text("ALTER TABLE tool_approvals DROP COLUMN original_args"))

            expected_error = (
                "unsupported decision_type labels"
                if drift == "unknown-enum"
                else (
                    "unsupported decision_type catalog dependencies"
                    if drift == "unsupported-dependent"
                    else "unsafe tool_approvals schema drift"
                )
            )
            with pytest.raises(pytest.fail.Exception, match=expected_error):
                _run_alembic(scratch_url, "upgrade", _RECONCILIATION_REVISION)
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                    _PRE_RECONCILIATION_HEAD
                )
        finally:
            engine.dispose()


def test_forward_repair_canonicalizes_stamped_legacy_enum_and_is_irreversible() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _OLD_HEAD)
        engine = create_engine(scratch_url)
        approval_id = uuid4()
        user_id = uuid4()
        conversation_id = uuid4()
        try:
            with engine.begin() as connection:
                canonical_table = _table_schema_snapshot(connection, "tool_approvals")
                _uppercase_decision_type(connection, include_respond=True)
                connection.execute(
                    text(
                        "INSERT INTO users "
                        "(id, username, email, password_hash, created_at, updated_at) "
                        "VALUES (:id, 'forward-user', 'forward@example.invalid', "
                        "'not-a-real-password', now(), now())"
                    ),
                    {"id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO conversations "
                        "(id, owner_id, title, created_at, updated_at, planning_mode_enabled, "
                        "next_message_sequence) VALUES "
                        "(:id, :owner_id, 'forward', now(), now(), false, 1)"
                    ),
                    {"id": conversation_id, "owner_id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO tool_approvals "
                        "(id, conversation_id, user_id, interrupt_id, tool_name, tool_call_id, "
                        "original_args, decision) VALUES "
                        "(:id, :conversation_id, :user_id, 'interrupt', 'tool', 'call', "
                        "'{}'::jsonb, 'ACCEPT')"
                    ),
                    {
                        "id": approval_id,
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                    },
                )

            _run_alembic(scratch_url, "upgrade", "head")
            with engine.connect() as connection:
                assert _decision_type_labels(connection) == [
                    "accept",
                    "edit",
                    "reject",
                    "respond",
                ]
                assert (
                    connection.scalar(
                        text("SELECT decision::text FROM tool_approvals WHERE id = :id"),
                        {"id": approval_id},
                    )
                    == "accept"
                )
                assert _table_schema_snapshot(connection, "tool_approvals") == canonical_table

            _run_alembic(scratch_url, "downgrade", _OLD_HEAD)
            with engine.connect() as connection:
                assert _decision_type_labels(connection) == [
                    "accept",
                    "edit",
                    "reject",
                    "respond",
                ]
            _run_alembic(scratch_url, "upgrade", "head")
        finally:
            engine.dispose()


def test_forward_repair_normalizes_original_6c_shape_without_rewriting_data() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _OLD_HEAD)
        engine = create_engine(scratch_url)
        user_id = uuid4()
        conversation_id = uuid4()
        approval_id = uuid4()
        try:
            with engine.begin() as connection:
                canonical_table = _table_schema_snapshot(connection, "tool_approvals")
                original_type_oid = connection.scalar(
                    text("SELECT 'public.decision_type'::regtype::oid")
                )
                for label in ("accept", "edit", "reject"):
                    connection.execute(
                        text(
                            f"ALTER TYPE public.decision_type RENAME VALUE '{label}' "
                            f"TO '{label.upper()}'"
                        )
                    )
                for column_name in ("created_at", "updated_at", "decided_at"):
                    connection.execute(
                        text(
                            "ALTER TABLE public.tool_approvals "
                            f"ALTER COLUMN {column_name} DROP DEFAULT"
                        )
                    )
                for constraint_name in (
                    "fk_tool_approvals_conversation_id",
                    "fk_tool_approvals_user_id",
                ):
                    connection.execute(
                        text(f"ALTER TABLE public.tool_approvals DROP CONSTRAINT {constraint_name}")
                    )
                connection.execute(
                    text(
                        "ALTER TABLE public.tool_approvals ADD FOREIGN KEY "
                        "(conversation_id) REFERENCES public.conversations(id)"
                    )
                )
                connection.execute(
                    text(
                        "ALTER TABLE public.tool_approvals ADD FOREIGN KEY "
                        "(user_id) REFERENCES public.users(id)"
                    )
                )
                legacy_fk_names = {
                    row[0]
                    for row in connection.execute(
                        text(
                            "SELECT conname FROM pg_constraint "
                            "WHERE conrelid = 'public.tool_approvals'::regclass "
                            "AND contype = 'f'"
                        )
                    )
                }
                assert {
                    "tool_approvals_conversation_id_fkey",
                    "tool_approvals_user_id_fkey",
                } <= legacy_fk_names
                connection.execute(
                    text(
                        "INSERT INTO users "
                        "(id, username, email, password_hash, created_at, updated_at) "
                        "VALUES (:id, 'legacy-user', 'legacy@example.invalid', "
                        "'not-a-real-password', now(), now())"
                    ),
                    {"id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO conversations "
                        "(id, owner_id, title, created_at, updated_at, planning_mode_enabled, "
                        "next_message_sequence) VALUES "
                        "(:id, :owner_id, 'legacy', now(), now(), false, 1)"
                    ),
                    {"id": conversation_id, "owner_id": user_id},
                )
                connection.execute(
                    text(
                        "INSERT INTO tool_approvals "
                        "(id, created_at, updated_at, conversation_id, user_id, interrupt_id, "
                        "tool_name, tool_call_id, original_args, decision, decided_at) VALUES "
                        "(:id, '2026-01-02T03:04:05Z', '2026-01-02T03:04:06Z', "
                        ":conversation_id, :user_id, 'legacy', 'tool', 'call', "
                        "'{\"legacy\": true}'::jsonb, 'ACCEPT', '2026-01-02T03:04:07Z')"
                    ),
                    {
                        "id": approval_id,
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                    },
                )
                legacy_row = connection.execute(
                    text("SELECT * FROM tool_approvals WHERE id = :id"),
                    {"id": approval_id},
                ).one()

            _run_alembic(scratch_url, "upgrade", "head")
            with engine.connect() as connection:
                assert _table_schema_snapshot(connection, "tool_approvals") == canonical_table
                assert (
                    connection.scalar(text("SELECT 'public.decision_type'::regtype::oid"))
                    == original_type_oid
                )
                repaired_row = connection.execute(
                    text("SELECT * FROM tool_approvals WHERE id = :id"),
                    {"id": approval_id},
                ).one()
                assert repaired_row._mapping["decision"] == "accept"
                assert {
                    key: value for key, value in repaired_row._mapping.items() if key != "decision"
                } == {key: value for key, value in legacy_row._mapping.items() if key != "decision"}
        finally:
            engine.dispose()


def test_forward_repair_rejects_not_valid_foreign_key_with_orphan() -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _OLD_HEAD)
        engine = create_engine(scratch_url)
        user_id = uuid4()
        orphan_conversation_id = uuid4()
        approval_id = uuid4()
        try:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO users "
                        "(id, username, email, password_hash, created_at, updated_at) "
                        "VALUES (:id, 'orphan-user', 'orphan@example.invalid', "
                        "'not-a-real-password', now(), now())"
                    ),
                    {"id": user_id},
                )
                connection.execute(
                    text(
                        "ALTER TABLE public.tool_approvals DROP CONSTRAINT "
                        "fk_tool_approvals_conversation_id"
                    )
                )
                connection.execute(
                    text(
                        "INSERT INTO tool_approvals "
                        "(id, conversation_id, user_id, interrupt_id, tool_name, tool_call_id, "
                        "original_args, decision) VALUES "
                        "(:id, :conversation_id, :user_id, 'orphan', 'tool', 'call', "
                        "'{}'::jsonb, 'accept')"
                    ),
                    {
                        "id": approval_id,
                        "conversation_id": orphan_conversation_id,
                        "user_id": user_id,
                    },
                )
                connection.execute(
                    text(
                        "ALTER TABLE public.tool_approvals ADD CONSTRAINT "
                        "fk_tool_approvals_conversation_id FOREIGN KEY (conversation_id) "
                        "REFERENCES public.conversations(id) NOT VALID"
                    )
                )

            with pytest.raises(
                pytest.fail.Exception,
                match="NOT VALID.*repair orphan rows.*VALIDATE CONSTRAINT",
            ):
                _run_alembic(scratch_url, "upgrade", "head")
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                    _OLD_HEAD
                )
                assert (
                    connection.scalar(
                        text("SELECT count(*) FROM tool_approvals WHERE id = :id"),
                        {"id": approval_id},
                    )
                    == 1
                )
        finally:
            engine.dispose()


@pytest.mark.parametrize(
    "dependency_ddl",
    [
        (
            "CREATE FUNCTION public.echo_decision(public.decision_type) "
            "RETURNS public.decision_type LANGUAGE sql IMMUTABLE AS 'SELECT $1'"
        ),
        (
            "CREATE FUNCTION public.echo_decision(public.decision_type[]) "
            "RETURNS public.decision_type[] LANGUAGE sql IMMUTABLE AS 'SELECT $1'"
        ),
        "CREATE DOMAIN public.decision_domain AS public.decision_type",
        ("CREATE VIEW public.decision_view AS SELECT decision FROM public.tool_approvals"),
    ],
    ids=["function", "array-function", "domain", "view"],
)
def test_forward_repair_rejects_enum_catalog_dependency_transactionally(
    dependency_ddl: str,
) -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _OLD_HEAD)
        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                connection.execute(text(dependency_ddl))

            with pytest.raises(
                pytest.fail.Exception, match="unsupported decision_type catalog dependencies"
            ):
                _run_alembic(scratch_url, "upgrade", "head")
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                    _OLD_HEAD
                )
        finally:
            engine.dispose()


@pytest.mark.parametrize(
    "drift",
    [
        "partial",
        "invalid",
        "include",
        "expression",
        "hash",
        "nondefault-opclass",
        "nulls-first",
        "reloptions",
    ],
)
def test_forward_repair_rejects_noncanonical_index_catalog_state(drift: str) -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _OLD_HEAD)
        engine = create_engine(scratch_url)
        index_name = "ix_tool_approvals_interrupt_id"
        try:
            with engine.begin() as connection:
                if drift == "partial":
                    connection.execute(text(f"DROP INDEX public.{index_name}"))
                    connection.execute(
                        text(
                            f"CREATE INDEX {index_name} ON public.tool_approvals "
                            "(interrupt_id) WHERE deleted_at IS NULL"
                        )
                    )
                elif drift == "invalid":
                    connection.execute(
                        text(
                            "UPDATE pg_index SET indisvalid = false, indisready = false "
                            "WHERE indexrelid = to_regclass(:index_name)"
                        ),
                        {"index_name": f"public.{index_name}"},
                    )
                else:
                    definitions = {
                        "include": "(interrupt_id) INCLUDE (deleted_at)",
                        "expression": "(lower(interrupt_id))",
                        "hash": "USING hash (interrupt_id)",
                        "nondefault-opclass": "(interrupt_id varchar_pattern_ops)",
                        "nulls-first": "(interrupt_id ASC NULLS FIRST)",
                        "reloptions": "(interrupt_id) WITH (fillfactor = 80)",
                    }
                    connection.execute(text(f"DROP INDEX public.{index_name}"))
                    connection.execute(
                        text(
                            f"CREATE INDEX {index_name} ON public.tool_approvals "
                            f"{definitions[drift]}"
                        )
                    )

            with pytest.raises(
                pytest.fail.Exception, match="unsafe tool_approvals schema drift: indexes mismatch"
            ):
                _run_alembic(scratch_url, "upgrade", "head")
            with engine.connect() as connection:
                assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
                    _OLD_HEAD
                )
        finally:
            engine.dispose()


@pytest.mark.parametrize(
    "drift",
    [
        "missing-table",
        "missing-table-uppercase-enum",
        "missing-table-missing-enum",
        "unknown-enum",
        "unsupported-dependent",
        "missing-column",
    ],
)
def test_forward_repair_handles_only_safe_current_head_drift(drift: str) -> None:
    with _scratch_database(_postgres_test_url()) as scratch_url:
        _run_alembic(scratch_url, "upgrade", _OLD_HEAD)
        engine = create_engine(scratch_url)
        try:
            with engine.begin() as connection:
                canonical_table = _table_schema_snapshot(connection, "tool_approvals")
                if drift.startswith("missing-table"):
                    connection.execute(text("DROP TABLE tool_approvals"))
                    if drift == "missing-table-uppercase-enum":
                        _uppercase_decision_type(connection, include_respond=True)
                    elif drift == "missing-table-missing-enum":
                        connection.execute(text("DROP TYPE decision_type"))
                elif drift == "unknown-enum":
                    connection.execute(
                        text("ALTER TYPE decision_type RENAME VALUE 'accept' TO 'maybe'")
                    )
                elif drift == "unsupported-dependent":
                    connection.execute(
                        text("CREATE TABLE rogue_decisions (decision decision_type)")
                    )
                else:
                    connection.execute(text("ALTER TABLE tool_approvals DROP COLUMN original_args"))

            if drift.startswith("missing-table"):
                _run_alembic(scratch_url, "upgrade", "head")
                with engine.connect() as connection:
                    assert _table_schema_snapshot(connection, "tool_approvals") == canonical_table
                    assert _decision_type_labels(connection) == [
                        "accept",
                        "edit",
                        "reject",
                        "respond",
                    ]
            else:
                expected_error = (
                    "unsupported decision_type labels"
                    if drift == "unknown-enum"
                    else (
                        "unsupported decision_type catalog dependencies"
                        if drift == "unsupported-dependent"
                        else "unsafe tool_approvals schema drift"
                    )
                )
                with pytest.raises(pytest.fail.Exception, match=expected_error):
                    _run_alembic(scratch_url, "upgrade", "head")
                with engine.connect() as connection:
                    assert (
                        connection.scalar(text("SELECT version_num FROM alembic_version"))
                        == _OLD_HEAD
                    )
        finally:
            engine.dispose()
