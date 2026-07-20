from __future__ import annotations

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.models.model_usage import ModelUsageEvent, ModelUsageMinute
from app.models.user import User

NULLABLE_TOKEN_COLUMNS = [
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "input_text_tokens",
    "input_image_tokens",
    "output_text_tokens",
    "output_image_tokens",
]

REQUIRED_SCALAR_COLUMNS = [
    "event_key",
    "operation_id",
    "attempt",
    "provider",
    "model",
    "operation",
    "status",
    "usage_source",
    "generated_images",
    "latency_ms",
    "started_at",
    "completed_at",
    "created_at",
]


def _named_constraint(table, constraint_type, name):
    return next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, constraint_type) and constraint.name == name
    )


def _checks(table):
    return {
        constraint.name: str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }


def test_model_usage_events_table_name_and_primary_key():
    table = ModelUsageEvent.__table__
    assert table.name == "model_usage_events"
    assert isinstance(table.c.id.type, UUID)
    assert table.c.id.primary_key is True


def test_model_usage_events_token_columns_are_nullable_biginteger():
    table = ModelUsageEvent.__table__
    for name in NULLABLE_TOKEN_COLUMNS:
        column = table.c[name]
        assert isinstance(column.type, BigInteger), name
        assert column.nullable is True, name


def test_model_usage_events_required_scalar_columns_are_not_null():
    table = ModelUsageEvent.__table__
    for name in REQUIRED_SCALAR_COLUMNS:
        assert table.c[name].nullable is False, name

    assert isinstance(table.c.generated_images.type, Integer)
    assert isinstance(table.c.latency_ms.type, BigInteger)
    assert isinstance(table.c.started_at.type, DateTime)
    assert table.c.started_at.type.timezone is True
    assert isinstance(table.c.completed_at.type, DateTime)
    assert table.c.completed_at.type.timezone is True
    assert str(table.c.created_at.server_default.arg) == "now()"


def test_model_usage_events_nullable_fk_and_metadata_columns():
    table = ModelUsageEvent.__table__
    assert table.c.user_id.nullable is True
    assert table.c.conversation_id.nullable is True
    assert table.c.request_message_id.nullable is True
    assert table.c.document_id.nullable is True
    assert table.c.correlation_id.nullable is True
    assert table.c.langsmith_run_id.nullable is True
    assert table.c.provider_request_id.nullable is True
    assert table.c.agent_id.nullable is True
    assert table.c.error_code.nullable is True


def test_model_usage_events_event_key_and_operation_attempt_are_unique():
    table = ModelUsageEvent.__table__

    event_key_unique = _named_constraint(table, UniqueConstraint, "uq_model_usage_events_event_key")
    assert [column.name for column in event_key_unique.columns] == ["event_key"]

    operation_attempt_unique = _named_constraint(
        table, UniqueConstraint, "uq_model_usage_events_operation_attempt"
    )
    assert [column.name for column in operation_attempt_unique.columns] == [
        "operation_id",
        "attempt",
    ]


def test_model_usage_events_foreign_key_delete_behavior():
    table = ModelUsageEvent.__table__

    assert next(iter(table.c.user_id.foreign_keys)).ondelete == "CASCADE"
    assert next(iter(table.c.conversation_id.foreign_keys)).ondelete == "SET NULL"
    assert next(iter(table.c.request_message_id.foreign_keys)).ondelete == "SET NULL"
    assert next(iter(table.c.document_id.foreign_keys)).ondelete == "SET NULL"


def test_model_usage_events_operation_id_is_indexed():
    table = ModelUsageEvent.__table__
    assert any(index.name == "ix_model_usage_events_operation_id" for index in table.indexes)


def test_model_usage_events_nonnegative_checks_are_database_enforced():
    checks = _checks(ModelUsageEvent.__table__)

    expected = {
        "ck_model_usage_events_attempt_positive": "attempt >= 1",
        "ck_model_usage_events_generated_images_nonnegative": "generated_images >= 0",
        "ck_model_usage_events_latency_ms_nonnegative": "latency_ms >= 0",
    }
    for name in NULLABLE_TOKEN_COLUMNS:
        expected[f"ck_model_usage_events_{name}_nonnegative"] = f"{name} IS NULL OR {name} >= 0"

    assert checks == expected


def test_model_usage_minute_table_name_and_primary_key():
    table = ModelUsageMinute.__table__
    assert table.name == "model_usage_minute"
    assert table.c.rollup_key.primary_key is True
    assert isinstance(table.c.rollup_key.type, String)
    assert table.c.rollup_key.type.length == 64
    assert table.c.rollup_key.nullable is False


def test_model_usage_minute_bucket_and_dimension_columns():
    table = ModelUsageMinute.__table__

    assert isinstance(table.c.bucket_start_utc.type, DateTime)
    assert table.c.bucket_start_utc.type.timezone is True
    assert table.c.bucket_start_utc.nullable is False

    for name in ["provider", "model", "operation", "status", "usage_source"]:
        assert table.c[name].nullable is False

    assert table.c.agent_id.nullable is True
    assert table.c.user_id.nullable is True
    assert table.c.conversation_id.nullable is True
    assert str(table.c.created_at.server_default.arg) == "now()"


def test_model_usage_minute_foreign_keys_cascade_on_delete():
    table = ModelUsageMinute.__table__
    assert next(iter(table.c.user_id.foreign_keys)).ondelete == "CASCADE"
    assert next(iter(table.c.conversation_id.foreign_keys)).ondelete == "CASCADE"


def test_model_usage_minute_sum_and_known_count_columns_are_nonnegative_biginteger():
    table = ModelUsageMinute.__table__
    checks = _checks(table)

    for name in NULLABLE_TOKEN_COLUMNS:
        sum_column = f"{name}_sum"
        count_column = f"{name}_known_count"

        assert isinstance(table.c[sum_column].type, BigInteger), sum_column
        assert table.c[sum_column].nullable is False, sum_column
        assert isinstance(table.c[count_column].type, BigInteger), count_column
        assert table.c[count_column].nullable is False, count_column

        # model_usage_minute uses an abbreviated check-constraint naming
        # scheme ("ck_mu_minute_<column>_nonneg") because the full-length
        # scheme pushes several *_known_count names past PostgreSQL's
        # 63-character identifier limit.
        assert checks[f"ck_mu_minute_{sum_column}_nonneg"] == f"{sum_column} >= 0"
        assert checks[f"ck_mu_minute_{count_column}_nonneg"] == f"{count_column} >= 0"

    for name in ["generated_images_sum", "request_count", "latency_ms_sum"]:
        assert isinstance(table.c[name].type, BigInteger), name
        assert table.c[name].nullable is False, name
        assert checks[f"ck_mu_minute_{name}_nonneg"] == f"{name} >= 0"


def test_expected_named_indexes_exist_on_both_tables():
    expected_indexes = {
        "ix_model_usage_events_user_started",
        "ix_model_usage_events_conversation_started",
        "ix_model_usage_events_provider_model_started",
        "ix_model_usage_events_operation_started",
        "ix_model_usage_minute_user_bucket",
        "ix_model_usage_minute_conversation_bucket",
    }
    actual_indexes = {
        index.name
        for table in (ModelUsageEvent.__table__, ModelUsageMinute.__table__)
        for index in table.indexes
    }
    assert expected_indexes <= actual_indexes


def test_user_has_usage_events_relationship_without_eager_loading():
    relationship_property = User.usage_events.property
    assert relationship_property.mapper.class_ is ModelUsageEvent
    assert relationship_property.back_populates == "user"
    assert relationship_property.lazy == "select"


def test_all_constraint_and_index_names_fit_postgres_identifier_limit():
    # PostgreSQL truncates identifiers at 63 bytes; SQLAlchemy raises
    # IdentifierError rather than emit a name that would silently collide
    # after truncation. Guard every constraint/index name generated for
    # these two tables so a future column addition can't regress this
    # (caught this exact failure against real PostgreSQL: several
    # model_usage_minute *_known_count check names were 64-65 chars).
    postgres_max_identifier_length = 63

    for table in (ModelUsageEvent.__table__, ModelUsageMinute.__table__):
        for constraint in table.constraints:
            if constraint.name is not None:
                assert len(constraint.name) <= postgres_max_identifier_length, constraint.name
        for index in table.indexes:
            assert len(index.name) <= postgres_max_identifier_length, index.name
