"""Immutable v1 PostgreSQL repair contract for tool-approval migrations.

Do not adapt this migration-only module to future ORM changes. Create a new versioned helper
when a later revision needs a different schema contract.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic.operations import Operations
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection

BASE_DECISION_LABELS = ("accept", "edit", "reject")
CURRENT_DECISION_LABELS = (*BASE_DECISION_LABELS, "respond")

_BASE_COLUMNS = (
    ("id", "uuid", False, None, None),
    ("created_at", "timestamptz", False, "now()", None),
    ("updated_at", "timestamptz", False, "now()", None),
    ("deleted_at", "timestamptz", True, None, None),
    ("conversation_id", "uuid", False, None, None),
    ("user_id", "uuid", False, None, None),
    ("interrupt_id", "varchar", False, None, 255),
    ("tool_name", "varchar", False, None, 255),
    ("tool_call_id", "varchar", False, None, 255),
    ("original_args", "jsonb", False, None, None),
    ("modified_args", "jsonb", True, None, None),
    ("decision", "decision_type", False, None, None),
    ("decided_at", "timestamptz", False, "now()", None),
)
_CURRENT_EXTRA_COLUMNS = (
    ("device_id", "uuid", True, None, None),
    ("tool_origin", "varchar", True, None, 32),
    ("server_name", "varchar", True, None, 255),
    ("qualified_tool_id", "varchar", True, None, 512),
    ("session_id", "varchar", True, None, 255),
    ("catalog_version", "int4", True, None, None),
    ("tool_instance_id", "varchar", True, None, 64),
)
_BASE_INDEXES = {
    "ix_tool_approvals_conversation_id": ("conversation_id",),
    "ix_tool_approvals_decided_at": ("decided_at",),
    "ix_tool_approvals_id": ("id",),
    "ix_tool_approvals_interrupt_id": ("interrupt_id",),
    "ix_tool_approvals_user_id": ("user_id",),
}
_CURRENT_EXTRA_INDEXES = {
    "ix_tool_approvals_device_id": ("device_id",),
    "ix_tool_approvals_qualified_tool_id": ("qualified_tool_id",),
    "ix_tool_approvals_session_id": ("session_id",),
    "ix_tool_approvals_tool_instance_id": ("tool_instance_id",),
    "ix_tool_approvals_tool_origin": ("tool_origin",),
}
_BASE_FOREIGN_KEYS = {
    "fk_tool_approvals_conversation_id": (
        ("conversation_id",),
        "public",
        "conversations",
        ("id",),
        (),
    ),
    "fk_tool_approvals_user_id": (("user_id",), "public", "users", ("id",), ()),
}
_CURRENT_EXTRA_FOREIGN_KEYS = {
    "fk_tool_approvals_device_id_client_devices": (
        ("device_id",),
        "public",
        "client_devices",
        ("id",),
        (),
    )
}


def require_online(operations: Operations, revision: str) -> None:
    """Reject ``--sql`` because these repairs inspect live PostgreSQL catalogs."""
    if operations.get_context().as_sql:
        raise RuntimeError(
            f"migration {revision} requires an online PostgreSQL connection; "
            "offline SQL generation is unsupported"
        )


def canonicalize_decision_type(
    connection: Connection,
    expected_labels: Sequence[str],
) -> None:
    """Replace a known lowercase/uppercase enum with the canonical labels."""
    expected = tuple(expected_labels)
    type_kind = connection.scalar(
        sa.text(
            "SELECT t.typtype FROM pg_type t "
            "JOIN pg_namespace n ON n.oid = t.typnamespace "
            "WHERE n.nspname = 'public' AND t.typname = 'decision_type'"
        )
    )
    if type_kind is None:
        _create_decision_type(connection, expected)
        return
    if type_kind != "e":
        raise RuntimeError("unsupported public.decision_type: expected a PostgreSQL enum")

    labels = tuple(
        connection.execute(
            sa.text(
                "SELECT e.enumlabel FROM pg_enum e "
                "JOIN pg_type t ON t.oid = e.enumtypid "
                "JOIN pg_namespace n ON n.oid = t.typnamespace "
                "WHERE n.nspname = 'public' AND t.typname = 'decision_type' "
                "ORDER BY e.enumsortorder"
            )
        ).scalars()
    )
    allowed = {
        label for expected_label in expected for label in (expected_label, expected_label.upper())
    }
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise RuntimeError(f"unsupported decision_type labels: {unknown}")
    normalized_labels = tuple(label.lower() for label in labels)
    if normalized_labels != expected:
        raise RuntimeError(
            f"unsupported decision_type label set or ordering: expected {expected}, found {labels}"
        )
    dependents = tuple(
        tuple(row)
        for row in connection.execute(
            sa.text(
                "SELECT n.nspname, c.relname, a.attname "
                "FROM pg_type t "
                "JOIN pg_namespace tn ON tn.oid = t.typnamespace "
                "JOIN pg_attribute a ON a.atttypid = t.oid "
                "AND a.attnum > 0 AND NOT a.attisdropped "
                "JOIN pg_class c ON c.oid = a.attrelid "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE tn.nspname = 'public' AND t.typname = 'decision_type'"
            )
        ).all()
    )
    supported_dependent = (("public", "tool_approvals", "decision"),)
    if dependents not in ((), supported_dependent):
        raise RuntimeError(f"unsupported decision_type dependent columns: {list(dependents)}")
    if dependents:
        decision_default = connection.scalar(
            sa.text(
                "SELECT column_default FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'tool_approvals' "
                "AND column_name = 'decision'"
            )
        )
        if decision_default is not None:
            raise RuntimeError(
                "unsupported tool_approvals.decision default while canonicalizing decision_type"
            )
    if labels == expected:
        return

    temporary_exists = connection.scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM pg_type t "
            "JOIN pg_namespace n ON n.oid = t.typnamespace "
            "WHERE n.nspname = 'public' "
            "AND t.typname = 'decision_type_canonical_tmp')"
        )
    )
    if temporary_exists:
        raise RuntimeError("unsafe schema drift: decision_type_canonical_tmp already exists")

    _create_decision_type(connection, expected, type_name="decision_type_canonical_tmp")
    if dependents:
        connection.execute(
            sa.text(
                "ALTER TABLE public.tool_approvals ALTER COLUMN decision "
                "TYPE public.decision_type_canonical_tmp "
                "USING lower(decision::text)::public.decision_type_canonical_tmp"
            )
        )
    connection.execute(sa.text("DROP TYPE public.decision_type"))
    connection.execute(
        sa.text("ALTER TYPE public.decision_type_canonical_tmp RENAME TO decision_type")
    )


def validate_tool_approvals_schema(connection: Connection, *, current: bool) -> None:
    """Fail on table drift that cannot be repaired without guessing or data loss."""
    expected_columns = _BASE_COLUMNS + (_CURRENT_EXTRA_COLUMNS if current else ())
    actual_columns = tuple(
        connection.execute(
            sa.text(
                "SELECT column_name, udt_name, is_nullable = 'YES', column_default, "
                "character_maximum_length "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'tool_approvals' "
                "ORDER BY ordinal_position"
            )
        ).all()
    )
    normalized_columns = tuple(
        (
            name,
            udt_name,
            nullable,
            _normalize_default(default),
            length,
        )
        for name, udt_name, nullable, default, length in actual_columns
    )
    if normalized_columns != expected_columns:
        raise RuntimeError(
            "unsafe tool_approvals schema drift: columns do not match the "
            f"{'current' if current else 'historical'} contract"
        )

    inspector = sa.inspect(connection)
    primary_key = inspector.get_pk_constraint("tool_approvals", schema="public")
    if primary_key.get("name") != "tool_approvals_pkey" or tuple(
        primary_key.get("constrained_columns") or ()
    ) != ("id",):
        raise RuntimeError("unsafe tool_approvals schema drift: primary key mismatch")

    expected_foreign_keys = dict(_BASE_FOREIGN_KEYS)
    if current:
        expected_foreign_keys.update(_CURRENT_EXTRA_FOREIGN_KEYS)
    actual_foreign_keys = {
        foreign_key["name"]: (
            tuple(foreign_key["constrained_columns"]),
            foreign_key["referred_schema"],
            foreign_key["referred_table"],
            tuple(foreign_key["referred_columns"]),
            tuple(sorted((foreign_key.get("options") or {}).items())),
        )
        for foreign_key in inspector.get_foreign_keys("tool_approvals", schema="public")
    }
    if actual_foreign_keys != expected_foreign_keys:
        raise RuntimeError("unsafe tool_approvals schema drift: foreign keys mismatch")

    expected_indexes = dict(_BASE_INDEXES)
    if current:
        expected_indexes.pop("ix_tool_approvals_id")
        expected_indexes.update(_CURRENT_EXTRA_INDEXES)
    actual_indexes = {
        index["name"]: (tuple(index.get("column_names") or ()), bool(index.get("unique")))
        for index in inspector.get_indexes("tool_approvals", schema="public")
    }
    expected_index_contract = {name: (columns, False) for name, columns in expected_indexes.items()}
    if actual_indexes != expected_index_contract:
        raise RuntimeError("unsafe tool_approvals schema drift: indexes mismatch")
    if inspector.get_unique_constraints("tool_approvals", schema="public"):
        raise RuntimeError("unsafe tool_approvals schema drift: unexpected unique constraints")
    if inspector.get_check_constraints("tool_approvals", schema="public"):
        raise RuntimeError("unsafe tool_approvals schema drift: unexpected check constraints")


def create_tool_approvals(operations: Operations, *, current: bool) -> None:
    """Create the exact historical-6c or current-head table contract."""
    columns = [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interrupt_id", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column("tool_call_id", sa.String(length=255), nullable=False),
        sa.Column("original_args", postgresql.JSONB(), nullable=False),
        sa.Column("modified_args", postgresql.JSONB(), nullable=True),
        sa.Column(
            "decision",
            postgresql.ENUM(
                *(CURRENT_DECISION_LABELS if current else BASE_DECISION_LABELS),
                name="decision_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    ]
    if current:
        columns.extend(
            [
                sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=True),
                sa.Column("tool_origin", sa.String(length=32), nullable=True),
                sa.Column("server_name", sa.String(length=255), nullable=True),
                sa.Column("qualified_tool_id", sa.String(length=512), nullable=True),
                sa.Column("session_id", sa.String(length=255), nullable=True),
                sa.Column("catalog_version", sa.Integer(), nullable=True),
                sa.Column("tool_instance_id", sa.String(length=64), nullable=True),
            ]
        )
    constraints = [
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_tool_approvals_conversation_id",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_tool_approvals_user_id",
        ),
    ]
    if current:
        constraints.append(
            sa.ForeignKeyConstraint(
                ["device_id"],
                ["client_devices.id"],
                name="fk_tool_approvals_device_id_client_devices",
            )
        )
    operations.create_table("tool_approvals", *columns, *constraints)

    indexes = dict(_BASE_INDEXES)
    if current:
        indexes.pop("ix_tool_approvals_id")
        indexes.update(_CURRENT_EXTRA_INDEXES)
    for index_name, column_names in indexes.items():
        operations.create_index(index_name, "tool_approvals", list(column_names))


def _create_decision_type(
    connection: Connection,
    labels: Sequence[str],
    *,
    type_name: str = "decision_type",
) -> None:
    label_sql = ", ".join(f"'{label}'" for label in labels)
    connection.execute(sa.text(f"CREATE TYPE public.{type_name} AS ENUM ({label_sql})"))


def _normalize_default(default: str | None) -> str | None:
    if default is None:
        return None
    return "now()" if default in {"now()", "CURRENT_TIMESTAMP"} else default
