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
    ),
    "fk_tool_approvals_user_id": (("user_id",), "public", "users", ("id",)),
}
_CURRENT_EXTRA_FOREIGN_KEYS = {
    "fk_tool_approvals_device_id_client_devices": (
        ("device_id",),
        "public",
        "client_devices",
        ("id",),
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
    *,
    allowed_future_labels: Sequence[str] = (),
) -> None:
    """Normalize known case-only enum drift in place after dependency checks."""
    expected = tuple(expected_labels)
    future = tuple(allowed_future_labels)
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
    accepted_label_sequences = (expected, expected + future) if future else (expected,)
    allowed = {
        label
        for expected_label in expected + future
        for label in (expected_label, expected_label.upper())
    }
    unknown = sorted(set(labels) - allowed)
    if unknown:
        raise RuntimeError(f"unsupported decision_type labels: {unknown}")
    unsupported_dependencies = tuple(
        connection.execute(
            sa.text(
                "SELECT pg_describe_object(d.classid, d.objid, d.objsubid) "
                "FROM pg_type t "
                "JOIN pg_namespace n ON n.oid = t.typnamespace "
                "JOIN pg_depend d ON d.refclassid = 'pg_type'::regclass "
                "AND d.refobjid IN (t.oid, t.typarray) AND d.refobjsubid = 0 "
                "WHERE n.nspname = 'public' AND t.typname = 'decision_type' "
                "AND NOT ("
                "  (d.refobjid = t.oid AND d.classid = 'pg_type'::regclass "
                "   AND d.objid = t.typarray "
                "   AND d.objsubid = 0 AND d.deptype = 'i') "
                "  OR "
                "  (d.refobjid = t.oid AND d.classid = 'pg_class'::regclass "
                "   AND d.objid = to_regclass('public.tool_approvals') "
                "   AND d.objsubid = ("
                "     SELECT a.attnum FROM pg_attribute a "
                "     WHERE a.attrelid = to_regclass('public.tool_approvals') "
                "     AND a.attname = 'decision' AND a.attnum > 0 "
                "     AND NOT a.attisdropped"
                "   ) AND d.deptype = 'n')"
                ") ORDER BY 1"
            )
        ).scalars()
    )
    if unsupported_dependencies:
        raise RuntimeError(
            f"unsupported decision_type catalog dependencies: {list(unsupported_dependencies)}"
        )
    normalized_labels = tuple(label.lower() for label in labels)
    if normalized_labels not in accepted_label_sequences:
        raise RuntimeError(
            "unsupported decision_type label set or ordering: "
            f"expected one of {accepted_label_sequences}, found {labels}"
        )
    decision_column_exists = connection.scalar(
        sa.text(
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_attribute a "
            "WHERE a.attrelid = to_regclass('public.tool_approvals') "
            "AND a.attname = 'decision' AND a.atttypid = "
            "'public.decision_type'::regtype AND a.attnum > 0 "
            "AND NOT a.attisdropped)"
        )
    )
    if decision_column_exists:
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
    if labels == normalized_labels:
        return
    for source_label, target_label in zip(labels, normalized_labels, strict=True):
        if source_label != target_label:
            connection.execute(
                sa.text(
                    "ALTER TYPE public.decision_type "
                    f"RENAME VALUE '{source_label}' TO '{target_label}'"
                )
            )


def repair_current_legacy_tool_approvals(connection: Connection) -> None:
    """Normalize only the exact, known original-6c differences at current head."""
    defaults = dict(
        connection.execute(
            sa.text(
                "SELECT column_name, column_default FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'tool_approvals' "
                "AND column_name IN ('created_at', 'updated_at', 'decided_at')"
            )
        ).all()
    )
    if set(defaults) != {"created_at", "updated_at", "decided_at"}:
        raise RuntimeError("unsafe tool_approvals schema drift: timestamp columns are missing")
    for column_name, default in defaults.items():
        normalized_default = _normalize_default(default)
        if normalized_default is None:
            connection.execute(
                sa.text(
                    "ALTER TABLE public.tool_approvals "
                    f"ALTER COLUMN {column_name} SET DEFAULT now()"
                )
            )
        elif normalized_default != "now()":
            raise RuntimeError(
                "unsafe tool_approvals schema drift: unsupported timestamp default "
                f"on {column_name}"
            )

    foreign_keys = _foreign_key_catalog(connection)
    _raise_for_not_valid_foreign_keys(foreign_keys)
    expected_contracts = _expected_foreign_key_contracts(current=True)
    legacy_names = {
        "tool_approvals_conversation_id_fkey": "fk_tool_approvals_conversation_id",
        "tool_approvals_user_id_fkey": "fk_tool_approvals_user_id",
    }
    for legacy_name, canonical_name in legacy_names.items():
        if legacy_name not in foreign_keys:
            continue
        if canonical_name in foreign_keys:
            raise RuntimeError(
                "unsafe tool_approvals schema drift: duplicate legacy and canonical foreign keys"
            )
        if foreign_keys[legacy_name] != expected_contracts[canonical_name]:
            raise RuntimeError(
                f"unsafe tool_approvals schema drift: legacy foreign key {legacy_name} mismatch"
            )
        connection.execute(
            sa.text(
                "ALTER TABLE public.tool_approvals "
                f"RENAME CONSTRAINT {legacy_name} TO {canonical_name}"
            )
        )


def _foreign_key_catalog(connection: Connection) -> dict[str, tuple]:
    rows = connection.execute(
        sa.text(
            "SELECT con.conname, "
            "ARRAY("
            " SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord) "
            " JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum "
            " ORDER BY k.ord"
            "), ref_ns.nspname, ref.relname, "
            "ARRAY("
            " SELECT a.attname FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum, ord) "
            " JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum "
            " ORDER BY k.ord"
            "), con.convalidated, con.condeferrable, con.condeferred, "
            "con.confupdtype, con.confdeltype, con.confmatchtype, con.conislocal, "
            "con.coninhcount, con.connoinherit, "
            "ARRAY("
            " SELECT p.proname FROM pg_trigger t JOIN pg_proc p ON p.oid = t.tgfoid "
            " WHERE t.tgconstraint = con.oid ORDER BY p.proname"
            "), NOT EXISTS ("
            " SELECT 1 FROM pg_trigger t WHERE t.tgconstraint = con.oid "
            " AND (NOT t.tgisinternal OR t.tgenabled <> 'O')"
            ") "
            "FROM pg_constraint con "
            "JOIN pg_class ref ON ref.oid = con.confrelid "
            "JOIN pg_namespace ref_ns ON ref_ns.oid = ref.relnamespace "
            "WHERE con.conrelid = 'public.tool_approvals'::regclass "
            "AND con.contype = 'f' ORDER BY con.conname"
        )
    ).all()
    return {
        name: (
            tuple(columns),
            referred_schema,
            referred_table,
            tuple(referred_columns),
            validated,
            deferrable,
            deferred,
            update_action,
            delete_action,
            match_type,
            is_local,
            inherit_count,
            no_inherit,
            tuple(trigger_functions),
            triggers_enabled,
        )
        for (
            name,
            columns,
            referred_schema,
            referred_table,
            referred_columns,
            validated,
            deferrable,
            deferred,
            update_action,
            delete_action,
            match_type,
            is_local,
            inherit_count,
            no_inherit,
            trigger_functions,
            triggers_enabled,
        ) in rows
    }


def _expected_foreign_key_contracts(*, current: bool) -> dict[str, tuple]:
    definitions = dict(_BASE_FOREIGN_KEYS)
    if current:
        definitions.update(_CURRENT_EXTRA_FOREIGN_KEYS)
    trigger_functions = (
        "RI_FKey_check_ins",
        "RI_FKey_check_upd",
        "RI_FKey_noaction_del",
        "RI_FKey_noaction_upd",
    )
    return {
        name: (
            *definition,
            True,
            False,
            False,
            "a",
            "a",
            "s",
            True,
            0,
            True,
            trigger_functions,
            True,
        )
        for name, definition in definitions.items()
    }


def _raise_for_not_valid_foreign_keys(foreign_keys: dict[str, tuple]) -> None:
    for name, contract in foreign_keys.items():
        if not contract[4]:
            raise RuntimeError(
                f"foreign key {name} is NOT VALID; repair orphan rows and run "
                f"ALTER TABLE tool_approvals VALIDATE CONSTRAINT {name} before retrying"
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

    actual_foreign_keys = _foreign_key_catalog(connection)
    _raise_for_not_valid_foreign_keys(actual_foreign_keys)
    if actual_foreign_keys != _expected_foreign_key_contracts(current=current):
        raise RuntimeError("unsafe tool_approvals schema drift: foreign keys mismatch")

    expected_indexes = dict(_BASE_INDEXES)
    if current:
        expected_indexes.pop("ix_tool_approvals_id")
        expected_indexes.update(_CURRENT_EXTRA_INDEXES)
    actual_indexes = {
        name: (tuple(column_names), catalog_valid)
        for name, column_names, catalog_valid in connection.execute(
            sa.text(
                "SELECT idx.relname, "
                "ARRAY("
                "  SELECT a.attname "
                "  FROM unnest(i.indkey::smallint[]) WITH ORDINALITY AS k(attnum, ord) "
                "  LEFT JOIN pg_attribute a ON a.attrelid = i.indrelid "
                "  AND a.attnum = k.attnum ORDER BY k.ord"
                "), "
                "(idx.relkind = 'i' AND idx.relpersistence = 'p' "
                " AND am.amname = 'btree' AND NOT i.indisunique "
                " AND NOT i.indisprimary AND NOT i.indisexclusion "
                " AND i.indimmediate AND i.indisvalid AND i.indisready "
                " AND i.indislive AND NOT i.indisclustered "
                " AND NOT i.indisreplident AND NOT i.indcheckxmin "
                " AND i.indpred IS NULL AND i.indexprs IS NULL "
                " AND i.indnkeyatts = i.indnatts "
                " AND cardinality(i.indkey::smallint[]) = i.indnatts "
                " AND cardinality(i.indoption::smallint[]) = i.indnkeyatts "
                " AND cardinality(i.indclass::oid[]) = i.indnkeyatts "
                " AND cardinality(i.indcollation::oid[]) = i.indnkeyatts "
                " AND (idx.reloptions IS NULL OR cardinality(idx.reloptions) = 0) "
                " AND NOT EXISTS ("
                "   SELECT 1 FROM unnest(i.indoption::smallint[]) AS options(value) "
                "   WHERE options.value <> 0"
                " ) AND NOT EXISTS ("
                "   SELECT 1 FROM unnest(i.indclass::oid[]) AS classes(opclass_oid) "
                "   LEFT JOIN pg_opclass opc ON opc.oid = classes.opclass_oid "
                "   WHERE opc.oid IS NULL OR NOT opc.opcdefault "
                "   OR opc.opcmethod <> idx.relam"
                " ) AND NOT EXISTS ("
                "   SELECT 1 "
                "   FROM unnest(i.indkey::smallint[], i.indcollation::oid[]) "
                "     AS pairs(attnum, collation_oid) "
                "   LEFT JOIN pg_attribute a ON a.attrelid = i.indrelid "
                "   AND a.attnum = pairs.attnum "
                "   WHERE a.attnum IS NULL OR pairs.collation_oid <> a.attcollation"
                " )) AS catalog_valid "
                "FROM pg_index i "
                "JOIN pg_class table_class ON table_class.oid = i.indrelid "
                "JOIN pg_namespace table_ns ON table_ns.oid = table_class.relnamespace "
                "JOIN pg_class idx ON idx.oid = i.indexrelid "
                "JOIN pg_am am ON am.oid = idx.relam "
                "WHERE table_ns.nspname = 'public' "
                "AND table_class.relname = 'tool_approvals' "
                "AND NOT i.indisprimary ORDER BY idx.relname"
            )
        ).all()
    }
    expected_index_contract = {name: (columns, True) for name, columns in expected_indexes.items()}
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
