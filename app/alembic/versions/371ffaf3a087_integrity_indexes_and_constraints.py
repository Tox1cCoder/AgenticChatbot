"""foreign-key indexes, redundant-index cleanup, integrity constraints

Revision ID: 371ffaf3a087
Revises: c0033ee1e8cd
Create Date: 2026-09-30 00:00:00.000000

Four groups of change, all checked read-only against the application database
on 2026-09-30 before this was written (PostgreSQL 18.3):

1. **Foreign keys without an index.** Deleting the referenced row, or its
   ``ON DELETE SET NULL``, scans the whole child table without one.
   ``model_usage_events`` (2,731 rows and append-only) gets its two indexes
   ``CONCURRENTLY`` so a large ledger never blocks writes during startup.
2. **One live ``chat_images`` row per (user_id, sha256).** The storage service
   reads-then-inserts, so two concurrent persists of one image both inserted.
   A partial unique index closes that window and the repository inserts with
   ``ON CONFLICT DO NOTHING``. The application database had 0 duplicate groups.
3. **Redundant single-column indexes.** Each one dropped here is the leading
   column of another btree index on the same table, confirmed in the
   application database's ``pg_indexes`` (not only in the models), so every
   lookup it served is still served. ``idx_feedbacks_message_user`` is the one
   composite: its leading column is already UNIQUE on its own.
4. **CHECK constraints and one missing foreign key.** The allowed values are
   the ones the code writes. Live data conformed: 0 bad ratings out of 38
   feedbacks, generation statuses {active}, chunk statuses {indexed}, image
   states {pending, released, selected}, and 0 of 401 non-null
   ``hitl_interrupts.resolved_by_user_id`` values were orphans.

Existing rows can still violate (2) or (4) by the time this runs, and a failed
``ADD CONSTRAINT`` would block API startup with a bare PostgreSQL error. So the
upgrade counts violations first and refuses with a message naming each table,
the count, and the rule, before it changes anything. The one repair it makes
itself is nulling ``resolved_by_user_id`` values whose user no longer exists,
which is exactly what the new ``ON DELETE SET NULL`` would have done.
"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "371ffaf3a087"
down_revision: str | Sequence[str] | None = "c0033ee1e8cd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Built without blocking writes. Each entry is (index, table, column).
_CONCURRENT_FK_INDEXES = (
    ("ix_model_usage_events_request_message_id", "model_usage_events", "request_message_id"),
    ("ix_model_usage_events_document_id", "model_usage_events", "document_id"),
)

_FK_INDEXES = (
    ("ix_generations_assistant_message_id", "generations", "assistant_message_id"),
    ("ix_project_custom_agents_custom_agent_id", "project_custom_agents", "custom_agent_id"),
    ("ix_user_memories_project_id", "user_memories", "project_id"),
    ("ix_hitl_interrupts_resolved_by_user_id", "hitl_interrupts", "resolved_by_user_id"),
)

#: (index, table, columns). The trailing comment names the index that still
#: covers each one on the application database.
_REDUNDANT_INDEXES = (
    # uq_messages_conversation_sequence, idx_message_conversation_created
    ("ix_messages_conversation_id", "messages", ("conversation_id",)),
    # uq_task_plan_conversation_order
    ("ix_task_plans_conversation_id", "task_plans", ("conversation_id",)),
    # ix_generations_owner_status
    ("ix_generations_user_id", "generations", ("user_id",)),
    # ix_tool_execution_receipts_owner_status
    ("ix_tool_execution_receipts_user_id", "tool_execution_receipts", ("user_id",)),
    # ix_tool_execution_receipts_conversation_turn
    (
        "ix_tool_execution_receipts_conversation_id",
        "tool_execution_receipts",
        ("conversation_id",),
    ),
    # idx_agent_model_configs_user_agent (unique)
    ("idx_agent_model_configs_user_id", "agent_model_configs", ("user_id",)),
    # uq_client_devices_user_device_id
    ("ix_client_devices_user_id", "client_devices", ("user_id",)),
    # uq_skill_settings_user_skill
    ("ix_skill_settings_user_id", "skill_settings", ("user_id",)),
    # uq_tool_approval_settings_user_device_origin_scope
    ("ix_tool_approval_settings_user_id", "tool_approval_settings", ("user_id",)),
    # uq_document_parse_artifact_path
    ("ix_document_parse_artifacts_document_id", "document_parse_artifacts", ("document_id",)),
    # uq_document_chunk_generation_index
    ("idx_document_chunks_document_id", "document_chunks", ("document_id",)),
    # ix_custom_agents_owner_deleted
    ("ix_custom_agents_owner_id", "custom_agents", ("owner_id",)),
    # ix_projects_owner_deleted
    ("ix_projects_owner_id", "projects", ("owner_id",)),
    # ix_feedbacks_message_id is UNIQUE on the leading column alone
    ("idx_feedbacks_message_user", "feedbacks", ("message_id", "user_id")),
)

#: The one redundant index on the large ledger; dropped and rebuilt concurrently.
_REDUNDANT_LEDGER_INDEX = (
    "ix_model_usage_events_operation_id",
    "model_usage_events",
    "operation_id",
)

_CHAT_IMAGE_UNIQUE = "uq_chat_images_user_sha256_live"

#: (table, constraint, condition). Values mirror the constants in the models.
_CHECKS = (
    ("feedbacks", "ck_feedbacks_rating_range", "rating BETWEEN 1 AND 5"),
    (
        "document_index_generations",
        "ck_document_index_generations_status",
        "status IN ('building', 'ready', 'active', 'retired', 'failed')",
    ),
    (
        "document_chunks",
        "ck_document_chunks_index_status",
        "index_status IN ('pending', 'indexed', 'failed', 'needs_reindex')",
    ),
    (
        "web_image_references",
        "ck_web_image_references_lifecycle_state",
        "lifecycle_state IN ('pending', 'selected', 'released')",
    ),
)

_RESOLVER_FK = "fk_hitl_interrupts_resolved_by_user_id_users"


def _require_online() -> None:
    if op.get_context().as_sql:
        raise RuntimeError(
            "371ffaf3a087 counts violating rows before adding constraints and "
            "requires an online PostgreSQL connection"
        )


def _violations(bind) -> list[str]:
    """Every reason the constraints below would fail, counted up front."""
    problems: list[str] = []
    duplicate_groups = bind.execute(
        sa.text(
            "SELECT count(*) FROM (SELECT 1 FROM chat_images WHERE deleted_at IS NULL "
            "GROUP BY user_id, sha256 HAVING count(*) > 1) duplicates"
        )
    ).scalar_one()
    if duplicate_groups:
        problems.append(
            f"chat_images: {duplicate_groups} (user_id, sha256) groups have more than one "
            f"live row (deleted_at IS NULL); {_CHAT_IMAGE_UNIQUE} allows one"
        )
    for table, name, condition in _CHECKS:
        # NOT (condition) skips NULL, exactly as a CHECK constraint does.
        violating = bind.execute(
            sa.text(f"SELECT count(*) FROM {table} WHERE NOT ({condition})")
        ).scalar_one()
        if violating:
            problems.append(f"{table}: {violating} rows violate {name} ({condition})")
    return problems


def _null_orphaned_resolvers(bind) -> None:
    result = bind.execute(
        sa.text(
            "UPDATE hitl_interrupts SET resolved_by_user_id = NULL "
            "WHERE resolved_by_user_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM users WHERE users.id = hitl_interrupts.resolved_by_user_id)"
        )
    )
    if result.rowcount:
        logger.warning(
            "371ffaf3a087 nulled %d hitl_interrupts.resolved_by_user_id values "
            "whose user no longer exists",
            result.rowcount,
        )


def _build_concurrently(name: str, table: str, column: str) -> None:
    # An interrupted CONCURRENTLY build leaves an INVALID index behind under the
    # same name; dropping first makes a retry rebuild a valid one.
    op.drop_index(name, table_name=table, if_exists=True, postgresql_concurrently=True)
    op.create_index(name, table, [column], postgresql_concurrently=True)


def upgrade() -> None:
    """Refuse on violating rows, then add indexes and constraints."""

    _require_online()
    bind = op.get_bind()
    problems = _violations(bind)
    if problems:
        raise RuntimeError(
            "371ffaf3a087 changed nothing: fix these rows, then start again. "
            + "; ".join(problems)
        )

    # Ahead of the transactional part: each step here is idempotent, so a
    # failure later in this revision is retried without leftovers.
    with op.get_context().autocommit_block():
        for name, table, column in _CONCURRENT_FK_INDEXES:
            _build_concurrently(name, table, column)
        name, table, _column = _REDUNDANT_LEDGER_INDEX
        op.drop_index(name, table_name=table, if_exists=True, postgresql_concurrently=True)

    for name, table, column in _FK_INDEXES:
        op.create_index(name, table, [column])

    op.create_index(
        _CHAT_IMAGE_UNIQUE,
        "chat_images",
        ["user_id", "sha256"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    for name, table, _columns in _REDUNDANT_INDEXES:
        op.drop_index(name, table_name=table, if_exists=True)

    for table, name, condition in _CHECKS:
        op.create_check_constraint(name, table, condition)

    _null_orphaned_resolvers(bind)
    op.create_foreign_key(
        _RESOLVER_FK,
        "hitl_interrupts",
        "users",
        ["resolved_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Remove what the upgrade added and rebuild every index it dropped."""

    op.drop_constraint(_RESOLVER_FK, "hitl_interrupts", type_="foreignkey")
    for table, name, _condition in reversed(_CHECKS):
        op.drop_constraint(name, table, type_="check")
    for name, table, columns in reversed(_REDUNDANT_INDEXES):
        op.create_index(name, table, list(columns))
    op.drop_index(_CHAT_IMAGE_UNIQUE, table_name="chat_images")
    for name, table, _column in reversed(_FK_INDEXES):
        op.drop_index(name, table_name=table)

    with op.get_context().autocommit_block():
        name, table, column = _REDUNDANT_LEDGER_INDEX
        _build_concurrently(name, table, column)
        for name, table, _column in reversed(_CONCURRENT_FK_INDEXES):
            op.drop_index(name, table_name=table, if_exists=True, postgresql_concurrently=True)
