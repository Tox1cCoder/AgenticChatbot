# Graph and Database Codebase Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trim `app/ai/graph.py`, delete obsolete compatibility paths, consolidate the database/session layer, and bring SQLAlchemy models, Alembic, and the live PostgreSQL schema into a production-ready contract.

**Architecture:** Keep the public workflow/API contracts stable first, then move behavior out of `MultiAgentWorkflow` behind focused modules. Treat PostgreSQL/Alembic as the source of schema truth, with SQLAlchemy metadata required to match the live schema after the cleanup migrations. Keep LangGraph checkpoint tables owned by LangGraph, not by application autogenerate.

**Tech Stack:** FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, LangGraph, Qdrant, Redis, Celery, pytest, Ruff.

---

## Requirements And Decisions

- Full refactor permission is granted, including schema and API changes, when they are production-ready.
- Database schema changes are allowed.
- Legacy/redundant functions should be deleted, not wrapped indefinitely.
- API changes must update `plans/AI_SDK_FE_CONTRACT.md`.
- Because the authoritative database source was unclear, this plan uses all three available sources: SQLAlchemy models, Alembic history, and the reachable local PostgreSQL database.
- No frontend contract change is planned in the recommended path. The AI SDK and Streamlit SSE wire formats should remain compatible.

## Approach Options

**Recommended: staged contract-first refactor.** First add schema and stream safety tests, then fix Alembic/model drift, then split `graph.py` by behavior. This gives a reliable rollback point for each phase and avoids changing graph behavior while the database is unstable.

**Alternative: graph-only extraction first.** This reduces the visible bloat quickly, but leaves Alembic autogenerate trying to drop external tables and leaves the local DB carrying orphaned tables. Not recommended because database drift can break future deploys.

**Alternative: database-only hardening first.** This is safe but does not address the user's main pain point: the graph has accumulated streaming, tool, planning, custom-agent, checkpoint, and compatibility concerns in one 5,202-line class.

## Current Findings

### Graph And Runtime

- `app/ai/graph.py` is 5,202 lines and `MultiAgentWorkflow` owns topology, routing, all agent nodes, RAG loops, planning subagents, tool execution, HITL, checkpoint compaction, request execution, resume, streaming, and legacy event projection.
- `app/ai/graph.py` still contains a no-op `_summarization_node` and a `summarize` node for old checkpoints. The live DB still has checkpoint rows containing `summarize`, so deleting it requires checkpoint cleanup first.
- `app/services/event_streaming/*` already contains the canonical v3, AI SDK v6, internal SSE, and compatibility adapters. The graph still projects canonical events back into legacy dicts, then `AIService` maps those dicts back to canonical `V3StreamEvent`.
- `IWorkflowRuntime.resume()` is still called by `AIService.resume_workflow()`, so it is not dead code unless the non-stream resume API is removed or redirected.

### Live Database Snapshot

Checked on 2026-07-08 against the configured local PostgreSQL database.

| Table | Rows | Status |
|---|---:|---|
| `users` | 749 | app model |
| `conversations` | 1,827 | app model |
| `messages` | 6,729 | app model |
| `feedbacks` | 38 | app model, pluralized |
| `documents` | 277 | app model |
| `document_chunks` | 829 | app model, SQL content source for RAG |
| `document_images` | 947 | app model |
| `document_parse_artifacts` | 30 | app model |
| `conversation_memory_summaries` | 0 | app model |
| `task_plans` | 953 | app model |
| `tool_approvals` | 367 | app model |
| `tool_approval_settings` | 1 | app model |
| `hitl_interrupts` | 439 | app model |
| `client_devices` | 519 | app model |
| `skill_settings` | 0 | app model |
| `custom_agents` | 7 | app model |
| `conversation_custom_agents` | 16 | app model |
| `agent_model_configs` | 4 | app model |
| `model_providers` | 2 | app model |
| `tool_result_blobs` | 83 | app model |
| `user_memories` | 0 | app model |
| `conversation_device_bindings` | 4 | live DB only, no current model/repository usage |
| `alembic_version` | 1 | Alembic-owned |
| `checkpoint_migrations` | 10 | LangGraph-owned |
| `checkpoints` | 16,664 | LangGraph-owned |
| `checkpoint_blobs` | 54,626 | LangGraph-owned |
| `checkpoint_writes` | 247,524 | LangGraph-owned |

### Database Issues To Fix

- `alembic heads --verbose` shows one head: `g0h1i2j3k4l5`.
- `alembic current --verbose` reports the live DB at `g0h1i2j3k4l5`.
- `alembic check` currently fails. It detects:
  - attempted removal of LangGraph tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`);
  - attempted removal of `conversation_device_bindings`;
  - `client_devices.status` type drift (`VARCHAR(32)` live vs SQLAlchemy non-native enum metadata);
  - index name drift on `conversation_memory_summaries`;
  - `document_chunks.qdrant_point_id` unique/index drift;
  - duplicate or missing model indexes around document images, blobs, and user memories.
- `conversation_device_bindings` appears only in an old migration and old plans. Current code and newer design notes say conversation ownership must be user-based, not device-bound. Plan: drop it with a reversible migration.
- `6c6598a9eb26_create_missing_tool_approvals_table.py` shows a previous autogenerate hazard: because checkpoint tables are not in app metadata, Alembic generated destructive drop operations for LangGraph tables. Plan: add explicit autogenerate filters for external tables.
- `app/database/session.py` and `app/database/database.py` create separate SQLAlchemy engines/session factories. Plan: one engine/session factory, with repository sessions and FastAPI `get_db()` using the same provider.
- `Database.create_database()` calls `Base.metadata.create_all()`. It is not used in current app code and should be removed so migrations are the only schema mutation path.
- Several primary keys have redundant `index=True` model declarations or live `ix_*_id` indexes. PostgreSQL primary keys already create indexes.

## Target File Structure

Create:

- `app/alembic/autogenerate_filters.py` - Alembic include filters for external tables and naming exclusions.
- `app/alembic/versions/v1w2x3y4z5a6_schema_contract_cleanup.py` - one explicit migration for the owned-schema cleanup.
- `app/services/checkpoint_retention_service.py` - checkpoint cleanup and retention orchestration.
- `app/services/event_streaming/graph_public_projection.py` - graph canonical/legacy public projection currently embedded in `MultiAgentWorkflow`.
- `app/ai/workflow/graph_builder.py` - graph topology builder.
- `app/ai/workflow/tool_loop.py` - shared tool/HITL loop helpers.
- `app/ai/workflow/rag_loop.py` - RAG tool-loop helpers.
- `app/ai/workflow/planning_loop.py` - planning and subagent execution helpers.
- `app/ai/workflow/custom_agents.py` - custom-agent runtime, handoff, and activity helpers.
- `tests/test_alembic_autogenerate_filters.py`
- `tests/test_database_schema_contract.py`
- `tests/test_database_session_provider.py`
- `tests/test_checkpoint_retention_service.py`
- `tests/test_graph_stream_projection.py`
- `tests/test_graph_refactor_contract.py`

Modify:

- `app/alembic/env.py`
- `app/database/session.py`
- `app/database/database.py`
- `app/core/container.py`
- `app/main.py`
- `app/workers/cleanup_tasks.py`
- `app/ai/checkpoint.py`
- `app/ai/graph.py`
- `app/interfaces/workflow_runtime_interface.py` only if non-stream resume is intentionally removed.
- ORM model files with redundant indexes or schema drift:
  - `app/models/agent_model_config.py`
  - `app/models/client_device.py`
  - `app/models/conversation_memory_summary.py`
  - `app/models/custom_agent.py`
  - `app/models/document_chunk.py`
  - `app/models/document_image.py`
  - `app/models/document_parse_artifact.py`
  - `app/models/feedback.py`
  - `app/models/message.py`
  - `app/models/model_provider.py`
  - `app/models/skill_setting.py`
  - `app/models/task_plan.py`
  - `app/models/tool_approval.py`
  - `app/models/tool_approval_setting.py`
  - `app/models/tool_result_blob.py`
  - `app/models/user.py`
  - `app/models/user_memory.py`
- `plans/AI_SDK_FE_CONTRACT.md` only if stream event names, endpoint paths, metadata shape, or terminal sequences change.

## Task 1: Add Schema Drift Safety Tests

**Files:**

- Create: `tests/test_database_schema_contract.py`

- [x] **Step 1: Write the failing schema contract test**

```python
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
```

- [x] **Step 2: Run the test and confirm it fails before cleanup**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_database_schema_contract.py -q
```

Expected before implementation: FAIL on `conversation_device_bindings` and redundant primary-key indexes.

## Task 2: Guard Alembic Autogenerate From External Tables

**Files:**

- Create: `app/alembic/autogenerate_filters.py`
- Create: `tests/test_alembic_autogenerate_filters.py`
- Modify: `app/alembic/env.py`

- [x] **Step 1: Add the filter module**

```python
"""Alembic autogenerate filters.

Application migrations own only application tables. LangGraph checkpoint
tables and Alembic's version table are managed externally and must never be
dropped by app-model autogenerate.
"""

from __future__ import annotations

from typing import Any

EXTERNAL_TABLE_NAMES = {
    "alembic_version",
    "checkpoint_blobs",
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
}


def include_name(
    name: str | None,
    type_: str,
    parent_names: dict[str, Any],
) -> bool:
    if type_ == "table" and name in EXTERNAL_TABLE_NAMES:
        return False
    return True
```

- [x] **Step 2: Wire the filter into Alembic**

In `app/alembic/env.py`, import `include_name` and pass it to both `context.configure(...)` calls:

```python
from app.alembic.autogenerate_filters import include_name
```

```python
context.configure(
    url=url,
    target_metadata=target_metadata,
    literal_binds=True,
    dialect_opts={"paramstyle": "named"},
    include_name=include_name,
)
```

```python
context.configure(
    connection=connection,
    target_metadata=target_metadata,
    include_name=include_name,
)
```

- [x] **Step 3: Add unit coverage**

```python
from app.alembic.autogenerate_filters import EXTERNAL_TABLE_NAMES, include_name


def test_external_tables_are_excluded_from_autogenerate():
    for table_name in EXTERNAL_TABLE_NAMES:
        assert include_name(table_name, "table", {}) is False


def test_application_tables_are_included_in_autogenerate():
    assert include_name("messages", "table", {}) is True
    assert include_name("document_chunks", "table", {}) is True
```

- [x] **Step 4: Verify checkpoint drop operations disappear from Alembic check output**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_alembic_autogenerate_filters.py -q
.\.venv\Scripts\alembic.exe check
```

Expected after this task: pytest PASS. `alembic check` may still fail, but its output must no longer include `remove_table` for `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, or `checkpoint_migrations`.

## Task 3: Align ORM Metadata With The Live App Schema

**Files:**

- Modify ORM model files listed in "Target File Structure"
- Create: `app/alembic/versions/v1w2x3y4z5a6_schema_contract_cleanup.py`
- Modify: `tests/test_database_schema_contract.py` if the final index contract is more specific than Task 1

- [x] **Step 1: Remove redundant primary-key indexes from models**

Remove `index=True` from primary-key `id` columns. PostgreSQL primary keys already create indexes. Apply this to models such as `User`, `Conversation`, `Message`, `Feedback`, `TaskPlan`, `ToolApproval`, `ClientDevice`, `CustomAgent`, `SkillSetting`, `AgentModelConfig`, `ModelProvider`, `ToolApprovalSetting`, `ToolResultBlob`, and `UserMemory`.

- [x] **Step 2: Replace implicit duplicate indexes with explicit named indexes**

Use explicit `Index(...)` or `UniqueConstraint(...)` in `__table_args__` when the index name is part of the schema contract.

Specific decisions:

- Keep `idx_agent_model_configs_user_id`; remove `ix_agent_model_configs_user_id`.
- Keep `idx_model_providers_user_id`; remove `ix_model_providers_user_id`.
- Keep `idx_document_images_document_id`; remove `ix_document_images_document_id`.
- Keep `idx_document_images_chunk_id`; remove `index=True` on `DocumentImage.chunk_id`.
- Keep the unique constraint `document_chunks_qdrant_point_id_key`; remove the non-unique `idx_document_chunks_qdrant_point_id`.
- Keep `uq_document_chunk_document_index`.
- Match existing `conversation_memory_summaries` names: `ux_conversation_memory_summaries_conversation_id` and `ix_conversation_memory_summaries_last_message`, or explicitly migrate the live DB to the model names in the same revision. Prefer matching the current DB names to avoid unnecessary rename churn.

- [x] **Step 3: Align `ClientDevice.status` metadata**

Keep the DB column as `VARCHAR(32)` and make SQLAlchemy metadata match it. Use either `SQLEnum(..., native_enum=False, length=32, values_callable=...)` or replace the column with `String(32)` plus Python-level validation in the service/schema layer. Prefer the smaller change: add `length=32` to the current non-native enum metadata.

- [x] **Step 4: Drop obsolete `conversation_device_bindings`**

Create `app/alembic/versions/v1w2x3y4z5a6_schema_contract_cleanup.py` with:

```python
"""schema contract cleanup

Revision ID: v1w2x3y4z5a6
Revises: g0h1i2j3k4l5
Create Date: 2026-07-08
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "v1w2x3y4z5a6"
down_revision: str | None = "g0h1i2j3k4l5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_agent_model_configs_user_id", table_name="agent_model_configs")
    op.drop_index("ix_agent_model_configs_id", table_name="agent_model_configs")
    op.drop_index("ix_model_providers_user_id", table_name="model_providers")
    op.drop_index("ix_model_providers_id", table_name="model_providers")
    op.drop_index("ix_document_images_document_id", table_name="document_images")
    op.drop_index("idx_document_chunks_qdrant_point_id", table_name="document_chunks")

    for table_name, index_name in (
        ("client_devices", "ix_client_devices_id"),
        ("conversation_custom_agents", "ix_conversation_custom_agents_id"),
        ("conversations", "ix_conversations_id"),
        ("custom_agents", "ix_custom_agents_id"),
        ("document_parse_artifacts", "ix_document_parse_artifacts_id"),
        ("feedbacks", "ix_feedbacks_id"),
        ("messages", "ix_messages_id"),
        ("skill_settings", "ix_skill_settings_id"),
        ("task_plans", "ix_task_plans_id"),
        ("tool_approval_settings", "ix_tool_approval_settings_id"),
        ("tool_approvals", "ix_tool_approvals_id"),
        ("users", "ix_users_id"),
    ):
        op.drop_index(index_name, table_name=table_name)

    op.drop_table("conversation_device_bindings")


def downgrade() -> None:
    op.create_table(
        "conversation_device_bindings",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("device_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("bound_by_user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "bound_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["bound_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"]),
        sa.ForeignKeyConstraint(["device_id"], ["client_devices.id"]),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversation_device_bindings_bound_at",
        "conversation_device_bindings",
        ["bound_at"],
    )
    op.create_index(
        "ix_conversation_device_bindings_bound_by_user_id",
        "conversation_device_bindings",
        ["bound_by_user_id"],
    )
    op.create_index(
        "ix_conversation_device_bindings_conversation_id",
        "conversation_device_bindings",
        ["conversation_id"],
    )
    op.create_index(
        "ix_conversation_device_bindings_device_id",
        "conversation_device_bindings",
        ["device_id"],
    )

    op.create_index("ix_agent_model_configs_user_id", "agent_model_configs", ["user_id"])
    op.create_index("ix_agent_model_configs_id", "agent_model_configs", ["id"])
    op.create_index("ix_model_providers_user_id", "model_providers", ["user_id"])
    op.create_index("ix_model_providers_id", "model_providers", ["id"])
    op.create_index("ix_document_images_document_id", "document_images", ["document_id"])
    op.create_index("idx_document_chunks_qdrant_point_id", "document_chunks", ["qdrant_point_id"])

    for table_name, index_name in (
        ("client_devices", "ix_client_devices_id"),
        ("conversation_custom_agents", "ix_conversation_custom_agents_id"),
        ("conversations", "ix_conversations_id"),
        ("custom_agents", "ix_custom_agents_id"),
        ("document_parse_artifacts", "ix_document_parse_artifacts_id"),
        ("feedbacks", "ix_feedbacks_id"),
        ("messages", "ix_messages_id"),
        ("skill_settings", "ix_skill_settings_id"),
        ("task_plans", "ix_task_plans_id"),
        ("tool_approval_settings", "ix_tool_approval_settings_id"),
        ("tool_approvals", "ix_tool_approvals_id"),
        ("users", "ix_users_id"),
    ):
        op.create_index(index_name, table_name, ["id"])
```

- [x] **Step 5: Verify Alembic and schema tests**

Run:

```powershell
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\alembic.exe check
.\.venv\Scripts\python.exe -m pytest tests/test_database_schema_contract.py -q
```

Expected after this task: `alembic check` PASS and schema contract tests PASS. If `alembic check` still reports app-table differences, fix models or the cleanup migration in this task before proceeding.

## Task 4: Consolidate Database Session Ownership

**Files:**

- Modify: `app/database/session.py`
- Modify: `app/database/database.py`
- Modify: `app/core/container.py`
- Modify: tests that instantiate `Database(settings.database_url)`
- Create: `tests/test_database_session_provider.py`

- [x] **Step 1: Add explicit session context helpers**

Update `app/database/session.py` to expose one engine and one session factory:

```python
from collections.abc import Generator, Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.api_debug,
)

SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False, bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    return engine


def get_session_factory():
    return SessionLocal
```

Do not use `session_scope()` inside repositories that already commit internally. Use `SessionLocal` as the repository factory there.

- [x] **Step 2: Retire `Database.create_database()`**

Modify `app/database/database.py` so it no longer creates a second engine and no longer exposes `create_database()`. Keep a small compatibility wrapper only if tests or dependency-injector wiring still need `.session()`:

```python
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from app.database.session import SessionLocal


class Database:
    def __init__(self, db_url: str | None = None) -> None:
        self._session_factory = SessionLocal

    @contextmanager
    def session(self) -> Iterator[Session]:
        db = self._session_factory()
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
```

Then plan a later deletion of `Database` after container/tests no longer require it.

- [x] **Step 3: Update dependency injection**

In `app/core/container.py`, prefer `SessionLocal` or `get_session_factory()` for repositories. Keep `Database` only if a test fixture still uses `db.provided.session`.

- [x] **Step 4: Add tests**

```python
from app.database.session import SessionLocal, get_engine, get_session_factory


def test_database_session_factory_is_singleton_provider():
    assert get_session_factory() is SessionLocal


def test_database_engine_is_bound_to_session_factory():
    assert SessionLocal.kw["bind"] is get_engine()
```

- [x] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_database_session_provider.py tests/test_custom_agents_service.py tests/test_custom_agents_message_service.py tests/test_custom_agents_api.py -q
```

Expected: PASS.

## Task 5: Harden Checkpoint Cleanup And Retention

**Files:**

- Create: `app/services/checkpoint_retention_service.py`
- Modify: `app/ai/checkpoint.py`
- Modify: `app/workers/cleanup_tasks.py`
- Modify: `app/core/config.py`
- Create: `tests/test_checkpoint_retention_service.py`
- Modify: `tests/test_checkpoint_serializer.py`

- [x] **Step 1: Make checkpoint deletion schema-aware**

Update `CheckpointManager.delete_thread()` so the SQL fallback uses the configured `checkpoint_schema` and deletes in dependency-safe order:

```python
tables = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
schema = getattr(self.settings, "checkpoint_schema", "public") or "public"
for table_name in tables:
    await conn.execute(
        f'DELETE FROM "{schema}"."{table_name}" WHERE thread_id = %s',
        (normalized_thread_id,),
    )
```

Keep table names from a fixed tuple only. Do not accept table names from user input.

- [x] **Step 2: Add retention service**

`app/services/checkpoint_retention_service.py` should:

- expire stale pending HITL records first;
- delete checkpoint threads for expired HITL records;
- delete checkpoint threads for soft-deleted conversations;
- report counts for records inspected and rows/threads cleaned;
- never drop LangGraph checkpoint tables.

Public method:

```python
class CheckpointRetentionService:
    def __init__(self, checkpoint_manager, hitl_interrupt_repository, conversation_repository):
        ...

    async def cleanup_expired_and_deleted_threads(self, *, now: datetime) -> dict[str, int]:
        ...
```

- [x] **Step 3: Move Celery cleanup through the service**

Update `app/workers/cleanup_tasks.py` so `cleanup_abandoned_interrupts()` calls `CheckpointRetentionService` instead of duplicating DB/Redis/checkpoint cleanup logic.

- [x] **Step 4: Add focused tests**

Test cases:

- expired pending interrupts are marked expired;
- expired interrupt thread IDs are passed to `CheckpointManager.delete_thread`;
- unexpired pending interrupts are not deleted;
- cleanup swallows per-thread checkpoint deletion errors and continues;
- schema-qualified fallback SQL is used when `adelete_thread` is unavailable.

- [x] **Step 5: Run cleanup once in development before deleting graph compatibility**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_checkpoint_retention_service.py tests/test_checkpoint_serializer.py -q
```

Then run the cleanup task manually in the local dev environment:

```powershell
.\.venv\Scripts\python.exe -c "from app.workers.cleanup_tasks import cleanup_abandoned_interrupts; print(cleanup_abandoned_interrupts())"
```

Expected: expired pending HITL interrupts become expired, and checkpoint threads for expired interrupts are deleted. The local snapshot already showed `unexpired_pending_hitl: 0`, so this is safe for local cleanup.

## Task 6: Delete The Legacy `summarize` Graph Node

**Files:**

- Modify: `app/ai/graph.py`
- Modify: `tests/test_graph_streaming_summarization.py`
- Create or modify: `tests/test_graph_refactor_contract.py`

- [ ] **Step 1: Add a guard test that no active checkpoint needs `summarize`**

```python
from sqlalchemy import create_engine, text

from app.core.config import settings


def test_no_pending_checkpoint_requires_summarize_node():
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        count = conn.execute(
            text("SELECT COUNT(*) FROM checkpoints WHERE checkpoint::text ILIKE '%summarize%'")
        ).scalar_one()

    assert count == 0
```

Run it after Task 5 cleanup. It should pass before deleting the node.

- [ ] **Step 2: Remove the compatibility node**

Delete:

- `MultiAgentWorkflow._summarization_node`
- `workflow.add_node("summarize", self._summarization_node)`
- the `metadata.get("langgraph_node") == "summarize"` fallback in `_is_internal_stream_chunk`
- tests that assert the no-op node exists

Keep durable summary refresh in `MessageService.refresh_summary_after_turn()`.

- [ ] **Step 3: Update tests**

Replace `test_summarization_node_is_a_noop_passthrough` with:

```python
def test_workflow_has_no_summarize_node_or_method():
    import inspect

    from app.ai.graph import MultiAgentWorkflow

    assert not hasattr(MultiAgentWorkflow, "_summarization_node")
    source = inspect.getsource(MultiAgentWorkflow._build_graph)
    assert '"summarize"' not in source
    assert 'add_edge(START, "route")' in source
```

- [ ] **Step 4: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_streaming_summarization.py tests/test_message_history_pipeline.py -q
```

Expected: PASS.

## Task 7: Move Stream Projection Out Of `graph.py`

**Files:**

- Create: `app/services/event_streaming/graph_public_projection.py`
- Modify: `app/ai/graph.py`
- Modify: `app/services/ai_service.py` only if event names change internally
- Create: `tests/test_graph_stream_projection.py`
- Modify existing stream tests:
  - `tests/test_graph_streaming_tool_events.py`
  - `tests/test_graph_handoff_streaming.py`
  - `tests/test_internal_sse_stream_contract.py`
  - `tests/test_ai_sdk_v6_stream_contract.py`

- [ ] **Step 1: Extract `_StreamMapCtx` and stream mapping helpers**

Move these from `MultiAgentWorkflow` into `graph_public_projection.py`:

- `_StreamMapCtx`
- `_consume_stream_text_chunk`
- `_map_v3_stream_event`
- `_emit_tool_start_from_canonical`
- `_map_legacy_message_chunk`
- `_map_legacy_update_node`
- `_map_v3_values_snapshot`

Expose:

```python
class GraphPublicStreamProjector:
    def __init__(self, *, tool_end_events_from_node_state, suppress_internal_stream_chunks: bool):
        ...

    def map_event(self, event, ctx: StreamProjectionContext):
        ...
```

Keep behavior identical: graph stream dicts still include `token`, `thinking`, `tool_start`, `tool_end`, `node_complete`, `continuation_start`, `agent_selected`, `interrupt`, `complete`, and `error` until `AIService` maps them to canonical `V3StreamEvent`.

- [ ] **Step 2: Make `MultiAgentWorkflow` delegate projection**

In `execute_request_stream()` and `resume_with_decisions_stream()`, instantiate the projector and call `projector.map_event(event, ctx)`.

- [ ] **Step 3: Add projector unit tests**

Cover:

- cumulative text chunk delta calculation;
- tool-call dedupe;
- `ToolMessage` to `tool_end`;
- handoff `selected_agent` to second `agent_selected`;
- planning node snapshot to `node_complete`;
- subagent events pass through unchanged.

- [ ] **Step 4: Verify no wire contract change**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_streaming_tool_events.py tests/test_graph_handoff_streaming.py tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/test_graph_stream_projection.py -q
```

Expected: PASS. Do not update `plans/AI_SDK_FE_CONTRACT.md` in this task because the public AI SDK stream must remain unchanged.

## Task 8: Extract Graph Topology And Node Domains

**Files:**

- Create: `app/ai/workflow/graph_builder.py`
- Create: `app/ai/workflow/custom_agents.py`
- Create: `app/ai/workflow/tool_loop.py`
- Create: `app/ai/workflow/rag_loop.py`
- Create: `app/ai/workflow/planning_loop.py`
- Modify: `app/ai/graph.py`
- Create: `tests/test_graph_refactor_contract.py`

- [ ] **Step 1: Extract topology first**

Move `_build_graph()` body to:

```python
def build_workflow_graph(workflow: Any, *, checkpointer: Any | None):
    ...
```

`MultiAgentWorkflow._build_graph()` becomes a thin call:

```python
def _build_graph(self) -> StateGraph:
    return build_workflow_graph(self, checkpointer=self.checkpointer)
```

- [ ] **Step 2: Add topology contract tests**

Assertions:

- `START` routes directly to `route`;
- no `summarize` node exists;
- top-level route targets include all base agents plus `custom_agent`;
- standard tool-calling agents route through `approval`, `tools`, or `END`;
- `planning_tools` can route to every base agent plus `custom_agent` and `END`.

- [ ] **Step 3: Extract custom-agent helpers**

Move custom-agent-specific helpers to `app/ai/workflow/custom_agents.py`:

- `_resolve_runtime_agent`
- `_custom_handoff_targets`
- `_custom_handoff_target_descriptions`
- `_multi_agent_kwargs`
- `_build_multi_agent_activity_block`
- `_reset_agent_trail`
- `_record_agent_invocation`
- `_custom_agent_descriptors`
- `_sticky_custom_agent`
- `_build_custom_agent`
- `_is_attached_custom_agent`
- `_route_target_for` if it only exists to support custom routing

Use functions or a small helper class. Keep `MultiAgentWorkflow` as the orchestrator and avoid a mixin hierarchy unless it is the smallest low-risk step.

- [ ] **Step 4: Extract generic tool/HITL loop helpers**

Move generic tool-loop code to `app/ai/workflow/tool_loop.py`:

- `_tool_node`
- `_apply_hand_off_if_present`
- `_needs_approval`
- `_prepare_interrupt_payload`
- `_approval_node`
- `_execute_agent_tool_calls`
- `_apply_tool_outputs_to_state`
- `_tool_error_signature`
- `_update_tool_error_streak`
- `_tool_end_events_from_node_state`

Keep all current test names passing; only imports or monkeypatch paths should change.

- [ ] **Step 5: Extract RAG loop helpers**

Move RAG-specific orchestration to `app/ai/workflow/rag_loop.py`:

- `_rag_node`
- `_rag_tools_node`
- `_should_call_rag_tools`
- `_should_continue_rag`
- the RAG branch of `_run_agent_in_isolated_context`

Do not move `RAGAgent` retrieval internals in this task.

- [ ] **Step 6: Extract planning loop helpers**

Move planning-specific orchestration to `app/ai/workflow/planning_loop.py`:

- `_build_planning_internal_tools`
- `_planning_node`
- `_review_planning_todos_with_rubric`
- `_planning_tools_node`
- `_should_call_planning_tools`
- `_should_continue_planning`
- the generic worker branch of `_run_agent_in_isolated_context`

- [ ] **Step 7: Keep public imports stable**

`app.ai.graph.MultiAgentWorkflow` and `app.ai.graph.create_workflow` must continue to exist. Do not require callers to import from the new modules.

- [ ] **Step 8: Verify graph tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph*.py tests/test_custom_agents_graph.py tests/test_custom_agents_planning.py tests/test_rag_tool_loop_finalization.py tests/test_rag_agent.py -q
```

Expected: PASS. If monkeypatch paths such as `app.ai.graph.execute_tool_calls` fail, keep re-export aliases in `app/ai/graph.py` until tests and call sites are updated.

## Task 9: Clean Repository Patterns Without Broad Rewrites

**Files:**

- Modify: `app/repositories/command_strategy.py`
- Modify: `app/repositories/query_strategy.py`
- Modify repositories that inherit generic strategies:
  - `app/repositories/user.py`
  - `app/repositories/conversation.py`
  - `app/repositories/message.py`
  - `app/repositories/feedback.py`
  - `app/repositories/task_plan.py`
  - `app/repositories/tool_approval.py`
- Add focused tests only where behavior changes.

- [ ] **Step 1: Make generic strategies safe for models without `deleted_at`**

Either delete the generic strategies and inline repository methods, or update them to check `hasattr(self.model, "deleted_at")` before filtering. Prefer deletion only when the repository method is already custom enough that the strategy adds no value.

- [ ] **Step 2: Replace inefficient counts**

Change `DefaultQueryStrategy.count_all()` from materializing rows to SQL `COUNT`.

Implementation:

```python
from sqlalchemy import func, select

def count_all(self, db: Session) -> int:
    statement = select(func.count(self.model.id))
    if hasattr(self.model, "deleted_at"):
        statement = statement.where(self.model.deleted_at.is_(None))
    return int(db.execute(statement).scalar() or 0)
```

- [ ] **Step 3: Audit soft-delete expectations**

Rules:

- Tables with `deleted_at`: `users`, `conversations`, `messages`, `feedbacks`, `custom_agents`, `model_providers`, `tool_approvals`, `tool_result_blobs`, `user_memories`.
- Tables without `deleted_at`: do not route through a strategy that assumes soft delete.

- [ ] **Step 4: Verify repository tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_conversation_memory_summary_repository.py tests/test_tool_approval_setting_repository.py tests/test_document_parse_artifact_repository.py tests/test_document_service_deletion.py tests/test_custom_agents_service.py -q
```

Expected: PASS.

## Task 10: Verify API Contract And Documentation

**Files:**

- Modify: `plans/AI_SDK_FE_CONTRACT.md` only if behavior changes
- Modify: `README.md` for database/checkpoint ownership updates

- [ ] **Step 1: Confirm no AI SDK stream change**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ai_sdk_v6_stream_contract.py tests/test_ai_sdk_assistant_ui_compat.py tests/test_ai_sdk_context_window.py tests/test_rich_response_streaming.py -q
```

Expected: PASS.

- [ ] **Step 2: Update `plans/AI_SDK_FE_CONTRACT.md` only for contract changes**

If implementation changes endpoint paths, event names, terminal stream order, metadata keys, resume behavior, or rich item delivery, update the matching sections in `plans/AI_SDK_FE_CONTRACT.md` in the same task as the code change.

No update is required if all stream tests pass unchanged and the event catalog remains the same.

- [ ] **Step 3: Update README database notes**

Document:

- Alembic owns app tables.
- LangGraph owns checkpoint tables.
- `conversation_device_bindings` was dropped because conversation ownership is user-based.
- `Base.metadata.create_all()` is not used in application startup.
- Checkpoint cleanup is handled by `CheckpointRetentionService` and Celery beat.

## Final Verification

Run the focused and broad checks:

```powershell
.\.venv\Scripts\alembic.exe heads --verbose
.\.venv\Scripts\alembic.exe current --verbose
.\.venv\Scripts\alembic.exe check
.\.venv\Scripts\python.exe -m pytest tests/test_database_schema_contract.py tests/test_alembic_autogenerate_filters.py tests/test_database_session_provider.py tests/test_checkpoint_retention_service.py tests/test_graph_stream_projection.py tests/test_graph_refactor_contract.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_graph*.py tests/test_custom_agents_graph.py tests/test_ai_sdk_v6_stream_contract.py tests/test_internal_sse_stream_contract.py tests/test_message_history_pipeline.py -q
.\.venv\Scripts\ruff.exe check app tests
```

Expected:

- Alembic has one head.
- Live DB is at head.
- `alembic check` reports no upgrade operations.
- No app model includes LangGraph checkpoint tables.
- `conversation_device_bindings` is gone.
- `app/ai/graph.py` is under 1,800 lines and contains only orchestration/public workflow entrypoints plus thin delegates.
- No `summarize` graph node or `_summarization_node` remains.
- Existing AI SDK and Streamlit stream tests pass without frontend contract changes.

## Rollout Notes

- Take a database backup before applying `v1w2x3y4z5a6_schema_contract_cleanup.py` outside local development.
- Apply the Alembic external-table guard before generating any new schema migration.
- Run checkpoint cleanup before deleting `summarize` compatibility.
- Keep `app.ai.graph.MultiAgentWorkflow` and `create_workflow` stable so existing tests, container wiring, and service imports do not break.
- Treat LangGraph checkpoint row pruning as operational cleanup, not an Alembic table migration.

---

## Implementation Log

Execution started 2026-07-08 at base commit `b1c61d2` on branch `Thai-Postgre-FastAPI`.
Baseline verified: single Alembic head `g0h1i2j3k4l5`; live DB at head; `alembic check` FAILS with exactly the documented drift; `graph.py` = 5202 lines.

### Design Decisions Log

_Records deviations, judgment calls, and clarifications made during implementation._

- **Task 1:** Test file written verbatim from plan. Confirmed the 14 redundant PK indexes in the live DB match the Task 3 migration's drop list exactly (2 explicit `ix_agent_model_configs_id`/`ix_model_providers_id` + 12 in the loop), so Task 3's PK-index migration is pre-verified against ground truth.

- **Task 2:** `env.py` imports `Base` from `app.database.base` (not `app.models.base` as the schema-contract test uses); both aggregate the same model metadata (verified: alembic detected all app tables; the contract test found all 14 offenders). `include_name` wired into both offline and online `context.configure` calls.
- **Task 3:**
  - Step 1 bulk removal (16 identical PK-`id` `index=True` lines across 15 files) done via a verified full-line Python string replacement rather than 16 read+edit round-trips. Two of those (`tool_result_blobs`, `user_memories`) had no live index yet — removal just prevents autogenerate from adding one.
  - `document_chunks.document_id` (gap in plan Step 2): model produced `ix_document_chunks_document_id` via `index=True` while live had `idx_document_chunks_document_id`. Resolved per the plan's "match live names to avoid rename churn" rule — removed `index=True`, added explicit `Index("idx_document_chunks_document_id", "document_id")`. Migration leaves it untouched (already matches).
  - `document_chunks.qdrant_point_id`: changed `unique=True, index=True` → `unique=True` only, so metadata declares a unique *constraint* (Postgres-named `document_chunks_qdrant_point_id_key`, matching live) instead of a unique index; migration drops the redundant plain `idx_document_chunks_qdrant_point_id`.
  - `conversation_memory_summaries`: added `__table_args__` with explicit `ux_conversation_memory_summaries_conversation_id` (unique) and `ix_conversation_memory_summaries_last_message`, matching live names; kept `user_id` `index=True` (matches live `ix_conversation_memory_summaries_user_id`).
  - `ClientDevice.status`: added `length=32` to the existing non-native `SQLEnum` (smallest change, per plan).
  - **INCIDENT + lesson:** verifying downgrade reversibility, I ran an alembic round-trip in the background concurrently with a foreground check. Two alembic processes deadlocked on DDL locks against the running uvicorn dev server, leaving orphaned migration backends and the DB downgraded. Reversibility itself was confirmed (downgrade `v1w2x3y4z5a6`→`g0h1i2j3k4l5` ran clean). Recovery required terminating 4 stuck DB sessions (user-authorized after classifier denials). **Lesson: never run DB-mutating alembic in the background or concurrently — always sequential/foreground.** Re-applied migration cleanly with `lock_timeout=8s`.
- **Task 4:**
  - Step 3 (DI) kept intentionally minimal: rather than rewrite ~25 `session_factory=db.provided.session` sites in `container.py`, `Database` was made a thin adapter over the shared `SessionLocal`/engine from `app.database.session`. This achieves the single-engine goal (the actual bug: `Database` previously built a *second* engine) without a risky 25-site DI churn, and keeps `db.provided.session` and the 4 test files that do `Database(settings.database_url)` working unchanged. Verified: `Container().db().session()` binds to the single shared engine.
  - `SessionLocal` gained `expire_on_commit=False` (per plan target) so ORM objects stay usable after commit/session-close (avoids DetachedInstanceError); 46 repo/service/API regression tests pass with it.
  - `create_database()` (the only `Base.metadata.create_all` caller) removed — migrations are now the sole schema-mutation path. `database.py` no longer imports `Base`.
- **Task 5** (subagent-implemented, controller-verified live):
  - New `CheckpointRetentionService.cleanup_expired_and_deleted_threads(*, now)` returns counts: `pending_interrupts_inspected`, `hitl_interrupts_expired`, `hitl_checkpoint_threads_deleted`, `soft_deleted_conversations_inspected`, `conversation_checkpoint_threads_deleted`. Order: expire HITL first (DB authoritative) → delete their threads → sweep soft-deleted conversations. Per-thread errors swallowed+logged; never issues DDL.
  - `CheckpointManager.delete_thread()` SQL fallback now schema-qualified (`"<schema>"."<table>"`) and dependency-safe order (`checkpoint_writes`→`checkpoint_blobs`→`checkpoints`); fixed tuple only.
  - `cleanup_abandoned_interrupts()` delegates to the service; Redis scan preserved+guarded; return shape preserved (one additive key `soft_deleted_conversations_inspected`).
  - New repo method `ConversationRepository.get_soft_deleted()` (genuinely missing — the rest of the repo filters `deleted_at IS NULL` everywhere). Verified thread_id==`str(conversation.id)` matches `ConversationService.delete_conversation`'s convention.
  - **Controller bug fix during live verify:** the task's `asyncio.new_event_loop()` produced a Windows ProactorEventLoop → psycopg async pool `PoolTimeout` in `CheckpointManager.setup()`; retention silently returned zeros. Fixed by creating a SelectorEventLoop on `win32` via `WindowsSelectorEventLoopPolicy().new_event_loop()` (mirrors the existing `app/main.py:7-9` convention). This was a latent pre-existing incompatibility (original per-invocation manager had the same pattern), surfaced only by actually running the task.
  - **Live Step 5 result:** 64 pending-expired HITL → expired; 837 checkpoint threads cleaned; 797 soft-deleted conversations swept; after-state = 0 pending HITL. This satisfies Task 6's precondition (no active interrupts needing the old node). Unit tests 12/12, ruff clean.
  - Non-blocking follow-up (subagent concern #2): `get_soft_deleted()` is unbounded with no "cleaned" marker, so each 10-min beat re-sweeps all soft-deleted rows (idempotent no-ops). Candidate for a future pagination/marker task; out of Task 5 scope.

### Progress

- **Task 1 — DONE** (commit 899bff9). `tests/test_database_schema_contract.py` created. RED confirmed: 3 failed (unmodeled `conversation_device_bindings`, table present, 14 redundant PK indexes), 2 passed (checkpoint ownership disjoint, 0 unexpired pending HITL). Matches plan's expected pre-implementation state.
- **Task 2 — DONE** (commit e328e9e). `autogenerate_filters.py` + `test_alembic_autogenerate_filters.py` created; `env.py` wired. Filter tests 2/2 pass; `alembic check` output no longer references any checkpoint table (still fails overall on remaining app-table drift, as expected).
- **Task 3 — DONE** (commit 4916e88). 18 model files edited + `v1w2x3y4z5a6_schema_contract_cleanup.py` migration created. Live DB migrated to `v1w2x3y4z5a6`. Verified: `alembic check` = "No new upgrade operations detected"; schema contract test 5/5 green; regression (memory-summary repo 7/7, document chunk/model/parse-artifact 25/25) green. Downgrade reversibility confirmed. `conversation_device_bindings` dropped.
- **Task 4 — DONE** (commit 020ec02). `session.py` (one engine + `SessionLocal` w/ `expire_on_commit=False` + `session_scope()`), `database.py` (thin adapter over shared `SessionLocal`, no 2nd engine, `create_database` removed), `test_database_session_provider.py` created. Verified: provider tests + custom-agents service/message/api + hitl_api = 46/46 pass; container session bound to single shared engine.
- **Task 5 — DONE** (commit pending). Subagent-implemented, controller-verified. `checkpoint_retention_service.py` + `test_checkpoint_retention_service.py` created; `checkpoint.py` (schema-aware fallback), `cleanup_tasks.py` (delegates + Windows selector-loop fix), `conversation.py` (`get_soft_deleted`), `test_checkpoint_serializer.py` updated. Unit 12/12; live cleanup expired 64 HITL + cleaned 837 threads; after-state 0 pending HITL; ruff clean.

