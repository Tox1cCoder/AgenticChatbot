from __future__ import annotations

import importlib
import inspect


def _migration_module():
    return importlib.import_module(
        "app.alembic.versions.x1y2z3a4b5c6_production_conversation_compaction"
    )


def test_compaction_migration_is_the_single_new_head():
    migration = _migration_module()

    assert migration.revision == "x1y2z3a4b5c6"
    assert migration.down_revision == "w7x8y9z0a1b2"


def test_upgrade_backfills_sequences_deterministically():
    source = inspect.getsource(_migration_module().upgrade).lower()

    assert "row_number() over" in source
    assert "partition by conversation_id" in source
    assert "order by created_at, id" in source
    assert "max(sequence) + 1" in source
    assert "uq_messages_conversation_sequence" in source
    assert "ix_messages_prompt_history" in source


def test_upgrade_invalidates_legacy_memory_and_creates_durable_jobs():
    source = inspect.getsource(_migration_module().upgrade).lower()

    assert "conversation_memory_summaries_legacy" in source
    assert "'{}'::jsonb" in source
    assert "false" in source
    assert "conversation_summary_jobs" in source
    assert "fk_memory_summary_conversation_sequence" in source
    assert "fk_summary_job_conversation_sequence" in source


def test_downgrade_restores_legacy_summary_shape_and_removes_sequences():
    source = inspect.getsource(_migration_module().downgrade).lower()

    assert "summary_text" in source
    assert "estimated_tokens" in source
    assert 'drop_column("messages", "sequence")' in source
    assert 'drop_column("conversations", "next_message_sequence")' in source
