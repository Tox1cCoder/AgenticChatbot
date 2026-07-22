"""Contract checks for cascading raw usage when a conversation is deleted."""

from __future__ import annotations

import inspect
from importlib import import_module


def test_conversation_fk_cascade_migration_contract() -> None:
    module = import_module(
        "app.alembic.versions.c6d7e8f9a0b1_cascade_model_usage_events_conversation"
    )

    assert module.revision == "c6d7e8f9a0b1"
    assert module.down_revision == "b5c6d7e8f9a0"
    upgrade = inspect.getsource(module.upgrade)
    downgrade = inspect.getsource(module.downgrade)
    helper = inspect.getsource(module._replace_conversation_fk)
    assert 'ondelete="CASCADE"' in upgrade
    assert 'ondelete="SET NULL"' in downgrade
    assert "fk_model_usage_events_conversation" in helper
    assert "op.drop_constraint" in helper
    assert "op.create_foreign_key" in helper
