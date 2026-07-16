"""Structural guard for the ToolApprovalSetting model + migration registration."""

from pathlib import Path

import app.models as models
from app.models.tool_approval_setting import ToolApprovalSetting


def test_model_columns_and_table():
    assert ToolApprovalSetting.__tablename__ == "tool_approval_settings"
    cols = set(ToolApprovalSetting.__table__.columns.keys())
    assert {
        "id",
        "created_at",
        "updated_at",
        "user_id",
        "scope_type",
        "scope_value",
        "require_approval",
    } <= cols
    uniques = {
        tuple(sorted(c.name for c in con.columns))
        for con in ToolApprovalSetting.__table__.constraints
        if con.__class__.__name__ == "UniqueConstraint"
    }
    assert ("scope_type", "scope_value", "user_id") in uniques
    checks = {
        str(con.sqltext)
        for con in ToolApprovalSetting.__table__.constraints
        if con.__class__.__name__ == "CheckConstraint"
    }
    assert any("scope_type" in check and "server" in check and "tool" in check for check in checks)


def test_model_exported():
    assert "ToolApprovalSetting" in models.__all__
    assert models.ToolApprovalSetting is ToolApprovalSetting


def test_migration_chains_from_current_head():
    text = Path("app/alembic/versions/g0h1i2j3k4l5_add_tool_approval_settings.py").read_text(
        encoding="utf-8"
    )
    assert 'revision: str = "g0h1i2j3k4l5"' in text
    assert 'down_revision: str | None = "f03e63aa5a33"' in text
    assert "tool_approval_settings" in text
