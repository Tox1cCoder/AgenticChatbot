"""Structural guard for the device-scoped HITL settings migration."""

from pathlib import Path

MIGRATION = Path("app/alembic/versions/z3a4b5c6d7e8_device_scope_tool_approval_settings.py")


def test_migration_follows_current_head_and_resets_ambiguous_legacy_rows():
    text = MIGRATION.read_text(encoding="utf-8")

    assert 'revision: str = "z3a4b5c6d7e8"' in text
    assert 'down_revision: str | None = "y2z3a4b5c6d7"' in text
    assert 'op.execute("DELETE FROM tool_approval_settings")' in text
    assert text.index('op.execute("DELETE FROM tool_approval_settings")') < text.index(
        'op.alter_column("tool_approval_settings", "device_id", nullable=False)'
    )


def test_migration_installs_device_origin_constraints_and_indexes():
    text = MIGRATION.read_text(encoding="utf-8")

    for expected in (
        'sa.Column("device_id", UUID(as_uuid=True), nullable=True)',
        'sa.Column("tool_origin", sa.String(length=32), nullable=True)',
        '"fk_tool_approval_settings_device_id_client_devices"',
        '"ck_tool_approval_settings_tool_origin"',
        '"uq_tool_approval_settings_user_device_origin_scope"',
        '"ix_tool_approval_settings_device_id"',
        '"ix_tool_approval_settings_tool_origin"',
    ):
        assert expected in text


def test_downgrade_restores_legacy_shape_without_claiming_to_restore_data():
    text = MIGRATION.read_text(encoding="utf-8")

    assert 'op.drop_column("tool_approval_settings", "tool_origin")' in text
    assert 'op.drop_column("tool_approval_settings", "device_id")' in text
    assert '"uq_tool_approval_settings_user_scope"' in text
    assert "Legacy rows cannot be restored" in text
