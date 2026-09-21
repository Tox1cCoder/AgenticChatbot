"""Add projects, project default agents, and conversation membership.

Purely additive. Existing conversations get ``project_id = NULL`` and behave
exactly as before, so there is no backfill and no data migration.

No enum type is created here. If a later revision adds one to these tables,
pass ``create_type=False`` to the column type — ``op.create_table`` re-creates
a column's enum without ``checkfirst`` and will fail on a second run.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "9a8b7c6d5e4f"
down_revision: str | None = "a3b4c5d6e7f9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
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
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("instructions", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name="fk_projects_owner_id_users"),
    )
    op.create_index("ix_projects_owner_id", "projects", ["owner_id"])
    op.create_index("ix_projects_owner_deleted", "projects", ["owner_id", "deleted_at"])

    op.create_table(
        "project_custom_agents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("custom_agent_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent_order", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_project_custom_agents_owner_id_users"
        ),
        sa.ForeignKeyConstraint(
            ["project_id"], ["projects.id"], name="fk_project_custom_agents_project_id_projects"
        ),
        sa.ForeignKeyConstraint(
            ["custom_agent_id"],
            ["custom_agents.id"],
            name="fk_project_custom_agents_custom_agent_id_custom_agents",
        ),
        sa.UniqueConstraint(
            "project_id", "custom_agent_id", name="uq_project_custom_agents_project_agent"
        ),
    )
    op.create_index(
        "ix_project_custom_agents_owner_project",
        "project_custom_agents",
        ["owner_id", "project_id"],
    )

    op.add_column(
        "conversations",
        sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_conversations_project_id_projects",
        "conversations",
        "projects",
        ["project_id"],
        ["id"],
    )
    op.create_index("ix_conversations_project_id", "conversations", ["project_id"])
    op.create_index(
        "ix_conversations_project_updated",
        "conversations",
        ["project_id", "updated_at"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_conversations_project_updated", table_name="conversations")
    op.drop_index("ix_conversations_project_id", table_name="conversations")
    op.drop_constraint("fk_conversations_project_id_projects", "conversations", type_="foreignkey")
    op.drop_column("conversations", "project_id")

    op.drop_index("ix_project_custom_agents_owner_project", table_name="project_custom_agents")
    op.drop_table("project_custom_agents")

    op.drop_index("ix_projects_owner_deleted", table_name="projects")
    op.drop_index("ix_projects_owner_id", table_name="projects")
    op.drop_table("projects")
