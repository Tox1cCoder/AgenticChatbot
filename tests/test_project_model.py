"""Shape of the project tables and the conversation membership column."""

from sqlalchemy import inspect

from app.models.conversation import Conversation
from app.models.project import Project, ProjectCustomAgent


def test_project_columns():
    columns = {c.name for c in inspect(Project).columns}
    assert columns == {
        "id",
        "owner_id",
        "name",
        "description",
        "instructions",
        "created_at",
        "updated_at",
        "deleted_at",
    }


def test_project_has_no_slug_and_no_unique_name():
    """Duplicate project names are allowed, matching Claude."""
    columns = {c.name for c in inspect(Project).columns}
    assert "slug" not in columns

    unique_constraints = {c.name for c in Project.__table__.constraints}
    assert "uq_projects_owner_name" not in unique_constraints


def test_project_custom_agent_columns():
    columns = {c.name for c in inspect(ProjectCustomAgent).columns}
    assert columns == {
        "id",
        "created_at",
        "owner_id",
        "project_id",
        "custom_agent_id",
        "agent_order",
    }


def test_project_custom_agent_is_unique_per_project_and_agent():
    names = {c.name for c in ProjectCustomAgent.__table__.constraints}
    assert "uq_project_custom_agents_project_agent" in names


def test_conversation_project_id_is_nullable():
    column = Conversation.__table__.columns["project_id"]
    assert column.nullable is True
    assert {fk.column.table.name for fk in column.foreign_keys} == {"projects"}


def test_conversation_has_project_listing_index():
    assert "ix_conversations_project_updated" in {
        index.name for index in Conversation.__table__.indexes
    }
