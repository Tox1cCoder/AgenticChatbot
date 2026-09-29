"""Model-side server defaults must render as valid PostgreSQL DDL.

``create_all`` builds tables from these declarations (the integration suites
do), so a default PostgreSQL rejects is a table that cannot be created.
"""

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.models.task_plan import TaskPlan


def test_task_metadata_default_is_a_jsonb_expression_not_a_quoted_string():
    ddl = str(CreateTable(TaskPlan.__table__).compile(dialect=postgresql.dialect()))

    assert "task_metadata JSONB DEFAULT '{}'::jsonb" in ddl
