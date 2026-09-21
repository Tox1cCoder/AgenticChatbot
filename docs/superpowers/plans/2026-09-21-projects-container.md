# Projects Container (Slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a user-owned Project that groups conversations, applies shared instructions to all of them, and seeds a default set of custom agents into conversations created in or moved into it.

**Architecture:** Three new tables (`projects`, `project_custom_agents`, plus a nullable `conversations.project_id`), all additive. Project instructions are composed into the *existing* `persona` string at its three assembly sites rather than added as a new pipeline field, so `app/ai/` is untouched. Default agents are copied into `conversation_custom_agents` rows at create and attach time, so the request-time resolver `build_runtime_state` is untouched.

**Tech Stack:** Python 3.13 (`.venv`), FastAPI, SQLAlchemy 2.x (Core `select`/`update` style), Alembic, PostgreSQL, `dependency_injector`, Pydantic v2, Streamlit (`demo.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-projects-container-design.md`

## Global Constraints

- **Run everything with `.venv/Scripts/python.exe`.** The conda `agents` env is Python 3.11 and cannot run this suite. If the app misbehaves while tests are green, check the interpreter first.
- **Line length 100.** `E501` is enforced by ruff. Run `.venv/Scripts/python.exe -m ruff check <paths>` before every commit.
- **Zero warnings.** The suite is fully green today, so there is no pre-existing failure to attribute a new one to.
- **Instruction cap: 8000 characters per part, applied independently.** Never call `sanitize_persona` on a composed string.
- **Error contract:** `ProjectForbiddenError` 403 `PROJECT_FORBIDDEN` when the project exists but belongs to another user; `ProjectNotFoundError` 404 `PROJECT_NOT_FOUND` when missing or soft-deleted; `ProjectConversationNotFoundError` 404 `PROJECT_CONVERSATION_NOT_FOUND` when a conversation is not in the named project.
- **Alembic head at plan time is `a3b4c5d6e7f9`.** The new revision is `9a8b7c6d5e4f`. If `alembic heads` reports something else when you start, use that instead and say so in the commit.
- **No new enum types.** This repository has been bitten by `op.create_table` re-creating a column's enum without `checkfirst`. Nothing here needs an enum; if you find yourself adding one, use `create_type=False`.
- **Do not use `git stash`** and do not run a broad `git add`. Stage the exact paths each task lists — other sessions may share this working tree.
- **Commit message attribution:** end every commit body with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

---

## File Structure

**Created:**

| Path | Responsibility |
| --- | --- |
| `app/models/project.py` | `Project` and `ProjectCustomAgent` ORM models |
| `app/alembic/versions/9a8b7c6d5e4f_add_projects.py` | additive migration |
| `app/core/exceptions/project.py` | the four project exception classes |
| `app/schemas/project.py` | `ProjectCreate`, `ProjectUpdate`, `ProjectRead`, `ProjectCustomAgentsUpdate` |
| `app/repositories/project.py` | `ProjectRepository` — all project SQL, including seeding |
| `app/services/project_service.py` | `ProjectService` — CRUD, ownership, agent set, attach/detach |
| `app/services/project_context_service.py` | `ProjectContextService` — resolves a conversation's system instruction |
| `app/api/projects.py` | the project router |
| `plans/PROJECTS_FE_CONTRACT.md` | frontend contract |
| `tests/test_project_instruction_composition.py` | composition unit tests |
| `tests/test_project_model.py` | model and migration shape |
| `tests/test_project_repository.py` | repository behavior |
| `tests/test_project_service.py` | service behavior and ownership |
| `tests/test_project_context_service.py` | resolver + the three call sites |
| `tests/test_projects_api.py` | HTTP contract and cross-user access |
| `tests/test_project_membership.py` | conversation create/list with a project |
| `tests/test_demo_projects.py` | Streamlit helpers and sidebar |

**Modified:**

| Path | Change |
| --- | --- |
| `app/utils/text_processing.py` | add `compose_system_instruction` |
| `app/models/conversation.py` | `project_id` column + partial index |
| `app/models/__init__.py` | export `Project`, `ProjectCustomAgent` |
| `app/schemas/conversation.py` | `project_id` on `ConversationCreate` and `ConversationRead` |
| `app/factories/conversation_factory.py` | `project_id` in both factory methods |
| `app/services/conversation_service.py` | validate project on create, seed agents, `project_id` filter |
| `app/repositories/conversation.py` | `project_id` filter on the owner listing |
| `app/api/conversations.py` | `projectId` query parameter |
| `app/services/ai_service.py` | use the resolver in `_prepare_request` |
| `app/services/message_service.py` | use the resolver at both persona sites |
| `app/core/container.py` | register the three new providers, wire `app.api.projects` |
| `app/main.py` | include the projects router |
| `demo.py` | project client helpers, sidebar section, project view |

Files that change together live together: all project SQL is in one repository, all project HTTP is in one router. `message_service.py` is already 181K and unwieldy, but this plan does not restructure it — the change there is two call-site replacements, and a split is out of scope.

---

## Task 1: Instruction composition

Pure text function with no dependencies. Nothing else in the plan can be correct if this is wrong, so it goes first.

**Files:**
- Modify: `app/utils/text_processing.py` (add after `sanitize_persona`, which ends at line 130)
- Test: `tests/test_project_instruction_composition.py`

**Interfaces:**
- Consumes: `sanitize_persona(persona: str | None) -> str | None` from the same module.
- Produces: `compose_system_instruction(project_instructions: str | None, persona_prompt: str | None) -> str | None`. Task 5 is its only caller.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_project_instruction_composition.py`:

```python
"""Composition of project-level and conversation-level instructions."""

from app.utils.text_processing import compose_system_instruction, sanitize_persona


def test_returns_none_when_both_absent():
    assert compose_system_instruction(None, None) is None
    assert compose_system_instruction("   ", "") is None


def test_persona_only_passes_through_unchanged():
    assert compose_system_instruction(None, "Be terse.") == "Be terse."


def test_project_only_passes_through_unchanged():
    assert compose_system_instruction("Answer in Vietnamese.", None) == "Answer in Vietnamese."


def test_both_present_are_headered_with_project_first():
    result = compose_system_instruction("Answer in Vietnamese.", "Be terse.")
    assert result == (
        "Project instructions:\nAnswer in Vietnamese.\n\n"
        "Conversation-specific instructions:\nBe terse."
    )


def test_each_part_is_capped_independently_so_the_persona_survives():
    """The bug this guards: compose-then-truncate would drop the persona entirely,
    because the project text leads and the cap is 8000."""
    result = compose_system_instruction("P" * 9000, "Q" * 9000)

    # 8001, not 8000: the header "Project instructions:" contributes one P.
    assert result.count("P") == 8001
    assert result.count("Q") == 8000
    assert result.endswith("Q" * 100)


def test_project_less_output_is_byte_identical_to_the_previous_behaviour():
    """Every conversation that predates projects must render exactly as before."""
    persona = "  Be   terse.\n\n\n\nAlways answer in full sentences.  "

    assert compose_system_instruction(None, persona) == sanitize_persona(persona)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_instruction_composition.py -v`
Expected: all six FAIL with `ImportError: cannot import name 'compose_system_instruction'`.

- [ ] **Step 3: Write the implementation**

Append to `app/utils/text_processing.py`, immediately after `sanitize_persona`:

```python
PROJECT_INSTRUCTION_HEADER = "Project instructions:"
CONVERSATION_INSTRUCTION_HEADER = "Conversation-specific instructions:"


def compose_system_instruction(
    project_instructions: str | None,
    persona_prompt: str | None,
) -> str | None:
    """Combine a project's instructions with a conversation's persona.

    Each part is sanitized independently against its own 8000-character cap.
    Composing first and truncating after would silently discard the persona,
    because the project text leads — never call :func:`sanitize_persona` on
    the value returned here.

    Headers are added only when both parts are present, so a conversation
    with no project renders byte-identically to how it rendered before
    projects existed.
    """
    project = sanitize_persona(project_instructions)
    persona = sanitize_persona(persona_prompt)

    if project and persona:
        return (
            f"{PROJECT_INSTRUCTION_HEADER}\n{project}\n\n"
            f"{CONVERSATION_INSTRUCTION_HEADER}\n{persona}"
        )
    return project or persona
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_instruction_composition.py -v`
Expected: 6 passed.

- [ ] **Step 5: Lint**

Run: `.venv/Scripts/python.exe -m ruff check app/utils/text_processing.py tests/test_project_instruction_composition.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add app/utils/text_processing.py tests/test_project_instruction_composition.py
git commit -m "feat: compose project and conversation instructions"
```

---

## Task 2: Models and migration

**Files:**
- Create: `app/models/project.py`
- Create: `app/alembic/versions/9a8b7c6d5e4f_add_projects.py`
- Modify: `app/models/conversation.py:23-30` (`__table_args__`) and after line 43
- Modify: `app/models/__init__.py`
- Test: `tests/test_project_model.py`

**Interfaces:**
- Produces: `Project` (columns `id`, `owner_id`, `name`, `description`, `instructions`, `created_at`, `updated_at`, `deleted_at`), `ProjectCustomAgent` (columns `id`, `created_at`, `owner_id`, `project_id`, `custom_agent_id`, `agent_order`), and `Conversation.project_id`. Tasks 3 onward import these from `app.models.project`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_project_model.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_model.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'app.models.project'`.

- [ ] **Step 3: Create the models**

Create `app/models/project.py`:

```python
"""Models for user-owned projects and their default custom agents."""

import uuid

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base


class Project(Base):
    """A container grouping conversations under one set of instructions.

    Owner-scoped and soft-deleted, mirroring :class:`CustomAgent`. There is
    deliberately no slug and no uniqueness on ``name``: a project is addressed
    by id everywhere, and duplicate names are allowed.
    """

    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_owner_deleted", "owner_id", "deleted_at"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}', owner_id={self.owner_id})>"


class ProjectCustomAgent(Base):
    """A custom agent in a project's default set.

    Structurally identical to :class:`ConversationCustomAgent` so that seeding
    a conversation is a straight row copy rather than a translation.
    """

    __tablename__ = "project_custom_agents"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "custom_agent_id",
            name="uq_project_custom_agents_project_agent",
        ),
        Index("ix_project_custom_agents_owner_project", "owner_id", "project_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    custom_agent_id = Column(UUID(as_uuid=True), ForeignKey("custom_agents.id"), nullable=False)
    agent_order = Column(Integer, nullable=False, default=0, server_default=text("0"))

    def __repr__(self) -> str:
        return (
            f"<ProjectCustomAgent(project_id={self.project_id}, "
            f"custom_agent_id={self.custom_agent_id}, order={self.agent_order})>"
        )
```

- [ ] **Step 4: Add the conversation column**

In `app/models/conversation.py`, add `Index` to the `sqlalchemy` import list, replace `__table_args__` with:

```python
    __table_args__ = (
        CheckConstraint(
            "next_message_sequence > 0",
            name="ck_conversations_next_message_sequence_positive",
        ),
        Index(
            "ix_conversations_project_updated",
            "project_id",
            "updated_at",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )
```

and add this column immediately after `plan_lifecycle`:

```python
    # Nullable: a conversation may belong to at most one project, or none.
    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True, index=True
    )
```

- [ ] **Step 5: Export the models**

In `app/models/__init__.py`, add `from app.models.project import Project, ProjectCustomAgent` after the `model_provider` import, and add `"Project"` and `"ProjectCustomAgent"` to `__all__`. This registry is what Alembic autogenerate reads; omitting it produces an empty migration.

- [ ] **Step 6: Run the model test**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_model.py -v`
Expected: 6 passed.

- [ ] **Step 7: Write the migration**

Create `app/alembic/versions/9a8b7c6d5e4f_add_projects.py`:

```python
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
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
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
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
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
```

- [ ] **Step 8: Verify the revision chain is single-headed**

Run: `.venv/Scripts/python.exe -m alembic heads`
Expected: `9a8b7c6d5e4f (head)` and nothing else. Two heads means `down_revision` is wrong.

- [ ] **Step 9: Apply and roll back against PostgreSQL**

Run:
```bash
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe -m alembic downgrade -1
.venv/Scripts/python.exe -m alembic upgrade head
```
Expected: three clean runs, no error. If the downgrade fails, the `upgrade` is not reversible and must be fixed — do not proceed with a one-way migration.

- [ ] **Step 10: Confirm autogenerate sees no drift**

Run: `.venv/Scripts/python.exe -m alembic check`
Expected: no new operations detected. A diff here means the models and the migration disagree — `create_all` never ALTERs an existing table, so trust this over a passing test.

- [ ] **Step 11: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/models/project.py app/models/conversation.py app/models/__init__.py app/alembic/versions/9a8b7c6d5e4f_add_projects.py tests/test_project_model.py
git add app/models/project.py app/models/conversation.py app/models/__init__.py app/alembic/versions/9a8b7c6d5e4f_add_projects.py tests/test_project_model.py
git commit -m "feat: add projects tables and conversation membership"
```

---

## Task 3: Project repository

**Files:**
- Create: `app/repositories/project.py`
- Test: `tests/test_project_repository.py`

**Interfaces:**
- Consumes: `Project`, `ProjectCustomAgent` from Task 2; `ConversationCustomAgent`, `CustomAgent` from `app.models.custom_agent`; `Conversation` from `app.models.conversation`; `RepositorySessionMixin` from `app.repositories.session_transport`.
- Produces: `ProjectRepository` with `list_by_owner`, `get_owned`, `get_live`, `create`, `update`, `soft_delete_and_detach`, `conversation_counts`, `list_agents`, `replace_agents`, `seed_conversation_agents`, `attach_conversation`, `detach_conversation`. Signatures are in Step 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_project_repository.py`:

```python
"""Behaviour of ProjectRepository against PostgreSQL."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from app.core.config import settings
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User
from app.repositories.project import ProjectRepository


@pytest.fixture
def repo_env():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    agent_a = uuid4()
    agent_b = uuid4()

    with sf() as s:
        for uid in (owner_id, other_id):
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
        for aid, name in ((agent_a, "A"), (agent_b, "B")):
            s.add(
                CustomAgent(
                    id=aid,
                    owner_id=owner_id,
                    name=name,
                    slug=name.lower(),
                    prompt="p",
                    provider_type="openai",
                    model="gpt-4.1-mini",
                )
            )
        s.commit()

    repository = ProjectRepository(session_factory=sf)
    try:
        yield repository, sf, owner_id, other_id, agent_a, agent_b
    finally:
        with sf() as s:
            s.execute(delete(ConversationCustomAgent))
            s.execute(delete(ProjectCustomAgent))
            for uid in (owner_id, other_id):
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(Project).where(Project.owner_id == uid))
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def test_create_and_list_by_owner(repo_env):
    repository, _sf, owner_id, other_id, _a, _b = repo_env

    created = repository.create(owner_id, {"name": "Roadmap", "instructions": "Be brief."})

    assert created.name == "Roadmap"
    assert [p.id for p in repository.list_by_owner(owner_id)] == [created.id]
    assert repository.list_by_owner(other_id) == []


def test_get_owned_returns_none_for_another_owner(repo_env):
    repository, _sf, owner_id, other_id, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})

    assert repository.get_owned(owner_id, project.id) is not None
    assert repository.get_owned(other_id, project.id) is None


def test_get_live_is_ownership_agnostic_but_hides_deleted(repo_env):
    repository, _sf, owner_id, _other, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})

    assert repository.get_live(project.id) is not None

    repository.soft_delete_and_detach(owner_id, project.id)

    assert repository.get_live(project.id) is None


def test_soft_delete_detaches_conversations_without_deleting_them(repo_env):
    repository, sf, owner_id, _other, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    conversation_id = uuid4()
    with sf() as s:
        s.add(
            Conversation(
                id=conversation_id, owner_id=owner_id, title="t", project_id=project.id
            )
        )
        s.commit()

    assert repository.soft_delete_and_detach(owner_id, project.id) is True

    with sf() as s:
        conversation = s.get(Conversation, conversation_id)
        assert conversation is not None, "the conversation must survive the project delete"
        assert conversation.project_id is None


def test_replace_agents_sets_order(repo_env):
    repository, _sf, owner_id, _other, agent_a, agent_b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})

    repository.replace_agents(owner_id, project.id, [agent_b, agent_a])

    assert [a.id for a in repository.list_agents(owner_id, project.id)] == [agent_b, agent_a]


def test_seed_inserts_project_agents_onto_a_conversation(repo_env):
    repository, sf, owner_id, _other, agent_a, agent_b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    repository.replace_agents(owner_id, project.id, [agent_a, agent_b])
    conversation_id = uuid4()
    with sf() as s:
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.commit()

    inserted = repository.seed_conversation_agents(owner_id, project.id, conversation_id)

    assert inserted == 2
    with sf() as s:
        rows = s.execute(
            select(ConversationCustomAgent.custom_agent_id)
            .where(ConversationCustomAgent.conversation_id == conversation_id)
            .order_by(ConversationCustomAgent.agent_order.asc())
        ).scalars().all()
    assert rows == [agent_a, agent_b]


def test_seed_is_idempotent_and_never_removes(repo_env):
    repository, sf, owner_id, _other, agent_a, agent_b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    repository.replace_agents(owner_id, project.id, [agent_a])
    conversation_id = uuid4()
    with sf() as s:
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.add(
            ConversationCustomAgent(
                owner_id=owner_id,
                conversation_id=conversation_id,
                custom_agent_id=agent_b,
                agent_order=0,
            )
        )
        s.commit()

    first = repository.seed_conversation_agents(owner_id, project.id, conversation_id)
    second = repository.seed_conversation_agents(owner_id, project.id, conversation_id)

    assert (first, second) == (1, 0)
    with sf() as s:
        rows = s.execute(
            select(ConversationCustomAgent.custom_agent_id)
            .where(ConversationCustomAgent.conversation_id == conversation_id)
            .order_by(ConversationCustomAgent.agent_order.asc())
        ).scalars().all()
    assert rows == [agent_b, agent_a], "the pre-existing attachment is kept and stays first"


def test_conversation_counts_excludes_soft_deleted(repo_env):
    repository, sf, owner_id, _other, _a, _b = repo_env
    project = repository.create(owner_id, {"name": "Roadmap"})
    with sf() as s:
        s.add(Conversation(owner_id=owner_id, title="live", project_id=project.id))
        s.add(
            Conversation(
                owner_id=owner_id,
                title="gone",
                project_id=project.id,
                deleted_at="2026-09-01T00:00:00+00:00",
            )
        )
        s.commit()

    assert repository.conversation_counts(owner_id) == {project.id: 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_repository.py -v`
Expected: collection error — `No module named 'app.repositories.project'`.

- [ ] **Step 3: Write the repository**

Create `app/repositories/project.py`:

```python
"""Repository for projects, their default agents, and conversation membership."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select, update

from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.repositories.session_transport import RepositorySessionMixin


class ProjectRepository(RepositorySessionMixin):
    """Session-factory backed repository for projects."""

    def __init__(
        self,
        session_factory: Callable[[], Any],
        async_session_factory: Callable[[], Any] | None = None,
    ):
        super().__init__(
            session_factory=session_factory,
            async_session_factory=async_session_factory,
        )

    # ----------------------------------------------------------------- reads

    def list_by_owner(self, owner_id: UUID) -> list[Project]:
        """All live projects for an owner, newest first."""
        with self.session_factory() as session:
            stmt = (
                select(Project)
                .where(Project.owner_id == owner_id, Project.deleted_at.is_(None))
                .order_by(Project.created_at.desc())
            )
            projects = list(session.execute(stmt).scalars().all())
            for project in projects:
                session.expunge(project)
            return projects

    def get_owned(self, owner_id: UUID, project_id: UUID) -> Project | None:
        """A live project if it exists and belongs to ``owner_id``."""
        with self.session_factory() as session:
            project = session.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.owner_id == owner_id,
                    Project.deleted_at.is_(None),
                )
            ).scalar_one_or_none()
            if project is not None:
                session.expunge(project)
            return project

    def get_live(self, project_id: UUID) -> Project | None:
        """A live project regardless of owner.

        Used by the instruction resolver, which already holds a conversation
        the caller was authorised to read, and by ownership checks that need
        to tell "missing" apart from "someone else's".
        """
        with self.session_factory() as session:
            project = session.execute(
                select(Project).where(
                    Project.id == project_id, Project.deleted_at.is_(None)
                )
            ).scalar_one_or_none()
            if project is not None:
                session.expunge(project)
            return project

    def conversation_counts(self, owner_id: UUID) -> dict[UUID, int]:
        """Live conversation count per project, for the owner's project list."""
        with self.session_factory() as session:
            stmt = (
                select(Conversation.project_id, func.count(Conversation.id))
                .where(
                    Conversation.owner_id == owner_id,
                    Conversation.project_id.is_not(None),
                    Conversation.deleted_at.is_(None),
                )
                .group_by(Conversation.project_id)
            )
            return {project_id: count for project_id, count in session.execute(stmt).all()}

    def list_agents(self, owner_id: UUID, project_id: UUID) -> list[CustomAgent]:
        """Live custom agents in the project's default set, in attachment order.

        Returns the agent rows rather than ids so the service can convert them
        to ``CustomAgentRead``, matching the conversation agent route.
        """
        with self.session_factory() as session:
            stmt = (
                select(CustomAgent)
                .join(
                    ProjectCustomAgent,
                    ProjectCustomAgent.custom_agent_id == CustomAgent.id,
                )
                .where(
                    ProjectCustomAgent.project_id == project_id,
                    ProjectCustomAgent.owner_id == owner_id,
                    CustomAgent.deleted_at.is_(None),
                )
                .order_by(ProjectCustomAgent.agent_order.asc())
            )
            agents = list(session.execute(stmt).scalars().all())
            for agent in agents:
                session.expunge(agent)
            return agents

    # ---------------------------------------------------------------- writes

    def create(self, owner_id: UUID, fields: dict[str, Any]) -> Project:
        with self.session_factory() as session:
            project = Project(owner_id=owner_id, **fields)
            session.add(project)
            session.commit()
            session.refresh(project)
            session.expunge(project)
            return project

    def update(self, owner_id: UUID, project_id: UUID, fields: dict[str, Any]) -> Project | None:
        with self.session_factory() as session:
            project = session.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.owner_id == owner_id,
                    Project.deleted_at.is_(None),
                )
            ).scalar_one_or_none()
            if project is None:
                return None
            for key, value in fields.items():
                setattr(project, key, value)
            session.commit()
            session.refresh(project)
            session.expunge(project)
            return project

    def soft_delete_and_detach(self, owner_id: UUID, project_id: UUID) -> bool:
        """Soft-delete the project and release its conversations.

        Conversations survive as loose conversations. The detach is a hard
        write while the delete is soft, so restoring a project would restore
        it empty; there is no restore in this slice.
        """
        with self.session_factory() as session:
            project = session.execute(
                select(Project).where(
                    Project.id == project_id,
                    Project.owner_id == owner_id,
                    Project.deleted_at.is_(None),
                )
            ).scalar_one_or_none()
            if project is None:
                return False
            session.execute(
                update(Conversation)
                .where(Conversation.project_id == project_id)
                .values(project_id=None)
            )
            project.deleted_at = func.now()
            session.commit()
            return True

    def replace_agents(
        self, owner_id: UUID, project_id: UUID, custom_agent_ids: list[UUID]
    ) -> None:
        """Replace the project's default set with the given ordered ids.

        This does not reach into conversations already in the project; seeding
        happens at create and attach time only.
        """
        with self.session_factory() as session:
            session.execute(
                delete(ProjectCustomAgent).where(
                    ProjectCustomAgent.project_id == project_id,
                    ProjectCustomAgent.owner_id == owner_id,
                )
            )
            for order, custom_agent_id in enumerate(custom_agent_ids):
                session.add(
                    ProjectCustomAgent(
                        owner_id=owner_id,
                        project_id=project_id,
                        custom_agent_id=custom_agent_id,
                        agent_order=order,
                    )
                )
            session.commit()

    def seed_conversation_agents(
        self, owner_id: UUID, project_id: UUID, conversation_id: UUID
    ) -> int:
        """Copy the project's default agents onto a conversation, insert-if-absent.

        Returns the number of rows inserted. Never removes or reorders an
        existing attachment; new rows continue the conversation's ordering.
        Shared by conversation creation and by attach.
        """
        with self.session_factory() as session:
            project_agent_ids = list(
                session.execute(
                    select(ProjectCustomAgent.custom_agent_id)
                    .where(
                        ProjectCustomAgent.project_id == project_id,
                        ProjectCustomAgent.owner_id == owner_id,
                    )
                    .order_by(ProjectCustomAgent.agent_order.asc())
                )
                .scalars()
                .all()
            )
            if not project_agent_ids:
                return 0

            existing = set(
                session.execute(
                    select(ConversationCustomAgent.custom_agent_id).where(
                        ConversationCustomAgent.conversation_id == conversation_id
                    )
                )
                .scalars()
                .all()
            )
            next_order = (
                session.execute(
                    select(
                        func.coalesce(func.max(ConversationCustomAgent.agent_order), -1)
                    ).where(ConversationCustomAgent.conversation_id == conversation_id)
                ).scalar_one()
                + 1
            )

            inserted = 0
            for custom_agent_id in project_agent_ids:
                if custom_agent_id in existing:
                    continue
                session.add(
                    ConversationCustomAgent(
                        owner_id=owner_id,
                        conversation_id=conversation_id,
                        custom_agent_id=custom_agent_id,
                        agent_order=next_order,
                    )
                )
                next_order += 1
                inserted += 1
            session.commit()
            return inserted

    def set_conversation_project(
        self, conversation_id: UUID, project_id: UUID | None
    ) -> None:
        """Point a conversation at a project, or release it when ``None``."""
        with self.session_factory() as session:
            session.execute(
                update(Conversation)
                .where(Conversation.id == conversation_id)
                .values(project_id=project_id)
            )
            session.commit()

    def conversation_project_id(self, conversation_id: UUID) -> UUID | None:
        """The project a conversation currently belongs to, if any."""
        with self.session_factory() as session:
            return session.execute(
                select(Conversation.project_id).where(Conversation.id == conversation_id)
            ).scalar_one_or_none()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_repository.py -v`
Expected: 8 passed.

- [ ] **Step 5: Prove the seeding test catches a real regression**

Temporarily change `if custom_agent_id in existing: continue` to `pass`, re-run
`tests/test_project_repository.py::test_seed_is_idempotent_and_never_removes`, and
confirm it FAILS on the unique constraint. Restore the line and confirm it passes
again. A test that cannot fail is not pinning anything.

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/repositories/project.py tests/test_project_repository.py
git add app/repositories/project.py tests/test_project_repository.py
git commit -m "feat: add project repository with agent seeding"
```

---

## Task 4: Exceptions, schemas, and the project service

**Files:**
- Create: `app/core/exceptions/project.py`
- Create: `app/schemas/project.py`
- Create: `app/services/project_service.py`
- Modify: `app/core/exceptions/__init__.py`
- Modify: `app/schemas/__init__.py`
- Test: `tests/test_project_service.py`

**Interfaces:**
- Consumes: `ProjectRepository` (Task 3); `CustomAgentRepository.get_owned`, `CustomAgentRepository.list_by_owner` from `app.repositories.custom_agent`; `ConversationValidationUtils.validate_conversation_access` from `app.utils.validation.conversation_validation`.
- Produces: `ProjectService` with `list_projects(owner_id) -> list[ProjectRead]`, `create_project(owner_id, payload) -> ProjectRead`, `get_project(owner_id, project_id, *, include_agents=False) -> ProjectRead`, `update_project(owner_id, project_id, payload) -> ProjectRead`, `delete_project(owner_id, project_id) -> None`, `list_agents(owner_id, project_id) -> list[CustomAgentRead]`, `set_agents(owner_id, project_id, custom_agent_ids) -> list[CustomAgentRead]`, `attach_conversation(owner_id, project_id, conversation_id) -> None`, `detach_conversation(owner_id, project_id, conversation_id) -> None`, `require_owned(owner_id, project_id) -> Project`. Also `ProjectCreate`, `ProjectUpdate`, `ProjectRead` (with `conversation_count` and optional `custom_agents`), `ProjectCustomAgentsUpdate`.
- Note: `ProjectService` exposes its repository as `self.repository`, which Task 6 uses to call `seed_conversation_agents`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_project_service.py`:

```python
"""ProjectService behaviour, including the reverse-direction ownership checks."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete

from app.core.config import settings
from app.core.exceptions.project import (
    ProjectConversationNotFoundError,
    ProjectForbiddenError,
    ProjectNotFoundError,
)
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User
from app.repositories.conversation import ConversationRepository
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.project import ProjectRepository
from app.schemas.project import ProjectCreate, ProjectUpdate
from app.services.project_service import ProjectService
from app.utils.validation.conversation_validation import ConversationValidationUtils


@pytest.fixture
def service_env():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    agent_id = uuid4()
    owner_conversation = uuid4()
    other_conversation = uuid4()

    with sf() as s:
        for uid in (owner_id, other_id):
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
        s.add(
            CustomAgent(
                id=agent_id,
                owner_id=owner_id,
                name="A",
                slug="a",
                prompt="p",
                provider_type="openai",
                model="gpt-4.1-mini",
            )
        )
        s.add(Conversation(id=owner_conversation, owner_id=owner_id, title="mine"))
        s.add(Conversation(id=other_conversation, owner_id=other_id, title="theirs"))
        s.commit()

    service = ProjectService(
        repository=ProjectRepository(session_factory=sf),
        custom_agent_repository=CustomAgentRepository(session_factory=sf),
        # ConversationValidationUtils takes a session factory and builds its own
        # repository internally, exposing it as `.conversation_repository`.
        conversation_validation_utils=ConversationValidationUtils(session_factory=sf),
    )
    try:
        yield service, owner_id, other_id, agent_id, owner_conversation, other_conversation
    finally:
        with sf() as s:
            s.execute(delete(ConversationCustomAgent))
            s.execute(delete(ProjectCustomAgent))
            for uid in (owner_id, other_id):
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(Project).where(Project.owner_id == uid))
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def test_create_then_read(service_env):
    service, owner_id, *_ = service_env

    created = service.create_project(owner_id, ProjectCreate(name="Roadmap", instructions="Brief."))
    fetched = service.get_project(owner_id, created.id)

    assert fetched.name == "Roadmap"
    assert fetched.instructions == "Brief."
    assert fetched.conversation_count == 0


def test_update_changes_instructions(service_env):
    service, owner_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    updated = service.update_project(owner_id, created.id, ProjectUpdate(instructions="New."))

    assert updated.instructions == "New."


def test_missing_project_is_404(service_env):
    service, owner_id, *_ = service_env

    with pytest.raises(ProjectNotFoundError):
        service.get_project(owner_id, uuid4())


def test_another_users_project_is_403(service_env):
    """Reverse direction: unauthorised access must be refused, not merely absent."""
    service, owner_id, other_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.get_project(other_id, created.id)


def test_deleted_project_is_404_not_403(service_env):
    service, owner_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    service.delete_project(owner_id, created.id)

    with pytest.raises(ProjectNotFoundError):
        service.get_project(owner_id, created.id)


def test_cannot_attach_another_users_conversation(service_env):
    service, owner_id, _other_id, _agent, _mine, theirs = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(Exception) as exc:
        service.attach_conversation(owner_id, created.id, theirs)

    assert exc.type is not ProjectNotFoundError, "must refuse on ownership, not report missing"


def test_cannot_attach_to_another_users_project(service_env):
    service, owner_id, other_id, _agent, mine, _theirs = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.attach_conversation(other_id, created.id, mine)


def test_attach_seeds_agents_and_detach_keeps_them(service_env):
    service, owner_id, _other, agent_id, mine, _theirs = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    service.set_agents(owner_id, created.id, [agent_id])

    service.attach_conversation(owner_id, created.id, mine)
    seeded = service.repository.seed_conversation_agents(owner_id, created.id, mine)
    service.detach_conversation(owner_id, created.id, mine)

    assert seeded == 0, "attach already seeded the agent, so a re-seed inserts nothing"
    assert service.repository.conversation_project_id(mine) is None


def test_detach_against_the_wrong_project_is_404(service_env):
    service, owner_id, _other, _agent, mine, _theirs = service_env
    first = service.create_project(owner_id, ProjectCreate(name="First"))
    second = service.create_project(owner_id, ProjectCreate(name="Second"))
    service.attach_conversation(owner_id, first.id, mine)

    with pytest.raises(ProjectConversationNotFoundError):
        service.detach_conversation(owner_id, second.id, mine)


def test_attach_moves_a_conversation_between_projects(service_env):
    service, owner_id, _other, _agent, mine, _theirs = service_env
    first = service.create_project(owner_id, ProjectCreate(name="First"))
    second = service.create_project(owner_id, ProjectCreate(name="Second"))

    service.attach_conversation(owner_id, first.id, mine)
    service.attach_conversation(owner_id, second.id, mine)

    assert service.repository.conversation_project_id(mine) == second.id


def test_set_agents_rejects_an_agent_the_user_does_not_own(service_env):
    service, owner_id, *_ = service_env
    created = service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.set_agents(owner_id, created.id, [uuid4()])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_service.py -v`
Expected: collection error — `No module named 'app.core.exceptions.project'`.

- [ ] **Step 3: Write the exceptions**

Create `app/core/exceptions/project.py`:

```python
"""Project exception classes mapped to the API error contract.

Matches the convention already set by ``custom_agent.py``: 403 when the
resource exists but belongs to someone else, 404 only when it is genuinely
missing or soft-deleted.
"""

from fastapi import status

from .http import CustomHTTPException


class ProjectValidationError(CustomHTTPException):
    """400 — invalid name, instructions, or agent id list."""

    def __init__(
        self,
        detail: str = "Project validation failed",
        error_code: str = "PROJECT_VALIDATION_FAILED",
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST, detail=detail, error_code=error_code
        )


class ProjectForbiddenError(CustomHTTPException):
    """403 — the user does not own the project, conversation, or agent."""

    def __init__(
        self,
        detail: str = "Access denied to this project resource",
        error_code: str = "PROJECT_FORBIDDEN",
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN, detail=detail, error_code=error_code
        )


class ProjectNotFoundError(CustomHTTPException):
    """404 — the project is missing or soft-deleted."""

    def __init__(
        self,
        detail: str = "Project not found",
        error_code: str = "PROJECT_NOT_FOUND",
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code
        )


class ProjectConversationNotFoundError(CustomHTTPException):
    """404 — the conversation is not a member of this project."""

    def __init__(
        self,
        detail: str = "Conversation is not in this project",
        error_code: str = "PROJECT_CONVERSATION_NOT_FOUND",
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code
        )
```

Add all four names to `app/core/exceptions/__init__.py` following the existing import-and-`__all__` pattern in that file.

- [ ] **Step 4: Write the schemas**

Create `app/schemas/project.py`:

```python
"""Request and response schemas for projects."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.custom_agent import CustomAgentRead
from app.utils.case_conversion import to_camel_case as to_camel


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: str | None = Field(None, max_length=2000, description="Short description")
    instructions: str | None = Field(
        None,
        max_length=8000,
        description="Instructions applied to every conversation in this project",
    )

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=2000)
    instructions: str | None = Field(None, max_length=8000)

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ProjectRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, alias_generator=to_camel, populate_by_name=True)

    id: UUID
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    owner_id: UUID
    name: str
    description: str | None = None
    instructions: str | None = None
    conversation_count: int = Field(
        default=0, description="Live conversations currently in this project"
    )
    custom_agents: list[CustomAgentRead] | None = Field(
        default=None,
        description="Ordered default agents (when requested)",
    )


class ProjectCustomAgentsUpdate(BaseModel):
    """Replace the ordered set of default custom agents for a project."""

    custom_agent_ids: list[UUID] = Field(default_factory=list)

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    @field_validator("custom_agent_ids")
    @classmethod
    def _no_duplicates(cls, value: list[UUID]) -> list[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("custom_agent_ids must not contain duplicates")
        return value
```

Export the four names from `app/schemas/__init__.py` following the existing pattern there.

- [ ] **Step 5: Write the service**

Create `app/services/project_service.py`:

```python
"""Service layer for projects: CRUD, ownership, default agents, membership."""

from __future__ import annotations

import logging
from uuid import UUID

from app.core.exceptions.project import (
    ProjectConversationNotFoundError,
    ProjectForbiddenError,
    ProjectNotFoundError,
)
from app.models.project import Project
from app.repositories.custom_agent import CustomAgentRepository
from app.repositories.project import ProjectRepository
from app.schemas.custom_agent import CustomAgentRead
from app.schemas.project import ProjectCreate, ProjectRead, ProjectUpdate
from app.utils.validation.conversation_validation import ConversationValidationUtils

logger = logging.getLogger(__name__)


class ProjectService:
    """Owner-scoped project operations."""

    def __init__(
        self,
        repository: ProjectRepository,
        custom_agent_repository: CustomAgentRepository,
        conversation_validation_utils: ConversationValidationUtils,
    ):
        self.repository = repository
        self.custom_agent_repository = custom_agent_repository
        self.conversation_validation_utils = conversation_validation_utils

    # ------------------------------------------------------------ ownership

    def require_owned(self, owner_id: UUID, project_id: UUID) -> Project:
        """Return the project, or raise 404 if missing and 403 if not the caller's.

        Distinguishing the two is deliberate and matches ``custom_agents``.
        """
        project = self.repository.get_live(project_id)
        if project is None:
            raise ProjectNotFoundError()
        if project.owner_id != owner_id:
            raise ProjectForbiddenError()
        return project

    # ----------------------------------------------------------------- CRUD

    def list_projects(self, owner_id: UUID) -> list[ProjectRead]:
        counts = self.repository.conversation_counts(owner_id)
        return [
            self._to_read(project, counts.get(project.id, 0))
            for project in self.repository.list_by_owner(owner_id)
        ]

    def create_project(self, owner_id: UUID, payload: ProjectCreate) -> ProjectRead:
        project = self.repository.create(
            owner_id,
            {
                "name": payload.name,
                "description": payload.description,
                "instructions": payload.instructions,
            },
        )
        return self._to_read(project, 0)

    def get_project(
        self, owner_id: UUID, project_id: UUID, *, include_agents: bool = False
    ) -> ProjectRead:
        project = self.require_owned(owner_id, project_id)
        counts = self.repository.conversation_counts(owner_id)
        agents = (
            [
                CustomAgentRead.model_validate(agent)
                for agent in self.repository.list_agents(owner_id, project_id)
            ]
            if include_agents
            else None
        )
        return self._to_read(project, counts.get(project.id, 0), agents)

    def update_project(
        self, owner_id: UUID, project_id: UUID, payload: ProjectUpdate
    ) -> ProjectRead:
        self.require_owned(owner_id, project_id)
        fields = payload.model_dump(exclude_unset=True)
        project = self.repository.update(owner_id, project_id, fields)
        if project is None:
            raise ProjectNotFoundError()
        counts = self.repository.conversation_counts(owner_id)
        return self._to_read(project, counts.get(project.id, 0))

    def delete_project(self, owner_id: UUID, project_id: UUID) -> None:
        self.require_owned(owner_id, project_id)
        if not self.repository.soft_delete_and_detach(owner_id, project_id):
            raise ProjectNotFoundError()

    # --------------------------------------------------------------- agents

    def list_agents(self, owner_id: UUID, project_id: UUID) -> list[CustomAgentRead]:
        self.require_owned(owner_id, project_id)
        return [
            CustomAgentRead.model_validate(agent)
            for agent in self.repository.list_agents(owner_id, project_id)
        ]

    def set_agents(
        self, owner_id: UUID, project_id: UUID, custom_agent_ids: list[UUID]
    ) -> list[CustomAgentRead]:
        """Replace the default set. Does not touch conversations already in the project."""
        self.require_owned(owner_id, project_id)
        for custom_agent_id in custom_agent_ids:
            if self.custom_agent_repository.get_owned(owner_id, custom_agent_id) is None:
                raise ProjectForbiddenError(
                    detail=f"Custom agent {custom_agent_id} is not available to this user"
                )
        self.repository.replace_agents(owner_id, project_id, custom_agent_ids)
        return self.list_agents(owner_id, project_id)

    # ----------------------------------------------------------- membership

    def attach_conversation(
        self, owner_id: UUID, project_id: UUID, conversation_id: UUID
    ) -> None:
        """Move a conversation into the project and seed the project's agents.

        Attaching a conversation that already belongs to another project is a
        move, not an error.
        """
        self.require_owned(owner_id, project_id)
        self.conversation_validation_utils.validate_conversation_access(
            owner_id, conversation_id
        )
        self.repository.set_conversation_project(conversation_id, project_id)
        self.repository.seed_conversation_agents(owner_id, project_id, conversation_id)

    def detach_conversation(
        self, owner_id: UUID, project_id: UUID, conversation_id: UUID
    ) -> None:
        """Release a conversation from the project, keeping its seeded agents."""
        self.require_owned(owner_id, project_id)
        self.conversation_validation_utils.validate_conversation_access(
            owner_id, conversation_id
        )
        if self.repository.conversation_project_id(conversation_id) != project_id:
            raise ProjectConversationNotFoundError()
        self.repository.set_conversation_project(conversation_id, None)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _to_read(
        project: Project,
        conversation_count: int,
        agents: list[CustomAgentRead] | None = None,
    ) -> ProjectRead:
        return ProjectRead(
            id=project.id,
            created_at=project.created_at,
            updated_at=project.updated_at,
            deleted_at=project.deleted_at,
            owner_id=project.owner_id,
            name=project.name,
            description=project.description,
            instructions=project.instructions,
            conversation_count=conversation_count,
            custom_agents=agents,
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_service.py -v`
Expected: 11 passed.

- [ ] **Step 7: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/core/exceptions/project.py app/core/exceptions/__init__.py app/schemas/project.py app/schemas/__init__.py app/services/project_service.py tests/test_project_service.py
git add app/core/exceptions/project.py app/core/exceptions/__init__.py app/schemas/project.py app/schemas/__init__.py app/services/project_service.py tests/test_project_service.py
git commit -m "feat: add project service, schemas, and exceptions"
```

---

## Task 5: Resolver and the three persona call sites

The highest-risk task: it changes what the model receives on every turn. The invariant test from Task 1 is what protects conversations with no project.

**Files:**
- Create: `app/services/project_context_service.py`
- Modify: `app/services/ai_service.py:150-158`
- Modify: `app/services/message_service.py:1051-1061` and `:2515-2516` and `:3912`
- Test: `tests/test_project_context_service.py`

**Interfaces:**
- Consumes: `compose_system_instruction` (Task 1), `ProjectRepository.get_live` (Task 3).
- Produces: `ProjectContextService.resolve_system_instruction(conversation: Any) -> str | None`. Tasks 6 and 7 wire it through the container.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_project_context_service.py`:

```python
"""Resolution of a conversation's system instruction, project included."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from app.services.project_context_service import ProjectContextService


class _StubRepository:
    def __init__(self, project=None):
        self._project = project
        self.calls: list = []

    def get_live(self, project_id):
        self.calls.append(project_id)
        return self._project


def test_conversation_without_a_project_returns_its_persona():
    service = ProjectContextService(project_repository=_StubRepository())
    conversation = SimpleNamespace(project_id=None, persona_prompt="Be terse.")

    assert service.resolve_system_instruction(conversation) == "Be terse."


def test_conversation_without_a_project_never_queries(): 
    repository = _StubRepository()
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=None, persona_prompt="Be terse.")

    service.resolve_system_instruction(conversation)

    assert repository.calls == []


def test_project_instructions_lead_the_persona():
    project_id = uuid4()
    repository = _StubRepository(SimpleNamespace(instructions="Answer in Vietnamese."))
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=project_id, persona_prompt="Be terse.")

    result = service.resolve_system_instruction(conversation)

    assert result == (
        "Project instructions:\nAnswer in Vietnamese.\n\n"
        "Conversation-specific instructions:\nBe terse."
    )


def test_soft_deleted_project_reads_as_project_less():
    """get_live filters deleted_at, so a stale pointer must not inherit."""
    repository = _StubRepository(None)
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=uuid4(), persona_prompt="Be terse.")

    assert service.resolve_system_instruction(conversation) == "Be terse."


def test_missing_conversation_does_not_raise():
    service = ProjectContextService(project_repository=_StubRepository())

    assert service.resolve_system_instruction(None) is None


def test_long_project_and_persona_are_not_truncated_together():
    """The resume path used to re-sanitize; 16000 characters must survive."""
    repository = _StubRepository(SimpleNamespace(instructions="P" * 9000))
    service = ProjectContextService(project_repository=repository)
    conversation = SimpleNamespace(project_id=uuid4(), persona_prompt="Q" * 9000)

    result = service.resolve_system_instruction(conversation)

    assert result.count("P") == 8000
    assert result.count("Q") == 8000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_context_service.py -v`
Expected: collection error — `No module named 'app.services.project_context_service'`.

- [ ] **Step 3: Write the resolver**

Create `app/services/project_context_service.py`:

```python
"""Resolves the system instruction a conversation runs with.

This is the single place project-derived prompt context is assembled. It reads
the project live, so editing a project's instructions takes effect on the next
turn of every conversation in it.
"""

from __future__ import annotations

import logging
from typing import Any

from app.repositories.project import ProjectRepository
from app.utils.text_processing import compose_system_instruction

logger = logging.getLogger(__name__)


class ProjectContextService:
    """Combines a conversation's project instructions with its own persona."""

    def __init__(self, project_repository: ProjectRepository):
        self.project_repository = project_repository

    def resolve_system_instruction(self, conversation: Any) -> str | None:
        """The sanitized system instruction for ``conversation``, or ``None``.

        A conversation with no project costs no extra query and returns exactly
        what the pre-project code returned.
        """
        if conversation is None:
            return None

        persona = getattr(conversation, "persona_prompt", None)
        project_id = getattr(conversation, "project_id", None)
        if project_id is None:
            return compose_system_instruction(None, persona)

        project_instructions = None
        try:
            project = self.project_repository.get_live(project_id)
            if project is not None:
                project_instructions = project.instructions
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to resolve project instructions: %s", exc)

        return compose_system_instruction(project_instructions, persona)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_context_service.py -v`
Expected: 6 passed.

- [ ] **Step 5: Wire `ai_service._prepare_request`**

In `app/services/ai_service.py`, add `project_context_service` as an optional
constructor parameter (default `None`, stored on `self`), and replace the body of
`_prepare_request` (lines 150-158) with:

```python
    def _prepare_request(self, request: WorkflowExecutionRequest) -> WorkflowExecutionRequest:
        if request.persona is not None or not request.conversation_id:
            return request
        try:
            conversation = self.conversation_repository.get_by_id(UUID(request.conversation_id))
        except Exception:
            conversation = None
        if self.project_context_service is None:
            return request.model_copy(
                update={
                    "persona": sanitize_persona(
                        conversation.persona_prompt if conversation else None
                    )
                }
            )
        return request.model_copy(
            update={
                "persona": self.project_context_service.resolve_system_instruction(conversation)
            }
        )
```

The `None` branch keeps every existing direct construction of `AIService` in the
test suite working without edits.

- [ ] **Step 6: Wire both message_service sites**

In `app/services/message_service.py`, add `project_context_service` as an optional
constructor parameter (default `None`, stored on `self`), then:

Replace `_get_conversation_context` (lines 1051-1061) with:

```python
    def _get_conversation_context(
        self, conversation_id: UUID, user_id: UUID | None = None
    ) -> tuple[UUID | None, str | None]:
        """Get user_id and the composed system instruction from a conversation.

        Returns the instruction already sanitized and composed. The caller must
        NOT run ``sanitize_persona`` over it: the composed string can exceed the
        8000-character cap legitimately, and truncating it would silently drop
        the conversation's own persona.
        """
        conversation = self.conversation_validation_utils.conversation_repository.get_by_id(
            conversation_id
        )
        resolved_user_id = user_id or (conversation.owner_id if conversation else None)
        if self.project_context_service is None:
            return resolved_user_id, sanitize_persona(
                conversation.persona_prompt if conversation else None
            )
        return resolved_user_id, self.project_context_service.resolve_system_instruction(
            conversation
        )
```

At line 2515-2516, delete the re-sanitize:

```python
            user_id, sanitized_persona = self._get_conversation_context(conversation_id, user_id)
```

replacing the two-line `user_id, persona = ...` / `sanitized_persona = sanitize_persona(persona)`
pair. This deletion is the point of the change — leaving it would reintroduce the
truncation bug on the HITL resume path.

At line 3912, in `_build_user_message_workflow_request`, replace:

```python
        sanitized_persona = sanitize_persona(conversation.persona_prompt if conversation else None)
```

with:

```python
        if self.project_context_service is None:
            sanitized_persona = sanitize_persona(
                conversation.persona_prompt if conversation else None
            )
        else:
            sanitized_persona = self.project_context_service.resolve_system_instruction(
                conversation
            )
```

- [ ] **Step 7: Confirm the schema parity guard still passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_workflow_request_schema_parity.py -v`
Expected: 1 passed. This approach adds no request field, so this test should never
have been at risk — if it fails, a field was added somewhere it should not have been.

- [ ] **Step 8: Run the affected existing suites**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_project_context_service.py tests/test_workflow_request_schema_parity.py tests/test_custom_agents_message_service.py tests/test_ai_service_initialization.py tests/test_demo_persona_editor_seed.py -v
```
Expected: all passed. Any failure here is a regression in persona handling, not a flake.

- [ ] **Step 9: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/services/project_context_service.py app/services/ai_service.py app/services/message_service.py tests/test_project_context_service.py
git add app/services/project_context_service.py app/services/ai_service.py app/services/message_service.py tests/test_project_context_service.py
git commit -m "feat: resolve project instructions into the system prompt"
```

---

## Task 6: Conversation membership

**Files:**
- Modify: `app/schemas/conversation.py:11-24` and `:41-70`
- Modify: `app/factories/conversation_factory.py:16-42`
- Modify: `app/services/conversation_service.py:44-96` and `get_by_user_id`
- Modify: `app/repositories/conversation.py` (owner listing query)
- Modify: `app/api/conversations.py:76-110`
- Test: `tests/test_project_membership.py`

**Interfaces:**
- Consumes: `ProjectService.require_owned` and `ProjectRepository.seed_conversation_agents` (Tasks 3-4).
- Produces: `ConversationCreate.project_id`, `ConversationRead.project_id`, and `ConversationService.get_by_user_id(..., project_id: UUID | None = None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_project_membership.py`:

```python
"""Creating and listing conversations inside a project."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from app.core.config import settings
from app.core.exceptions.project import ProjectForbiddenError
from app.database.database import Database
from app.factories.conversation_factory import ConversationFactory
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User
from app.schemas.conversation import ConversationCreate


def test_factory_carries_project_id_from_schema():
    owner_id = uuid4()
    project_id = uuid4()

    fields = ConversationFactory.create_from_schema(
        ConversationCreate(title="t", project_id=project_id), owner_id
    )

    assert fields["project_id"] == project_id


def test_factory_carries_project_id_from_dict():
    """Both factory paths, because updating only one is how a field goes missing."""
    owner_id = uuid4()
    project_id = uuid4()

    fields = ConversationFactory.create_from_dict(
        {"owner_id": owner_id, "title": "t", "project_id": project_id}
    )

    assert fields["project_id"] == project_id


def test_factory_defaults_project_id_to_none():
    fields = ConversationFactory.create_from_dict({"owner_id": uuid4(), "title": "t"})

    assert fields["project_id"] is None


@pytest.fixture
def membership_env():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    agent_id = uuid4()
    with sf() as s:
        for uid in (owner_id, other_id):
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
        s.add(
            CustomAgent(
                id=agent_id,
                owner_id=owner_id,
                name="A",
                slug="a",
                prompt="p",
                provider_type="openai",
                model="gpt-4.1-mini",
            )
        )
        s.commit()
    try:
        yield sf, owner_id, other_id, agent_id
    finally:
        with sf() as s:
            s.execute(delete(ConversationCustomAgent))
            s.execute(delete(ProjectCustomAgent))
            for uid in (owner_id, other_id):
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(Project).where(Project.owner_id == uid))
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def _service(sf):
    from app.repositories.conversation import ConversationRepository
    from app.repositories.custom_agent import CustomAgentRepository
    from app.repositories.project import ProjectRepository
    from app.services.conversation_service import ConversationService
    from app.services.project_service import ProjectService
    from app.utils.validation.conversation_validation import ConversationValidationUtils
    from app.utils.validation.user_validation import UserValidationUtils

    conversation_repository = ConversationRepository(session_factory=sf)
    project_repository = ProjectRepository(session_factory=sf)
    # Takes a session factory, not a repository; it builds its own internally.
    conversation_validation_utils = ConversationValidationUtils(session_factory=sf)
    project_service = ProjectService(
        repository=project_repository,
        custom_agent_repository=CustomAgentRepository(session_factory=sf),
        conversation_validation_utils=conversation_validation_utils,
    )
    return (
        ConversationService(
            conversation_repository=conversation_repository,
            # Like ConversationValidationUtils, this takes a session factory.
            user_validation_utils=UserValidationUtils(session_factory=sf),
            conversation_validation_utils=conversation_validation_utils,
            project_service=project_service,
        ),
        project_service,
    )


def test_creating_in_a_project_seeds_its_agents(membership_env):
    sf, owner_id, _other_id, agent_id = membership_env
    service, project_service = _service(sf)
    from app.schemas.project import ProjectCreate

    project = project_service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    project_service.set_agents(owner_id, project.id, [agent_id])

    created = service.create_conversation(
        ConversationCreate(title="t", project_id=project.id), owner_id
    )

    assert created.project_id == project.id
    with sf() as s:
        attached = s.execute(
            select(ConversationCustomAgent.custom_agent_id).where(
                ConversationCustomAgent.conversation_id == created.id
            )
        ).scalars().all()
    assert attached == [agent_id]


def test_creating_in_another_users_project_is_refused(membership_env):
    sf, owner_id, other_id, _agent_id = membership_env
    service, project_service = _service(sf)
    from app.schemas.project import ProjectCreate

    project = project_service.create_project(owner_id, ProjectCreate(name="Roadmap"))

    with pytest.raises(ProjectForbiddenError):
        service.create_conversation(
            ConversationCreate(title="t", project_id=project.id), other_id
        )


def test_listing_filters_by_project(membership_env):
    sf, owner_id, _other_id, _agent_id = membership_env
    service, project_service = _service(sf)
    from app.schemas.project import ProjectCreate

    project = project_service.create_project(owner_id, ProjectCreate(name="Roadmap"))
    inside = service.create_conversation(
        ConversationCreate(title="inside", project_id=project.id), owner_id
    )
    service.create_conversation(ConversationCreate(title="outside"), owner_id)

    filtered = service.get_by_user_id(owner_id, project_id=project.id)
    unfiltered = service.get_by_user_id(owner_id)

    assert [c.id for c in filtered.items] == [inside.id]
    assert len(unfiltered.items) == 2, "the flat list still shows every conversation"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_membership.py -v`
Expected: failures on `ConversationCreate(... project_id=...)` — the field does not exist yet.

- [ ] **Step 3: Add the schema fields**

In `app/schemas/conversation.py`, add to `ConversationCreate`:

```python
    project_id: UUID | None = Field(
        None, description="Project this conversation belongs to, if any"
    )
```

and to both `ConversationRead` and `ConversationInDB`:

```python
    project_id: UUID | None = Field(
        default=None, description="Project this conversation belongs to, if any"
    )
```

`ConversationUpdate` is deliberately left alone: `None` there means "unchanged", so
it cannot express "remove from project". Attach and detach are their own routes.

- [ ] **Step 4: Add the factory fields**

In `app/factories/conversation_factory.py`, add `"project_id": conversation_data.project_id,`
to the dict returned by `create_from_schema`, and
`"project_id": conversation_data.get("project_id"),` to the dict returned by
`create_from_dict`.

- [ ] **Step 5: Validate and seed on create**

In `app/services/conversation_service.py`, add `project_service: Any | None = None` to
`__init__` (stored as `self.project_service`), add `"project_id": getattr(conversation_entity, "project_id", None),`
to the dict in `_convert_to_read_schema`, and replace `create_conversation` with:

```python
    def create_conversation(
        self, conversation_create_data: ConversationCreate, owner_id: UUID
    ) -> ConversationRead:
        self.user_validation_utils.validate_user_exists(owner_id)
        project_id = conversation_create_data.project_id
        if project_id is not None and self.project_service is not None:
            self.project_service.require_owned(owner_id, project_id)
        conversation_entity = ConversationFactory.create_from_schema(
            conversation_create_data, owner_id
        )
        created_conversation = self.repository.create(conversation_entity)
        if project_id is not None and self.project_service is not None:
            self.project_service.repository.seed_conversation_agents(
                owner_id, project_id, created_conversation.id
            )
        return self._convert_to_read_schema(created_conversation, include=[])
```

- [ ] **Step 6: Add the listing filter**

In `app/services/conversation_service.py`, add `project_id: UUID | None = None` to the
`get_by_user_id` signature and pass it through to the repository call in that method.
In `app/repositories/conversation.py`, thread the same optional parameter into the
owner-scoped listing query, adding `.where(Conversation.project_id == project_id)`
only when it is not `None`.

- [ ] **Step 7: Add the query parameter**

In `app/api/conversations.py`, add to `get_conversations`:

```python
    project_id: UUID | None = Query(
        default=None,
        alias="projectId",
        description="Only conversations in this project",
    ),
```

and pass `project_id=project_id` into `conversation_service.get_by_user_id(...)`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_project_membership.py -v`
Expected: 6 passed.

- [ ] **Step 9: Run the existing conversation suites**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_conversation_search.py tests/test_demo_conversation_manager.py tests/test_take100_api.py -v
```
Expected: all passed. These exercise conversation creation and listing.

- [ ] **Step 10: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/schemas/conversation.py app/factories/conversation_factory.py app/services/conversation_service.py app/repositories/conversation.py app/api/conversations.py tests/test_project_membership.py
git add app/schemas/conversation.py app/factories/conversation_factory.py app/services/conversation_service.py app/repositories/conversation.py app/api/conversations.py tests/test_project_membership.py
git commit -m "feat: let conversations belong to a project"
```

---

## Task 7: API routes, wiring, and the frontend contract

**Files:**
- Create: `app/api/projects.py`
- Create: `plans/PROJECTS_FE_CONTRACT.md`
- Modify: `app/core/container.py:154-168` (wiring modules) and the provider block near `custom_agent_service`
- Modify: `app/api/__init__.py`
- Modify: `app/main.py:398-403`
- Test: `tests/test_projects_api.py`

**Interfaces:**
- Consumes: `ProjectService` (Task 4), `ProjectContextService` (Task 5), `ProjectRepository` (Task 3).
- Produces: the nine HTTP routes and the container providers `project_repository`, `project_service`, `project_context_service`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_projects_api.py`:

```python
"""HTTP contract for the project API, including cross-user access."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.projects import router
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
from app.models.user import User


def _build_app(user_id):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    return app


@pytest.fixture
def api():
    db = Database(settings.database_url)
    sf = db.session
    owner_id = uuid4()
    other_id = uuid4()
    conversation_id = uuid4()
    with sf() as s:
        for uid in (owner_id, other_id):
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.commit()

    owner_client = TestClient(_build_app(owner_id))
    other_client = TestClient(_build_app(other_id))
    try:
        yield owner_client, other_client, owner_id, other_id, conversation_id
    finally:
        with sf() as s:
            s.execute(delete(ConversationCustomAgent))
            s.execute(delete(ProjectCustomAgent))
            for uid in (owner_id, other_id):
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(Project).where(Project.owner_id == uid))
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def test_crud_flow(api):
    owner, _other, _oid, _otid, _cid = api

    created = owner.post("/projects", json={"name": "Roadmap", "instructions": "Be brief."})
    assert created.status_code == 201, created.text
    project = created.json()["data"]
    assert project["instructions"] == "Be brief."
    assert project["conversationCount"] == 0

    listed = owner.get("/projects")
    assert listed.status_code == 200
    assert len(listed.json()["data"]) == 1

    patched = owner.patch(f"/projects/{project['id']}", json={"name": "Renamed"})
    assert patched.status_code == 200
    assert patched.json()["data"]["name"] == "Renamed"

    deleted = owner.delete(f"/projects/{project['id']}")
    assert deleted.status_code == 200
    assert owner.get("/projects").json()["data"] == []


def test_instructions_over_the_cap_are_rejected(api):
    owner, *_ = api

    response = owner.post("/projects", json={"name": "X", "instructions": "P" * 8001})

    assert response.status_code == 422


def test_attach_then_detach_a_conversation(api):
    owner, _other, _oid, _otid, conversation_id = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    attached = owner.put(f"/projects/{project_id}/conversations/{conversation_id}")
    assert attached.status_code == 200
    assert owner.get(f"/projects/{project_id}").json()["data"]["conversationCount"] == 1

    detached = owner.delete(f"/projects/{project_id}/conversations/{conversation_id}")
    assert detached.status_code == 200
    assert owner.get(f"/projects/{project_id}").json()["data"]["conversationCount"] == 0


def test_detaching_from_the_wrong_project_is_404(api):
    owner, _other, _oid, _otid, conversation_id = api
    first = owner.post("/projects", json={"name": "First"}).json()["data"]["id"]
    second = owner.post("/projects", json={"name": "Second"}).json()["data"]["id"]
    owner.put(f"/projects/{first}/conversations/{conversation_id}")

    response = owner.delete(f"/projects/{second}/conversations/{conversation_id}")

    assert response.status_code == 404


def test_missing_project_is_404(api):
    owner, *_ = api

    assert owner.get(f"/projects/{uuid4()}").status_code == 404


def test_another_user_cannot_read_the_project(api):
    """Reverse direction: verify the block, not just the happy path."""
    owner, other, *_ = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    assert other.get(f"/projects/{project_id}").status_code == 403


def test_another_user_cannot_attach_to_the_project(api):
    owner, other, _oid, _otid, conversation_id = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    assert (
        other.put(f"/projects/{project_id}/conversations/{conversation_id}").status_code == 403
    )


def test_another_user_cannot_attach_the_owners_conversation_to_their_own_project(api):
    owner, other, _oid, _otid, conversation_id = api
    their_project = other.post("/projects", json={"name": "Theirs"}).json()["data"]["id"]

    response = other.put(f"/projects/{their_project}/conversations/{conversation_id}")

    assert response.status_code == 403


def test_another_user_cannot_delete_the_project(api):
    owner, other, *_ = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    assert other.delete(f"/projects/{project_id}").status_code == 403
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_projects_api.py -v`
Expected: collection error — `No module named 'app.api.projects'`.

- [ ] **Step 3: Write the router**

Create `app/api/projects.py`:

```python
"""Project CRUD, default agents, and conversation membership routes."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, status

from app.core.dependency_injection import AppAutoInjector
from app.schemas.custom_agent import CustomAgentRead
from app.schemas.project import (
    ProjectCreate,
    ProjectCustomAgentsUpdate,
    ProjectRead,
    ProjectUpdate,
)
from app.schemas.responses import ApiResponse
from app.services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=ApiResponse[list[ProjectRead]])
@AppAutoInjector.auto_inject()
async def list_projects(
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[list[ProjectRead]]:
    """List the authenticated user's projects."""
    result = project_service.list_projects(user_id)
    return ApiResponse(success=True, message="Projects retrieved", data=result)


@router.post("", response_model=ApiResponse[ProjectRead], status_code=status.HTTP_201_CREATED)
@AppAutoInjector.auto_inject()
async def create_project(
    payload: ProjectCreate,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[ProjectRead]:
    """Create a project."""
    result = project_service.create_project(user_id, payload)
    return ApiResponse(success=True, message="Project created", data=result)


@router.get("/{project_id}", response_model=ApiResponse[ProjectRead])
@AppAutoInjector.auto_inject()
async def get_project(
    project_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[ProjectRead]:
    """Read one project, including its default agent ids."""
    result = project_service.get_project(user_id, project_id, include_agents=True)
    return ApiResponse(success=True, message="Project retrieved", data=result)


@router.patch("/{project_id}", response_model=ApiResponse[ProjectRead])
@AppAutoInjector.auto_inject()
async def update_project(
    project_id: UUID,
    payload: ProjectUpdate,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[ProjectRead]:
    """Update a project's name, description, or instructions."""
    result = project_service.update_project(user_id, project_id, payload)
    return ApiResponse(success=True, message="Project updated", data=result)


@router.delete("/{project_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def delete_project(
    project_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Soft-delete a project; its conversations are detached, not deleted."""
    project_service.delete_project(user_id, project_id)
    return ApiResponse(success=True, message="Project deleted", data=None)


@router.get("/{project_id}/custom-agents", response_model=ApiResponse[list[CustomAgentRead]])
@AppAutoInjector.auto_inject()
async def list_project_custom_agents(
    project_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[list[CustomAgentRead]]:
    """The project's ordered default agents."""
    result = project_service.list_agents(user_id, project_id)
    return ApiResponse(success=True, message="Project agents retrieved", data=result)


@router.put("/{project_id}/custom-agents", response_model=ApiResponse[list[CustomAgentRead]])
@AppAutoInjector.auto_inject()
async def set_project_custom_agents(
    project_id: UUID,
    payload: ProjectCustomAgentsUpdate,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[list[CustomAgentRead]]:
    """Replace the default set. Conversations already in the project are untouched."""
    result = project_service.set_agents(user_id, project_id, payload.custom_agent_ids)
    return ApiResponse(success=True, message="Project agents updated", data=result)


@router.put("/{project_id}/conversations/{conversation_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def attach_conversation(
    project_id: UUID,
    conversation_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Move a conversation into the project and seed the project's agents."""
    project_service.attach_conversation(user_id, project_id, conversation_id)
    return ApiResponse(success=True, message="Conversation attached", data=None)


@router.delete("/{project_id}/conversations/{conversation_id}", response_model=ApiResponse[Any])
@AppAutoInjector.auto_inject()
async def detach_conversation(
    project_id: UUID,
    conversation_id: UUID,
    project_service: ProjectService,
    user_id: UUID,
) -> ApiResponse[Any]:
    """Release a conversation from the project, keeping its seeded agents."""
    project_service.detach_conversation(user_id, project_id, conversation_id)
    return ApiResponse(success=True, message="Conversation detached", data=None)
```

- [ ] **Step 4: Register the providers**

In `app/core/container.py`, add `"app.api.projects"` to the `wiring_config` modules
list, import `ProjectRepository`, `ProjectService`, and `ProjectContextService`, and
add these providers next to `custom_agent_service`:

```python
    project_repository = providers.Factory(
        ProjectRepository,
        session_factory=db.provided.session,
        async_session_factory=db.provided.async_session,
    )

    project_service = providers.Factory(
        ProjectService,
        repository=project_repository,
        custom_agent_repository=custom_agent_repository,
        conversation_validation_utils=conversation_validation_utils,
    )

    project_context_service = providers.Factory(
        ProjectContextService,
        project_repository=project_repository,
    )
```

Match the `session_factory` / `async_session_factory` argument style used by
`custom_agent_repository` in that same file rather than copying the snippet
blindly — the container deep-copies provider arguments and a raw `sessionmaker`
does not survive it.

Then add `project_service=project_service` to the `conversation_service` provider,
`project_context_service=project_context_service` to the `message_service` provider,
and `project_context_service=container.project_context_service()` inside
`_create_ai_service`.

- [ ] **Step 5: Register the router**

In `app/api/__init__.py`, add `from app.api.projects import router as projects_router  # noqa: E402`
and `"projects_router"` to `__all__`. In `app/main.py`, add
`app.include_router(projects_router)` immediately after the conversations router,
and add `projects_router` to the import from `app.api`. Register it once — only
`custom_agents` is dual-registered under `/ai`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_projects_api.py -v`
Expected: 9 passed.

- [ ] **Step 7: Verify the container still builds**

Run: `.venv/Scripts/python.exe -m pytest tests/test_container_import.py tests/test_container_async_wiring.py tests/test_container_reuse.py -v`
Expected: all passed. A broken provider graph shows up here, not in the route tests.

- [ ] **Step 8: Write the frontend contract**

Create `plans/PROJECTS_FE_CONTRACT.md` following the structure of
`plans/CUSTOM_AGENTS_FE_CONTRACT.md`. It must cover:

- All nine endpoints with camelCase request and response bodies, taken from the
  passing tests above rather than written from memory.
- The `projectId` query parameter on `GET /conversations`, and `projectId` on
  `ConversationCreate` and `ConversationRead`.
- The error contract: 403 `PROJECT_FORBIDDEN` for cross-user access, 404
  `PROJECT_NOT_FOUND` for missing or soft-deleted, 404
  `PROJECT_CONVERSATION_NOT_FOUND` for a wrong-project detach, 422 for
  instructions over 8000 characters.
- Seeding: happens on conversation create and on attach, insert-if-absent, never
  removes; `PUT /projects/{id}/custom-agents` does not reach existing conversations.
- Attach on a conversation already in another project is a move.
- Delete detaches conversations rather than deleting them.
- Composition: project instructions lead, separated from the conversation persona
  by the two literal headers `Project instructions:` and
  `Conversation-specific instructions:`, each part capped at 8000 characters
  independently, so the frontend can preview what the model receives.

- [ ] **Step 9: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check app/api/projects.py app/api/__init__.py app/core/container.py app/main.py tests/test_projects_api.py
git add app/api/projects.py app/api/__init__.py app/core/container.py app/main.py plans/PROJECTS_FE_CONTRACT.md tests/test_projects_api.py
git commit -m "feat: expose the project API and wire the container"
```

---

## Task 8: Streamlit UI

**Files:**
- Modify: `demo.py` — client helpers near `set_conversation_custom_agents` (line 1261), sidebar at `render_sidebar` (line 5583), new `render_project_view`, and the `active_view` dispatch in `main` (line 12870)
- Test: `tests/test_demo_projects.py`

**Interfaces:**
- Consumes: the endpoints from Task 7.
- Produces: `list_projects()`, `create_project(name, description, instructions)`, `update_project(project_id, fields)`, `delete_project(project_id)`, `get_project(project_id)`, `set_project_custom_agents(project_id, custom_agent_ids)`, `attach_conversation_to_project(project_id, conversation_id)`, `detach_conversation_from_project(project_id, conversation_id)`, `render_project_view()`, and a `project_id` keyword on the existing `get_conversations`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_demo_projects.py`:

```python
"""Streamlit project helpers and sidebar wiring."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from typing import Any
from uuid import uuid4

import pytest


class _SessionState(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class _CacheDecorator:
    def __call__(self, *args: Any, **kwargs: Any):
        return lambda func: func

    def clear(self) -> None:
        return None


class _StreamlitStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("streamlit")
        self.session_state = _SessionState()
        self.query_params: dict[str, str] = {}
        self.cache_data = _CacheDecorator()
        self.cache_resource = _CacheDecorator()

    def set_page_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def markdown(self, *args: Any, **kwargs: Any) -> None:
        return None

    def __getattr__(self, name: str):
        def _noop(*args: Any, **kwargs: Any):
            return None

        return _noop


def _import_demo_with_ui_stubs(monkeypatch: pytest.MonkeyPatch):
    streamlit_stub = _StreamlitStub()
    components_module = types.ModuleType("streamlit.components")
    components_v1_module = types.ModuleType("streamlit.components.v1")
    components_v1_module.html = lambda *args, **kwargs: None
    components_v1_module.declare_component = lambda *args, **kwargs: (
        lambda **_component_kwargs: _component_kwargs.get("default")
    )
    components_module.v1 = components_v1_module
    streamlit_stub.components = components_module
    markdown_stub = types.ModuleType("markdown")
    markdown_stub.markdown = lambda text, **_kwargs: text

    monkeypatch.setitem(sys.modules, "streamlit", streamlit_stub)
    monkeypatch.setitem(sys.modules, "streamlit.components", components_module)
    monkeypatch.setitem(sys.modules, "streamlit.components.v1", components_v1_module)
    monkeypatch.setitem(sys.modules, "markdown", markdown_stub)
    sys.modules.pop("demo", None)
    return importlib.import_module("demo")


def test_project_client_helpers_exist(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)

    for name in (
        "list_projects",
        "create_project",
        "update_project",
        "delete_project",
        "get_project",
        "set_project_custom_agents",
        "attach_conversation_to_project",
        "detach_conversation_from_project",
        "render_project_view",
    ):
        assert hasattr(demo, name), f"demo.{name} is missing"


def test_get_conversations_accepts_a_project_filter(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)

    assert "project_id" in inspect.signature(demo.get_conversations).parameters


def test_list_projects_calls_the_endpoint(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    calls = []

    def _fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {"success": True, "data": [{"id": str(uuid4()), "name": "Roadmap"}]}

    monkeypatch.setattr(demo, "make_api_request", _fake_request)

    result = demo.list_projects()

    assert calls == [("GET", "/projects")]
    assert result[0]["name"] == "Roadmap"


def test_attach_calls_put_on_the_membership_route(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    project_id, conversation_id = uuid4(), uuid4()
    calls = []

    monkeypatch.setattr(
        demo,
        "make_api_request",
        lambda method, path, **kwargs: calls.append((method, path)) or {"success": True},
    )

    demo.attach_conversation_to_project(str(project_id), str(conversation_id))

    assert calls == [("PUT", f"/projects/{project_id}/conversations/{conversation_id}")]


def test_project_settings_are_not_rendered_inside_tabs(monkeypatch):
    """Streamlit garbage-collects widget state for tabs that are not open, so an
    unsaved 8000-character instruction edit would vanish on a tab switch."""
    demo = _import_demo_with_ui_stubs(monkeypatch)

    source = inspect.getsource(demo.render_project_view)

    assert "st.tabs" not in source


def test_sidebar_renders_a_projects_section(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)

    source = inspect.getsource(demo.render_sidebar)

    assert "Projects" in source
    assert "render_project_view" in inspect.getsource(demo.main)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_demo_projects.py -v`
Expected: failures on the missing `demo.list_projects` attribute.

- [ ] **Step 3: Add the client helpers**

Insert immediately after `set_conversation_custom_agents` in `demo.py` (line 1261):

```python
def list_projects() -> list[dict[str, Any]]:
    """The signed-in user's projects, newest first."""
    response = make_api_request("GET", "/projects")
    if not response or not response.get("success"):
        return []
    return response.get("data") or []


def get_project(project_id: str) -> dict[str, Any] | None:
    response = make_api_request("GET", f"/projects/{project_id}")
    if not response or not response.get("success"):
        return None
    return response.get("data")


def create_project(
    name: str, description: str | None = None, instructions: str | None = None
) -> dict[str, Any] | None:
    payload = {"name": name, "description": description, "instructions": instructions}
    response = make_api_request("POST", "/projects", payload)
    if not response or not response.get("success"):
        return None
    return response.get("data")


def update_project(project_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    response = make_api_request("PATCH", f"/projects/{project_id}", fields)
    if not response or not response.get("success"):
        return None
    return response.get("data")


def delete_project(project_id: str) -> bool:
    response = make_api_request("DELETE", f"/projects/{project_id}")
    return bool(response and response.get("success"))


def set_project_custom_agents(project_id: str, custom_agent_ids: list[str]) -> bool:
    response = make_api_request(
        "PUT",
        f"/projects/{project_id}/custom-agents",
        {"customAgentIds": custom_agent_ids},
    )
    return bool(response and response.get("success"))


def attach_conversation_to_project(project_id: str, conversation_id: str) -> bool:
    response = make_api_request(
        "PUT", f"/projects/{project_id}/conversations/{conversation_id}"
    )
    return bool(response and response.get("success"))


def detach_conversation_from_project(project_id: str, conversation_id: str) -> bool:
    response = make_api_request(
        "DELETE", f"/projects/{project_id}/conversations/{conversation_id}"
    )
    return bool(response and response.get("success"))
```

Add a `project_id: str | None = None` keyword to the existing `get_conversations`
and include `projectId` in its query parameters when it is set.

- [ ] **Step 4: Add the sidebar section**

In `render_sidebar` (`demo.py:5583`), after the "Manage Conversations" button and
before the first `st.divider()`, add a Projects block that loads
`st.session_state.projects_list` once via `list_projects()` guarded by
`st.session_state.projects_loaded` (mirroring the `conversations_loaded` pattern
directly below it), renders a "New Project" button, and renders one button per
project that sets `st.session_state.current_project_id`,
`st.session_state.active_view = "project"`, and calls `st.rerun()`.

Initialise `projects_list = []`, `projects_loaded = False`, and
`current_project_id = None` alongside the existing `conversations_list` defaults.

- [ ] **Step 5: Add the project view**

Add `render_project_view()` to `demo.py`. It must render, on one full page and
**not** inside `st.tabs()`: a name input, a description input, an instructions
`st.text_area` with `max_chars=8000`, an agents `st.multiselect` backed by
`list_custom_agents()`, an explicit Save button calling `update_project` and
`set_project_custom_agents`, a Delete button, the project's conversations from
`get_conversations(project_id=...)`, and a "New chat in this project" button that
sets `current_conversation_id = "pending_new"` and `active_view = "chat"` while
keeping `current_project_id` so the new conversation is created with it.

In `main()` (`demo.py:12870`), dispatch `active_view == "project"` to
`render_project_view()`. NOTE: `active_view` is write-only today (9 assignments, 0 reads), so there are no existing branches to sit alongside — you are adding the first read site, and the else-branch must preserve the current default rendering.

- [ ] **Step 6: Add the move control**

In the existing "Manage Conversations" dialog, add a project selectbox per
conversation whose options are "No project" plus the user's projects. Selecting a
project calls `attach_conversation_to_project`; selecting "No project" on a
conversation that has one calls `detach_conversation_from_project`. After either,
set `st.session_state.conversations_loaded = False` so the sidebar refetches.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_demo_projects.py -v`
Expected: 6 passed.

- [ ] **Step 8: Run the demo suites that touch the sidebar**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/test_demo_conversation_manager.py tests/test_demo_custom_agents.py tests/test_demo_refactor_contract.py tests/test_streamlit_width_deprecation.py -v
```
Expected: all passed.

- [ ] **Step 9: Run the whole suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: no failures. The suite was fully green before this work, so any failure
here belongs to this branch — do not attribute it to a pre-existing condition.

- [ ] **Step 10: Lint and commit**

```bash
.venv/Scripts/python.exe -m ruff check demo.py tests/test_demo_projects.py
git add demo.py tests/test_demo_projects.py
git commit -m "feat: add the projects sidebar and project view"
```

---

## Manual verification

After Task 8, confirm the feature works in the running app, not only in tests:

1. Start the API and Streamlit as the project's run instructions describe.
2. Create a project with instructions `Always answer in Vietnamese.`
3. Attach a custom agent to it.
4. Create a conversation inside the project; confirm the agent is pre-attached.
5. Send a message; confirm the reply honours the project instruction.
6. Add a conversation-level persona; send another message; confirm both apply.
7. Move an existing project-less conversation into the project; confirm the
   instruction takes effect on its next turn.
8. Delete the project; confirm both conversations survive and lose the instruction.
