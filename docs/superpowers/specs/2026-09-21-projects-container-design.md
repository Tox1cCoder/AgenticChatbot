# Projects: container, instructions, and membership (slice 1)

Date: 2026-09-21
Status: approved design, not yet planned or implemented

## Goal

Give users a Claude/ChatGPT-style **Project**: a named container that holds many
conversations, carries instructions that apply to all of them, and supplies a
default set of custom agents to conversations created in or moved into it.

## Scope

This document specifies **slice 1 only**. The full feature request also covered
shared documents, multiple plans, conversation forking, and project-level
retained context. Those are deliberately excluded here and are listed under
[Deferred slices](#deferred-slices) with the constraints already discovered for
each, so a later spec does not have to rediscover them.

Slice 1 is additive. No existing table is re-scoped, no data is backfilled, and
every conversation that exists today keeps behaving exactly as it does now.

### In scope

- `projects` table, owner-scoped, soft-deleted.
- Project instructions, composed with the existing per-conversation
  `persona_prompt`.
- Conversation membership: at most one project per conversation.
- Project default custom agents, seeded into conversations on create and on
  attach.
- REST API, Streamlit UI, and a frontend contract document.

### Out of scope

- Sharing a project with another user. Projects are owner-scoped, like every
  other owned entity in this codebase.
- A project-level default model or provider.
- A project-level planning-mode default.
- Project badges on conversation buttons in the flat sidebar chat list.
- Restoring a soft-deleted project.

## Background: what the codebase already constrains

| Area | Existing fact | Consequence |
| --- | --- | --- |
| Per-conversation instruction | `conversations.persona_prompt`, capped at 8000 chars by `sanitize_persona` (`app/utils/text_processing.py:111`) | Project instructions must meet it somewhere |
| Persona plumbing | `persona` is threaded through roughly 15 sites in `app/ai/` — `graph.py:1492`, `workflow/specialists.py:524`, `workflow/routing.py:282`, and every agent (chat, search, planning, canvas, router) | A parallel pipeline field would have to follow it everywhere, and every missed site fails silently |
| Persona assembly | Only three sites build it: `message_service.py:3912`, `message_service.py:1059` (feeding the HITL resume path at `:2516`), `ai_service.py:153` | Composition has exactly three call sites to change |
| Persona rendering | `_build_persona_block` (`app/ai/prompts.py:477`) wraps the text in an injection-resistant sandbox | Composed text can reuse that sandbox unchanged |
| Custom agent resolution | `build_runtime_state(owner_id, conversation_id)` reads `conversation_custom_agents` at request time | Seeding rows leaves the resolver untouched |
| Attachment shape | `ConversationCustomAgent` (`app/models/custom_agent.py:78`) | A project attachment table can copy it structurally |
| Router registration | Only `custom_agents` is dual-registered under `/ai` (`app/main.py:402`); `conversations` is registered once | Projects register once |

## Approach

Project instructions are **composed into the existing `persona` string** at the
three assembly sites, rather than carried as a new field on
`WorkflowExecutionRequest`.

Rejected alternatives:

- **A parallel `project_instructions` field.** Cleaner separation downstream,
  but it must be added to both request schemas and threaded through all ~15
  `app/ai/` sites. It also re-exposes the known hazard where `_to_ai_request`
  silently drops fields the AI-layer schema lacks. Composition avoids the
  hazard entirely by not adding a field.
- **Copying project instructions into `persona_prompt` at conversation
  creation.** Requires no prompt-layer change at all, but edits to a project's
  instructions would never reach conversations already created, and the two
  texts would become one indistinguishable blob. That is a conversation
  template, not project instructions.

## Data model

### `projects`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | primary key |
| `owner_id` | UUID | FK `users.id`, indexed |
| `name` | String(255) | not null |
| `description` | Text | nullable |
| `instructions` | Text | nullable; the project system instruction |
| `created_at` | timestamptz | not null |
| `updated_at` | timestamptz | not null, `onupdate` |
| `deleted_at` | timestamptz | nullable; soft delete |

Index `ix_projects_owner_deleted (owner_id, deleted_at)`, mirroring
`ix_custom_agents_owner_deleted`.

There is no slug and no uniqueness on `name`. `custom_agents` needs a slug for
its picker; a project is addressed by id everywhere. Duplicate project names are
allowed, matching Claude. Adding uniqueness later is a migration; removing it is
not, so the looser constraint is the safer default.

### `project_custom_agents`

A structural copy of `ConversationCustomAgent`.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | primary key |
| `created_at` | timestamptz | not null |
| `owner_id` | UUID | FK `users.id` |
| `project_id` | UUID | FK `projects.id` |
| `custom_agent_id` | UUID | FK `custom_agents.id` |
| `agent_order` | Integer | not null, default 0 |

Unique constraint `uq_project_custom_agents_project_agent (project_id,
custom_agent_id)`. Index `ix_project_custom_agents_owner_project (owner_id,
project_id)`.

Matching `ConversationCustomAgent`'s shape means seeding is a straight row copy
rather than a translation.

### `conversations.project_id`

Nullable UUID, FK `projects.id`. Partial index
`ix_conversations_project_updated (project_id, updated_at) WHERE deleted_at IS
NULL` for the project's conversation list.

### Migration

Purely additive: two `create_table` calls and one `add_column`. Existing rows
get `project_id = NULL` and behave identically to today, so there is no backfill
and no data migration.

No new enum types are introduced. This repository has previously been bitten by
`op.create_table` re-creating a column's enum without `checkfirst`; the
migration docstring states that no enum is created here, so that nobody adds one
without reaching for `create_type=False`.

The `down_revision` is taken from `alembic heads` at implementation time, not
guessed. `Project` and `ProjectCustomAgent` must be exported from
`app/models/__init__.py`, because that registry is what Alembic autogenerate
reads; omitting them produces an empty migration.

## Instruction composition

### The pure function

In `app/utils/text_processing.py`, beside `sanitize_persona`:

```python
def compose_system_instruction(
    project_instructions: str | None,
    persona_prompt: str | None,
) -> str | None:
```

| Input | Output |
| --- | --- |
| Both `None` or blank | `None` — no persona block is built, as today |
| Persona only | the sanitized persona, verbatim |
| Project instructions only | the sanitized project instructions, verbatim |
| Both | the headered composition below |

When both are present:

```
Project instructions:
<project text>

Conversation-specific instructions:
<persona text>
```

Headers appear only when both parts are present. A single part passes through
unchanged.

### Invariant

When `project_id IS NULL`, the composed output is byte-identical to what the
current code produces. This is what makes the change safe for every existing
conversation, and it is asserted by a test rather than assumed.

### Character budget

Each part is sanitized **independently** — `sanitize_persona` applied to the
project text and to the persona text separately, 8000 characters each — and the
composed result is never re-truncated.

Composing first and truncating after is a silent data-loss bug: project text
leads, so an 8000-character cut would discard the conversation's own persona
entirely. The implementation must not call `sanitize_persona` on the composed
string.

`projects.instructions` is validated with `Field(None, max_length=8000)` at the
schema layer, matching `persona_prompt`. The worst case reaching the model is
roughly 16,060 characters of system instruction.

This is safe downstream: `app/ai/workflow/routing.py:282` already clips `persona`
against its own budget and records the clip in `truncated_fields`, so a longer
string degrades through a path built for it.

### The resolver

`app/services/project_context_service.py`:

```python
def resolve_system_instruction(self, conversation) -> str | None:
```

Reads `conversation.project_id`, fetches the project's instructions when set
(filtering `deleted_at IS NULL`), and calls `compose_system_instruction`. One
additional `SELECT` per turn against a small owner-indexed table, which is
negligible beside the LLM calls in the same turn.

A `lazy="joined"` relationship on `Conversation` is explicitly **not** used: it
would add a LEFT JOIN to every conversation query in the application, including
the list endpoints, to serve three call sites.

Because the resolver reads live, editing a project's instructions takes effect
on the next turn of every conversation in that project.

### Call sites replaced

- `app/services/message_service.py:3912` —
  `_build_user_message_workflow_request`
- `app/services/message_service.py:1059` — `_get_conversation_context`, feeding
  the HITL resume path at `:2516`
- `app/services/ai_service.py:153` — `_prepare_request`

`app/ai/prompts.py` is not modified. Composed text lands inside the existing
`_build_persona_block` sandbox. Project instructions and persona are both
user-authored and carry the same trust level, so a single sandbox is correct.
The block's `--- BEGIN USER PERSONA ---` label is narrower than what it now
wraps, but rewording it would change the prompt for every project-less
conversation and break the invariant above. Slice 1 accepts the imprecise label.

## Membership lifecycle

### Creating a conversation in a project

`ConversationCreate.project_id: UUID | None`.
`ConversationService.create_conversation` validates that the project exists, is
not soft-deleted, and is owned by the caller before the factory runs.

**Error contract.** Projects follow the convention already established for
owner-scoped entities in this API: `ProjectForbiddenError` (403,
`PROJECT_FORBIDDEN`) when the project exists but belongs to another user, and
`ProjectNotFoundError` (404, `PROJECT_NOT_FOUND`) only when it is genuinely
missing or soft-deleted. This mirrors `CustomAgentForbiddenError` /
`CustomAgentNotFoundError` (`app/core/exceptions/custom_agent.py`), which
`tests/test_custom_agents_api.py:148` pins at 403 for cross-user reads, and the
`AuthorizationException` that conversations raise.

An earlier draft of this spec specified 404 for cross-user access to avoid
confirming that another user's project exists. That was reversed: it would have
made projects the only owner-scoped entity in the API behaving differently, and
a frontend branching on 403 everywhere else would need a special case.

`ConversationFactory` gains `project_id` in **both** `create_from_schema`
(`app/factories/conversation_factory.py:23`) and `create_from_dict` (`:40`).
Both paths, because updating only one is how a field ends up silently absent
depending on the caller.

After the conversation row commits, `project_custom_agents` rows are copied into
`conversation_custom_agents`, preserving `agent_order`.

### Attaching and detaching

Not expressed through `ConversationUpdate`: every field there treats `None` as
"unchanged", so `project_id=None` could not distinguish "leave it" from "remove
it". Instead, two dedicated routes, mirroring the separate `conversation_router`
that `custom_agents` already ships (`app/api/custom_agents.py:25`):

```
PUT    /projects/{project_id}/conversations/{conversation_id}
DELETE /projects/{project_id}/conversations/{conversation_id}
```

Both validate that the caller owns the project **and** the conversation.

**Attach** sets `project_id` and seeds agents: the project's agents are inserted
into `conversation_custom_agents` where not already present. It is a union —
nothing is removed and nothing duplicates, guaranteed by the unique
`(conversation_id, custom_agent_id)` constraint. Re-attaching is idempotent.

Attaching a conversation that already belongs to a **different** project is a
move, not an error: `project_id` is overwritten with the new project and the new
project's agents are seeded. Agents seeded from the previous project are left in
place, for the same reason detach leaves them — the user may have curated them
since, and the endpoint cannot distinguish seeded rows from hand-added ones.

Seeding on create and seeding on attach are the same insert-if-absent helper.

**Detach** sets `project_id = NULL` and leaves seeded agents in place;
un-seeding would delete attachments the user may have curated since. Project
instructions stop applying on the next turn.

Detach requires that the conversation currently belongs to the project named in
the path. `DELETE /projects/{Y}/conversations/{cid}` for a conversation in
project X returns 404 `PROJECT_CONVERSATION_NOT_FOUND` — the membership, not the
project, is what is missing — so a stale client cannot detach a conversation
from a project it is no longer in.

### Deleting a project

Soft-delete the project, then
`UPDATE conversations SET project_id = NULL WHERE project_id = :id`.

Conversations survive as loose conversations. This differs from Claude, which
deletes a project's chats along with it; the non-destructive behavior was chosen
deliberately because every other entity here soft-deletes, and because silently
destroying conversation history is discovered only after it happens.

Known asymmetry: the detach is a hard write while the project delete is soft, so
a future "restore project" feature would restore an empty project. There is no
restore in slice 1.

`project_custom_agents` rows are left in place, unreachable behind the soft
delete, consistent with soft-delete semantics elsewhere in this codebase.

### Defensive read

`resolve_system_instruction` filters `deleted_at IS NULL` on the project, so a
conversation still pointing at a soft-deleted project behaves as project-less
rather than inheriting a deleted project's instructions.

## API

New module `app/api/projects.py`, one router, registered once in `app/main.py`.
Not dual-registered under `/ai`: only `custom_agents` does that, and
`conversations` — which projects sit alongside — is registered once.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/projects` | list the caller's projects |
| `POST` | `/projects` | create |
| `GET` | `/projects/{id}` | read, including default agents |
| `PATCH` | `/projects/{id}` | update name, description, instructions |
| `DELETE` | `/projects/{id}` | soft-delete and detach conversations |
| `GET` | `/projects/{id}/custom-agents` | list default agents |
| `PUT` | `/projects/{id}/custom-agents` | replace the ordered set |
| `PUT` | `/projects/{id}/conversations/{cid}` | attach and seed agents |
| `DELETE` | `/projects/{id}/conversations/{cid}` | detach |

`GET /projects` returns a plain list rather than a paginated envelope, matching
`GET /custom-agents`. Projects are few per user; pagination can be added later
without breaking clients that ignore it.

`PUT /projects/{id}/custom-agents` changes only the project's default set. It
does **not** reach into conversations already in the project, consistent with
seeding happening at create and attach time and nowhere else.

There is no `GET /projects/{id}/conversations`. The existing `GET /conversations`
(`app/api/conversations.py:76`) gains an optional `projectId` query parameter
instead; it already implements pagination, ordering, and the `include`
machinery, and a nested route would mean maintaining a second copy of all of it.
A `projectId` the caller does not own returns 403, matching the error contract
above, rather than an empty page.

No sentinel value for "unassigned" is provided, because nothing in the UI needs
that list: the flat chat list shows every conversation, as Claude's does.

### Schemas

`app/schemas/project.py`, following the `alias_generator=to_camel` convention
used throughout:

- `ProjectCreate` — `name`, `description`, `instructions`
  (`Field(None, max_length=8000)`)
- `ProjectUpdate` — all optional
- `ProjectRead` — plus `conversation_count`, and
  `custom_agents: list[CustomAgentRead] | None` populated only when requested,
  mirroring the same field on `ConversationRead`
- `ProjectCustomAgentsUpdate` — a direct copy of `ConversationCustomAgentsUpdate`
  (`app/schemas/custom_agent.py:219`), including its no-duplicates validator

`conversation_count` counts live conversations only (`deleted_at IS NULL`). It
is computed on the list endpoint with one grouped subquery, the way
`message_count` already works on `ConversationRead` — not per row, which would
be an N+1 across the sidebar.

Existing schemas touched: `ConversationCreate.project_id` (optional) and
`ConversationRead.project_id`. `ConversationUpdate` is deliberately unchanged.

### Wiring

`ProjectRepository`, `ProjectService`, and `ProjectContextService` are
registered in `app/core/container.py` and reached through
`AppAutoInjector.auto_inject()`, like every other route dependency.

## Streamlit UI

`demo.py` is the only frontend in this repository.

**Sidebar** (`demo.py:5583`) gains a Projects section above Conversations: a
"New Project" button and the project list. Selecting a project sets
`current_project_id` and `active_view = "project"` — a third view alongside the
existing `"chat"` and `"planning"`. Session state mirrors the conversation
pattern: `projects_list`, `projects_loaded`, `projects_last_fetch_params`.

**`render_project_view()`** shows name, description, an instructions
`text_area`, the default-agents multiselect, the project's conversations via
`get_conversations(project_id=...)`, and a "New chat in this project" button.

Project settings live on the full-page project view and **not** inside
`st.tabs()`. Only the open tab renders in Streamlit, so widget-keyed
`session_state` is garbage-collected on tab switch, and an unsaved
8000-character instruction edit would vanish silently. A full page plus an
explicit Save button avoids this.

**Move and detach** live in the existing "Manage Conversations" dialog, already
the place conversations are administered.

**Client helpers** beside `set_conversation_custom_agents` (`demo.py:1261`):
`list_projects`, `create_project`, `update_project`, `delete_project`,
`set_project_custom_agents`, `attach_conversation_to_project`,
`detach_conversation_from_project`, and a `project_id` parameter on
`get_conversations`.

## Frontend contract

`plans/PROJECTS_FE_CONTRACT.md`, following `CUSTOM_AGENTS_FE_CONTRACT.md`:
endpoints with camelCase request and response bodies, the error contract (403
`PROJECT_FORBIDDEN` for cross-user access, 404 `PROJECT_NOT_FOUND` for missing
or soft-deleted, 404 `PROJECT_CONVERSATION_NOT_FOUND` for a wrong-project
detach), seeding behavior on both create and attach, the
8000-characters-per-part cap, and the composition rule. The frontend needs the composition rule to
preview what the model will actually receive.

## Testing

| Target | What it pins |
| --- | --- |
| `compose_system_instruction` unit tests | the four-case table above |
| Cap regression test | 8000-char project text plus 8000-char persona; asserts the persona survives at the tail |
| Invariant test | `project_id IS NULL` yields byte-identical output to the pre-change assembly |
| Resolver tests | a soft-deleted project reads as project-less; a missing project does not raise |
| Seeding tests | create seeds; attach unions insert-if-absent; `agent_order` preserved; re-attach is idempotent |
| Lifecycle tests | project delete detaches and leaves conversations intact |
| Ownership tests | the four cross-user cases below, each asserting 403 |
| Migration test | upgrade then downgrade against the Postgres integration database |

Ownership tests run in the reverse direction — verifying that unauthorized
access is blocked, not only that authorized access works. User B cannot: read
user A's project; attach their own conversation to A's project; attach A's
conversation to their own project; create a conversation naming A's project.
Each asserts 403, matching `tests/test_custom_agents_api.py:148`.

Postgres integration tests derive `TEST_DATABASE_URL` from
`settings.database_url` and are runnable in this environment, so "skipped,
unavailable" is not an acceptable outcome for the migration test.

## Deferred slices

Each gets its own design, plan, and implementation cycle. The constraints below
were found while designing slice 1 and are recorded so a later spec does not
have to rediscover them.

1. **Project-scoped documents.** `documents.conversation_id` is NOT NULL with a
   unique `(conversation_id, filename_key)`, and Qdrant payloads filter on
   `conversation_id` (`app/services/rag_retrieval.py:626`). A PostgreSQL
   migration alone will not re-scope this; a Qdrant payload backfill is
   required. This is the riskiest slice and the reason slice 1 was kept
   additive.
2. **Fork conversation.** Nothing exists today.
   `conversation_memory_summaries` has a composite FK to
   `(messages.conversation_id, messages.sequence)`, so a fork needs its own
   summary row rather than a shared one.
3. **Multiple plans.** `task_plans.conversation_id` is NOT NULL with a unique
   `(conversation_id, task_order)`, and `conversations.plan_lifecycle` is a
   single enum. One plan per conversation is a structural fact, not a UI
   limitation; multiple plans require a plan-container entity and moving
   `plan_lifecycle` onto it.
4. **Project-level retained context.** `ConversationMemorySummary` is 1:1 with a
   conversation. Project-level memory is a new entity, not a widening of that
   one. It would compose through `ProjectContextService`, the same seam slice 1
   introduces.
