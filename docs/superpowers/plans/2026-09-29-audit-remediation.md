# 2026-09 Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every issue the September 2026 audit reported but deliberately did not fix.

**Architecture:** The audit ran as seven area passes (DB, API, services, AI orchestration,
AI tools, core/infra, client sidecar). Clear-cut defects were fixed in commits
`0a51a1e7`..`0fdb9804`. What remains needs a decision, a migration, or a behaviour
change. Phase 0 is ready to run and is written out step by step. Phases 1–5 are one
task per issue, with the decision each one needs. Expand a phase into its own detailed
plan once its decisions are made. Writing code for an undecided design would be guesswork.

**Tech Stack:** FastAPI, SQLAlchemy 2 + Alembic (PostgreSQL), Qdrant, Celery/Redis,
LangGraph, Streamlit (`demo.py`), local sidecar `client_backend/` on :8100.

**Spec:** the audit reports, summarised in the Appendix. The session note is at
`~/claude-assistant/state/sessions/2026-09-25-codebase-audit.md`.

## Status (2026-10-01, end of round 4)

Round 4 shipped: `3adb3eee` (epoch carried through pauses, epoch cap enforced, research
accounting kept), `99ee77e3` + `3b1d7bc7` (Phase 5 splits, all ≤ 8), `b71f8609` (dead
state keys), `1c743e7f`, `496429b6`, `41b93954` (generation row settled on every turn
ending — reproduced on PostgreSQL), `aa8157b4`, `4697d43d` (fallback keeps tool
exclusions), `26ade534` (Python 3.11 + ruff py311; UP042 ignored on purpose). Full suite
6128 passed, 0 failed. Every task in this plan is done except 6 (deferred by Thai) and
10/24 (Thai's hands).

### Still open after round 4 (found while fixing; each needs a decision or its own plan)
| Item | Note |
|---|---|
| Resume holds no turn lock | approval pauses no longer block the conversation, so a new message can start while an approval is pending; approving during it runs the resume untracked. Hold the lock, or refuse the resume? |
| Approval inside a continued epoch | nothing persisted, no HITL record, so it can never be resumed; the row now ends `failed` |
| Stop during a resume | not acted on until the resume finishes (no registry entry / stop watch) |
| Stop on an approval pause | row → `completed_partial` but the HITL interrupt stays pending |
| Continue doesn't restamp `producer_token` | if another worker serves the Continue and dies, the reaper won't clear the row |
| Continued-epoch partial text | lost on disconnect/Stop |
| `planning_budget_reached` / `planning_call_count` | documented in AI_SDK_FE_CONTRACT but never produced (always false / 0) — wire or drop from the contract |
| `WorkerTask.parent_context` | built, never reaches the worker prompt |
| `tool_execution_consecutive_errors_limit` | setting no longer read |
| Upload staging | still writes to disk on the event loop |
| Flaky | `test_tool_execution_recovery::test_safe_session_reconnect_counts_as_next_attempt` fails alone (timing bound) |

## Status (2026-10-01)

Full suite after round 3 (with another session's uncommitted preview-stream work in the
tree): 6062 passed, 312 skipped, 1 failed. The failure, a stale call-site manifest entry
left by the Task 26 deletions, is fixed in `280801dc`. `ruff check .` is clean.

Rounds 2–3 (2026-09-30 → 10-01), decided by Thai: SECRET_KEY / MODEL_ENCRYPTION_KEY
auto-generated into the DB, `api_host` → 127.0.0.1, sidecar logout requires a session,
Python floor 3.11 everywhere (do last), skill-file passwords handled by Thai.

| Item | State | Commit / note |
|---|---|---|
| 12 leftovers | done | `39aba1df` (`rag_agent.get_status`), `95e47436` (forwarded-path guard in `ServerAPIClient`, cmd.exe second hop removed, demo token moved from URL to a SameSite=Strict cookie) |
| 13 SpecialistFactory leak | done | `39aba1df`: definitions resolved per request |
| 14 gap | done | `1bbebdc0` |
| 17 receipts owner-scoped | done | `ce1d95f5` |
| 18 device-bound results | done | `1bbebdc0` |
| 19 leftovers | done | `1bbebdc0`, `39aba1df`, `95e47436`, `d43bbb93`. Still open: `documents.py` handlers await async `DocumentService` methods that do sync repository calls, and `get_processing_status` makes a sync Celery result call in an async def |
| 20, 21, 23 migrations | done | `ce1d95f5`: revisions `c0033ee1e8cd`, `371ffaf3a087` |
| 24 dead modules | open | the classifier refuses the `git rm`; Thai runs it |
| 25 dead settings | done | `e7c22a39` (Thai removes the `.env.example` lines) |
| 26 test-only code | done | `280801dc` |
| 27 unused dependencies | done | `e7c22a39` |
| 28 Python 3.11 + ruff py311 | open | run last, as one mechanical commit |
| Secrets in the DB | done | `38c10b16`: `server_secrets`, revisions `e983df693ada` and `9b132832d676` (partial `model_providers` unique index) |
| api_host / sidecar logout | done | `29a14b11`, `95e47436` |
| Second Continue refused; epoch limit never enforced; evidence counter never released | in progress | found in round 2 |
| Phase 5 | in progress | non-orchestration splits first; the orchestration ones come after the epoch fix |

Round 1 table (2026-09-30):

| Task | State | Commit / note |
|---|---|---|
| 1 bundle ships project schemas | done | `cf4d4636` (both builders; the `.ps1` twin was missing from the plan) |
| 2 DB tests off the app database | done | `762e9c5b`, `02c8ce3a` |
| 3 CORS | done | `fcd46dfa`. Thai's `.env` origins are now enforced |
| 4 dead Celery shim | done | `fcd46dfa` |
| 5 sidecar launch token | done | `0c7413cf`. Needs `CLIENT_TRUST_CHECKS_ENABLED=true` in `.env.client.example` (Thai) |
| 6 server MCP admin gate | **deferred by Thai** | leave as is for now |
| 7 token_version revocation | done | `67f128f2`, migration `60adc43e534e` (the API lifespan applies it at startup). No password-change route or logout-all route exists yet |
| 8 tool recovery scope | done | `eda376e6`. Custom agents keep `allow_all_server_tools` by design |
| 9 widget owner/session binding | done | `00bcfa0a` |
| 10 skill-file passwords | open | Thai's untracked files |
| 11 production-safe defaults | done | `fcd46dfa`. Non-dev needs `SECRET_KEY` ≥32 chars and explicit `CORS_ORIGINS`; add both to `.env.example` (Thai). `api_host` still defaults to `0.0.0.0` |
| 12 smaller leaks | mostly done | `fcd46dfa`, `9736dff3`, `00bcfa0a`, `eda376e6`. Open: sandbox launcher's second hop (`cmd.exe /c set`), `rag_agent.get_status` `str(e)`, the demo token in the URL/localStorage, the remaining sidecar routes with dot-segment path params (`common.proxy_server_request`) |
| 13 SpecialistFactory leak | open | needs its own plan |
| 14 close singletons on shutdown | done | `fcd46dfa`. Gap: a checkpoint pool built lazily by `ConversationService` with checkpoints disabled |
| 15 RAG tool scope | done | `eda376e6` |
| 16 deferred state empty key | done | `eda376e6` |
| 17, 18 | open | |
| 19 small correctness | partly done | `00bcfa0a`: task-status 404, conversation delete off the loop. Open: other sync-in-async routes, widget WATCH, dead resume chain, event listeners, planning call budget, document upload size limit (needs a new client setting), key_preview, MCP URL masking |
| 20, 21, 23 migrations | open | |
| 22 project memories | done | `cab22e26`, decided: soft-deleted with the project |
| 24–28 debloat | open | 24 needs Thai to run the `git rm` |
| Phase 5 | open | |

## Global Constraints

- Python: run everything with `./.venv/Scripts/python.exe` (3.13). The conda `agents` env is 3.11 and cannot run the suite.
- Lint: `./.venv/Scripts/python.exe -m ruff check .` must stay clean. Line length is 100 and is enforced.
- Tests: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider <paths>`. Every bug fix needs a test you have watched fail without the fix.
- Never run tests against the app database. PostgreSQL tests use `TEST_DATABASE_URL` (a `chatbot_test` database); `tests/integration/conftest.py` refuses the app database.
- Schema changes go through a new Alembic revision. Never `create_all` against a real database.
- Commits: imperative mood, subject ≤72 characters, one logical change each. Stage explicit paths, never `git add -A`: other sessions (Claude and Codex) share this checkout.
- `.env.example` and `.env.client.example` cannot be read by the assistant (global guard). A task that needs a template change says so, and Thai makes the edit.

## Review Focus

1. **Empty `CORS_ORIGINS`.** The default is `[]`, which today allows every origin. After Task 3 an unset value must still let the local Streamlit/demo frontend work. Task 3 pins what an empty list means.
2. **A sidecar bundle built on a clean machine.** The bundle tests catch a missing module only when the import is at module level. Task 1 runs the real import path, not just the static scan.
3. **A PostgreSQL test run with no `TEST_DATABASE_URL`.** Tests moved off the live database must skip with a clear reason, not error. Task 2 pins this.
4. **Existing tokens after the auth work (Phase 1).** Already-issued access tokens and stored sidecar sessions must keep working through the rollout, or users get logged out. Each Phase 1 task must say how it treats them.
5. **Migrations on a database with real data (Phase 3).** A new unique index or CHECK fails if existing rows violate it. Every Phase 3 migration first queries for violating rows.

---

## Phase 0 — Ready now (no decision needed)

### Task 1: Ship the project schemas in the sidecar bundle

`client_backend/api/projects.py` (commit `44403b25`) imports server schemas for its
OpenAPI `response_model`s. The bundle builder does not copy them, so the packaged
sidecar fails at import. `tests/test_client_backend_bundle.py` has 2 failures:
`test_bundle_ships_every_first_party_module_it_imports` and
`test_bundle_can_import_the_real_server_startup_path`.

**Files:**
- Modify: `scripts/build_client_backend_bundle.py:87-133` (copy list and package `__init__`s)
- Modify: `client_backend/api/projects.py:17` (import `ApiResponse` from its module, not the package)
- Test: `tests/test_client_backend_bundle.py` (existing, currently failing)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: a bundle whose `app/` also contains `schemas/custom_agent.py`, `schemas/project.py`, `schemas/responses/api_response.py`, `schemas/responses/paginated_response.py`, `repositories/utils/pagination.py`, `utils/case_conversion.py`.

- [ ] **Step 1: Confirm the failure**

Run: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_client_backend_bundle.py`
Expected: 2 failed, naming `app.schemas.project`, `app.schemas.custom_agent`, `app.schemas.responses`, `app.schemas.responses.paginated_response`.

- [ ] **Step 2: Import `ApiResponse` from its own module**

The bundle's package `__init__` files are written empty. The repo's
`app/schemas/responses/__init__.py` also re-exports two dead modules (Task 24). In
`client_backend/api/projects.py` replace

```python
from app.schemas.responses import ApiResponse
```

with

```python
from app.schemas.responses.api_response import ApiResponse
```

- [ ] **Step 3: Copy the modules and create their packages**

In `scripts/build_client_backend_bundle.py`, after the existing
`shutil.copy2(... "widget_runtime.py" ...)` call, add:

```python
    # client_backend/api/projects.py documents its routes with the server's own
    # project schemas; these are that import closure.
    for relative in (
        ("schemas", "custom_agent.py"),
        ("schemas", "project.py"),
        ("schemas", "responses", "api_response.py"),
        ("schemas", "responses", "paginated_response.py"),
        ("repositories", "utils", "pagination.py"),
        ("utils", "case_conversion.py"),
    ):
        target = app_dir.joinpath(*relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / "app" / Path(*relative), target)
```

Then, next to the existing `(app_dir / "services" / "__init__.py").write_text(...)` line, add:

```python
    for package in (
        ("schemas", "responses"),
        ("repositories",),
        ("repositories", "utils"),
        ("utils",),
    ):
        (app_dir.joinpath(*package) / "__init__.py").write_text("", encoding="utf-8")
```

Check that `Path` is already imported from `pathlib` at the top of the script; add it if not.

- [ ] **Step 4: Run the bundle tests**

Run: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_client_backend_bundle.py tests/client_backend/test_projects_proxy.py`
Expected: all pass. If the static scan names another module, add it to the tuple and rerun. Do not add it to `INTENTIONALLY_ABSENT_FROM_BUNDLE`: the sidecar imports it at startup.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_client_backend_bundle.py client_backend/api/projects.py
git commit -m "fix(bundle): ship the project schemas the sidecar imports"
```

### Task 2: Stop repository tests writing to the live app database

These tests build `Database(settings.database_url)` or use the module-level
`AsyncSessionLocal`, both bound to the app database, so every suite run inserts and
deletes rows there:
`test_project_repository`, `test_projects_api`, `test_project_service`,
`test_project_membership`, `test_user_memory_repository`, `test_conversation_search`,
`test_custom_agents_api`, `test_custom_agents_service`,
`test_custom_agents_message_service`, `test_hitl_api`, `test_repository_async_twins`,
`test_message_ordering_defaults`, `test_database_schema_contract`,
`test_graph_refactor_contract`. `tests/conftest.py::_async_db_available` probes the same engine.

**Decision (recommended default in bold):** **point the whole suite at the test database
when `TEST_DATABASE_URL` is set, and skip these modules when it is not.** The
alternative, gating each module individually, leaves `settings.database_url` pointing at
the app database for anything that later forgets the gate.

**Files:**
- Modify: `tests/conftest.py` (set `DATABASE_URL` from `TEST_DATABASE_URL` before any `app` import)
- Create: `tests/database_isolation.py` (the skip marker; `tests/` is a package and other tests already import `tests.*` helpers, and importing `conftest` directly can load it twice)
- Modify: each module listed above (apply the marker)
- Test: `tests/test_suite_database_isolation.py` (new)

**Interfaces:**
- Produces: `tests.database_isolation.requires_test_database`, a `pytest.mark.skipif` object.

- [ ] **Step 1: Write the failing guard test**

```python
"""The default suite must never be bound to the application database."""

import os

import pytest
from sqlalchemy.engine import make_url


def test_settings_database_is_the_test_database_when_one_is_configured():
    test_url = os.getenv("TEST_DATABASE_URL")
    if not test_url:
        pytest.skip("TEST_DATABASE_URL is not set; DB-backed modules are skipped instead")

    from app.core.config import settings

    assert make_url(settings.database_url).database == make_url(test_url).database


def test_db_backed_modules_are_marked():
    import pathlib
    import re

    root = pathlib.Path(__file__).parent
    offenders = [
        p.name
        for p in root.glob("test_*.py")
        if re.search(r"Database\(settings\.database_url\)|AsyncSessionLocal", p.read_text("utf-8"))
        and "requires_test_database" not in p.read_text("utf-8")
    ]
    assert offenders == []
```

- [ ] **Step 2: Run it to see it fail**

Run: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_suite_database_isolation.py`
Expected: `test_db_backed_modules_are_marked` FAILS listing the modules above.

- [ ] **Step 3: Rebind settings in `tests/conftest.py`**

Near the existing `os.environ["LANGSMITH_TRACING"] = "false"` lines (before any `app` import), add:

```python
# Tests that open real sessions must never reach the application database: they seed
# and delete rows by id. With a test database configured, the whole suite uses it;
# without one, modules marked requires_test_database are skipped.
_TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
if _TEST_DATABASE_URL:
    os.environ["DATABASE_URL"] = _TEST_DATABASE_URL
```

And create `tests/database_isolation.py`:

```python
"""Skip marker for tests that open real database sessions."""

import os

import pytest

requires_test_database = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="needs TEST_DATABASE_URL (a dedicated database, never the app database)",
)
```

Verify the env var name the settings class reads for `database_url` (Grep `database_url` in `app/core/config.py` for an `alias`/`validation_alias`) and use that name.

- [ ] **Step 4: Mark each module**

At the top of each listed module:

```python
from tests.database_isolation import requires_test_database

pytestmark = requires_test_database
```

If a module already defines `pytestmark`, make it a list that includes `requires_test_database`.

- [ ] **Step 5: Run with and without a test database**

Run: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_suite_database_isolation.py <the listed modules>`
Expected with no `TEST_DATABASE_URL`: the listed modules are skipped with the reason above, and the guard tests pass.
Then build `chatbot_test` with `alembic upgrade head` (not `create_all`), derive `TEST_DATABASE_URL` as in the `pg-integration-tests-are-runnable` memory, and rerun. Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/database_isolation.py tests/test_suite_database_isolation.py <the listed modules>
git commit -m "test: keep DB-backed tests off the application database"
```

### Task 3: Make CORS honour `settings.cors_origins`

`app/main.py:368-388`: with `*` configured, the API echoes any origin and allows
credentials. Otherwise it allows every origin and ignores the configured list. The
frontend authenticates with a bearer header, not cookies, so credentials are not needed
for `*`.

**Decision pinned here:** `*` → any origin, no credentials. An explicit list → exactly
those origins, with credentials. **An empty list keeps today's behaviour (any origin, no
credentials)**, so an unset variable doesn't break local development. Production is
guarded by Task 11.

**Files:**
- Modify: `app/main.py:367-388`
- Test: `tests/test_cors_policy.py` (new)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from fastapi.testclient import TestClient

import app.main as main


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


def test_wildcard_never_combines_any_origin_with_credentials(monkeypatch):
    monkeypatch.setattr(main.settings, "cors_origins", ["*"])
    response = _preflight(TestClient(main.create_app()), "https://evil.example")
    assert response.headers.get("access-control-allow-credentials") != "true"


def test_explicit_origins_are_the_only_ones_allowed(monkeypatch):
    monkeypatch.setattr(main.settings, "cors_origins", ["http://localhost:8501"])
    client = TestClient(main.create_app())
    allowed = _preflight(client, "http://localhost:8501")
    refused = _preflight(client, "https://evil.example")
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-origin" not in refused.headers
```

If `/health` is not the probe route, Grep `app/main.py` for a GET route that needs no auth and use it.

- [ ] **Step 2: Run to see both fail**

Run: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_cors_policy.py`
Expected: both FAIL. The first gets `allow-credentials: true`; the second gets `*` or the evil origin.

- [ ] **Step 3: Replace the CORS block**

```python
    # Bearer tokens travel in a header, so "*" never needs credentials; only an
    # explicit origin list is trusted with them. An empty list is the local-dev
    # default and keeps any-origin without credentials (production is refused
    # an empty or wildcard list by the settings validator).
    configured = [origin for origin in settings.cors_origins if origin]
    explicit = [origin for origin in configured if origin != "*"]
    use_explicit = bool(explicit) and "*" not in configured
    app.add_middleware(
        CORSMiddleware,
        allow_origins=explicit if use_explicit else ["*"],
        allow_credentials=use_explicit,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_private_network=True,
        expose_headers=["x-vercel-ai-ui-message-stream"],
    )
```

- [ ] **Step 4: Run the new tests and the existing CORS/sidecar tests**

Run: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_cors_policy.py` and Grep `tests/` for `cors` and run those files too.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_cors_policy.py
git commit -m "fix(api): stop reflecting any origin with credentials"
```

### Task 4: Remove the unconsumed `process_document_task` shim

`app/workers/document_processor.py:508` registers `process_document_task` on the default
`celery` queue, which no worker started by `start_worker` consumes. Nothing in `app/`
or `tests/` calls it.

**Files:**
- Modify: `app/workers/document_processor.py:505-~530` (delete the task)
- Test: `tests/test_celery_worker_config.py::test_every_beat_task_lands_on_a_queue_start_worker_consumes` (existing)

- [ ] **Step 1: Confirm no references**

Grep the repo, excluding `.venv`, `.worktrees`, `dist` and `docs`, for `process_document_task`.
Expected: only the definition and its `name=` string.

- [ ] **Step 2: Delete the task function and its decorator.**

- [ ] **Step 3: Run the worker tests**

Run: `./.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/test_celery_worker_config.py tests/test_cleanup_tasks.py` and Grep `tests/` for `document_processor` and run those files.
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add app/workers/document_processor.py
git commit -m "chore(workers): drop the unconsumed process_document_task shim"
```

---

## Phase 1 — Security (decision needed first)

Each task lists the decision and the recommended option. Expand the phase into a
detailed plan once the decisions are made.

### Task 5: Sidecar session takeover via `/auth/restore` — CRITICAL
- **Where:** `client_backend/api/auth.py:254` (`POST /auth/restore?user_id=`) and `:315` (`GET /auth/users`); no Host or Origin check on the sidecar.
- **Failure:** any local process, a command run as the KaniSandbox account, or a DNS-rebinding web page calls `/auth/users`, then `/auth/restore`. It gets access, refresh and local-session tokens, and can then switch the sandbox off or add an MCP server that runs as the real user.
- **Decision:** how the Streamlit frontend proves it is the legitimate client.
- **Recommended:** (1) At each launch the sidecar writes a random token to an owner-only file in `<profile>` that the sandbox account cannot read. `demo.py` sends it as `X-Kani-Client`, and middleware rejects every request without it except `/health`. (2) A Host allowlist of `localhost`, `127.0.0.1`, `[::1]` and `backend_host` (add `testserver` for tests). (3) An Origin check on POST, PUT, PATCH and DELETE.
- **Tests:** a request without the header gets 401. A request with `Host: evil.example` gets 400. The sandbox account cannot read the token file (extend `tests/integration/test_sandbox_windows.py`).
- **Existing sessions (Review Focus 4):** stored `session/credentials.json` stays valid; only the transport check is new.

### Task 6: Any user can register a server-side stdio MCP server — CRITICAL
- **Where:** `app/api/mcp.py:22` (router guarded only by `get_current_user`), `POST /servers` (`:65`), `POST /servers/from-url` (`:78`); commands launched at `app/ai/mcp_integration.py` (~`:193`).
- **Failure:** signup is open. A new user posts `{"command": "powershell", ...}` and it runs on the API host, shared across all tenants. Delete, toggle and execute are not scoped either.
- **Decision:** admin-only, or no stdio on the canonical server at all.
- **Recommended:** add `admin_user_ids: list[UUID]` to settings and a `require_admin` dependency on every mutating MCP route. Refuse `stdio` transport from the API unless `mcp_allow_api_stdio` is set; bundled servers still load from `mcp_config.json`. Needs a `.env.example` line (Thai edits).
- **Tests:** a non-admin POST gets 403. An admin stdio POST gets 403 unless the flag is set. A GET still masks env and header values (already covered by `tests/test_mcp_service_redaction.py`).

### Task 7: Deleted users keep working tokens
- **Where:** `app/core/auth.py` `get_current_user_id` (never loads the user); `app/api/auth.py` logout does not revoke anything.
- **Decision:** a per-request user lookup, or a token version.
- **Recommended:** add a `token_version` column to `users`. Put `ver` in access and refresh tokens and compare it in `get_current_user_id` through a small TTL cache. Soft delete and "log out everywhere" both bump it. This needs a migration (it can ride with Phase 3).
- **Existing tokens (Review Focus 4):** treat a missing `ver` as version 0 so issued tokens stay valid until they expire.

### Task 8: MCP tool recovery skips the agent's tool scope
- **Where:** `app/ai/tool_execution.py` `_recover_missing_tool` (~999–1011).
- **Failure:** with tool_search off, any MCP tool the model names is re-bound from the manager. That skips the agent allowlist and the custom agent's tool refs, and approval ran before recovery, when the tool had no metadata.
- **Fix:** filter recovered tools through the agent's binding filter, then re-run the approval policy on the recovered tool before executing it. No decision needed; it moves here because it is security.
- **Test:** a custom agent limited to tool A names tool B. B must be refused, not executed.

### Task 9: Widget cross-user access and session binding
- **Where:** `app/api/widgets.py` recovery (~289–320) and `widget_connect` (~607).
- **Failure:** recovery matches any of user B's tool outputs that contain A's widget id as a substring. `widget_connect` never compares the token's `sid` to the widget's stored `session_id`. So B, knowing A's widget UUID, can read and patch A's widget.
- **Fix:** only recover when the stored id is not a UUID, and compare `sid` to `record.session_id` on connect. No decision needed.

### Task 10: Credentials in skill files — HIGH (untracked files)
- **Where:** `skills/take100/SKILL.md` and `skills/attendance/SKILL.md` contain plaintext `Password:` lines, and SKILL.md is sent to the server and the LLM on activation. `skills/take100/take100_api.py:243` sets `verify=False`.
- **Fix:** move the passwords to skill secret bindings (`secrets:` front matter, read from env), rotate them, and remove `verify=False` or pin the site's CA. Thai's own files, so Thai makes the edit.

### Task 11: Production-safe defaults
- **Where:** `app/core/config.py`.
- **Failure:** `environment` defaults to `development`, so a production deploy that forgets `ENVIRONMENT` gets the dev key and debug. Outside dev there is no minimum `SECRET_KEY` length. `main.py:597` hard-codes `0.0.0.0:8000`, so `api_host`/`api_port` do nothing.
- **Recommended:** in `_cross_field_checks`, when `environment != "development"`, require `len(secret_key) >= 32` and a non-empty `cors_origins` with no `*`. Make `main.py`'s `__main__` read `settings.api_host`/`api_port`, defaulting to 127.0.0.1. Keep the `development` default, so local setup doesn't change.
- **Tests:** extend `tests/test_dev_secret_key.py`.

### Task 12: Smaller leaks
- `app/api/messages.py:47` `_stream_error_event` still sends `str(exc)` for exceptions that are not `CustomHTTPException`. Reuse `message_service._client_error_text`.
- Planning rubric errors put raw exception text into `planning_rubric` metadata (`graph.py` ~1603, `planning_agent.py:751`). Use the type name, and update `tests/test_planning_agent_rubric.py`, which pins the text.
- `base_agent._build_error_response` (~1908), `chat_agent:100` and `rag_agent:1417` store `str(e)` in `AgentResponse.error` and metadata. Fix once in `_build_error_response`.
- `workflow/finalization.py` (~395) builds `response_validation_failed` details without `sanitize_error_details`. Build them through `workflow_error()`.
- Health endpoints return raw `str(e)` and are unauthenticated. Return the exception class name.
- `repositories/hitl_interrupt.py:218` logs the new session id. Drop it.
- Sidecar: `operations.py:325` and `upstream_auth.py:94` put `str(exc)` into install receipts and logs. Use the type name.
- Sidecar `sandbox/launch.py` passes Desktop Commander env values on the command line. Pass them via stdin.
- Sidecar POSIX only: `.local_session_secret` is created with the default umask. Use `os.open(..., 0o600)`.
- `demo.py` (~2705–2745) keeps the token in localStorage and restores it through `?__t=` in the URL. Use sessionStorage or a component return value.
- Sidecar proxy path parameters accept `%2E%2E`. Reject `.` and `..`, or validate UUIDs.

---

## Phase 2 — Correctness (no schema change)

### Task 13: Custom-agent definitions leak on the shared `SpecialistFactory` — HIGH
- **Where:** `graph.py` (~1744) and `SpecialistFactory.register`.
- **Failure:** definitions are registered on the shared factory and never removed, and planning custom-agent workers never register one. A dispatched worker fails with `agent_unavailable`, or runs a stale definition (old tools, handoff targets, instructions). Memory grows with every custom agent used.
- **Fix:** resolve the definition per request (pass it explicitly, or through a callback) and stop registering on the shared factory. Architecture-level, so it needs its own plan.

### Task 14: Close singletons on shutdown
- **Where:** `app/main.py` lifespan.
- **Fix:** close `checkpoint_manager` (psycopg pool), `generation_control_bus` (Redis subscriber task) and `qdrant_client`. The sidecar equivalent is already fixed (`tests/client_backend/test_lifespan_shutdown.py` is the pattern).

### Task 15: RAG workers ignore their tool scope
- **Where:** `workflow/rag_execution.py`.
- **Fix:** `RagExecutionRequest.allowed_tool_ids` is never enforced. Filter like `WorkerToolScopeMiddleware` does.

### Task 16: Deferred tool state keyed by an empty conversation id
- **Where:** `deferred_tool_state._get_key`.
- **Fix:** a `None` conversation id becomes `""`, which every user shares. Refuse, or skip storing, when there is no conversation id.

### Task 17: Tool receipt completion is not owner-scoped
- **Where:** `repositories/tool_execution_receipt.py` `_atransition` (~155) and `services/tool_execution_receipt_service.py` (~207–217).
- **Fix:** add `user_id`/`conversation_id` to the transition WHERE. Update the two unit fakes and the postgres test calls. Practically unreachable (the key embeds the thread id), so low priority.

### Task 18: Device runtime accepts results for any request id
- **Where:** `client_runtime_store.publish_result` and `api/device_runtime.py:153`.
- **Fix:** compare the result against the request→device record before accepting it.

### Task 19: Remaining small correctness items
- Async routes call sync DB code, which blocks the event loop: `api/conversations.py:173`, `api/ai_sdk.py:412`, and others. Make them `def` routes, or use the async repositories.
- `api/documents.py:380`: a task id with no linked document returns its status to any user. Return 404.
- `widget_runtime.py:457`: the Redis `close` read-then-write has no WATCH, so a racing update can produce duplicate version numbers.
- `ai_service.resume_workflow` uses `thread_id=str(conversation_id)`, but threads are per turn, and no route calls it. Delete the whole non-stream resume chain (also `message_service.resume_workflow`, `MultiAgentWorkflow.resume`).
- `events.py` listeners are registered only in the API process, while `PROCESSING_*` events are emitted in Celery workers. Remove the listeners, or move them to the worker.
- Planning model-to-model loops are bounded only by the graph `recursion_limit`. Add a Planning model-call budget.
- Sidecar uploads: there is no size limit before Starlette spools the file, and `await file.read()` loads it all into memory. Add Content-Length middleware (`api/documents.py`, skill uploads).
- `take100_api.py` writes its session cookies inside the skill bundle, which changes the bundle hash, so the next run fails as `SKILL_RUNTIME_STALE`. Write them outside the bundle.
- `provider_service.key_preview` shows the last 4 characters of the ciphertext, not of the key.
- `mcp_service`: an HTTP server's `url` can carry an API key in its query string, and it is not masked.

---

## Phase 3 — Database migrations (one Alembic revision per task)

Each task first counts violating or affected rows in the app database (read-only), then
writes the revision, then runs `tests/test_alembic_full_chain_postgres.py` against
`chatbot_test`. After each one, add the matching `server_default`/`Index` to the model,
so `tests/test_repository_query_contracts.py::test_models_match_the_migrated_head_schema`
stays green.

### Task 20: `document_chunks` defaults and legacy columns
- `op.alter_column("document_chunks", c, server_default=sa.text("now()"))` for `created_at` and `updated_at`. The model has a Python default since `0a51a1e7`, so this is defence for raw SQL.
- `op.drop_column` for the legacy `page_number` and `content_preview`, after Grep confirms nothing reads them.

### Task 21: Indexes and constraints
- **Missing FK indexes:** `generations.assistant_message_id`, `model_usage_events.request_message_id`, `model_usage_events.document_id`, `project_custom_agents.custom_agent_id`, `user_memories.project_id`. Use `CREATE INDEX CONCURRENTLY` for `model_usage_events` (large table).
- **`chat_images` dedup race:** a partial unique index on `(user_id, sha256) WHERE deleted_at IS NULL`, plus `ON CONFLICT DO NOTHING` in the repository insert. Check for existing duplicates first.
- **Redundant single-column indexes** covered by a composite or unique index that starts with the same column: `messages.conversation_id`, `task_plans.conversation_id`, `generations.user_id`, `tool_execution_receipts.user_id` and `.conversation_id`, `agent_model_configs idx_…_user_id`, `client_devices.user_id`, `skill_settings.user_id`, `tool_approval_settings.user_id`, `document_parse_artifacts.document_id`, `document_chunks idx_…_document_id`, `model_usage_events.operation_id`, `custom_agents.owner_id`, `projects.owner_id`, `feedbacks idx_feedbacks_message_user`. Check `pg_stat_user_indexes` before dropping.
- **CHECK constraints:** `feedbacks.rating BETWEEN 1 AND 5`, and enumerations for `document_index_generations.status`, `document_chunks.index_status`, `web_image_references.lifecycle_state`. Add an FK for `hitl_interrupts.resolved_by_user_id`.

### Task 22: Project soft delete strands its memories
- **Where:** `repositories/project.py` (~155) `soft_delete_and_detach`.
- **Failure:** a soft delete never fires `SET NULL`, and conversations are detached, so that project's memories are unreachable. Migration `9778bb07ea35` says they should "fall back to global".
- **Decision:** move them to global (`UPDATE user_memories SET project_id = NULL WHERE project_id = :pid`), or soft-delete them with the project. The demo's delete confirmation text (commit `44403b25`) describes one of these, so check that text and match it.

### Task 23: ORM cascade loading
- `DocumentIndexGeneration.chunks` and the `Document.*` relationships cascade without `passive_deletes=True`, so purging a generation loads every chunk and then each chunk's images. Add `passive_deletes=True` where the database already cascades. SQLite tests then need `PRAGMA foreign_keys=ON`, which changes their behaviour, so check which ones.
- `Message.feedback` uses `lazy="joined"`, which adds a join to every message read. Switch to `selectin`, or load it explicitly.
- `get_with_messages` includes deleted messages. `MessageCRUDStrategy.get_by_user_id` does a join *and* an EXISTS, and includes soft-deleted conversations. `get_soft_deleted` is unbounded. `_lookup_sequence` is not scoped to the conversation.

---

## Phase 4 — Debloat

### Task 24: Delete five dead modules (the classifier blocked the assistant; Thai runs it)

```bash
git rm app/schemas/responses/error_response.py app/schemas/responses/success_response.py \
  app/core/exceptions/skills.py client_backend/schemas/mcp.py client_backend/schemas/messages.py
```

Then remove the matching imports and `__all__` entries from
`app/schemas/responses/__init__.py` (`ErrorResponse`, `SuccessResponse`) and
`app/core/exceptions/__init__.py` (`SkillNotFoundError`). Grep found no other references
(2026-09-29). Run ruff and the full suite.

### Task 25: Dead settings kept only because `.env.example` lists them
Thai removes each line from `.env.example`, then the field from `config.py`, then the
contract test entry:
`smithery_api_key`, `media_resolution`, `rerank_top_k`, `table_format`,
`memory_max_messages`, `memory_load_batch_size`, `react_agent_quality_threshold`,
`max_auto_plan_tasks`, `execution_call_budget`, `planning_consecutive_errors_limit`,
`planning_subagents_enabled` (no longer gates anything), `tool_validation_enabled`,
`confidence_threshold_abstain`, `enable_structured_output_validation`,
`confidence_weight_*`, `client_runtime_require_connected_device_for_local_tools`,
`rich_image_candidate_max_count`, `langsmith_project`, `enable_citation_verification`.
`api_host` and `api_port` are wired up in Task 11 instead of deleted.
Test-only: `workflow_graph_version`, `web_search_max_results`,
`web_search_result_max_chars`, `web_open_max_*`. README-only:
`rich_image_group_max_items`, `rich_image_anchor_min_score`.

### Task 26: Code referenced only by tests
Delete each item together with its test, one commit per group:
- **graph.py:** `_finalize_agent_response`, `_interrupt_payload_from_pending_interrupts`, `_conversation_has_documents`, `_aconversation_has_documents`, `_get_agent_type`.
- **tool_loop.py:** `_apply_hand_off_if_present`, `_needs_approval`, `_prepare_interrupt_payload`, `_execute_agent_tool_calls`, `_apply_tool_outputs_to_state`. The production gate is the middleware.
- **Routing:** `agents/router.py` `Router` (5 test files), `RoutingContextBuilder.build_messages`, `inventory.routable_ids`.
- **Other orchestration:** `hitl_config.requires_human_approval`, `execution_budget.begin_epoch`, `custom_agents._route_target_for`, `IMAGE_GENERATOR_SYSTEM_PROMPT`. `rag_agent.regenerate_grounded_answer` is asserted exactly once by `test_streamable_grounding`, so update that test.
- **Services:** `GenerationRegistry.find_by_user`, `WidgetConnectionManager.connection_count`, `WidgetRecord.to_live_widget_metadata`, `rag_grounding.evidence_pack_from_payloads`, `_excel_rows_to_markdown`.
- **Repositories:** Message `get_latest_by_conversation`/`a…`, `acount_by_conversation_id`; Document `get_routing_descriptors`, `filename_exists_in_conversation`; Chunk `mark_indexed`, `get_by_ids`, `get_by_qdrant_point_ids`; `DocumentImage.get_by_document_id`; DocumentIndexGeneration `get_latest_failed`, `list_for_document`, `retired_before`; `DocumentParseArtifact.list_by_document`; `Generation.aget_active_for_conversation`; `Receipt.alist_unresolved`; `ModelUsage.get_minute_series`; `asearch`; `Conversation.aupdate`; `UserMemory.aresolve_project_id`; `WebImage.aget_pending_for_user`; `get_async_engine`/`get_async_session_factory`; and now also `FeedbackRepository.get_by_user_id`, `ConversationRepository.get_with_messages`.
- **Core:** `ConversationFactory.create_from_dict`, `AppContainerInjector.wiring_map`, `assess_corpus`, `_mark_needs_reindex`. The `rich_images` counters `record_discovery`, `record_candidate`, `record_anchor` and `record_discovery_outcome` always read zero: wire them up or delete them.
- **API:** `DeviceRuntimeGateway.dispatch_tool_call`.
- **Sidecar:** `persist_for_test`, `staging_dir_for_test`, `recover_install_transactions`, `is_single_skill`, `retry_pending_sync`, `close_skill_catalog_service`, `SkillMetadata.to_dict`, `LIFECYCLE_AUDIT_FIELDS`, upload_support `render_upload_section`/`upload_document`, server_api `stream_message`/`upload_documents_bytes`.
- **Kept on purpose (plans/docs reference them):** `get_bot_response_sync`, `MessageService.update_message`/`delete_message`, `RAGEmbeddingService`, `build_image_preview_inline_data`, `ProjectValidationError`, `ConversationInDB`.

### Task 27: Unused dependencies
No imports anywhere for `langchain-text-splitters`, `langchain-tavily`, `nltk`, `pypdf`,
`python-docx`, `docx2txt`, `pdfplumber`, `tabulate` or `fastapi-radar`. Frozen manifests
and bundle tests reference them, so remove each from `pyproject.toml`,
`requirements.txt` and `environment.yml` together and update those tests.

### Task 28: Python floor and ruff target
The app needs Python ≥3.11 (`asyncio.timeout` in `conversation_compactor.py:272`), but
the wheel also ships `client_backend`, whose bundle documents 3.10+. Decide the floor.
Raising ruff's `target-version` produces 338 findings (py311) or 347 (py312). Do that as
its own mechanical commit with `ruff check --fix`, reviewed separately.

---

## Phase 5 — Complexity (refactor with tests green before and after)

Top candidates by audit area. Split one function per task, and never mix a split with behaviour changes.

| Function | Complexity | Split into |
|---|---|---|
| `base_agent.invoke_model_with_history` | 45, 165 statements | model resolution, budget preflight, the invoke/retry loop, response metadata |
| `message_service._create_message_stream_holding_turn` | 34, 154 statements | per-event handlers (delta, tool end, interrupt, complete), the cancel/partial-persist path, the error path |
| `tool_execution_policy.resolve_tool_execution_policy` | 34 | trusted metadata, deployment rules, safety caps, client-deadline checks |
| `message_service._validate_and_claim_interrupt_resume` | 27 | durable-record checks, device/tool-instance checks, the legacy Redis expiry check |
| `base_agent._get_tools_for_binding` | 27 | deferred vs full binding, client tools, internal/handoff merge |
| `config._cross_field_checks` | 25 | one validator per domain (generation, usage, tools, Redis, security, summary, production) |
| `message_service.resume_message_creation_stream` | 23 | the handler split plus one shared "fail, persist, yield error" helper |
| `rag_agent._invoke_agentic_rag_model` | 22 | vision fallback, usage recording, response parsing |
| `ui/subagent_activity.build_subagent_activity_view` | 22 | a dispatch table of per-event-kind handlers |
| sidecar `environment.prepare` | 19 | build commands, run setup, write metadata, promote stage |
| `widgets._extract_widget_snapshot_from_metadata` | 18 | find the live widget, fold artifacts |
| `tool_execution.execute_tool_calls` | 16 | a list of per-call checks, each returning an error payload or None |
| `widgets.widget_connect` | 14, 61 statements | `_handle_user_state_patch` |
| `conversation_compaction.persist_memory_cas` | 78 lines, 14 args | a `MemoryWrite` dataclass; split the insert branch from the CAS branch |
| `model_usage._reconcile_minute_chunk` | 96 lines | the aggregate-statement builder, the advisory-lock step |

State-schema cleanup (from AI orchestration). These keys are read but never written:
`force_final_response`, `tool_budget`, `continuation_signal`. `planning_call_count` is
never updated but still published (always 0). These are written but never read:
`pending_action_requests`, `interrupt_metadata`, `tool_error_streak`,
`WorkerTask.parent_context`. Wire each one up or delete it, one commit per key.

---

## Appendix — where each finding came from

| Area | Fix commit | Reported items in this plan |
|---|---|---|
| DB | `0a51a1e7` | Tasks 2, 17, 20–23, 26 (repositories) |
| API/security | `93268faa` | Tasks 6, 7, 9, 12, 19, 26 (API), Phase 5 widgets |
| Services | `94c12081` | Tasks 7, 12, 17–19, 26 (services), Phase 5 message_service |
| AI orchestration | `694ad0b1` | Tasks 8, 12, 13, 15, 16, 19, 26 (orchestration), Phase 5 state keys |
| AI tools | `b0a65dff` | none: every finding was fixed |
| Core/infra | `16aee360` | Tasks 3, 4, 11, 14, 19, 25, 27, 28 |
| Client sidecar | `0fdb9804` | Tasks 5, 10, 12, 19, 26 (sidecar) |
| Codex Projects work | `44403b25` | Task 1 |
