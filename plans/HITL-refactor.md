# Granular Per-User HITL Approval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single global, static `hitl_tools_require_approval` name-list with a per-user, runtime-managed approval policy that can gate **a specific tool** or **all tools from an MCP server** (tool-overrides-server precedence), wired through every HITL gate (chat, custom-agent, RAG, planning) and surfaced as toggles in the demo's MCP panel via the sidecar.

**Architecture:** Five workstreams.
- **(A) Policy core** — a checkpoint-safe policy dict + a provenance resolver in `app/ai/hitl_config.py`. The resolver derives `(name, server_name, qualified_tool_id, origin)` for each pending tool call (client tools from `client__<server>__<tool>` name + metadata; server tools from metadata first, then `McpManager.get_server_for_tool` fallback), then applies the precedence ladder. The existing `requires_human_approval(names)` stays as a back-compat shim.
- **(B) Persistence + API** — `ToolApprovalSetting` model (per-user, scope = server|tool), a session-factory repository, a thin `HitlSettingsService`, and an `app/api/hitl.py` router. Mirrors the `SkillSetting`/`CustomAgent`/`MCP` patterns exactly.
- **(C) Gate wiring** — one shared async `MultiAgentWorkflow._needs_approval(...)` helper that all 5 gate sites call; per-turn policy loaded by `MessageService` and carried into graph `context` exactly like `inline_rich_response_v1`.
- **(D) Sidecar + demo UX** — additive `/hitl/settings` proxy routes in `client_backend/api/proxy.py`; per-server toggle + per-tool tri-state in the demo MCP panel; the global master shown read-only.
- **(E) Docs + verification** — README, full suite, lint, manual smoke (incl. the explicit "HITL works with client sidecar + deferred tools" requirement).

**Tech Stack:** FastAPI server (`app/`), FastAPI sidecar (`client_backend/`), Streamlit (`demo.py`), LangChain `@tool` + LangGraph, SQLAlchemy + Alembic (PostgreSQL), pytest.

**Environment:** Run tests with `.conda\python.exe -m pytest` (Python 3.14, pytest 9.0.2 — the `.venv` runtime env has no pytest). Always exclude `tests/client_backend/test_live_server_integration.py` (needs a live server on :8000). API tests (`TestClient`) need a reachable Postgres at `settings.database_url`; if unavailable, run them against the dev DB or skip with a documented reason — never weaken assertions to get green. Run commands from the repo root.

## Global Constraints

- **Per-user scope** — the approval policy is keyed by `user_id` (it follows the user across any sidecar/UI). Persisted in the server DB. The gate reads it server-side.
- **Precedence (verbatim)** — for each pending call, after resolving `server_name` + `qualified_tool_id`:
  1. `master_enabled == False` → **NOT** gated (global kill-switch, unchanged).
  2. tool rule for `qualified_tool_id` exists → its bool (override wins).
  3. tool rule for bare `name` exists → its bool.
  4. server rule for `server_name` exists → its bool.
  5. bare `name ∈ global_tools` (legacy `hitl_tools_require_approval`) → gated.
  6. otherwise → **NOT** gated.
- **Back-compat** — keep `enable_human_in_the_loop` (master), keep `hitl_tools_require_approval` (now the global default floor at step 5), keep decision types `accept/edit/reject/respond`, keep `/resume-interrupt`. Existing `tests/test_hitl_config.py` and `tests/test_client_tool_isolation.py::test_client_tools_follow_explicit_hitl_allowlist` MUST keep passing unchanged.
- **Checkpoint-safety** — the policy lives in graph `context` as a plain JSON-serializable dict (`{"master_enabled": bool, "servers": {str: bool}, "tools": {str: bool}, "global_tools": [str]}`). No dataclasses/objects in graph state.
- **Additive sidecar contract** — new sidecar routes are additive only; do not change existing route signatures (FR-5 of `plans/client_toolset_isolation.md`).
- **Naming (verbatim):** client tool name = `client__<server>__<tool>`; `qualified_tool_id` = `"<server>::<tool>"`; client-tool prefix constant `CLIENT_TOOL_PREFIX = "client__"`.
- **Do not touch** `client_backend/services/local_skills_registry.py`, `shared/skills/front_matter.py`, or the `client_runtime_tools`/`deferred_tool_state` v1 isolation internals beyond reading them.

**Key research facts (verified — trust these):**

- HITL gate call sites in `app/ai/graph.py`: **five** — `_should_call_tools` (1209, sync conditional edge shared by `chat_agent`/`search_agent`/`image_generator_agent`/`canvas_agent`/`custom_agent`, wired at 768-776), RAG (2453, calls `interrupt()`), RAG sub-worker (2892, sets `response.metadata["requires_approval"]`), generic worker loop (2996, same metadata flag), planning (3297, calls `interrupt()`).
- `requires_human_approval(tool_names)` (`app/ai/hitl_config.py:22-40`) is **exact-name membership** against `settings.hitl_tools_require_approval`. No server/wildcard logic.
- Provenance: **client tools** carry `server_name`, `qualified_tool_id`, `tool_origin` in `.metadata` (`app/ai/client_runtime_tools.py:299-323`). **Server MCP tools currently also carry metadata** after `clone_mcp_tool(..., server_name=...)` (`app/core/mcp_adapter_utils.py:230-245`), but the resolver must still fall back to `McpManager.get_server_for_tool(tool: BaseTool) -> str | None` (`app/ai/mcp_integration.py:307-309`) for older/raw/fake tools. The server-side global manager is obtained with `await get_global_mcp_manager()` from `app.ai.mcp_registry` (`app/ai/mcp_registry.py:227`), not from `app.ai.tool_execution`. `_prepare_interrupt_payload` (`app/ai/graph.py:1073-1147`) already extracts those metadata fields.
- `ensure_agent_tool_map(agent, conversation_id=, user_id=, device_id=)` (`app/ai/tool_execution.py:568`) is cached per conversation/agent — safe to call in the router.
- Persistence patterns: model → `app/models/skill_setting.py`; **wired** session-factory repo → `app/repositories/custom_agent.py` (sync `with self.session_factory() as session:`); container provider → `app/core/container.py:195` (repo) and `:384` (service); `MessageService` already receives `tool_approval_repository` (`app/core/container.py:412`); DI service wiring → `app/core/dependency_injection.py:121-135` (`AppAutoInjector.wiring_map`).
- API: routers use `@AppAutoInjector.auto_inject()` with `user_id: UUID` auto-injected and `service: SomeService` auto-injected (no `Depends`), return `ApiResponse[T]`; registered in `app/main.py:226-245`. `get_current_user_id` → `UUID` (`app/core/auth.py:36`).
- Alembic: current head revision = `f03e63aa5a33` (`app/alembic/versions/f03e63aa5a33_add_content_to_tool_result_blobs.py`), verified with `.conda\python.exe -m alembic heads`; the new migration must revise that head. Revision ids are 12-char pseudo-hex; `skill_settings` create_table is the template (`j1k2l3m4n5o6_...py:187-213`); register the model in `app/models/__init__.py` (import + `__all__`).
- Per-turn injection precedent: `inline_rich_response_v1` rides the dual `WorkflowExecutionRequest` schemas into graph `context` at `app/ai/graph.py:512`. Mirror it for `hitl_policy`. (See `[[dual-workflow-request-schema-drift]]`: `_to_ai_request` silently drops fields missing from the AI-layer schema — the new field MUST be added to **both** schemas and verified through `_to_ai_request`.)
- Sidecar: `proxy_server_request(request, upstream_path=...)` (`client_backend/api/common.py`) forwards with the upstream token automatically; routes guarded by `Depends(require_local_session)`; registered (unprefixed + `/api`) in `client_backend/main.py:100-105`. HITL is per-user → **no** device stamping.
- Demo: `make_api_request(method, endpoint, data=None)` (`demo.py:2816`) returns the `{success, message, data}` envelope and attaches the Bearer token. MCP panel: Configured Servers loop `demo.py:7876-7919` (`st.columns([3,1,1])`), Tools section `demo.py:7922-7978`; helpers `get_mcp_servers`/`get_mcp_tools`/`toggle_mcp_server` at `demo.py:3335-3347,3672-3676`.
- Test harnesses: repo unit tests use a hand-rolled `_FakeSession`/`_FakeQuery` + `@contextmanager` factory, **no real DB** (`tests/test_document_parse_artifact_repository.py:20-70`); API tests use `Database(settings.database_url)`, seed a `User`, override `get_current_user_id`, and drive a `TestClient` (`tests/test_custom_agents_api.py:37-70`).

---

### Task 1: Policy core — resolver, precedence, and back-compat shim

**Files:**
- Modify: `app/ai/hitl_config.py` (add dataclass + functions; keep `requires_human_approval`)
- Test: `tests/test_hitl_policy.py` (new)

**Interfaces:**
- Produces:
  - `CallIdentity` (frozen dataclass): `name: str`, `server_name: str | None`, `qualified_tool_id: str | None`, `origin: str`
  - `resolve_call_identity(tool_call: dict, *, tool_map: dict | None = None, mcp_manager=None) -> CallIdentity`
  - `identity_requires_approval(identity: CallIdentity, policy: dict) -> bool`
  - `any_call_requires_approval(tool_calls: list[dict], *, policy: dict, tool_map: dict | None = None, mcp_manager=None) -> bool`
  - `build_global_policy() -> dict` and `policy_from_context(context: dict | None) -> dict`
  - unchanged: `requires_human_approval(tool_names: list[str]) -> bool`
- Consumes: `app/core/config.settings`. Do not import `app.ai.utils` here; gate sites pass normalized dicts and `hitl_config` must stay dependency-light.

- [x] **Step 1: Write the failing tests — `tests/test_hitl_policy.py`**

```python
"""Per-user HITL policy resolution and precedence (tool-overrides-server)."""

from types import SimpleNamespace

import pytest

from app.ai.hitl_config import (
    any_call_requires_approval,
    build_global_policy,
    identity_requires_approval,
    policy_from_context,
    resolve_call_identity,
)


def _tool(name, **metadata):
    return SimpleNamespace(name=name, metadata=dict(metadata))


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping  # id(tool) -> server

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def _policy(master=True, servers=None, tools=None, global_tools=None):
    return {
        "master_enabled": master,
        "servers": servers or {},
        "tools": tools or {},
        "global_tools": global_tools or [],
    }


def test_resolve_identity_for_client_tool_uses_name_and_metadata():
    tool = _tool(
        "client__desktop_commander__start_process",
        server_name="desktop_commander",
        qualified_tool_id="desktop_commander::start_process",
        tool_origin="client_mcp",
    )
    identity = resolve_call_identity(
        {"name": "client__desktop_commander__start_process"},
        tool_map={"client__desktop_commander__start_process": tool},
    )
    assert identity.server_name == "desktop_commander"
    assert identity.qualified_tool_id == "desktop_commander::start_process"
    assert identity.origin == "client_mcp"


def test_resolve_identity_for_client_tool_parses_name_without_metadata():
    identity = resolve_call_identity({"name": "client__excel__write_cell"})
    assert identity.server_name == "excel"
    assert identity.qualified_tool_id == "excel::write_cell"
    assert identity.origin == "client_mcp"


def test_resolve_identity_for_server_tool_uses_manager():
    tool = _tool("search")  # bare server-tool name, no metadata
    manager = _FakeManager({id(tool): "tavily"})
    identity = resolve_call_identity(
        {"name": "search"}, tool_map={"search": tool}, mcp_manager=manager
    )
    assert identity.server_name == "tavily"
    assert identity.qualified_tool_id == "tavily::search"
    assert identity.origin == "server_mcp"


def test_resolve_identity_for_server_tool_prefers_metadata():
    tool = _tool(
        "calculate",
        server_name="calculator",
        qualified_tool_id="calculator::calculate",
        tool_origin="server_mcp",
    )
    identity = resolve_call_identity({"name": "calculate"}, tool_map={"calculate": tool})
    assert identity.server_name == "calculator"
    assert identity.qualified_tool_id == "calculator::calculate"
    assert identity.origin == "server_mcp"


def test_resolve_identity_for_aliased_deferred_server_tool_without_manager_id_hit():
    tool = _tool(
        "brave__search",
        tool_origin="server_mcp",
        aliased_from_tool_name="search",
        call_name="brave__search",
    )
    identity = resolve_call_identity({"name": "brave__search"}, tool_map={"brave__search": tool})
    assert identity.server_name == "brave"
    assert identity.qualified_tool_id == "brave::search"
    assert identity.origin == "server_mcp"


def test_precedence_tool_qualified_overrides_server():
    policy = _policy(
        servers={"desktop_commander": True},
        tools={"desktop_commander::list_files": False},
    )
    gated = SimpleNamespace(
        name="client__desktop_commander__run", server_name="desktop_commander",
        qualified_tool_id="desktop_commander::run", origin="client_mcp",
    )
    exempt = SimpleNamespace(
        name="client__desktop_commander__list_files", server_name="desktop_commander",
        qualified_tool_id="desktop_commander::list_files", origin="client_mcp",
    )
    assert identity_requires_approval(gated, policy) is True   # inherits server ON
    assert identity_requires_approval(exempt, policy) is False  # tool override SKIP


def test_precedence_tool_can_force_on_when_server_off():
    policy = _policy(servers={"excel": False}, tools={"excel::delete_sheet": True})
    ident = SimpleNamespace(
        name="client__excel__delete_sheet", server_name="excel",
        qualified_tool_id="excel::delete_sheet", origin="client_mcp",
    )
    assert identity_requires_approval(ident, policy) is True


def test_precedence_master_off_disables_everything():
    policy = _policy(master=False, servers={"excel": True})
    ident = SimpleNamespace(
        name="client__excel__x", server_name="excel",
        qualified_tool_id="excel::x", origin="client_mcp",
    )
    assert identity_requires_approval(ident, policy) is False


def test_precedence_legacy_global_floor_still_gates():
    policy = _policy(global_tools=["dangerous_tool"])
    ident = SimpleNamespace(
        name="dangerous_tool", server_name=None, qualified_tool_id=None, origin="internal",
    )
    assert identity_requires_approval(ident, policy) is True


def test_any_call_requires_approval_short_circuits_on_first_gated():
    policy = _policy(servers={"desktop_commander": True})
    calls = [
        {"name": "client__time_server__now"},
        {"name": "client__desktop_commander__run"},
    ]
    assert any_call_requires_approval(calls, policy=policy) is True


def test_policy_from_context_falls_back_to_global(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "enable_human_in_the_loop", True)
    monkeypatch.setattr(settings, "hitl_tools_require_approval", ["legacy_tool"])
    policy = policy_from_context(None)
    assert policy == build_global_policy()
    assert policy["global_tools"] == ["legacy_tool"]
    assert policy["master_enabled"] is True
```

- [x] **Step 2: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/test_hitl_policy.py -q`
Expected: FAIL — `ImportError`/`AttributeError` (none of the new symbols exist yet).

- [x] **Step 3: Implement in `app/ai/hitl_config.py`**

Add these at the top (after the existing imports). Do **not** import from `app.ai.utils` here — `hitl_config` must stay dependency-light to avoid a circular import (`utils` references HITL decision logic). A tiny inline name reader is enough since gate sites already pass normalized dicts:

```python
from dataclasses import dataclass

CLIENT_TOOL_PREFIX = "client__"


def _tool_call_name(tool_call) -> str:
    """Best-effort tool name from a normalized dict or a tool-call object."""
    if isinstance(tool_call, dict):
        return tool_call.get("name") or ""
    return getattr(tool_call, "name", "") or ""
```

Add the new public API (place it directly after `requires_human_approval`, keeping that function unchanged):

```python
@dataclass(frozen=True)
class CallIdentity:
    """Resolved provenance for a single pending tool call."""

    name: str
    server_name: str | None
    qualified_tool_id: str | None
    origin: str  # "client_mcp" | "server_mcp" | "internal"


def build_global_policy() -> dict:
    """Back-compat policy derived only from process settings (no per-user rules)."""
    return {
        "master_enabled": is_hitl_enabled(),
        "servers": {},
        "tools": {},
        "global_tools": list(get_tools_requiring_approval()),
    }


def policy_from_context(context: dict | None) -> dict:
    """Return the per-turn policy stashed in graph context, or the global fallback."""
    if isinstance(context, dict):
        policy = context.get("hitl_policy")
        if isinstance(policy, dict):
            return policy
    return build_global_policy()


def resolve_call_identity(tool_call, *, tool_map: dict | None = None, mcp_manager=None) -> CallIdentity:
    """Derive (name, server_name, qualified_tool_id, origin) for one tool call.

    Client tools resolve from their ``client__<server>__<tool>`` name (and metadata
    when bound); server tools resolve their server via the MCP manager.
    """
    name = _tool_call_name(tool_call)

    tool = tool_map.get(name) if (tool_map and name) else None
    meta = getattr(tool, "metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    server_name = meta.get("server_name")
    qualified = meta.get("qualified_tool_id")
    origin = meta.get("tool_origin")

    if name.startswith(CLIENT_TOOL_PREFIX):
        remainder = name[len(CLIENT_TOOL_PREFIX):]
        parsed_server, _, base = remainder.partition("__")
        server_name = server_name or (parsed_server or None)
        if not qualified and server_name and base:
            qualified = f"{server_name}::{base}"
        origin = origin or "client_mcp"
    else:
        # Server MCP tools usually carry these metadata fields after clone_mcp_tool(),
        # but fake/raw tools and older objects may not. Ambiguous deferred server
        # tools can be copied under a public alias such as "brave__search"; when
        # that alias metadata is present, recover the server from the alias prefix
        # and the raw tool name from "aliased_from_tool_name".
        aliased_from = meta.get("aliased_from_tool_name")
        if server_name is None and aliased_from and "__" in name:
            parsed_server, _, _ = name.partition("__")
            server_name = parsed_server or None
        if server_name is None and tool is not None and mcp_manager is not None:
            server_name = mcp_manager.get_server_for_tool(tool)
        base_tool_name = aliased_from or name
        if not qualified and server_name and base_tool_name:
            qualified = f"{server_name}::{base_tool_name}"
        origin = origin or ("server_mcp" if server_name else "internal")

    return CallIdentity(
        name=name, server_name=server_name, qualified_tool_id=qualified, origin=origin
    )


def identity_requires_approval(identity: CallIdentity, policy: dict) -> bool:
    """Apply the precedence ladder (tool override > server default > legacy floor)."""
    if not policy.get("master_enabled", True):
        return False

    tools = policy.get("tools") or {}
    if identity.qualified_tool_id and identity.qualified_tool_id in tools:
        return bool(tools[identity.qualified_tool_id])
    if identity.name in tools:
        return bool(tools[identity.name])

    servers = policy.get("servers") or {}
    if identity.server_name and identity.server_name in servers:
        return bool(servers[identity.server_name])

    if identity.name and identity.name in set(policy.get("global_tools") or []):
        return True

    return False


def any_call_requires_approval(
    tool_calls, *, policy: dict, tool_map: dict | None = None, mcp_manager=None
) -> bool:
    """True if any pending call requires approval under the given policy."""
    if not policy.get("master_enabled", True):
        return False
    for tool_call in tool_calls or []:
        identity = resolve_call_identity(tool_call, tool_map=tool_map, mcp_manager=mcp_manager)
        if identity_requires_approval(identity, policy):
            return True
    return False
```

- [x] **Step 4: Run the new tests + the existing HITL guards**

Run: `.conda\python.exe -m pytest tests/test_hitl_policy.py tests/test_hitl_config.py tests/test_client_tool_isolation.py -q`
Expected: PASS. (The existing `test_hitl_config.py` / `test_client_tool_isolation.py` exercise the unchanged `requires_human_approval` shim and must stay green.)

- [x] **Step 5: Commit**

```powershell
git add app/ai/hitl_config.py tests/test_hitl_policy.py
git commit -m "feat(hitl): policy resolver + tool-overrides-server precedence (core)"
```

---

### Task 2: `ToolApprovalSetting` model + migration + registration

**Files:**
- Create: `app/models/tool_approval_setting.py`
- Modify: `app/models/__init__.py` (import + `__all__`)
- Create: `app/alembic/versions/g0h1i2j3k4l5_add_tool_approval_settings.py`
- Test: `tests/test_tool_approval_setting_model.py` (new)

**Interfaces:**
- Produces: `ToolApprovalSetting` ORM model — columns `id, created_at, updated_at, user_id, scope_type, scope_value, require_approval`; table `tool_approval_settings`; unique `(user_id, scope_type, scope_value)`.

- [x] **Step 1: Write the failing test — `tests/test_tool_approval_setting_model.py`**

```python
"""Structural guard for the ToolApprovalSetting model + migration registration."""

from pathlib import Path

import app.models as models
from app.models.tool_approval_setting import ToolApprovalSetting


def test_model_columns_and_table():
    assert ToolApprovalSetting.__tablename__ == "tool_approval_settings"
    cols = set(ToolApprovalSetting.__table__.columns.keys())
    assert {"id", "created_at", "updated_at", "user_id", "scope_type", "scope_value",
            "require_approval"} <= cols
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
    assert any(
        "scope_type" in check and "server" in check and "tool" in check
        for check in checks
    )


def test_model_exported():
    assert "ToolApprovalSetting" in models.__all__
    assert models.ToolApprovalSetting is ToolApprovalSetting


def test_migration_chains_from_current_head():
    text = Path(
        "app/alembic/versions/g0h1i2j3k4l5_add_tool_approval_settings.py"
    ).read_text(encoding="utf-8")
    assert 'revision: str = "g0h1i2j3k4l5"' in text
    assert 'down_revision: str | None = "f03e63aa5a33"' in text
    assert "tool_approval_settings" in text
```

- [x] **Step 2: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/test_tool_approval_setting_model.py -q`
Expected: FAIL — module/file does not exist.

- [x] **Step 3: Create `app/models/tool_approval_setting.py`**

```python
"""Per-user Human-in-the-Loop approval setting (server-scoped or tool-scoped)."""

import uuid

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ToolApprovalSetting(Base):
    """
    A per-user rule that gates tool calls behind human approval.

    ``scope_type`` is "server" (``scope_value`` = MCP server name, applies to all
    its tools) or "tool" (``scope_value`` = qualified tool id ``"<server>::<tool>"``
    or a bare tool name, overrides the server default). ``require_approval`` is the
    explicit decision for that scope.
    """

    __tablename__ = "tool_approval_settings"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "scope_type", "scope_value", name="uq_tool_approval_settings_user_scope"
        ),
        CheckConstraint(
            "scope_type IN ('server', 'tool')",
            name="ck_tool_approval_settings_scope_type",
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    scope_type = Column(String(16), nullable=False)  # "server" | "tool"
    scope_value = Column(String(512), nullable=False, index=True)
    require_approval = Column(Boolean, nullable=False, default=True)

    user = relationship("User", backref="tool_approval_settings")

    def __repr__(self) -> str:
        return (
            f"<ToolApprovalSetting(user_id={self.user_id}, scope_type='{self.scope_type}', "
            f"scope_value='{self.scope_value}', require_approval={self.require_approval})>"
        )
```

- [x] **Step 4: Register in `app/models/__init__.py`**

Add the import next to the other model imports (after the `SkillSetting` import):

```python
from app.models.tool_approval_setting import ToolApprovalSetting
```

Add to the `__all__` list (after `"SkillSetting",`):

```python
    "ToolApprovalSetting",
```

- [x] **Step 5: Create the migration `app/alembic/versions/g0h1i2j3k4l5_add_tool_approval_settings.py`**

```python
"""add tool_approval_settings (per-user HITL approval policy)

Revision ID: g0h1i2j3k4l5
Revises: f03e63aa5a33
Create Date: 2026-06-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "g0h1i2j3k4l5"
down_revision: str | None = "f03e63aa5a33"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "tool_approval_settings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope_type", sa.String(length=16), nullable=False),
        sa.Column("scope_value", sa.String(length=512), nullable=False),
        sa.Column(
            "require_approval", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id", "scope_type", "scope_value", name="uq_tool_approval_settings_user_scope"
        ),
        sa.CheckConstraint(
            "scope_type IN ('server', 'tool')",
            name="ck_tool_approval_settings_scope_type",
        ),
    )
    op.create_index(
        op.f("ix_tool_approval_settings_id"), "tool_approval_settings", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_tool_approval_settings_user_id"),
        "tool_approval_settings", ["user_id"], unique=False,
    )
    op.create_index(
        op.f("ix_tool_approval_settings_scope_value"),
        "tool_approval_settings", ["scope_value"], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_tool_approval_settings_scope_value"), table_name="tool_approval_settings")
    op.drop_index(op.f("ix_tool_approval_settings_user_id"), table_name="tool_approval_settings")
    op.drop_index(op.f("ix_tool_approval_settings_id"), table_name="tool_approval_settings")
    op.drop_table("tool_approval_settings")
```

- [x] **Step 6: Verify model import + tests; confirm single migration head**

Run: `.conda\python.exe -m pytest tests/test_tool_approval_setting_model.py -q` → PASS
Run: `.conda\python.exe -c "import app.models; import app.main"` → no ImportError
Run: `.conda\python.exe -m alembic heads` → exactly one head: `g0h1i2j3k4l5 (head)`, confirming we extend the current head and create no second head.

- [x] **Step 7: Commit**

```powershell
git add app/models/tool_approval_setting.py app/models/__init__.py app/alembic/versions/g0h1i2j3k4l5_add_tool_approval_settings.py tests/test_tool_approval_setting_model.py
git commit -m "feat(hitl): ToolApprovalSetting model + migration"
```

---

### Task 3: `ToolApprovalSettingRepository` (session-factory, sync)

**Files:**
- Create: `app/repositories/tool_approval_setting.py`
- Test: `tests/test_tool_approval_setting_repository.py` (new)

**Interfaces:**
- Consumes: `ToolApprovalSetting` (Task 2).
- Produces: `ToolApprovalSettingRepository(session_factory)` with:
  - `list_by_user(user_id: UUID) -> list[ToolApprovalSetting]`
  - `set(user_id: UUID, scope_type: str, scope_value: str, require_approval: bool) -> ToolApprovalSetting`
  - `bulk_set(user_id: UUID, items: list[dict]) -> list[ToolApprovalSetting]` (each `{scope_type, scope_value, require_approval}`)
  - `delete(user_id: UUID, scope_type: str, scope_value: str) -> bool`
  - `build_policy(user_id: UUID) -> dict` → `{"servers": {str: bool}, "tools": {str: bool}}`

- [x] **Step 1: Write the failing test — `tests/test_tool_approval_setting_repository.py`**

```python
"""ToolApprovalSettingRepository uses the sync session-factory style (no real DB)."""

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

from app.repositories.tool_approval_setting import ToolApprovalSettingRepository


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.added = []
        self.deleted = []
        self.committed = 0

    def execute(self, _stmt):
        rows = self._rows

        class _Result:
            def scalars(self_inner):
                return SimpleNamespace(all=lambda: list(rows))

            def scalar_one_or_none(self_inner):
                return rows[0] if rows else None

        return _Result()

    def add(self, obj):
        self.added.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)

    def commit(self):
        self.committed += 1

    def refresh(self, _obj):
        pass

    def expunge(self, _obj):
        pass


def _factory(session):
    @contextmanager
    def _cm():
        yield session

    return _cm


def test_set_creates_when_missing():
    session = _FakeSession(rows=[])
    repo = ToolApprovalSettingRepository(_factory(session))
    user_id = uuid4()

    setting = repo.set(user_id, "server", "desktop_commander", True)

    assert session.added and session.committed >= 1
    assert setting.scope_type == "server"
    assert setting.scope_value == "desktop_commander"
    assert setting.require_approval is True


def test_set_updates_when_present():
    existing = SimpleNamespace(
        scope_type="tool", scope_value="excel::delete_sheet", require_approval=False
    )
    session = _FakeSession(rows=[existing])
    repo = ToolApprovalSettingRepository(_factory(session))

    setting = repo.set(uuid4(), "tool", "excel::delete_sheet", True)

    assert setting is existing
    assert setting.require_approval is True
    assert not session.added  # updated in place, not inserted


def test_build_policy_groups_by_scope():
    rows = [
        SimpleNamespace(scope_type="server", scope_value="desktop_commander", require_approval=True),
        SimpleNamespace(scope_type="tool", scope_value="desktop_commander::list_files", require_approval=False),
        SimpleNamespace(scope_type="garbage", scope_value="ignored", require_approval=True),
    ]
    repo = ToolApprovalSettingRepository(_factory(_FakeSession(rows=rows)))

    policy = repo.build_policy(uuid4())

    assert policy == {
        "servers": {"desktop_commander": True},
        "tools": {"desktop_commander::list_files": False},
    }


def test_rejects_invalid_scope_type():
    repo = ToolApprovalSettingRepository(_factory(_FakeSession(rows=[])))
    try:
        repo.set(uuid4(), "garbage", "ignored", True)
    except ValueError as exc:
        assert "scope_type" in str(exc)
    else:
        raise AssertionError("invalid scope_type should fail")
```

- [x] **Step 2: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/test_tool_approval_setting_repository.py -q`
Expected: FAIL — module does not exist.

- [x] **Step 3: Implement `app/repositories/tool_approval_setting.py`**

```python
"""Session-factory backed repository for per-user HITL approval settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.models.tool_approval_setting import ToolApprovalSetting


VALID_SCOPE_TYPES = {"server", "tool"}


class ToolApprovalSettingRepository:
    """Sync, session-factory style repository (mirrors CustomAgentRepository)."""

    def __init__(self, session_factory: Callable[[], Any]):
        self.session_factory = session_factory

    @staticmethod
    def _validate_scope(scope_type: str, scope_value: str) -> tuple[str, str]:
        normalized_type = str(scope_type or "").strip()
        normalized_value = str(scope_value or "").strip()
        if normalized_type not in VALID_SCOPE_TYPES:
            raise ValueError("scope_type must be 'server' or 'tool'")
        if not normalized_value:
            raise ValueError("scope_value is required")
        return normalized_type, normalized_value

    def list_by_user(self, user_id: UUID) -> list[ToolApprovalSetting]:
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(ToolApprovalSetting.user_id == user_id)
            return list(session.execute(stmt).scalars().all())

    def set(
        self, user_id: UUID, scope_type: str, scope_value: str, require_approval: bool
    ) -> ToolApprovalSetting:
        scope_type, scope_value = self._validate_scope(scope_type, scope_value)
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(
                ToolApprovalSetting.user_id == user_id,
                ToolApprovalSetting.scope_type == scope_type,
                ToolApprovalSetting.scope_value == scope_value,
            )
            setting = session.execute(stmt).scalar_one_or_none()
            if setting is None:
                setting = ToolApprovalSetting(
                    user_id=user_id,
                    scope_type=scope_type,
                    scope_value=scope_value,
                    require_approval=require_approval,
                )
                session.add(setting)
            else:
                setting.require_approval = require_approval
            session.commit()
            session.refresh(setting)
            session.expunge(setting)
            return setting

    def bulk_set(self, user_id: UUID, items: list[dict[str, Any]]) -> list[ToolApprovalSetting]:
        return [
            self.set(
                user_id,
                str(item["scope_type"]),
                str(item["scope_value"]),
                bool(item["require_approval"]),
            )
            for item in items
        ]

    def delete(self, user_id: UUID, scope_type: str, scope_value: str) -> bool:
        scope_type, scope_value = self._validate_scope(scope_type, scope_value)
        with self.session_factory() as session:
            stmt = select(ToolApprovalSetting).where(
                ToolApprovalSetting.user_id == user_id,
                ToolApprovalSetting.scope_type == scope_type,
                ToolApprovalSetting.scope_value == scope_value,
            )
            setting = session.execute(stmt).scalar_one_or_none()
            if setting is None:
                return False
            session.delete(setting)
            session.commit()
            return True

    def build_policy(self, user_id: UUID) -> dict[str, dict[str, bool]]:
        servers: dict[str, bool] = {}
        tools: dict[str, bool] = {}
        for row in self.list_by_user(user_id):
            if row.scope_type == "server":
                servers[row.scope_value] = bool(row.require_approval)
            elif row.scope_type == "tool":
                tools[row.scope_value] = bool(row.require_approval)
        return {"servers": servers, "tools": tools}
```

- [x] **Step 4: Run + commit**

Run: `.conda\python.exe -m pytest tests/test_tool_approval_setting_repository.py -q` → PASS

```powershell
git add app/repositories/tool_approval_setting.py tests/test_tool_approval_setting_repository.py
git commit -m "feat(hitl): ToolApprovalSettingRepository + build_policy"
```

---

### Task 4: `HitlSettingsService` + DI wiring + `/hitl/settings` API

**Files:**
- Create: `app/services/hitl_settings_service.py`
- Create: `app/schemas/hitl.py`
- Create: `app/api/hitl.py`
- Modify: `app/core/container.py` (repo provider ~line 198; service provider ~line 390; do **not** inject into `message_service` until Task 6)
- Modify: `app/core/dependency_injection.py` (import + add `HitlSettingsService` to `AppAutoInjector.wiring_map` after line 135)
- Modify: `app/api/__init__.py` (export `router as hitl_router`)
- Modify: `app/main.py` (`include_router(hitl_router)` in the block at 226-245)
- Test: `tests/test_hitl_api.py` (new)

**Interfaces:**
- Consumes: `ToolApprovalSettingRepository` (Task 3), `app/ai/hitl_config.{is_hitl_enabled,get_tools_requiring_approval}`.
- Produces:
  - `HitlSettingsService(repository)` with `get_settings(user_id) -> dict`, `apply(user_id, items) -> dict`, `clear(user_id, scope_type, scope_value) -> dict`, `build_turn_policy(user_id) -> dict`.
  - REST: `GET /hitl/settings`, `POST /hitl/settings`, `DELETE /hitl/settings`.

- [x] **Step 1: Write the failing API test — `tests/test_hitl_api.py`**

```python
"""HTTP-level tests for the per-user HITL settings API."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.hitl import router
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.database.database import Database
from app.models.tool_approval_setting import ToolApprovalSetting
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
    user_id = uuid4()
    with sf() as s:
        s.add(User(id=user_id, username=f"u_{user_id.hex[:12]}",
                   email=f"{user_id.hex[:12]}@test.local", password_hash="x"))
        s.commit()
    client = TestClient(_build_app(user_id))
    try:
        yield client, user_id, sf
    finally:
        with sf() as s:
            s.execute(delete(ToolApprovalSetting).where(ToolApprovalSetting.user_id == user_id))
            s.execute(delete(User).where(User.id == user_id))
            s.commit()


def test_post_then_get_roundtrips_server_and_tool_rules(api):
    client, _user_id, _sf = api
    resp = client.post("/hitl/settings", json={"items": [
        {"scopeType": "server", "scopeValue": "desktop_commander", "requireApproval": True},
        {"scopeType": "tool", "scopeValue": "desktop_commander::list_files", "requireApproval": False},
    ]})
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    data = client.get("/hitl/settings").json()["data"]
    servers = {s["scopeValue"]: s["requireApproval"] for s in data["servers"]}
    tools = {t["scopeValue"]: t["requireApproval"] for t in data["tools"]}
    assert servers == {"desktop_commander": True}
    assert tools == {"desktop_commander::list_files": False}
    assert "masterEnabled" in data


def test_delete_clears_rule(api):
    client, _user_id, _sf = api
    client.post("/hitl/settings", json={"items": [
        {"scopeType": "server", "scopeValue": "excel", "requireApproval": True},
    ]})
    resp = client.request("DELETE", "/hitl/settings",
                          params={"scope_type": "server", "scope_value": "excel"})
    assert resp.status_code == 200
    data = client.get("/hitl/settings").json()["data"]
    assert all(s["scopeValue"] != "excel" for s in data["servers"])


def test_rejects_invalid_scope_type(api):
    client, _user_id, _sf = api
    resp = client.post("/hitl/settings", json={"items": [
        {"scopeType": "garbage", "scopeValue": "ignored", "requireApproval": True},
    ]})
    assert resp.status_code == 422
```

- [x] **Step 2: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/test_hitl_api.py -q`
Expected: FAIL — `app.api.hitl` does not exist.

- [x] **Step 3: Create `app/schemas/hitl.py`**

```python
"""Schemas for the per-user HITL approval settings API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class HitlScopeRule(_CamelModel):
    scope_type: Literal["server", "tool"] = Field(..., description='"server" or "tool"')
    scope_value: str = Field(..., description="server name or qualified tool id")
    require_approval: bool


class HitlSettingsUpdate(_CamelModel):
    items: list[HitlScopeRule]


class HitlSettingsResponse(_CamelModel):
    master_enabled: bool
    global_tools: list[str]
    servers: list[HitlScopeRule]
    tools: list[HitlScopeRule]
```

(If `app/schemas/` already defines a shared camel-case base model, import that instead of `_CamelModel` — grep `alias_generator=to_camel` in `app/schemas` and reuse the existing base to stay DRY.)

- [x] **Step 4: Create `app/services/hitl_settings_service.py`**

```python
"""Per-user HITL approval settings service."""

from __future__ import annotations

from uuid import UUID

from app.ai.hitl_config import get_tools_requiring_approval, is_hitl_enabled
from app.repositories.tool_approval_setting import ToolApprovalSettingRepository


class HitlSettingsService:
    def __init__(self, repository: ToolApprovalSettingRepository):
        self.repository = repository

    def get_settings(self, user_id: UUID) -> dict:
        rows = self.repository.list_by_user(user_id)
        return {
            "master_enabled": is_hitl_enabled(),
            "global_tools": list(get_tools_requiring_approval()),
            "servers": [
                {"scope_type": "server", "scope_value": r.scope_value,
                 "require_approval": bool(r.require_approval)}
                for r in rows if r.scope_type == "server"
            ],
            "tools": [
                {"scope_type": "tool", "scope_value": r.scope_value,
                 "require_approval": bool(r.require_approval)}
                for r in rows if r.scope_type == "tool"
            ],
        }

    def apply(self, user_id: UUID, items: list[dict]) -> dict:
        self.repository.bulk_set(user_id, items)
        return self.get_settings(user_id)

    def clear(self, user_id: UUID, scope_type: str, scope_value: str) -> dict:
        self.repository.delete(user_id, scope_type, scope_value)
        return self.get_settings(user_id)

    def build_turn_policy(self, user_id: UUID) -> dict:
        """Full policy dict consumed by the graph gate (checkpoint-safe)."""
        grouped = self.repository.build_policy(user_id)
        return {
            "master_enabled": is_hitl_enabled(),
            "servers": grouped["servers"],
            "tools": grouped["tools"],
            "global_tools": list(get_tools_requiring_approval()),
        }
```

- [x] **Step 5: Create `app/api/hitl.py`**

```python
from uuid import UUID

from fastapi import APIRouter, Query

from app.core.dependency_injection import AppAutoInjector
from app.schemas.hitl import HitlScopeRule, HitlSettingsResponse, HitlSettingsUpdate
from app.schemas.responses import ApiResponse
from app.services.hitl_settings_service import HitlSettingsService

# No router-level auth dependency: the auto-injected ``user_id: UUID`` resolves to
# ``Depends(get_current_user_id)`` (DI magic), which both authenticates the request
# and yields the caller's id. This mirrors the custom-agents router and lets the API
# test authenticate by overriding ``get_current_user_id`` alone.
router = APIRouter(prefix="/hitl", tags=["hitl"])


def _to_response(data: dict) -> HitlSettingsResponse:
    return HitlSettingsResponse(
        master_enabled=data["master_enabled"],
        global_tools=data["global_tools"],
        servers=[HitlScopeRule(**r) for r in data["servers"]],
        tools=[HitlScopeRule(**r) for r in data["tools"]],
    )


@router.get("/settings")
@AppAutoInjector.auto_inject()
async def get_hitl_settings(
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
) -> ApiResponse[HitlSettingsResponse]:
    data = hitl_settings_service.get_settings(user_id)
    return ApiResponse(success=True, message="HITL settings retrieved", data=_to_response(data))


@router.post("/settings")
@AppAutoInjector.auto_inject()
async def update_hitl_settings(
    payload: HitlSettingsUpdate,
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
) -> ApiResponse[HitlSettingsResponse]:
    items = [r.model_dump() for r in payload.items]
    data = hitl_settings_service.apply(user_id, items)
    return ApiResponse(success=True, message="HITL settings updated", data=_to_response(data))


@router.delete("/settings")
@AppAutoInjector.auto_inject()
async def clear_hitl_setting(
    hitl_settings_service: HitlSettingsService,
    user_id: UUID,
    scope_type: str = Query(...),
    scope_value: str = Query(...),
) -> ApiResponse[HitlSettingsResponse]:
    data = hitl_settings_service.clear(user_id, scope_type, scope_value)
    return ApiResponse(success=True, message="HITL setting cleared", data=_to_response(data))
```

- [x] **Step 6: Register the repo + service in `app/core/container.py`**

Add the repository provider after `custom_agent_repository` (~line 198):

```python
    tool_approval_setting_repository = providers.Factory(
        ToolApprovalSettingRepository,
        session_factory=db.provided.session,
    )
```

Add the import near the other repository imports (with `CustomAgentRepository`):

```python
from app.repositories.tool_approval_setting import ToolApprovalSettingRepository
```

Add the service provider after `custom_agent_service` (~line 390):

```python
    hitl_settings_service = providers.Factory(
        HitlSettingsService,
        repository=tool_approval_setting_repository,
    )
```

Add the import near the other service imports (with `CustomAgentService`):

```python
from app.services.hitl_settings_service import HitlSettingsService
```

Do not inject the repository into `message_service` in this task. Task 6 adds the constructor parameter and provider argument together so every task-level commit remains boot-importable.

- [x] **Step 7: Wire `HitlSettingsService` for auto-injection in `app/core/dependency_injection.py`**

Add the import (with the other service imports near line 28-31):

```python
from app.services.hitl_settings_service import HitlSettingsService
```

Add to `AppAutoInjector.wiring_map` (after `CustomAgentService: container_ref.custom_agent_service,` at line 135):

```python
            HitlSettingsService: container_ref.hitl_settings_service,
```

- [x] **Step 8: Export + mount the router**

In `app/api/__init__.py`, add an export mirroring the existing `mcp` router export:

```python
from app.api.hitl import router as hitl_router
```

In `app/main.py`, add inside the include block (after `app.include_router(mcp_router)` ~line 237):

```python
    app.include_router(hitl_router)
```

(Import `hitl_router` alongside the other router imports at the top of `app/main.py`, mirroring `mcp_router`.)

- [x] **Step 9: Run boot-import + API tests**

Run: `.conda\python.exe -c "import app.main"` → no ImportError
Run: `.conda\python.exe -m pytest tests/test_hitl_api.py -q` → PASS
(If the DB at `settings.database_url` is unreachable in this environment, the fixture will error on connect — start the dev Postgres or run this task's test against it; do not stub it out.)

- [x] **Step 10: Commit**

```powershell
git add app/services/hitl_settings_service.py app/schemas/hitl.py app/api/hitl.py app/api/__init__.py app/core/container.py app/core/dependency_injection.py app/main.py tests/test_hitl_api.py
git commit -m "feat(hitl): per-user settings service + /hitl/settings API"
```

---

### Task 5: Wire the policy into all five graph gates

**Files:**
- Modify: `app/ai/graph.py` (import; new `_needs_approval` helper near `_prepare_interrupt_payload` ~1073; make `_should_call_tools` async at 1209; replace the 5 `requires_human_approval(...)` calls at 1219/2453/2892/2996/3297)
- Modify: `tests/test_custom_agents_graph.py:180,188` (`_should_call_tools` is now a coroutine → `await`)
- Test: `tests/test_hitl_gate_policy.py` (new)

**Interfaces:**
- Consumes: `any_call_requires_approval`, `policy_from_context` (Task 1); `ensure_agent_tool_map` (`app/ai/tool_execution.py`), `get_global_mcp_manager` (`app/ai/mcp_registry.py`); `normalize_tool_call`, `GraphStateView` (already imported in `graph.py`).
- Produces: `MultiAgentWorkflow._needs_approval(state, normalized_calls, *, agent=None, tool_map=None) -> bool` (async).

- [x] **Step 1: Write the failing gate test — `tests/test_hitl_gate_policy.py`**

```python
"""The graph gate honors a per-turn per-user policy and resolves server provenance."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.ai import graph as graph_module
from app.ai.schemas import AgentMessage, AgentResponse, AgentType, MessageRole


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def _workflow_stub(tool_map, manager, monkeypatch):
    """A bare object exposing just what _needs_approval / _should_call_tools touch."""
    wf = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    agent = object()
    wf.agents = {"chat_agent": agent}

    async def _fake_tool_map(*_a, **_k):
        return tool_map

    async def _fake_manager():
        return manager

    monkeypatch.setattr(graph_module, "ensure_agent_tool_map", _fake_tool_map)
    monkeypatch.setattr(graph_module, "get_global_mcp_manager", _fake_manager)
    return wf


@pytest.mark.asyncio
async def test_gate_gates_all_tools_from_a_server_via_policy(monkeypatch):
    server_tool = SimpleNamespace(name="search", metadata={})
    tool_map = {"search": server_tool}
    manager = _FakeManager({id(server_tool): "tavily"})
    wf = _workflow_stub(tool_map, manager, monkeypatch)

    state = {
        "selected_agent": "chat_agent",
        "conversation_id": "c1", "user_id": "u1", "device_id": None,
        "context": {"hitl_policy": {
            "master_enabled": True, "servers": {"tavily": True}, "tools": {}, "global_tools": [],
        }},
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "search", "args": {}, "id": "call-1"}
        ])],
    }
    assert await wf._should_call_tools(state) == "approval"


@pytest.mark.asyncio
async def test_gate_lets_tool_override_exempt_a_server_tool(monkeypatch):
    tool = SimpleNamespace(
        name="client__desktop_commander__list_files",
        metadata={"server_name": "desktop_commander",
                  "qualified_tool_id": "desktop_commander::list_files",
                  "tool_origin": "client_mcp"},
    )
    tool_map = {tool.name: tool}
    wf = _workflow_stub(tool_map, _FakeManager({}), monkeypatch)

    state = {
        "selected_agent": "chat_agent",
        "conversation_id": "c1", "user_id": "u1", "device_id": None,
        "context": {"hitl_policy": {
            "master_enabled": True,
            "servers": {"desktop_commander": True},
            "tools": {"desktop_commander::list_files": False},
            "global_tools": [],
        }},
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "client__desktop_commander__list_files", "args": {}, "id": "call-1"}
        ])],
    }
    assert await wf._should_call_tools(state) == "tools"


@pytest.mark.asyncio
async def test_gate_master_off_never_gates(monkeypatch):
    wf = _workflow_stub({}, _FakeManager({}), monkeypatch)
    state = {
        "selected_agent": "chat_agent",
        "conversation_id": "c1", "user_id": "u1", "device_id": None,
        "context": {"hitl_policy": {"master_enabled": False, "servers": {"tavily": True},
                                    "tools": {}, "global_tools": []}},
        "messages": [AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "x"}])],
    }
    assert await wf._should_call_tools(state) == "tools"


@pytest.mark.asyncio
async def test_generic_worker_uses_parent_state_hitl_policy(monkeypatch):
    server_tool = SimpleNamespace(name="search", metadata={})
    tool_map = {"search": server_tool}
    manager = _FakeManager({id(server_tool): "tavily"})
    wf = _workflow_stub(tool_map, manager, monkeypatch)

    class _FakeAgent:
        agent_config_key = "chat"

        async def invoke_model_with_history(self, **_kwargs):
            return AgentResponse(
                agent_type=AgentType.CHAT,
                agent_id="chat_agent",
                message=AgentMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[{"name": "search", "args": {}, "id": "call-1"}],
                ),
                metadata={},
            )

    wf.agents = {"chat_agent": _FakeAgent()}
    parent_state = {
        "conversation_id": "c1",
        "user_id": "u1",
        "device_id": None,
        "context": {"hitl_policy": {
            "master_enabled": True, "servers": {"tavily": True}, "tools": {}, "global_tools": [],
        }},
    }

    response = await wf._run_agent_in_isolated_context(
        agent_name="chat_agent",
        task_prompt="use search",
        parent_state=parent_state,
    )
    assert response.metadata["requires_approval"] is True
    assert response.metadata["pause_reason"] == "awaiting_approval"
```

- [x] **Step 2: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/test_hitl_gate_policy.py -q`
Expected: FAIL — `_should_call_tools` is currently sync (returns a str, not awaitable) and ignores `hitl_policy`/server resolution, so awaiting it errors and/or the server-scope case returns `"tools"`.

- [x] **Step 3: Add imports + the `_needs_approval` helper in `app/ai/graph.py`**

Update the hitl_config import (line 52). All five gate sites move to `_needs_approval`, so `requires_human_approval` becomes unused **in graph.py** and must be dropped from this import (it stays defined/public in `hitl_config` for the legacy tests). Replace:

```python
from .hitl_config import build_interrupt_response, requires_human_approval
```

with:

```python
from .hitl_config import (
    any_call_requires_approval,
    build_interrupt_response,
    policy_from_context,
)
```

Add the import for the global MCP manager near the other local imports:

```python
from .mcp_registry import get_global_mcp_manager
```

Do not import it from `app.ai.tool_execution`; that module only imports the function locally inside helper functions and does not export it.

Add this method to `MultiAgentWorkflow`, directly above `_prepare_interrupt_payload` (~line 1073):

```python
    async def _needs_approval(
        self,
        state: GraphState,
        normalized_calls: list[dict[str, Any]],
        *,
        agent: Any | None = None,
        tool_map: dict[str, Any] | None = None,
    ) -> bool:
        """Resolve per-call provenance and apply the per-turn HITL policy."""
        policy = policy_from_context(state.get("context"))
        if not policy.get("master_enabled", True):
            return False

        mcp_manager = None
        if tool_map is None and agent is not None:
            view = GraphStateView(state)
            tool_map = await ensure_agent_tool_map(
                agent,
                conversation_id=view.conversation_id(),
                user_id=view.user_id(),
                device_id=view.device_id(),
            )
        if tool_map is not None:
            mcp_manager = await get_global_mcp_manager()

        return any_call_requires_approval(
            normalized_calls, policy=policy, tool_map=tool_map, mcp_manager=mcp_manager
        )
```

- [x] **Step 4: Make `_should_call_tools` async + policy-aware (graph.py:1209-1222)**

Replace the whole method:

```python
    async def _should_call_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "end"

        normalized_calls = [normalize_tool_call(tc) for tc in last_message.tool_calls]
        selected_agent_name = state.get("selected_agent")
        agent = self.agents.get(selected_agent_name) if selected_agent_name else None
        if await self._needs_approval(state, normalized_calls, agent=agent):
            return "approval"

        return "tools"
```

- [x] **Step 5: Replace the RAG gate (graph.py:2452-2453)**

Change:

```python
            tool_names = [tc.get("name") for tc in non_search_tool_calls]
            if requires_human_approval(tool_names):
```

to:

```python
            if await self._needs_approval(state, non_search_tool_calls, agent=agent):
```

- [x] **Step 6: Replace the RAG sub-worker gate (graph.py:2891-2892)**

Change:

```python
                tool_call_names = [tc.get("name") or "" for tc in normalized_calls]
                if requires_human_approval(tool_call_names):
```

to:

```python
                if await self._needs_approval(parent_state, normalized_calls, agent=agent):
```

- [x] **Step 7: Replace the generic worker gate (graph.py:2995-2996)**

Change:

```python
            tool_call_names = [normalize_tool_call(tc).get("name") or "" for tc in tool_calls]
            if requires_human_approval(tool_call_names):
```

to:

```python
            normalized_worker_calls = [normalize_tool_call(tc) for tc in tool_calls]
            if await self._needs_approval(parent_state, normalized_worker_calls, agent=agent):
```

- [x] **Step 8: Replace the planning gate (graph.py:3296-3297)**

Change:

```python
            ext_tool_names = [tc.get("name") for tc in external_tool_calls]
            if requires_human_approval(ext_tool_names):
```

to (the planning node already built `tool_map` at line 3271 — pass it to avoid a second build):

```python
            if await self._needs_approval(
                state, external_tool_calls, agent=self.planning_agent, tool_map=tool_map
            ):
```

- [x] **Step 9: Fix `tests/test_custom_agents_graph.py:180,188` (now a coroutine)**

Change the two assertions to await the coroutine. Mark the test(s) async if not already:

```python
    assert await wf._should_call_tools(done_state) == "end"
    ...
    assert await wf._should_call_tools(tool_state) == "tools"
```

If those assertions live in a sync test, convert that test to `async def` + `@pytest.mark.asyncio` (mirror the style of other async tests in the file). The state dicts used there have no `hitl_policy` → `policy_from_context` returns the global policy → with the default empty `hitl_tools_require_approval` the gate returns `"tools"`, preserving the original expectation.

- [x] **Step 10: Run gate + custom-agent graph suites**

Run: `.conda\python.exe -m pytest tests/test_hitl_gate_policy.py tests/test_custom_agents_graph.py tests/test_hitl_config.py tests/test_client_tool_isolation.py -q`
Expected: PASS. (`requires_human_approval` is no longer referenced in `graph.py` after this task — Step 3 dropped it from the import — but stays defined and public in `hitl_config.py`, which is what the two legacy tests import and exercise.)

- [x] **Step 11: Commit**

```powershell
git add app/ai/graph.py tests/test_hitl_gate_policy.py tests/test_custom_agents_graph.py
git commit -m "feat(hitl): route all five gates through the per-user policy resolver"
```

---

### Task 6: Load the per-user policy per turn and carry it into graph context

**Files:**
- Modify: `app/services/message_service.py` (`__init__` accepts `tool_approval_setting_repository`; add `_resolve_hitl_policy`; call it in `_build_user_message_workflow_request` ~2155-2198)
- Modify: `app/core/container.py` (pass `tool_approval_setting_repository` to `MessageService` only after the constructor parameter exists)
- Modify: `app/schemas/workflow.py` + `app/ai/schemas.py` AI-layer request/`GraphContext` schema (mirror `inline_rich_response_v1`; `_to_ai_request` should then preserve the field)
- Modify: `app/ai/graph.py:_build_initial_state_from_request` (~512, inject into `context`)
- Test: `tests/test_hitl_turn_policy_injection.py` (new)

**Interfaces:**
- Consumes: `ToolApprovalSettingRepository.build_policy` (same policy shape exposed by `HitlSettingsService.build_turn_policy` in Task 4); the `hitl_policy` carrier field.
- Produces: `WorkflowExecutionRequest.hitl_policy: dict | None`; `context["hitl_policy"]` in initial graph state.

- [x] **Step 1: Locate the `inline_rich_response_v1` carrier and mirror it**

Run: `rg -n "inline_rich_response_v1" app`
This field is the working precedent for carrying a per-turn flag across the dual `WorkflowExecutionRequest` schemas into graph context (see `[[dual-workflow-request-schema-drift]]`). For EVERY location it appears in `app/schemas/workflow.py` and the AI-layer request schema, add a sibling `hitl_policy: dict | None = None` (default `None`). Do not skip the `_to_ai_request` conversion test below — if the AI-layer schema is missing the field, Pydantic will silently drop it there.

- [x] **Step 2: Write the failing injection test — `tests/test_hitl_turn_policy_injection.py`**

```python
"""The per-turn HITL policy reaches graph context via the workflow request."""

from app.schemas.workflow import WorkflowExecutionRequest


def test_request_carries_hitl_policy_field():
    policy = {"master_enabled": True, "servers": {"excel": True}, "tools": {}, "global_tools": []}
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1", user_id="u1",
                                   hitl_policy=policy)
    assert req.hitl_policy == policy


def test_initial_state_injects_hitl_policy_into_context():
    from app.ai.graph import MultiAgentWorkflow

    wf = MultiAgentWorkflow.__new__(MultiAgentWorkflow)
    policy = {"master_enabled": True, "servers": {"excel": True}, "tools": {}, "global_tools": []}
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1", user_id="u1",
                                   hitl_policy=policy)
    state = wf._build_initial_state_from_request(req)
    assert state["context"]["hitl_policy"] == policy


def test_ai_service_conversion_preserves_hitl_policy():
    from app.services.ai_service import AIService

    policy = {"master_enabled": True, "servers": {"excel": True}, "tools": {}, "global_tools": []}
    req = WorkflowExecutionRequest(message="hi", conversation_id="c1", user_id="u1",
                                   hitl_policy=policy)
    ai_req = AIService._to_ai_request(req)
    assert ai_req.hitl_policy == policy
```

- [x] **Step 3: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/test_hitl_turn_policy_injection.py -q`
Expected: FAIL — `hitl_policy` is not a field yet, is not present on the AI-layer schema, and is not injected into context.

- [x] **Step 4: Inject into context in `app/ai/graph.py:_build_initial_state_from_request`**

Where the `context` dict is built (~line 512, alongside `inline_rich_response_v1`), add:

```python
        "hitl_policy": getattr(request, "hitl_policy", None),
```

(Keep the existing keys; this just adds one entry. `policy_from_context` ignores a `None` value and falls back to the global policy, so unauthenticated/legacy turns are unaffected.)

Also add `hitl_policy: dict[str, Any] | None = None` to both `app/schemas/workflow.py::WorkflowExecutionRequest` and `app/ai/schemas.py::WorkflowExecutionRequest`, and add `hitl_policy: dict[str, Any]` to `app/ai/schemas.py::GraphContext` near `inline_rich_response_v1`.

- [x] **Step 5: Accept the repository + resolve the policy in `app/services/message_service.py`**

Add the parameter to `MessageService.__init__` (mirror how `custom_agent_service` is accepted and stored):

```python
        tool_approval_setting_repository=None,
```

and in the body:

```python
        self.tool_approval_setting_repository = tool_approval_setting_repository
```

In `app/core/container.py`, add this line to the `MessageService` provider block (after `custom_agent_service=custom_agent_service,` ~line 417):

```python
        tool_approval_setting_repository=tool_approval_setting_repository,
```

Add the resolver method (mirror `_resolve_custom_agents_state`):

```python
    def _resolve_hitl_policy(self, user_id) -> dict | None:
        repo = getattr(self, "tool_approval_setting_repository", None)
        if repo is None or not user_id:
            return None
        try:
            from app.ai.hitl_config import get_tools_requiring_approval, is_hitl_enabled

            grouped = repo.build_policy(user_id)
            return {
                "master_enabled": is_hitl_enabled(),
                "servers": grouped["servers"],
                "tools": grouped["tools"],
                "global_tools": list(get_tools_requiring_approval()),
            }
        except Exception as exc:
            logging.warning("Failed to resolve HITL policy: %s", exc)
            return None
```

In `_build_user_message_workflow_request`, near where `custom_agents_state` is resolved (~line 2176), add:

```python
        hitl_policy = self._resolve_hitl_policy(resolved_user_id)
```

and pass it into the `WorkflowExecutionRequest(...)` construction (~line 2191):

```python
            hitl_policy=hitl_policy,
```

- [x] **Step 6: Run injection + a message-service smoke**

Run: `.conda\python.exe -m pytest tests/test_hitl_turn_policy_injection.py tests/test_custom_agents_message_service.py -q`
Expected: PASS. (`tool_approval_setting_repository` defaults to `None`, so existing `MessageService` constructions in tests that don't pass it keep working and `_resolve_hitl_policy` returns `None` → global-policy fallback.)

- [x] **Step 7: Commit**

```powershell
git add app/services/message_service.py app/core/container.py app/schemas/workflow.py app/ai/schemas.py app/ai/graph.py tests/test_hitl_turn_policy_injection.py
git commit -m "feat(hitl): load per-user policy per turn and inject into graph context"
```

---

### Task 7: Verify HITL round-trips for client (sidecar) and deferred tools

This is the explicit "test HITL works with the client sidecar + deferred tool loading" requirement. It adds behavioral guards on top of Task 5's gate and proves provenance + interrupt enrichment for the two origins the previous global-name list never modeled.

**Files:**
- Test: `tests/test_hitl_client_and_deferred.py` (new)
- Modify (only if a guard fails): `app/ai/graph.py` `_prepare_interrupt_payload` / `_needs_approval`

**Interfaces:**
- Consumes: `resolve_call_identity`, `any_call_requires_approval` (Task 1); `MultiAgentWorkflow._needs_approval` (Task 5); `_prepare_interrupt_payload` (existing).

- [ ] **Step 1: Write the behavioral guards — `tests/test_hitl_client_and_deferred.py`**

```python
"""HITL fires correctly for client (sidecar) tools and deferred (search-loaded) tools."""

from types import SimpleNamespace

import pytest

from app.ai.hitl_config import any_call_requires_approval, resolve_call_identity


class _FakeManager:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_server_for_tool(self, tool):
        return self._mapping.get(id(tool))


def test_client_server_rule_gates_a_sidecar_tool_by_name_alone():
    # Sidecar tool, NOT yet in any tool_map (e.g. resolved purely from the call name).
    policy = {"master_enabled": True, "servers": {"desktop_commander": True},
              "tools": {}, "global_tools": []}
    calls = [{"name": "client__desktop_commander__start_process", "args": {}, "id": "c1"}]
    assert any_call_requires_approval(calls, policy=policy) is True


def test_deferred_server_tool_gated_by_server_after_autoload():
    # A server tool discovered + autoloaded via tool_search this turn: bare name, no
    # metadata, server resolved through the MCP manager (the deferred-binding path).
    loaded_tool = SimpleNamespace(name="run_query", metadata={})
    tool_map = {"run_query": loaded_tool}
    manager = _FakeManager({id(loaded_tool): "postgres"})
    policy = {"master_enabled": True, "servers": {"postgres": True},
              "tools": {}, "global_tools": []}

    identity = resolve_call_identity({"name": "run_query"}, tool_map=tool_map, mcp_manager=manager)
    assert identity.server_name == "postgres"

    calls = [{"name": "run_query", "args": {}, "id": "c1"}]
    assert any_call_requires_approval(calls, policy=policy, tool_map=tool_map, mcp_manager=manager) is True


@pytest.mark.asyncio
async def test_prepare_interrupt_payload_carries_client_provenance():
    from app.ai import graph as graph_module

    wf = graph_module.MultiAgentWorkflow.__new__(graph_module.MultiAgentWorkflow)
    client_tool = SimpleNamespace(
        name="client__excel__delete_sheet",
        metadata={"server_name": "excel", "qualified_tool_id": "excel::delete_sheet",
                  "tool_origin": "client_mcp"},
    )

    # agent=None + explicit tool_map => _prepare_interrupt_payload skips building a real map.
    payload = await wf._prepare_interrupt_payload(
        {"context": {}, "device_id": "dev-1"},
        tool_calls=[{"name": "client__excel__delete_sheet", "args": {}, "id": "c1"}],
        agent=None,
        tool_map={client_tool.name: client_tool},
    )
    prov = payload["metadata"]["tool_provenance"]
    entry = next(iter(prov.values()))
    assert entry["server_name"] == "excel"
    assert entry["qualified_tool_id"] == "excel::delete_sheet"
    assert entry["tool_origin"] == "client_mcp"
```

- [ ] **Step 2: Run the guards**

Run: `.conda\python.exe -m pytest tests/test_hitl_client_and_deferred.py -q`
Expected: PASS with the Task-1/Task-5 implementation. If the `_prepare_interrupt_payload` provenance test fails, the metadata extraction at `graph.py:1109-1124` is the place to fix (it already reads `server_name`/`qualified_tool_id`/`tool_origin`) — bring it in line with the resolver rather than weakening the test.

- [ ] **Step 3: Commit**

```powershell
git add tests/test_hitl_client_and_deferred.py app/ai/graph.py
git commit -m "test(hitl): guard client-sidecar + deferred-tool approval round-trips"
```

---

### Task 8: Sidecar proxy routes for `/hitl/settings`

**Files:**
- Modify: `client_backend/api/proxy.py` (add 3 additive routes)
- Test: `tests/client_backend/test_hitl_proxy.py` (new)

**Interfaces:**
- Consumes: `proxy_server_request`, `require_local_session`, `LocalSessionPayload` (existing in `client_backend/api/proxy.py` / `common.py`).
- Produces: sidecar routes `GET/POST/DELETE /hitl/settings` proxying to the same upstream paths (no device stamping).

- [ ] **Step 1: Write the failing proxy test — `tests/client_backend/test_hitl_proxy.py`**

```python
"""The sidecar proxies /hitl/settings to the canonical server without device stamping."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import proxy as proxy_module
from client_backend.api.proxy import router
from client_backend.core.auth import require_local_session


@pytest.fixture
def client(monkeypatch):
    captured = {}

    async def _fake_proxy(request, *, upstream_path, **kwargs):
        from fastapi.responses import JSONResponse

        captured["path"] = upstream_path
        captured["method"] = request.method
        captured["params_override"] = kwargs.get("params_override")
        return JSONResponse(status_code=200, content={"success": True, "data": {}})

    monkeypatch.setattr(proxy_module, "proxy_server_request", _fake_proxy)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_local_session] = lambda: object()
    return TestClient(app), captured


def test_get_hitl_settings_proxies_without_device_param(client):
    test_client, captured = client
    resp = test_client.get("/hitl/settings")
    assert resp.status_code == 200
    assert captured["path"] == "/hitl/settings"
    assert captured["method"] == "GET"
    assert captured["params_override"] is None  # per-user, no device stamping


def test_post_and_delete_hitl_settings_proxy(client):
    test_client, captured = client
    assert test_client.post("/hitl/settings", json={"items": []}).status_code == 200
    assert captured["path"] == "/hitl/settings"
    assert test_client.request(
        "DELETE", "/hitl/settings", params={"scope_type": "server", "scope_value": "excel"}
    ).status_code == 200
    assert captured["method"] == "DELETE"
```

- [ ] **Step 2: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/client_backend/test_hitl_proxy.py -q`
Expected: FAIL — routes not defined (404).

- [ ] **Step 3: Add the routes to `client_backend/api/proxy.py`**

Add near the other `@router.api_route(...)` proxy definitions (mirror the per-user `/providers` style, which passes no `params_override`):

```python
@router.api_route("/hitl/settings", methods=["GET", "POST", "DELETE"])
async def proxy_hitl_settings(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    # Per-user policy: forward verbatim, no device stamping.
    return await proxy_server_request(request, upstream_path="/hitl/settings")
```

- [ ] **Step 4: Run + commit**

Run: `.conda\python.exe -m pytest tests/client_backend/test_hitl_proxy.py -q` → PASS

```powershell
git add client_backend/api/proxy.py tests/client_backend/test_hitl_proxy.py
git commit -m "feat(hitl): sidecar proxies /hitl/settings (additive)"
```

---

### Task 9: Demo MCP panel — per-server toggle + per-tool tri-state

**Files:**
- Modify: `demo.py` (3 helpers near 3335-3347; per-server column in the Configured Servers loop ~7876-7919; per-tool tri-state + read-only master in the Tools section ~7922-7978)
- Test: `tests/test_hitl_demo_panel.py` (new — static source assertions, mirroring `tests/test_skills_architecture.py`)

**Interfaces:**
- Consumes: `make_api_request` (demo.py:2816). Calls the sidecar's `/hitl/settings` (Task 8) which proxies to the server (Task 4).
- Produces: demo helpers `get_hitl_settings()`, `set_hitl_setting(scope_type, scope_value, require_approval)`, `clear_hitl_setting(scope_type, scope_value)`.

- [ ] **Step 1: Write the failing static test — `tests/test_hitl_demo_panel.py`**

```python
"""Static guard: the demo MCP panel manages HITL approval via the sidecar API."""

from pathlib import Path


def _demo_source() -> str:
    return Path("demo.py").read_text(encoding="utf-8")


def test_demo_defines_hitl_helpers_and_calls_settings_endpoint():
    src = _demo_source()
    assert "def get_hitl_settings(" in src
    assert "def set_hitl_setting(" in src
    assert "def clear_hitl_setting(" in src
    assert 'make_api_request("GET", "/hitl/settings")' in src
    assert 'make_api_request("POST", "/hitl/settings"' in src


def test_demo_renders_per_server_and_per_tool_controls():
    src = _demo_source()
    assert "Approval: ON" in src              # per-server toggle label
    assert "hitl_tool_mode_" in src           # per-tool tri-state widget key prefix
    assert "qualified_tool_options" in src    # duplicate tool names select by server::tool
```

- [ ] **Step 2: Run and verify FAIL**

Run: `.conda\python.exe -m pytest tests/test_hitl_demo_panel.py -q`
Expected: FAIL — helpers/labels absent.

- [ ] **Step 3: Add the helpers (near `get_mcp_tools`, ~demo.py:3347)**

```python
def get_hitl_settings() -> dict[str, Any] | None:
    """Fetch this user's HITL approval settings (via the sidecar proxy)."""
    response = make_api_request("GET", "/hitl/settings")
    return response.get("data") if response else None


def set_hitl_setting(scope_type: str, scope_value: str, require_approval: bool) -> dict[str, Any] | None:
    """Upsert one HITL approval rule (server- or tool-scoped)."""
    response = make_api_request(
        "POST", "/hitl/settings",
        {"items": [{"scopeType": scope_type, "scopeValue": scope_value,
                    "requireApproval": require_approval}]},
    )
    return response.get("data") if response else None


def clear_hitl_setting(scope_type: str, scope_value: str) -> dict[str, Any] | None:
    """Delete one HITL approval rule (revert to inherit/default)."""
    response = make_api_request(
        "DELETE", f"/hitl/settings?scope_type={scope_type}&scope_value={scope_value}"
    )
    return response.get("data") if response else None
```

- [ ] **Step 4: Render the per-server toggle in the Configured Servers loop (~demo.py:7876-7919)**

Before the loop, fetch settings once:

```python
        hitl_settings = get_hitl_settings() or {}
        hitl_master = bool(hitl_settings.get("masterEnabled", True))
        hitl_servers = {s["scopeValue"]: s["requireApproval"] for s in hitl_settings.get("servers", [])}
        if not hitl_master:
            st.caption(":material/info: Human-in-the-loop is globally disabled (admin setting); approval rules below are inactive until it is enabled.")
```

Change the per-row column split from `st.columns([3, 1, 1])` to `st.columns([3, 1, 1, 1])`, keep `col1/col2/col3` as-is, and add a fourth column that toggles the server rule:

```python
        with col4:
            server_gated = bool(hitl_servers.get(server_name, False))
            approval_label = "Approval: ON" if server_gated else "Approval: OFF"
            if st.button(approval_label, key=f"hitl_server_{server_name}",
                         help="Require human approval for all tools from this server"):
                with st.spinner("Updating approval rule..."):
                    if set_hitl_setting("server", server_name, not server_gated) is not None:
                        st.rerun()
```

- [ ] **Step 5: Make the tool selector use a qualified key (near demo.py:7953-7966)**

Replace the current selectbox that uses only `tool.get("name")` as the option value with a qualified key. This prevents the UI from editing the wrong rule when two servers expose the same tool name:

```python
    qualified_tool_options = {
        f"{tool.get('serverName', '')}::{tool.get('name')}": tool
        for tool in filtered_tools
        if tool.get("name")
    }
    selected_tool_key = st.selectbox(
        "Select a tool to test",
        options=list(qualified_tool_options.keys()),
        format_func=lambda key: (
            f"{qualified_tool_options[key].get('name')} "
            f"({qualified_tool_options[key].get('serverName', '')})"
        ),
    )

    if not selected_tool_key:
        return

    selected_tool = qualified_tool_options.get(selected_tool_key)
    if not selected_tool:
        return
    selected_tool_name = selected_tool.get("name")
```

- [ ] **Step 6: Render the per-tool tri-state in the Tools section (after the selected-tool details, ~demo.py:7978)**

```python
        st.markdown("**Human approval**")
        qualified_id = selected_tool_key
        hitl_settings = get_hitl_settings() or {}
        tool_rules = {t["scopeValue"]: t["requireApproval"] for t in hitl_settings.get("tools", [])}

        if qualified_id in tool_rules:
            current_mode = "Require" if tool_rules[qualified_id] else "Skip"
        else:
            current_mode = "Inherit"

        modes = ["Inherit", "Require", "Skip"]
        chosen = st.radio(
            "Approval mode for this tool",
            modes,
            index=modes.index(current_mode),
            key=f"hitl_tool_mode_{qualified_id}",
            horizontal=True,
            help="Inherit = follow the server rule; Require = always prompt; Skip = never prompt",
        )
        if chosen != current_mode:
            with st.spinner("Updating tool approval..."):
                if chosen == "Inherit":
                    result = clear_hitl_setting("tool", qualified_id)
                else:
                    result = set_hitl_setting("tool", qualified_id, chosen == "Require")
                if result is not None:
                    st.rerun()
```

- [ ] **Step 7: Run static test + demo import**

Run: `.conda\python.exe -m pytest tests/test_hitl_demo_panel.py -q` → PASS
Run: `.conda\python.exe -c "import ast, pathlib; ast.parse(pathlib.Path('demo.py').read_text(encoding='utf-8'))"` → no SyntaxError

- [ ] **Step 8: Commit**

```powershell
git add demo.py tests/test_hitl_demo_panel.py
git commit -m "feat(hitl): demo MCP panel toggles approval per-server and per-tool"
```

---

### Task 10: Documentation + full verification

**Files:**
- Modify: `README.md` (HITL section)
- Append: `plans/HITL-refactor.md` (this file — execution log)

- [ ] **Step 1: README — document the new contract**

In the HITL/MCP section, add:

> **Human-in-the-loop approval (per-user).** Approval is governed by a per-user policy stored server-side (`tool_approval_settings`). A rule is either **server-scoped** (gates every tool from an MCP server) or **tool-scoped** (a `"<server>::<tool>"` rule that overrides its server). Precedence: tool rule > server rule > the legacy global floor `hitl_tools_require_approval`; the global `enable_human_in_the_loop` switch is the master kill-switch. Manage it from the demo's MCP panel (per-server "Approval" toggle; per-tool Inherit/Require/Skip), which calls `GET/POST/DELETE /hitl/settings` through the sidecar proxy.

- [ ] **Step 2: Full suite**

Run: `.conda\python.exe -m pytest tests --ignore=tests/client_backend/test_live_server_integration.py -q`
Expected: 0 failures. Fix any failure before proceeding — do not skip tests to get green. (If `tests/test_hitl_api.py` can't reach Postgres in this environment, run it separately against the dev DB and note the result; everything else must pass offline.)

- [ ] **Step 3: Lint touched files**

Run: `.conda\python.exe -m ruff check app/ai/hitl_config.py app/ai/graph.py app/ai/schemas.py app/models/tool_approval_setting.py app/repositories/tool_approval_setting.py app/services/hitl_settings_service.py app/schemas/hitl.py app/api/hitl.py app/core/container.py app/core/dependency_injection.py app/services/message_service.py app/schemas/workflow.py app/alembic/versions/g0h1i2j3k4l5_add_tool_approval_settings.py client_backend/api/proxy.py demo.py`
Expected: clean, or only findings already present on HEAD (verify against HEAD if unsure). Fix anything new.

- [ ] **Step 4: Manual smoke (dev parity — the live HITL round-trip)**

Start server (:8000), sidecar (:8100, `CLIENT_SKILLS_ROOTS=<repo>\skills`), and `streamlit run demo.py` per the README's three-process dev setup. Then:
1. MCP panel → toggle **Approval: ON** for a client server that exposes a real tool (e.g. `desktop_commander`). (Writes a `server` rule.)
2. Ask the model to use one of that server's tools → the turn pauses with an approval interrupt naming that tool; **Approve** → it dispatches to the sidecar and runs; repeat and **Reject** → it does not run.
3. Set that one tool's mode to **Skip** → calling it no longer prompts (tool overrides server).
4. Set a single low-risk tool to **Require** while its server is OFF → only that tool prompts.
5. Trigger a tool that was **discovered via `tool_search` this turn** (deferred load) and is covered by a server rule → it still prompts (proves deferred-tool gating).
6. Confirm a non-gated tool (e.g. `time`) runs without a prompt.

Record the outcome of each step in the execution log below — do not mark this task done on partial smoke results.

- [ ] **Step 5: Final commit + push**

```powershell
git add -A
git commit -m "docs(hitl): per-user granular approval contract + dev smoke"
git push
```

---

## Execution notes

- **Order:** Tasks 1→7 are sequenced (policy core → persistence → API → gate wiring → turn injection → round-trip guards). Tasks 8 and 9 depend on Task 4's API existing but are independent of each other. Run Task 10 last.
- **Keep every commit green** (each task ends green; there is no intentional-red commit in this plan — failing tests are written and made green within the same task).
- **Back-compat is load-bearing:** `requires_human_approval`, `enable_human_in_the_loop`, and `hitl_tools_require_approval` all keep their current meaning. `tool_approval_setting_repository` is injected with a `None`-tolerant default so every existing `MessageService`/graph construction in the test suite keeps working with the global-policy fallback.
- **If a step's expected result doesn't match reality, STOP** and re-verify the "Key research facts" before improvising — especially the dual-schema `inline_rich_response_v1` mirror (Task 6) and the async conversion of `_should_call_tools` (Task 5), which are the two highest-risk edits.

## Execution log

### Task 1 — Policy core (resolver, precedence, back-compat shim) ✅ 2026-06-22

- **Tests:** `tests/test_hitl_policy.py` (12 cases) written; Step-2 verify-fail confirmed `ImportError: cannot import name 'any_call_requires_approval'`.
- **Implementation:** Added `CLIENT_TOOL_PREFIX`, `_tool_call_name`, `CallIdentity` (frozen dataclass), `build_global_policy`, `policy_from_context`, `resolve_call_identity`, `identity_requires_approval`, `any_call_requires_approval` to `app/ai/hitl_config.py`. `requires_human_approval` left unchanged.
- **Verification:** `pytest tests/test_hitl_policy.py tests/test_hitl_config.py tests/test_client_tool_isolation.py -q` → **24 passed**. The two legacy guards stay green (unchanged shim).
- **Commit:** `feat(hitl): policy resolver + tool-overrides-server precedence (core)` (2 files, +278).
- **Pre-verified facts:** `is_hitl_enabled()`/`get_tools_requiring_approval()` already existed in `hitl_config.py` (build_global_policy depends on them); alembic head confirmed `f03e63aa5a33`; on branch `Thai-Postgre-FastAPI` (not master).
- **Design decisions:**
  - Kept `hitl_config` dependency-light per the plan (no `app.ai.utils` import) — `_tool_call_name` inlined to read normalized dicts/objects, avoiding a circular import.
  - `identity_requires_approval` checks `qualified_tool_id` before bare `name` in the `tools` map, so a `"<server>::<tool>"` rule wins over a bare-name rule, matching the precedence ladder verbatim.
  - `any_call_requires_approval` short-circuits the master kill-switch up front (before iterating calls) in addition to the per-identity check, so a master-off policy is O(1).

### Task 2 — ToolApprovalSetting model + migration + registration ✅ 2026-06-22

- **Tests:** `tests/test_tool_approval_setting_model.py` (3 cases: columns/constraints, `__all__` export, migration chains from head). Step-2 verify-fail confirmed `ModuleNotFoundError: No module named 'app.models.tool_approval_setting'`.
- **Implementation:** New `app/models/tool_approval_setting.py` (`ToolApprovalSetting`, table `tool_approval_settings`, unique `(user_id, scope_type, scope_value)`, check `scope_type IN ('server','tool')`); registered in `app/models/__init__.py` (import + `__all__`); migration `g0h1i2j3k4l5` revising `f03e63aa5a33`.
- **Verification:** `pytest …model.py` → **3 passed**; `import app.models; import app.main` → no ImportError; `alembic heads` → exactly one head `g0h1i2j3k4l5 (head)` (no branch).
- **Commit:** `feat(hitl): ToolApprovalSetting model + migration` (4 files, +171).
- **Design decisions:**
  - Named the model `ToolApprovalSetting` to sit alongside the pre-existing `ToolApproval` (the per-call decision record) without collision — distinct table, distinct concept (policy vs. decision).
  - `scope_value` sized `String(512)` to comfortably hold qualified ids `"<server>::<tool>"`; indexed for the per-user policy read.

### Task 3 — ToolApprovalSettingRepository (session-factory, sync) ✅ 2026-06-22

- **Tests:** `tests/test_tool_approval_setting_repository.py` (4 cases: create-when-missing, update-in-place, build_policy grouping + garbage-scope drop, invalid scope_type rejection) using the repo's hand-rolled `_FakeSession`/`@contextmanager` factory (no real DB). Step-2 verify-fail confirmed `ModuleNotFoundError`.
- **Implementation:** `app/repositories/tool_approval_setting.py` — `ToolApprovalSettingRepository(session_factory)` with `list_by_user`, `set` (upsert), `bulk_set`, `delete`, `build_policy`. `set`/`delete` validate scope via `_validate_scope`.
- **Verification:** `pytest …repository.py -q` → **4 passed**.
- **Commit:** `feat(hitl): ToolApprovalSettingRepository + build_policy` (2 files, +200).
- **Design decisions:**
  - `build_policy` silently ignores unknown `scope_type` rows (defensive against future scope kinds / dirty data) rather than raising — the read path must never break a turn.
  - `set` upserts (select-then-update-or-insert) keyed on the `(user_id, scope_type, scope_value)` unique tuple, matching the DB constraint so concurrent writers converge on update rather than violating the constraint.

### Task 4 — HitlSettingsService + DI wiring + /hitl/settings API ✅ 2026-06-22

- **Tests:** `tests/test_hitl_api.py` (3 cases: POST→GET roundtrip of server+tool rules, DELETE clears a rule, invalid scope_type → 422). Step-2 verify-fail confirmed `ModuleNotFoundError: No module named 'app.api.hitl'`.
- **Implementation:** `app/schemas/hitl.py` (camel schemas), `app/services/hitl_settings_service.py` (`get_settings`/`apply`/`clear`/`build_turn_policy`), `app/api/hitl.py` (`GET/POST/DELETE /hitl/settings`). Container: repo provider `tool_approval_setting_repository` + service provider `hitl_settings_service` (NOT injected into `message_service` yet — deferred to Task 6 so every commit boots). DI: `HitlSettingsService` added to `AppAutoInjector.wiring_map`. Router exported in `app/api/__init__.py` and mounted in `app/main.py`.
- **Verification:** `import app.main` → BOOT OK; `pytest tests/test_hitl_api.py -q` → **3 passed** against the live dev Postgres (DB reachable in this environment; the `tool_approval_settings` table exists, exercised by real insert/select/delete).
- **Commit:** `feat(hitl): per-user settings service + /hitl/settings API` (8 files, +231).
- **Design decisions:**
  - **Camel base divergence from the plan's literal code:** the plan's `app/schemas/hitl.py` snippet imported `pydantic.alias_generators.to_camel`. The codebase convention (12 schema modules) is a *local* `_CamelModel` built on the shared `app.utils.case_conversion.to_camel_case` helper. I followed the codebase convention (reuse the shared helper) instead of introducing pydantic's generator — same camelCase output, consistent with every other schema module. The API test asserts the exact camelCase keys (`scopeType`/`scopeValue`/`requireApproval`/`masterEnabled`), which pass.
  - 422 on invalid `scope_type` comes for free from the `Literal["server","tool"]` field on `HitlScopeRule` (Pydantic validation at the request boundary), so the bad value never reaches the repository.
  - Service holds the `is_hitl_enabled()`/`get_tools_requiring_approval()` reads so the master switch + legacy floor are always reported from live settings, not persisted per-user.

### Task 5 — Wire the policy into all five graph gates ✅ 2026-06-22

- **Tests:** `tests/test_hitl_gate_policy.py` (4 cases: server-scope gates via `_should_call_tools`, tool-override exempts a server tool, master-off never gates, generic worker reads `parent_state` policy). Step-2 verify-fail confirmed `AttributeError: module 'app.ai.graph' has no attribute 'get_global_mcp_manager'`.
- **Implementation:** `app/ai/graph.py` — dropped `requires_human_approval` from the import (now `any_call_requires_approval`, `policy_from_context`), added `from .mcp_registry import get_global_mcp_manager`, added async `MultiAgentWorkflow._needs_approval(...)` helper above `_prepare_interrupt_payload`, made `_should_call_tools` async, and replaced all five `requires_human_approval(...)` sites (`_should_call_tools`, RAG gate, RAG sub-worker, generic worker, planning) with `await self._needs_approval(...)`. `tests/test_custom_agents_graph.py` test converted to `async def` (+`await`).
- **Verification:** `import app.ai.graph` → OK (no circular import from `mcp_registry`); `ruff check app/ai/graph.py` → All checks passed; `pytest test_hitl_gate_policy.py test_custom_agents_graph.py test_hitl_config.py test_client_tool_isolation.py -q` → **44 passed**.
- **Commit:** `feat(hitl): route all five gates through the per-user policy resolver` (3 files, +209/-40).
- **Design decisions:**
  - **Planning gate nested-`if` collapse (deviation from plan's literal code):** the plan kept `if external_tool_calls:` wrapping `if await self._needs_approval(...):`. Removing the old intervening `ext_tool_names = [...]` assignment made ruff flag a real **new** SIM102 (collapsible nested-if). I collapsed them to `if external_tool_calls and await self._needs_approval(...):` and dedented the block body — behavior-identical (short-circuit on empty calls; `approved_external_calls` is already captured above the block) and keeps the touched file ruff-clean (the plan's Task 10 requires fixing new findings).
  - Async conversion of `_should_call_tools` is safe: its only internal caller is the LangGraph conditional-edge registration (`self._should_call_tools` at graph.py:775), and LangGraph awaits async edge callables. `asyncio_mode = "auto"` (pyproject.toml) means the converted sync test needed only `async def`/`await`, no `@pytest.mark.asyncio` (which couldn't be used there anyway — `pytest` is imported lower in that file).
  - `_needs_approval` only builds a `tool_map`/fetches the MCP manager when `master_enabled` is true and a tool_map isn't already supplied, so the master kill-switch path stays cheap and the planning gate reuses its pre-built `tool_map`.

### Task 6 — Load per-user policy per turn + carry into graph context ✅ 2026-06-22

- **Tests:** `tests/test_hitl_turn_policy_injection.py` (3 cases: service request carries `hitl_policy`, `_build_initial_state_from_request` injects it into `context`, `AIService._to_ai_request` preserves it through the dual-schema round-trip). Step-3 verify-fail confirmed all 3 failed (field absent).
- **Implementation:** Added `hitl_policy: dict[str, Any] | None = None` to **both** `app/schemas/workflow.py::WorkflowExecutionRequest` and `app/ai/schemas.py::WorkflowExecutionRequest`, and `hitl_policy: dict[str, Any]` to `app/ai/schemas.py::GraphContext`. `_build_initial_state_from_request` injects `"hitl_policy": getattr(request, "hitl_policy", None)` alongside `inline_rich_response_v1`. `MessageService.__init__` accepts `tool_approval_setting_repository=None`; new `_resolve_hitl_policy(user_id)` (mirrors `_resolve_custom_agents_state`) called in `_build_user_message_workflow_request` and passed as `hitl_policy=`. Container passes `tool_approval_setting_repository=` into the `MessageService` provider.
- **Verification:** `import app.main` → BOOT OK; `pytest test_hitl_turn_policy_injection.py test_custom_agents_message_service.py -q` → **11 passed**; `ruff check` on touched files → All checks passed (after fixing one I001).
- **Commit:** `feat(hitl): load per-user policy per turn and inject into graph context` (6 files, +75/-2).
- **Design decisions:**
  - **Verified the dual-schema drift trap (`[[dual-workflow-request-schema-drift]]`):** `_to_ai_request` does `AIWorkflowExecutionRequest.model_validate(request.model_dump())`, which silently drops any field absent from the AI-layer schema. Test 3 is the guard — it passes only because the field was added to BOTH schemas, not just the service one.
  - **Fixed an I001 introduced in Task 4:** the `ToolApprovalSettingRepository` import was placed after `custom_agent` instead of after `tool_approval` (alphabetical), tripping ruff's import-sort. Task 4's plan steps had no ruff gate so it slipped through; corrected here via ruff's safe autofix (import reorder only — both symbols still imported, boot verified).
  - `_resolve_hitl_policy` returns `None` (not an empty policy) when no repo/user, so `policy_from_context` falls back to the global policy and legacy/test `MessageService` constructions (which omit the new param) keep working unchanged.
