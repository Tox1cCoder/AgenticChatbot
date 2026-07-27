"""Tests for the widget runtime service — stores, tokens, metadata extraction."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, get_type_hints

import jwt as pyjwt
import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from app.ai.tool_execution import build_tool_artifact, execute_tool_calls
from app.core.response_constants import build_bot_metadata, extract_live_widgets_from_artifacts
from app.services.widget_runtime import (
    MAX_WIDGET_STATE_BYTES,
    InMemoryWidgetStore,
    WidgetConnectionManager,
    WidgetRecord,
    WidgetStatus,
    WidgetTokenService,
    _validate_state_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def store():
    return InMemoryWidgetStore()


@pytest.fixture()
def token_service():
    return WidgetTokenService(secret="test-secret-key-long-enough-32bytes!")


# ---------------------------------------------------------------------------
# InMemoryWidgetStore — CRUD
# ---------------------------------------------------------------------------
class TestInMemoryWidgetStoreCRUD:
    async def test_create_returns_record(self, store):
        record = await store.create("session-1", {"cols": ["a", "b"]}, title="T")
        assert isinstance(record, WidgetRecord)
        assert record.session_id == "session-1"
        assert not hasattr(record, "widget_type")
        assert record.title == "T"
        assert record.state == {"cols": ["a", "b"]}
        assert record.status == WidgetStatus.ACTIVE
        assert record.version == 1

    async def test_get_returns_created_widget(self, store):
        created = await store.create("s1", {})
        fetched = await store.get(created.widget_id)
        assert fetched is not None
        assert fetched.widget_id == created.widget_id

    async def test_get_returns_none_for_missing(self, store):
        assert await store.get("nonexistent") is None

    async def test_update_replaces_state_and_increments_version(self, store):
        created = await store.create("s1", {"v": 1})
        updated = await store.update(created.widget_id, {"v": 2})
        assert updated.state == {"v": 2}
        assert updated.version == 2

    async def test_update_with_correct_expected_version(self, store):
        created = await store.create("s1", {})
        updated = await store.update(created.widget_id, {"x": 1}, expected_version=1)
        assert updated.version == 2

    async def test_update_with_wrong_expected_version_raises(self, store):
        created = await store.create("s1", {})
        with pytest.raises(ValueError, match="Version mismatch"):
            await store.update(created.widget_id, {}, expected_version=999)

    async def test_update_closed_widget_raises(self, store):
        created = await store.create("s1", {})
        await store.close(created.widget_id)
        with pytest.raises(ValueError, match="closed"):
            await store.update(created.widget_id, {})

    async def test_update_missing_widget_raises(self, store):
        with pytest.raises(KeyError):
            await store.update("missing", {})

    async def test_restore_recreates_widget_with_same_id(self, store):
        restored = await store.restore(
            widget_id="w-restored",
            session_id="s-restored",
            state={"labels": ["A"], "datasets": [{"data": [1]}]},
            title="Recovered Widget",
            status="active",
            version=4,
        )

        fetched = await store.get("w-restored")

        assert restored.widget_id == "w-restored"
        assert restored.version == 4
        assert fetched is not None
        assert fetched.widget_id == "w-restored"
        assert fetched.state["labels"] == ["A"]


# ---------------------------------------------------------------------------
# InMemoryWidgetStore — Patch (shallow merge)
# ---------------------------------------------------------------------------
class TestInMemoryWidgetStorePatch:
    async def test_patch_merges_shallow(self, store):
        created = await store.create("s1", {"a": 1, "b": 2})
        patched = await store.patch(created.widget_id, {"b": 99, "c": 3})
        assert patched.state == {"a": 1, "b": 99, "c": 3}
        assert patched.version == 2

    async def test_patch_closed_widget_raises(self, store):
        created = await store.create("s1", {})
        await store.close(created.widget_id)
        with pytest.raises(ValueError, match="closed"):
            await store.patch(created.widget_id, {"x": 1})

    async def test_patch_missing_widget_raises(self, store):
        with pytest.raises(KeyError):
            await store.patch("missing", {})


# ---------------------------------------------------------------------------
# InMemoryWidgetStore — Close
# ---------------------------------------------------------------------------
class TestInMemoryWidgetStoreClose:
    async def test_close_sets_status(self, store):
        created = await store.create("s1", {})
        closed = await store.close(created.widget_id)
        assert closed.status == WidgetStatus.CLOSED

    async def test_close_missing_raises(self, store):
        with pytest.raises(KeyError):
            await store.close("missing")


# ---------------------------------------------------------------------------
# InMemoryWidgetStore — list_by_session
# ---------------------------------------------------------------------------
class TestInMemoryWidgetStoreListBySession:
    async def test_list_returns_session_widgets(self, store):
        await store.create("s1", {})
        await store.create("s1", {})
        await store.create("s2", {})
        widgets = await store.list_by_session("s1")
        assert len(widgets) == 2

    async def test_list_empty_session(self, store):
        assert await store.list_by_session("no-such-session") == []


# ---------------------------------------------------------------------------
# InMemoryWidgetStore — TTL eviction
# ---------------------------------------------------------------------------
class TestInMemoryWidgetStoreTTL:
    async def test_expired_widget_returns_none(self, store):
        created = await store.create("s1", {}, ttl_seconds=0)
        await asyncio.sleep(0.01)
        assert await store.get(created.widget_id) is None


# ---------------------------------------------------------------------------
# State size validation
# ---------------------------------------------------------------------------
class TestStateSizeValidation:
    def test_valid_size_passes(self):
        _validate_state_size({"key": "value"})

    def test_oversized_state_raises(self):
        huge_state = {"data": "x" * (MAX_WIDGET_STATE_BYTES + 1)}
        with pytest.raises(ValueError, match="exceeds maximum size"):
            _validate_state_size(huge_state)

    async def test_create_with_oversized_state_raises(self):
        store = InMemoryWidgetStore()
        huge = {"data": "x" * (MAX_WIDGET_STATE_BYTES + 1)}
        with pytest.raises(ValueError, match="exceeds maximum size"):
            await store.create("s1", huge)


# ---------------------------------------------------------------------------
# WidgetRecord — metadata conversion
# ---------------------------------------------------------------------------
class TestWidgetRecordMetadata:
    def test_to_live_widget_metadata(self):
        record = WidgetRecord(
            widget_id="w-123",
            session_id="s-456",
            title="My Table",
            state={"rows": []},
            status=WidgetStatus.ACTIVE,
            version=3,
            created_at=1000.0,
            updated_at=1001.0,
            expires_at=2000.0,
        )
        meta = record.to_live_widget_metadata()
        assert meta["widget_id"] == "w-123"
        assert meta["session_id"] == "s-456"
        assert "widget_type" not in meta
        assert meta["title"] == "My Table"
        assert meta["status"] == "active"
        assert meta["version"] == 3
        assert meta["connection_endpoint"] == "/widgets/w-123/connection"
        assert "state" not in meta

    def test_to_dict_includes_state(self):
        record = WidgetRecord(
            widget_id="w-1",
            session_id="s-1",
            title=None,
            state={"data": [1, 2, 3]},
            status=WidgetStatus.ACTIVE,
            version=1,
            created_at=1000.0,
            updated_at=1000.0,
            expires_at=2000.0,
        )
        d = record.to_dict()
        assert d["state"] == {"data": [1, 2, 3]}
        assert d["status"] == "active"


# ---------------------------------------------------------------------------
# WidgetTokenService
# ---------------------------------------------------------------------------
class TestWidgetTokenService:
    def test_mint_and_verify(self, token_service):
        token, expires_at = token_service.mint(widget_id="w-1", session_id="s-1", user_id="u-1")
        claims = token_service.verify(token)
        assert claims["wid"] == "w-1"
        assert claims["sid"] == "s-1"
        assert claims["sub"] == "u-1"
        assert claims["type"] == "widget"

    def test_expired_token_raises(self, token_service):
        token, _ = token_service.mint(
            widget_id="w-1", session_id="s-1", user_id="u-1", ttl_seconds=0
        )
        time.sleep(0.1)
        with pytest.raises(pyjwt.ExpiredSignatureError):
            token_service.verify(token)

    def test_wrong_secret_raises(self, token_service):
        token, _ = token_service.mint(widget_id="w-1", session_id="s-1", user_id="u-1")
        other = WidgetTokenService(secret="different-secret-key-long-enough!")
        with pytest.raises(pyjwt.InvalidSignatureError):
            other.verify(token)

    def test_non_widget_token_rejected(self, token_service):
        payload = {"sub": "u-1", "wid": "w-1", "sid": "s-1", "type": "access"}
        raw = pyjwt.encode(payload, "test-secret-key-long-enough-32bytes!", algorithm="HS256")
        with pytest.raises(pyjwt.InvalidTokenError, match="Not a widget token"):
            token_service.verify(raw)


# ---------------------------------------------------------------------------
# WidgetConnectionManager — unit tests
# ---------------------------------------------------------------------------
class TestWidgetConnectionManager:
    def test_connection_count_starts_at_zero(self):
        mgr = WidgetConnectionManager()
        assert mgr.connection_count("w-1") == 0


# ---------------------------------------------------------------------------
# extract_live_widgets_from_artifacts
# ---------------------------------------------------------------------------
class TestExtractLiveWidgets:
    def test_extracts_from_widget_create_artifact(self):
        artifacts = [
            {
                "tool_call_id": "tc-1",
                "tool": "widget_create",
                "args": {},
                "output": json.dumps(
                    {
                        "widget_id": "w-abc",
                        "session_id": "s-1",
                        "widget_type": "table",
                        "title": "Result Table",
                        "status": "active",
                        "version": 1,
                    }
                ),
                "error": None,
                "status": "success",
            }
        ]
        widgets = extract_live_widgets_from_artifacts(artifacts)
        assert len(widgets) == 1
        assert widgets[0]["widget_id"] == "w-abc"
        assert widgets[0]["connection_endpoint"] == "/widgets/w-abc/connection"

    def test_ignores_non_widget_tools(self):
        artifacts = [
            {
                "tool": "tavily_search",
                "output": json.dumps({"widget_id": "fake"}),
                "status": "success",
            }
        ]
        assert extract_live_widgets_from_artifacts(artifacts) == []

    def test_ignores_error_artifacts(self):
        artifacts = [
            {
                "tool": "widget_create",
                "output": None,
                "error": "boom",
                "status": "error",
            }
        ]
        assert extract_live_widgets_from_artifacts(artifacts) == []

    def test_deduplicates_by_widget_id(self):
        base = {
            "widget_id": "w-1",
            "session_id": "s-1",
            "widget_type": "table",
            "status": "active",
            "version": 1,
        }
        artifacts = [
            {"tool": "widget_create", "output": json.dumps(base), "status": "success"},
            {
                "tool": "widget_update",
                "output": json.dumps({**base, "version": 2}),
                "status": "success",
            },
        ]
        widgets = extract_live_widgets_from_artifacts(artifacts)
        assert len(widgets) == 1

    def test_handles_none_and_empty(self):
        assert extract_live_widgets_from_artifacts(None) == []
        assert extract_live_widgets_from_artifacts([]) == []

    def test_handles_malformed_output(self):
        artifacts = [
            {"tool": "widget_create", "output": "not json", "status": "success"},
        ]
        assert extract_live_widgets_from_artifacts(artifacts) == []

    def test_treats_missing_status_as_success_for_legacy_artifacts(self):
        artifacts = [
            {
                "tool": "widget_create",
                "output": json.dumps(
                    {
                        "widget_id": "w-legacy",
                        "session_id": "s-1",
                        "widget_type": "table",
                        "title": "Legacy Widget",
                        "status": "active",
                        "version": 1,
                    }
                ),
            }
        ]
        widgets = extract_live_widgets_from_artifacts(artifacts)
        assert len(widgets) == 1
        assert widgets[0]["widget_id"] == "w-legacy"

    def test_supports_legacy_tool_name_and_tool_output_keys(self):
        artifacts = [
            {
                "tool_name": "widget_update",
                "tool_output": json.dumps(
                    {
                        "widget_id": "w-legacy-shape",
                        "session_id": "s-1",
                        "widget_type": "chart",
                        "title": "Legacy Shape",
                        "status": "active",
                        "version": 2,
                    }
                ),
            }
        ]
        widgets = extract_live_widgets_from_artifacts(artifacts)
        assert len(widgets) == 1
        assert widgets[0]["widget_id"] == "w-legacy-shape"
        assert widgets[0]["version"] == 2


# ---------------------------------------------------------------------------
# build_bot_metadata — live_widgets integration
# ---------------------------------------------------------------------------
class TestBuildBotMetadataWidgets:
    def test_includes_live_widgets_when_present(self):
        class FakeResponse:
            metadata = {}
            tool_artifacts = [
                {
                    "tool": "widget_create",
                    "output": json.dumps(
                        {
                            "widget_id": "w-1",
                            "session_id": "s-1",
                            "widget_type": "chart",
                            "title": "Chart",
                            "status": "active",
                            "version": 1,
                        }
                    ),
                    "status": "success",
                }
            ]

        metadata = build_bot_metadata(FakeResponse())
        assert "live_widgets" in metadata
        assert len(metadata["live_widgets"]) == 1
        assert metadata["live_widgets"][0]["widget_id"] == "w-1"

    def test_no_live_widgets_without_widget_artifacts(self):
        class FakeResponse:
            metadata = {}
            tool_artifacts = [{"tool": "add", "output": "5", "status": "success"}]

        metadata = build_bot_metadata(FakeResponse())
        assert "live_widgets" not in metadata

    def test_no_live_widgets_with_none_response(self):
        metadata = build_bot_metadata(None)
        assert "live_widgets" not in metadata

    def test_merges_response_tool_artifacts_with_metadata_tool_artifacts(self):
        class FakeResponse:
            metadata = {"tool_artifacts": [{"tool": "add", "output": "5", "status": "success"}]}
            tool_artifacts = [
                {
                    "tool": "widget_create",
                    "output": json.dumps(
                        {
                            "widget_id": "w-merged",
                            "session_id": "s-1",
                            "widget_type": "table",
                            "title": "Merged Widget",
                            "status": "active",
                            "version": 1,
                        }
                    ),
                    "status": "success",
                }
            ]

        metadata = build_bot_metadata(FakeResponse())
        assert "live_widgets" in metadata
        assert metadata["live_widgets"][0]["widget_id"] == "w-merged"

    def test_preserves_render_payload_in_tool_artifacts(self):
        render = {
            "version": 1,
            "type": "mcp_app",
            "template_uri": "ui://canva/presentation-viewer.html",
        }

        class FakeResponse:
            metadata = {}
            tool_artifacts = [
                {
                    "tool": "canva_create_presentation",
                    "output": "Created presentation",
                    "status": "success",
                    "render": render,
                }
            ]

        metadata = build_bot_metadata(FakeResponse())

        assert metadata["tool_artifacts"][0]["render"] == render
        assert "live_widgets" not in metadata


# ---------------------------------------------------------------------------
# build_tool_artifact — widget output compaction
# ---------------------------------------------------------------------------
class TestBuildToolArtifactWidgets:
    def test_widget_artifact_output_remains_parseable(self):
        raw_output = json.dumps(
            {
                "widget_id": "w-big",
                "session_id": "s-1",
                "widget_type": "table",
                "title": "Large Widget",
                "status": "active",
                "version": 1,
                "state": {"rows": [["x" * 5000]]},
            }
        )
        artifact = build_tool_artifact(
            tool_call_id="tc-1",
            tool_name="widget_create",
            tool_args={},
            output_text=raw_output,
            error=None,
        )

        parsed = json.loads(artifact["output"])
        assert parsed["widget_id"] == "w-big"
        assert parsed["version"] == 1
        assert "state" not in parsed

    def test_non_widget_artifact_preserves_full_output_by_default(self):
        artifact = build_tool_artifact(
            tool_call_id="tc-2",
            tool_name="some_other_tool",
            tool_args={},
            output_text="x" * 1200,
            error=None,
        )
        assert len(artifact["output"]) == 1200

    def test_tool_artifact_preserves_render_payload(self):
        render = {
            "version": 1,
            "type": "mcp_app",
            "template_uri": "ui://canva/presentation-viewer.html",
            "structured_content": {"presentation_id": "deck_123"},
        }

        artifact = build_tool_artifact(
            tool_call_id="tc-render",
            tool_name="canva_create_presentation",
            tool_args={"prompt": "roadmap"},
            output_text="Created presentation",
            error=None,
            render=render,
        )

        assert artifact["output"] == "Created presentation"
        assert artifact["render"] == render

    def test_error_tool_artifact_preserves_error_render_payload(self):
        render = {
            "version": 1,
            "type": "error",
            "model_content": "Error: permission denied",
            "error": "permission denied",
        }

        artifact = build_tool_artifact(
            tool_call_id="tc-error",
            tool_name="dangerous_tool",
            tool_args={},
            output_text="Error: permission denied",
            error="permission denied",
            render=render,
        )

        assert artifact["status"] == "error"
        assert artifact["render"]["type"] == "error"


# ---------------------------------------------------------------------------
# execute_tool_calls — widget session binding
# ---------------------------------------------------------------------------
class _CaptureTool:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return {"ok": True}


class TestWidgetToolExecutionBinding:
    async def test_widget_create_binds_session_id_to_active_conversation(self):
        tool = _CaptureTool("widget_create")
        conversation_id = "22222222-2222-2222-2222-222222222222"

        await execute_tool_calls(
            tool_calls=[
                {
                    "id": "tc-widget-create",
                    "name": "widget_create",
                    "args": {
                        "session_id": "current_session",
                        "initial_state": {"html": "<!doctype html><html></html>", "height": 320},
                    },
                }
            ],
            tool_map={"widget_create": tool},
            conversation_id=conversation_id,
        )

        assert tool.calls[0]["session_id"] == conversation_id

    async def test_session_list_widgets_binds_session_id_to_active_conversation(self):
        tool = _CaptureTool("session_list_widgets")
        conversation_id = "33333333-3333-3333-3333-333333333333"

        await execute_tool_calls(
            tool_calls=[
                {
                    "id": "tc-widget-list",
                    "name": "session_list_widgets",
                    "args": {"session_id": "stale-session"},
                }
            ],
            tool_map={"session_list_widgets": tool},
            conversation_id=conversation_id,
        )

        assert tool.calls[0]["session_id"] == conversation_id

    async def test_widget_create_keeps_original_session_without_conversation_context(self):
        tool = _CaptureTool("widget_create")

        await execute_tool_calls(
            tool_calls=[
                {
                    "id": "tc-widget-create-no-context",
                    "name": "widget_create",
                    "args": {
                        "session_id": "manual-session",
                        "initial_state": {"html": "<!doctype html><html></html>", "height": 320},
                    },
                }
            ],
            tool_map={"widget_create": tool},
            conversation_id=None,
        )

        assert tool.calls[0]["session_id"] == "manual-session"


# ---------------------------------------------------------------------------
# BaseAgent tool binding — widget session binding
# ---------------------------------------------------------------------------
class _WidgetCreateInput(BaseModel):
    session_id: str
    initial_state: dict[str, Any]


class TestWidgetToolBindingWrappers:
    async def test_bound_widget_tool_forces_active_conversation(self):
        from app.ai.agents.base_agent import _bind_widget_session_tools

        captured_calls: list[dict[str, object]] = []

        async def _widget_create(**kwargs):
            captured_calls.append(kwargs)
            return json.dumps({"ok": True})

        tool = StructuredTool.from_function(
            coroutine=_widget_create,
            name="widget_create",
            description="Create widget",
            args_schema=_WidgetCreateInput,
            infer_schema=False,
        )

        wrapped_tools = _bind_widget_session_tools(
            [tool],
            "44444444-4444-4444-4444-444444444444",
        )

        await wrapped_tools[0].ainvoke(
            {
                "session_id": "current_session",
                "initial_state": {"html": "<!doctype html><html></html>", "height": 320},
            }
        )

        assert captured_calls[0]["session_id"] == "44444444-4444-4444-4444-444444444444"

    async def test_non_widget_tools_are_not_wrapped(self):
        from app.ai.agents.base_agent import _bind_widget_session_tools

        async def _search_documents(query: str) -> str:
            return query

        tool = StructuredTool.from_function(
            coroutine=_search_documents,
            name="search_documents",
            description="Search documents",
        )

        wrapped_tools = _bind_widget_session_tools(
            [tool],
            "55555555-5555-5555-5555-555555555555",
        )

        assert wrapped_tools[0] is tool


# ---------------------------------------------------------------------------
# MCP server tool definitions
# ---------------------------------------------------------------------------
class TestWidgetsMCPServer:
    async def test_server_defines_expected_tools(self):
        from app.ai.mcp_servers.widgets_server import mcp

        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "widget_create",
            "widget_update",
            "widget_get_state",
            "widget_close",
            "session_list_widgets",
        }
        assert expected == tool_names


# ---------------------------------------------------------------------------
# Agent scoping — widget tool exclusion
# ---------------------------------------------------------------------------
class TestWidgetAgentScoping:
    def test_permanent_exclusion_is_reserved_for_non_visual_agents(self):
        from app.ai.agents.base_agent import _WIDGET_EXCLUDED_AGENT_KEYS

        assert "canvas" not in _WIDGET_EXCLUDED_AGENT_KEYS
        assert "image_generator" in _WIDGET_EXCLUDED_AGENT_KEYS
        assert "planning" in _WIDGET_EXCLUDED_AGENT_KEYS

    def test_target_agents_not_excluded(self):
        from app.ai.agents.base_agent import _WIDGET_EXCLUDED_AGENT_KEYS

        assert "chat" not in _WIDGET_EXCLUDED_AGENT_KEYS
        assert "rag" not in _WIDGET_EXCLUDED_AGENT_KEYS
        assert "search" not in _WIDGET_EXCLUDED_AGENT_KEYS

    def test_target_agent_allowlist_keeps_widgets_access(self, monkeypatch):
        from app.ai.agents.base_agent import _get_effective_tool_allowlist
        from app.core.config import settings

        monkeypatch.setattr(settings, "chat_agent_allowed_tools", ["tavily"], raising=False)

        allowlist = _get_effective_tool_allowlist("chat")

        assert "tavily" in allowlist
        assert "widgets" in allowlist

    def test_non_target_agent_allowlist_not_modified(self, monkeypatch):
        from app.ai.agents.base_agent import _get_effective_tool_allowlist
        from app.core.config import settings

        monkeypatch.setattr(settings, "planning_agent_allowed_tools", ["tavily"], raising=False)

        assert _get_effective_tool_allowlist("planning") == ["tavily"]

    def test_canvas_edit_deferred_binding_omits_widget_mutations(self):
        from app.ai.canvas_state import CANVAS_EDIT_DENIED_TOOL_NAMES
        from app.ai.deferred_tool_binding import build_deferred_tool_list

        async def _noop(**kwargs):
            return kwargs

        tools = [
            StructuredTool.from_function(
                coroutine=_noop,
                name=name,
                description=name,
            )
            for name in ("widget_create", "widget_update", "widget_get_state")
        ]

        bound = build_deferred_tool_list(
            conversation_id=None,
            agent_key="canvas",
            mcp_manager=None,
            all_mcp_tools=[],
            internal_tools=tools,
            excluded_tool_names=CANVAS_EDIT_DENIED_TOOL_NAMES,
        )

        names = {tool.name for tool in bound}
        assert "widget_create" not in names
        assert "widget_update" not in names
        assert "widget_get_state" in names
        assert "tool_search" in names

    def test_canvas_edit_omits_loaded_client_widget_mutations(self, monkeypatch):
        from app.ai.agents.base_agent import BaseAgent
        from app.ai.canvas_state import CANVAS_EDIT_DENIED_TOOL_NAMES
        from app.ai.schemas import AgentType

        class _CanvasAgent(BaseAgent):
            def _init_gemini(self) -> None:
                self.gemini_client = None
                self.langchain_model = None

            @property
            def agent_type(self) -> AgentType:
                return AgentType.CANVAS

            @property
            def agent_id(self) -> str:
                return "canvas_agent"

            def _get_base_system_prompt(self) -> str:
                return "Canvas"

        async def _noop(**kwargs):
            return kwargs

        client_tools = [
            StructuredTool.from_function(
                coroutine=_noop,
                name=name,
                description=name,
            )
            for name in ("widget_create", "widget_get_state")
        ]
        agent = _CanvasAgent(agent_config_key="canvas")
        monkeypatch.setattr(
            "app.ai.agents.base_agent.should_use_deferred_loading",
            lambda _agent_key: False,
        )
        monkeypatch.setattr(agent, "_get_client_runtime_tools", lambda **_kwargs: client_tools)

        names = {
            tool.name
            for tool in agent._get_tools_for_binding(
                conversation_id="conversation-1",
                user_id="user-1",
                device_id="device-1",
                excluded_tool_names=CANVAS_EDIT_DENIED_TOOL_NAMES,
            )
        }

        assert "widget_create" not in names
        assert "widget_get_state" in names


# ---------------------------------------------------------------------------
# Deferred binding — pinned widget tools
# ---------------------------------------------------------------------------
class TestDeferredWidgetBinding:
    def test_chat_agent_gets_widget_tools_pinned(self, monkeypatch):
        from app.ai.deferred_tool_binding import _get_pinned_specs
        from app.core.config import settings

        monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

        pinned_specs = _get_pinned_specs("chat")

        assert "widgets::widget_create" in pinned_specs
        assert "widgets::widget_update" in pinned_specs
        assert "widgets::widget_get_state" in pinned_specs

    def test_non_target_agent_does_not_auto_pin_widgets(self, monkeypatch):
        from app.ai.deferred_tool_binding import _get_pinned_specs
        from app.core.config import settings

        monkeypatch.setattr(settings, "mcp_tool_search_pinned_tools", [], raising=False)

        assert _get_pinned_specs("canvas") == []


# ---------------------------------------------------------------------------
# Prompt updates verification
# ---------------------------------------------------------------------------
class TestPromptUpdates:
    def test_chat_prompt_mentions_live_widgets_when_helpful(self):
        from app.ai.prompts import CHAT_SYSTEM_PROMPT

        assert "live widget" in CHAT_SYSTEM_PROMPT.lower()

    def test_router_prompt_routes_visual_aids_to_chat(self):
        from app.ai.prompts import ROUTER_SYSTEM_PROMPT

        assert "LiveUI widgets are for compact in-chat aids" in ROUTER_SYSTEM_PROMPT
        assert "Do not route to canvas_agent merely because a widget" in ROUTER_SYSTEM_PROMPT

    def test_prompt_mentions_html_micro_app_widgets(self):
        from app.ai.prompts import CHAT_SYSTEM_PROMPT

        prompt = CHAT_SYSTEM_PROMPT.lower()
        assert "html" in prompt
        assert "micro-app" in prompt
        assert "slider" in prompt
        assert "animation" in prompt

    def test_prompt_drops_structured_widget_guidance(self):
        from app.ai.prompts import CHAT_SYSTEM_PROMPT

        prompt = CHAT_SYSTEM_PROMPT.lower()
        assert "chart_type" not in prompt
        assert "dashboard" not in prompt
        assert "presentation block" not in prompt


_VALID_HTML_STATE = {
    "html": "<!doctype html><div>hi</div>",
    "height": 540,
    "caption": "demo",
}


# ---------------------------------------------------------------------------
# Widget tool HTML-only contract enforcement
# ---------------------------------------------------------------------------
class TestWidgetToolHtmlContract:
    def test_widget_create_schema_has_no_widget_type(self):
        import inspect

        from app.ai.mcp_servers import widgets_server

        parameters = inspect.signature(widgets_server.widget_create).parameters
        assert list(parameters) == ["session_id", "initial_state", "title"]
        assert get_type_hints(widgets_server.widget_create)["initial_state"] == dict[str, Any]

    async def test_widget_create_accepts_quote_heavy_native_object(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)
        html = (
            "<!doctype html><script>const state = "
            + json.dumps({"label": 'He said "hello"', "rows": list(range(500))})
            + ";</script>"
        )
        result = await widgets_server.widget_create(
            session_id="conv-native",
            initial_state={"html": html, "height": 620},
        )
        assert json.loads(result)["state"]["html"] == html

    async def test_widget_create_rejects_serialized_state(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)

        with pytest.raises(ValueError, match="initial_state must be an object"):
            await widgets_server.widget_create(
                session_id="conv-serialized",
                initial_state=json.dumps(_VALID_HTML_STATE),
            )

        assert await store.list_by_session("conv-serialized") == []

    async def test_widget_create_accepts_valid_html(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)

        result = await widgets_server.widget_create(
            session_id="conv-2",
            initial_state=_VALID_HTML_STATE,
            title="Oscillator",
        )

        payload = json.loads(result)
        assert "widget_type" not in payload
        assert payload["status"] == "active"
        assert "quality_guidance" not in payload

    @pytest.mark.parametrize(
        "bad_state,match",
        [
            ("just-a-string", "object"),
            ({"html": "", "height": 540}, "html content"),
            ({"html": "<div>hi</div>"}, "height"),
            ({"html": "<div>hi</div>", "height": "tall"}, "numeric"),
            ({"html": "<div>hi</div>", "height": 50}, "between"),
            ({"html": "<div>hi</div>", "height": 5000}, "between"),
        ],
    )
    async def test_widget_create_rejects_invalid_html_state(self, monkeypatch, bad_state, match):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)

        with pytest.raises(ValueError, match=match):
            await widgets_server.widget_create(
                session_id="conv-bad",
                initial_state=bad_state,
            )

        assert await store.list_by_session("conv-bad") == []

    async def test_widget_create_rejects_state_field_aliases(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)

        # `document` and `min_height` are former aliases that the new contract
        # must not accept in place of `html`/`height`.
        with pytest.raises(ValueError, match="html content"):
            await widgets_server.widget_create(
                session_id="conv-alias",
                initial_state={"document": "<div>hi</div>", "min_height": 540},
            )

    async def test_widget_update_rejects_invalid_html_state(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)

        created = await widgets_server.widget_create(
            session_id="conv-3",
            initial_state=_VALID_HTML_STATE,
        )
        widget_id = json.loads(created)["widget_id"]

        with pytest.raises(ValueError, match="html content"):
            await widgets_server.widget_update(
                widget_id=widget_id,
                state={"html": "", "height": 540},
            )

    async def test_widget_update_rejects_serialized_state(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)
        created = await widgets_server.widget_create(
            session_id="conv-update-serialized",
            initial_state=_VALID_HTML_STATE,
        )
        widget_id = json.loads(created)["widget_id"]

        with pytest.raises(ValueError, match="state must be an object"):
            await widgets_server.widget_update(
                widget_id=widget_id,
                state=json.dumps(_VALID_HTML_STATE),
            )

    async def test_widget_update_preserves_widget_identity(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)

        created = await widgets_server.widget_create(
            session_id="conv-4",
            initial_state=_VALID_HTML_STATE,
        )
        widget_id = json.loads(created)["widget_id"]

        result = await widgets_server.widget_update(
            widget_id=widget_id,
            state={**_VALID_HTML_STATE, "caption": "updated"},
        )
        payload = json.loads(result)
        assert "widget_type" not in payload
        assert payload["version"] == 2
        assert "quality_guidance" not in payload

    async def test_widget_update_can_replace_legacy_state_with_valid_html(self, monkeypatch):
        import app.services.widget_runtime as widget_runtime
        from app.ai.mcp_servers import widgets_server

        store = InMemoryWidgetStore()
        monkeypatch.setattr(widget_runtime, "_widget_store", store)

        legacy = await store.create("conv-5", {"labels": ["A", "B"]})

        result = await widgets_server.widget_update(
            widget_id=legacy.widget_id,
            state=_VALID_HTML_STATE,
        )
        assert json.loads(result)["state"] == _VALID_HTML_STATE


# ---------------------------------------------------------------------------
# MCP config
# ---------------------------------------------------------------------------
class TestMCPConfig:
    def test_widgets_in_config(self):
        with open("app/ai/mcp_config.json") as f:
            config = json.load(f)
        assert "widgets" in config["servers"]
        assert config["servers"]["widgets"]["enabledByDefault"] is True

    def test_widgets_in_default_servers(self):
        from app.ai.mcp_integration import MCPManager

        assert "widgets" in MCPManager.DEFAULT_SERVERS
