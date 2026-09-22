"""
Lifecycle and staleness tests for deferred tool state.

Covers the long-lived behaviour of app/ai/deferred_tool_state.py:
- idle TTL measured from last *use*, not from load time
- abandoned scopes swept, and a hard ceiling on tracked scopes
- clear_conversation dropping both server and client scopes
- ownership-validated session resolution (no unvalidated fallback)
- snapshot/restore honouring the staleness it records
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ai.client_tool_catalog import ClientToolReference
from app.ai.deferred_tool_state import (
    _SWEEP_MIN_INTERVAL_SECONDS,
    RUNTIME_ID,
    ClientToolScope,
    ConversationToolSet,
    DeferredToolState,
    _now,
)
from app.ai.mcp_tool_catalog import ToolReference

CONV = "conv-lifecycle"
AGENT = "chat"


def _server_ref(name: str = "search", server: str = "brave") -> ToolReference:
    return ToolReference(tool_name=name, server_name=server, call_name=f"{server}__{name}")


def _client_ref(name: str = "client__t1", device_id: str = "dev-a") -> ClientToolReference:
    return ClientToolReference(
        tool_name=name,
        server_name="native",
        device_id=device_id,
        session_id="sess-a",
        catalog_version=3,
        tool_instance_id=f"inst-{name}",
    )


def _age_tools(scope, minutes: float) -> None:
    """Backdate every tool's last use by ``minutes``."""
    for tool in scope.loaded.values():
        tool.last_used = _now() - minutes * 60.0


# ---------------------------------------------------------------------------
# 1. Idle TTL is measured from last use
# ---------------------------------------------------------------------------


class TestIdleTTL:
    def test_tool_in_active_use_survives_past_its_load_time(self, monkeypatch):
        """A tool used every turn must not expire just because it was loaded
        long ago - that is what mark_tool_used exists to prevent."""
        monkeypatch.setattr(
            "app.core.config.settings.mcp_tool_search_loaded_tools_ttl_minutes", 30
        )
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])

        tool_set = state._conversation_tools[(CONV, AGENT)]
        tool = tool_set.loaded["brave__search"]
        tool.loaded_at = _now() - 120 * 60.0  # loaded two hours ago
        tool.last_used = _now() - 60.0  # used a minute ago

        assert state.get_loaded(CONV, AGENT) != []

    def test_unused_tool_expires_after_the_idle_ttl(self, monkeypatch):
        monkeypatch.setattr(
            "app.core.config.settings.mcp_tool_search_loaded_tools_ttl_minutes", 30
        )
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])
        _age_tools(state._conversation_tools[(CONV, AGENT)], minutes=45)

        assert state.get_loaded(CONV, AGENT) == []

    def test_marking_a_tool_used_resets_the_idle_timer(self, monkeypatch):
        monkeypatch.setattr(
            "app.core.config.settings.mcp_tool_search_loaded_tools_ttl_minutes", 30
        )
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])
        _age_tools(state._conversation_tools[(CONV, AGENT)], minutes=29)

        assert state.mark_tool_used(CONV, AGENT, "brave__search") is True

        _age_tools(state._conversation_tools[(CONV, AGENT)], minutes=5)
        assert [ref.call_name for ref in state.get_loaded(CONV, AGENT)] == ["brave__search"]

    def test_ttl_of_zero_disables_expiry(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.mcp_tool_search_loaded_tools_ttl_minutes", 0)
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])
        _age_tools(state._conversation_tools[(CONV, AGENT)], minutes=10_000)

        assert state.get_loaded(CONV, AGENT) != []


# ---------------------------------------------------------------------------
# 2. Scope tables stay bounded
# ---------------------------------------------------------------------------


class TestScopeBounds:
    def test_abandoned_scopes_are_swept(self, monkeypatch):
        """An emptied scope must not outlive its tools; otherwise every
        conversation ever seen keeps an entry for the life of the process."""
        monkeypatch.setattr(
            "app.core.config.settings.mcp_tool_search_loaded_tools_ttl_minutes", 30
        )
        state = DeferredToolState()
        for index in range(5):
            state.autoload(f"conv-{index}", AGENT, [_server_ref()])
            _age_tools(state._conversation_tools[(f"conv-{index}", AGENT)], minutes=90)
            state._conversation_tools[(f"conv-{index}", AGENT)].last_active = _now() - 90 * 60.0

        assert len(state._conversation_tools) == 5

        # Make the next write eligible to sweep.
        state._last_sweep = _now() - _SWEEP_MIN_INTERVAL_SECONDS - 1
        state.autoload("conv-live", AGENT, [_server_ref()])

        assert list(state._conversation_tools) == [("conv-live", AGENT)]

    def test_active_scopes_survive_the_sweep(self, monkeypatch):
        monkeypatch.setattr(
            "app.core.config.settings.mcp_tool_search_loaded_tools_ttl_minutes", 30
        )
        state = DeferredToolState()
        state.autoload("conv-busy", AGENT, [_server_ref()])

        state._last_sweep = _now() - _SWEEP_MIN_INTERVAL_SECONDS - 1
        state.autoload("conv-other", AGENT, [_server_ref()])

        assert ("conv-busy", AGENT) in state._conversation_tools

    def test_scope_cap_evicts_the_least_recently_active_scope(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.mcp_tool_search_max_tracked_scopes", 2)
        state = DeferredToolState()

        state.autoload("conv-a", AGENT, [_server_ref()])
        state.autoload("conv-b", AGENT, [_server_ref()])
        state._conversation_tools[("conv-a", AGENT)].last_active = _now() - 600.0
        state.autoload("conv-c", AGENT, [_server_ref()])

        assert ("conv-a", AGENT) not in state._conversation_tools
        assert len(state._conversation_tools) == 2

    def test_client_scope_cap_is_independent_of_server_scopes(self, monkeypatch):
        monkeypatch.setattr("app.core.config.settings.mcp_tool_search_max_tracked_scopes", 2)
        state = DeferredToolState()

        for device in ("dev-a", "dev-b", "dev-c"):
            state.autoload_client_tools(
                conversation_id=CONV,
                agent_key=AGENT,
                references=[_client_ref(device_id=device)],
                device_id=device,
                session_id=f"sess-{device}",
            )

        assert len(state._client_tool_scopes) == 2


# ---------------------------------------------------------------------------
# 3. Conversation lifecycle
# ---------------------------------------------------------------------------


class TestClearConversation:
    def test_clear_drops_server_and_client_scopes(self):
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])
        state.autoload_client_tools(
            conversation_id=CONV,
            agent_key=AGENT,
            references=[_client_ref()],
            device_id="dev-a",
            session_id="sess-a",
        )

        assert state.clear_conversation(CONV) == 2
        assert state.get_loaded(CONV, AGENT) == []
        assert (
            state.get_loaded_client_tools(CONV, AGENT, device_id="dev-a", session_id="sess-a") == []
        )

    def test_clear_for_one_agent_leaves_the_others(self):
        state = DeferredToolState()
        state.autoload(CONV, "chat", [_server_ref()])
        state.autoload(CONV, "rag", [_server_ref()])

        assert state.clear_conversation(CONV, "chat") == 1
        assert state.get_loaded(CONV, "rag") != []

    def test_clear_of_an_unknown_conversation_is_a_no_op(self):
        state = DeferredToolState()
        assert state.clear_conversation("never-seen") == 0


# ---------------------------------------------------------------------------
# 4. Session resolution is ownership-validated
# ---------------------------------------------------------------------------


class TestSessionResolution:
    @pytest.fixture
    def state_with_client_tool(self):
        state = DeferredToolState()
        state.autoload_client_tools(
            conversation_id=CONV,
            agent_key=AGENT,
            references=[_client_ref()],
            device_id="dev-a",
            session_id="sess-a",
        )
        return state

    def test_user_id_routes_through_the_validated_resolver(
        self, monkeypatch, state_with_client_tool
    ):
        """A device the user does not own must resolve to nothing, even though
        the unvalidated device-only lookup would happily answer."""
        monkeypatch.setattr(
            "app.ai.client_runtime_tools.get_active_client_runtime_session",
            lambda **kwargs: None,
        )
        monkeypatch.setattr(
            "app.services.client_device_service.ClientDeviceService.lookup_active_session",
            lambda _device_id: SimpleNamespace(session_id="sess-a"),
        )

        assert (
            state_with_client_tool.get_loaded_client_tools(
                CONV, AGENT, device_id="dev-a", user_id="user-1"
            )
            == []
        )

    def test_validated_resolver_supplies_the_session_when_it_matches(
        self, monkeypatch, state_with_client_tool
    ):
        monkeypatch.setattr(
            "app.ai.client_runtime_tools.get_active_client_runtime_session",
            lambda **kwargs: SimpleNamespace(session_id="sess-a"),
        )

        tools = state_with_client_tool.get_loaded_client_tools(
            CONV, AGENT, device_id="dev-a", user_id="user-1"
        )
        assert [tool.tool_name for tool in tools] == ["client__t1"]

    def test_no_scope_at_all_returns_nothing(self, state_with_client_tool):
        assert state_with_client_tool.get_loaded_client_tools(CONV, AGENT) == []


# ---------------------------------------------------------------------------
# 5. Snapshot and restore honour recorded staleness
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_snapshot_records_the_producing_runtime(self):
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])

        snapshot = state.snapshot(CONV, AGENT)
        assert snapshot["runtime_id"] == RUNTIME_ID
        assert [tool["call_name"] for tool in snapshot["server_tools"]] == ["brave__search"]

    def test_round_trip_restores_the_same_tools(self):
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])
        snapshot = state.snapshot(CONV, AGENT)

        target = DeferredToolState()
        assert target.restore(CONV, AGENT, snapshot)["server_tools"] == 1
        assert [ref.call_name for ref in target.get_loaded(CONV, AGENT)] == ["brave__search"]

    def test_same_runtime_snapshot_drops_tools_from_an_older_generation(self, monkeypatch):
        """An MCP reload invalidates what was loaded before it; restoring the
        snapshot must not launder a stale entry back in as current."""
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])
        snapshot = state.snapshot(CONV, AGENT)

        monkeypatch.setattr(
            "app.ai.deferred_tool_state.get_mcp_tools_generation",
            lambda: 99,
        )
        target = DeferredToolState()
        assert target.restore(CONV, AGENT, snapshot)["server_tools"] == 0

    def test_snapshot_from_another_runtime_is_restored_as_is(self, monkeypatch):
        """Generation counters restart at zero per process, so a snapshot from
        a different one cannot be compared against them."""
        state = DeferredToolState()
        state.autoload(CONV, AGENT, [_server_ref()])
        snapshot = state.snapshot(CONV, AGENT)
        snapshot["runtime_id"] = "some-other-process"

        monkeypatch.setattr(
            "app.ai.deferred_tool_state.get_mcp_tools_generation",
            lambda: 99,
        )
        target = DeferredToolState()
        assert target.restore(CONV, AGENT, snapshot)["server_tools"] == 1

    def test_client_entries_from_another_device_are_dropped(self):
        state = DeferredToolState()
        snapshot = {
            "runtime_id": RUNTIME_ID,
            "server_tools": [],
            "client_tools": [
                {
                    "tool_name": "client__t1",
                    "server_name": "native",
                    "device_id": "dev-other",
                    "session_id": "sess-a",
                    "catalog_version": 3,
                    "tool_instance_id": "inst-1",
                }
            ],
        }
        restored = state.restore(
            CONV, AGENT, snapshot, device_id="dev-a", session_id="sess-a"
        )
        assert restored["client_tools"] == 0

    def test_client_entries_from_a_superseded_session_are_dropped(self):
        """tool_instance_ids rotate on reconnect, so an entry from the previous
        session is uninvokable and must not eat the new session's budget."""
        state = DeferredToolState()
        snapshot = {
            "runtime_id": RUNTIME_ID,
            "server_tools": [],
            "client_tools": [
                {
                    "tool_name": "client__t1",
                    "server_name": "native",
                    "device_id": "dev-a",
                    "session_id": "sess-old",
                    "catalog_version": 3,
                    "tool_instance_id": "inst-old",
                }
            ],
        }
        restored = state.restore(
            CONV, AGENT, snapshot, device_id="dev-a", session_id="sess-new"
        )
        assert restored["client_tools"] == 0
        assert (
            state.get_loaded_client_tools(
                CONV, AGENT, device_id="dev-a", session_id="sess-new"
            )
            == []
        )

    def test_client_entries_for_the_current_session_are_restored(self):
        state = DeferredToolState()
        snapshot = {
            "runtime_id": RUNTIME_ID,
            "server_tools": [],
            "client_tools": [
                {
                    "tool_name": "client__t1",
                    "server_name": "native",
                    "device_id": "dev-a",
                    "session_id": "sess-a",
                    "catalog_version": 3,
                    "tool_instance_id": "inst-1",
                }
            ],
        }
        restored = state.restore(CONV, AGENT, snapshot, device_id="dev-a", session_id="sess-a")
        assert restored["client_tools"] == 1

        tools = state.get_loaded_client_tools(CONV, AGENT, device_id="dev-a", session_id="sess-a")
        assert [tool.tool_instance_id for tool in tools] == ["inst-1"]

    def test_a_turn_without_a_device_restores_no_client_tools(self):
        state = DeferredToolState()
        snapshot = {
            "runtime_id": RUNTIME_ID,
            "server_tools": [],
            "client_tools": [
                {
                    "tool_name": "client__t1",
                    "server_name": "native",
                    "device_id": "dev-a",
                    "session_id": "sess-a",
                }
            ],
        }
        assert state.restore(CONV, AGENT, snapshot)["client_tools"] == 0

    def test_a_malformed_snapshot_is_ignored(self):
        state = DeferredToolState()
        assert state.restore(CONV, AGENT, None) == {"server_tools": 0, "client_tools": 0}
        assert state.restore(CONV, AGENT, {"server_tools": ["nonsense", 7]}) == {
            "server_tools": 0,
            "client_tools": 0,
        }


# ---------------------------------------------------------------------------
# 6. Catalog-version staleness for client tools
# ---------------------------------------------------------------------------


class TestClientCatalogStaleness:
    def test_a_republished_catalog_drops_the_superseded_tool(self):
        scope = ClientToolScope()
        scope.add("client__t1", "native", "dev-a", 3, max_tools=5, tool_instance_id="inst-1")

        assert scope.cleanup(ttl_minutes=0, current_catalog_version=4) == 1
        assert "client__t1" not in scope

    def test_an_unknown_catalog_version_never_unloads_a_tool(self):
        """A catalog lookup that fails reports 0; treating that as a mismatch
        would silently unload every client tool on the device."""
        scope = ClientToolScope()
        scope.add("client__t1", "native", "dev-a", 3, max_tools=5, tool_instance_id="inst-1")

        assert scope.cleanup(ttl_minutes=0, current_catalog_version=0) == 0
        assert "client__t1" in scope

    def test_a_tool_loaded_before_any_version_was_known_is_kept(self):
        scope = ClientToolScope()
        scope.add("client__t1", "native", "dev-a", 0, max_tools=5)

        assert scope.cleanup(ttl_minutes=0, current_catalog_version=4) == 0
        assert "client__t1" in scope


# ---------------------------------------------------------------------------
# 7. Alias identity
# ---------------------------------------------------------------------------


class TestCallNameIdentity:
    def test_an_unaliased_tool_reports_its_own_name_as_the_call_name(self):
        tool_set = ConversationToolSet()
        tool_set.add("get_time", "time_server", generation=1, max_tools=5)

        [ref] = tool_set.list_tools()
        assert ref.get_call_name() == "get_time"

    def test_an_aliased_tool_is_stored_and_returned_under_the_alias(self):
        tool_set = ConversationToolSet()
        tool_set.add("search", "brave", generation=1, max_tools=5, call_name="brave__search")

        assert "brave__search" in tool_set
        [ref] = tool_set.list_tools()
        assert (ref.tool_name, ref.get_call_name()) == ("search", "brave__search")

    def test_rebinding_an_alias_to_another_server_updates_both_halves(self):
        tool_set = ConversationToolSet()
        tool_set.add("search", "brave", generation=1, max_tools=5, call_name="dual__search")
        tool_set.add("find", "tavily", generation=2, max_tools=5, call_name="dual__search")

        tool = tool_set.loaded["dual__search"]
        assert (tool.tool_name, tool.server_name, tool.generation) == ("find", "tavily", 2)
