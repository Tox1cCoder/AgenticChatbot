"""Tests for syncing ready skill capabilities as client tool catalog entries.

Covers client_backend.services.skill_runtime.manager.capability_catalog_entries
(pure catalog-entry building) and the RuntimeBridgeService._build_tool_catalog
merge point where skill capability entries join the MCP tool catalog.
"""

from types import SimpleNamespace

import pytest

from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services.runtime_bridge import RuntimeBridgeService
from client_backend.services.skill_runtime.manager import SkillReadiness, SkillRuntimeManager
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from shared.skills.manifest import load_manifest


def _skill_manifest_dict(name: str = "example-calendar") -> dict:
    """A provider-neutral, ready-to-execute manifest with two capabilities."""
    return {
        "schema_version": "1.0",
        "name": name,
        "description": "A skill used to exercise the capability catalog.",
        "runtime": {"type": "python_module", "module": "skills.example.cli"},
        "dependencies": {"python": [], "node": [], "system": []},
        "secrets": [],
        "permissions": [],
        "capabilities": [
            {
                "name": "event_list",
                "description": "List calendar events.",
                "input_schema": {
                    "type": "object",
                    "properties": {"time_min": {"type": "string"}},
                    "required": ["time_min"],
                },
                "execution": {"argv": ["event-list"]},
                "permissions": [],
                "secrets": [],
                "mutation": False,
            },
            {
                "name": "event_create",
                "description": "Create a calendar event.",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                },
                "execution": {"argv": ["event-create"]},
                "permissions": [],
                "secrets": [],
                "mutation": True,
            },
        ],
    }


class TestCapabilityCatalogEntries:
    def test_ready_skill_produces_one_entry_per_capability(self):
        manifest = load_manifest(_skill_manifest_dict("example-calendar"))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))
        readiness = manager.evaluate_readiness(manifest)
        assert readiness.status == "ready"

        entries = manager.capability_catalog_entries("example-calendar", manifest, readiness)

        assert len(entries) == 2
        assert {entry["name"] for entry in entries} == {"event_list", "event_create"}

        event_list = next(entry for entry in entries if entry["name"] == "event_list")
        assert event_list["description"] == "List calendar events."
        assert event_list["origin"] == "skill"
        assert event_list["server_name"] == "skill_example_calendar"
        assert event_list["qualified_id"] == "skill::example-calendar::event_list"
        assert event_list["input_schema"] == {
            "type": "object",
            "properties": {"time_min": {"type": "string"}},
            "required": ["time_min"],
        }
        assert event_list["readiness"] == {"status": "ready"}

    def test_hyphenated_skill_name_maps_to_underscored_server_name(self):
        manifest = load_manifest(_skill_manifest_dict("example-calendar"))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))
        readiness = manager.evaluate_readiness(manifest)

        entries = manager.capability_catalog_entries("example-calendar", manifest, readiness)

        assert all(entry["server_name"] == "skill_example_calendar" for entry in entries)
        assert all(
            entry["qualified_id"].startswith("skill::example-calendar::") for entry in entries
        )

    def test_not_ready_skill_produces_no_entries(self):
        manifest = load_manifest(_skill_manifest_dict("example-calendar"))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))
        readiness = SkillReadiness(status="not_ready")

        entries = manager.capability_catalog_entries("example-calendar", manifest, readiness)

        assert entries == []

    @pytest.mark.parametrize("status", ["instruction_only", "invalid"])
    def test_non_ready_statuses_produce_no_entries(self, status):
        manifest = load_manifest(_skill_manifest_dict("example-calendar"))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))
        readiness = SkillReadiness(status=status)

        entries = manager.capability_catalog_entries("example-calendar", manifest, readiness)

        assert entries == []


class TestRuntimeBridgeMergesSkillTools:
    @pytest.mark.asyncio
    async def test_build_tool_catalog_merges_mcp_and_skill_entries_without_collision(
        self, monkeypatch
    ):
        bridge = RuntimeBridgeService()

        monkeypatch.setattr(
            runtime_bridge_module,
            "get_mcp_manager",
            lambda: SimpleNamespace(
                get_tool_catalog=lambda: {
                    "tools": [
                        {
                            "name": "start_process",
                            "qualified_id": "desktop_commander::start_process",
                            "origin": "mcp",
                            "server_name": "desktop_commander",
                            "description": "Start a local process.",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    "server_count": 1,
                    "active_servers": ["desktop_commander"],
                }
            ),
        )

        manifest = load_manifest(_skill_manifest_dict("example-calendar"))
        skill = SimpleNamespace(
            name="example-calendar",
            enabled=True,
            manifest=manifest,
            manifest_error=None,
        )
        monkeypatch.setattr(
            runtime_bridge_module,
            "get_skills_registry",
            lambda: SimpleNamespace(get_enabled_skills=lambda: [skill]),
        )

        catalog = await bridge._build_tool_catalog()

        tools_by_qualified_id = {tool["qualified_id"]: tool for tool in catalog["tools"]}
        assert "desktop_commander::start_process" in tools_by_qualified_id

        skill_qualified_ids = [
            qid for qid in tools_by_qualified_id if qid.startswith("skill::example-calendar::")
        ]
        assert skill_qualified_ids

        mcp_entry = tools_by_qualified_id["desktop_commander::start_process"]
        skill_entry = tools_by_qualified_id[skill_qualified_ids[0]]

        assert mcp_entry["tool_instance_id"]
        assert skill_entry["tool_instance_id"]
        assert mcp_entry["qualified_id"] != skill_entry["qualified_id"]

    @pytest.mark.asyncio
    async def test_build_tool_catalog_skips_not_ready_skill(self, monkeypatch):
        bridge = RuntimeBridgeService()

        monkeypatch.setattr(
            runtime_bridge_module,
            "get_mcp_manager",
            lambda: SimpleNamespace(
                get_tool_catalog=lambda: {"tools": [], "server_count": 0, "active_servers": []}
            ),
        )

        # A required secret that is never satisfied keeps readiness at "not_ready".
        manifest_dict = _skill_manifest_dict("example-calendar")
        manifest_dict["secrets"] = [
            {"name": "EXAMPLE_TOKEN", "required": True, "description": "token"}
        ]
        manifest = load_manifest(manifest_dict)
        skill = SimpleNamespace(
            name="example-calendar",
            enabled=True,
            manifest=manifest,
            manifest_error=None,
        )
        monkeypatch.setattr(
            runtime_bridge_module,
            "get_skills_registry",
            lambda: SimpleNamespace(get_enabled_skills=lambda: [skill]),
        )

        catalog = await bridge._build_tool_catalog()

        assert catalog["tools"] == []

    @pytest.mark.asyncio
    async def test_build_tool_catalog_survives_skill_collection_failure(self, monkeypatch):
        """A skill-runtime hiccup must never break MCP tool catalog sync."""
        bridge = RuntimeBridgeService()

        monkeypatch.setattr(
            runtime_bridge_module,
            "get_mcp_manager",
            lambda: SimpleNamespace(
                get_tool_catalog=lambda: {
                    "tools": [
                        {
                            "name": "start_process",
                            "qualified_id": "desktop_commander::start_process",
                            "origin": "mcp",
                            "server_name": "desktop_commander",
                            "description": "Start a local process.",
                            "input_schema": {"type": "object", "properties": {}},
                        }
                    ],
                    "server_count": 1,
                    "active_servers": ["desktop_commander"],
                }
            ),
        )

        def _raise():
            raise RuntimeError("skills registry unavailable")

        monkeypatch.setattr(runtime_bridge_module, "get_skills_registry", _raise)

        catalog = await bridge._build_tool_catalog()

        assert [tool["qualified_id"] for tool in catalog["tools"]] == [
            "desktop_commander::start_process"
        ]
