import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.ai.mcp_integration import MCPManager, compute_catalog_version
from app.core.exceptions.mcp import AmbiguousToolNameError


def _fake_tool(name: str, schema: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"{name} desc",
        args_schema=schema or {"type": "object"},
    )


@pytest.mark.asyncio
async def test_list_tool_descriptors_keeps_same_name_across_servers():
    """Duplicate bare tool names on different servers both survive in the listing,
    each stamped with its own server provenance (no collapse-by-name)."""
    manager = MCPManager.__new__(MCPManager)
    manager._server_tools = {
        "alpha": [_fake_tool("inspect")],
        "beta": [_fake_tool("inspect")],
    }

    async def _noop_get_tools():
        return []

    manager.get_tools = _noop_get_tools  # type: ignore[method-assign]

    descriptors = await manager.list_tool_descriptors(None)
    pairs = sorted((d["server_name"], d["name"]) for d in descriptors)
    assert pairs == [("alpha", "inspect"), ("beta", "inspect")]


@pytest.mark.asyncio
async def test_list_tool_descriptors_scoped_loads_only_that_server():
    """A scoped request loads only the requested server's tools (scoped LOAD, not
    get-all-then-filter)."""
    manager = MCPManager.__new__(MCPManager)
    manager._server_tools = {"beta": [_fake_tool("inspect")]}
    loaded: list[str] = []

    async def _fake_get_server_tools(server_name):
        loaded.append(server_name)
        return manager._server_tools.get(server_name, [])

    manager.get_server_tools = _fake_get_server_tools  # type: ignore[method-assign]

    descriptors = await manager.list_tool_descriptors("beta")
    assert loaded == ["beta"]
    assert {d["server_name"] for d in descriptors} == {"beta"}


@pytest.mark.asyncio
async def test_list_tool_descriptors_scoped_known_empty_server_returns_empty():
    """A known server currently exposing zero tools yields an empty list, not an error."""
    manager = MCPManager.__new__(MCPManager)
    manager._server_tools = {}

    async def _fake_get_server_tools(server_name):
        return []

    manager.get_server_tools = _fake_get_server_tools  # type: ignore[method-assign]

    assert await manager.list_tool_descriptors("brave") == []


def test_compute_catalog_version_is_deterministic_and_order_independent():
    a = [
        {"server_name": "s1", "name": "t1", "args_schema": {"type": "object"}},
        {"server_name": "s2", "name": "t2", "args_schema": {"type": "object"}},
    ]
    b = list(reversed(a))
    assert compute_catalog_version(a) == compute_catalog_version(b)
    assert compute_catalog_version(a).startswith("sha256:")


def test_compute_catalog_version_changes_on_schema_change():
    base = [{"server_name": "s1", "name": "t1", "args_schema": {"type": "object"}}]
    changed = [{"server_name": "s1", "name": "t1", "args_schema": {"type": "string"}}]
    assert compute_catalog_version(base) != compute_catalog_version(changed)


def test_default_stdio_servers_resolve_from_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    manager = MCPManager()
    manager._ensure_config_loaded()
    server_config = manager._build_server_config()

    assert server_config
    for config in server_config.values():
        assert config["command"] == sys.executable
        script = Path(config["args"][0])
        assert script.is_absolute()
        assert script.is_file()


def test_explicit_stdio_script_resolves_relative_to_config_file(tmp_path, monkeypatch):
    config_dir = tmp_path / "custom"
    config_dir.mkdir()
    server_script = config_dir / "server.py"
    server_script.write_text("print('ok')", encoding="utf-8")
    config_path = config_dir / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "custom": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": ["server.py"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    manager = MCPManager(config_path=str(config_path))
    manager._ensure_config_loaded()

    resolved = manager._build_server_config()["custom"]

    assert resolved["args"] == [str(server_script.resolve())]


def test_explicit_stdio_resolves_all_path_fields_without_touching_path_commands_or_urls(
    tmp_path, monkeypatch
):
    config_dir = tmp_path / "custom"
    command = config_dir / "bin" / "custom-runner"
    asset = config_dir / "assets" / "schema.json"
    executable_arg = config_dir / "helper.exe"
    runtime_dir = config_dir / "runtime"
    for path in (command, asset, executable_arg):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    runtime_dir.mkdir()

    # A same-named config-relative file must not turn a bare PATH command into
    # a file path. Commands such as npx/python are intentionally PATH-resolved.
    (config_dir / "npx").write_text("fixture", encoding="utf-8")
    (config_dir / "npx.cmd").write_text("fixture", encoding="utf-8")
    config_path = config_dir / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "custom": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": "bin/custom-runner",
                        "args": [
                            "--watch",
                            "assets/schema.json",
                            "helper.exe",
                            "missing-server.py",
                            "postgresql://db.example/service",
                        ],
                        "cwd": "runtime",
                    },
                    "package": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["--yes", "@scope/package", "https://example.test/spec"],
                    },
                    "windows-package": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": "npx.cmd",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    manager = MCPManager(config_path=str(config_path))
    manager._ensure_config_loaded()
    resolved = manager._build_server_config()

    assert resolved["custom"]["command"] == str(command.resolve())
    assert resolved["custom"]["args"] == [
        "--watch",
        str(asset.resolve()),
        str(executable_arg.resolve()),
        str((config_dir / "missing-server.py").resolve()),
        "postgresql://db.example/service",
    ]
    assert resolved["custom"]["cwd"] == str(runtime_dir.resolve())
    assert resolved["package"]["command"] == "npx"
    assert resolved["package"]["args"] == [
        "--yes",
        "@scope/package",
        "https://example.test/spec",
    ]
    assert resolved["windows-package"]["command"] == "npx.cmd"


@pytest.mark.asyncio
async def test_server_mcp_manager_cleans_up_stdio_session_without_cancel_scope_error(
    tmp_path,
):
    config_path = tmp_path / "mcp_config.json"
    server_script = Path("app/ai/mcp_servers/time_server.py").resolve()
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "time": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager(config_path=str(config_path))

    await manager.initialize()
    tool = await manager.get_tool_by_name("get_current_time", server_name="time")
    assert tool is not None

    result = await tool.ainvoke({"timezone": "UTC", "format": "%Y-%m-%d"})
    assert "UTC" in str(result)

    await manager.cleanup()


@pytest.mark.asyncio
async def test_server_mcp_manager_closes_session_opened_by_another_task_without_warning(
    tmp_path,
    caplog,
):
    config_path = tmp_path / "mcp_config.json"
    server_script = Path("app/ai/mcp_servers/time_server.py").resolve()
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "time": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager(config_path=str(config_path))

    await manager.initialize()
    load_task = asyncio.create_task(
        manager.get_tool_by_name("get_current_time", server_name="time")
    )
    tool = await load_task
    assert tool is not None

    with caplog.at_level(logging.WARNING, logger="app.ai.mcp_integration"):
        await manager.cleanup()

    assert "Error closing session for time" not in caplog.text


@pytest.mark.asyncio
async def test_server_mcp_tool_cancellation_does_not_escape_as_unbound_local_error(
    tmp_path,
):
    server_script = tmp_path / "slow_mcp_server.py"
    server_script.write_text(
        "\n".join(
            [
                "import asyncio",
                "from mcp.server.fastmcp import FastMCP",
                "",
                "mcp = FastMCP('Slow')",
                "",
                "@mcp.tool()",
                "async def slow_echo(value: str, delay: float = 10.0) -> str:",
                "    await asyncio.sleep(delay)",
                "    return value",
                "",
                "if __name__ == '__main__':",
                "    mcp.run(transport='stdio')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "mcp_config.json"
    config_path.write_text(
        json.dumps(
            {
                "servers": {
                    "slow": {
                        "enabled": True,
                        "transport": "stdio",
                        "command": sys.executable,
                        "args": [str(server_script)],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = MCPManager(config_path=str(config_path))

    await manager.initialize()
    tool = await manager.get_tool_by_name("slow_echo", server_name="slow")
    assert tool is not None

    task = asyncio.create_task(tool.ainvoke({"value": "hello", "delay": 5.0}))
    await asyncio.sleep(0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await manager.cleanup()


# ---------------------------------------------------------------------------
# Execution-path provenance (T006 follow-up)
#
# FR-MCP-003: server ownership is attached when tools are loaded and retained in
# a stable descriptor. The execution path previously resolved it through an
# ``id(tool)``-keyed map, which is address-based: a tool the manager did not
# personally index (a clone, a rebuilt binding, a tool handed back by another
# layer) resolved to "unknown", and a recycled CPython id could attribute a tool
# to the wrong server.
# ---------------------------------------------------------------------------


def _stamped_tool(name: str, server: str) -> SimpleNamespace:
    """A tool as ``clone_mcp_tool`` produces it: provenance in metadata."""
    return SimpleNamespace(
        name=name,
        description=f"{name} desc",
        args_schema={"type": "object"},
        metadata={
            "tool_origin": "server_mcp",
            "server_name": server,
            "source_tool_name": name,
            "qualified_tool_id": f"{server}::{name}",
        },
    )


def test_get_server_for_tool_reads_stamped_provenance_not_identity():
    """A stamped tool resolves to its server even when this manager instance
    never indexed that exact object."""
    manager = MCPManager.__new__(MCPManager)
    manager._server_tools = {}
    manager._tool_index = {}

    tool = _stamped_tool("search", "brave_image_search")
    assert manager.get_server_for_tool(tool) == "brave_image_search"


def test_get_server_for_tool_falls_back_to_server_index_for_unstamped_tool():
    """Tools without metadata (older adapters, test doubles) still resolve via
    the per-server index by identity."""
    manager = MCPManager.__new__(MCPManager)
    tool = _fake_tool("legacy")
    manager._server_tools = {"legacy_server": [tool]}
    manager._tool_index = {"legacy": [tool]}

    assert manager.get_server_for_tool(tool) == "legacy_server"
    assert manager.get_server_for_tool(_fake_tool("unknown")) is None


@pytest.mark.asyncio
async def test_get_servers_for_tool_name_uses_stamped_provenance():
    """The same bare name on two servers reports BOTH servers."""
    manager = MCPManager.__new__(MCPManager)
    alpha = _stamped_tool("inspect", "alpha")
    beta = _stamped_tool("inspect", "beta")
    manager._server_tools = {"alpha": [alpha], "beta": [beta]}
    manager._tool_index = {"inspect": [alpha, beta]}

    async def _noop_get_tools():
        return []

    manager.get_tools = _noop_get_tools  # type: ignore[method-assign]

    assert sorted(await manager.get_servers_for_tool_name("inspect")) == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Ambiguous execution requires server qualification (T006 follow-up)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ambiguous_bare_name_lookup_is_rejected_not_silently_first_wins():
    """Two servers exposing the same bare name must not silently resolve to
    whichever was indexed first — the caller has to qualify the server."""
    manager = MCPManager.__new__(MCPManager)
    alpha = _stamped_tool("inspect", "alpha")
    beta = _stamped_tool("inspect", "beta")
    manager._server_tools = {"alpha": [alpha], "beta": [beta]}
    manager._tool_index = {"inspect": [alpha, beta]}

    async def _noop_get_tools():
        return []

    manager.get_tools = _noop_get_tools  # type: ignore[method-assign]

    with pytest.raises(AmbiguousToolNameError) as excinfo:
        await manager.get_tool_by_name("inspect")

    assert excinfo.value.status_code == 409
    assert "alpha" in excinfo.value.detail and "beta" in excinfo.value.detail

    assert await manager.get_tool_by_name("inspect", server_name="beta") is beta


@pytest.mark.asyncio
async def test_unambiguous_bare_name_lookup_still_resolves():
    """One server owning the name keeps working without qualification."""
    manager = MCPManager.__new__(MCPManager)
    only = _stamped_tool("inspect", "alpha")
    manager._server_tools = {"alpha": [only]}
    manager._tool_index = {"inspect": [only]}

    async def _noop_get_tools():
        return []

    manager.get_tools = _noop_get_tools  # type: ignore[method-assign]

    assert await manager.get_tool_by_name("inspect") is only


@pytest.mark.asyncio
async def test_execute_tool_accepts_server_qualification_for_duplicate_names():
    """``execute_tool`` routes to the qualified server and reports it back."""
    manager = MCPManager.__new__(MCPManager)

    class _Invocable(SimpleNamespace):
        async def ainvoke(self, arguments):
            return f"{self.metadata['server_name']}:{arguments['q']}"

    alpha = _Invocable(
        name="inspect",
        description="",
        args_schema=None,
        metadata={"server_name": "alpha", "qualified_tool_id": "alpha::inspect"},
    )
    beta = _Invocable(
        name="inspect",
        description="",
        args_schema=None,
        metadata={"server_name": "beta", "qualified_tool_id": "beta::inspect"},
    )
    manager._server_tools = {"alpha": [alpha], "beta": [beta]}
    manager._tool_index = {"inspect": [alpha, beta]}

    async def _noop_get_tools():
        return []

    manager.get_tools = _noop_get_tools  # type: ignore[method-assign]

    result = await manager.execute_tool("inspect", {"q": "x"}, server_name="beta")
    assert result["success"] is True
    assert result["result"] == "beta:x"
    assert result["server_name"] == "beta"


def test_get_server_for_tool_ignores_device_local_tool_provenance():
    """A device-local (client) MCP tool must NEVER be attributed to the backend
    MCP catalog.

    Client runtime tools carry their own ``metadata["server_name"]`` (the MCP
    server name ON THE USER'S DEVICE) alongside ``tool_origin="client_mcp"``.
    If ``MCPManager`` reported that name as backend provenance, a user could
    name a local server after a backend one and have their local tools admitted
    by a base agent's server allowlist
    (``base_agent._filter_tools_by_allowlist``) — a scope-broadening fail-open
    across the device boundary.
    """
    manager = MCPManager.__new__(MCPManager)
    manager._server_tools = {}
    manager._tool_index = {}

    client_tool = SimpleNamespace(
        name="tavily__search",
        description="",
        args_schema={"type": "object"},
        metadata={
            "tool_origin": "client_mcp",
            "is_client_tool": True,
            "server_name": "tavily",  # a LOCAL server that shadows a backend name
            "qualified_tool_id": "tavily::search",
        },
    )
    client_skill = SimpleNamespace(
        name="do_thing",
        description="",
        args_schema={"type": "object"},
        metadata={"tool_origin": "client_skill", "server_name": "widgets"},
    )

    assert manager.get_server_for_tool(client_tool) is None
    assert manager.get_server_for_tool(client_skill) is None


def test_get_server_for_tool_accepts_backend_mcp_provenance():
    """The backend stamp (``tool_origin="server_mcp"``) still resolves."""
    manager = MCPManager.__new__(MCPManager)
    manager._server_tools = {}
    manager._tool_index = {}

    assert manager.get_server_for_tool(_stamped_tool("search", "tavily")) == "tavily"
