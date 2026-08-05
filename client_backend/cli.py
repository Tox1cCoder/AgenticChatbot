"""
Command-line entrypoint for the client backend runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


def _apply_env_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "config", None):
        os.environ["CLIENT_ENV_FILE"] = str(Path(args.config).expanduser())
    if getattr(args, "host", None):
        os.environ["CLIENT_BACKEND_HOST"] = args.host
    if getattr(args, "port", None):
        os.environ["CLIENT_BACKEND_PORT"] = str(args.port)
    if getattr(args, "log_level", None):
        os.environ["CLIENT_LOG_LEVEL"] = args.log_level
    if getattr(args, "environment", None):
        os.environ["CLIENT_ENVIRONMENT"] = args.environment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kani-client-backend")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the client backend")
    run_parser.add_argument("--config", help="Path to the .env.client file")
    run_parser.add_argument("--host", help="Override CLIENT_BACKEND_HOST")
    run_parser.add_argument("--port", type=int, help="Override CLIENT_BACKEND_PORT")
    run_parser.add_argument("--log-level", help="Override CLIENT_LOG_LEVEL")
    run_parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        help="Override CLIENT_ENVIRONMENT",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Validate local runtime configuration")
    doctor_parser.add_argument("--config", help="Path to the .env.client file")
    doctor_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Manage the active device-scoped MCP profile",
    )
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", required=True)
    mcp_migrate_parser = mcp_subparsers.add_parser(
        "migrate",
        help="Migrate the authenticated user's legacy MCP profile to schema v2",
    )
    mcp_doctor_parser = mcp_subparsers.add_parser(
        "doctor",
        help="Discover tools from selected MCP servers",
    )
    mcp_doctor_parser.add_argument(
        "--servers",
        required=True,
        help="Comma-separated server names",
    )
    for command_parser in (mcp_migrate_parser, mcp_doctor_parser):
        command_parser.add_argument(
            "--user-id",
            help="Explicit local profile user ID when no in-process session is active",
        )
        command_parser.add_argument(
            "--device-identifier",
            help="Override the stable installation identifier",
        )

    return parser


def _run_doctor(args: argparse.Namespace) -> int:
    _apply_env_overrides(args)

    from client_backend.core.config import get_client_settings, initialize_client_environment
    settings = get_client_settings()
    initialize_client_environment(settings)

    parsed_server = urlparse(settings.server_api_base_url)
    checks = {
        "server_api_base_url_has_scheme": bool(parsed_server.scheme and parsed_server.netloc),
        "profile_root_exists": Path(settings.profile_root).exists(),
        # An unset skills root resolves per user under the profile and is created
        # by the first install, so only a configured one can be missing here.
        "skills_root_exists": (
            Path(settings.skills_root).exists() if settings.skills_root else True
        ),
    }

    result = {
        "status": "ok" if all(checks.values()) else "warning",
        "checks": checks,
        "config": {
            "server_api_base_url": settings.server_api_base_url,
            "backend_host": settings.backend_host,
            "backend_port": settings.backend_port,
            "profile_root": settings.profile_root,
            "environment": settings.environment,
            "skills_root": settings.skills_root or "(profile default)",
        },
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Status: {result['status']}")
        for name, passed in checks.items():
            print(f"- {name}: {'OK' if passed else 'FAIL'}")
        print("Config:")
        for key, value in result["config"].items():
            print(f"  {key}: {value}")

    return 0 if all(checks.values()) else 1


def _run_server(args: argparse.Namespace) -> int:
    _apply_env_overrides(args)
    from client_backend.main import main as run_main

    run_main()
    return 0


def _resolve_mcp_scope(
    user_id: str | None = None,
    device_identifier: str | None = None,
):
    from client_backend.core.security import generate_device_identifier
    from client_backend.schemas.mcp_config import MCPProfileScope
    from client_backend.services.local_mcp_manager import resolve_current_mcp_scope

    if not user_id:
        return resolve_current_mcp_scope()
    return MCPProfileScope(
        user_id=user_id,
        device_identifier=device_identifier or generate_device_identifier(),
    )


def migrate_mcp_config(
    user_id: str | None = None,
    device_identifier: str | None = None,
) -> int:
    """Migrate the active authenticated installation without printing secrets."""

    from client_backend.core.config import (
        client_settings,
        initialize_client_environment,
    )
    from client_backend.services.mcp_config_migration import (
        prepare_mcp_config_store,
    )

    initialize_client_environment(client_settings)
    scope = _resolve_mcp_scope(user_id, device_identifier)
    store, result = prepare_mcp_config_store(scope)
    print(f"Status: {result.status}")
    print(f"Profile: {store.profile_path}")
    if result.migrated_servers:
        print(f"Servers: {','.join(result.migrated_servers)}")
    if result.backup_path is not None:
        print(f"Backup: {result.backup_path}")
    if result.receipt_path is not None:
        print(f"Receipt: {result.receipt_path}")
    return 0


def doctor_mcp_servers(
    server_names: list[str],
    user_id: str | None = None,
    device_identifier: str | None = None,
) -> int:
    """Discover selected servers and report only redacted runtime status."""

    import asyncio

    from client_backend.core.config import (
        client_settings,
        initialize_client_environment,
    )
    from client_backend.services.local_mcp_manager import (
        LocalMCPManager,
    )
    from client_backend.services.mcp_config_store import MCPConfigStore

    requested = {name.strip() for name in server_names if name.strip()}
    if not requested:
        print("No MCP server names were provided.", file=sys.stderr)
        return 2

    async def run() -> int:
        initialize_client_environment(client_settings)
        manager = LocalMCPManager(
            store=MCPConfigStore(
                _resolve_mcp_scope(user_id, device_identifier)
            )
        )
        try:
            effective = {
                server.name: server
                for server in manager.store.list_effective_servers()
            }
            await manager.initialize(server_names=requested)
            failed = False
            for name in sorted(requested):
                definition = effective.get(name)
                runtime = manager.servers.get(name)
                if definition is None:
                    failed = True
                    print(f"- {name}: NOT_FOUND")
                elif not definition.enabled:
                    failed = True
                    print(f"- {name}: DISABLED")
                elif runtime is None or not runtime.is_running():
                    failed = True
                    error = (
                        runtime.error_message
                        if runtime and runtime.error_message
                        else "server did not initialize"
                    )
                    print(f"- {name}: FAILED ({error})")
                else:
                    print(f"- {name}: OK ({len(runtime.tools)} tools)")
            return 1 if failed else 0
        finally:
            await manager.shutdown()

    return asyncio.run(run())


def _run_mcp(args: argparse.Namespace) -> int:
    try:
        if args.mcp_command == "migrate":
            return migrate_mcp_config(
                args.user_id,
                args.device_identifier,
            )
        if args.mcp_command == "doctor":
            return doctor_mcp_servers(
                args.servers.split(","),
                args.user_id,
                args.device_identifier,
            )
        return 2
    except Exception as exc:
        print(f"MCP command failed: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "run"):
        return _run_server(args)
    if args.command == "doctor":
        return _run_doctor(args)
    if args.command == "mcp":
        return _run_mcp(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
