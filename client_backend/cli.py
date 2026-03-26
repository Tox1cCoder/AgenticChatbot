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
    parser = argparse.ArgumentParser(prog="codex-client-backend")
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

    return parser


def _run_doctor(args: argparse.Namespace) -> int:
    _apply_env_overrides(args)

    from client_backend.core.config import get_client_settings
    from client_backend.services.local_mcp_manager import LocalMCPManager

    settings = get_client_settings()
    mcp_manager = LocalMCPManager()

    parsed_server = urlparse(settings.server_api_base_url)
    checks = {
        "server_api_base_url_has_scheme": bool(parsed_server.scheme and parsed_server.netloc),
        "profile_root_exists": Path(settings.profile_root).exists(),
        "skills_roots_exist": all(
            Path(root).expanduser().exists() for root in settings.skills_roots
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
            "mcp_config_path": str(mcp_manager.config_path),
            "skills_roots": settings.skills_roots,
            "allowed_shells": settings.allowed_shells,
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "run"):
        return _run_server(args)
    if args.command == "doctor":
        return _run_doctor(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
