from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from app.ai import agent_config
from app.ai.agents.router import Router
from app.core.config import Settings

ROOT = Path(__file__).resolve().parents[1]
TRACKED_AUDIT_ROOTS = (
    ".env.example",
    "README.md",
    "app",
    "client_backend",
    "docs",
    "scripts",
    "shared",
    "skills",
)


def _tracked_audit_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", *TRACKED_AUDIT_ROOTS],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in completed.stdout.split(b"\0") if item]


def _env_example_values() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            values[name] = value
    return values


def _env_default(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def test_tracked_runtime_and_docs_do_not_embed_developer_home_paths() -> None:
    windows_home = re.compile(r"(?i)[a-z]:[\\/](?:users|documents and settings)[\\/][^\\/\s]+")
    posix_home = re.compile(r"(?:/home/[^/\s]+|/Users/[^/\s]+)")
    violations: list[str] = []

    for path in _tracked_audit_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if windows_home.search(text) or posix_home.search(text):
            violations.append(path.relative_to(ROOT).as_posix())

    assert not violations, f"machine-specific home paths found in tracked files: {violations}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows event-loop policy regression")
def test_importing_server_app_preserves_event_loop_policy() -> None:
    script = """
import asyncio

policy = asyncio.WindowsProactorEventLoopPolicy()
asyncio.set_event_loop_policy(policy)
import app.main

assert asyncio.get_event_loop_policy() is policy
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_model_defaults_are_settings_backed() -> None:
    configured = Settings(
        _env_file=None,
        router_model="router-model",
        image_generator_tool_model="image-tool-model",
        canvas_agent_model="canvas-model",
        suggestion_model="suggestion-model",
        title_generator_model="title-model",
    )

    assert configured.router_model == "router-model"
    assert configured.image_generator_tool_model == "image-tool-model"
    assert configured.canvas_agent_model == "canvas-model"
    assert configured.suggestion_model == "suggestion-model"
    assert configured.title_generator_model == "title-model"

    source = (ROOT / "app" / "ai" / "agent_config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "AGENT_CONFIG" for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Dict)

    model_values: list[ast.expr] = []
    for config_node in assignment.value.values:
        assert isinstance(config_node, ast.Dict)
        for key, value in zip(config_node.keys, config_node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value in {"model", "langchain_model"}:
                model_values.append(value)

    assert model_values
    assert all(
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "settings"
        for value in model_values
    )


def test_model_env_example_matches_runtime_defaults() -> None:
    example_values = _env_example_values()

    setting_names = (
        "rag_agent_model",
        "chat_agent_model",
        "search_agent_model",
        "form_filler_model",
        "router_model",
        "image_generator_tool_model",
        "canvas_agent_model",
        "suggestion_model",
        "title_generator_model",
        "image_generator_model",
        "image_caption_model",
    )
    for setting_name in setting_names:
        env_name = setting_name.upper()
        assert example_values.get(env_name) == Settings.model_fields[setting_name].default


def test_every_model_usage_setting_is_discoverable_with_accurate_default() -> None:
    usage_fields = {
        name: field
        for name, field in Settings.model_fields.items()
        if name.startswith("model_usage_")
    }
    example_values = _env_example_values()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert len(usage_fields) == 17
    for name, field in usage_fields.items():
        env_name = name.upper()
        expected = _env_default(field.default)
        assert example_values.get(env_name) == expected
        assert f"| `{env_name}` | `{expected or 'blank'}` |" in readme

    assert example_values["MODEL_USAGE_USER_HASH_SECRET"] == ""
    assert "app/core/config.py::Settings" in readme


def test_model_usage_operator_constraints_are_documented() -> None:
    env_text = " ".join(
        (ROOT / ".env.example").read_text(encoding="utf-8").replace("#", "").split()
    )
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())

    required_guidance = (
        "reconcile window must be shorter than raw retention",
        "rollup retention must be at least raw retention",
        "failure-store TTL must cover the health window plus 60 seconds",
        "unhealthy rollup lag must be at least degraded rollup lag",
        "MODEL_USAGE_USER_HASH_SECRET must be set in production when LangSmith tracing is enabled",
        "retention, reconciliation, cleanup, retry, lookback, failure-window, "
        "and TTL integers must be positive",
        "unattributed ratio must be between 0 and 1",
        "rollup lag thresholds must be nonnegative",
        "failure window cannot exceed 3600 seconds",
        "failure-store TTL must be between 60 and 86400 seconds",
        "failure-store timeout must be positive",
    )
    for guidance in required_guidance:
        assert guidance in env_text
        assert guidance in readme


def test_router_uses_agent_config_model(monkeypatch) -> None:
    monkeypatch.setitem(agent_config.AGENT_CONFIG, "router", {"model": "configured-router"})
    monkeypatch.setattr(Router, "_init_gemini", lambda self: None)

    assert Router().model_name == "configured-router"


def test_validation_error_paths_do_not_depend_on_starlette_422_names() -> None:
    script = """
import asyncio

from fastapi import HTTPException, status

for name in ("HTTP_422_UNPROCESSABLE_CONTENT", "HTTP_422_UNPROCESSABLE_ENTITY"):
    status.__dict__.pop(name, None)

from app.core.exceptions.validation import ValidationException
from client_backend.api import auth, mcp

assert ValidationException().status_code == 422

calls = (
    lambda: mcp._parse_server_url_payload({}),
    lambda: asyncio.run(mcp.add_mcp_server({}, None)),
)
for call in calls:
    try:
        call()
    except HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("expected HTTP 422")

class InvalidRequest:
    password = "unused"
    def resolved_email(self):
        raise ValueError("invalid login")

auth.get_upstream_auth_service = lambda: object()
try:
    asyncio.run(auth.login(InvalidRequest()))
except HTTPException as exc:
    assert exc.status_code == 422
else:
    raise AssertionError("expected HTTP 422")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_production_no_longer_depends_on_sunset_langchain_community() -> None:
    source_violations: list[str] = []
    for path in (ROOT / "app").rglob("*.py"):
        if "langchain_community" in path.read_text(encoding="utf-8"):
            source_violations.append(path.relative_to(ROOT).as_posix())

    manifest_violations = [
        name
        for name in ("pyproject.toml", "requirements.txt", "environment.yml")
        if "langchain-community" in (ROOT / name).read_text(encoding="utf-8").lower()
    ]

    assert not source_violations, f"sunset imports remain in: {source_violations}"
    assert not manifest_violations, f"sunset direct dependencies remain in: {manifest_violations}"


def test_default_stdio_mcp_servers_only_launch_tracked_portable_scripts() -> None:
    config = json.loads((ROOT / "app" / "ai" / "mcp_config.json").read_text(encoding="utf-8"))
    tracked = {
        path.replace("\\", "/")
        for path in subprocess.run(
            ["git", "ls-files", "--", "app/ai/mcp_servers"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    }

    for name, server in config["mcp_servers"].items():
        if server.get("transport") != "stdio":
            continue
        command = str(server.get("command") or "")
        args = [str(value) for value in server.get("args", [])]
        launcher_tokens = {command.lower(), *(value.lower() for value in args)}
        assert command
        assert not PurePosixPath(command).is_absolute()
        assert not PureWindowsPath(command).is_absolute()
        assert not {"cmd", "conda", "powershell", "pwsh", "/c"}.intersection(launcher_tokens), (
            f"{name} contains a developer launcher chain: {args!r}"
        )

        script_args = [value for value in args if Path(value).suffix in {".js", ".mjs", ".py"}]
        assert len(script_args) == 1, f"{name} must identify one server script"
        script = Path(script_args[0])
        assert not PurePosixPath(script_args[0]).is_absolute()
        assert not PureWindowsPath(script_args[0]).is_absolute()
        normalized = script.as_posix()
        assert normalized in tracked, f"{name} references an untracked server: {normalized}"
        assert (ROOT / script).is_file(), f"{name} references a missing server: {normalized}"


def test_readme_only_claims_tracked_mcp_assets_are_bundled() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "boring_reader" not in readme
    assert "ships its own Tesseract" not in readme
    assert "YOLO artifacts" not in readme


def test_httpx2_testclient_dependency_is_declared_in_every_manifest() -> None:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8").splitlines()

    assert "httpx2>=2.0.0,<3.0.0" in dependencies
    assert "httpx>=0.25.0" in dependencies
    assert "httpx2==2.0.0" in requirements
    assert "httpx==0.28.1" in requirements
    assert "      - httpx2==2.0.0" in environment
    assert any(line.strip().startswith("- httpx=") for line in environment)


def test_fastapi_testclient_uses_httpx2_without_deprecation_warning() -> None:
    script = """
import warnings

warnings.simplefilter("error", DeprecationWarning)

from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

with TestClient(app) as client:
    response = client.get("/health")

assert response.status_code == 200
assert response.json() == {"status": "ok"}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_plain_text_loader_returns_langchain_document_with_source(tmp_path: Path) -> None:
    from app.services.plain_text_loader import load_utf8_text_document

    source = tmp_path / "notes.txt"
    source.write_text("Xin chào", encoding="utf-8")

    document = load_utf8_text_document(source)

    assert document.page_content == "Xin chào"
    assert document.metadata == {"source": str(source)}


def test_plain_text_loader_preserves_loader_error_boundary(tmp_path: Path) -> None:
    from app.services.plain_text_loader import load_utf8_text_document

    source = tmp_path / "invalid.txt"
    source.write_bytes(b"\xff")

    with pytest.raises(RuntimeError, match=re.escape(f"Error loading {source}")):
        load_utf8_text_document(source)
