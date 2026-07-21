from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

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
    example_values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            example_values[name] = value

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


def test_router_uses_agent_config_model(monkeypatch) -> None:
    monkeypatch.setitem(agent_config.AGENT_CONFIG, "router", {"model": "configured-router"})
    monkeypatch.setattr(Router, "_init_gemini", lambda self: None)

    assert Router().model_name == "configured-router"


def test_production_code_avoids_deprecated_fastapi_422_name() -> None:
    violations: list[str] = []
    for root_name in ("app", "client_backend"):
        for path in (ROOT / root_name).rglob("*.py"):
            if "HTTP_422_UNPROCESSABLE_ENTITY" in path.read_text(encoding="utf-8"):
                violations.append(path.relative_to(ROOT).as_posix())

    assert not violations, f"deprecated HTTP 422 constant used in: {violations}"


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
