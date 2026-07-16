import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import skills as skills_api
from client_backend.core.config import client_settings
from client_backend.core.paths import get_installed_skills_root
from client_backend.services import local_skills_registry as registry_module
from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_runtime import install as install_module
from client_backend.services.skill_runtime.install import SkillBundleInstaller
from shared.skills.errors import (
    SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID,
    SKILL_SETUP_REQUIRED,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)

USER_ID = "user-a"


def _write_skill(path: Path, *, name: str = "demo-skill") -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo skill.\n---\n\nUse `{name}-cli`.\n",
        encoding="utf-8",
    )
    return path


class _EnvironmentManager:
    def __init__(self):
        self.prepared = []
        self.removed = []

    def preview(self, skill):
        python_project = bool(skill.executable_assets["python_project"])
        return {
            "python_project": python_project,
            "dependencies": ["example>=1"] if python_project else [],
            "build_requirements": [],
            "declared_commands": ["demo-skill-cli"] if python_project else [],
            "confirmation_required": python_project,
        }

    def prepare(self, skill, *, approve_setup, force=False):
        if skill.executable_assets["python_project"] and not approve_setup:
            raise SkillRuntimeError(SKILL_SETUP_REQUIRED, "approval required")
        self.prepared.append((skill.name, skill.bundle_root, approve_setup, force))
        return {"status": "ready", "commands": ["demo-skill-cli"]}

    def inspect(self, skill):
        return {"status": "setup_required", "commands": []}

    def remove_skill(self, name):
        self.removed.append(name)


class _SecretStore:
    def __init__(self):
        self.removed = []

    def remove_skill(self, name):
        self.removed.append(name)
        return True


@pytest.fixture
def install_env(tmp_path, monkeypatch):
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profiles")
    monkeypatch.setattr(
        registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: USER_ID),
    )
    monkeypatch.setattr(
        install_module,
        "get_runtime_bridge",
        lambda: SimpleNamespace(
            is_connected=lambda: False,
            get_registered_device_id=lambda: None,
        ),
    )
    configured = tmp_path / "configured"
    configured.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    environment = _EnvironmentManager()
    secrets = _SecretStore()
    try:
        yield SimpleNamespace(
            configured=configured,
            sources=sources,
            environment=environment,
            secrets=secrets,
        )
    finally:
        client_settings.profile_root = original_profile_root


def _installer(install_env):
    registry = LocalSkillsRegistry(skill_roots=[str(install_env.configured)])
    return registry, SkillBundleInstaller(
        registry=registry,
        environment_manager=install_env.environment,
        secret_store=install_env.secrets,
    )


@pytest.mark.asyncio
async def test_preview_discovers_direct_portable_bundle_without_local_paths(install_env):
    source = _write_skill(install_env.sources / "demo")
    (source / "bin").mkdir()
    (source / "bin" / "demo-skill-cli.py").write_text("print('ok')\n", encoding="utf-8")
    _, installer = _installer(install_env)

    preview = await installer.preview(source)

    assert preview["name"] == "demo-skill"
    assert len(preview["source_hash"]) == 64
    assert preview["bundle_shape"] == "direct"
    assert preview["executable_assets"]["bin"] == ["demo-skill-cli.py"]
    assert preview["setup"]["confirmation_required"] is False
    assert str(source) not in json.dumps(preview)


@pytest.mark.asyncio
async def test_install_copies_complete_nested_bundle_and_preserves_bundle_root(install_env):
    source = install_env.sources / "plugin"
    _write_skill(source / "skills" / "demo-skill")
    (source / "bin").mkdir(parents=True)
    (source / "bin" / "demo-skill-cli.py").write_text("print('ok')\n", encoding="utf-8")
    registry, installer = _installer(install_env)

    preview = await installer.preview(source)
    result = await installer.install(
        source,
        expected_source_hash=preview["source_hash"],
        approve_setup=False,
    )

    installed = get_installed_skills_root(USER_ID) / result["install_id"]
    assert (installed / "skills" / "demo-skill" / "SKILL.md").is_file()
    assert (installed / "bin" / "demo-skill-cli.py").is_file()
    skill = registry.get_skill("demo-skill")
    assert skill is not None
    assert skill.bundle_root == installed.resolve()
    assert result["runtime_status"] == "ready"


@pytest.mark.asyncio
async def test_install_rejects_changed_source_hash_before_copy(install_env):
    source = _write_skill(install_env.sources / "demo")
    _, installer = _installer(install_env)
    preview = await installer.preview(source)
    (source / "SKILL.md").write_text("# changed\n", encoding="utf-8")

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source, expected_source_hash=preview["source_hash"])

    assert exc_info.value.code == SKILL_INSTALL_INVALID
    install_root = get_installed_skills_root(USER_ID)
    assert not install_root.exists() or not any(install_root.iterdir())


@pytest.mark.asyncio
async def test_install_rejects_bundle_changed_while_it_is_being_copied(install_env, monkeypatch):
    source = _write_skill(install_env.sources / "demo")
    (source / "bin").mkdir()
    (source / "bin" / "demo-skill-cli.py").write_text("print('original')\n", encoding="utf-8")
    _, installer = _installer(install_env)
    preview = await installer.preview(source)
    real_copytree = install_module.shutil.copytree

    def copy_then_tamper(src, dst, **kwargs):
        monkeypatch.setattr(install_module.shutil, "copytree", real_copytree)
        result = real_copytree(src, dst, **kwargs)
        (Path(dst) / "bin" / "demo-skill-cli.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(install_module.shutil, "copytree", copy_then_tamper)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source, expected_source_hash=preview["source_hash"])

    assert exc_info.value.code == SKILL_INSTALL_INVALID
    install_root = get_installed_skills_root(USER_ID)
    assert not install_root.exists() or not any(install_root.iterdir())


@pytest.mark.asyncio
async def test_python_project_requires_approval_then_prepares_installed_copy(install_env):
    source = _write_skill(install_env.sources / "demo")
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    _, installer = _installer(install_env)
    preview = await installer.preview(source)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source, expected_source_hash=preview["source_hash"])
    assert exc_info.value.code == SKILL_SETUP_REQUIRED

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source, approve_setup=True)
    assert exc_info.value.code == SKILL_INSTALL_INVALID

    result = await installer.install(
        source,
        expected_source_hash=preview["source_hash"],
        approve_setup=True,
    )

    assert result["runtime_status"] == "ready"
    _, prepared_root, approved, _ = install_env.environment.prepared[0]
    assert approved is True
    assert prepared_root.parent == get_installed_skills_root(USER_ID)


@pytest.mark.asyncio
async def test_setup_rehashes_installed_bundle_and_requires_expected_hash(install_env):
    source = _write_skill(install_env.sources / "demo")
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    registry, installer = _installer(install_env)
    preview = await installer.preview(source)
    await installer.install(
        source,
        expected_source_hash=preview["source_hash"],
        approve_setup=True,
    )
    installed_skill = registry.get_skill("demo-skill")

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.setup(
            "demo-skill",
            expected_source_hash=None,
            approve_setup=True,
        )
    assert exc_info.value.code == SKILL_INSTALL_INVALID

    (installed_skill.bundle_root / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: changed\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.setup(
            "demo-skill",
            expected_source_hash=preview["source_hash"],
            approve_setup=True,
        )
    assert exc_info.value.code == SKILL_INSTALL_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 2])
async def test_install_rejects_zero_or_multiple_skills(install_env, count):
    source = install_env.sources / f"bundle-{count}"
    source.mkdir()
    for index in range(count):
        _write_skill(source / "skills" / f"skill-{index}", name=f"skill-{index}")
    _, installer = _installer(install_env)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.preview(source)

    assert exc_info.value.code == SKILL_INSTALL_INVALID


@pytest.mark.asyncio
async def test_preview_rejects_nonstandard_skill_name(install_env):
    source = _write_skill(install_env.sources / "invalid", name="invalid::skill")
    _, installer = _installer(install_env)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.preview(source)

    assert exc_info.value.code == SKILL_INSTALL_INVALID
    assert "name" in exc_info.value.message.lower()


@pytest.mark.asyncio
async def test_duplicate_enabled_skill_is_rejected(install_env):
    source = _write_skill(install_env.sources / "demo")
    registry, installer = _installer(install_env)
    await installer.install(source)
    assert registry.get_skill("demo-skill").enabled is True

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source)

    assert exc_info.value.code == SKILL_INSTALL_CONFLICT


@pytest.mark.asyncio
async def test_uninstall_removes_bundle_runtime_and_registry_entry(install_env):
    source = _write_skill(install_env.sources / "demo")
    registry, installer = _installer(install_env)
    result = await installer.install(source)
    installed = get_installed_skills_root(USER_ID) / result["install_id"]

    await installer.uninstall("demo-skill")

    assert not installed.exists()
    assert registry.get_skill("demo-skill") is None
    assert install_env.environment.removed == ["demo-skill"]
    assert install_env.secrets.removed == ["demo-skill"]


@pytest.mark.asyncio
async def test_install_rejects_symlink_in_bundle(install_env):
    source = _write_skill(install_env.sources / "demo")
    outside = install_env.sources / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        os.symlink(outside, source / "escape.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _, installer = _installer(install_env)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.preview(source)

    assert exc_info.value.code == UNSAFE_BUNDLE_PATH


class _InstallerStub:
    def __init__(self):
        self.preview_calls = []
        self.install_calls = []

    async def preview(self, source):
        self.preview_calls.append(source)
        return {"name": "demo", "source_hash": "abc", "setup": {}}

    async def install(self, source, *, expected_source_hash=None, approve_setup=False):
        self.install_calls.append((source, expected_source_hash, approve_setup))
        return {
            "name": "demo",
            "install_id": "demo-abc",
            "source_hash": "abc",
            "runtime_status": "ready",
        }


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(skills_api.router)
    app.dependency_overrides[skills_api.require_local_session] = lambda: object()
    return app


def test_install_preview_and_confirmed_install_endpoints(monkeypatch):
    stub = _InstallerStub()
    monkeypatch.setattr(skills_api, "get_skill_installer", lambda: stub)

    with TestClient(_build_app()) as client:
        preview = client.post("/skills/install/preview", json={"source_path": "C:/bundle"})
        install = client.post(
            "/skills/install",
            json={
                "source_path": "C:/bundle",
                "expected_source_hash": "abc",
                "approve_setup": True,
            },
        )

    assert preview.status_code == 200
    assert install.status_code == 200
    assert stub.preview_calls == ["C:/bundle"]
    assert stub.install_calls == [("C:/bundle", "abc", True)]
