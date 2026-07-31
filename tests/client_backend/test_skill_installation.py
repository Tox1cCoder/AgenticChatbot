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
    SKILL_CONFIGURED_ROOT_CONFLICT,
    SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID,
    SKILL_SETUP_REQUIRED,
    SKILL_SOURCE_CHANGED,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)

USER_ID = "user-a"


def _write_skill(path: Path, *, name: str = "demo-skill", body: str | None = None) -> Path:
    """Write a minimal valid bundle. ``body`` distinguishes versions by content."""
    path.mkdir(parents=True)
    instructions = body if body is not None else f"Use `{name}-cli`."
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Demo skill.\n---\n\n{instructions}\n",
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
    # install.py no longer touches the runtime bridge at all; a bridge double
    # here would hide a regression rather than prevent one.
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

    async def install(
        self,
        source,
        *,
        expected_source_hash=None,
        approve_setup=False,
        replace_source_hash=None,
        source_kind="path",
        observer=None,
    ):
        self.install_calls.append(
            (source, expected_source_hash, approve_setup, replace_source_hash, source_kind)
        )
        return {
            "name": "demo",
            "install_id": "demo-abc",
            "source_hash": "abc",
            "runtime_status": "ready",
            "action": "installed",
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
    assert stub.install_calls == [("C:/bundle", "abc", True, None, "path")]


@pytest.mark.asyncio
async def test_existing_profile_skill_requires_matching_replace_hash(install_env):
    """A name collision is never resolved silently.

    Before this guard an existing *disabled* skill was overwritten without any
    confirmation. Now every collision needs the caller to name the exact hash it
    intends to replace.
    """
    first = _write_skill(install_env.sources / "v1", body="v1")
    second = _write_skill(install_env.sources / "v2", body="v2")
    registry, installer = _installer(install_env)
    installed = await installer.install(first)
    old_hash = installed["source_hash"]
    assert installed["action"] == "installed"

    with pytest.raises(SkillRuntimeError) as conflict:
        await installer.install(second)
    assert conflict.value.code == SKILL_INSTALL_CONFLICT

    updated = await installer.install(second, replace_source_hash=old_hash)

    assert updated["action"] == "updated"
    assert "v2" in registry.get_skill("demo-skill").content


@pytest.mark.asyncio
async def test_disabled_existing_skill_is_also_protected(install_env):
    source = _write_skill(install_env.sources / "v1", body="v1")
    replacement = _write_skill(install_env.sources / "v2", body="v2")
    registry, installer = _installer(install_env)
    await installer.install(source)
    registry.set_skill_enabled("demo-skill", False)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(replacement)

    assert exc_info.value.code == SKILL_INSTALL_CONFLICT


@pytest.mark.asyncio
async def test_update_rejects_stale_hash_and_preserves_previous_bundle(install_env):
    first = _write_skill(install_env.sources / "v1", body="v1")
    second = _write_skill(install_env.sources / "v2", body="v2")
    registry, installer = _installer(install_env)
    await installer.install(first)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(second, replace_source_hash="0" * 64)

    assert exc_info.value.code == SKILL_SOURCE_CHANGED
    assert "v1" in registry.get_skill("demo-skill").content


@pytest.mark.asyncio
async def test_replace_hash_without_an_installed_skill_is_rejected(install_env):
    source = _write_skill(install_env.sources / "demo")
    _, installer = _installer(install_env)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source, replace_source_hash="0" * 64)

    assert exc_info.value.code == SKILL_SOURCE_CHANGED


@pytest.mark.asyncio
async def test_failed_update_preserves_the_previous_bundle_and_runtime(install_env):
    first = _write_skill(install_env.sources / "v1", body="v1")
    second = _write_skill(install_env.sources / "v2", body="v2")
    (second / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    registry, installer = _installer(install_env)
    installed = await installer.install(first)

    # Runtime preparation is the realistic mid-install failure: it runs after the
    # copy but before the promotion.
    def failing_prepare(skill, *, approve_setup, force=False):
        raise SkillRuntimeError("SKILL_SETUP_FAILED", "dependency install failed")

    install_env.environment.prepare = failing_prepare
    preview = await installer.preview(second)

    with pytest.raises(SkillRuntimeError):
        await installer.install(
            second,
            expected_source_hash=preview["source_hash"],
            approve_setup=True,
            replace_source_hash=installed["source_hash"],
        )

    surviving = registry.get_skill("demo-skill")
    assert "v1" in surviving.content
    assert surviving.source_hash == installed["source_hash"]
    assert (get_installed_skills_root(USER_ID) / installed["install_id"]).is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_configured_root_skill_cannot_be_replaced(install_env, enabled):
    """A skills root the user manages is never rewritten by an install."""
    _write_skill(install_env.configured / "demo", body="configured")
    registry, installer = _installer(install_env)
    await registry.initialize()
    registry.set_skill_enabled("demo-skill", enabled)
    replacement = _write_skill(install_env.sources / "replacement", body="new")

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(
            replacement,
            replace_source_hash=registry.get_skill("demo-skill").source_hash,
        )

    assert exc_info.value.code == SKILL_CONFIGURED_ROOT_CONFLICT
    assert "configured" in registry.get_skill("demo-skill").content


@pytest.mark.asyncio
async def test_upload_provenance_never_persists_staging_path(install_env):
    source = _write_skill(install_env.sources / "upload")
    _, installer = _installer(install_env)

    result = await installer.install(source, source_kind="upload")

    metadata = install_module._read_install_metadata(
        get_installed_skills_root(USER_ID) / result["install_id"]
    )
    assert metadata["source"] == "upload"
    assert "source_path" not in metadata
    assert str(source) not in json.dumps(metadata)


@pytest.mark.asyncio
async def test_path_install_still_records_its_source_path(install_env):
    source = _write_skill(install_env.sources / "demo")
    _, installer = _installer(install_env)

    result = await installer.install(source)

    metadata = install_module._read_install_metadata(
        get_installed_skills_root(USER_ID) / result["install_id"]
    )
    assert metadata["source"] == "profile"
    assert metadata["source_path"] == str(source)


@pytest.mark.asyncio
async def test_observer_reports_phases_and_gates_the_commit(install_env):
    source = _write_skill(install_env.sources / "demo")
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    _, installer = _installer(install_env)
    preview = await installer.preview(source)
    phases: list[str] = []
    commits: list[str] = []

    class _Observer:
        async def phase(self, name):
            phases.append(name)

        async def before_commit(self):
            commits.append("before_commit")

    await installer.install(
        source,
        expected_source_hash=preview["source_hash"],
        approve_setup=True,
        observer=_Observer(),
    )

    assert phases == [
        "validating",
        "waitingForLock",
        "copying",
        "preparingRuntime",
        "committing",
    ]
    assert commits == ["before_commit"]


@pytest.mark.asyncio
async def test_observer_cancellation_before_commit_installs_nothing(install_env):
    source = _write_skill(install_env.sources / "demo")
    registry, installer = _installer(install_env)

    class _CancellingObserver:
        async def phase(self, name):
            return None

        async def before_commit(self):
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        await installer.install(source, observer=_CancellingObserver())

    await registry.initialize()
    assert registry.get_skill("demo-skill") is None
    install_root = get_installed_skills_root(USER_ID)
    assert not install_root.exists() or not any(install_root.iterdir())


@pytest.mark.asyncio
async def test_failed_cleanup_is_resumable_without_changing_the_404_contract(install_env):
    source = _write_skill(install_env.sources / "demo")
    _, installer = _installer(install_env)
    await installer.install(source)

    def failing_remove(name):
        raise RuntimeError("runtime directory is locked")

    install_env.environment.remove_skill = failing_remove

    first = await installer.uninstall("demo-skill")
    assert first["removed"] is True
    assert first["cleanup_status"] == "pending"

    install_env.environment.remove_skill = install_env.environment.removed.append
    second = await installer.uninstall("demo-skill")
    assert second["removed"] is False
    assert second["cleanup_status"] == "complete"
    assert install_env.environment.removed == ["demo-skill"]

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.uninstall("demo-skill")
    assert exc_info.value.code == SKILL_INSTALL_INVALID


@pytest.mark.asyncio
async def test_successful_uninstall_leaves_no_cleanup_receipt(install_env):
    source = _write_skill(install_env.sources / "demo")
    _, installer = _installer(install_env)
    await installer.install(source)

    result = await installer.uninstall("demo-skill")

    assert result == {"name": "demo-skill", "removed": True, "cleanup_status": "complete"}
    operations_root = install_module.get_skill_operations_root(USER_ID)
    assert not operations_root.exists() or not any(operations_root.iterdir())


@pytest.mark.asyncio
async def test_install_setup_and_uninstall_never_touch_the_runtime_bridge(install_env):
    """Catalog synchronization belongs to the catalog service, not the installer.

    A bridge call here would let a network failure fail an install that has
    already been committed locally, so the symbol must not even be reachable.
    """
    source = _write_skill(install_env.sources / "demo")
    (source / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    _, installer = _installer(install_env)
    preview = await installer.preview(source)

    assert "get_runtime_bridge" not in SkillBundleInstaller.install.__globals__

    await installer.install(
        source,
        expected_source_hash=preview["source_hash"],
        approve_setup=True,
    )
    installed = installer._registry.get_skill("demo-skill")
    await installer.setup(
        "demo-skill",
        expected_source_hash=installed.source_hash,
        approve_setup=True,
    )
    await installer.uninstall("demo-skill")
