import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import skills as skills_api
from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.services import local_skills_registry as local_skills_registry_module
from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_runtime import install as install_module
from client_backend.services.skill_runtime.install import SkillBundleInstaller
from shared.skills.errors import (
    SKILL_INSTALL_CONFLICT,
    SKILL_INSTALL_INVALID,
    SKILL_MANIFEST_INVALID,
    UNSAFE_BUNDLE_PATH,
    SkillRuntimeError,
)

_USER_ID = "user-a"


def _valid_manifest_json(name: str = "demo-skill") -> str:
    """A minimal, provider-neutral skill.json payload."""
    return json.dumps(
        {
            "schema_version": "1.0",
            "name": name,
            "description": "Demo skill for install tests.",
            "runtime": {
                "type": "python_module",
                "module": "skills.demo.cli",
                "entrypoint": "cli",
            },
            "permissions": ["filesystem:read"],
            "capabilities": [
                {
                    "name": "do_thing",
                    "description": "Do the thing.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": ["do-thing"]},
                }
            ],
        }
    )


def _make_bundle(
    parent: "object",
    dir_name: str,
    *,
    front_matter_name: str | None = None,
    skill_json: str | None = None,
    with_skill_md: bool = True,
    body: str = "Body.",
):
    """Create a bundle directory containing SKILL.md and optional skill.json."""
    bundle = parent / dir_name
    bundle.mkdir(parents=True)
    if with_skill_md:
        if front_matter_name is not None:
            content = (
                f"---\nname: {front_matter_name}\n"
                f"description: Test skill {front_matter_name}.\n---\n\n"
                f"# {front_matter_name}\n\n{body}\n"
            )
        else:
            content = f"# {dir_name}\n\n{body}\n"
        (bundle / "SKILL.md").write_text(content, encoding="utf-8")
    if skill_json is not None:
        (bundle / "skill.json").write_text(skill_json, encoding="utf-8")
    return bundle


@pytest.fixture
def install_env(tmp_path, monkeypatch):
    """Set up a real profile root + fake user id, restoring profile_root after.

    Mirrors ``test_skills_registry.test_skill_enabled_state_is_isolated_per_user_profile``:
    it monkeypatches ``local_skills_registry_module.get_upstream_auth_service`` so the
    registry resolves a stable user id and installs land under a tmp profile root.
    """
    profile_root = tmp_path / "profiles"
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(profile_root)

    monkeypatch.setattr(
        local_skills_registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: _USER_ID),
    )
    # Keep the installer's post-install catalog refresh offline + deterministic
    # (no real runtime bridge / network) regardless of singleton state.
    monkeypatch.setattr(
        install_module,
        "get_runtime_bridge",
        lambda: SimpleNamespace(
            is_connected=lambda: False,
            get_registered_device_id=lambda: None,
        ),
    )

    configured_root = tmp_path / "configured_skills"
    configured_root.mkdir()
    source_root = tmp_path / "src"
    source_root.mkdir()

    try:
        yield SimpleNamespace(
            tmp_path=tmp_path,
            configured_root=configured_root,
            source_root=source_root,
        )
    finally:
        client_settings.profile_root = original_profile_root


def _new_registry(install_env) -> LocalSkillsRegistry:
    return LocalSkillsRegistry(skill_roots=[str(install_env.configured_root)])


def _install_root() -> "object":
    return get_profile_subdir(_USER_ID, "skills") / "installed"


@pytest.mark.asyncio
async def test_install_markdown_only_skill(install_env):
    source = _make_bundle(
        install_env.source_root, "notes-skill", front_matter_name="notes-skill"
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    result = await installer.install(source)

    assert result["name"] == "notes-skill"
    assert result["manifest_status"] == "instruction_only"

    install_root = _install_root()
    installed_dir = install_root / result["install_id"]
    assert installed_dir.is_dir()
    assert (installed_dir / "SKILL.md").exists()
    assert (installed_dir / "install.json").exists()

    skill = registry.get_skill("notes-skill")
    assert skill is not None
    assert skill.manifest is None

    catalog = registry.get_skill_catalog(include_content=True)
    names = [entry["name"] for entry in catalog["skills"]]
    assert "notes-skill" in names


@pytest.mark.asyncio
async def test_install_markdown_only_without_front_matter_uses_dir_name(install_env):
    source = _make_bundle(install_env.source_root, "plain-skill", front_matter_name=None)
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    result = await installer.install(source)

    assert result["name"] == "plain-skill"
    assert registry.get_skill("plain-skill") is not None


@pytest.mark.asyncio
async def test_install_manifest_backed_skill_no_path_leak(install_env):
    source = _make_bundle(
        install_env.source_root,
        "demo-skill",
        front_matter_name="demo-skill",
        skill_json=_valid_manifest_json("demo-skill"),
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    result = await installer.install(source)

    assert result["manifest_status"] == "manifest_present"

    skill = registry.get_skill("demo-skill")
    assert skill is not None
    assert skill.manifest is not None
    assert skill.install_metadata is not None
    assert skill.install_metadata["bundle_name"] == "demo-skill"

    sync_dict = skill.to_sync_dict()
    assert sync_dict["install"]["installed"] is True
    assert sync_dict["install"]["source"] == "profile"

    catalog = registry.get_skill_catalog(include_content=True)
    serialized_catalog = json.dumps(catalog)
    assert str(source.resolve()) not in serialized_catalog
    assert str(install_env.source_root.resolve()) not in serialized_catalog


@pytest.mark.asyncio
async def test_install_rejects_duplicate_active_name(install_env):
    source = _make_bundle(
        install_env.source_root, "dup-skill", front_matter_name="dup-skill"
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    await installer.install(source)
    assert registry.get_skill("dup-skill").enabled is True

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source)

    assert exc_info.value.code == SKILL_INSTALL_CONFLICT


@pytest.mark.asyncio
async def test_install_rejects_malformed_manifest(install_env):
    bad_manifest = json.dumps(
        {
            "schema_version": "1.0",
            "name": "bad-skill",
            "description": "Has an unknown runtime type.",
            "runtime": {"type": "totally_unknown"},
            "capabilities": [],
        }
    )
    source = _make_bundle(
        install_env.source_root,
        "bad-skill",
        front_matter_name="bad-skill",
        skill_json=bad_manifest,
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source)

    assert exc_info.value.code == SKILL_MANIFEST_INVALID
    # A rejected bundle must not have been copied into the profile. The
    # install root is only created once validation passes, so a rejection
    # this early leaves it either absent or empty.
    install_root = _install_root()
    assert not install_root.exists() or not any(install_root.iterdir())


@pytest.mark.asyncio
async def test_install_rejects_missing_skill_md(install_env):
    source = _make_bundle(
        install_env.source_root, "empty-skill", with_skill_md=False
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source)

    assert exc_info.value.code == SKILL_INSTALL_INVALID


@pytest.mark.asyncio
async def test_uninstall_removes_bundle_and_registry_entry(install_env):
    source = _make_bundle(
        install_env.source_root, "gone-skill", front_matter_name="gone-skill"
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    result = await installer.install(source)
    installed_dir = _install_root() / result["install_id"]
    assert installed_dir.is_dir()
    assert registry.get_skill("gone-skill") is not None

    await installer.uninstall("gone-skill")

    assert not installed_dir.exists()
    assert registry.get_skill("gone-skill") is None


@pytest.mark.asyncio
async def test_uninstall_unknown_name_raises_invalid(install_env):
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.uninstall("never-installed")

    assert exc_info.value.code == SKILL_INSTALL_INVALID


@pytest.mark.asyncio
async def test_install_rejects_front_matter_without_name(install_env):
    # Front matter is present (---...---) but declares no name — malformed.
    bundle = install_env.source_root / "nameless"
    bundle.mkdir(parents=True)
    (bundle / "SKILL.md").write_text(
        "---\ndescription: No name here.\n---\n\n# Nameless\n\nBody.\n", encoding="utf-8"
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(bundle)

    assert exc_info.value.code == SKILL_INSTALL_INVALID


@pytest.mark.asyncio
async def test_install_rejects_symlink_in_bundle(install_env):
    # A symlink in the bundle could point outside it (copying arbitrary files
    # into the installed skill) or form a cycle — it must be rejected.
    outside_secret = install_env.tmp_path / "outside_secret.txt"
    outside_secret.write_text("secret payload", encoding="utf-8")

    source = _make_bundle(
        install_env.source_root, "sneaky-skill", front_matter_name="sneaky-skill"
    )
    link_path = source / "escape.txt"
    try:
        os.symlink(outside_secret, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    with pytest.raises(SkillRuntimeError) as exc_info:
        await installer.install(source)

    assert exc_info.value.code == UNSAFE_BUNDLE_PATH
    # Nothing was copied into the profile for a rejected bundle.
    install_root = _install_root()
    assert not install_root.exists() or not any(install_root.iterdir())


def test_compute_source_hash_rejects_symlink_cross_platform(install_env, monkeypatch):
    # Guarantees the symlink-rejection branch is exercised on every platform,
    # including Windows without symlink privileges (where the integration test
    # above skips). Mocks the OS boundary (is_symlink) rather than the logic.
    source = _make_bundle(install_env.source_root, "linky", front_matter_name="linky")

    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "SKILL.md")

    with pytest.raises(SkillRuntimeError) as exc_info:
        install_module._compute_source_hash(source)

    assert exc_info.value.code == UNSAFE_BUNDLE_PATH


@pytest.mark.asyncio
async def test_reinstall_over_disabled_bundle_replaces_and_stays_discoverable(install_env):
    source = _make_bundle(
        install_env.source_root,
        "swap-skill",
        front_matter_name="swap-skill",
        body="Original body.",
    )
    registry = _new_registry(install_env)
    installer = SkillBundleInstaller(registry=registry)

    first = await installer.install(source)
    # Disabling the active skill is what the plan says unblocks a reinstall.
    registry.set_skill_enabled("swap-skill", False)

    # Reinstall with MODIFIED content (new source hash -> new install dir).
    (source / "SKILL.md").write_text(
        "---\nname: swap-skill\ndescription: Test skill swap-skill.\n---\n\n"
        "# swap-skill\n\nUpdated body.\n",
        encoding="utf-8",
    )
    second = await installer.install(source)

    assert second["source_hash"] != first["source_hash"]

    # The stale bundle was replaced, not orphaned: exactly one installed dir.
    install_root = _install_root()
    installed_dirs = [entry for entry in install_root.iterdir() if entry.is_dir()]
    assert len(installed_dirs) == 1
    assert installed_dirs[0].name == second["install_id"]

    # The skill is still discoverable under its name and reflects the new body.
    skill = registry.get_skill("swap-skill")
    assert skill is not None
    assert "Updated body." in skill.content


# --- API layer -----------------------------------------------------------


class _InstallerStub:
    def __init__(self, *, result: dict | None = None, error: Exception | None = None):
        self._result = result or {}
        self._error = error
        self.install_calls: list[str] = []

    async def install(self, source):
        self.install_calls.append(source)
        if self._error is not None:
            raise self._error
        return self._result


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(skills_api.router)
    app.dependency_overrides[skills_api.require_local_session] = lambda: object()
    return app


def test_install_endpoint_returns_success_envelope(monkeypatch):
    stub = _InstallerStub(
        result={
            "name": "demo",
            "install_id": "demo-abc123456789",
            "source_hash": "deadbeef",
            "manifest_status": "instruction_only",
        }
    )
    monkeypatch.setattr(skills_api, "get_skill_installer", lambda: stub)

    with TestClient(_build_app()) as client:
        response = client.post("/skills/install", json={"source_path": "/some/bundle"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["name"] == "demo"
    assert stub.install_calls == ["/some/bundle"]


def test_install_endpoint_maps_conflict_to_409(monkeypatch):
    stub = _InstallerStub(
        error=SkillRuntimeError(SKILL_INSTALL_CONFLICT, "a skill named 'demo' is already active")
    )
    monkeypatch.setattr(skills_api, "get_skill_installer", lambda: stub)

    with TestClient(_build_app()) as client:
        response = client.post("/skills/install", json={"source_path": "/some/bundle"})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == SKILL_INSTALL_CONFLICT
