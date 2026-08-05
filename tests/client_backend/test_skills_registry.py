import json
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from client_backend.core.config import client_settings
from client_backend.core.paths import (
    get_profile_subdir,
    profile_subdir_path,
    resolve_skills_root,
)
from client_backend.services import local_skills_registry as local_skills_registry_module
from client_backend.services.local_skills_registry import LocalSkillsRegistry, SkillMetadata
from client_backend.services.skill_runtime.install import _compute_source_hash as install_hash


def _write_skill(skill_dir: Path, *, name: str = "demo-skill") -> Path:
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test bundled commands.\n---\n\n"
        f"Use `{name}-cli --json`.\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.mark.asyncio
async def test_direct_bundle_records_root_hash_and_executable_assets(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = _write_skill(skill_root / "demo-skill")
    (skill_dir / "bin").mkdir()
    (skill_dir / "bin" / "demo-skill-cli.py").write_text("print('ok')\n", encoding="utf-8")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "inspect.py").write_text("print('inspect')\n", encoding="utf-8")

    registry = LocalSkillsRegistry(skill_root=str(skill_root))
    await registry.initialize()

    skill = registry.get_skill("demo-skill")
    assert skill is not None
    assert skill.bundle_root == skill_dir.resolve()
    assert len(skill.source_hash) == 64
    assert skill.executable_assets == {
        "bin": ["demo-skill-cli.py"],
        "scripts": ["scripts/inspect.py"],
        "python_project": False,
    }


@pytest.mark.asyncio
async def test_nested_installed_bundle_uses_install_marker_as_authoritative_root(tmp_path):
    install_root = tmp_path / "installed"
    bundle_root = install_root / "demo-skill-abc123"
    skill_dir = _write_skill(bundle_root / "skills" / "demo-skill")
    (bundle_root / "bin").mkdir(parents=True)
    (bundle_root / "bin" / "demo-skill-cli.py").write_text("print('ok')\n", encoding="utf-8")
    (bundle_root / "install.json").write_text(
        json.dumps(
            {
                "installed": True,
                "bundle_name": "demo-skill",
                "source_hash": "a" * 64,
                "source": "profile",
            }
        ),
        encoding="utf-8",
    )

    registry = LocalSkillsRegistry(skill_root=str(install_root))
    await registry.initialize()

    skill = registry.get_skill("demo-skill")
    assert skill is not None
    assert skill.path == skill_dir / "SKILL.md"
    assert skill.bundle_root == bundle_root.resolve()
    assert skill.source_hash != "a" * 64
    assert skill.source_hash == registry._compute_source_hash(bundle_root)
    assert skill.executable_assets["bin"] == ["demo-skill-cli.py"]


@pytest.mark.asyncio
async def test_source_hash_changes_when_executable_changes(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = _write_skill(skill_root / "demo-skill")
    script = skill_dir / "scripts" / "run.py"
    script.parent.mkdir()
    script.write_text("print('one')\n", encoding="utf-8")

    registry = LocalSkillsRegistry(skill_root=str(skill_root))
    await registry.initialize()
    first_hash = registry.get_skill("demo-skill").source_hash

    script.write_text("print('two')\n", encoding="utf-8")
    await registry.refresh()

    assert registry.get_skill("demo-skill").source_hash != first_hash


def test_source_hash_records_are_unambiguous_and_shared_by_installer(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "SKILL.md").write_text("same", encoding="utf-8")
    (second / "SKILL.md").write_text("same", encoding="utf-8")
    # The old path+content concatenation encoded both trailing records as b"abc".
    (first / "ab").write_bytes(b"c")
    (second / "a").write_bytes(b"bc")

    first_hash = LocalSkillsRegistry._compute_source_hash(first)
    second_hash = LocalSkillsRegistry._compute_source_hash(second)

    assert first_hash != second_hash
    assert install_hash(first) == first_hash


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "mkfifo"),
    reason="POSIX FIFO regression",
)
def test_source_hash_rejects_fifo_without_blocking(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text("same", encoding="utf-8")
    os.mkfifo(bundle / "blocking-input")

    with pytest.raises(ValueError, match="regular file"):
        LocalSkillsRegistry._compute_source_hash(bundle)


@pytest.mark.asyncio
async def test_declared_secret_names_reach_the_catalogued_skill(tmp_path):
    """The credential names a skill declares are what the UI suggests binding."""
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "calendar"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: calendar\ndescription: Demo.\nsecrets: CALENDAR_TOKEN, bad-name\n---\n\nBody\n",
        encoding="utf-8",
    )

    registry = LocalSkillsRegistry(skill_root=str(skill_root))
    await registry.initialize()

    skill = registry.get_skill("calendar")
    assert skill.declared_secrets == ["CALENDAR_TOKEN"]
    # Names stay device-local: the server sync payload has no place for them.
    assert "CALENDAR_TOKEN" not in json.dumps(skill.to_sync_dict())


@pytest.mark.asyncio
async def test_plain_markdown_skill_declares_no_secrets(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "plain"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("Just instructions.\n", encoding="utf-8")

    registry = LocalSkillsRegistry(skill_root=str(skill_root))
    await registry.initialize()

    assert registry.get_skill("plain").declared_secrets == []


@pytest.mark.asyncio
async def test_scanner_skips_installer_staging_and_backup_copies(tmp_path):
    """The installer stages inside the root it owns, which is now this root too.

    A staging copy is half-written and a backup copy is superseded; publishing
    either would surface a skill that is mid-install or already replaced.
    """
    skill_root = tmp_path / "skills"
    _write_skill(skill_root / "demo-skill", name="demo-skill")
    _write_skill(skill_root / "demo-skill-abc.stage-1234", name="staged-skill")
    _write_skill(skill_root / "demo-skill-abc.backup-1234", name="backed-up-skill")

    registry = LocalSkillsRegistry(skill_root=str(skill_root))
    await registry.initialize()

    assert [skill.name for skill in registry.get_all_skills()] == ["demo-skill"]


def test_publishes_single_skill_distinguishes_a_bundle_from_a_container(tmp_path):
    """Replacement swaps a whole directory, so it must publish only one skill."""
    lone = _write_skill(tmp_path / "lone" / "demo", name="demo-skill").parent
    shared = tmp_path / "shared"
    _write_skill(shared / "first", name="first-skill")
    _write_skill(shared / "second", name="second-skill")

    assert local_skills_registry_module.publishes_single_skill(lone) is True
    assert local_skills_registry_module.publishes_single_skill(shared) is False


@pytest.mark.asyncio
async def test_scanner_omits_front_matter_with_nonstandard_skill_name(tmp_path):
    skill_root = tmp_path / "skills"
    _write_skill(skill_root / "invalid", name="invalid::skill")
    registry = LocalSkillsRegistry(skill_root=str(skill_root))

    await registry.initialize()

    assert registry.get_all_skills() == []


@pytest.mark.asyncio
async def test_scanner_rejects_symlinked_skill_document_outside_root(tmp_path):
    skill_root = tmp_path / "skills"
    bundle = skill_root / "demo-skill"
    bundle.mkdir(parents=True)
    outside = _write_skill(tmp_path / "outside") / "SKILL.md"
    try:
        os.symlink(outside, bundle / "SKILL.md")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    registry = LocalSkillsRegistry(skill_root=str(skill_root))

    await registry.initialize()

    assert registry.get_all_skills() == []


@pytest.mark.asyncio
async def test_loader_rejects_resolved_skill_document_outside_scan_root(tmp_path):
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    outside_skill = _write_skill(tmp_path / "outside") / "SKILL.md"
    registry = LocalSkillsRegistry(skill_root=str(skill_root))

    loaded = await registry._load_skill(outside_skill, skill_root.resolve())

    assert loaded is None


@pytest.mark.asyncio
async def test_scanner_rejects_bundle_with_symlinked_executable(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = _write_skill(skill_root / "demo-skill")
    (skill_dir / "bin").mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    try:
        os.symlink(outside, skill_dir / "bin" / "demo-cli.py")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    registry = LocalSkillsRegistry(skill_root=str(skill_root))

    await registry.initialize()

    assert registry.get_all_skills() == []


@pytest.mark.asyncio
async def test_scanner_rejects_link_like_asset_without_platform_symlink_support(
    tmp_path, monkeypatch
):
    skill_root = tmp_path / "skills"
    skill_dir = _write_skill(skill_root / "demo-skill")
    (skill_dir / "bin").mkdir()
    (skill_dir / "bin" / "demo-cli.py").write_text("print('x')\n", encoding="utf-8")
    real_is_link_like = local_skills_registry_module.is_link_like
    monkeypatch.setattr(
        local_skills_registry_module,
        "is_link_like",
        lambda path: path.name == "demo-cli.py" or real_is_link_like(path),
    )
    registry = LocalSkillsRegistry(skill_root=str(skill_root))

    await registry.initialize()

    assert registry.get_all_skills() == []


@pytest.mark.asyncio
async def test_unsupported_bin_assets_are_not_published_as_commands(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = _write_skill(skill_root / "demo-skill")
    bin_dir = skill_dir / "bin"
    bin_dir.mkdir()
    (bin_dir / "notes.txt").write_text("not a command", encoding="utf-8")
    (bin_dir / "blocked.cmd").write_text("echo blocked", encoding="utf-8")
    registry = LocalSkillsRegistry(skill_root=str(skill_root))

    await registry.initialize()

    assert registry.get_skill("demo-skill").executable_assets["bin"] == []


@pytest.mark.asyncio
async def test_skill_catalog_sync_omits_absolute_paths_and_roots(tmp_path):
    skill_root = tmp_path / "skills"
    _write_skill(skill_root / "demo-skill")

    registry = LocalSkillsRegistry(skill_root=str(skill_root))
    await registry.initialize()

    catalog = registry.get_skill_catalog(include_content=True)
    serialized_catalog = json.dumps(catalog)

    assert "skill_root" not in catalog
    assert "skill_root_count" not in catalog
    assert catalog["skills"][0]["name"] == "demo-skill"
    assert "content" in catalog["skills"][0]
    assert "path" not in catalog["skills"][0]
    assert "bundle_root" not in catalog["skills"][0]
    assert str(skill_root.resolve()) not in serialized_catalog
    assert catalog["skills"][0]["execution"]["source_hash"]


@pytest.mark.asyncio
async def test_refresh_only_reports_new_skills_not_removed_ones(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = _write_skill(skill_root / "demo-skill")

    registry = LocalSkillsRegistry(skill_root=str(skill_root))
    await registry.initialize()
    shutil.rmtree(skill_dir)

    discovered = await registry.refresh()

    assert discovered == 0
    assert registry.get_all_skills() == []


@pytest.mark.asyncio
async def test_skill_enabled_state_is_isolated_per_user_profile(tmp_path, monkeypatch):
    skill_root = tmp_path / "skills"
    _write_skill(skill_root / "demo-skill")
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profiles")
    auth_state = SimpleNamespace(current_user_id="user-a")
    monkeypatch.setattr(
        local_skills_registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: auth_state.current_user_id),
    )

    try:
        registry = LocalSkillsRegistry(skill_root=str(skill_root))
        await registry.initialize()
        registry.set_skill_enabled("demo-skill", False)

        auth_state.current_user_id = "user-b"
        await registry.initialize()
        assert registry.get_skill("demo-skill").enabled is True

        auth_state.current_user_id = "user-a"
        await registry.initialize()
        assert registry.get_skill("demo-skill").enabled is False
        state = json.loads(
            (get_profile_subdir("user-a", "skills") / "state.json").read_text(encoding="utf-8")
        )
        assert state == {"enabled": {"demo-skill": False}}
    finally:
        client_settings.profile_root = original_profile_root


def test_install_metadata_never_leaks_absolute_paths_in_sync_dict(tmp_path):
    absolute_path = str((tmp_path / "profiles" / "user-a" / "skills").resolve())
    skill = SkillMetadata(
        name="demo-skill",
        path=tmp_path / "SKILL.md",
        bundle_root=tmp_path,
        source_hash="abc123",
        executable_assets={"bin": [], "scripts": [], "python_project": False},
        description="demo",
        content="content",
        install_metadata={
            "installed": True,
            "source_hash": "abc123",
            "bundle_name": "demo-skill",
            "source": absolute_path,
            "install_root": absolute_path,
        },
    )

    sync_dict = skill.to_sync_dict()
    serialized = json.dumps(sync_dict)

    assert absolute_path not in serialized
    assert "install_root" not in sync_dict["install"]
    assert "source" not in sync_dict["install"]
    assert sync_dict["install"]["installed"] is True


@pytest.mark.asyncio
async def test_missing_configured_root_warns(tmp_path, monkeypatch, caplog):
    """A root the operator named and that does not exist is a misconfiguration."""
    missing_configured_root = tmp_path / "configured-but-absent"
    monkeypatch.setattr(
        local_skills_registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: "user-a"),
    )

    registry = LocalSkillsRegistry(skill_root=str(missing_configured_root))
    with caplog.at_level(logging.WARNING):
        await registry.initialize()

    assert any(
        "Skill root does not exist" in record.getMessage()
        and str(missing_configured_root) in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_absent_profile_root_is_silent(tmp_path, monkeypatch, caplog):
    """The profile-owned root is absent until the first install; that is normal."""
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profiles")
    monkeypatch.setattr(
        local_skills_registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: "user-a"),
    )

    try:
        registry = LocalSkillsRegistry()
        with caplog.at_level(logging.WARNING):
            await registry.initialize()
        resolved_root = str(resolve_skills_root("user-a"))
    finally:
        client_settings.profile_root = original_profile_root

    assert not Path(resolved_root).exists()
    assert registry.skill_root == resolved_root
    assert not [
        record.getMessage()
        for record in caplog.records
        if "Skill root does not exist" in record.getMessage()
    ]


def test_read_only_profile_access_does_not_create_skill_directories(tmp_path, monkeypatch):
    """Resolving a profile path must not be what creates it on disk."""
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profiles")
    monkeypatch.setattr(
        local_skills_registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: "user-a"),
    )

    try:
        registry = LocalSkillsRegistry()
        assert registry._load_persisted_skill_state() == {}
        assert registry._resolve_skill_root()

        assert not profile_subdir_path("user-a", "skills").exists()
    finally:
        client_settings.profile_root = original_profile_root


def test_install_metadata_defaults_to_not_installed_when_absent(tmp_path):
    skill = SkillMetadata(
        name="demo-skill",
        path=tmp_path / "SKILL.md",
        bundle_root=tmp_path,
        source_hash="abc123",
        executable_assets={"bin": [], "scripts": [], "python_project": False},
        description="demo",
        content="content",
    )

    assert skill.to_dict()["install"] == {"installed": False}
    assert skill.to_sync_dict()["install"] == {"installed": False}
