import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.services import local_skills_registry as local_skills_registry_module
from client_backend.services.local_skills_registry import LocalSkillsRegistry, SkillMetadata


def _valid_skill_manifest_dict() -> dict:
    """A minimal, provider-neutral skill.json payload for registry tests."""
    return {
        "schema_version": "1.0",
        "name": "demo-skill",
        "description": "Demo skill for registry tests.",
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


def _write_demo_skill(skill_root: Path, skill_json_content: str | None = None) -> Path:
    skill_dir = skill_root / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Demo Skill\n\nUse this skill for testing.\n",
        encoding="utf-8",
    )
    if skill_json_content is not None:
        (skill_dir / "skill.json").write_text(skill_json_content, encoding="utf-8")
    return skill_dir


@pytest.mark.asyncio
async def test_skill_catalog_sync_omits_absolute_paths_and_roots(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Demo Skill\n\nUse this skill for testing.\n",
        encoding="utf-8",
    )

    registry = LocalSkillsRegistry(skill_roots=[str(skill_root)])
    await registry.initialize()

    catalog = registry.get_skill_catalog(include_content=True)
    serialized_catalog = json.dumps(catalog)

    assert catalog["skill_root_count"] == 1
    assert "skill_roots" not in catalog
    assert catalog["skills"][0]["name"] == "demo-skill"
    assert "content" in catalog["skills"][0]
    assert "path" not in catalog["skills"][0]
    assert str(skill_root.resolve()) not in serialized_catalog


@pytest.mark.asyncio
async def test_refresh_only_reports_new_skills_not_removed_ones(tmp_path):
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Demo Skill\n\nUse this skill for testing.\n",
        encoding="utf-8",
    )

    registry = LocalSkillsRegistry(skill_roots=[str(skill_root)])
    await registry.initialize()

    shutil.rmtree(skill_dir)

    discovered = await registry.refresh()

    assert discovered == 0
    assert registry.get_all_skills() == []


@pytest.mark.asyncio
async def test_skill_enabled_state_is_isolated_per_user_profile(tmp_path, monkeypatch):
    skill_root = tmp_path / "skills"
    skill_dir = skill_root / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "# Demo Skill\n\nUse this skill for testing.\n",
        encoding="utf-8",
    )

    profile_root = tmp_path / "profiles"
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(profile_root)

    auth_state = SimpleNamespace(current_user_id="user-a")
    monkeypatch.setattr(
        local_skills_registry_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: auth_state.current_user_id),
    )

    try:
        registry = LocalSkillsRegistry(skill_roots=[str(skill_root)])
        await registry.initialize()

        assert registry.get_skill("demo-skill").enabled is True

        registry.set_skill_enabled("demo-skill", False)
        assert registry.get_skill("demo-skill").enabled is False

        auth_state.current_user_id = "user-b"
        await registry.initialize()
        assert registry.get_skill("demo-skill").enabled is True

        registry.set_skill_enabled("demo-skill", False)
        assert registry.get_skill("demo-skill").enabled is False

        auth_state.current_user_id = "user-a"
        await registry.initialize()
        assert registry.get_skill("demo-skill").enabled is False

        user_a_state = json.loads(
            (get_profile_subdir("user-a", "skills") / "state.json").read_text(encoding="utf-8")
        )
        user_b_state = json.loads(
            (get_profile_subdir("user-b", "skills") / "state.json").read_text(encoding="utf-8")
        )
        assert user_a_state == {"enabled": {"demo-skill": False}}
        assert user_b_state == {"enabled": {"demo-skill": False}}
    finally:
        client_settings.profile_root = original_profile_root


@pytest.mark.asyncio
async def test_markdown_only_skill_has_instruction_only_execution_summary(tmp_path):
    skill_root = tmp_path / "skills"
    _write_demo_skill(skill_root)

    registry = LocalSkillsRegistry(skill_roots=[str(skill_root)])
    await registry.initialize()

    skill = registry.get_skill("demo-skill")
    assert skill is not None
    assert skill.manifest is None
    assert skill.manifest_error is None
    assert skill.manifest_path is None

    execution = skill.to_dict()["execution"]
    assert execution == {
        "manifest_present": False,
        "status": "instruction_only",
        "capability_count": 0,
        "permissions": [],
    }

    catalog = registry.get_skill_catalog(include_content=True)
    assert catalog["skills"][0]["execution"]["status"] == "instruction_only"
    assert catalog["skills"][0]["execution"]["manifest_present"] is False


@pytest.mark.asyncio
async def test_manifest_backed_skill_has_manifest_present_execution_summary(tmp_path):
    skill_root = tmp_path / "skills"
    manifest_dict = _valid_skill_manifest_dict()
    _write_demo_skill(skill_root, skill_json_content=json.dumps(manifest_dict))

    registry = LocalSkillsRegistry(skill_roots=[str(skill_root)])
    await registry.initialize()

    skill = registry.get_skill("demo-skill")
    assert skill is not None
    assert skill.manifest is not None
    assert skill.manifest_error is None
    assert skill.manifest_path is not None

    execution = skill.to_dict()["execution"]
    assert execution["manifest_present"] is True
    assert execution["status"] == "manifest_present"
    assert execution["capability_count"] == len(manifest_dict["capabilities"])
    assert execution["permissions"] == manifest_dict["permissions"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "skill_json_content",
    [
        "{this is not valid json",
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "demo-skill",
                "description": "Has an unknown runtime type.",
                "runtime": {"type": "totally_unknown"},
                "capabilities": [],
            }
        ),
    ],
    ids=["malformed-json", "invalid-manifest-unknown-runtime-type"],
)
async def test_invalid_skill_json_still_loads_skill_as_instruction_only(
    tmp_path, skill_json_content
):
    skill_root = tmp_path / "skills"
    _write_demo_skill(skill_root, skill_json_content=skill_json_content)

    registry = LocalSkillsRegistry(skill_roots=[str(skill_root)])
    await registry.initialize()

    skill = registry.get_skill("demo-skill")
    assert skill is not None
    assert skill.manifest is None
    assert skill.manifest_error is not None
    assert "skill.json" in skill.manifest_error

    execution = skill.to_dict()["execution"]
    assert execution["manifest_present"] is False
    assert execution["status"] == "invalid"
    assert execution["capability_count"] == 0
    assert execution["permissions"] == []


@pytest.mark.asyncio
async def test_sync_dict_and_catalog_omit_manifest_path_and_absolute_paths(tmp_path):
    skill_root = tmp_path / "skills"
    _write_demo_skill(skill_root, skill_json_content=json.dumps(_valid_skill_manifest_dict()))

    registry = LocalSkillsRegistry(skill_roots=[str(skill_root)])
    await registry.initialize()

    catalog = registry.get_skill_catalog(include_content=True)
    serialized_catalog = json.dumps(catalog)

    assert str(skill_root.resolve()) not in serialized_catalog
    assert "manifest_path" not in catalog["skills"][0]
    assert catalog["skills"][0]["execution"]["status"] == "manifest_present"

    skill = registry.get_skill("demo-skill")
    sync_dict = skill.to_sync_dict()
    assert "manifest_path" not in sync_dict
    assert "path" not in sync_dict


def test_install_metadata_never_leaks_absolute_paths_in_sync_dict(tmp_path):
    absolute_path = str((tmp_path / "profiles" / "user-a" / "skills" / "demo-skill").resolve())

    skill = SkillMetadata(
        name="demo-skill",
        path=tmp_path / "SKILL.md",
        description="demo",
        content="content",
        install_metadata={
            "installed": True,
            "source_hash": "abc123",
            "bundle_name": "demo-skill",
            # Misuse: a category label field populated with an absolute path
            # instead of e.g. "profile" — must still be redacted defensively.
            "source": absolute_path,
            # Not in the safe-key allow-list at all.
            "install_root": absolute_path,
        },
    )

    sync_dict = skill.to_sync_dict()
    serialized = json.dumps(sync_dict)

    assert absolute_path not in serialized
    assert "install_root" not in sync_dict["install"]
    assert "source" not in sync_dict["install"]
    assert sync_dict["install"]["installed"] is True
    assert sync_dict["install"]["source_hash"] == "abc123"
    assert sync_dict["install"]["bundle_name"] == "demo-skill"


def test_install_metadata_redacts_absolute_path_embedded_in_larger_string(tmp_path):
    absolute_path = str((tmp_path / "profiles" / "user-a" / "skills" / "demo").resolve())

    skill = SkillMetadata(
        name="demo-skill",
        path=tmp_path / "SKILL.md",
        description="demo",
        content="content",
        install_metadata={
            "installed": True,
            # Path embedded as a substring in a larger human string, not a
            # bare path value — must still be caught and dropped.
            "bundle_name": f"copied from {absolute_path}",
        },
    )

    serialized = json.dumps(skill.to_sync_dict())

    assert absolute_path not in serialized
    assert "bundle_name" not in skill.to_sync_dict()["install"]


def test_install_metadata_defaults_to_not_installed_when_absent():
    skill = SkillMetadata(
        name="demo-skill",
        path=Path("SKILL.md"),
        description="demo",
        content="content",
    )

    assert skill.to_dict()["install"] == {"installed": False}
    assert skill.to_sync_dict()["install"] == {"installed": False}
