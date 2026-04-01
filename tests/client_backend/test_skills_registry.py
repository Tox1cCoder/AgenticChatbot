import json
import shutil
from types import SimpleNamespace

import pytest

from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.services import local_skills_registry as local_skills_registry_module
from client_backend.services.local_skills_registry import LocalSkillsRegistry


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
