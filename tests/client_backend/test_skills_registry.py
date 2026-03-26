import json

import pytest

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
