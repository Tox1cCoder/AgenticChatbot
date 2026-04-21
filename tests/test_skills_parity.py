"""
Parity tests for skills parsing between server (SkillsRegistry) and client
(LocalSkillsRegistry).

Both parsers must produce identical name, description, and activation body
for the same SKILL.md files.  A divergence here means client-synced summaries
will not match server-resolved content, breaking skill activation.
"""

from pathlib import Path

import pytest

from app.ai.skills_registry import SkillsRegistry
from client_backend.services.local_skills_registry import LocalSkillsRegistry

# ---------------------------------------------------------------------------
# Shared Anthropic-style front-matter fixtures
# ---------------------------------------------------------------------------

FRONTMATTER_SIMPLE = """\
---
name: my-skill
description: Does something useful.
---

## Instructions

Follow these steps carefully.
"""

FRONTMATTER_FOLDED_DESCRIPTION = """\
---
name: folded-skill
description: >
  This is a longer description
  that spans multiple lines.
---

# Body

Body content here.
"""

FRONTMATTER_WITH_EXTRAS = """\
---
name: tagged-skill
description: A skill with tags and category.
category: productivity
tags: writing, editing
---

Skill body without front matter.
"""

PLAIN_MARKDOWN = """\
# Plain Skill

No front matter — just a plain markdown skill file.
"""

FRONTMATTER_MISSING_NAME = """\
---
description: Missing name should be rejected.
---

Body content.
"""

FRONTMATTER_NAME_ONLY = """\
---
name: name-only
---

Body content.
"""

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SKILLS_DIR = REPO_ROOT / "skills"


# ---------------------------------------------------------------------------
# Client parser unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_parser_extracts_name_from_front_matter(tmp_path):
    """Name must come from YAML front matter, not directory name."""
    skill_dir = tmp_path / "wrong-dirname"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_SIMPLE, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    skill = registry.get_skill("my-skill")
    assert skill is not None, "Skill not found under front-matter name 'my-skill'"
    assert skill.name == "my-skill"


@pytest.mark.asyncio
async def test_client_parser_extracts_description_from_front_matter(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_SIMPLE, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    skill = registry.get_skill("my-skill")
    assert skill.description == "Does something useful."


@pytest.mark.asyncio
async def test_client_parser_strips_front_matter_from_content(tmp_path):
    """Activation payload (content) must not contain raw YAML delimiters."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_SIMPLE, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    skill = registry.get_skill("my-skill")
    assert "---" not in skill.content
    assert "name: my-skill" not in skill.content
    assert "Follow these steps carefully." in skill.content


@pytest.mark.asyncio
async def test_client_parser_folded_description(tmp_path):
    skill_dir = tmp_path / "folded-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_FOLDED_DESCRIPTION, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    skill = registry.get_skill("folded-skill")
    assert skill is not None
    # Folded scalar collapses newlines into spaces
    assert "longer description" in skill.description
    assert "spans multiple lines" in skill.description


@pytest.mark.asyncio
async def test_client_parser_extracts_category_and_tags(tmp_path):
    skill_dir = tmp_path / "tagged-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_WITH_EXTRAS, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    skill = registry.get_skill("tagged-skill")
    assert skill is not None
    assert skill.category == "productivity"
    assert "writing" in skill.tags
    assert "editing" in skill.tags


@pytest.mark.asyncio
async def test_client_parser_falls_back_to_dirname_without_front_matter(tmp_path):
    """Plain markdown skills still load via directory-name fallback."""
    skill_dir = tmp_path / "plain-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(PLAIN_MARKDOWN, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    skill = registry.get_skill("plain-skill")
    assert skill is not None
    assert skill.name == "plain-skill"
    # Full raw content preserved (no front matter to strip)
    assert "No front matter" in skill.content


@pytest.mark.asyncio
async def test_client_parser_rejects_front_matter_without_name(tmp_path):
    """Invalid front matter should not silently fall back to the directory name."""
    skill_dir = tmp_path / "missing-name"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_MISSING_NAME, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    assert registry.get_skill("missing-name") is None
    assert registry.get_all_skills() == []


@pytest.mark.asyncio
async def test_client_parser_keeps_empty_description_when_front_matter_omits_it(tmp_path):
    """Front-matter skills should match the server contract for empty descriptions."""
    skill_dir = tmp_path / "name-only"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_NAME_ONLY, encoding="utf-8")

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()

    skill = registry.get_skill("name-only")
    assert skill is not None
    assert skill.description == ""


# ---------------------------------------------------------------------------
# Server parser unit tests (static methods)
# ---------------------------------------------------------------------------


def test_server_split_front_matter_extracts_yaml_and_body():
    result = SkillsRegistry._split_front_matter(FRONTMATTER_SIMPLE)
    assert result is not None
    yaml_block, body = result
    assert "name: my-skill" in yaml_block
    assert "Follow these steps carefully." in body
    assert "---" not in body


def test_server_split_front_matter_returns_none_without_front_matter():
    result = SkillsRegistry._split_front_matter(PLAIN_MARKDOWN)
    assert result is None


def test_server_extract_yaml_value_simple():
    yaml_block = "name: my-skill\ndescription: Does something."
    assert SkillsRegistry._extract_yaml_value(yaml_block, "name") == "my-skill"
    assert SkillsRegistry._extract_yaml_value(yaml_block, "description") == "Does something."


def test_server_extract_yaml_value_folded():
    result = SkillsRegistry._split_front_matter(FRONTMATTER_FOLDED_DESCRIPTION)
    assert result is not None
    yaml_block, _ = result
    desc = SkillsRegistry._extract_yaml_value(yaml_block, "description")
    assert desc is not None
    assert "longer description" in desc
    assert "spans multiple lines" in desc


def test_server_registry_rejects_front_matter_without_name(tmp_path):
    skill_dir = tmp_path / "missing-name"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_MISSING_NAME, encoding="utf-8")

    registry = SkillsRegistry(
        skills_dir=str(tmp_path),
        config_path=str(tmp_path / "skills_config.json"),
    )

    assert registry.get_all_skills() == []


# ---------------------------------------------------------------------------
# Cross-parser parity: server and client must agree on the same SKILL.md
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parity_name_matches_for_front_matter_skill(tmp_path):
    """Server and client must resolve the same name for the same SKILL.md."""
    skill_dir = tmp_path / "wrong-dirname"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(FRONTMATTER_SIMPLE, encoding="utf-8")

    # Server path
    server_parsed = SkillsRegistry._split_front_matter(FRONTMATTER_SIMPLE)
    assert server_parsed is not None
    server_yaml, server_body = server_parsed
    server_name = SkillsRegistry._extract_yaml_value(server_yaml, "name")

    # Client path
    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()
    client_skill = registry.get_skill("my-skill")
    assert client_skill is not None, "Client did not find skill under front-matter name"

    assert client_skill.name == server_name


@pytest.mark.asyncio
async def test_parity_description_matches_for_front_matter_skill(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_SIMPLE, encoding="utf-8")

    server_parsed = SkillsRegistry._split_front_matter(FRONTMATTER_SIMPLE)
    assert server_parsed is not None
    server_yaml, _ = server_parsed
    server_description = SkillsRegistry._extract_yaml_value(server_yaml, "description") or ""

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()
    client_skill = registry.get_skill("my-skill")
    assert client_skill is not None

    assert client_skill.description == server_description


@pytest.mark.asyncio
async def test_parity_activation_body_matches_for_front_matter_skill(tmp_path):
    """Both parsers must produce the same body content (front matter stripped)."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_SIMPLE, encoding="utf-8")

    server_parsed = SkillsRegistry._split_front_matter(FRONTMATTER_SIMPLE)
    assert server_parsed is not None
    _, server_body = server_parsed

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()
    client_skill = registry.get_skill("my-skill")
    assert client_skill is not None

    assert client_skill.content == server_body


@pytest.mark.asyncio
async def test_parity_folded_description_matches(tmp_path):
    skill_dir = tmp_path / "folded-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_FOLDED_DESCRIPTION, encoding="utf-8")

    server_parsed = SkillsRegistry._split_front_matter(FRONTMATTER_FOLDED_DESCRIPTION)
    assert server_parsed is not None
    server_yaml, server_body = server_parsed
    server_desc = SkillsRegistry._extract_yaml_value(server_yaml, "description") or ""

    registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await registry.initialize()
    client_skill = registry.get_skill("folded-skill")
    assert client_skill is not None

    assert client_skill.description == server_desc
    assert client_skill.content == server_body


@pytest.mark.asyncio
async def test_parity_missing_description_matches_server_contract(tmp_path):
    skill_dir = tmp_path / "name-only"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(FRONTMATTER_NAME_ONLY, encoding="utf-8")

    server_registry = SkillsRegistry(
        skills_dir=str(tmp_path),
        config_path=str(tmp_path / "skills_config.json"),
    )
    server_skill = server_registry.get_skill("name-only")

    client_registry = LocalSkillsRegistry(skill_roots=[str(tmp_path)])
    await client_registry.initialize()
    client_skill = client_registry.get_skill("name-only")

    assert client_skill is not None
    assert client_skill.description == server_skill.description
    assert client_skill.content == server_skill.content


def test_server_registry_loads_real_repo_skills_folder(tmp_path):
    """The server registry must load the production repo skills directory."""
    registry = SkillsRegistry(
        skills_dir=str(REPO_SKILLS_DIR),
        config_path=str(tmp_path / "skills_config.json"),
    )

    skills_by_name = {skill.name: skill for skill in registry.get_all_skills()}

    assert {"playwright-cli", "take100-timesheet"} <= set(skills_by_name)
    assert skills_by_name["playwright-cli"].description.startswith("Automate browser interactions")
    assert "Browser Automation with playwright-cli" in skills_by_name["playwright-cli"].content
    assert skills_by_name["take100-timesheet"].description.startswith(
        "Automates entering daily working time schedules"
    )
    assert "Take100 Timesheet Automation" in skills_by_name["take100-timesheet"].content


@pytest.mark.asyncio
async def test_real_repo_skill_folder_has_server_client_parity(tmp_path):
    """The checked-in repo skills must parse the same on server and client."""
    server_registry = SkillsRegistry(
        skills_dir=str(REPO_SKILLS_DIR),
        config_path=str(tmp_path / "skills_config.json"),
    )
    server_skills = {
        skill.name: (skill.description, skill.content) for skill in server_registry.get_all_skills()
    }

    client_registry = LocalSkillsRegistry(skill_roots=[str(REPO_SKILLS_DIR)])
    await client_registry.initialize()
    client_skills = {
        skill.name: (skill.description, skill.content) for skill in client_registry.get_all_skills()
    }

    assert client_skills == server_skills
