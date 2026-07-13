from pathlib import Path

from client_backend.services.local_skills_registry import SkillMetadata
from client_backend.services.skill_runtime.manager import (
    RUN_SKILL_COMMAND,
    SkillReadiness,
    SkillRuntimeManager,
)


def _skill(tmp_path: Path) -> SkillMetadata:
    root = tmp_path / "example-calendar"
    root.mkdir()
    skill_path = root / "SKILL.md"
    skill_path.write_text("# Calendar\n", encoding="utf-8")
    return SkillMetadata(
        name="example-calendar",
        path=skill_path,
        bundle_root=root,
        source_hash="b" * 64,
        executable_assets={"bin": ["calendar-cli.py"], "scripts": [], "python_project": False},
        description="Calendar commands",
        content="# Calendar\n",
    )


def test_ready_skill_publishes_exactly_one_fixed_command_tool(tmp_path):
    skill = _skill(tmp_path)
    readiness = SkillReadiness(
        status="ready",
        setup_status="not_applicable",
        commands=["calendar-cli"],
    )

    entries = SkillRuntimeManager().capability_catalog_entries(skill, readiness)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == RUN_SKILL_COMMAND
    assert entry["qualified_id"] == "skill::example-calendar::run_skill_command"
    assert entry["origin"] == "skill"
    assert entry["server_name"] == "skill_example_calendar"
    assert entry["mutation"] is True
    assert entry["source_hash"] == "b" * 64
    assert entry["input_schema"] == {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "cwd": {
                "type": "string",
                "enum": ["skill", "workspace"],
                "default": "workspace",
            },
        },
        "required": ["argv"],
        "additionalProperties": False,
    }


def test_non_ready_skill_publishes_no_tool(tmp_path):
    skill = _skill(tmp_path)
    readiness = SkillReadiness(status="not_ready", setup_status="setup_required")

    assert SkillRuntimeManager().capability_catalog_entries(skill, readiness) == []
