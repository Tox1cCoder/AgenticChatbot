from pathlib import Path

from client_backend.services.local_skills_registry import SkillMetadata
from client_backend.services.skill_runtime.manager import SkillRuntimeManager


def _skill(tmp_path: Path, *, assets: dict) -> SkillMetadata:
    root = tmp_path / "demo-skill"
    root.mkdir()
    skill_path = root / "SKILL.md"
    skill_path.write_text("# Demo\n", encoding="utf-8")
    return SkillMetadata(
        name="demo-skill",
        path=skill_path,
        bundle_root=root,
        source_hash="a" * 64,
        executable_assets=assets,
        description="Demo skill",
        content="# Demo\n",
    )


class _EnvironmentManager:
    def __init__(self, inspection: dict):
        self.inspection = inspection

    def inspect(self, skill):
        return self.inspection


def test_skill_without_executable_assets_is_instruction_only(tmp_path):
    skill = _skill(
        tmp_path,
        assets={"bin": [], "scripts": [], "python_project": False},
    )

    readiness = SkillRuntimeManager(
        environment_manager=_EnvironmentManager({"status": "not_applicable"})
    ).evaluate_readiness(skill)

    assert readiness.status == "instruction_only"
    assert readiness.setup_status == "not_applicable"
    assert readiness.commands == []


def test_bundled_bin_and_python_script_are_ready_without_global_path(tmp_path):
    skill = _skill(
        tmp_path,
        assets={
            "bin": ["demo-cli.py"],
            "scripts": ["scripts/inspect.py"],
            "python_project": False,
        },
    )

    readiness = SkillRuntimeManager(
        environment_manager=_EnvironmentManager({"status": "not_applicable"})
    ).evaluate_readiness(skill)

    assert readiness.status == "ready"
    assert readiness.setup_status == "not_applicable"
    assert readiness.commands == ["demo-cli", "scripts/inspect.py"]


def test_python_project_without_runtime_requires_setup(tmp_path):
    skill = _skill(
        tmp_path,
        assets={"bin": [], "scripts": [], "python_project": True},
    )

    readiness = SkillRuntimeManager(
        environment_manager=_EnvironmentManager({"status": "setup_required", "commands": []})
    ).evaluate_readiness(skill)

    assert readiness.status == "not_ready"
    assert readiness.setup_status == "setup_required"
    assert readiness.repair_hints == [{"type": "setup_skill", "skill": "demo-skill"}]


def test_prepared_python_project_uses_runtime_commands(tmp_path):
    skill = _skill(
        tmp_path,
        assets={"bin": [], "scripts": [], "python_project": True},
    )

    readiness = SkillRuntimeManager(
        environment_manager=_EnvironmentManager(
            {"status": "ready", "commands": ["demo-cli"], "runtime_id": "runtime-1"}
        )
    ).evaluate_readiness(skill)

    assert readiness.status == "ready"
    assert readiness.setup_status == "ready"
    assert readiness.commands == ["demo-cli"]
    assert readiness.runtime_id == "runtime-1"


def test_stale_runtime_is_not_ready(tmp_path):
    skill = _skill(
        tmp_path,
        assets={"bin": [], "scripts": [], "python_project": True},
    )

    readiness = SkillRuntimeManager(
        environment_manager=_EnvironmentManager({"status": "stale", "commands": []})
    ).evaluate_readiness(skill)

    assert readiness.status == "not_ready"
    assert readiness.setup_status == "stale"
    assert readiness.repair_hints[0]["type"] == "rebuild_skill_runtime"


def test_readiness_summary_never_exposes_runtime_paths(tmp_path):
    skill = _skill(
        tmp_path,
        assets={"bin": [], "scripts": [], "python_project": True},
    )
    readiness = SkillRuntimeManager(
        environment_manager=_EnvironmentManager(
            {
                "status": "ready",
                "commands": ["demo-cli"],
                "runtime_id": "runtime-1",
                "runtime_root": str(tmp_path / "private"),
            }
        )
    ).evaluate_readiness(skill)

    summary = readiness.to_summary()

    assert "runtime_root" not in summary
    assert str(tmp_path) not in str(summary)
