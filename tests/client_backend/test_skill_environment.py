import json
import subprocess
from pathlib import Path

import pytest

from client_backend.services.local_skills_registry import SkillMetadata
from client_backend.services.skill_runtime.environment import SkillEnvironmentManager
from shared.skills.errors import SKILL_SETUP_FAILED, SKILL_SETUP_REQUIRED, SkillRuntimeError


def _python_skill(tmp_path: Path, *, source_hash: str = "a" * 64) -> SkillMetadata:
    root = tmp_path / "bundle"
    root.mkdir(exist_ok=True)
    skill_path = root / "SKILL.md"
    skill_path.write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "demo-skill"
version = "1.0.0"
dependencies = ["httpx>=0.25"]

[project.scripts]
demo-cli = "demo:main"
""".strip(),
        encoding="utf-8",
    )
    return SkillMetadata(
        name="demo-skill",
        path=skill_path,
        bundle_root=root,
        source_hash=source_hash,
        executable_assets={"bin": [], "scripts": [], "python_project": True},
        description="demo",
        content="demo",
    )


def _fake_venv_builder(venv_root: Path) -> None:
    scripts = venv_root / ("Scripts" if __import__("os").name == "nt" else "bin")
    scripts.mkdir(parents=True)
    python_name = "python.exe" if __import__("os").name == "nt" else "python"
    command_name = "demo-cli.exe" if __import__("os").name == "nt" else "demo-cli"
    (scripts / python_name).write_text("python", encoding="utf-8")
    (scripts / command_name).write_text("command", encoding="utf-8")
    dependency_command = "dependency-cli.exe" if __import__("os").name == "nt" else "dependency-cli"
    (scripts / dependency_command).write_text("dependency", encoding="utf-8")


def test_preview_discloses_python_setup_inputs_and_requires_confirmation(tmp_path):
    skill = _python_skill(tmp_path)
    manager = SkillEnvironmentManager(runtime_base=tmp_path / "runtimes")

    preview = manager.preview(skill)

    assert preview["python_project"] is True
    assert preview["dependencies"] == ["httpx>=0.25"]
    assert preview["build_requirements"] == ["setuptools>=68"]
    assert preview["declared_commands"] == ["demo-cli"]
    assert preview["confirmation_required"] is True
    assert str(tmp_path) not in json.dumps(preview)


def test_prepare_rejects_unapproved_python_setup(tmp_path):
    skill = _python_skill(tmp_path)
    manager = SkillEnvironmentManager(runtime_base=tmp_path / "runtimes")

    with pytest.raises(SkillRuntimeError) as exc_info:
        manager.prepare(skill, approve_setup=False)

    assert exc_info.value.code == SKILL_SETUP_REQUIRED
    assert exc_info.value.repair["source_hash"] == skill.source_hash


def test_prepare_stages_runtime_and_records_discovered_commands(tmp_path):
    skill = _python_skill(tmp_path)
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"installed", stderr=b"")

    manager = SkillEnvironmentManager(
        runtime_base=tmp_path / "runtimes",
        venv_builder=_fake_venv_builder,
        runner=runner,
    )

    result = manager.prepare(skill, approve_setup=True)

    assert result["status"] == "ready"
    assert result["commands"] == ["demo-cli"]
    assert calls and calls[0][1:4] == ["-m", "pip", "install"]
    inspection = manager.inspect(skill)
    assert inspection["status"] == "ready"
    assert inspection["commands"] == ["demo-cli"]
    runtime_json = Path(inspection["runtime_root"]) / "runtime.json"
    metadata = json.loads(runtime_json.read_text(encoding="utf-8"))
    assert metadata["source_hash"] == skill.source_hash
    assert metadata["commands"] == ["demo-cli"]


def test_prepare_installs_dependency_lock_before_local_project(tmp_path):
    skill = _python_skill(tmp_path)
    lock_path = skill.bundle_root / "requirements.lock"
    lock_path.write_text("httpx==0.28.1\n", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"installed", stderr=b"")

    manager = SkillEnvironmentManager(
        runtime_base=tmp_path / "runtimes",
        venv_builder=_fake_venv_builder,
        runner=runner,
    )

    preview = manager.preview(skill)
    manager.prepare(skill, approve_setup=True)

    assert preview["dependency_lock"] == "requirements.lock"
    assert calls[0][-2:] == ["-r", str(lock_path)]
    assert calls[1][-3:-1] == ["--no-build-isolation", "--no-deps"]
    assert calls[1][-1] == str(skill.bundle_root)


def test_inspect_marks_runtime_stale_when_source_hash_changes(tmp_path):
    skill = _python_skill(tmp_path)
    manager = SkillEnvironmentManager(
        runtime_base=tmp_path / "runtimes",
        venv_builder=_fake_venv_builder,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, b"", b""),
    )
    manager.prepare(skill, approve_setup=True)
    changed = SkillMetadata(
        **{**skill.__dict__, "source_hash": "b" * 64}
    )

    assert manager.inspect(changed)["status"] == "stale"
    assert manager.inspect_existing_for_skill(changed)["status"] == "stale"


def test_failed_setup_preserves_previous_ready_runtime_and_removes_stage(tmp_path):
    skill = _python_skill(tmp_path)
    runtime_base = tmp_path / "runtimes"
    good = SkillEnvironmentManager(
        runtime_base=runtime_base,
        venv_builder=_fake_venv_builder,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, b"", b""),
    )
    good.prepare(skill, approve_setup=True)

    failing = SkillEnvironmentManager(
        runtime_base=runtime_base,
        venv_builder=_fake_venv_builder,
        runner=lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, b"", b"build failed"
        ),
    )
    with pytest.raises(SkillRuntimeError) as exc_info:
        failing.prepare(skill, approve_setup=True, force=True)

    assert exc_info.value.code == SKILL_SETUP_FAILED
    assert good.inspect(skill)["status"] == "ready"
    assert not list(runtime_base.rglob("*.stage-*"))


def test_remove_skill_runtimes_is_confined_to_selected_skill(tmp_path):
    runtime_base = tmp_path / "runtimes"
    selected = runtime_base / "demo-skill"
    other = runtime_base / "other-skill"
    selected.mkdir(parents=True)
    other.mkdir(parents=True)
    (selected / "marker").write_text("selected", encoding="utf-8")
    (other / "marker").write_text("other", encoding="utf-8")
    manager = SkillEnvironmentManager(runtime_base=runtime_base)

    manager.remove_skill("demo-skill")

    assert not selected.exists()
    assert other.exists()
