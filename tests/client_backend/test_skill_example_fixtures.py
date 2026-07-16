import os
from pathlib import Path

import pytest

from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_runtime.execution import SkillExecutionEngine
from client_backend.services.skill_runtime.manager import SkillRuntimeManager


class _Secrets:
    def get_for_skill(self, name):
        return {}


class _Audit:
    def write(self, **record):
        return None


@pytest.mark.asyncio
async def test_standard_skill_bundle_executes_named_cli_without_global_install(monkeypatch):
    fixture_root = Path(__file__).parents[1] / "fixtures" / "skills"
    registry = LocalSkillsRegistry(skill_roots=[str(fixture_root)])
    await registry.initialize()
    skill = registry.get_skill("echo-python")
    assert skill is not None

    readiness = SkillRuntimeManager().evaluate_readiness(skill)
    entries = SkillRuntimeManager().capability_catalog_entries(skill, readiness)
    monkeypatch.setenv("PATH", "")
    envelope = await SkillExecutionEngine(
        registry=registry,
        secret_store=_Secrets(),
        audit_writer=_Audit(),
    ).execute(
        "skill::echo-python::run_skill_command",
        {"argv": ["echo-python", "hello fixtures"]},
        {"mutation_approved": True},
    )

    assert readiness.status == "ready"
    assert [entry["qualified_id"] for entry in entries] == ["skill::echo-python::run_skill_command"]
    assert envelope["ok"] is True
    assert envelope["result"] == {"message": "hello fixtures"}
    assert "echo-python" not in os.environ.get("PATH", "")


@pytest.mark.asyncio
async def test_calendar_skill_dry_run_uses_its_bundled_cli_without_global_install(monkeypatch):
    fixture_root = Path(__file__).parents[1] / "fixtures" / "skills"
    registry = LocalSkillsRegistry(skill_roots=[str(fixture_root)])
    await registry.initialize()
    skill = registry.get_skill("cli-anything-google-calendar")
    assert skill is not None
    monkeypatch.setenv("PATH", "")

    envelope = await SkillExecutionEngine(
        registry=registry,
        secret_store=_Secrets(),
        audit_writer=_Audit(),
    ).execute(
        "skill::cli-anything-google-calendar::run_skill_command",
        {
            "argv": [
                "cli-anything-google-calendar",
                "--json",
                "schedule",
                "create",
                "--calendar-id",
                "primary",
                "--title",
                "Planning",
                "--start",
                "2026-07-13T16:00:00+07:00",
                "--end",
                "2026-07-13T16:30:00+07:00",
                "--dry-run",
            ]
        },
        {"mutation_approved": True},
    )

    assert envelope["ok"] is True
    assert envelope["result"]["dry_run"] is True
    assert envelope["result"]["body"]["summary"] == "Planning"
