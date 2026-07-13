import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.services.local_skills_registry import LocalSkillsRegistry, SkillMetadata
from client_backend.services.skill_runtime import audit as audit_module
from client_backend.services.skill_runtime.audit import SkillAuditWriter, new_audit_id
from client_backend.services.skill_runtime.execution import SkillExecutionEngine
from shared.skills.errors import PERMISSION_REQUIRED


class _Registry:
    def __init__(self, skill):
        self.skill = skill

    def get_skill(self, name):
        return self.skill if name == self.skill.name else None


class _Secrets:
    def get_for_skill(self, name):
        return {"TOKEN": 'sëcret"\\value'}


def _skill(tmp_path: Path) -> SkillMetadata:
    root = tmp_path / "demo"
    root.mkdir()
    (root / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (root / "bin").mkdir()
    (root / "bin" / "demo-cli.py").write_text("print('ok')\n", encoding="utf-8")
    return SkillMetadata(
        name="demo",
        path=root / "SKILL.md",
        bundle_root=root,
        source_hash=LocalSkillsRegistry._compute_source_hash(root),
        executable_assets={"bin": ["demo-cli.py"], "scripts": [], "python_project": False},
        description="demo",
        content="demo",
    )


@pytest.fixture
def audit_profile(tmp_path, monkeypatch):
    original = client_settings.profile_root
    client_settings.profile_root = str(tmp_path / "profiles")
    monkeypatch.setattr(
        audit_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: "user-a"),
    )
    try:
        yield
    finally:
        client_settings.profile_root = original


@pytest.mark.asyncio
async def test_successful_command_writes_device_bound_audit_record(tmp_path, audit_profile):
    skill = _skill(tmp_path)
    envelope = await SkillExecutionEngine(
        registry=_Registry(skill),
        secret_store=_Secrets(),
        audit_writer=SkillAuditWriter(),
    ).execute(
        "skill::demo::run_skill_command",
        {"argv": ["demo-cli"]},
        {
            "mutation_approved": True,
            "device_id": "device-a",
            "session_id": "session-a",
        },
    )

    records = [
        json.loads(line)
        for line in (get_profile_subdir("user-a", "skills") / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert envelope["ok"] is True
    assert records[0]["qualified_id"] == "skill::demo::run_skill_command"
    assert records[0]["capability"] == "run_skill_command"
    assert records[0]["device_id"] == "device-a"
    assert records[0]["session_id"] == "session-a"
    assert records[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_unapproved_command_is_audited_before_secret_access(tmp_path, audit_profile):
    skill = _skill(tmp_path)
    envelope = await SkillExecutionEngine(
        registry=_Registry(skill),
        secret_store=_Secrets(),
        audit_writer=SkillAuditWriter(),
    ).execute(
        "skill::demo::run_skill_command",
        {"argv": ["demo-cli"]},
        {"mutation_approved": False},
    )

    record = json.loads(
        (get_profile_subdir("user-a", "skills") / "audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert envelope["error"]["code"] == PERMISSION_REQUIRED
    assert record["error_code"] == PERMISSION_REQUIRED


def test_argument_redaction_handles_unicode_quotes_and_backslashes():
    secret = 'sëcret"\\value'

    redacted = SkillAuditWriter._redact_arguments(
        {"argv": ["demo-cli", f"prefix-{secret}-suffix"]},
        {secret},
    )

    assert redacted == {"argv": ["demo-cli", "prefix-<redacted>-suffix"]}


def test_audit_ids_are_unique():
    assert new_audit_id() != new_audit_id()
