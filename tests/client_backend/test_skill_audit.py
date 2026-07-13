"""Tests for client_backend.services.skill_runtime.audit.SkillAuditWriter.

Mirrors the real-bundle pattern in test_skill_execution_engine.py: every
scenario builds a real skill bundle on disk (SKILL.md + skill.json) under
``tmp_path``, scans it with a real ``LocalSkillsRegistry``, and drives the
real ``SkillExecutionEngine`` (which owns a real ``SkillAuditWriter``). Only
the upstream auth service is faked (to pin an active user id without a real
login), mirroring the ``fake_profile`` fixture in test_skill_secrets.py --
but here the auth service is monkeypatched in the AUDIT module, since that
is where ``SkillAuditWriter`` resolves the current user id from.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from client_backend.core.config import client_settings
from client_backend.core.paths import get_profile_subdir
from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_runtime import audit as audit_module
from client_backend.services.skill_runtime.audit import SkillAuditWriter, new_audit_id
from client_backend.services.skill_runtime.execution import SkillExecutionEngine
from client_backend.services.skill_runtime.manager import SkillRuntimeManager
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from shared.skills.errors import INVALID_ARGUMENTS, PERMISSION_REQUIRED


@pytest.fixture
def fake_profile(tmp_path, monkeypatch):
    """Point the profile root at a tmp dir and fake an active user id.

    Yields a mutable ``SimpleNamespace`` so a test can flip to "no active
    user" mid-test by setting ``current_user_id = None``.
    """
    profile_root = tmp_path / "profiles"
    original_profile_root = client_settings.profile_root
    client_settings.profile_root = str(profile_root)

    auth_state = SimpleNamespace(current_user_id="user-a")
    monkeypatch.setattr(
        audit_module,
        "get_upstream_auth_service",
        lambda: SimpleNamespace(get_current_user_id=lambda: auth_state.current_user_id),
    )

    try:
        yield auth_state
    finally:
        client_settings.profile_root = original_profile_root


def _write_skill_md(skill_dir: Path, name: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: Test skill {name}.\n---\n\n# {name}\n\nBody.\n"
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")


def _write_manifest(skill_dir: Path, manifest: dict) -> None:
    (skill_dir / "skill.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_runner(skill_dir: Path, script_name: str, source: str) -> None:
    (skill_dir / script_name).write_text(source, encoding="utf-8")


def _manifest_dict(
    *,
    name: str = "example-skill",
    runtime: dict | None = None,
    capability: dict | None = None,
    secrets: list[dict] | None = None,
) -> dict:
    """A minimal, provider-neutral manifest dict for audit tests."""
    return {
        "schema_version": "1.0",
        "name": name,
        "description": "A skill used to exercise the audit trail.",
        "runtime": runtime or {"type": "python_script", "script": "runner.py"},
        "dependencies": {"python": [], "node": [], "system": []},
        "secrets": secrets or [],
        "permissions": [],
        "capabilities": [
            capability
            or {
                "name": "do_thing",
                "description": "Does a thing.",
                "input_schema": {"type": "object", "properties": {}},
                "execution": {"argv": [], "json_output": False},
                "permissions": [],
                "secrets": [],
                "mutation": False,
            }
        ],
    }


async def _build_registry(root: Path) -> LocalSkillsRegistry:
    registry = LocalSkillsRegistry(skill_roots=[str(root)])
    await registry.scan_skills()
    return registry


def _read_audit_records(user_id: str) -> list[dict]:
    audit_path = get_profile_subdir(user_id, "skills") / "audit.jsonl"
    if not audit_path.exists():
        return []
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.asyncio
class TestSuccessAudit:
    async def test_success_writes_one_record(self, tmp_path, fake_profile):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(
            skill_dir,
            "runner.py",
            "import json\nprint(json.dumps({'ok': True}))\n",
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "list_items",
                    "description": "List items.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                    },
                    "execution": {"argv": [], "json_output": True},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=SkillSecretStore(environ={}))

        envelope = await engine.execute(
            "skill::example-skill::list_items",
            {"label": "not-secret"},
            {"device_id": "device-123", "session_id": "session-456"},
        )

        assert envelope["ok"] is True
        records = _read_audit_records("user-a")
        assert len(records) == 1
        record = records[0]
        assert record["status"] == "ok"
        assert record["error_code"] is None
        assert record["skill"] == "example-skill"
        assert record["capability"] == "list_items"
        assert record["qualified_id"] == "skill::example-skill::list_items"
        assert isinstance(record["duration_ms"], int)
        assert record["audit_id"] == envelope["audit_id"]
        assert record["audit_id"] is not None
        assert record["device_id"] == "device-123"
        assert record["session_id"] == "session-456"
        assert record["arguments_redacted"] == {"label": "not-secret"}
        assert "stdout" not in record
        assert "stderr" not in record


@pytest.mark.asyncio
class TestFailureAudit:
    async def test_failure_writes_an_error_record(self, tmp_path, fake_profile):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "raise SystemExit('should never run')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "event_list",
                    "description": "List events.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"time_min": {"type": "string"}},
                        "required": ["time_min"],
                    },
                    "execution": {"argv": ["{time_min}"], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=SkillSecretStore(environ={}))

        envelope = await engine.execute("skill::example-skill::event_list", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == INVALID_ARGUMENTS
        records = _read_audit_records("user-a")
        assert len(records) == 1
        record = records[0]
        assert record["status"] == "error"
        assert record["error_code"] == INVALID_ARGUMENTS
        assert record["audit_id"] == envelope["audit_id"]
        assert "stdout" not in record
        assert "stderr" not in record


@pytest.mark.asyncio
class TestPermissionDeniedAudit:
    async def test_blocked_mutation_is_audited(self, tmp_path, fake_profile):
        # A blocked mutation returns (does not raise) — it must still be
        # audited as a security-relevant outcome.
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "raise SystemExit('should never run')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "mutate_it",
                    "description": "Mutates state.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": True,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        # Default engine policy has allow_mutation=False -> the mutation is blocked.
        engine = SkillExecutionEngine(registry=registry, secret_store=SkillSecretStore(environ={}))

        envelope = await engine.execute("skill::example-skill::mutate_it", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == PERMISSION_REQUIRED
        records = _read_audit_records("user-a")
        assert len(records) == 1
        assert records[0]["status"] == "error"
        assert records[0]["error_code"] == PERMISSION_REQUIRED
        assert records[0]["audit_id"] == envelope["audit_id"]


@pytest.mark.asyncio
class TestSecretRedactionInAudit:
    async def test_secret_value_never_appears_in_audit_file(self, tmp_path, fake_profile):
        secret_value = "super-secret-value-123"
        secret_store = SkillSecretStore(environ={"API_TOKEN": secret_value})
        manager = SkillRuntimeManager(secret_store=secret_store)

        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(
            skill_dir,
            "runner.py",
            "import os\nprint(os.environ.get('API_TOKEN', ''))\n",
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                secrets=[{"name": "API_TOKEN", "required": True}],
                capability={
                    "name": "echo_secret",
                    "description": "Echo the injected secret.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "note": {"type": "string"},
                            "leaked": {"type": "string"},
                        },
                    },
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": ["API_TOKEN"],
                    "mutation": False,
                },
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=secret_store, manager=manager
        )

        envelope = await engine.execute(
            "skill::example-skill::echo_secret",
            {"note": "harmless-value", "leaked": secret_value},
        )

        assert envelope["ok"] is True
        audit_path = get_profile_subdir("user-a", "skills") / "audit.jsonl"
        raw_bytes = audit_path.read_bytes()
        assert secret_value.encode("utf-8") not in raw_bytes
        assert b"harmless-value" in raw_bytes

        records = _read_audit_records("user-a")
        assert records[0]["arguments_redacted"]["note"] == "harmless-value"
        assert records[0]["arguments_redacted"]["leaked"] == "<redacted>"

    async def test_secret_with_json_special_chars_is_still_redacted(self, tmp_path, fake_profile):
        # Secrets containing a quote, backslash, or non-ASCII char must still be
        # redacted — redacting the json.dumps()'d text (which escapes them)
        # would silently miss them and round-trip the raw secret back.
        secret_value = 'p@ss"word\\with/café'
        secret_store = SkillSecretStore(environ={"API_TOKEN": secret_value})
        manager = SkillRuntimeManager(secret_store=secret_store)

        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "print('done')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                secrets=[{"name": "API_TOKEN", "required": True}],
                capability={
                    "name": "do_it",
                    "description": "Do it.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"leaked": {"type": "string"}, "nested": {"type": "object"}},
                    },
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": ["API_TOKEN"],
                    "mutation": False,
                },
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=secret_store, manager=manager)

        envelope = await engine.execute(
            "skill::example-skill::do_it",
            {"leaked": secret_value, "nested": {"deep": secret_value}},
        )

        assert envelope["ok"] is True
        audit_path = get_profile_subdir("user-a", "skills") / "audit.jsonl"
        assert secret_value.encode("utf-8") not in audit_path.read_bytes()
        record = _read_audit_records("user-a")[0]
        assert record["arguments_redacted"]["leaked"] == "<redacted>"
        # Nested string leaves are redacted too.
        assert record["arguments_redacted"]["nested"]["deep"] == "<redacted>"


@pytest.mark.asyncio
class TestNoUserProfile:
    async def test_no_active_user_skips_audit_without_crashing(self, tmp_path, fake_profile):
        fake_profile.current_user_id = None
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "print('hello')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "say_hello",
                    "description": "Say hello.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=SkillSecretStore(environ={}))

        envelope = await engine.execute("skill::example-skill::say_hello", {})

        assert envelope["ok"] is True
        profile_root = Path(client_settings.profile_root)
        audit_files = list(profile_root.rglob("audit.jsonl")) if profile_root.exists() else []
        assert audit_files == []


@pytest.mark.asyncio
class TestAuditWriteFailureIsSwallowed:
    async def test_audit_append_failure_never_breaks_execution(
        self, tmp_path, fake_profile, monkeypatch
    ):
        def _raise_on_append(_path, _line):
            raise OSError("disk full")

        monkeypatch.setattr(SkillAuditWriter, "_append_line", staticmethod(_raise_on_append))

        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "print('hello')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "say_hello",
                    "description": "Say hello.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=SkillSecretStore(environ={}))

        envelope = await engine.execute("skill::example-skill::say_hello", {})

        assert envelope["ok"] is True
        assert envelope["result"].strip() == "hello"


class TestNewAuditId:
    def test_new_audit_id_format(self):
        assert new_audit_id().startswith("skill-exec-")
