"""Tests for client_backend.services.skill_runtime.manager: readiness checks."""

import shutil

import pytest

from client_backend.services.skill_runtime.manager import SkillReadiness, SkillRuntimeManager
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from shared.skills.manifest import load_manifest


def _manifest_dict(
    *,
    runtime: dict | None = None,
    python_deps: list[str] | None = None,
    secrets: list[dict] | None = None,
    capability_secrets: list[str] | None = None,
) -> dict:
    """Build a minimal, provider-neutral manifest dict for readiness tests."""
    return {
        "schema_version": "1.0",
        "name": "example-skill",
        "description": "A skill used to exercise readiness checks.",
        "runtime": runtime or {"type": "python_module", "module": "skills.example.cli"},
        "dependencies": {"python": python_deps or [], "node": [], "system": []},
        "secrets": secrets or [],
        "permissions": [],
        "capabilities": [
            {
                "name": "do_thing",
                "description": "Does a thing.",
                "input_schema": {"type": "object", "properties": {}},
                "execution": {"argv": []},
                "permissions": [],
                "secrets": capability_secrets or [],
                "mutation": False,
            }
        ],
    }


class TestReadySkill:
    def test_python_module_skill_with_installed_dependency_is_ready(self):
        manifest = load_manifest(_manifest_dict(python_deps=["pydantic"]))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "ready"
        assert readiness.missing_dependencies == []
        assert readiness.missing_secrets == []
        assert readiness.missing_executable is None
        assert readiness.unsupported_runtime is None
        assert readiness.repair_hints == []


class TestMissingSecret:
    def test_missing_required_secret_is_not_ready_with_repair_hint(self):
        manifest = load_manifest(
            _manifest_dict(
                secrets=[{"name": "EXAMPLE_TOKEN", "required": True, "description": "token"}]
            )
        )
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "not_ready"
        assert readiness.missing_secrets == ["EXAMPLE_TOKEN"]
        assert {"type": "configure_secret", "secret": "EXAMPLE_TOKEN"} in readiness.repair_hints

    def test_capability_secret_is_also_required(self):
        manifest = load_manifest(
            _manifest_dict(capability_secrets=["CAPABILITY_TOKEN"]),
        )
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "not_ready"
        assert "CAPABILITY_TOKEN" in readiness.missing_secrets


class TestSecretPresent:
    def test_secret_present_in_injected_environ_is_not_missing(self):
        manifest = load_manifest(
            _manifest_dict(
                secrets=[{"name": "EXAMPLE_TOKEN", "required": True, "description": "token"}]
            )
        )
        manager = SkillRuntimeManager(SkillSecretStore(environ={"EXAMPLE_TOKEN": "shh"}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "ready"
        assert readiness.missing_secrets == []


class TestMissingBinary:
    def test_missing_binary_executable_is_not_ready_with_repair_hint(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda command: None)
        manifest = load_manifest(
            _manifest_dict(runtime={"type": "binary", "command": "totally-missing-cmd-xyz"})
        )
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "not_ready"
        assert readiness.missing_executable == "totally-missing-cmd-xyz"
        assert {
            "type": "install_command",
            "command": "totally-missing-cmd-xyz",
        } in readiness.repair_hints

    def test_present_binary_executable_is_not_missing(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda command: f"/usr/bin/{command}")
        manifest = load_manifest(_manifest_dict(runtime={"type": "binary", "command": "ls"}))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "ready"
        assert readiness.missing_executable is None


class TestBinaryMissingCommand:
    def test_binary_runtime_without_command_is_not_ready_with_invalid_manifest_hint(self):
        manifest = load_manifest(_manifest_dict(runtime={"type": "binary", "command": None}))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "not_ready"
        assert readiness.missing_executable is None
        assert {
            "type": "invalid_manifest",
            "reason": "binary runtime missing command",
        } in readiness.repair_hints
        assert readiness.detail == "binary runtime requires a 'command'"


class TestMissingDependency:
    def test_missing_python_dependency_is_not_ready_with_repair_hint(self):
        manifest = load_manifest(
            _manifest_dict(python_deps=["this-distribution-does-not-exist-zzz>=1"])
        )
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "not_ready"
        assert "this-distribution-does-not-exist-zzz>=1" in readiness.missing_dependencies
        assert {
            "type": "install_dependency",
            "ecosystem": "python",
            "dependency": "this-distribution-does-not-exist-zzz>=1",
        } in readiness.repair_hints

    def test_leading_whitespace_dependency_is_still_checked(self):
        # A requirement with surrounding whitespace is valid PEP 508 and must
        # NOT be silently skipped — it must be presence-checked like any other.
        manifest = load_manifest(
            _manifest_dict(python_deps=["  this-distribution-does-not-exist-zzz"])
        )
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.status == "not_ready"
        assert "  this-distribution-does-not-exist-zzz" in readiness.missing_dependencies

    @pytest.mark.parametrize("unparseable", ["   ", ";", "==1.0"])
    def test_genuinely_unparseable_dependency_is_skipped_not_reported(self, unparseable):
        # Only truly unparseable requirement strings are skipped, so we never
        # false-fail readiness on a string we cannot interpret.
        manifest = load_manifest(_manifest_dict(python_deps=[unparseable]))
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(manifest)

        assert readiness.missing_dependencies == []
        assert readiness.status == "ready"


class TestInvalidManifest:
    def test_manifest_error_yields_invalid_status_with_detail(self):
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(None, manifest_error="invalid skill.json: bad json")

        assert readiness.status == "invalid"
        assert readiness.detail == "invalid skill.json: bad json"


class TestInstructionOnlySkill:
    def test_no_manifest_and_no_error_is_instruction_only(self):
        manager = SkillRuntimeManager(SkillSecretStore(environ={}))

        readiness = manager.evaluate_readiness(None, None)

        assert readiness.status == "instruction_only"
        assert readiness.missing_dependencies == []
        assert readiness.missing_secrets == []


class TestToSummary:
    def test_to_summary_is_json_safe_and_excludes_catalog_fields(self):
        readiness = SkillReadiness(
            status="not_ready",
            missing_dependencies=["click>=8"],
            missing_secrets=["TOKEN"],
            missing_executable=None,
            unsupported_runtime=None,
            repair_hints=[
                {"type": "install_dependency", "ecosystem": "python", "dependency": "click>=8"}
            ],
            detail=None,
        )

        summary = readiness.to_summary()

        assert summary["status"] == "not_ready"
        assert summary["missing_dependencies"] == ["click>=8"]
        assert summary["missing_secrets"] == ["TOKEN"]
        assert "capability_count" not in summary
        assert "permissions" not in summary


@pytest.mark.parametrize("environ", [{}])
def test_default_secret_store_defaults_to_os_environ(environ):
    # Sanity check that SkillRuntimeManager() with no args doesn't blow up
    # and uses SkillSecretStore() (which defaults to os.environ) rather than
    # requiring an explicit store.
    manager = SkillRuntimeManager()
    manifest = load_manifest(_manifest_dict())

    readiness = manager.evaluate_readiness(manifest)

    assert readiness.status == "ready"
