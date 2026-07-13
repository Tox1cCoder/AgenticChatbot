"""Tests for the generic (provider-neutral) example skill fixtures.

Exercises the REAL ``LocalSkillsRegistry`` + ``SkillExecutionEngine`` against
two static, tracked fixture bundles under ``tests/fixtures/skills/``: one
``python_script`` skill (``echo-python``) and one ``binary`` skill
(``binary-probe``). Nothing here is mocked beyond an empty, in-memory secret
store -- these are dry runs that require no live credentials.

These tests prove two things:

1. The skill runtime is a universal, provider-neutral contract -- it can
   load and execute skills it has never seen before, without any per-skill
   special casing.
2. The core runtime source code never hardcodes an example skill's name (a
   regression test that would fail if someone "made it work" by special
   casing ``echo-python``/``binary-probe`` instead of the generic contract).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_runtime.execution import SkillExecutionEngine
from client_backend.services.skill_runtime.manager import SkillRuntimeManager
from client_backend.services.skill_runtime.secrets import SkillSecretStore

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "skills"

# Tokens that would indicate a fixture has drifted away from being
# provider-neutral. Checked case-insensitively against fixture file contents.
_BANNED_PROVIDER_TOKENS = ("google", "gcal", "calendar", "oauth")

# Fixture skill names plus provider tokens that core runtime code must never
# hardcode. If the core runtime contained any of these literals, it would be
# special-casing a specific example skill instead of dispatching generically
# on the manifest contract (name/runtime/capabilities).
_BANNED_RUNTIME_LITERALS = (
    "echo-python",
    "echo_python",
    "binary-probe",
    "binary_probe",
    "google",
    "gcal",
    "google_calendar",
)


async def _build_registry() -> LocalSkillsRegistry:
    registry = LocalSkillsRegistry(skill_roots=[str(FIXTURES_ROOT)])
    await registry.scan_skills()
    return registry


def _core_runtime_files() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    skill_runtime_dir = repo_root / "client_backend" / "services" / "skill_runtime"
    files = [
        repo_root / "shared" / "skills" / "manifest.py",
        repo_root / "shared" / "skills" / "errors.py",
        *sorted(skill_runtime_dir.glob("*.py")),
    ]
    for path in files:
        assert path.is_file(), f"expected core runtime file to exist: {path}"
    return files


@pytest.mark.asyncio
class TestFixturesLoadAndValidate:
    async def test_both_fixtures_load_with_a_parsed_manifest(self):
        registry = await _build_registry()

        echo_skill = registry.get_skill("echo-python")
        binary_skill = registry.get_skill("binary-probe")

        assert echo_skill is not None
        assert echo_skill.manifest is not None
        assert echo_skill.manifest_error is None
        assert echo_skill.manifest.runtime.type == "python_script"

        assert binary_skill is not None
        assert binary_skill.manifest is not None
        assert binary_skill.manifest_error is None
        assert binary_skill.manifest.runtime.type == "binary"


@pytest.mark.asyncio
class TestEchoPythonExecution:
    async def test_echo_python_executes_end_to_end_without_credentials(self):
        registry = await _build_registry()
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute(
            "skill::echo-python::echo", {"message": "hello fixtures"}
        )

        assert envelope["ok"] is True
        assert envelope["result"] == {
            "echoed": "hello fixtures",
            "runtime": "python_script",
        }


@pytest.mark.asyncio
class TestBinaryProbeExecution:
    async def test_binary_probe_executes_or_skips_when_python_unresolvable(self):
        registry = await _build_registry()
        skill = registry.get_skill("binary-probe")
        assert skill is not None
        assert skill.manifest is not None

        manager = SkillRuntimeManager(secret_store=SkillSecretStore(environ={}))
        readiness = manager.evaluate_readiness(skill.manifest, skill.manifest_error)

        if shutil.which("python") is None:
            # Environments without a bare "python" on PATH (e.g. only
            # "python3") still validated the manifest correctly; readiness
            # must report the missing executable rather than "invalid".
            assert readiness.status != "invalid"
            assert readiness.missing_executable == "python"
            pytest.skip("'python' is not resolvable on PATH in this environment")

        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )
        envelope = await engine.execute("skill::binary-probe::probe", {})

        assert envelope["ok"] is True
        assert envelope["result"].strip() == "binary probe ok"


class TestFixturesAreProviderNeutral:
    def test_fixture_files_contain_no_banned_provider_tokens(self):
        candidate_files = [
            path
            for path in FIXTURES_ROOT.rglob("*")
            if path.is_file() and path.suffix in (".md", ".json")
        ]
        # Guards against a silent no-op: if the glob ever matched nothing
        # (e.g. fixtures moved/renamed), the assertions below would vacuously
        # "pass" without checking anything.
        assert len(candidate_files) == 4

        for path in candidate_files:
            text = path.read_text(encoding="utf-8").lower()
            for token in _BANNED_PROVIDER_TOKENS:
                assert token not in text, f"{path} contains banned provider token {token!r}"


class TestCoreRuntimeHasNoExampleSpecificChecks:
    def test_core_runtime_source_never_hardcodes_a_fixture_skill_name(self):
        core_files = _core_runtime_files()
        assert len(core_files) >= 3

        for path in core_files:
            text = path.read_text(encoding="utf-8").lower()
            for literal in _BANNED_RUNTIME_LITERALS:
                assert literal not in text, (
                    f"{path} contains banned literal {literal!r}; core skill "
                    "runtime code must never special-case a specific skill"
                )
