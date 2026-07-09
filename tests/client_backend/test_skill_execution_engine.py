"""Tests for client_backend.services.skill_runtime.execution.SkillExecutionEngine.

Every scenario builds a real skill bundle on disk (SKILL.md + skill.json)
under ``tmp_path``, scans it with a real ``LocalSkillsRegistry``, and runs
the engine against a real short-lived Python subprocess (the
``python_script`` runtime, invoked via ``sys.executable`` so it is
deterministic and cross-platform). Only the secret store is faked (via an
injected environ mapping); nothing here mocks subprocess or jsonschema.
"""

import json
from pathlib import Path

import pytest

from client_backend.services.local_skills_registry import LocalSkillsRegistry
from client_backend.services.skill_runtime import execution as execution_module
from client_backend.services.skill_runtime.execution import SkillExecutionEngine
from client_backend.services.skill_runtime.manager import SkillReadiness, SkillRuntimeManager
from client_backend.services.skill_runtime.secrets import SkillSecretStore
from shared.skills.errors import (
    COMMAND_NOT_FOUND,
    EXECUTION_TIMEOUT,
    INVALID_ARGUMENTS,
    MISSING_SECRET,
    NON_JSON_OUTPUT,
    OUTPUT_TOO_LARGE,
    PERMISSION_REQUIRED,
    RUNTIME_ERROR,
)


class _AlwaysReadyManager:
    """Readiness stub that always reports ``ready``.

    ``SkillRuntimeManager.evaluate_readiness`` also independently checks
    binary-command presence and required secrets (see
    ``test_skill_runtime_manager.py``), which would otherwise short-circuit
    a couple of these scenarios with ``SKILL_NOT_READY`` before the
    execution engine's OWN command-not-found / missing-secret checks ever
    run. Injecting this stub via the engine's ``manager=`` parameter isolates
    the engine-level check under test from the (separately-tested) readiness
    evaluator.
    """

    def evaluate_readiness(self, manifest, manifest_error=None) -> SkillReadiness:
        return SkillReadiness(status="ready")


def _write_skill_md(skill_dir: Path, name: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\n\n# {name}\n\nBody.\n"
    )
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
    """A minimal, provider-neutral manifest dict for execution-engine tests."""
    return {
        "schema_version": "1.0",
        "name": name,
        "description": "A skill used to exercise the execution engine.",
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


@pytest.mark.asyncio
class TestSuccessJsonOutput:
    async def test_json_output_capability_returns_parsed_result(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(
            skill_dir,
            "runner.py",
            "import json\nprint(json.dumps({'items': [1, 2, 3]}))\n",
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "list_items",
                    "description": "List items.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": True},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute("skill::example-skill::list_items", {})

        assert envelope["ok"] is True
        assert envelope["skill"] == "example-skill"
        assert envelope["capability"] == "list_items"
        assert envelope["result"] == {"items": [1, 2, 3]}
        assert isinstance(envelope["duration_ms"], int)
        assert envelope["audit_id"] is None


@pytest.mark.asyncio
class TestSuccessTextOutput:
    async def test_text_output_capability_returns_raw_stdout(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "print('hello from skill')\n")
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
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute("skill::example-skill::say_hello", {})

        assert envelope["ok"] is True
        assert envelope["result"].strip() == "hello from skill"


@pytest.mark.asyncio
class TestInvalidArguments:
    async def test_missing_required_argument_is_rejected_before_spawn(self, tmp_path):
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
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute("skill::example-skill::event_list", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == INVALID_ARGUMENTS


@pytest.mark.asyncio
class TestCommandNotFound:
    async def test_missing_binary_command_is_reported(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                runtime={"type": "binary", "command": "totally-missing-cmd-xyz"},
                capability={
                    "name": "run_thing",
                    "description": "Run a thing.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                },
            ),
        )
        registry = await _build_registry(tmp_path)
        # Bypass readiness's own (separately-tested) binary-missing check so
        # this test isolates the engine's own COMMAND_NOT_FOUND path.
        engine = SkillExecutionEngine(
            registry=registry,
            secret_store=SkillSecretStore(environ={}),
            manager=_AlwaysReadyManager(),
        )

        envelope = await engine.execute("skill::example-skill::run_thing", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == COMMAND_NOT_FOUND


@pytest.mark.asyncio
class TestTimeout:
    async def test_slow_capability_is_killed_and_reported_as_timeout(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "import time\ntime.sleep(5)\nprint('done')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "slow_thing",
                    "description": "Take a while.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute(
            "skill::example-skill::slow_thing", {}, {"timeout_seconds": 1}
        )

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == EXECUTION_TIMEOUT


@pytest.mark.asyncio
class TestOutputTooLarge:
    async def test_oversized_stdout_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(execution_module, "MAX_OUTPUT_BYTES", 16)
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "print('x' * 1000)\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "big_output",
                    "description": "Print a lot.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute("skill::example-skill::big_output", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == OUTPUT_TOO_LARGE


@pytest.mark.asyncio
class TestNonJsonOutput:
    async def test_declared_json_output_that_is_not_json_is_rejected(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "print('not json')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "bad_json",
                    "description": "Claims JSON, isn't.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": True},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute("skill::example-skill::bad_json", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == NON_JSON_OUTPUT


@pytest.mark.asyncio
class TestPermissionBlockedMutation:
    async def test_mutation_capability_is_blocked_by_default_policy(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "raise SystemExit('should never run')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "delete_thing",
                    "description": "Delete a thing.",
                    "input_schema": {"type": "object", "properties": {"target": {}}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": True,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        # Default engine policy: allow_mutation=False.
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )
        sensitive_value = "super-secret-should-not-appear"

        envelope = await engine.execute(
            "skill::example-skill::delete_thing", {"target": sensitive_value}
        )

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == PERMISSION_REQUIRED
        serialized = json.dumps(envelope)
        assert sensitive_value not in serialized


@pytest.mark.asyncio
class TestMissingSecret:
    async def test_missing_required_secret_is_reported(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "raise SystemExit('should never run')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "call_api",
                    "description": "Call an API.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": ["API_TOKEN"],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        # Bypass readiness's own (separately-tested) missing-secret check so
        # this test isolates the engine's own MISSING_SECRET resolution path.
        engine = SkillExecutionEngine(
            registry=registry,
            secret_store=SkillSecretStore(environ={}),
            manager=_AlwaysReadyManager(),
        )

        envelope = await engine.execute("skill::example-skill::call_api", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == MISSING_SECRET
        assert envelope["error"]["repair"] == {"type": "configure_secret", "secret": "API_TOKEN"}


@pytest.mark.asyncio
class TestSecretRedaction:
    async def test_secret_reaches_child_but_never_appears_in_the_envelope(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(
            skill_dir,
            "runner.py",
            "import os\n"
            "token = os.environ.get('API_TOKEN', '')\n"
            "print(token)\n"
            "print('received-length:' + str(len(token)))\n",
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "echo_secret",
                    "description": "Echo the injected secret.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": ["API_TOKEN"],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        secret_value = "super-secret-value-123"
        secret_store = SkillSecretStore(environ={"API_TOKEN": secret_value})
        # Give the readiness manager the SAME secret store, so readiness
        # legitimately sees the secret as present (this is not a stub --
        # the secret really is configured end to end).
        manager = SkillRuntimeManager(secret_store=secret_store)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=secret_store, manager=manager
        )

        envelope = await engine.execute("skill::example-skill::echo_secret", {})

        assert envelope["ok"] is True
        serialized = json.dumps(envelope)
        assert secret_value not in serialized
        assert f"received-length:{len(secret_value)}" in envelope["stdout"]


@pytest.mark.asyncio
class TestArgvPlaceholderRendering:
    async def test_placeholder_value_is_a_discrete_argv_element(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(
            skill_dir,
            "runner.py",
            "import json\nimport sys\nprint(json.dumps(sys.argv[1:]))\n",
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "echo_argv",
                    "description": "Echo argv.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "execution": {"argv": ["--name", "{name}"], "json_output": True},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute(
            "skill::example-skill::echo_argv", {"name": "hello world"}
        )

        assert envelope["ok"] is True
        # Exactly two argv elements -- proves "hello world" arrived as ONE
        # element, never split by a shell into ["--name", "hello", "world"].
        assert envelope["result"] == ["--name", "hello world"]


@pytest.mark.asyncio
class TestPathEscape:
    async def test_script_path_escaping_skill_dir_is_refused(self, tmp_path):
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        marker_path = tmp_path / "evil_ran.marker"
        (tmp_path / "evil.py").write_text(
            f"open(r'{marker_path}', 'w').write('pwned')\n", encoding="utf-8"
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                runtime={"type": "python_script", "script": "../evil.py"},
                capability={
                    "name": "run_evil",
                    "description": "Attempt to escape the skill dir.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                },
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(
            registry=registry, secret_store=SkillSecretStore(environ={})
        )

        envelope = await engine.execute("skill::example-skill::run_evil", {})

        assert envelope["ok"] is False
        assert envelope["error"]["code"] == RUNTIME_ERROR
        assert not marker_path.exists()


@pytest.mark.asyncio
class TestScopedEnvironment:
    async def test_child_env_excludes_unrelated_process_env_but_keeps_path(
        self, tmp_path, monkeypatch
    ):
        # A capability that declares NO secrets and dumps its whole os.environ
        # must NOT see an unrelated env var (e.g. another skill's env-backed
        # secret) from the sidecar process -- only allow-listed infra vars.
        monkeypatch.setenv("OTHER_SKILL_SECRET_XYZ", "leak-me-if-you-can")
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(
            skill_dir,
            "runner.py",
            "import json, os\nprint(json.dumps(dict(os.environ)))\n",
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                capability={
                    "name": "dump_env",
                    "description": "Dump the environment.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": True},
                    "permissions": [],
                    "secrets": [],
                    "mutation": False,
                }
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=SkillSecretStore(environ={}))

        envelope = await engine.execute("skill::example-skill::dump_env", {})

        assert envelope["ok"] is True
        child_env = envelope["result"]
        assert "OTHER_SKILL_SECRET_XYZ" not in child_env
        assert "leak-me-if-you-can" not in json.dumps(envelope)
        assert "PATH" in child_env  # infra var still passes through


@pytest.mark.asyncio
class TestSecretRedactionSubstringCollision:
    async def test_shorter_secret_substring_of_longer_is_fully_redacted(self, tmp_path):
        # One secret value is a prefix of another; redaction must strip both
        # completely regardless of iteration order (longest-first).
        short_value = "ZZSECRETZZ"
        long_value = "ZZSECRETZZ_EXTRA_TAIL"
        secret_store = SkillSecretStore(
            environ={"SHORT_TOKEN": short_value, "LONG_TOKEN": long_value}
        )
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(
            skill_dir,
            "runner.py",
            "import os\nprint(os.environ['SHORT_TOKEN'])\nprint(os.environ['LONG_TOKEN'])\n",
        )
        _write_manifest(
            skill_dir,
            _manifest_dict(
                secrets=[
                    {"name": "SHORT_TOKEN", "required": True},
                    {"name": "LONG_TOKEN", "required": True},
                ],
                capability={
                    "name": "echo_secrets",
                    "description": "Echo both secrets.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": ["SHORT_TOKEN", "LONG_TOKEN"],
                    "mutation": False,
                },
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=secret_store)

        envelope = await engine.execute("skill::example-skill::echo_secrets", {})

        serialized = json.dumps(envelope)
        assert short_value not in serialized
        assert long_value not in serialized
        # The tail that would leak if the shorter value were replaced first.
        assert "_EXTRA_TAIL" not in serialized


@pytest.mark.asyncio
class TestManagerSharesSecretStore:
    async def test_injected_secret_store_is_used_by_readiness(self, tmp_path):
        # Constructing the engine with only a custom secret_store (no manager)
        # must make readiness use that SAME store -- otherwise a required
        # secret present only in the injected store would make readiness
        # report not_ready and the capability would fail with SKILL_NOT_READY.
        secret_store = SkillSecretStore(environ={"REQUIRED_TOKEN": "present"})
        skill_dir = tmp_path / "example-skill"
        _write_skill_md(skill_dir, "example-skill")
        _write_runner(skill_dir, "runner.py", "print('ok')\n")
        _write_manifest(
            skill_dir,
            _manifest_dict(
                secrets=[{"name": "REQUIRED_TOKEN", "required": True}],
                capability={
                    "name": "needs_secret",
                    "description": "Needs a secret.",
                    "input_schema": {"type": "object", "properties": {}},
                    "execution": {"argv": [], "json_output": False},
                    "permissions": [],
                    "secrets": ["REQUIRED_TOKEN"],
                    "mutation": False,
                },
            ),
        )
        registry = await _build_registry(tmp_path)
        engine = SkillExecutionEngine(registry=registry, secret_store=secret_store)

        envelope = await engine.execute("skill::example-skill::needs_secret", {})

        assert envelope["ok"] is True


class TestClampTimeout:
    def test_clamp_timeout_bounds_and_fallbacks(self):
        default = float(execution_module.DEFAULT_TIMEOUT_SECONDS)
        maximum = float(execution_module.MAX_TIMEOUT_SECONDS)
        clamp = SkillExecutionEngine._clamp_timeout

        assert clamp(None) == default
        assert clamp(0) == default
        assert clamp(-5) == default
        assert clamp("30") == default  # non-numeric
        assert clamp(True) == default  # bool is not a real timeout
        assert clamp(10) == 10.0
        assert clamp(10_000) == maximum
        assert clamp(float("nan")) == default  # NaN must not slip through min()
        assert clamp(float("inf")) == default
