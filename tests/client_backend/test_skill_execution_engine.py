import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from client_backend.core.config import client_settings
from client_backend.services.local_skills_registry import LocalSkillsRegistry, SkillMetadata
from client_backend.services.skill_runtime import execution as execution_module
from client_backend.services.skill_runtime.execution import SkillExecutionEngine
from shared.skills.errors import (
    COMMAND_NOT_FOUND,
    EXECUTION_TIMEOUT,
    INVALID_ARGUMENTS,
    OUTPUT_TOO_LARGE,
    PERMISSION_REQUIRED,
    SKILL_RUNTIME_STALE,
)


class _Registry:
    def __init__(self, skill):
        self.skill = skill

    def get_skill(self, name):
        return self.skill if name == self.skill.name else None


class _Environment:
    def __init__(self, command_dir: Path | None = None, python: Path | None = None):
        self._command_dir = command_dir
        self._python = python

    def inspect(self, _skill):
        if self._command_dir is None:
            return {"status": "setup_required", "commands": []}
        return {"status": "ready", "commands": ["runtime-cli"], "runtime_id": "runtime"}

    def command_directory(self, _skill):
        return self._command_dir

    def python_executable(self, _skill):
        return self._python


class _Secrets:
    def __init__(self, values=None):
        self.values = values or {}
        self.calls = []

    def get_for_skill(self, name):
        self.calls.append(name)
        return dict(self.values)


class _Audit:
    def __init__(self):
        self.records = []

    def write(self, **record):
        self.records.append(record)


class _FakeJob:
    def close(self):
        pass


def _skill(tmp_path: Path, script: str, *, command_name: str = "demo-cli"):
    root = tmp_path / "demo-skill"
    root.mkdir()
    (root / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    executable = bin_dir / f"{command_name}.py"
    executable.write_text(script, encoding="utf-8")
    return SkillMetadata(
        name="demo-skill",
        path=root / "SKILL.md",
        bundle_root=root,
        source_hash=LocalSkillsRegistry._compute_source_hash(root),
        executable_assets={
            "bin": [executable.name],
            "scripts": [],
            "python_project": False,
        },
        description="demo",
        content="demo",
    )


def _engine(skill, *, environment=None, secrets=None, audit=None):
    environment = environment or _Environment()
    return SkillExecutionEngine(
        registry=_Registry(skill),
        environment_manager=environment,
        secret_store=secrets or _Secrets(),
        audit_writer=audit or _Audit(),
    )


def _context(**extra):
    return {
        "mutation_approved": True,
        "device_id": "device-a",
        "session_id": "session-a",
        **extra,
    }


@pytest.mark.asyncio
async def test_bundled_command_runs_when_absent_from_global_path(tmp_path, monkeypatch):
    skill = _skill(
        tmp_path,
        "import json, sys\nprint(json.dumps({'argv': sys.argv[1:]}))\n",
    )
    monkeypatch.setenv("PATH", "")
    engine = _engine(skill)

    envelope = await engine.execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli", "hello world", "$(not-a-shell)"]},
        _context(),
    )

    assert envelope["ok"] is True
    assert envelope["result"] == {"argv": ["hello world", "$(not-a-shell)"]}


@pytest.mark.asyncio
async def test_relative_python_script_under_scripts_is_supported(tmp_path):
    skill = _skill(tmp_path, "print('bin')\n")
    scripts = skill.bundle_root / "scripts"
    scripts.mkdir()
    (scripts / "inspect.py").write_text("print('script-ok')\n", encoding="utf-8")
    skill.executable_assets["scripts"] = ["scripts/inspect.py"]
    skill.source_hash = LocalSkillsRegistry._compute_source_hash(skill.bundle_root)

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["scripts/inspect.py"], "cwd": "skill"},
        _context(),
    )

    assert envelope["ok"] is True
    assert envelope["result"] == "script-ok\n"


@pytest.mark.asyncio
async def test_workspace_cwd_defaults_to_first_configured_workspace_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill = _skill(
        tmp_path,
        "import json, os\nprint(json.dumps({'cwd': os.getcwd()}))\n",
    )
    monkeypatch.setattr(client_settings, "workspace_roots", [str(workspace)])

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"], "cwd": "workspace"},
        _context(),
    )

    assert envelope["ok"] is True
    assert Path(envelope["result"]["cwd"]).resolve() == workspace.resolve()


@pytest.mark.asyncio
async def test_approval_is_required_before_secret_resolution(tmp_path):
    skill = _skill(tmp_path, "print('no')\n")
    secrets = _Secrets({"TOKEN": "secret"})

    envelope = await _engine(skill, secrets=secrets).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        {"mutation_approved": False},
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == PERMISSION_REQUIRED
    assert secrets.calls == []


@pytest.mark.asyncio
async def test_live_bundle_tamper_is_rejected_before_secret_resolution(tmp_path):
    skill = _skill(tmp_path, "print('original')\n")
    skill.source_hash = LocalSkillsRegistry._compute_source_hash(skill.bundle_root)
    secrets = _Secrets({"TOKEN": "must-not-reach-tampered-code"})
    (skill.bundle_root / "bin" / "demo-cli.py").write_text(
        "import os\nprint(os.environ.get('TOKEN'))\n",
        encoding="utf-8",
    )

    envelope = await _engine(skill, secrets=secrets).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(),
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == SKILL_RUNTIME_STALE
    assert secrets.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"argv": "demo-cli --help"},
        {"argv": []},
        {"argv": ["demo-cli"], "cwd": "outside"},
        {"argv": ["demo-cli"], "extra": True},
    ],
)
async def test_fixed_command_schema_rejects_invalid_arguments(tmp_path, arguments):
    skill = _skill(tmp_path, "print('no')\n")

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        arguments,
        _context(),
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == INVALID_ARGUMENTS


@pytest.mark.asyncio
async def test_system_command_and_path_traversal_are_not_owned_commands(tmp_path):
    skill = _skill(tmp_path, "print('no')\n")
    engine = _engine(skill)

    system = await engine.execute(
        "skill::demo-skill::run_skill_command",
        {"argv": [Path(sys.executable).name]},
        _context(),
    )
    traversal = await engine.execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["../outside.py"]},
        _context(),
    )

    assert system["error"]["code"] == COMMAND_NOT_FOUND
    assert traversal["error"]["code"] == COMMAND_NOT_FOUND


@pytest.mark.asyncio
async def test_child_receives_scoped_roots_path_and_only_own_secrets(tmp_path, monkeypatch):
    skill = _skill(
        tmp_path,
        """
import json, os
print(json.dumps({
    'root': os.environ.get('SKILL_ROOT'),
    'runtime': os.environ.get('SKILL_RUNTIME_ROOT'),
    'path0': os.environ.get('PATH', '').split(os.pathsep)[0],
    'path': os.environ.get('PATH', ''),
    'token': os.environ.get('DEMO_TOKEN'),
    'foreign': os.environ.get('FOREIGN_TOKEN'),
}))
""".strip(),
    )
    secrets = _Secrets({"DEMO_TOKEN": "very-secret"})
    monkeypatch.setenv("FOREIGN_TOKEN", "must-not-leak")
    global_path = str(tmp_path / "global-bin")
    monkeypatch.setenv("PATH", global_path)

    envelope = await _engine(skill, secrets=secrets).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(),
    )

    assert envelope["ok"] is True
    assert envelope["result"]["root"] == str(skill.bundle_root)
    assert envelope["result"]["runtime"] == ""
    assert envelope["result"]["path0"] == str(skill.bundle_root / "bin")
    assert global_path not in envelope["result"]["path"]
    assert envelope["result"]["token"] == "<redacted>"
    assert envelope["result"]["foreign"] is None


@pytest.mark.asyncio
async def test_runtime_command_directory_is_resolved_without_global_path(tmp_path):
    skill = _skill(tmp_path, "print('bin')\n")
    skill.executable_assets = {"bin": [], "scripts": [], "python_project": True}
    runtime_bin = tmp_path / "runtime" / "Scripts"
    runtime_bin.mkdir(parents=True)
    (runtime_bin / "runtime-cli.py").write_text("print('runtime-ok')\n", encoding="utf-8")
    environment = _Environment(runtime_bin, Path(sys.executable))

    envelope = await _engine(skill, environment=environment).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["runtime-cli"]},
        _context(),
    )

    assert envelope["ok"] is True
    assert envelope["result"] == "runtime-ok\n"


@pytest.mark.asyncio
async def test_prepared_runtime_rejects_undeclared_environment_commands(tmp_path):
    skill = _skill(tmp_path, "print('bin')\n")
    skill.executable_assets = {"bin": [], "scripts": [], "python_project": True}
    runtime_bin = tmp_path / "runtime" / "Scripts"
    runtime_bin.mkdir(parents=True)
    (runtime_bin / "runtime-cli.py").write_text("print('owned')\n", encoding="utf-8")
    (runtime_bin / "pip.py").write_text("print('not owned')\n", encoding="utf-8")
    environment = _Environment(runtime_bin, Path(sys.executable))

    envelope = await _engine(skill, environment=environment).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["pip"]},
        _context(),
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == COMMAND_NOT_FOUND


@pytest.mark.asyncio
async def test_timeout_is_normalized(tmp_path):
    skill = _skill(tmp_path, "import time\ntime.sleep(2)\n")

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(timeout_seconds=0.05),
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == EXECUTION_TIMEOUT


@pytest.mark.asyncio
async def test_timeout_includes_process_wait_after_child_closes_output(tmp_path):
    skill = _skill(
        tmp_path,
        "import os, time\nos.close(1)\nos.close(2)\ntime.sleep(2)\n",
    )
    started = time.monotonic()

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(timeout_seconds=0.05),
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == EXECUTION_TIMEOUT
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_timeout_terminates_descendant_processes(tmp_path):
    marker = tmp_path / "descendant-survived.txt"
    child_code = (
        "import time; from pathlib import Path; "
        f"time.sleep(0.4); Path({str(marker)!r}).write_text('survived')"
    )
    parent_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "time.sleep(2)\n"
    )
    skill = _skill(tmp_path, parent_code)

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(timeout_seconds=0.05),
    )
    await asyncio.sleep(0.6)

    assert envelope["error"]["code"] == EXECUTION_TIMEOUT
    assert not marker.exists()


@pytest.mark.asyncio
async def test_timeout_terminates_descendant_after_parent_already_exited(tmp_path):
    marker = tmp_path / "orphan-survived.txt"
    child_code = (
        "import time; from pathlib import Path; "
        f"time.sleep(0.4); Path({str(marker)!r}).write_text('survived')"
    )
    parent_code = (
        f"import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
    )
    skill = _skill(tmp_path, parent_code)

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(timeout_seconds=0.05),
    )
    await asyncio.sleep(0.6)

    assert envelope["error"]["code"] == EXECUTION_TIMEOUT
    assert not marker.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows process-containment protocol")
@pytest.mark.asyncio
async def test_windows_skill_command_is_released_only_after_job_attachment(tmp_path, monkeypatch):
    events = []

    class FakeStdin:
        def write(self, payload):
            events.append(("write", json.loads(payload.decode("utf-8"))))

        async def drain(self):
            pass

        def close(self):
            pass

    class FakeProcess:
        pid = 123
        stdin = FakeStdin()

    async def fake_spawn(*args, **kwargs):
        events.append(("spawn", list(args), kwargs))
        return FakeProcess()

    def fake_attach(pid):
        events.append(("attach", pid))
        return _FakeJob()

    monkeypatch.setattr(execution_module.asyncio, "create_subprocess_exec", fake_spawn)
    monkeypatch.setattr(execution_module._WindowsKillJob, "attach", fake_attach)

    process, job = await SkillExecutionEngine._spawn_contained_process(
        [r"C:\skill\fast.exe", "arg"],
        tmp_path,
        {"PATH": ""},
    )

    assert process.pid == 123
    assert isinstance(job, _FakeJob)
    assert [event[0] for event in events] == ["spawn", "attach", "write"]
    assert r"C:\skill\fast.exe" not in events[0][1]
    assert events[2][1] == [r"C:\skill\fast.exe", "arg"]


@pytest.mark.asyncio
async def test_output_limit_is_enforced(tmp_path, monkeypatch):
    skill = _skill(tmp_path, "print('x' * 100)\n")
    monkeypatch.setattr(execution_module, "MAX_OUTPUT_BYTES", 20)

    envelope = await _engine(skill).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(),
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == OUTPUT_TOO_LARGE


@pytest.mark.asyncio
async def test_audit_record_is_bound_to_device_session_and_fixed_capability(tmp_path):
    skill = _skill(tmp_path, "print('ok')\n")
    audit = _Audit()

    envelope = await _engine(skill, audit=audit).execute(
        "skill::demo-skill::run_skill_command",
        {"argv": ["demo-cli"]},
        _context(),
    )

    assert envelope["ok"] is True
    assert audit.records[0]["skill"] == "demo-skill"
    assert audit.records[0]["capability"] == "run_skill_command"
    assert audit.records[0]["device_id"] == "device-a"
    assert audit.records[0]["session_id"] == "session-a"
