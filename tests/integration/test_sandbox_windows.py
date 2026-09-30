"""The sandbox account on a real machine: what it can and cannot change.

Skips unless ``python -m client_backend sandbox setup`` has been run here.
Works in a scratch folder under ProgramData so the user's profile is never
touched, and runs commands through the real launcher as the account.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows sandbox only")

REPO = Path(__file__).resolve().parents[2]
REAL_PROFILE = Path(os.environ.get("LOCALAPPDATA", "")) / "KaniDesktop"


def _real_account_name() -> str | None:
    try:
        return str(json.loads((REAL_PROFILE / "sandbox" / "account.json").read_text())["username"])
    except (OSError, ValueError, KeyError):
        return None


def _account_is_set_up() -> bool:
    name = _real_account_name()
    return bool(name) and subprocess.run(["net", "user", name], capture_output=True).returncode == 0


needs_account = pytest.mark.skipif(
    not _account_is_set_up(), reason="the sandbox account is not set up on this machine"
)


def _as_sandbox(
    *command: str, cwd: Path, variables: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    from client_backend.services.sandbox.launcher import ENV_VARIABLE, encode_environment

    # conftest points CLIENT_PROFILE_ROOT at a temporary folder; the launcher
    # needs the real one, where setup stored the account's credentials.
    env = dict(os.environ, CLIENT_PROFILE_ROOT=str(REAL_PROFILE))
    if variables is not None:
        env[ENV_VARIABLE] = encode_environment(variables)
    return subprocess.run(
        [sys.executable, "-m", "client_backend.services.sandbox.launcher",
         "--cwd", str(cwd), "--", *command],
        capture_output=True, text=True, cwd=REPO, env=env, timeout=60,
    )


@pytest.fixture
def scratch():
    root = Path(os.environ["PROGRAMDATA"]) / f"KaniDesktop-test-{secrets.token_hex(6)}"
    root.mkdir()
    try:
        yield root
    finally:
        # Files the account created belong to it; ask it to remove them first.
        _as_sandbox("cmd", "/c", "rmdir", "/s", "/q", str(root / "clean"), cwd=root)
        shutil.rmtree(root, ignore_errors=True)


@needs_account
def test_a_repo_workspace_is_refused_and_left_ungranted(scratch):
    """A workspace with .git or an environment is never opened to the account:
    granting is inherited, and would hand it folders that run code as the user."""
    from client_backend.services.sandbox.runtime import (
        SandboxRuntimeError,
        ensure_workspace_access,
    )

    repo = scratch / "repo"
    (repo / ".git" / "hooks").mkdir(parents=True)
    (repo / ".venv").mkdir()
    (repo / ".venv" / "pyvenv.cfg").write_text("home = C:/Python\n", encoding="utf-8")

    with pytest.raises(SandboxRuntimeError):
        ensure_workspace_access(repo)

    # We applied no grant of our own -- the escape via .git/.venv is never opened.
    # (ProgramData grants Users inherited write, so a write probe here would say
    # nothing; a real profile workspace has no such inheritance.)
    listing = subprocess.run(["icacls", str(repo)], capture_output=True, text=True).stdout
    assert _real_account_name() not in listing


@needs_account
def test_variables_from_the_environment_reach_the_sandboxed_program(scratch):
    """The launcher takes the program's variables from its environment, not argv."""
    shown = _as_sandbox(
        "cmd", "/c", "set", "KANI_PROBE", cwd=scratch, variables={"KANI_PROBE": "sandbox-probe"}
    )

    assert shown.returncode == 0, shown.stderr
    assert "KANI_PROBE=sandbox-probe" in shown.stdout
    # The payload itself is consumed by the launcher, not handed to the account.
    assert "KANI_SANDBOX_ENV" not in _as_sandbox("cmd", "/c", "set", cwd=scratch).stdout


@needs_account
def test_a_clean_workspace_is_granted_and_usable(scratch):
    from client_backend.services.sandbox.runtime import ensure_workspace_access

    clean = scratch / "clean"
    clean.mkdir()

    ensure_workspace_access(clean, principal=_real_account_name())

    made = clean / "made-by-sandbox"
    _as_sandbox("cmd", "/c", "mkdir", str(made), cwd=clean)
    assert made.is_dir()


@needs_account
def test_account_cannot_read_the_launch_token(scratch, monkeypatch):
    """The token that admits /auth/restore must stay out of the account's reach.

    Written under ProgramData on purpose: every user inherits read access there,
    so only the file's own owner-only ACL can be what keeps the account out. The
    sibling file with inherited permissions shows the probe does read what it can.
    """
    from client_backend.core import launch_token
    from client_backend.core.config import client_settings

    profile = scratch / "profile"
    monkeypatch.setattr(client_settings, "profile_root", str(profile))
    token = launch_token.issue_launch_token()
    try:
        readable = profile / "inherited.txt"
        readable.write_text("inherited-permissions", encoding="utf-8")

        control = _as_sandbox("cmd", "/c", "type", str(readable), cwd=scratch)
        assert control.returncode == 0, control.stderr
        assert "inherited-permissions" in control.stdout

        probe = _as_sandbox("cmd", "/c", "type", str(launch_token.launch_token_path()), cwd=scratch)
        assert probe.returncode != 0
        assert token not in probe.stdout
    finally:
        launch_token.revoke_launch_token()


@needs_account
def test_account_cannot_read_the_real_profile_signing_secret(scratch):
    """The profile root itself is closed to the account: it holds the local-session
    signing secret and stored credentials. Output is never shown: it is a secret."""
    secret = REAL_PROFILE / ".local_session_secret"
    if not secret.is_file():
        pytest.skip("no local session secret in the real profile")

    probe = _as_sandbox("cmd", "/c", "type", str(secret), cwd=scratch)

    assert probe.returncode != 0, "the sandbox account read the local session secret"


@needs_account
def test_account_can_write_only_after_grant_and_cannot_change_runtime(scratch):
    from client_backend.services.sandbox.runtime import ensure_runtime_root, ensure_workspace_access

    identity = _as_sandbox("whoami", cwd=scratch)
    assert identity.returncode == 0
    assert identity.stdout.strip().lower().endswith("\\" + _real_account_name().lower())

    clean = scratch / "clean"
    clean.mkdir()
    user = f"{os.environ['USERDOMAIN']}\\{os.environ['USERNAME']}"
    locked = subprocess.run(
        ["icacls", str(clean), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F",
         "*S-1-5-18:(OI)(CI)F", "*S-1-5-32-544:(OI)(CI)F"],
        capture_output=True, text=True,
    )
    assert locked.returncode == 0, locked.stderr

    denied = clean / "denied"
    assert _as_sandbox("cmd", "/c", "mkdir", str(denied), cwd=scratch).returncode != 0
    assert not denied.exists()

    ensure_workspace_access(clean, principal=_real_account_name())
    allowed = clean / "allowed"
    assert _as_sandbox("cmd", "/c", "mkdir", str(allowed), cwd=clean).returncode == 0
    assert allowed.is_dir()

    runtime = ensure_runtime_root(principal=_real_account_name())
    forbidden = runtime / "sandbox-must-not-write"
    assert _as_sandbox("cmd", "/c", "mkdir", str(forbidden), cwd=clean).returncode != 0
    assert not forbidden.exists()
