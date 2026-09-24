"""The sandbox account on a real machine: what it can and cannot change.

Skips unless ``python -m client_backend sandbox setup`` has been run here.
Works in a scratch folder under ProgramData so the user's profile is never
touched, and runs commands through the real launcher as the account.
"""

from __future__ import annotations

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


def _account_is_set_up() -> bool:
    exists = subprocess.run(["net", "user", "KaniSandbox"], capture_output=True).returncode == 0
    return exists and (REAL_PROFILE / "sandbox" / "account.json").is_file()


needs_account = pytest.mark.skipif(
    not _account_is_set_up(), reason="the sandbox account is not set up on this machine"
)


def _as_sandbox(*command: str, cwd: Path) -> subprocess.CompletedProcess:
    # conftest points CLIENT_PROFILE_ROOT at a temporary folder; the launcher
    # needs the real one, where setup stored the account's credentials.
    env = dict(os.environ, CLIENT_PROFILE_ROOT=str(REAL_PROFILE))
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
    assert "KaniSandbox" not in listing


@needs_account
def test_a_clean_workspace_is_granted_and_usable(scratch):
    from client_backend.services.sandbox.runtime import ensure_workspace_access

    clean = scratch / "clean"
    clean.mkdir()

    ensure_workspace_access(clean)

    made = clean / "made-by-sandbox"
    _as_sandbox("cmd", "/c", "mkdir", str(made), cwd=clean)
    assert made.is_dir()
