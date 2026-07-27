"""T007 follow-up: a build identity in health diagnostics.

The plan's MCP investigation could not distinguish "the endpoint is wrong" from
"the running process predates the fix" (the 2026-07-24 baseline: *"the reported
all-server response is therefore either a stale process/build, ..."*). Health
must therefore report which build is answering, on BOTH the canonical server and
the sidecar, so a stale process is immediately visible.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.core import build_info

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def test_build_sha_prefers_explicit_deployment_env(monkeypatch):
    """A deploy pipeline stamps BUILD_SHA; that wins over anything local."""
    monkeypatch.setenv("BUILD_SHA", "deadbeefcafe")
    build_info.resolve_build_info.cache_clear()

    info = build_info.resolve_build_info()

    assert info["build_sha"] == "deadbeefcafe"
    assert info["build_source"] == "env"


def test_build_sha_falls_back_to_git_checkout(monkeypatch):
    """In a dev checkout with no stamp, report the real git HEAD commit."""
    monkeypatch.delenv("BUILD_SHA", raising=False)
    build_info.resolve_build_info.cache_clear()

    info = build_info.resolve_build_info()

    assert info["build_source"] in {"git", "unknown"}
    if info["build_source"] != "git":
        pytest.skip("not running from a git checkout")

    assert _SHA_RE.match(info["build_sha"]), info["build_sha"]
    if shutil.which("git") is None:
        pytest.skip("git executable unavailable for cross-check")
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=build_info._REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert info["build_sha"] == expected, (
        "the reported build SHA does not match the checkout's HEAD, so the "
        "diagnostic would point at the wrong source"
    )


def _write_git_dir(root, *, head: str, ref_value: str | None, packed: str | None = None):
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (git_dir / "HEAD").write_text(head, encoding="utf-8")
    if ref_value is not None:
        (git_dir / "refs" / "heads" / "main").write_text(ref_value, encoding="utf-8")
    if packed is not None:
        (git_dir / "packed-refs").write_text(packed, encoding="utf-8")
    return git_dir


@pytest.mark.parametrize(
    "head,ref_value,packed,expected",
    [
        ("ref: refs/heads/main\n", "a" * 40 + "\n", None, "a" * 40),
        ("ref: refs/heads/main\n", None, f"# pack-refs\n{'b' * 40} refs/heads/main\n", "b" * 40),
        ("c" * 40 + "\n", None, None, "c" * 40),
    ],
    ids=["loose-ref", "packed-refs", "detached-head"],
)
def test_git_fallback_is_used_for_each_head_layout(
    monkeypatch, tmp_path, head, ref_value, packed, expected
):
    """Runs unconditionally (no dependency on the test machine having a
    checkout), so removing the git fallback fails rather than skips."""
    monkeypatch.delenv("BUILD_SHA", raising=False)
    _write_git_dir(tmp_path, head=head, ref_value=ref_value, packed=packed)
    monkeypatch.setattr(build_info, "_REPO_ROOT", tmp_path)
    build_info.resolve_build_info.cache_clear()

    assert build_info.resolve_build_info() == {
        "build_sha": expected,
        "build_source": "git",
    }


def test_git_fallback_follows_a_worktree_gitdir_pointer(monkeypatch, tmp_path):
    """A git worktree/submodule stores ``.git`` as a pointer file."""
    monkeypatch.delenv("BUILD_SHA", raising=False)
    real = tmp_path / "real-git"
    (real / "refs" / "heads").mkdir(parents=True)
    (real / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (real / "refs" / "heads" / "main").write_text("d" * 40 + "\n", encoding="utf-8")

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text(f"gitdir: {real}\n", encoding="utf-8")
    monkeypatch.setattr(build_info, "_REPO_ROOT", checkout)
    build_info.resolve_build_info.cache_clear()

    assert build_info.resolve_build_info()["build_sha"] == "d" * 40


def test_build_sha_is_unknown_not_an_error_without_env_or_git(monkeypatch, tmp_path):
    """A packaged install has neither the env var nor a .git directory; that is
    a degraded diagnostic, never a health failure."""
    monkeypatch.delenv("BUILD_SHA", raising=False)
    monkeypatch.setattr(build_info, "_REPO_ROOT", tmp_path)
    build_info.resolve_build_info.cache_clear()

    info = build_info.resolve_build_info()

    assert info == {"build_sha": "unknown", "build_source": "unknown"}


def test_canonical_health_reports_the_build(monkeypatch):
    monkeypatch.setenv("BUILD_SHA", "abc1234")
    build_info.resolve_build_info.cache_clear()

    from app.main import app

    body = TestClient(app).get("/health").json()

    assert body["status"] == "healthy"
    assert body["build_sha"] == "abc1234"
    assert body["build_source"] == "env"


def test_sidecar_health_reports_the_build(monkeypatch):
    monkeypatch.setenv("BUILD_SHA", "abc1234")
    build_info.resolve_build_info.cache_clear()

    from client_backend.main import create_app

    body = TestClient(create_app()).get("/health").json()

    assert body["build_sha"] == "abc1234"
    assert body["build_source"] == "env"


def teardown_module(_module):
    build_info.resolve_build_info.cache_clear()
