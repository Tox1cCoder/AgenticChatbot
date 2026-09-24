"""The managed workspace: where the sandbox account works when none is configured.

It lives under the profile, holds nothing the guard forbids, and is created on
demand. Nothing here touches Windows, so it runs on any platform.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from client_backend.services.sandbox import runtime, workspace


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "KaniDesktop"))
    return tmp_path / "KaniDesktop"


def test_the_managed_workspace_is_a_folder_under_the_profile(profile):
    assert workspace.managed_workspace_root() == profile / "workspace"


def test_ensure_creates_the_workspace_and_returns_it(profile):
    created = workspace.ensure_managed_workspace()

    assert created == profile / "workspace"
    assert created.is_dir()


def test_ensure_is_idempotent_and_keeps_existing_contents(profile):
    first = workspace.ensure_managed_workspace()
    (first / "made-earlier.txt").write_text("keep me", encoding="utf-8")

    again = workspace.ensure_managed_workspace()

    assert again == first
    assert (again / "made-earlier.txt").read_text(encoding="utf-8") == "keep me"


def test_the_managed_workspace_is_never_refused_by_the_guard(profile):
    """The whole point of the managed workspace is that it is grantable: it holds
    no .git and no environment, so protected_paths finds nothing to refuse."""
    created = workspace.ensure_managed_workspace()

    assert runtime.protected_paths(created) == []


def test_it_sits_beside_the_credential_folder_not_inside_it(profile):
    """The DPAPI-encrypted account credentials live in <profile>/sandbox; the
    workspace must be a sibling, never a parent of it, so granting the workspace
    can never reach them."""
    created = workspace.managed_workspace_root()
    credentials = profile / "sandbox"

    assert not credentials.is_relative_to(created)
    assert Path(*created.parts[:-1]) == profile
