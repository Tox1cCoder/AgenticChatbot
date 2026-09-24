"""Folders between the user's profile and a workspace: attributes only, nothing below.

PowerShell and node read the attributes of every folder on the way to a path.
Profile folders are closed to other accounts, so a workspace under Documents
is unusable until the account may read the attributes of Documents (and each
folder below it on the way). Each such folder gets one entry, on that folder
alone: no listing, no contents, nothing inherited, nothing changed underneath.

Everything here runs against a stand-in profile under the test's temporary
folder; the stand-in principal is the built-in Users group.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs only")

USERS_SID = "S-1-5-32-545"
# As Windows writes it back: read attributes + synchronize, no inheritance, for
# "BU" -- the alias Windows uses for the built-in Users group.
ENTRY = "(A;;0x100080;;;BU)"


@pytest.fixture
def machine(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings
    from client_backend.services.sandbox import path_resolution

    home = tmp_path / "home"
    workspace = home / "Documents" / "Code Practice" / "project"
    workspace.mkdir(parents=True)
    (home / "Documents" / "notes.txt").write_text("private", encoding="utf-8")
    monkeypatch.setattr(path_resolution.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path / "profile"))
    return home, workspace


def _dacl(path: Path) -> str:
    from client_backend.services.sandbox.path_resolution import read_dacl

    return read_dacl(path)


def test_each_folder_between_profile_and_workspace_gets_read_attributes_only(machine):
    from client_backend.services.sandbox.path_resolution import grant_path_resolution

    home, workspace = machine

    grant_path_resolution(workspace, principal_sid=USERS_SID)

    for folder in (home / "Documents", home / "Documents" / "Code Practice"):
        assert ENTRY in _dacl(folder), folder


def test_the_profile_root_and_the_workspace_itself_are_left_alone(machine):
    from client_backend.services.sandbox.path_resolution import grant_path_resolution

    home, workspace = machine
    before = {home: _dacl(home), workspace: _dacl(workspace)}

    grant_path_resolution(workspace, principal_sid=USERS_SID)

    assert {home: _dacl(home), workspace: _dacl(workspace)} == before


def test_nothing_below_a_granted_folder_changes(machine):
    from client_backend.services.sandbox.path_resolution import grant_path_resolution

    home, workspace = machine
    private = home / "Documents" / "notes.txt"
    before = _dacl(private)

    grant_path_resolution(workspace, principal_sid=USERS_SID)

    assert _dacl(private) == before


def test_granting_twice_adds_one_entry(machine):
    from client_backend.services.sandbox.path_resolution import grant_path_resolution

    home, workspace = machine

    grant_path_resolution(workspace, principal_sid=USERS_SID)
    grant_path_resolution(workspace, principal_sid=USERS_SID)

    assert _dacl(home / "Documents").count(ENTRY) == 1


def test_explicit_entries_stay_ahead_of_inherited_ones(machine):
    """Windows requires explicit entries before inherited ones; out of order,
    access checks misbehave and Explorer offers to 'fix' the folder."""
    import re

    from client_backend.services.sandbox.path_resolution import (
        grant_path_resolution,
        write_dacl,
    )

    home, workspace = machine
    # An explicit deny, then entries inherited from above, as real folders have.
    write_dacl(
        home / "Documents",
        "D:AI(D;;DC;;;WD)(A;;FA;;;OW)(A;OICIID;FA;;;SY)(A;OICIID;FA;;;BA)",
    )

    grant_path_resolution(workspace, principal_sid=USERS_SID)

    entries = re.findall(r"\(([^()]*)\)", _dacl(home / "Documents"))
    assert entries[0].startswith("D;"), entries
    assert f"({entries[1]})" == ENTRY, entries
    inherited = ["ID" in entry.split(";")[1] for entry in entries]
    assert inherited == sorted(inherited), entries


def test_revoking_restores_every_folder_exactly(machine):
    from client_backend.services.sandbox.path_resolution import (
        grant_path_resolution,
        revoke_path_resolution,
    )

    home, workspace = machine
    folders = (home / "Documents", home / "Documents" / "Code Practice")
    before = {folder: _dacl(folder) for folder in folders}

    grant_path_resolution(workspace, principal_sid=USERS_SID)
    revoke_path_resolution(principal_sid=USERS_SID)

    assert {folder: _dacl(folder) for folder in folders} == before


def test_grant_and_revoke_keep_an_auto_inherited_folder_auto_inherited(machine):
    """Real profile folders are marked auto-inherited ("D:AI"). The legacy call
    that changes one folder drops that mark unless asked to keep it."""
    from client_backend.services.sandbox.path_resolution import (
        grant_path_resolution,
        revoke_path_resolution,
        write_dacl,
    )

    home, workspace = machine
    write_dacl(home / "Documents", "D:AI(A;;FA;;;OW)(A;OICIID;FA;;;SY)")
    before = _dacl(home / "Documents")
    assert before.startswith("D:AI")

    grant_path_resolution(workspace, principal_sid=USERS_SID)
    assert _dacl(home / "Documents").startswith("D:AI")
    revoke_path_resolution()

    assert _dacl(home / "Documents") == before


def test_revoking_works_after_the_account_is_gone(machine):
    """``sandbox remove`` may run after the account was deleted by hand, when its
    name no longer resolves; the recorded SID still finds the entries."""
    from client_backend.services.sandbox.path_resolution import (
        grant_path_resolution,
        revoke_path_resolution,
    )

    home, workspace = machine
    before = _dacl(home / "Documents")
    grant_path_resolution(workspace, principal_sid=USERS_SID)

    revoke_path_resolution()

    assert _dacl(home / "Documents") == before


def test_revoking_with_nothing_granted_changes_nothing(machine):
    from client_backend.services.sandbox.path_resolution import revoke_path_resolution

    home, _ = machine
    before = _dacl(home / "Documents")

    revoke_path_resolution()

    assert _dacl(home / "Documents") == before


def test_workspace_outside_the_profile_needs_no_entries(machine, tmp_path):
    from client_backend.services.sandbox.path_resolution import grant_path_resolution

    outside = tmp_path / "Workspaces" / "project"
    outside.mkdir(parents=True)
    before = _dacl(outside.parent)

    grant_path_resolution(outside, principal_sid=USERS_SID)

    assert _dacl(outside.parent) == before
