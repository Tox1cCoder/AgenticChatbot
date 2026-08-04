"""Discovering and installing a library of skills from one archive.

Skill libraries are distributed as one repository holding many skills, so an
archive containing several is the normal case, not an error. These tests pin
where a skill's root is (which decides whether its commands are found), how the
collection identifies itself, and that installing a set is all-or-nothing.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from client_backend.services.skill_runtime.collection import (
    CollectionManifest,
    discover_collection,
    find_skill_roots,
)

SKILL_MD = "---\nname: {name}\ndescription: {name} skill\n---\nBody"


def _write_skill(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(SKILL_MD.format(name=name), encoding="utf-8")
    return root


def test_finds_every_skill_folder_in_a_library(tmp_path):
    _write_skill(tmp_path / "skills" / "brainstorming", "brainstorming")
    _write_skill(tmp_path / "skills" / "systematic-debugging", "systematic-debugging")
    _write_skill(tmp_path / "skills" / "test-driven-development", "test-driven-development")

    roots = find_skill_roots(tmp_path)

    assert [root.name for root in roots] == [
        "brainstorming",
        "systematic-debugging",
        "test-driven-development",
    ]


def test_a_skill_root_is_the_folder_holding_skill_md_not_an_ancestor(tmp_path):
    """Commands resolve relative to that folder, so an ancestor would lose them."""
    _write_skill(tmp_path / "skills" / "demo", "demo")
    _write_skill(tmp_path / "skills" / "other", "other")

    roots = find_skill_roots(tmp_path)

    assert tmp_path not in roots
    assert (tmp_path / "skills") not in roots


def test_ignores_vendored_and_build_directories(tmp_path):
    _write_skill(tmp_path / "skills" / "real", "real")
    _write_skill(tmp_path / "node_modules" / "pkg" / "fixture", "fixture")
    _write_skill(tmp_path / ".git" / "worktree" / "copy", "copy")

    roots = find_skill_roots(tmp_path)

    assert [root.name for root in roots] == ["real"]


def test_a_single_skill_keeps_the_archive_root_as_its_bundle(tmp_path):
    """The documented nested shape puts assets at the top, beside `skills/`.

    Narrowing to the inner folder would silently drop the skill's commands, so a
    lone skill still installs from the archive root.
    """
    _write_skill(tmp_path / "skills" / "demo", "demo")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "demo.py").write_text("print('ok')\n", encoding="utf-8")

    collection = discover_collection(tmp_path, fallback_name="demo")

    assert collection.is_single_skill
    assert hasattr(collection, "skills"), "discovery must retain the exact SKILL.md"
    discovered = collection.skills[0]
    assert discovered.bundle_root == tmp_path
    assert discovered.skill_file == tmp_path / "skills" / "demo" / "SKILL.md"


def test_a_collection_gives_each_skill_its_own_root(tmp_path):
    _write_skill(tmp_path / "skills" / "one", "one")
    _write_skill(tmp_path / "skills" / "two", "two")

    collection = discover_collection(tmp_path, fallback_name="library")

    assert not collection.is_single_skill
    assert [skill.bundle_root.name for skill in collection.skills] == ["one", "two"]


@pytest.mark.parametrize(
    "manifest_path",
    [
        ".claude-plugin/plugin.json",
        ".codex-plugin/plugin.json",
        ".cursor-plugin/plugin.json",
        "plugin.json",
    ],
)
def test_reads_identity_from_any_supported_plugin_manifest(tmp_path, manifest_path):
    """One repository ships manifests for several harnesses; any of them names it."""
    _write_skill(tmp_path / "skills" / "one", "one")
    _write_skill(tmp_path / "skills" / "two", "two")
    manifest = tmp_path / manifest_path
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"name": "superpowers", "version": "6.2.0", "description": "Core skills"}),
        encoding="utf-8",
    )

    collection = discover_collection(tmp_path, fallback_name="ignored")

    assert collection.manifest == CollectionManifest(
        name="superpowers",
        version="6.2.0",
        description="Core skills",
    )


def test_falls_back_to_the_uploaded_name_without_a_manifest(tmp_path):
    _write_skill(tmp_path / "skills" / "one", "one")
    _write_skill(tmp_path / "skills" / "two", "two")

    collection = discover_collection(tmp_path, fallback_name="my-library")

    assert collection.manifest.name == "my-library"
    assert collection.manifest.version is None


def test_an_unreadable_manifest_does_not_break_discovery(tmp_path):
    _write_skill(tmp_path / "skills" / "one", "one")
    _write_skill(tmp_path / "skills" / "two", "two")
    manifest = tmp_path / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{not json", encoding="utf-8")

    collection = discover_collection(tmp_path, fallback_name="fallback")

    assert collection.manifest.name == "fallback"
    assert len(collection.skills) == 2


def test_an_archive_with_no_skills_discovers_none(tmp_path):
    (tmp_path / "README.md").write_text("no skills here", encoding="utf-8")

    collection = discover_collection(tmp_path, fallback_name="empty")

    assert collection.skills == []


def test_the_real_superpowers_archive_discovers_its_library(tmp_path):
    """The archive that could not be installed at all, read as a collection."""
    source = Path(__file__).resolve().parents[2] / "superpowers-main.zip"
    if not source.is_file():
        pytest.skip("the reference archive is not present in this checkout")

    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            if member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000:
                continue
            target = extracted / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))

    collection = discover_collection(extracted / "superpowers-main", fallback_name="superpowers")

    assert collection.manifest.name == "superpowers"
    assert collection.manifest.version
    assert len(collection.skills) == 14
    assert "brainstorming" in {skill.bundle_root.name for skill in collection.skills}
