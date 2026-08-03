"""Confinement for reading a skill's own companion files.

The path here arrives from a model, so this is an attack surface in the same
class as the command runtime: the whole value of the feature is reading files
inside one bundle, and the whole risk is reading anything else.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from client_backend.services.skill_runtime.resources import (
    MAX_RESOURCE_BYTES,
    SkillResourceError,
    list_skill_resources,
    read_skill_resource,
)


@pytest.fixture()
def bundle(tmp_path) -> Path:
    """A bundle shaped like a current skill: a router plus companion documents."""
    root = tmp_path / "profile" / "installed" / "systematic-debugging"
    (root / "references").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "SKILL.md").write_text("---\nname: demo\n---\nRead references/x.md", encoding="utf-8")
    (root / "root-cause-tracing.md").write_text("Trace to the root.", encoding="utf-8")
    (root / "references" / "x.md").write_text("Details about x.", encoding="utf-8")
    (root / "scripts" / "find-polluter.sh").write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    (root / "install.json").write_text('{"bundle_name": "demo"}', encoding="utf-8")

    outside = tmp_path / "profile" / "secrets.json"
    outside.write_text('{"token": "super-secret"}', encoding="utf-8")
    return root


def test_lists_companion_files_relative_to_the_bundle(bundle):
    listing = list_skill_resources(bundle)

    assert listing.paths == [
        "SKILL.md",
        "references/x.md",
        "root-cause-tracing.md",
        "scripts/find-polluter.sh",
    ]
    assert listing.truncated is False


def test_provenance_metadata_is_not_listed(bundle):
    """install.json is what the sidecar wrote, not authored skill content."""
    assert "install.json" not in list_skill_resources(bundle).paths


def test_listing_is_bounded_and_reports_truncation(tmp_path, monkeypatch):
    # Patched through __globals__, not a fresh import: another test in this
    # directory evicts every client_backend module, after which a re-imported one
    # is a different object and patching it silently does nothing.
    root = tmp_path / "big"
    root.mkdir()
    for index in range(12):
        (root / f"doc-{index:03d}.md").write_text("x", encoding="utf-8")
    monkeypatch.setitem(list_skill_resources.__globals__, "MAX_LISTED_RESOURCES", 5)

    listing = list_skill_resources(root)

    assert len(listing.paths) == 5
    assert listing.truncated is True


def test_reads_a_companion_document(bundle):
    assert read_skill_resource(bundle, "references/x.md") == "Details about x."
    assert read_skill_resource(bundle, "root-cause-tracing.md") == "Trace to the root."


def test_accepts_backslash_separators_from_a_windows_style_path(bundle):
    assert read_skill_resource(bundle, "references\\x.md") == "Details about x."


@pytest.mark.parametrize(
    "hostile",
    [
        "../secrets.json",
        "references/../../secrets.json",
        "./../../secrets.json",
        "references/../../../../../../etc/passwd",
    ],
)
def test_refuses_paths_that_leave_the_bundle(bundle, hostile):
    with pytest.raises(SkillResourceError) as exc_info:
        read_skill_resource(bundle, hostile)

    assert "outside" in exc_info.value.message


@pytest.mark.parametrize("hostile", ["/etc/passwd", "C:/Windows/win.ini"])
def test_refuses_absolute_paths(bundle, hostile):
    with pytest.raises(SkillResourceError) as exc_info:
        read_skill_resource(bundle, hostile)

    assert "relative" in exc_info.value.message


def test_refuses_an_empty_path(bundle):
    with pytest.raises(SkillResourceError):
        read_skill_resource(bundle, "   ")


def test_refuses_a_directory(bundle):
    with pytest.raises(SkillResourceError) as exc_info:
        read_skill_resource(bundle, "references")

    assert "not a file" in exc_info.value.message


def test_refuses_a_missing_file(bundle):
    with pytest.raises(SkillResourceError):
        read_skill_resource(bundle, "references/absent.md")


def test_refuses_a_link_out_of_the_bundle(bundle, tmp_path):
    """A link inside the bundle can still point anywhere outside it."""
    link = bundle / "escape.md"
    try:
        os.symlink(tmp_path / "profile" / "secrets.json", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform")

    with pytest.raises(SkillResourceError) as exc_info:
        read_skill_resource(bundle, "escape.md")

    assert "link" in exc_info.value.message.lower()
    assert "escape.md" not in [path for path in list_skill_resources(bundle).paths]


def test_refuses_a_file_over_the_read_limit(bundle):
    oversized = bundle / "huge.md"
    oversized.write_text("x" * (MAX_RESOURCE_BYTES + 1), encoding="utf-8")

    with pytest.raises(SkillResourceError) as exc_info:
        read_skill_resource(bundle, "huge.md")

    assert "limit" in exc_info.value.message


def test_refuses_binary_content(bundle):
    """Binary assets ship and run; they do not get read into a prompt."""
    (bundle / "model.bin").write_bytes(b"\x00\x01\x02\xff\xfe")

    with pytest.raises(SkillResourceError) as exc_info:
        read_skill_resource(bundle, "model.bin")

    assert "UTF-8" in exc_info.value.message


def test_cache_directories_are_neither_listed_nor_read(bundle):
    cache = bundle / "__pycache__"
    cache.mkdir()
    (cache / "x.cpython-313.pyc").write_text("compiled", encoding="utf-8")

    assert not [path for path in list_skill_resources(bundle).paths if "__pycache__" in path]
    with pytest.raises(SkillResourceError):
        read_skill_resource(bundle, "__pycache__/x.cpython-313.pyc")
