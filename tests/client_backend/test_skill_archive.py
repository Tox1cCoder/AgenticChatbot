"""Hostile-archive coverage for the skill ZIP validator.

This validator is the only thing standing between an arbitrary uploaded file and
the profile directory, so the cases here are adversarial by design: path escapes,
Windows and UNC absolute names, reserved device names, case- and Unicode-folded
collisions, entry types that are not regular files, declared-size lies, and
truncated or corrupted containers. Every rejection must happen before any byte
lands in the requested destination.
"""

from __future__ import annotations

import stat
import struct
import zipfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from client_backend.services.skill_runtime.archive import (
    SkillArchiveError,
    SkillArchiveLimits,
    SkillArchiveValidator,
)

VALID_SKILL_MD = "---\nname: demo\ndescription: Demo\n---\nBody"


def _limits(**overrides) -> SkillArchiveLimits:
    base = SkillArchiveLimits(
        max_compressed_bytes=25 * 1024 * 1024,
        max_expanded_bytes=100 * 1024 * 1024,
        max_file_bytes=50 * 1024 * 1024,
        max_entries=2000,
        max_compression_ratio=200,
        max_path_depth=20,
        max_path_chars=240,
    )
    return replace(base, **overrides) if overrides else base


def _validator(limits: SkillArchiveLimits | None = None) -> SkillArchiveValidator:
    return SkillArchiveValidator(limits or _limits())


def _zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def _zip_entries(path: Path, members: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return path


def _zip_with_info(path: Path, info: zipfile.ZipInfo, payload: bytes = b"data") -> Path:
    """Write one member from a caller-built ZipInfo, preserving its Unix mode."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info, payload)
    return path


def _patch_header_field(path: Path, *, local_offset: int, central_offset: int, value: int) -> Path:
    """Set a 16-bit header field in every local and central record.

    ``zipfile.writestr`` recomputes the general-purpose flags and the compression
    method from its own state, so a hostile value has to be written into the
    container bytes directly. Both copies of the field must agree or the reader
    rejects the file for the wrong reason.
    """
    raw = bytearray(path.read_bytes())
    for signature, offset in ((b"PK\x03\x04", local_offset), (b"PK\x01\x02", central_offset)):
        index = raw.find(signature)
        while index != -1:
            struct.pack_into("<H", raw, index + offset, value)
            index = raw.find(signature, index + 4)
    path.write_bytes(bytes(raw))
    return path


def test_extracts_valid_skill_zip_member_by_member(tmp_path):
    archive = _zip(
        tmp_path / "skill.zip",
        {
            "demo/SKILL.md": VALID_SKILL_MD,
            "demo/bin/demo.py": "print('ok')\n",
        },
    )
    destination = tmp_path / "out"

    summary = _validator().extract(archive, destination)

    assert summary.file_count == 2
    assert (destination / "demo" / "SKILL.md").is_file()
    assert (destination / "demo" / "bin" / "demo.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert summary.expanded_bytes == len(VALID_SKILL_MD.encode()) + len("print('ok')\n")
    assert summary.compressed_bytes == archive.stat().st_size


def test_directory_entries_are_created_without_counting_as_files(tmp_path):
    path = tmp_path / "skill.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("demo/", b"")
        archive.writestr("demo/empty/", b"")
        archive.writestr("demo/SKILL.md", VALID_SKILL_MD)
    destination = tmp_path / "out"

    summary = _validator().extract(path, destination)

    assert summary.file_count == 1
    assert (destination / "demo" / "empty").is_dir()


@pytest.mark.parametrize(
    ("limit_name", "value", "code"),
    [
        ("max_entries", 1, "SKILL_ARCHIVE_TOO_MANY_FILES"),
        ("max_expanded_bytes", 4, "SKILL_ARCHIVE_TOO_LARGE"),
        ("max_file_bytes", 4, "SKILL_ARCHIVE_TOO_LARGE"),
        ("max_compressed_bytes", 4, "SKILL_ARCHIVE_TOO_LARGE"),
    ],
)
def test_rejects_resource_limits_before_leaving_output(tmp_path, limit_name, value, code):
    archive = _zip_entries(
        tmp_path / "skill.zip",
        [("SKILL.md", b"12345"), ("extra.txt", b"12345")],
    )
    limits = _limits(**{limit_name: value})

    with pytest.raises(SkillArchiveError) as exc_info:
        SkillArchiveValidator(limits).extract(archive, tmp_path / "out")

    assert exc_info.value.code == code
    assert not (tmp_path / "out").exists()


def test_rejects_declared_compression_ratio_bomb(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("SKILL.md", b"A" * (1024 * 1024))

    with pytest.raises(SkillArchiveError) as exc_info:
        SkillArchiveValidator(_limits(max_compression_ratio=200)).extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_TOO_LARGE"
    assert not (tmp_path / "out").exists()


UNSAFE_MEMBER_NAMES = [
    "../escape",
    "nested/../../escape",
    "/absolute/SKILL.md",
    r"C:\escape\SKILL.md",
    r"\\server\share\SKILL.md",
    r"nested\..\escape",
    "NUL.txt",
    "demo/trailing. /SKILL.md",
]


@pytest.mark.parametrize("member", UNSAFE_MEMBER_NAMES)
def test_rejects_unsafe_member_names(tmp_path, member):
    archive = _zip_entries(tmp_path / "bad.zip", [(member, b"payload")])

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_PATH_UNSAFE"
    assert not (tmp_path / "out").exists()


# Writing two entries under one name is the point of the test; zipfile's warning
# about it is expected input, not a defect to surface.
@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
@pytest.mark.parametrize(
    ("members", "code"),
    [
        ([("a.txt", b"1"), ("a.txt", b"2")], "SKILL_ARCHIVE_PATH_UNSAFE"),
        ([("A.txt", b"1"), ("a.txt", b"2")], "SKILL_ARCHIVE_PATH_UNSAFE"),
        ([("e\u0301.txt", b"1"), ("\u00e9.txt", b"2")], "SKILL_ARCHIVE_PATH_UNSAFE"),
    ],
)
def test_rejects_duplicate_portable_paths(tmp_path, members, code):
    archive = _zip_entries(tmp_path / "bad.zip", members)

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == code


def test_rejects_path_depth_over_limit(tmp_path):
    deep = "/".join(f"level{index}" for index in range(21)) + "/SKILL.md"
    archive = _zip_entries(tmp_path / "deep.zip", [(deep, b"payload")])

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_PATH_UNSAFE"


def test_rejects_path_longer_than_portable_limit(tmp_path):
    long_name = "demo/" + ("n" * 250) + ".txt"
    archive = _zip_entries(tmp_path / "long.zip", [(long_name, b"payload")])

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_PATH_UNSAFE"


def test_rejects_encrypted_member(tmp_path):
    archive = _zip(tmp_path / "encrypted.zip", {"SKILL.md": VALID_SKILL_MD})
    _patch_header_field(archive, local_offset=6, central_offset=8, value=0x1)

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"
    assert not (tmp_path / "out").exists()


def test_rejects_unsupported_compression_method(tmp_path):
    """Method 99 is the WinZip AES marker; refuse it instead of failing mid-copy."""
    archive = _zip(tmp_path / "unsupported.zip", {"SKILL.md": VALID_SKILL_MD})
    _patch_header_field(archive, local_offset=8, central_offset=10, value=99)

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"
    assert not (tmp_path / "out").exists()


def test_rejects_corrupted_member_crc(tmp_path):
    path = tmp_path / "crc.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("SKILL.md", b"AAAAAAAAAAAAAAAA")
    raw = bytearray(path.read_bytes())
    index = raw.find(b"AAAAAAAAAAAAAAAA")
    raw[index] = ord("B")
    path.write_bytes(bytes(raw))

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(path, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"
    assert not (tmp_path / "out").exists()


def test_rejects_truncated_central_directory(tmp_path):
    path = tmp_path / "truncated.zip"
    _zip(path, {"SKILL.md": VALID_SKILL_MD})
    raw = path.read_bytes()
    path.write_bytes(raw[:-16])

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(path, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"


def test_rejects_non_zip_payload(tmp_path):
    path = tmp_path / "not-a-zip.zip"
    path.write_bytes(b"this is plain text, not a container")

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(path, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"


def test_rejects_empty_archive(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w"):
        pass

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(path, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"


def test_understated_declared_size_cannot_produce_an_oversized_write(tmp_path, monkeypatch):
    """A header that understates a member's size still cannot exceed the limits.

    Preflight can only read declared sizes. ``zipfile`` bounds each member read by
    that declaration and then fails the CRC check, so an understated size ends as
    a rejected corrupt archive rather than an over-limit write. The streaming
    copy's own byte counters remain as defense in depth for the day that changes.
    """
    path = tmp_path / "liar.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SKILL.md", b"B" * 4096)

    real_infolist = zipfile.ZipFile.infolist

    def understating_infolist(self):
        infos = real_infolist(self)
        for info in infos:
            info.file_size = 8
        return infos

    monkeypatch.setattr(zipfile.ZipFile, "infolist", understating_infolist)

    with pytest.raises(SkillArchiveError) as exc_info:
        SkillArchiveValidator(_limits(max_file_bytes=64, max_expanded_bytes=64)).extract(
            path, tmp_path / "out"
        )

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("limits", "already_expanded"),
    [
        (_limits(max_file_bytes=64), 0),
        (_limits(max_expanded_bytes=4096), 4000),
    ],
)
def test_streaming_counters_stop_an_oversized_member(tmp_path, limits, already_expanded):
    """The copy loop's own counters bound a member, not just its declaration.

    Exercised at the copy seam rather than through ``extract`` because the reader
    already refuses to hand over more bytes than a member declares, which would
    mask this guard. It exists so a member that gets past preflight -- by a future
    reader change or a size the caller could not know in advance -- still cannot
    write past the per-file or total limit.
    """
    path = tmp_path / "big.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("SKILL.md", b"B" * 4096)
    validator = SkillArchiveValidator(limits)
    target = tmp_path / "member.out"

    with zipfile.ZipFile(path) as handle:
        info = handle.infolist()[0]
        member = _planned_member(info)
        with pytest.raises(SkillArchiveError) as exc_info:
            validator._copy_member(handle, member, target, already_expanded)

    assert exc_info.value.code == "SKILL_ARCHIVE_TOO_LARGE"
    assert exc_info.value.status_code == 413


def _planned_member(info: zipfile.ZipInfo):
    from client_backend.services.skill_runtime.archive import _PlannedMember

    return _PlannedMember(
        info=info,
        relative_path=PurePosixPath(info.filename),
        is_directory=False,
    )


def test_existing_destination_is_never_partially_overwritten(tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("original", encoding="utf-8")
    archive = _zip_entries(tmp_path / "bad.zip", [("../escape", b"payload")])

    with pytest.raises(SkillArchiveError):
        _validator().extract(archive, destination)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "original"


def test_limits_from_settings_reads_configured_bounds(monkeypatch):
    # The proxy this module resolves against, which module eviction elsewhere in
    # the suite can make distinct from a freshly imported one.
    client_settings = SkillArchiveValidator.__init__.__globals__["client_settings"]

    monkeypatch.setattr(client_settings, "skill_upload_max_bytes", 111)
    monkeypatch.setattr(client_settings, "skill_upload_max_expanded_bytes", 222)
    monkeypatch.setattr(client_settings, "skill_upload_max_file_bytes", 333)
    monkeypatch.setattr(client_settings, "skill_upload_max_entries", 444)

    limits = SkillArchiveLimits.from_settings()

    assert limits.max_compressed_bytes == 111
    assert limits.max_expanded_bytes == 222
    assert limits.max_file_bytes == 333
    assert limits.max_entries == 444


def test_temporary_extraction_root_is_removed_on_success_and_failure(tmp_path):
    good = _zip(tmp_path / "good.zip", {"SKILL.md": VALID_SKILL_MD})
    _validator().extract(good, tmp_path / "good-out")
    bad = _zip_entries(tmp_path / "bad.zip", [("../escape", b"x")])
    with pytest.raises(SkillArchiveError):
        _validator().extract(bad, tmp_path / "bad-out")

    leftovers = [path.name for path in tmp_path.iterdir() if ".extract-" in path.name]

    assert leftovers == []


def _zip_with_symlink(path: Path, *, link_name: str, target: str, members: dict[str, str]) -> Path:
    """Build an archive carrying a real Unix symlink entry.

    This is how a source download of a repository that uses symlinks arrives: a
    member whose external_attr records S_IFLNK and whose content is the target
    path. It reproduces the exact shape that made a GitHub archive unusable.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        info = zipfile.ZipInfo(link_name)
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, target)
    return path


def test_symlink_entries_are_skipped_not_rejected(tmp_path):
    """A source archive with an alias link must still install.

    Refusing the whole upload over one unrelated top-level link -- an AGENTS.md
    aliasing CLAUDE.md, for instance -- blocks legitimate archives for no safety
    gain, because the link is never materialized either way.
    """
    archive = _zip_with_symlink(
        tmp_path / "repo.zip",
        link_name="repo/AGENTS.md",
        target="CLAUDE.md",
        members={"repo/SKILL.md": VALID_SKILL_MD, "repo/CLAUDE.md": "guidance"},
    )
    destination = tmp_path / "out"

    summary = _validator().extract(archive, destination)

    assert summary.skipped_link_count == 1
    assert summary.file_count == 2
    assert (destination / "repo" / "SKILL.md").is_file()
    assert (destination / "repo" / "CLAUDE.md").is_file()
    # The link must not exist in any form -- neither as a link nor as a plain
    # file whose contents are the target path, which is what zipfile would write
    # on Windows.
    assert not (destination / "repo" / "AGENTS.md").exists()


def test_a_skipped_link_never_becomes_a_file_containing_its_target(tmp_path):
    archive = _zip_with_symlink(
        tmp_path / "escape.zip",
        link_name="repo/passwd",
        target="/etc/passwd",
        members={"repo/SKILL.md": VALID_SKILL_MD},
    )
    destination = tmp_path / "out"

    _validator().extract(archive, destination)

    assert not (destination / "repo" / "passwd").exists()
    assert [path.name for path in (destination / "repo").iterdir()] == ["SKILL.md"]


def test_an_archive_of_only_links_has_nothing_to_install(tmp_path):
    archive = _zip_with_symlink(
        tmp_path / "links.zip",
        link_name="repo/AGENTS.md",
        target="CLAUDE.md",
        members={},
    )

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_INVALID"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("mode", "label"),
    [
        (0o010644, "fifo"),
        (0o020644, "character device"),
        (0o060644, "block device"),
        (0o140644, "socket"),
    ],
)
def test_other_non_regular_entry_types_are_still_rejected(tmp_path, mode, label):
    """Unlike a symlink, these have no benign reading inside a skill bundle."""
    info = zipfile.ZipInfo("demo/entry")
    info.create_system = 3
    info.external_attr = mode << 16
    archive = _zip_with_info(tmp_path / "hostile.zip", info, b"payload")

    with pytest.raises(SkillArchiveError) as exc_info:
        _validator().extract(archive, tmp_path / "out")

    assert exc_info.value.code == "SKILL_ARCHIVE_PATH_UNSAFE", label
    assert "device, socket, or pipe" in exc_info.value.message


def test_the_real_superpowers_archive_shape_extracts(tmp_path):
    """Regression for the reported failure, reproduced structurally.

    The uploaded archive was a repository download whose only unusual entry was
    one symlink; every other member was an ordinary file.
    """
    archive = tmp_path / "superpowers-main.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("superpowers-main/CLAUDE.md", "root guidance")
        handle.writestr("superpowers-main/skills/brainstorming/SKILL.md", VALID_SKILL_MD)
        info = zipfile.ZipInfo("superpowers-main/AGENTS.md")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(info, "CLAUDE.md")

    summary = _validator().extract(archive, tmp_path / "out")

    assert summary.skipped_link_count == 1
    assert summary.file_count == 2
