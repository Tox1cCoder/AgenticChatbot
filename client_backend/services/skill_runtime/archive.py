"""Validation and confined extraction of an uploaded skill ZIP.

This module is the trust boundary for browser-supplied archives. Everything
downstream (the staged upload store, the installer, the runtime preparer) assumes
it is handed a plain directory tree of regular files inside the profile, so every
way an archive can violate that has to be refused here:

* **path escapes** -- ``..`` segments, absolute POSIX/Windows paths, UNC names,
  and drive letters, which would write outside the destination;
* **name collisions the filesystem resolves but the archive does not** --
  ``A.txt``/``a.txt`` on case-insensitive volumes and NFD/NFC pairs on macOS,
  where the second member would silently overwrite the first;
* **reserved Windows names** (``NUL``, ``COM1``, trailing dots/spaces), which do
  not behave like files;
* **entry types that are not regular files** -- symlinks, FIFOs, sockets, and
  device nodes, which turn a later read or copy into an escape; and
* **resource exhaustion** -- declared or actual sizes, entry counts, and
  compression ratios that would fill the disk.

Two rules shape the implementation. First, nothing is trusted that the archive
merely *declares*: sizes are re-counted while copying, because a header can lie.
Second, extraction is atomic from the caller's point of view -- members are
written into a temporary sibling and promoted with ``os.replace`` only after every
one succeeds, so a rejected archive never leaves a partial tree behind.
"""

from __future__ import annotations

import os
import shutil
import stat
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import is_under_root

logger = get_logger(__name__)

_COPY_CHUNK_BYTES = 1024 * 1024

# Names Windows resolves to devices rather than files, with or without an
# extension. Checked per component, case-insensitively.
_RESERVED_WINDOWS_STEMS = frozenset(
    {"con", "prn", "aux", "nul", "clock$"}
    | {f"com{digit}" for digit in "123456789"}
    | {f"lpt{digit}" for digit in "123456789"}
)

_SUPPORTED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})

_ENCRYPTED_FLAG = 0x1

SKILL_ARCHIVE_INVALID = "SKILL_ARCHIVE_INVALID"
SKILL_ARCHIVE_PATH_UNSAFE = "SKILL_ARCHIVE_PATH_UNSAFE"
SKILL_ARCHIVE_TOO_LARGE = "SKILL_ARCHIVE_TOO_LARGE"
SKILL_ARCHIVE_TOO_MANY_FILES = "SKILL_ARCHIVE_TOO_MANY_FILES"
SKILL_ARCHIVE_TYPE_UNSUPPORTED = "SKILL_ARCHIVE_TYPE_UNSUPPORTED"


class SkillArchiveError(Exception):
    """A normalized, client-safe archive rejection.

    The message is written for a person choosing another file; it never quotes a
    member name, a staging path, or archive contents.
    """

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class SkillArchiveLimits:
    """Immutable resource bounds for one archive validation."""

    max_compressed_bytes: int
    max_expanded_bytes: int
    max_file_bytes: int
    max_entries: int
    max_compression_ratio: int
    max_path_depth: int
    max_path_chars: int

    @classmethod
    def from_settings(cls) -> SkillArchiveLimits:
        """Snapshot the configured bounds so one validation cannot see them change."""
        return cls(
            max_compressed_bytes=int(client_settings.skill_upload_max_bytes),
            max_expanded_bytes=int(client_settings.skill_upload_max_expanded_bytes),
            max_file_bytes=int(client_settings.skill_upload_max_file_bytes),
            max_entries=int(client_settings.skill_upload_max_entries),
            max_compression_ratio=int(client_settings.skill_upload_max_compression_ratio),
            max_path_depth=int(client_settings.skill_upload_max_path_depth),
            max_path_chars=int(client_settings.skill_upload_max_path_chars),
        )


@dataclass(frozen=True)
class SkillArchiveSummary:
    """Non-sensitive measurements of one validated archive."""

    compressed_bytes: int
    expanded_bytes: int
    file_count: int


@dataclass(frozen=True)
class _PlannedMember:
    """One archive member that passed preflight, with its portable path."""

    info: zipfile.ZipInfo
    relative_path: PurePosixPath
    is_directory: bool


class SkillArchiveValidator:
    """Validate a ZIP completely, then extract it into a confined destination."""

    def __init__(self, limits: SkillArchiveLimits | None = None) -> None:
        self._limits = limits or SkillArchiveLimits.from_settings()

    def extract(self, archive_path: Path, destination: Path) -> SkillArchiveSummary:
        """Validate ``archive_path`` and materialize it at ``destination``.

        Args:
            archive_path: The uploaded archive on disk.
            destination: Directory to create. Must not exist yet as a file.

        Returns:
            Measurements of the extracted tree.

        Raises:
            SkillArchiveError: Any structural, path, entry-type, or size violation.
        """
        compressed_bytes = self._require_container_within_size(archive_path)

        # Extension and MIME type are attacker-controlled labels, never evidence.
        # A real central directory that zipfile can parse is the only proof.
        if not zipfile.is_zipfile(archive_path):
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The uploaded file is not a readable ZIP archive.",
            )

        try:
            with zipfile.ZipFile(archive_path) as archive:
                members, expanded_bytes = self._plan_members(archive, compressed_bytes)
                return self._extract_planned(
                    archive,
                    members,
                    destination,
                    compressed_bytes=compressed_bytes,
                    declared_expanded_bytes=expanded_bytes,
                )
        except zipfile.BadZipFile as exc:
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The ZIP archive is malformed or incomplete.",
            ) from exc

    def _require_container_within_size(self, archive_path: Path) -> int:
        try:
            compressed_bytes = archive_path.stat().st_size
        except OSError as exc:
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The uploaded archive could not be read.",
            ) from exc
        if compressed_bytes > self._limits.max_compressed_bytes:
            raise SkillArchiveError(
                SKILL_ARCHIVE_TOO_LARGE,
                "The archive is larger than the allowed upload size.",
                status_code=413,
            )
        return compressed_bytes

    def _plan_members(
        self,
        archive: zipfile.ZipFile,
        compressed_bytes: int,
    ) -> tuple[list[_PlannedMember], int]:
        """Check every entry before creating anything on disk."""
        infos = archive.infolist()
        planned: list[_PlannedMember] = []
        collision_keys: dict[str, str] = {}
        declared_expanded = 0
        file_count = 0

        for info in infos:
            self._require_supported_entry(info)
            relative_path, is_directory = self._safe_relative_path(info)

            key = self._collision_key(relative_path)
            if key in collision_keys:
                raise SkillArchiveError(
                    SKILL_ARCHIVE_PATH_UNSAFE,
                    "The archive contains two entries whose paths collide on this "
                    "filesystem. Repackage it with distinct names.",
                )
            collision_keys[key] = str(relative_path)

            if is_directory:
                planned.append(_PlannedMember(info, relative_path, True))
                continue

            file_count += 1
            if file_count > self._limits.max_entries:
                raise SkillArchiveError(
                    SKILL_ARCHIVE_TOO_MANY_FILES,
                    "The archive contains more files than allowed.",
                    status_code=413,
                )
            if info.file_size > self._limits.max_file_bytes:
                raise SkillArchiveError(
                    SKILL_ARCHIVE_TOO_LARGE,
                    "A file inside the archive is larger than allowed.",
                    status_code=413,
                )
            declared_expanded += info.file_size
            if declared_expanded > self._limits.max_expanded_bytes:
                raise SkillArchiveError(
                    SKILL_ARCHIVE_TOO_LARGE,
                    "The archive expands to more data than allowed.",
                    status_code=413,
                )
            planned.append(_PlannedMember(info, relative_path, False))

        if file_count == 0:
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The archive contains no files.",
            )
        self._require_sane_ratio(declared_expanded, compressed_bytes)
        return planned, declared_expanded

    def _require_sane_ratio(self, expanded_bytes: int, compressed_bytes: int) -> None:
        if compressed_bytes <= 0:
            return
        if expanded_bytes // compressed_bytes > self._limits.max_compression_ratio:
            raise SkillArchiveError(
                SKILL_ARCHIVE_TOO_LARGE,
                "The archive expands far beyond its compressed size.",
                status_code=413,
            )

    def _require_supported_entry(self, info: zipfile.ZipInfo) -> None:
        if info.flag_bits & _ENCRYPTED_FLAG:
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "Encrypted archives are not supported. Upload an unencrypted ZIP.",
            )
        if info.compress_type not in _SUPPORTED_COMPRESSION:
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The archive uses an unsupported compression method.",
            )
        self._require_regular_file_mode(info)

    @staticmethod
    def _require_regular_file_mode(info: zipfile.ZipInfo) -> None:
        """Reject anything a Unix-created archive marks as a non-regular file.

        ``external_attr``'s high 16 bits carry st_mode when ``create_system`` is
        Unix. A symlink member is the classic escape: extracting it as a link and
        then copying "through" it reaches any absolute path the attacker chose.
        """
        if info.create_system != 3:
            return
        mode = info.external_attr >> 16
        if not mode:
            return
        if info.is_dir() and stat.S_ISDIR(mode):
            return
        if stat.S_ISREG(mode):
            return
        raise SkillArchiveError(
            SKILL_ARCHIVE_PATH_UNSAFE,
            "The archive contains an entry that is not a regular file or folder.",
        )

    def _safe_relative_path(self, info: zipfile.ZipInfo) -> tuple[PurePosixPath, bool]:
        """Convert a member name to a confined relative path, or reject it."""
        raw = info.filename
        is_directory = raw.endswith("/") or info.is_dir()

        if not raw or raw in {"/", "."}:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains an entry with no usable name.",
            )
        if len(raw) > self._limits.max_path_chars:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains a path longer than allowed.",
            )
        if "\x00" in raw:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains an entry with an invalid name.",
            )
        # A backslash is a legal character in a POSIX filename, so reinterpreting
        # it as a separator would let "nested\..\escape" mean different things on
        # different hosts. Refuse rather than guess.
        if "\\" in raw:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains a backslash in an entry name. Repackage it "
                "with forward-slash paths.",
            )
        if raw.startswith("/") or _looks_like_windows_absolute(raw):
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains an absolute path.",
            )

        components = [part for part in raw.split("/") if part]
        if not components:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains an entry with no usable name.",
            )
        if len(components) > self._limits.max_path_depth:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive nests folders more deeply than allowed.",
            )
        for component in components:
            self._require_portable_component(component)

        return PurePosixPath(*components), is_directory

    @staticmethod
    def _require_portable_component(component: str) -> None:
        if component in {".", ".."}:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains a path that points outside the bundle.",
            )
        if component != component.rstrip(". "):
            # Windows strips trailing dots and spaces, so "trailing. " and
            # "trailing" would resolve to the same file after extraction.
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains a name ending in a dot or space.",
            )
        if component.split(".", 1)[0].lower() in _RESERVED_WINDOWS_STEMS:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains a reserved device name.",
            )

    @staticmethod
    def _collision_key(relative_path: PurePosixPath) -> str:
        """Key two member names collide under on a real filesystem.

        Casefolding covers NTFS and APFS case-insensitivity; NFC normalization
        covers macOS decomposing names, where "é" written two ways is one file.
        """
        return unicodedata.normalize("NFC", str(relative_path)).casefold()

    def _extract_planned(
        self,
        archive: zipfile.ZipFile,
        members: list[_PlannedMember],
        destination: Path,
        *,
        compressed_bytes: int,
        declared_expanded_bytes: int,
    ) -> SkillArchiveSummary:
        """Stream every planned member into a temporary root, then promote it."""
        destination = destination.resolve()
        if destination.exists() and not destination.is_dir():
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The extraction destination is not a directory.",
            )
        staging = destination.with_name(f"{destination.name}.extract-{uuid.uuid4().hex}")
        expanded_bytes = 0
        file_count = 0

        try:
            staging.mkdir(parents=True)
            staging_root = staging.resolve()
            for member in members:
                target = staging_root / Path(*member.relative_path.parts)
                if member.is_directory:
                    self._require_confined(target, staging_root)
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                self._require_confined(target, staging_root)
                written = self._copy_member(archive, member, target, expanded_bytes)
                expanded_bytes += written
                file_count += 1
            if destination.exists():
                # The caller pre-created the destination, so the directory itself
                # cannot be replaced; move the validated entries into it instead.
                self._promote_into_existing(staging, destination)
            else:
                os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        self._require_sane_ratio(expanded_bytes, compressed_bytes)
        if expanded_bytes != declared_expanded_bytes:
            logger.debug(
                "archive declared %d expanded bytes and produced %d",
                declared_expanded_bytes,
                expanded_bytes,
            )
        return SkillArchiveSummary(
            compressed_bytes=compressed_bytes,
            expanded_bytes=expanded_bytes,
            file_count=file_count,
        )

    @staticmethod
    def _promote_into_existing(staging: Path, destination: Path) -> None:
        for entry in list(staging.iterdir()):
            os.replace(entry, destination / entry.name)
        shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _require_confined(target: Path, root: Path) -> None:
        """Re-check containment after the OS resolved the path, before opening it."""
        parent = target.parent.resolve() if target.parent.exists() else target.parent
        if not is_under_root(parent, root):
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains a path that points outside the bundle.",
            )

    def _copy_member(
        self,
        archive: zipfile.ZipFile,
        member: _PlannedMember,
        target: Path,
        already_expanded: int,
    ) -> int:
        """Copy one member in bounded chunks, counting real bytes as they land."""
        written = 0
        try:
            with archive.open(member.info, "r") as source, target.open("xb") as sink:
                while True:
                    chunk = source.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    # Declared sizes were checked in preflight; these two checks
                    # exist because a header can understate the real payload.
                    if written > self._limits.max_file_bytes:
                        raise SkillArchiveError(
                            SKILL_ARCHIVE_TOO_LARGE,
                            "A file inside the archive is larger than allowed.",
                            status_code=413,
                        )
                    if already_expanded + written > self._limits.max_expanded_bytes:
                        raise SkillArchiveError(
                            SKILL_ARCHIVE_TOO_LARGE,
                            "The archive expands to more data than allowed.",
                            status_code=413,
                        )
                    sink.write(chunk)
        except FileExistsError as exc:
            raise SkillArchiveError(
                SKILL_ARCHIVE_PATH_UNSAFE,
                "The archive contains two entries whose paths collide on this "
                "filesystem. Repackage it with distinct names.",
            ) from exc
        except zipfile.BadZipFile as exc:
            # Includes the CRC mismatch zipfile raises at end-of-member.
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The archive is corrupted and could not be extracted.",
            ) from exc
        except OSError as exc:
            raise SkillArchiveError(
                SKILL_ARCHIVE_INVALID,
                "The archive could not be extracted to local storage.",
            ) from exc
        return written


def _looks_like_windows_absolute(raw: str) -> bool:
    """Detect drive-qualified and UNC names before any separator handling."""
    if raw.startswith("//"):
        return True
    return len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha()
