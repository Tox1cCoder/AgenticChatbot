"""User-scoped staging of uploaded skill archives.

An upload is untrusted content that has to sit in the user's profile long enough
for them to look at a preview and decide. That waiting period is the risk this
module manages:

* **it is never executed.** Staging validates the archive and reads its metadata
  (front matter, declared commands, dependencies). It never prepares a runtime or
  runs a build, so nothing in an uploaded bundle runs before an explicit
  confirmation.
* **it belongs to one user.** Records live under the owner's profile and every
  lookup is by ``(owner, upload_id)``. A foreign id and an unknown id return the
  same not-found error, so an upload id cannot be probed for existence.
* **it expires and is bounded.** Outstanding count, total bytes, and attempt rate
  are capped per profile, and free disk space is checked before extraction, so a
  staged upload cannot fill the disk or be left indefinitely.
* **a failure leaves nothing.** Each attempt gets its own random staging
  directory that is removed whole on any error.

The persisted receipt is the source of truth, not memory: the sidecar restarts
often and a staged upload has to survive that, while orphaned bytes from a killed
process must not.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import (
    get_installed_skills_root,
    get_skill_locks_root,
    get_skill_uploads_root,
    is_under_root,
    sanitize_filename,
)
from client_backend.schemas.skill_installation import (
    SkillArchivePreview,
    SkillArchiveSummary,
    SkillCollectionInfo,
    SkillExistingSkill,
    SkillUploadRecord,
)
from client_backend.services.skill_runtime.archive import (
    SkillArchiveError,
    SkillArchiveValidator,
)
from client_backend.services.skill_runtime.audit import SkillLifecycleAuditWriter
from client_backend.services.skill_runtime.collection import (
    MAX_SKILLS_PER_COLLECTION,
    DiscoveredCollection,
    discover_collection,
)
from client_backend.services.skill_runtime.locks import profile_lock
from client_backend.services.skill_runtime.state import atomic_write_json, read_json_object
from shared.skills.errors import SkillRuntimeError
from shared.skills.front_matter import parse_skill_front_matter

logger = get_logger(__name__)

_READ_CHUNK_BYTES = 1024 * 1024
_ARCHIVE_FILENAME = "archive.zip"
_EXTRACTED_DIRNAME = "extracted"
_RECORD_FILENAME = "upload.json"
_ATTEMPTS_FILENAME = "attempts.json"

SKILL_ARCHIVE_TYPE_UNSUPPORTED = "SKILL_ARCHIVE_TYPE_UNSUPPORTED"
SKILL_BUNDLE_INVALID = "SKILL_BUNDLE_INVALID"
SKILL_UPLOAD_NOT_FOUND = "SKILL_UPLOAD_NOT_FOUND"
SKILL_UPLOAD_STATE_INVALID = "SKILL_UPLOAD_STATE_INVALID"
SKILL_UPLOAD_CONSUMED = "SKILL_UPLOAD_CONSUMED"
SKILL_UPLOAD_QUOTA_EXCEEDED = "SKILL_UPLOAD_QUOTA_EXCEEDED"
SKILL_STORAGE_INSUFFICIENT = "SKILL_STORAGE_INSUFFICIENT"


class AsyncReadable(Protocol):
    """The part of Starlette's ``UploadFile`` this service depends on."""

    async def read(self, size: int = -1) -> bytes: ...


class SkillUploadError(Exception):
    """A normalized, client-safe upload rejection."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class SkillUploadNotFoundError(SkillUploadError):
    """Raised for an unknown, expired, or foreign upload.

    One error for all three on purpose: distinguishing them would turn an upload
    id into an oracle for what another profile is holding.
    """

    def __init__(self, message: str = "The upload is unknown or has expired.") -> None:
        super().__init__(SKILL_UPLOAD_NOT_FOUND, message, status_code=404)


class SkillUploadStateError(SkillUploadError):
    """Raised when an upload cannot make the requested transition."""

    def __init__(self, message: str) -> None:
        super().__init__(SKILL_UPLOAD_STATE_INVALID, message, status_code=400)


def _resolve_archive_bundle_root(extracted: Path) -> Path:
    """Strip one redundant wrapper directory from an extracted archive.

    Zipping a folder -- the way a person actually produces one of these -- yields
    ``google-calendar/SKILL.md`` and ``google-calendar/bin/...`` rather than
    ``SKILL.md`` at the top. The bundle root is what publishes commands: asset
    discovery looks for ``bin/`` and ``scripts/`` directly beneath it. Treating the
    extraction directory as the root in that case finds no commands at all and
    installs a silently inert skill.

    Exactly one leading directory is removed, and only when nothing else sits
    beside it, so a bundle that legitimately has ``bin/`` next to ``skills/``
    keeps its real root.
    """
    try:
        entries = list(extracted.iterdir())
    except OSError:
        return extracted
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted


def available_bytes(path: Path) -> int:
    """Free bytes on the volume holding ``path``; a seam for tests."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


class SkillUploadService:
    """Stage, look up, claim, and expire per-user skill archive uploads."""

    def __init__(
        self,
        *,
        environment_manager: Any = None,
        registry: Any = None,
        clock: Callable[[], datetime] | None = None,
        audit: SkillLifecycleAuditWriter | None = None,
    ) -> None:
        self._environment = environment_manager
        self._registry = registry
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._audit = audit
        self._recovered: set[str] = set()

    # ---------------------------------------------------------------- staging

    async def stage(
        self,
        *,
        user_id: str,
        filename: str,
        stream: AsyncReadable,
    ) -> SkillUploadRecord:
        """Persist, validate, and preview one uploaded archive.

        Args:
            user_id: Authenticated owner.
            filename: Client-supplied name, used only for the echoed label.
            stream: Async reader over the multipart body.

        Returns:
            The staged record, including the preview a client confirms against.

        Raises:
            SkillUploadError: Any type, size, quota, storage, or bundle problem.
        """
        upload_id = uuid.uuid4().hex
        try:
            # Ordered so a rejection audits exactly once: the filename check runs
            # inside the same try as the staging work, and only the outer handler
            # writes the rejection event.
            safe_filename = self._require_zip_filename(filename)
            async with profile_lock(user_id, "uploads"):
                self._record_attempt(user_id)
                self._require_quota_headroom(user_id)
                staging = self._staging_dir(user_id, upload_id)
                try:
                    staging.mkdir(parents=True, exist_ok=False)
                    archive_path = staging / _ARCHIVE_FILENAME
                    compressed_bytes = await self._write_stream(stream, archive_path)
                    self._require_extraction_headroom(staging, compressed_bytes)
                    record = await self._validate_and_preview(
                        user_id=user_id,
                        upload_id=upload_id,
                        filename=safe_filename,
                        archive_path=archive_path,
                        staging=staging,
                        compressed_bytes=compressed_bytes,
                    )
                except BaseException:
                    self._discard(staging)
                    raise
        except SkillUploadError as exc:
            self._audit_event(
                "upload_rejected",
                user_id=user_id,
                upload_id=upload_id,
                error_code=exc.code,
                status="rejected",
            )
            raise

        self._audit_event(
            "upload_staged",
            user_id=user_id,
            upload_id=upload_id,
            skill=record.preview.name,
            source_hash=record.preview.source_hash,
            status="staged",
            metrics={
                "compressed_bytes": record.archive.compressed_bytes,
                "expanded_bytes": record.archive.expanded_bytes,
                "file_count": record.archive.file_count,
            },
        )
        return record

    @staticmethod
    def _require_zip_filename(filename: str) -> str:
        """Accept only a ``.zip`` name, and reduce it to a bare label.

        The extension is not evidence that the body is a ZIP -- the archive
        validator decides that -- but rejecting other names early gives the user a
        precise error instead of a generic "malformed archive". The sanitized
        result is echoed in responses and never joined into a path.
        """
        candidate = sanitize_filename(str(filename or "").strip())
        if not candidate or not candidate.lower().endswith(".zip"):
            raise SkillUploadError(
                SKILL_ARCHIVE_TYPE_UNSUPPORTED,
                "Only a single .zip skill archive can be uploaded.",
                status_code=415,
            )
        return candidate

    async def _write_stream(self, stream: AsyncReadable, target: Path) -> int:
        """Copy the request body to disk, stopping the moment it is too large."""
        limit = int(client_settings.skill_upload_max_bytes)
        written = 0
        try:
            with target.open("xb") as sink:
                while True:
                    chunk = await stream.read(_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise SkillUploadError(
                            "SKILL_ARCHIVE_TOO_LARGE",
                            "The archive is larger than the allowed upload size.",
                            status_code=413,
                        )
                    sink.write(chunk)
                sink.flush()
                self._fsync(sink)
        except OSError as exc:
            raise self._storage_error(exc) from exc
        if written == 0:
            raise SkillUploadError(
                "SKILL_ARCHIVE_INVALID",
                "The uploaded file is empty.",
            )
        return written

    @staticmethod
    def _fsync(handle) -> None:
        with contextlib.suppress(OSError):
            os.fsync(handle.fileno())

    def _require_extraction_headroom(self, staging: Path, compressed_bytes: int) -> None:
        """Refuse to expand an archive we may not have room for."""
        needed = min(
            int(client_settings.skill_upload_max_expanded_bytes),
            compressed_bytes * int(client_settings.skill_upload_max_compression_ratio),
        )
        try:
            free = available_bytes(staging)
        except OSError as exc:
            raise self._storage_error(exc) from exc
        if free < needed:
            raise SkillUploadError(
                SKILL_STORAGE_INSUFFICIENT,
                "There is not enough free disk space to install this archive.",
                status_code=507,
                retryable=True,
            )

    async def _validate_and_preview(
        self,
        *,
        user_id: str,
        upload_id: str,
        filename: str,
        archive_path: Path,
        staging: Path,
        compressed_bytes: int,
    ) -> SkillUploadRecord:
        """Extract under confinement, then read the bundle without running it."""
        validator = SkillArchiveValidator()
        extracted = staging / _EXTRACTED_DIRNAME
        try:
            summary = await asyncio.to_thread(validator.extract, archive_path, extracted)
        except SkillArchiveError as exc:
            raise SkillUploadError(
                exc.code,
                exc.message,
                status_code=exc.status_code,
            ) from exc

        bundle_root = _resolve_archive_bundle_root(extracted)
        collection = discover_collection(
            bundle_root,
            fallback_name=filename.removesuffix(".zip").removesuffix(".ZIP") or "skills",
        )
        previews = await self._preview_collection(user_id, bundle_root, collection)

        now = self._clock()
        record = SkillUploadRecord(
            upload_id=upload_id,
            owner=user_id,
            state="staged",
            created_at=now,
            expires_at=now + timedelta(seconds=int(client_settings.skill_upload_ttl_seconds)),
            archive=SkillArchiveSummary(
                filename=filename,
                compressed_bytes=compressed_bytes,
                expanded_bytes=summary.expanded_bytes,
                file_count=summary.file_count,
                skipped_link_count=summary.skipped_link_count,
            ),
            collection=SkillCollectionInfo(
                name=collection.manifest.name,
                version=collection.manifest.version,
                description=collection.manifest.description,
                skill_count=len(previews),
            ),
            # The first skill doubles as `preview` so a client written against the
            # single-skill contract keeps working unchanged.
            preview=previews[0],
            skills=previews,
        )
        self._write_record(record)
        return record

    async def _preview_collection(
        self,
        user_id: str,
        bundle_root: Path,
        collection: DiscoveredCollection,
    ) -> list[SkillArchivePreview]:
        """Preview every skill the archive contains, without running any of them."""
        if not collection.skill_roots:
            raise SkillUploadError(
                SKILL_BUNDLE_INVALID,
                "This archive contains no SKILL.md, so there is no skill to install. "
                "Zip the folder that holds the skill's SKILL.md.",
            )
        if len(collection.skill_roots) > MAX_SKILLS_PER_COLLECTION:
            raise SkillUploadError(
                SKILL_BUNDLE_INVALID,
                f"This archive contains {len(collection.skill_roots)} skills, more than "
                f"the {MAX_SKILLS_PER_COLLECTION} allowed in one upload. It is probably "
                "a whole workspace rather than a skill library.",
            )

        previews: list[SkillArchivePreview] = []
        for skill_root in collection.skill_roots:
            payload = await self._preview_bundle(skill_root, bundle_root)
            existing = await self._describe_existing_skill(user_id, str(payload["name"]))
            previews.append(
                SkillArchivePreview.from_installer_preview(payload, existing_skill=existing)
            )

        duplicates = self._duplicate_names(previews)
        if duplicates:
            raise SkillUploadError(
                SKILL_BUNDLE_INVALID,
                f"This archive declares the same skill name twice ({', '.join(duplicates)}). "
                "Each skill in a collection needs a distinct name.",
            )
        return previews

    @staticmethod
    def _duplicate_names(previews: list[SkillArchivePreview]) -> list[str]:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for preview in previews:
            if preview.name in seen:
                duplicates.add(preview.name)
            seen.add(preview.name)
        return sorted(duplicates)

    async def _preview_bundle(self, skill_root: Path, archive_root: Path) -> dict[str, Any]:
        """Read one skill's metadata through the installer's read-only preview."""
        installer = self._build_installer()
        try:
            return await installer.preview(skill_root)
        except SkillRuntimeError as exc:
            raise SkillUploadError(
                SKILL_BUNDLE_INVALID,
                self._bundle_rejection_message(skill_root, archive_root, exc),
            ) from exc

    @staticmethod
    def _bundle_rejection_message(
        skill_root: Path,
        archive_root: Path,
        exc: SkillRuntimeError,
    ) -> str:
        """Explain a rejected skill in terms of the archive the user uploaded."""
        try:
            location = skill_root.relative_to(archive_root).as_posix() or "."
        except ValueError:
            location = skill_root.name
        return f"The skill in '{location}' could not be read: {exc.message}"

    def _build_installer(self):
        from client_backend.services.skill_runtime.install import SkillBundleInstaller

        kwargs: dict[str, Any] = {}
        if self._registry is not None:
            kwargs["registry"] = self._registry
        if self._environment is not None:
            kwargs["environment_manager"] = self._environment
        return SkillBundleInstaller(**kwargs)

    async def _describe_existing_skill(
        self,
        user_id: str,
        name: str,
    ) -> SkillExistingSkill | None:
        """Report the installed skill this upload would replace, if any.

        Replaceability is decided by *location*: only a bundle under the profile's
        installed root is ours to overwrite. A skill discovered from a
        user-configured root stays untouched even though its metadata may look
        identical, so the UI must not offer an update for it.
        """
        if self._registry is None:
            return None
        await self._registry.initialize()
        existing = self._registry.get_skill(name)
        if existing is None:
            return None
        installed = is_under_root(existing.bundle_root, get_installed_skills_root(user_id))
        return SkillExistingSkill(
            name=existing.name,
            source_hash=existing.source_hash,
            install_source="profile" if installed else "configured_root",
            enabled=bool(getattr(existing, "enabled", False)),
            replaceable=installed,
        )

    # ---------------------------------------------------------------- lookups

    def get_owned(self, user_id: str, upload_id: str) -> SkillUploadRecord:
        """Return one live upload owned by ``user_id``.

        Raises:
            SkillUploadNotFoundError: Unknown, foreign, or expired.
        """
        record = self._read_record(user_id, upload_id)
        if record is None or record.owner != user_id:
            raise SkillUploadNotFoundError()
        if self._is_expired(record):
            self._discard(self._staging_dir(user_id, upload_id))
            raise SkillUploadNotFoundError()
        return record

    def extracted_root(self, user_id: str, upload_id: str) -> Path:
        """Return the validated bundle directory for a staged upload.

        For a collection this is the archive root; use :meth:`skill_roots` to
        install the individual skills inside it.
        """
        self.get_owned(user_id, upload_id)
        return _resolve_archive_bundle_root(
            self._staging_dir(user_id, upload_id) / _EXTRACTED_DIRNAME
        )

    def skill_roots(self, user_id: str, upload_id: str) -> list[tuple[str, Path]]:
        """Return ``(skill name, bundle directory)`` for every skill in an upload.

        Re-discovered from disk rather than read from the receipt: the receipt
        stores previews, and the installer needs the directories those previews
        were taken from. Discovery is deterministic over an extracted tree that
        nothing else writes to.
        """
        record = self.get_owned(user_id, upload_id)
        archive_root = self.extracted_root(user_id, upload_id)
        collection = discover_collection(archive_root, fallback_name=record.archive.filename)

        by_name: dict[str, Path] = {}
        for skill_root in collection.skill_roots:
            parsed = parse_skill_front_matter(
                (skill_root / "SKILL.md").read_text(encoding="utf-8")
            )
            name = (parsed.name if parsed else None) or skill_root.name
            by_name.setdefault(name, skill_root)

        # Ordered by the preview the user approved, so installation follows the
        # list they were shown rather than a filesystem ordering.
        ordered: list[tuple[str, Path]] = []
        for preview in record.skills or [record.preview]:
            root = by_name.get(preview.name)
            if root is not None:
                ordered.append((preview.name, root))
        return ordered

    # ------------------------------------------------------------ transitions

    def delete(self, user_id: str, upload_id: str) -> None:
        """Discard a staged upload and its bytes."""
        record = self.get_owned(user_id, upload_id)
        if record.state == "installing":
            raise SkillUploadStateError(
                "This upload is being installed and cannot be cancelled. "
                "Cancel the installation instead."
            )
        self._discard(self._staging_dir(user_id, upload_id))
        self._audit_event(
            "upload_cancelled",
            user_id=user_id,
            upload_id=upload_id,
            status="cancelled",
        )

    def claim_for_operation(
        self,
        user_id: str,
        upload_id: str,
        *,
        request_fingerprint: str,
        operation_id: str,
    ) -> SkillUploadRecord:
        """Bind this upload to exactly one installation request.

        Re-submitting the identical request returns the existing claim, which is
        what makes a retried POST idempotent instead of a second install. A
        different request for the same upload is a conflict: the user confirmed
        one thing, and a second confirmation must start from a fresh preview.
        """
        record = self.get_owned(user_id, upload_id)
        if record.state == "installed":
            raise SkillUploadStateError("This upload has already been installed.")
        if record.request_fingerprint is not None:
            if record.request_fingerprint != request_fingerprint:
                raise SkillUploadError(
                    SKILL_UPLOAD_CONSUMED,
                    "This upload already has a different installation request. "
                    "Upload the archive again to change it.",
                    status_code=409,
                )
            return record

        updated = record.model_copy(
            update={
                "state": "installing",
                "request_fingerprint": request_fingerprint,
                "operation_id": operation_id,
            }
        )
        self._write_record(updated)
        return updated

    def mark_succeeded(self, user_id: str, upload_id: str) -> SkillUploadRecord:
        """Record a completed install and drop the archive and extracted bytes.

        The receipt is kept so a client polling the operation can still resolve
        its upload, but the content is gone: it has been copied into the installed
        bundle and holding a second copy only widens exposure.
        """
        record = self.get_owned(user_id, upload_id)
        updated = record.model_copy(update={"state": "installed"})
        self._write_record(updated)
        staging = self._staging_dir(user_id, upload_id)
        with contextlib.suppress(OSError):
            (staging / _ARCHIVE_FILENAME).unlink(missing_ok=True)
        shutil.rmtree(staging / _EXTRACTED_DIRNAME, ignore_errors=True)
        return updated

    # ------------------------------------------------------------- maintenance

    def cleanup_expired(self, user_id: str) -> int:
        """Remove every expired upload for one profile. Returns the count."""
        removed = 0
        for record in self._iter_records(user_id):
            if self._is_expired(record):
                self._discard(self._staging_dir(user_id, record.upload_id))
                removed += 1
                self._audit_event(
                    "upload_expired",
                    user_id=user_id,
                    upload_id=record.upload_id,
                    status="expired",
                )
        return removed

    def recover(self, user_id: str) -> None:
        """Drop expired records and staging directories with no readable receipt.

        A directory without a receipt is the signature of a process killed
        mid-stage: bytes on disk that nothing will ever claim.
        """
        self.cleanup_expired(user_id)
        root = get_skill_uploads_root(user_id)
        if not root.is_dir():
            return
        for directory in list(root.iterdir()):
            if not directory.is_dir():
                continue
            if not (directory / _RECORD_FILENAME).is_file():
                logger.info("removing orphaned skill upload staging directory")
                shutil.rmtree(directory, ignore_errors=True)

    async def ensure_recovered(self, user_id: str) -> None:
        """Run recovery once per profile per process."""
        if user_id in self._recovered:
            return
        async with profile_lock(user_id, "uploads-recovery"):
            if user_id in self._recovered:
                return
            await asyncio.to_thread(self.recover, user_id)
            self._recovered.add(user_id)

    # ------------------------------------------------------------- quota/rate

    def _require_quota_headroom(self, user_id: str) -> None:
        records = [record for record in self._iter_records(user_id) if not self._is_expired(record)]
        outstanding = [record for record in records if record.state != "installed"]
        if len(outstanding) >= int(client_settings.skill_upload_max_outstanding):
            raise SkillUploadError(
                SKILL_UPLOAD_QUOTA_EXCEEDED,
                "Too many uploads are waiting to be installed. "
                "Install or cancel one before uploading another.",
                status_code=413,
            )
        if self._staged_bytes(user_id) >= int(client_settings.skill_upload_quota_bytes):
            raise SkillUploadError(
                SKILL_UPLOAD_QUOTA_EXCEEDED,
                "Staged uploads already use the allowed storage. "
                "Install or cancel one before uploading another.",
                status_code=413,
            )

    def _staged_bytes(self, user_id: str) -> int:
        root = get_skill_uploads_root(user_id)
        if not root.is_dir():
            return 0
        return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    def _record_attempt(self, user_id: str) -> None:
        """Count this attempt in a persisted sliding window.

        Rejected attempts count too. A rate limit that only counted successes
        would leave archive validation itself as an unbounded amount of work an
        attacker could ask for.
        """
        window = float(client_settings.skill_upload_rate_limit_window_seconds)
        limit = int(client_settings.skill_upload_rate_limit_count)
        path = get_skill_uploads_root(user_id) / _ATTEMPTS_FILENAME
        now = self._clock()
        payload = read_json_object(path) or {}
        stamps = [str(value) for value in payload.get("attempts", []) if isinstance(value, str)]
        recent: list[str] = []
        for stamp in stamps:
            try:
                moment = datetime.fromisoformat(stamp)
            except ValueError:
                continue
            if (now - moment).total_seconds() < window:
                recent.append(stamp)
        if len(recent) >= limit:
            raise SkillUploadError(
                SKILL_UPLOAD_QUOTA_EXCEEDED,
                "Too many upload attempts. Wait a moment and try again.",
                status_code=413,
                retryable=True,
            )
        recent.append(now.isoformat())
        atomic_write_json(path, {"version": 1, "attempts": recent})

    # ------------------------------------------------------------- persistence

    def _staging_dir(self, user_id: str, upload_id: str) -> Path:
        safe_upload_id = self._require_opaque_id(upload_id)
        return get_skill_uploads_root(user_id) / safe_upload_id

    @staticmethod
    def _require_opaque_id(upload_id: str) -> str:
        """Reject anything that is not an opaque id before it becomes a path."""
        candidate = str(upload_id or "").strip()
        if not candidate or not candidate.replace("-", "").isalnum():
            raise SkillUploadNotFoundError()
        return candidate

    def _write_record(self, record: SkillUploadRecord) -> None:
        path = self._staging_dir(record.owner, record.upload_id) / _RECORD_FILENAME
        atomic_write_json(path, record.to_record())

    def _read_record(self, user_id: str, upload_id: str) -> SkillUploadRecord | None:
        try:
            path = self._staging_dir(user_id, upload_id) / _RECORD_FILENAME
        except SkillUploadNotFoundError:
            return None
        try:
            payload = read_json_object(path)
        except Exception:
            logger.warning("discarding unreadable skill upload receipt", exc_info=True)
            return None
        if payload is None:
            return None
        try:
            return SkillUploadRecord.model_validate(payload)
        except Exception:
            logger.warning("discarding malformed skill upload receipt", exc_info=True)
            return None

    def _iter_records(self, user_id: str):
        root = get_skill_uploads_root(user_id)
        if not root.is_dir():
            return
        for directory in sorted(root.iterdir()):
            if not directory.is_dir():
                continue
            record = self._read_record(user_id, directory.name)
            if record is not None:
                yield record

    def _is_expired(self, record: SkillUploadRecord) -> bool:
        return self._clock() >= record.expires_at

    @staticmethod
    def _discard(staging: Path) -> None:
        shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _storage_error(exc: OSError) -> SkillUploadError:
        if exc.errno in {errno.ENOSPC, errno.EDQUOT}:
            return SkillUploadError(
                SKILL_STORAGE_INSUFFICIENT,
                "There is not enough free disk space to store this archive.",
                status_code=507,
                retryable=True,
            )
        return SkillUploadError(
            "SKILL_ARCHIVE_INVALID",
            "The archive could not be saved to local storage.",
        )

    def _audit_event(self, event: str, **fields: Any) -> None:
        if self._audit is None:
            return
        self._audit.write(event=event, **fields)

    # ------------------------------------------------------------- test seams

    def persist_for_test(self, record: SkillUploadRecord) -> None:
        """Write a record directly; used by tests to arrange stored state."""
        self._write_record(record)

    def staging_dir_for_test(self, user_id: str, upload_id: str) -> Path:
        """Expose the staging directory for assertions about stored bytes."""
        return self._staging_dir(user_id, upload_id)


_upload_service: SkillUploadService | None = None


def get_skill_upload_service() -> SkillUploadService:
    """Return the process-wide upload service."""
    global _upload_service
    if _upload_service is None:
        _upload_service = SkillUploadService(audit=SkillLifecycleAuditWriter())
    return _upload_service


def reset_skill_upload_service() -> None:
    """Drop the cached service; used by tests and profile transitions."""
    global _upload_service
    _upload_service = None


# Imported for monkeypatching in tests and to document the coupling.
__all__ = [
    "AsyncReadable",
    "SkillUploadError",
    "SkillUploadNotFoundError",
    "SkillUploadService",
    "SkillUploadStateError",
    "available_bytes",
    "get_installed_skills_root",
    "get_skill_locks_root",
    "get_skill_upload_service",
    "get_skill_uploads_root",
    "reset_skill_upload_service",
]
