"""Persisted asynchronous skill installations.

Installing a bundle can take minutes -- a Python project's dependency build is
the usual reason -- which is longer than a browser will hold a request open. The
work therefore runs in a background task while the client polls a receipt.

Everything hard about this design follows from that receipt having to be true
after a crash:

* **the receipt is written before the task starts.** A client that gets a 202 can
  always resolve its operation id, even if the process dies immediately after.
* **cancellation has a hard boundary.** Before the installer's atomic promotion,
  cancelling changes nothing on disk. After it, the skill is installed, so cancel
  reports ``SKILL_OPERATION_COMMITTED`` instead of pretending otherwise. The
  boundary is durably recorded, not held in memory.
* **recovery decides by evidence, not by state.** A ``running`` operation found at
  startup was interrupted, and the only reliable way to know whether it committed
  is to compare its source hash against what is actually installed.
* **the same request twice is one installation.** A fingerprint of the normalized
  request claims the upload, so a retried POST returns the original operation
  rather than installing again.
* **catalog publication cannot fail the operation.** A committed install with a
  pending sync is a success reporting ``catalogSyncStatus: "pending"``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from client_backend.core.config import client_settings
from client_backend.core.logging import get_logger
from client_backend.core.paths import get_skill_operations_root
from client_backend.schemas.skill_installation import (
    SkillInstallationFailure,
    SkillInstallationOperationModel,
    SkillInstallationRequest,
    SkillInstallationResult,
)
from client_backend.services.skill_catalog import get_skill_catalog_service
from client_backend.services.skill_runtime.audit import SkillLifecycleAuditWriter
from client_backend.services.skill_runtime.locks import SkillLockTimeoutError, profile_lock
from client_backend.services.skill_runtime.state import atomic_write_json, read_json_object
from client_backend.services.skill_runtime.uploads import (
    SkillUploadError,
    SkillUploadNotFoundError,
    get_skill_upload_service,
)
from shared.skills.errors import (
    SKILL_CONFIGURED_ROOT_CONFLICT,
    SKILL_INSTALL_CONFLICT,
    SKILL_SETUP_FAILED,
    SkillRuntimeError,
)

logger = get_logger(__name__)

_RECEIPT_PREFIX = "operation-"

SKILL_OPERATION_NOT_FOUND = "SKILL_OPERATION_NOT_FOUND"
SKILL_OPERATION_COMMITTED = "SKILL_OPERATION_COMMITTED"
SKILL_OPERATION_STATE_INVALID = "SKILL_OPERATION_STATE_INVALID"
SKILL_INSTALL_LOCKED = "SKILL_INSTALL_LOCKED"

# Failures worth retrying with the same inputs. Everything else needs the user to
# change something (a different archive, a fresh preview, more disk space).
_RETRYABLE_CODES = frozenset({SKILL_SETUP_FAILED, SKILL_INSTALL_LOCKED, "SKILL_INTERRUPTED"})


class SkillOperationError(Exception):
    """A normalized, client-safe operation failure."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class SkillOperationNotFoundError(SkillOperationError):
    """Raised for an unknown, expired, or foreign operation.

    One error for all three, so an operation id cannot be probed for existence.
    """

    def __init__(self) -> None:
        super().__init__(
            SKILL_OPERATION_NOT_FOUND,
            "The installation is unknown or has expired.",
            status_code=404,
        )


class SkillOperationConflictError(SkillOperationError):
    """Raised when an operation cannot make the requested transition."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, status_code=409)


class _OperationObserver:
    """Bridges the installer's lifecycle hooks to persisted operation state."""

    def __init__(self, service: SkillInstallationService, user_id: str, operation_id: str) -> None:
        self._service = service
        self._user_id = user_id
        self._operation_id = operation_id

    async def phase(self, name: str) -> None:
        self._service._record_phase(self._user_id, self._operation_id, name)

    async def before_commit(self) -> None:
        """Stop here if cancelled; otherwise mark the commit as begun.

        This is the only place both can be decided atomically enough to matter:
        after it returns, the installer promotes the bundle and no later
        cancellation can be honored.
        """
        operation = self._service._read(self._user_id, self._operation_id)
        if operation is None:
            return
        if operation.cancel_requested:
            raise asyncio.CancelledError()
        self._service._persist(
            operation.model_copy(update={"commit_started_at": self._service._now()})
        )


class SkillInstallationService:
    """Start, observe, cancel, and recover asynchronous skill installations."""

    def __init__(
        self,
        *,
        installer_factory: Callable[[], Any] | None = None,
        upload_service: Any = None,
        catalog_service: Any = None,
        clock: Callable[[], datetime] | None = None,
        audit: SkillLifecycleAuditWriter | None = None,
    ) -> None:
        self._installer_factory = installer_factory or self._default_installer
        self._uploads = upload_service
        self._catalog = catalog_service
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._audit = audit
        self._tasks: dict[str, asyncio.Task] = {}
        self._recovered: set[str] = set()

    @staticmethod
    def _default_installer():
        from client_backend.services.skill_runtime.install import SkillBundleInstaller

        return SkillBundleInstaller()

    def _upload_service(self):
        return self._uploads if self._uploads is not None else get_skill_upload_service()

    def _catalog_service(self):
        return self._catalog if self._catalog is not None else get_skill_catalog_service()

    def _now(self) -> datetime:
        return self._clock()

    # ------------------------------------------------------------------ start

    async def start(
        self,
        user_id: str,
        upload_id: str,
        request: SkillInstallationRequest,
    ) -> SkillInstallationOperationModel:
        """Claim the upload and begin installing it in the background.

        Returns the existing operation when the identical request is submitted
        again, which is what makes a retried POST safe.
        """
        uploads = self._upload_service()
        upload = uploads.get_owned(user_id, upload_id)
        fingerprint = self._fingerprint(request)

        if upload.request_fingerprint == fingerprint and upload.operation_id:
            existing = self._read(user_id, upload.operation_id)
            if existing is not None:
                return existing

        operation_id = uuid.uuid4().hex
        try:
            uploads.claim_for_operation(
                user_id,
                upload_id,
                request_fingerprint=fingerprint,
                operation_id=operation_id,
            )
        except SkillUploadError as exc:
            raise SkillOperationConflictError(exc.code, exc.message) from exc

        now = self._now()
        operation = SkillInstallationOperationModel(
            operation_id=operation_id,
            upload_id=upload_id,
            owner=user_id,
            state="pending",
            phase="validating",
            created_at=now,
            expires_at=now
            + timedelta(seconds=int(client_settings.skill_operation_receipt_ttl_seconds)),
            upload_source_hash=upload.preview.source_hash,
            request_fingerprint=fingerprint,
        )
        # Persisted before the task exists: a 202 must always be resolvable, even
        # if the process dies on the next line.
        self._persist(operation)
        self._audit_event(
            "install_queued",
            user_id=user_id,
            upload_id=upload_id,
            operation_id=operation_id,
            skill=upload.preview.name,
            source_hash=upload.preview.source_hash,
            status="pending",
        )

        task = asyncio.create_task(self._run(user_id, operation_id, request))
        self._tasks[operation_id] = task
        task.add_done_callback(lambda finished: self._finalize_task(operation_id, finished))
        return operation

    @staticmethod
    def _fingerprint(request: SkillInstallationRequest) -> str:
        """Hash the normalized request so an identical retry is recognizable."""
        return hashlib.sha256(
            json.dumps(
                request.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _finalize_task(self, operation_id: str, task: asyncio.Task) -> None:
        """Consume the task's exception so it is never an unretrieved warning."""
        self._tasks.pop(operation_id, None)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "skill installation task %s ended with an unhandled error",
                operation_id,
                exc_info=exception,
            )

    # ------------------------------------------------------------------ worker

    async def _run(
        self,
        user_id: str,
        operation_id: str,
        request: SkillInstallationRequest,
    ) -> None:
        started = self._now()
        operation = self._read(user_id, operation_id)
        if operation is None:
            return
        self._persist(operation.model_copy(update={"state": "running", "started_at": started}))

        uploads = self._upload_service()
        try:
            # Rechecked here, not just at start(): the upload can expire or be
            # deleted while this task was queued behind another install.
            upload = uploads.get_owned(user_id, operation.upload_id)
            if request.expected_source_hash != upload.preview.source_hash:
                raise SkillRuntimeError(
                    "SKILL_SOURCE_CHANGED",
                    "the staged archive changed after preview; upload it again",
                )
            bundle_root = uploads.extracted_root(user_id, operation.upload_id)

            installer = self._installer_factory()
            result = await installer.install(
                bundle_root,
                expected_source_hash=request.expected_source_hash,
                approve_setup=request.approve_setup,
                replace_source_hash=request.replace_source_hash,
                source_kind="upload",
                observer=_OperationObserver(self, user_id, operation_id),
            )
        except asyncio.CancelledError:
            self._mark_cancelled(user_id, operation_id)
            raise
        except (SkillUploadNotFoundError, SkillUploadError) as exc:
            self._fail(user_id, operation_id, exc.code, exc.message)
            return
        except SkillLockTimeoutError:
            self._fail(
                user_id,
                operation_id,
                SKILL_INSTALL_LOCKED,
                "Another skill operation is in progress. Try again in a moment.",
            )
            return
        except SkillRuntimeError as exc:
            self._fail(user_id, operation_id, self._public_code(exc.code), exc.message)
            return
        except Exception as exc:  # noqa: BLE001 - a receipt must never be left running
            logger.error("skill installation %s failed unexpectedly", operation_id, exc_info=True)
            self._fail(
                user_id,
                operation_id,
                "SKILL_INSTALL_INVALID",
                str(exc) or "The installation failed unexpectedly.",
            )
            return

        await self._complete(user_id, operation_id, result, started=started)

    async def _complete(
        self,
        user_id: str,
        operation_id: str,
        install_result: dict[str, Any],
        *,
        started: datetime,
    ) -> None:
        """Record success, then publish the catalog without risking that success."""
        self._record_phase(user_id, operation_id, "refreshingCatalog")
        catalog = await self._catalog_service().after_mutation(sync=True)
        self._record_phase(user_id, operation_id, "syncingCatalog")

        operation = self._read(user_id, operation_id)
        if operation is None:
            return
        finished = self._now()
        self._persist(
            operation.model_copy(
                update={
                    "state": "succeeded",
                    "phase": "syncingCatalog",
                    "finished_at": finished,
                    "result": SkillInstallationResult.from_installer_result(
                        install_result,
                        catalog=catalog,
                    ),
                }
            )
        )
        with contextlib.suppress(SkillUploadError):
            self._upload_service().mark_succeeded(user_id, operation.upload_id)
        self._audit_event(
            "install_succeeded",
            user_id=user_id,
            upload_id=operation.upload_id,
            operation_id=operation_id,
            skill=str(install_result.get("name") or ""),
            source_hash=str(install_result.get("source_hash") or ""),
            status="succeeded",
            sync_status=str(catalog.get("catalogSyncStatus") or ""),
            duration_ms=int((finished - started).total_seconds() * 1000),
        )

    @staticmethod
    def _public_code(code: str) -> str:
        """Map internal codes onto the published contract.

        ``SKILL_CONFIGURED_ROOT_CONFLICT`` is intentionally internal: the contract
        expresses non-replaceability through ``preview.existingSkill.replaceable``
        and publishes only ``SKILL_INSTALL_CONFLICT`` for the collision itself.
        """
        if code == SKILL_CONFIGURED_ROOT_CONFLICT:
            return SKILL_INSTALL_CONFLICT
        return code

    # ----------------------------------------------------------- transitions

    def _record_phase(self, user_id: str, operation_id: str, phase: str) -> None:
        operation = self._read(user_id, operation_id)
        if operation is None or operation.is_terminal:
            return
        if operation.phase == phase:
            return
        self._persist(operation.model_copy(update={"phase": phase}))

    def _fail(self, user_id: str, operation_id: str, code: str, message: str) -> None:
        operation = self._read(user_id, operation_id)
        if operation is None:
            return
        self._persist(
            operation.model_copy(
                update={
                    "state": "failed",
                    "finished_at": self._now(),
                    "failure": SkillInstallationFailure(
                        code=code,
                        message=message,
                        retryable=code in _RETRYABLE_CODES,
                    ),
                }
            )
        )
        self._audit_event(
            "install_failed",
            user_id=user_id,
            operation_id=operation_id,
            upload_id=operation.upload_id,
            error_code=code,
            phase=operation.phase,
            status="failed",
        )

    def _mark_cancelled(self, user_id: str, operation_id: str) -> None:
        operation = self._read(user_id, operation_id)
        if operation is None or operation.is_terminal:
            return
        self._persist(
            operation.model_copy(update={"state": "cancelled", "finished_at": self._now()})
        )
        self._audit_event(
            "install_cancelled",
            user_id=user_id,
            operation_id=operation_id,
            upload_id=operation.upload_id,
            status="cancelled",
        )

    async def cancel(self, user_id: str, operation_id: str) -> SkillInstallationOperationModel:
        """Request cancellation, if the commit boundary has not been crossed."""
        operation = self.get_owned(user_id, operation_id)
        if operation.is_terminal:
            raise SkillOperationConflictError(
                SKILL_OPERATION_STATE_INVALID,
                "This installation has already finished.",
            )
        if operation.commit_started_at is not None:
            raise SkillOperationConflictError(
                SKILL_OPERATION_COMMITTED,
                "The installation has already been committed. Keep polling for its result.",
            )

        self._persist(operation.model_copy(update={"cancel_requested": True}))
        task = self._tasks.get(operation_id)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._mark_cancelled(user_id, operation_id)
        return self.get_owned(user_id, operation_id)

    # --------------------------------------------------------------- lookups

    def get_owned(self, user_id: str, operation_id: str) -> SkillInstallationOperationModel:
        """Return one live operation owned by ``user_id``."""
        operation = self._read(user_id, operation_id)
        if operation is None or operation.owner != user_id:
            raise SkillOperationNotFoundError()
        if operation.is_terminal and self._now() >= operation.expires_at:
            self._discard(user_id, operation_id)
            raise SkillOperationNotFoundError()
        return operation

    # -------------------------------------------------------------- recovery

    async def recover(self, user_id: str) -> None:
        """Reconcile operations left mid-flight by a previous process.

        An interrupted operation is judged by evidence: if the bundle that is
        installed carries the hash this operation was installing, the atomic
        promotion completed and the operation succeeded even though nothing got to
        write that down. Otherwise nothing was committed and it failed retryably.
        """
        installed_hashes = await self._installed_source_hashes()
        for operation in self._iter_operations(user_id):
            if operation.is_terminal:
                continue
            committed = (
                operation.commit_started_at is not None
                and operation.upload_source_hash in installed_hashes
            )
            if committed:
                catalog = await self._catalog_service().snapshot(force=True, sync=False)
                self._persist(
                    operation.model_copy(
                        update={
                            "state": "succeeded",
                            "phase": "syncingCatalog",
                            "finished_at": self._now(),
                            "result": SkillInstallationResult(
                                action="installed",
                                name=self._skill_name_for(operation, catalog),
                                source_hash=str(operation.upload_source_hash or ""),
                                runtime_status="unknown",
                                catalog=catalog,
                            ),
                        }
                    )
                )
                self._audit_event(
                    "install_recovered",
                    user_id=user_id,
                    operation_id=operation.operation_id,
                    status="succeeded",
                )
                continue
            self._persist(
                operation.model_copy(
                    update={
                        "state": "failed",
                        "finished_at": self._now(),
                        "failure": SkillInstallationFailure(
                            code="SKILL_INTERRUPTED",
                            message="The installation was interrupted before it completed. "
                            "Upload the archive again to retry.",
                            retryable=True,
                        ),
                    }
                )
            )
            self._audit_event(
                "install_recovered",
                user_id=user_id,
                operation_id=operation.operation_id,
                status="failed",
                error_code="SKILL_INTERRUPTED",
            )

    @staticmethod
    def _skill_name_for(
        operation: SkillInstallationOperationModel,
        catalog: dict[str, Any],
    ) -> str:
        for entry in catalog.get("skills") or []:
            if entry.get("sourceHash") == operation.upload_source_hash:
                return str(entry.get("name") or "")
        return ""

    async def _installed_source_hashes(self) -> set[str]:
        installer = self._installer_factory()
        installed = await asyncio.to_thread(installer.list_installed)
        return {
            str(entry.get("source_hash"))
            for entry in installed
            if isinstance(entry, dict) and entry.get("source_hash")
        }

    async def ensure_recovered(self, user_id: str) -> None:
        """Run recovery and expiry once per profile per process."""
        if user_id in self._recovered:
            return
        async with profile_lock(user_id, "operations-recovery"):
            if user_id in self._recovered:
                return
            await self.recover(user_id)
            self.cleanup_expired(user_id)
            self._recovered.add(user_id)

    def cleanup_expired(self, user_id: str) -> int:
        """Delete terminal receipts past their TTL. Returns the count."""
        removed = 0
        now = self._now()
        for operation in self._iter_operations(user_id):
            if operation.is_terminal and now >= operation.expires_at:
                self._discard(user_id, operation.operation_id)
                removed += 1
        return removed

    async def shutdown(self) -> None:
        """Stop tracking in-flight tasks without abandoning their receipts.

        Tasks are cancelled rather than awaited: shutdown must not block on a
        dependency build, and an interrupted operation is exactly what recovery
        reconciles on the next start.
        """
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

    # ------------------------------------------------------------ persistence

    def _receipt_path(self, user_id: str, operation_id: str) -> Path:
        return get_skill_operations_root(user_id) / f"{_RECEIPT_PREFIX}{operation_id}.json"

    def _persist(self, operation: SkillInstallationOperationModel) -> None:
        atomic_write_json(
            self._receipt_path(operation.owner, operation.operation_id),
            operation.to_record(),
        )

    def _read(self, user_id: str, operation_id: str) -> SkillInstallationOperationModel | None:
        candidate = str(operation_id or "").strip()
        if not candidate or not candidate.replace("-", "").isalnum():
            return None
        try:
            payload = read_json_object(self._receipt_path(user_id, candidate))
        except Exception:
            logger.warning("discarding unreadable skill operation receipt", exc_info=True)
            return None
        if payload is None:
            return None
        try:
            return SkillInstallationOperationModel.model_validate(payload)
        except Exception:
            logger.warning("discarding malformed skill operation receipt", exc_info=True)
            return None

    def _iter_operations(self, user_id: str):
        root = get_skill_operations_root(user_id)
        if not root.is_dir():
            return
        for path in sorted(root.iterdir()):
            if not path.name.startswith(_RECEIPT_PREFIX) or path.suffix != ".json":
                continue
            operation = self._read(user_id, path.stem[len(_RECEIPT_PREFIX) :])
            if operation is not None:
                yield operation

    def _discard(self, user_id: str, operation_id: str) -> None:
        with contextlib.suppress(OSError):
            self._receipt_path(user_id, operation_id).unlink(missing_ok=True)

    def _audit_event(self, event: str, **fields: Any) -> None:
        if self._audit is None:
            return
        self._audit.write(event=event, **fields)

    # ------------------------------------------------------------- test seams

    def persist_for_test(self, operation: SkillInstallationOperationModel) -> None:
        """Write a receipt directly so tests can arrange stored state."""
        self._persist(operation)


_installation_service: SkillInstallationService | None = None


def get_skill_installation_service() -> SkillInstallationService:
    """Return the process-wide installation service."""
    global _installation_service
    if _installation_service is None:
        _installation_service = SkillInstallationService(audit=SkillLifecycleAuditWriter())
    return _installation_service


def reset_skill_installation_service() -> None:
    """Drop the cached service; used by tests and profile transitions."""
    global _installation_service
    _installation_service = None


__all__ = [
    "SKILL_INSTALL_LOCKED",
    "SKILL_OPERATION_COMMITTED",
    "SKILL_OPERATION_NOT_FOUND",
    "SKILL_OPERATION_STATE_INVALID",
    "SkillInstallationService",
    "SkillOperationConflictError",
    "SkillOperationError",
    "SkillOperationNotFoundError",
    "get_skill_installation_service",
    "reset_skill_installation_service",
]
