"""State machine, idempotency, cancellation, and recovery of install operations.

The receipt a client polls has to stay truthful across retries, cancellation
races, and a process that dies mid-install. These tests pin the transitions that
make that true, including the one boundary that cannot be undone: once the
installer's atomic promotion starts, the answer to "cancel" is no.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from client_backend.schemas.skill_installation import (
    SkillArchivePreview,
    SkillArchiveSummary,
    SkillInstallationOperationModel,
    SkillInstallationRequest,
    SkillUploadRecord,
)
from client_backend.services.skill_runtime.operations import (
    SkillInstallationService,
    SkillOperationConflictError,
    SkillOperationError,
    SkillOperationNotFoundError,
)

USER_A = "user-a"
USER_B = "user-b"
SOURCE_HASH = "a" * 64


def _service_globals() -> dict:
    """The operations module's globals, which is where its `except` clauses look.

    Exception classes must be raised from *this* namespace, not from a fresh
    import: another test in this directory evicts every ``client_backend`` module
    to prove an import boundary, after which a re-imported
    ``SkillUploadError`` is a different class object and
    ``except SkillUploadError`` no longer matches it.
    """
    return SkillInstallationService.__init__.__globals__


def _request(**overrides) -> SkillInstallationRequest:
    payload = {"expected_source_hash": SOURCE_HASH, "approve_setup": False}
    payload.update(overrides)
    return SkillInstallationRequest(**payload)


class _FrozenClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 7, 31, 10, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, *, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class _UploadServiceStub:
    def __init__(self, bundle_root: Path) -> None:
        self._bundle_root = bundle_root
        self._records: dict[str, SkillUploadRecord] = {}
        self.succeeded: list[str] = []
        self.missing = False

    def stage_record(
        self,
        *,
        owner: str = USER_A,
        upload_id: str = "upload-a",
    ) -> SkillUploadRecord:
        record = SkillUploadRecord(
            upload_id=upload_id,
            owner=owner,
            created_at=datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc),
            expires_at=datetime(2026, 7, 31, 10, 30, tzinfo=timezone.utc),
            archive=SkillArchiveSummary(
                filename="demo.zip",
                compressed_bytes=100,
                expanded_bytes=200,
                file_count=2,
            ),
            preview=SkillArchivePreview(
                name="demo",
                source_hash=SOURCE_HASH,
                bundle_shape="nested",
                executable_assets={},
                setup={},
            ),
        )
        self._records[upload_id] = record
        return record

    def get_owned(self, user_id: str, upload_id: str) -> SkillUploadRecord:
        not_found = _service_globals()["SkillUploadNotFoundError"]
        record = self._records.get(upload_id)
        if self.missing or record is None or record.owner != user_id:
            raise not_found()
        return record

    def claim_for_operation(self, user_id, upload_id, *, request_fingerprint, operation_id):
        record = self.get_owned(user_id, upload_id)
        if record.request_fingerprint is not None:
            if record.request_fingerprint != request_fingerprint:
                raise _service_globals()["SkillUploadError"](
                    "SKILL_UPLOAD_CONSUMED",
                    "This upload already has a different installation request.",
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
        self._records[upload_id] = updated
        return updated

    def extracted_root(self, user_id: str, upload_id: str) -> Path:
        self.get_owned(user_id, upload_id)
        return self._bundle_root

    def skill_roots(self, user_id: str, upload_id: str) -> list[tuple[str, Path]]:
        record = self.get_owned(user_id, upload_id)
        previews = record.skills or [record.preview]
        return [(preview.name, self._bundle_root / preview.name) for preview in previews]

    def mark_succeeded(self, user_id: str, upload_id: str):
        self.succeeded.append(upload_id)
        record = self._records[upload_id]
        self._records[upload_id] = record.model_copy(update={"state": "installed"})
        return self._records[upload_id]


class _InstallerStub:
    """Installer double that drives the observer exactly as the real one does."""

    def __init__(self) -> None:
        self.install_calls = 0
        self.installed_hashes: list[str] = []
        self.error: Exception | None = None
        self.commit_gate: asyncio.Event | None = None
        self.raise_after_commit = False
        self.observed_phases: list[str] = []
        self.uninstalled: list[str] = []
        self.installed_names: list[str] = []
        self.fail_on_name: str | None = None

    async def install(
        self,
        source,
        *,
        expected_source_hash=None,
        approve_setup=False,
        replace_source_hash=None,
        source_kind="path",
        observer=None,
    ):
        self.install_calls += 1
        skill_name = Path(source).name
        if self.fail_on_name and skill_name == self.fail_on_name:
            raise RuntimeError(f"install of {skill_name} failed")
        for phase in ("validating", "waitingForLock", "copying", "preparingRuntime"):
            if observer is not None:
                await observer.phase(phase)
                self.observed_phases.append(phase)
        if self.error is not None:
            raise self.error
        if self.commit_gate is not None:
            await self.commit_gate.wait()
        if observer is not None:
            await observer.phase("committing")
            await observer.before_commit()
        if self.raise_after_commit:
            raise RuntimeError("failed after the promotion started")
        self.installed_hashes.append(str(expected_source_hash))
        self.installed_names.append(skill_name)
        return {
            "name": skill_name,
            "install_id": "demo-abc123",
            "source_hash": expected_source_hash or SOURCE_HASH,
            "runtime_status": "ready",
            "action": "updated" if replace_source_hash else "installed",
        }

    def list_installed(self):
        return [{"bundle_name": "demo", "source_hash": value} for value in self.installed_hashes]

    async def uninstall(self, name: str) -> dict:
        self.uninstalled.append(name)
        return {"name": name, "removed": True, "cleanup_status": "complete"}


class _CatalogStub:
    def __init__(self) -> None:
        self.sync_status = "synced"
        self.after_mutation_calls = 0

    async def after_mutation(self, sync: bool = True) -> dict:
        self.after_mutation_calls += 1
        return self._payload()

    async def snapshot(self, force: bool = False, sync: bool = False) -> dict:
        return self._payload()

    def _payload(self) -> dict:
        return {
            "deviceId": "device-123",
            "catalogGeneration": 7,
            "catalogSyncStatus": self.sync_status,
            "skills": [{"name": "demo", "enabled": True, "sourceHash": SOURCE_HASH}],
            "totalCount": 1,
            "enabledCount": 1,
        }


@dataclass
class _OperationEnv:
    service: SkillInstallationService
    uploads: _UploadServiceStub
    installer: _InstallerStub
    catalog: _CatalogStub
    clock: _FrozenClock
    root: Path

    def staged_upload(self) -> SkillUploadRecord:
        return self.uploads.stage_record()

    async def start_and_wait(self, request: SkillInstallationRequest | None = None):
        upload = self.staged_upload()
        operation = await self.service.start(USER_A, upload.upload_id, request or _request())
        return await self.wait_terminal(operation.operation_id)

    async def wait_terminal(self, operation_id: str):
        for _ in range(200):
            operation = self.service.get_owned(USER_A, operation_id)
            if operation.is_terminal:
                return operation
            await asyncio.sleep(0.005)
        raise AssertionError("operation never reached a terminal state")

    def get(self, operation: SkillInstallationOperationModel):
        return self.service.get_owned(operation.owner, operation.operation_id)

    def persist_operation(
        self,
        *,
        state: str = "running",
        commit_started: bool = False,
        source_hash: str = SOURCE_HASH,
        owner: str = USER_A,
        operation_id: str | None = None,
    ) -> SkillInstallationOperationModel:
        now = self.clock()
        operation = SkillInstallationOperationModel(
            operation_id=operation_id or f"op{len(list(self.root.glob('operation-*.json')))}",
            upload_id="upload-a",
            owner=owner,
            state=state,
            phase="copying",
            created_at=now,
            expires_at=now + timedelta(seconds=3600),
            started_at=now,
            commit_started_at=now if commit_started else None,
            upload_source_hash=source_hash,
        )
        self.service.persist_for_test(operation)
        return operation

    def persist_installed_bundle(self, *, source_hash: str) -> None:
        self.installer.installed_hashes.append(source_hash)


@pytest.fixture()
def operation_env(tmp_path, monkeypatch) -> _OperationEnv:
    operations_globals = SkillInstallationService.__init__.__globals__
    locks_globals = operations_globals["profile_lock"].__wrapped__.__globals__
    root = tmp_path / "operations"
    monkeypatch.setitem(operations_globals, "get_skill_operations_root", lambda _user: root)
    monkeypatch.setitem(locks_globals, "get_skill_locks_root", lambda _user: tmp_path / "locks")

    bundle_root = tmp_path / "extracted"
    bundle_root.mkdir()
    uploads = _UploadServiceStub(bundle_root)
    installer = _InstallerStub()
    catalog = _CatalogStub()
    clock = _FrozenClock()
    service = SkillInstallationService(
        installer_factory=lambda: installer,
        upload_service=uploads,
        catalog_service=catalog,
        clock=clock,
        audit=None,
    )
    return _OperationEnv(
        service=service,
        uploads=uploads,
        installer=installer,
        catalog=catalog,
        clock=clock,
        root=root,
    )


@pytest.mark.asyncio
async def test_start_is_idempotent_for_identical_request(operation_env):
    upload = operation_env.staged_upload()
    request = _request()

    first = await operation_env.service.start(USER_A, upload.upload_id, request)
    await operation_env.wait_terminal(first.operation_id)
    second = await operation_env.service.start(USER_A, upload.upload_id, request)

    assert second.operation_id == first.operation_id
    assert operation_env.installer.install_calls == 1


@pytest.mark.asyncio
async def test_start_rejects_different_request_for_claimed_upload(operation_env):
    upload = operation_env.staged_upload()
    first = await operation_env.service.start(USER_A, upload.upload_id, _request())
    await operation_env.wait_terminal(first.operation_id)

    with pytest.raises(SkillOperationConflictError) as exc_info:
        await operation_env.service.start(
            USER_A,
            upload.upload_id,
            _request(approve_setup=True),
        )

    assert exc_info.value.code == "SKILL_UPLOAD_CONSUMED"
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_receipt_exists_before_the_worker_runs(operation_env):
    upload = operation_env.staged_upload()

    operation = await operation_env.service.start(USER_A, upload.upload_id, _request())

    assert operation.state == "pending"
    assert operation_env.service.get_owned(USER_A, operation.operation_id) is not None
    await operation_env.wait_terminal(operation.operation_id)


@pytest.mark.asyncio
async def test_successful_operation_reports_result_and_catalog(operation_env):
    operation = await operation_env.start_and_wait()

    assert operation.state == "succeeded"
    assert operation.phase == "syncingCatalog"
    assert operation.result.action == "installed"
    assert operation.result.catalog["catalogGeneration"] == 7
    assert operation_env.uploads.succeeded == ["upload-a"]


@pytest.mark.asyncio
async def test_update_action_is_reported_for_a_guarded_replacement(operation_env):
    operation = await operation_env.start_and_wait(
        _request(replace_source_hash="b" * 64),
    )

    assert operation.result.action == "updated"


@pytest.mark.asyncio
async def test_sync_failure_keeps_operation_succeeded(operation_env):
    operation_env.catalog.sync_status = "pending"

    operation = await operation_env.start_and_wait()

    assert operation.state == "succeeded"
    assert operation.result.catalog["catalogSyncStatus"] == "pending"


@pytest.mark.asyncio
async def test_api_payload_hides_internal_recovery_fields(operation_env):
    operation = await operation_env.start_and_wait()

    payload = operation.to_api()

    assert payload["operationId"] == operation.operation_id
    assert payload["state"] == "succeeded"
    assert payload["result"]["sourceHash"] == SOURCE_HASH
    for internal in ("owner", "commitStartedAt", "cancelRequested", "uploadSourceHash"):
        assert internal not in payload


@pytest.mark.asyncio
async def test_installer_failure_becomes_a_terminal_failed_operation(operation_env):
    runtime_error = _service_globals()["SkillRuntimeError"]
    setup_failed = _service_globals()["SKILL_SETUP_FAILED"]
    operation_env.installer.error = runtime_error(setup_failed, "dependency install failed")

    operation = await operation_env.start_and_wait()

    assert operation.state == "failed"
    assert operation.failure.code == setup_failed
    assert operation.failure.retryable is True
    assert operation.result is None


@pytest.mark.asyncio
async def test_configured_root_conflict_is_published_as_install_conflict(operation_env):
    operation_env.installer.error = _service_globals()["SkillRuntimeError"](
        _service_globals()["SKILL_CONFIGURED_ROOT_CONFLICT"],
        "skill comes from a configured skills root",
    )

    operation = await operation_env.start_and_wait()

    assert operation.failure.code == "SKILL_INSTALL_CONFLICT"
    assert operation.failure.retryable is False


@pytest.mark.asyncio
async def test_lock_timeout_is_reported_as_retryable(operation_env):
    lock_timeout = _service_globals()["SkillLockTimeoutError"]
    operation_env.installer.error = lock_timeout("skill:demo")

    operation = await operation_env.start_and_wait()

    assert operation.failure.code == "SKILL_INSTALL_LOCKED"
    assert operation.failure.retryable is True


@pytest.mark.asyncio
async def test_expired_upload_fails_the_operation_before_installing(operation_env):
    upload = operation_env.staged_upload()
    operation = await operation_env.service.start(USER_A, upload.upload_id, _request())
    operation_env.uploads.missing = True

    terminal = await operation_env.wait_terminal(operation.operation_id)

    assert terminal.state in {"failed", "succeeded"}


@pytest.mark.asyncio
async def test_stale_expected_hash_fails_without_installing(operation_env):
    upload = operation_env.staged_upload()

    operation = await operation_env.service.start(
        USER_A,
        upload.upload_id,
        _request(expected_source_hash="c" * 64),
    )
    terminal = await operation_env.wait_terminal(operation.operation_id)

    assert terminal.state == "failed"
    assert terminal.failure.code == "SKILL_SOURCE_CHANGED"
    assert operation_env.installer.install_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "commit_started", "expected"),
    [
        ("pending", False, "cancelled"),
        ("running", False, "cancelled"),
        ("running", True, "SKILL_OPERATION_COMMITTED"),
    ],
)
async def test_cancel_respects_atomic_commit_boundary(
    operation_env, state, commit_started, expected
):
    operation = operation_env.persist_operation(state=state, commit_started=commit_started)

    if expected == "SKILL_OPERATION_COMMITTED":
        with pytest.raises(SkillOperationError) as exc_info:
            await operation_env.service.cancel(USER_A, operation.operation_id)
        assert exc_info.value.code == expected
        assert exc_info.value.status_code == 409
    else:
        cancelled = await operation_env.service.cancel(USER_A, operation.operation_id)
        assert cancelled.state == expected


@pytest.mark.asyncio
async def test_cancel_of_a_terminal_operation_is_a_conflict(operation_env):
    operation = operation_env.persist_operation(state="succeeded")

    with pytest.raises(SkillOperationConflictError) as exc_info:
        await operation_env.service.cancel(USER_A, operation.operation_id)

    assert exc_info.value.code == "SKILL_OPERATION_STATE_INVALID"


@pytest.mark.asyncio
async def test_cancel_before_commit_stops_a_running_install(operation_env):
    """A live install observes the cancellation flag at the commit boundary."""
    gate = asyncio.Event()
    operation_env.installer.commit_gate = gate
    upload = operation_env.staged_upload()
    operation = await operation_env.service.start(USER_A, upload.upload_id, _request())
    for _ in range(200):
        if operation_env.installer.observed_phases:
            break
        await asyncio.sleep(0.005)

    cancel_task = asyncio.create_task(
        operation_env.service.cancel(USER_A, operation.operation_id)
    )
    await asyncio.sleep(0.01)
    gate.set()
    cancelled = await cancel_task

    assert cancelled.state == "cancelled"
    assert operation_env.installer.installed_hashes == []


@pytest.mark.asyncio
async def test_recovery_reconciles_commit_and_fails_precommit(operation_env):
    committed = operation_env.persist_operation(
        state="running",
        source_hash="a" * 64,
        commit_started=True,
        operation_id="opcommitted",
    )
    operation_env.persist_installed_bundle(source_hash="a" * 64)
    precommit = operation_env.persist_operation(
        state="running",
        source_hash="b" * 64,
        commit_started=False,
        operation_id="opprecommit",
    )

    await operation_env.service.recover(USER_A)

    assert operation_env.get(committed).state == "succeeded"
    assert operation_env.get(precommit).state == "failed"
    assert operation_env.get(precommit).failure.retryable is True


@pytest.mark.asyncio
async def test_recovery_fails_a_commit_that_left_no_installed_bundle(operation_env):
    """commit_started alone is not evidence; the installed hash is."""
    operation = operation_env.persist_operation(
        state="running",
        commit_started=True,
        source_hash="d" * 64,
    )

    await operation_env.service.recover(USER_A)

    assert operation_env.get(operation).state == "failed"
    assert operation_env.get(operation).failure.code == "SKILL_INTERRUPTED"


@pytest.mark.asyncio
async def test_recovery_leaves_terminal_operations_untouched(operation_env):
    succeeded = operation_env.persist_operation(state="succeeded", operation_id="opdone")

    await operation_env.service.recover(USER_A)

    assert operation_env.get(succeeded).state == "succeeded"


@pytest.mark.asyncio
async def test_ensure_recovered_runs_once_per_profile(operation_env, monkeypatch):
    calls: list[str] = []

    async def counting_recover(user_id: str) -> None:
        calls.append(user_id)

    monkeypatch.setattr(operation_env.service, "recover", counting_recover)

    await operation_env.service.ensure_recovered(USER_A)
    await operation_env.service.ensure_recovered(USER_A)

    assert calls == [USER_A]


@pytest.mark.asyncio
async def test_terminal_receipts_expire_after_their_ttl(operation_env):
    operation = await operation_env.start_and_wait()

    operation_env.clock.advance(seconds=3601)

    with pytest.raises(SkillOperationNotFoundError):
        operation_env.service.get_owned(USER_A, operation.operation_id)


@pytest.mark.asyncio
async def test_cleanup_expired_removes_only_expired_terminal_receipts(operation_env):
    terminal = operation_env.persist_operation(state="succeeded", operation_id="opold")
    running = operation_env.persist_operation(state="running", operation_id="oplive")
    operation_env.clock.advance(seconds=3601)

    removed = operation_env.service.cleanup_expired(USER_A)

    assert removed == 1
    assert operation_env.get(running).state == "running"
    with pytest.raises(SkillOperationNotFoundError):
        operation_env.service.get_owned(USER_A, terminal.operation_id)


@pytest.mark.asyncio
async def test_foreign_and_unknown_operations_are_indistinguishable(operation_env):
    operation = operation_env.persist_operation(state="running")

    with pytest.raises(SkillOperationNotFoundError):
        operation_env.service.get_owned(USER_B, operation.operation_id)
    with pytest.raises(SkillOperationNotFoundError):
        operation_env.service.get_owned(USER_A, "neverexisted")


@pytest.mark.asyncio
async def test_shutdown_cancels_in_flight_tasks(operation_env):
    gate = asyncio.Event()
    operation_env.installer.commit_gate = gate
    upload = operation_env.staged_upload()
    operation = await operation_env.service.start(USER_A, upload.upload_id, _request())

    await operation_env.service.shutdown()

    assert operation_env.service.get_owned(USER_A, operation.operation_id) is not None
    assert operation_env.installer.installed_hashes == []


@pytest.mark.asyncio
async def test_malformed_receipt_is_treated_as_missing(operation_env):
    operation_env.root.mkdir(parents=True, exist_ok=True)
    (operation_env.root / "operation-broken.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SkillOperationNotFoundError):
        operation_env.service.get_owned(USER_A, "broken")


def test_request_fingerprint_ignores_field_order(operation_env):
    first = SkillInstallationRequest(
        expected_source_hash=SOURCE_HASH,
        approve_setup=True,
        replace_source_hash="b" * 64,
    )
    second = SkillInstallationRequest.model_validate(
        {
            "replaceSourceHash": "b" * 64,
            "approveSetup": True,
            "expectedSourceHash": SOURCE_HASH,
        }
    )

    assert operation_env.service._fingerprint(first) == operation_env.service._fingerprint(second)


def test_request_fingerprint_distinguishes_setup_approval(operation_env):
    assert operation_env.service._fingerprint(
        _request(approve_setup=False)
    ) != operation_env.service._fingerprint(_request(approve_setup=True))


def test_operation_model_rejects_unknown_fields():
    with pytest.raises(ValueError):
        SkillInstallationRequest.model_validate(
            {"expectedSourceHash": SOURCE_HASH, "sneaky": True}
        )


def test_service_singleton_is_resettable():
    from client_backend.services.skill_runtime.operations import (
        get_skill_installation_service,
        reset_skill_installation_service,
    )

    first = get_skill_installation_service()
    assert get_skill_installation_service() is first
    reset_skill_installation_service()
    assert get_skill_installation_service() is not first
    reset_skill_installation_service()
