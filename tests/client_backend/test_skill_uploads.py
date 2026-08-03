"""Lifecycle, isolation, and quota behavior of the staged upload store.

A staged upload is untrusted content sitting in the user's profile. The rules
these tests pin down are what keep it harmless: it is validated once and never
executed, it belongs to exactly one user, it expires, it cannot crowd out the
disk, and a failed stage leaves nothing behind.
"""

from __future__ import annotations

import inspect
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from client_backend.schemas.skill_installation import (
    SkillArchivePreview,
    SkillArchiveSummary,
    SkillUploadRecord,
)
from client_backend.services.skill_runtime.uploads import (
    SkillUploadError,
    SkillUploadNotFoundError,
    SkillUploadService,
    SkillUploadStateError,
)


def _upload_globals() -> dict:
    """The upload module's globals, which is what the service resolves names in.

    Not ``sys.modules[...]``: ``test_document_upload_proxy_guard.py`` evicts every
    ``client_backend*`` module to prove an import boundary, so a later import can
    hand back a different module object than the one this file's classes already
    closed over. Patching that fresh object silently does nothing, and the
    ``sys.modules`` key may be gone entirely. ``__globals__`` is always the
    namespace the running code reads, so pair it with ``monkeypatch.setitem``.
    """
    return SkillUploadService.__init__.__globals__


def _live_settings():
    """The settings proxy the upload service resolves against."""
    return _upload_globals()["client_settings"]


USER_A = "user-a"
USER_B = "user-b"

SKILL_MD = "---\nname: demo\ndescription: Demo skill\n---\nInstructions body"


def _valid_skill_zip_bytes(name: str = "demo") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}/SKILL.md", SKILL_MD)
        archive.writestr(f"{name}/bin/demo.py", "print('ok')\n")
    return buffer.getvalue()


class _AsyncReader:
    """An UploadFile-shaped async stream that records how it was consumed."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0
        self.max_requested_chunk = 0
        self.read_calls = 0
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        self.max_requested_chunk = max(self.max_requested_chunk, size)
        if size < 0:
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


class _FrozenClock:
    def __init__(self) -> None:
        self._now = datetime(2026, 7, 31, 10, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, *, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class _EnvironmentStub:
    """Records whether anything asked to *prepare* a runtime. It must not."""

    def __init__(self) -> None:
        self.prepared: list[str] = []

    def preview(self, skill) -> dict:
        return {
            "python_project": False,
            "dependencies": [],
            "build_requirements": [],
            "declared_commands": [],
            "dependency_lock": None,
            "confirmation_required": False,
        }

    def prepare(self, skill, **_kwargs) -> dict:
        self.prepared.append(skill.name)
        return {"status": "ready", "commands": []}


class _RegistryStub:
    """Registry double that keeps real asset discovery.

    Only lookup and initialization are faked; ``_discover_executable_assets``
    delegates to the production static method so a staged preview reports the
    commands a real bundle would actually publish.
    """

    def __init__(self) -> None:
        self.skills: dict[str, object] = {}
        self.initialize_calls = 0

    async def initialize(self) -> None:
        self.initialize_calls += 1

    def get_skill(self, name: str):
        return self.skills.get(name)

    @staticmethod
    def _discover_executable_assets(bundle_root: Path) -> dict:
        from client_backend.services.local_skills_registry import LocalSkillsRegistry

        return LocalSkillsRegistry._discover_executable_assets(bundle_root)

    def _resolve_current_user_id(self) -> str:
        return USER_A


@dataclass
class _UploadEnv:
    service: SkillUploadService
    environment: _EnvironmentStub
    registry: _RegistryStub
    clock: _FrozenClock
    root: Path

    def stage_directories(self) -> list[str]:
        uploads_root = self.root
        if not uploads_root.is_dir():
            return []
        return sorted(path.name for path in uploads_root.iterdir() if path.is_dir())

    def persist_upload(self, *, owner: str, upload_id: str = "upload-a") -> SkillUploadRecord:
        record = SkillUploadRecord(
            upload_id=upload_id,
            owner=owner,
            created_at=self.clock(),
            expires_at=self.clock() + timedelta(seconds=1800),
            archive=SkillArchiveSummary(
                filename="demo.zip",
                compressed_bytes=100,
                expanded_bytes=200,
                file_count=2,
            ),
            preview=SkillArchivePreview(
                name="demo",
                source_hash="a" * 64,
                bundle_shape="nested",
                executable_assets={},
                setup={},
            ),
        )
        self.service.persist_for_test(record)
        return record

    def persist_uploads(self, owner: str, *, count: int, bytes_each: int) -> None:
        for index in range(count):
            record = self.persist_upload(owner=owner, upload_id=f"upload-{index}")
            directory = self.service.staging_dir_for_test(owner, record.upload_id)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "archive.zip").write_bytes(b"x" * bytes_each)


@pytest.fixture()
def upload_env(tmp_path, monkeypatch) -> _UploadEnv:
    uploads = _upload_globals()
    locks = inspect.unwrap(uploads["profile_lock"]).__globals__

    roots: dict[str, Path] = {}

    def fake_uploads_root(user_id: str) -> Path:
        if user_id in {"", ".", ".."} or "/" in user_id or "\\" in user_id:
            raise ValueError("user_id must be a non-empty path component")
        roots[user_id] = tmp_path / user_id / "uploads"
        return roots[user_id]

    monkeypatch.setitem(uploads, "get_skill_uploads_root", fake_uploads_root)
    monkeypatch.setitem(locks, "get_skill_locks_root", lambda user_id: tmp_path / "locks")

    clock = _FrozenClock()
    environment = _EnvironmentStub()
    registry = _RegistryStub()
    service = SkillUploadService(
        environment_manager=environment,
        registry=registry,
        clock=clock,
        audit=None,
    )
    return _UploadEnv(
        service=service,
        environment=environment,
        registry=registry,
        clock=clock,
        root=tmp_path / USER_A / "uploads",
    )


@pytest.mark.asyncio
async def test_stage_streams_once_previews_without_setup_and_persists_record(upload_env):
    stream = _AsyncReader(_valid_skill_zip_bytes())

    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="../../calendar.zip",
        stream=stream,
    )

    assert record.state == "staged"
    assert record.preview.name == "demo"
    assert record.preview.source_hash
    # The uploaded name is sanitized to a bare basename before it is echoed, and
    # is never used to build a path.
    assert record.archive.filename == "calendar.zip"
    assert record.archive.file_count == 2
    assert stream.max_requested_chunk <= 1024 * 1024
    assert upload_env.environment.prepared == []
    assert upload_env.service.get_owned(USER_A, record.upload_id).upload_id == record.upload_id


@pytest.mark.asyncio
async def test_stage_reports_existing_replaceable_skill(upload_env, monkeypatch):
    from types import SimpleNamespace

    uploads = _upload_globals()

    installed_root = upload_env.root.parent / "installed"
    monkeypatch.setitem(uploads, "get_installed_skills_root", lambda user_id: installed_root)
    bundle = installed_root / "demo-abc"
    bundle.mkdir(parents=True)
    upload_env.registry.skills["demo"] = SimpleNamespace(
        name="demo",
        source_hash="b" * 64,
        bundle_root=bundle,
        enabled=True,
    )

    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    assert record.preview.existing_skill.replaceable is True
    assert record.preview.existing_skill.install_source == "profile"
    assert record.preview.existing_skill.source_hash == "b" * 64


@pytest.mark.asyncio
async def test_stage_marks_configured_root_collision_not_replaceable(upload_env, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(
        _upload_globals(),
        "get_installed_skills_root",
        lambda user_id: upload_env.root.parent / "installed",
    )
    upload_env.registry.skills["demo"] = SimpleNamespace(
        name="demo",
        source_hash="c" * 64,
        bundle_root=upload_env.root.parent / "user-configured" / "demo",
        enabled=False,
    )

    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    assert record.preview.existing_skill.replaceable is False
    assert record.preview.existing_skill.install_source == "configured_root"


@pytest.mark.asyncio
async def test_api_payload_hides_owner_and_fingerprint(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    payload = record.to_api()

    assert "owner" not in payload
    assert "requestFingerprint" not in payload
    assert payload["uploadId"] == record.upload_id
    assert payload["preview"]["sourceHash"] == record.preview.source_hash
    assert "source_hash" not in payload["preview"]
    assert payload["preview"]["executableAssets"]["pythonProject"] is False
    assert payload["preview"]["setup"]["confirmationRequired"] is False
    assert payload["archive"]["compressedBytes"] > 0


def test_foreign_and_expired_uploads_are_indistinguishable(upload_env):
    record = upload_env.persist_upload(owner=USER_A)

    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned(USER_B, record.upload_id)

    upload_env.clock.advance(seconds=1801)
    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned(USER_A, record.upload_id)


def test_unknown_upload_is_not_found(upload_env):
    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned(USER_A, "never-existed")


@pytest.mark.parametrize(
    ("filename", "payload", "code"),
    [
        ("", b"PK", "SKILL_ARCHIVE_TYPE_UNSUPPORTED"),
        ("demo.tar", b"PK", "SKILL_ARCHIVE_TYPE_UNSUPPORTED"),
        ("demo.zip.exe", b"PK", "SKILL_ARCHIVE_TYPE_UNSUPPORTED"),
        ("demo.ZIP", b"not a zip container", "SKILL_ARCHIVE_INVALID"),
        ("demo.zip", b"", "SKILL_ARCHIVE_INVALID"),
    ],
)
@pytest.mark.asyncio
async def test_rejects_invalid_upload_input_and_removes_partial_stage(
    upload_env, filename, payload, code
):
    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename=filename,
            stream=_AsyncReader(payload),
        )

    assert exc_info.value.code == code
    assert upload_env.stage_directories() == []


@pytest.mark.asyncio
async def test_rejects_upload_larger_than_the_configured_limit(upload_env, monkeypatch):
    monkeypatch.setattr(_live_settings(), "skill_upload_max_bytes", 1024)

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="demo.zip",
            stream=_AsyncReader(b"x" * 2048),
        )

    assert exc_info.value.code == "SKILL_ARCHIVE_TOO_LARGE"
    assert exc_info.value.status_code == 413
    assert upload_env.stage_directories() == []


@pytest.mark.asyncio
async def test_oversized_upload_stops_reading_early(upload_env, monkeypatch):
    """The limit is enforced while streaming, not after buffering the whole body."""
    monkeypatch.setattr(_live_settings(), "skill_upload_max_bytes", 1024)
    stream = _AsyncReader(b"x" * (8 * 1024 * 1024))

    with pytest.raises(SkillUploadError):
        await upload_env.service.stage(user_id=USER_A, filename="demo.zip", stream=stream)

    assert stream.read_calls <= 2


@pytest.mark.asyncio
async def test_rejects_bundle_without_exactly_one_skill(upload_env):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("one/SKILL.md", SKILL_MD)
        archive.writestr("two/SKILL.md", SKILL_MD)

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="two.zip",
            stream=_AsyncReader(buffer.getvalue()),
        )

    assert exc_info.value.code == "SKILL_BUNDLE_INVALID"
    assert upload_env.stage_directories() == []


@pytest.mark.asyncio
async def test_outstanding_upload_quota_is_profile_scoped(upload_env):
    upload_env.persist_uploads(USER_A, count=5, bytes_each=1024)

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="sixth.zip",
            stream=_AsyncReader(_valid_skill_zip_bytes()),
        )

    assert exc_info.value.code == "SKILL_UPLOAD_QUOTA_EXCEEDED"
    assert exc_info.value.status_code == 413

    # The other user's allowance is untouched.
    other = await upload_env.service.stage(
        user_id=USER_B,
        filename="first.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    assert other.state == "staged"


@pytest.mark.asyncio
async def test_storage_byte_quota_is_enforced(upload_env, monkeypatch):
    monkeypatch.setattr(_live_settings(), "skill_upload_quota_bytes", 4096)
    upload_env.persist_uploads(USER_A, count=2, bytes_each=3000)

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="third.zip",
            stream=_AsyncReader(_valid_skill_zip_bytes()),
        )

    assert exc_info.value.code == "SKILL_UPLOAD_QUOTA_EXCEEDED"


@pytest.mark.asyncio
async def test_rate_limit_counts_accepted_and_rejected_attempts(upload_env, monkeypatch):
    monkeypatch.setattr(_live_settings(), "skill_upload_rate_limit_count", 10)
    monkeypatch.setattr(_live_settings(), "skill_upload_max_outstanding", 100)

    for index in range(10):
        payload = _valid_skill_zip_bytes() if index % 2 == 0 else b"not a zip"
        try:
            await upload_env.service.stage(
                user_id=USER_A,
                filename="demo.zip",
                stream=_AsyncReader(payload),
            )
        except SkillUploadError as exc:
            assert exc.code == "SKILL_ARCHIVE_INVALID"

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="eleventh.zip",
            stream=_AsyncReader(_valid_skill_zip_bytes()),
        )

    assert exc_info.value.code == "SKILL_UPLOAD_QUOTA_EXCEEDED"
    assert exc_info.value.retryable is True

    upload_env.clock.advance(seconds=61)
    accepted = await upload_env.service.stage(
        user_id=USER_A,
        filename="after-window.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    assert accepted.state == "staged"


@pytest.mark.asyncio
async def test_insufficient_disk_space_maps_to_storage_error(upload_env, monkeypatch):
    def no_space(_path):
        raise OSError(28, "No space left on device")

    monkeypatch.setitem(_upload_globals(), "available_bytes", no_space)

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="demo.zip",
            stream=_AsyncReader(_valid_skill_zip_bytes()),
        )

    assert exc_info.value.code == "SKILL_STORAGE_INSUFFICIENT"
    assert exc_info.value.status_code == 507
    assert exc_info.value.retryable is True
    assert upload_env.stage_directories() == []


@pytest.mark.asyncio
async def test_enospc_while_writing_maps_to_storage_error(upload_env, monkeypatch):
    monkeypatch.setitem(_upload_globals(), "available_bytes", lambda _path: 1 << 40)

    class _FullDisk(_AsyncReader):
        async def read(self, size: int = -1) -> bytes:
            raise OSError(28, "No space left on device")

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="demo.zip",
            stream=_FullDisk(b""),
        )

    assert exc_info.value.code == "SKILL_STORAGE_INSUFFICIENT"
    assert upload_env.stage_directories() == []


@pytest.mark.asyncio
async def test_delete_removes_bytes_and_record(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    upload_env.service.delete(USER_A, record.upload_id)

    assert upload_env.stage_directories() == []
    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned(USER_A, record.upload_id)


@pytest.mark.asyncio
async def test_delete_is_blocked_while_installing(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    upload_env.service.claim_for_operation(
        USER_A,
        record.upload_id,
        request_fingerprint="fingerprint-a",
        operation_id="operation-a",
    )

    with pytest.raises(SkillUploadStateError) as exc_info:
        upload_env.service.delete(USER_A, record.upload_id)

    assert exc_info.value.code == "SKILL_UPLOAD_STATE_INVALID"


@pytest.mark.asyncio
async def test_delete_rejects_a_foreign_upload(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.delete(USER_B, record.upload_id)

    assert upload_env.service.get_owned(USER_A, record.upload_id).state == "staged"


@pytest.mark.asyncio
async def test_claim_is_idempotent_for_the_same_fingerprint(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    first = upload_env.service.claim_for_operation(
        USER_A, record.upload_id, request_fingerprint="fp", operation_id="operation-a"
    )
    second = upload_env.service.claim_for_operation(
        USER_A, record.upload_id, request_fingerprint="fp", operation_id="operation-a"
    )

    assert first.operation_id == second.operation_id == "operation-a"
    assert second.state == "installing"


@pytest.mark.asyncio
async def test_claim_rejects_a_different_request_for_a_claimed_upload(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    upload_env.service.claim_for_operation(
        USER_A, record.upload_id, request_fingerprint="fp-1", operation_id="operation-a"
    )

    with pytest.raises(SkillUploadError) as exc_info:
        upload_env.service.claim_for_operation(
            USER_A, record.upload_id, request_fingerprint="fp-2", operation_id="operation-b"
        )

    assert exc_info.value.code == "SKILL_UPLOAD_CONSUMED"
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_mark_succeeded_discards_bytes_but_keeps_the_record(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    upload_env.service.claim_for_operation(
        USER_A, record.upload_id, request_fingerprint="fp", operation_id="operation-a"
    )

    upload_env.service.mark_succeeded(USER_A, record.upload_id)

    stored = upload_env.service.get_owned(USER_A, record.upload_id)
    assert stored.state == "installed"
    staging = upload_env.service.staging_dir_for_test(USER_A, record.upload_id)
    assert not any(staging.rglob("*.zip"))
    assert not (staging / "extracted").exists()


@pytest.mark.asyncio
async def test_cleanup_expired_removes_stale_uploads_only(upload_env):
    fresh = await upload_env.service.stage(
        user_id=USER_A,
        filename="fresh.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    upload_env.clock.advance(seconds=1801)
    newer = await upload_env.service.stage(
        user_id=USER_A,
        filename="newer.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    removed = upload_env.service.cleanup_expired(USER_A)

    assert removed == 1
    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned(USER_A, fresh.upload_id)
    assert upload_env.service.get_owned(USER_A, newer.upload_id).upload_id == newer.upload_id


@pytest.mark.asyncio
async def test_ensure_recovered_drops_expired_and_orphaned_stages(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    orphan = upload_env.root / "orphaned-stage"
    orphan.mkdir(parents=True)
    (orphan / "archive.zip").write_bytes(b"junk")
    upload_env.clock.advance(seconds=1801)

    await upload_env.service.ensure_recovered(USER_A)

    with pytest.raises(SkillUploadNotFoundError):
        upload_env.service.get_owned(USER_A, record.upload_id)
    assert not orphan.exists()


@pytest.mark.asyncio
async def test_recovered_records_survive_a_new_service_instance(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    reloaded = SkillUploadService(
        environment_manager=upload_env.environment,
        registry=upload_env.registry,
        clock=upload_env.clock,
        audit=None,
    )

    assert reloaded.get_owned(USER_A, record.upload_id).preview.name == "demo"


@pytest.mark.asyncio
async def test_extracted_bundle_root_strips_one_wrapper_directory(upload_env):
    """Zipping a folder is how these archives are made, and it adds a wrapper.

    Asset discovery looks for ``bin/`` directly beneath the bundle root, so
    treating the extraction directory as the root would install a skill whose
    commands are invisible.
    """
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    bundle_root = upload_env.service.extracted_root(USER_A, record.upload_id)

    assert bundle_root.name == "demo"
    assert (bundle_root / "SKILL.md").is_file()
    assert (bundle_root / "bin" / "demo.py").is_file()
    assert record.preview.executable_assets.bin == ["demo.py"]


@pytest.mark.asyncio
async def test_a_flat_archive_keeps_its_extraction_root(upload_env):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("SKILL.md", SKILL_MD)
        archive.writestr("bin/demo.py", "print('ok')\n")

    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="flat.zip",
        stream=_AsyncReader(buffer.getvalue()),
    )

    bundle_root = upload_env.service.extracted_root(USER_A, record.upload_id)

    assert bundle_root.name == "extracted"
    assert record.preview.executable_assets.bin == ["demo.py"]


@pytest.mark.asyncio
async def test_a_multi_entry_root_is_never_stripped(upload_env):
    """A bundle with bin/ beside skills/ already has its real root."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("skills/demo/SKILL.md", SKILL_MD)
        archive.writestr("bin/demo.py", "print('ok')\n")

    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="nested.zip",
        stream=_AsyncReader(buffer.getvalue()),
    )

    bundle_root = upload_env.service.extracted_root(USER_A, record.upload_id)

    assert bundle_root.name == "extracted"
    assert record.preview.bundle_shape == "nested"
    assert record.preview.executable_assets.bin == ["demo.py"]


@pytest.mark.asyncio
async def test_lifecycle_audit_records_accepted_and_rejected_uploads(upload_env, tmp_path):
    audit_path = tmp_path / "lifecycle.jsonl"
    writer_cls = _upload_globals()["SkillLifecycleAuditWriter"]
    upload_env.service._audit = writer_cls(path=audit_path)

    await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )
    with pytest.raises(SkillUploadError):
        await upload_env.service.stage(
            user_id=USER_A,
            filename="demo.tar",
            stream=_AsyncReader(b"PK"),
        )

    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2
    assert "demo.zip" not in audit_path.read_text(encoding="utf-8")


# --- Collections ------------------------------------------------------------


def _collection_zip(skills: dict[str, str], *, manifest: dict | None = None) -> bytes:
    """An archive shaped like a downloaded skill repository."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        if manifest is not None:
            archive.writestr("library/.claude-plugin/plugin.json", json.dumps(manifest))
        archive.writestr("library/README.md", "A library of skills.")
        for name, body in skills.items():
            archive.writestr(
                f"library/skills/{name}/SKILL.md",
                f"---\nname: {name}\ndescription: {name}\n---\n{body}",
            )
            archive.writestr(f"library/skills/{name}/notes.md", f"Notes for {name}.")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_stages_every_skill_in_a_library(upload_env):
    payload = _collection_zip(
        {"brainstorming": "Explore first.", "systematic-debugging": "Find the root cause."},
        manifest={"name": "superpowers", "version": "6.2.0", "description": "Core skills"},
    )

    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="superpowers-main.zip",
        stream=_AsyncReader(payload),
    )

    assert record.collection.name == "superpowers"
    assert record.collection.version == "6.2.0"
    assert record.collection.skill_count == 2
    assert [skill.name for skill in record.skills] == [
        "brainstorming",
        "systematic-debugging",
    ]
    # The first skill doubles as `preview`, so a single-skill client still works.
    assert record.preview.name == "brainstorming"


@pytest.mark.asyncio
async def test_a_single_skill_upload_still_reports_one_collection_of_one(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="demo.zip",
        stream=_AsyncReader(_valid_skill_zip_bytes()),
    )

    assert record.collection.skill_count == 1
    assert record.collection.version is None
    assert [skill.name for skill in record.skills] == ["demo"]
    assert record.preview.name == "demo"


@pytest.mark.asyncio
async def test_collection_previews_report_each_skills_own_collision(upload_env, monkeypatch):
    from types import SimpleNamespace

    installed_root = upload_env.root.parent / "installed"
    monkeypatch.setitem(
        _upload_globals(),
        "get_installed_skills_root",
        lambda user_id: installed_root,
    )
    bundle = installed_root / "brainstorming-abc"
    bundle.mkdir(parents=True)
    upload_env.registry.skills["brainstorming"] = SimpleNamespace(
        name="brainstorming",
        source_hash="c" * 64,
        bundle_root=bundle,
        enabled=True,
    )

    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="library.zip",
        stream=_AsyncReader(_collection_zip({"brainstorming": "a", "other": "b"})),
    )

    by_name = {skill.name: skill for skill in record.skills}
    assert by_name["brainstorming"].existing_skill.replaceable is True
    assert by_name["brainstorming"].existing_skill.source_hash == "c" * 64
    assert by_name["other"].existing_skill is None


@pytest.mark.asyncio
async def test_skill_roots_follow_the_previewed_order(upload_env):
    record = await upload_env.service.stage(
        user_id=USER_A,
        filename="library.zip",
        stream=_AsyncReader(_collection_zip({"alpha": "a", "zulu": "z"})),
    )

    roots = upload_env.service.skill_roots(USER_A, record.upload_id)

    assert [name for name, _ in roots] == [skill.name for skill in record.skills]
    for name, root in roots:
        assert (root / "SKILL.md").is_file()
        assert root.name == name


@pytest.mark.asyncio
async def test_rejects_an_archive_whose_skills_share_a_name(upload_env):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("first", "second"):
            archive.writestr(
                f"library/skills/{folder}/SKILL.md",
                "---\nname: duplicated\ndescription: d\n---\nBody",
            )

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="library.zip",
            stream=_AsyncReader(buffer.getvalue()),
        )

    assert exc_info.value.code == "SKILL_BUNDLE_INVALID"
    assert "duplicated" in exc_info.value.message


@pytest.mark.asyncio
async def test_rejects_an_archive_with_more_skills_than_allowed(upload_env, monkeypatch):
    """A tree with hundreds of SKILL.md files is a workspace, not a library."""
    monkeypatch.setitem(
        _upload_globals(),
        "MAX_SKILLS_PER_COLLECTION",
        2,
    )

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="huge.zip",
            stream=_AsyncReader(_collection_zip({"a": "1", "b": "2", "c": "3"})),
        )

    assert exc_info.value.code == "SKILL_BUNDLE_INVALID"
    assert "more than" in exc_info.value.message


@pytest.mark.asyncio
async def test_an_archive_with_no_skill_says_so(upload_env):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("library/README.md", "nothing installable here")

    with pytest.raises(SkillUploadError) as exc_info:
        await upload_env.service.stage(
            user_id=USER_A,
            filename="empty.zip",
            stream=_AsyncReader(buffer.getvalue()),
        )

    assert exc_info.value.code == "SKILL_BUNDLE_INVALID"
    assert "no SKILL.md" in exc_info.value.message
