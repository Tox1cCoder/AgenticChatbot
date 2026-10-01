from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.api.documents import _stage_create_and_enqueue_document
from app.services.document_processing_service import DocumentProcessingService


def _service(tmp_path):
    service = object.__new__(DocumentProcessingService)
    service.settings = SimpleNamespace(temp_storage_path=str(tmp_path), max_file_size_mb=1)
    return service


class _Upload:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    async def seek(self, offset):
        assert offset == 0

    async def read(self, size):
        assert size == 1024 * 1024
        return next(self.chunks, b"")


@pytest.mark.asyncio
async def test_staging_disk_operations_leave_the_event_loop(tmp_path, monkeypatch):
    loop_thread = threading.get_ident()
    calls = []

    class RecordedFile:
        def __init__(self, file):
            self.file = file

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            calls.append(("close", threading.get_ident()))
            return self.file.__exit__(*args)

        def write(self, chunk):
            calls.append(("write", threading.get_ident()))
            return self.file.write(chunk)

        def read(self):
            return self.file.read()

    for name in ("mkdir", "open", "unlink"):
        original = getattr(Path, name)

        def record(path, *args, _name=name, _original=original, **kwargs):
            calls.append((_name, threading.get_ident()))
            result = _original(path, *args, **kwargs)
            return RecordedFile(result) if _name == "open" else result

        monkeypatch.setattr(Path, name, record)

    service = _service(tmp_path / "staging")
    result = await service.stage_upload_file(_Upload([b"first", b"second"]), "notes.txt")
    path = Path(result["temp_file_path"])
    disk_calls = calls.copy()
    assert path.read_bytes() == b"firstsecond"
    assert result["file_size"] == 11
    assert disk_calls
    assert all(thread != loop_thread for _, thread in disk_calls)

    calls.clear()
    with pytest.raises(ValueError, match="exceeds maximum"):
        await service.stage_upload_file(_Upload([b"x" * (1024 * 1024), b"y"]), "big.txt")
    assert all(thread != loop_thread for _, thread in calls)
    assert list((tmp_path / "staging").iterdir()) == [path]


@pytest.mark.asyncio
async def test_read_failure_removes_the_partial_upload(tmp_path):
    class FailingUpload(_Upload):
        async def read(self, size):
            chunk = await super().read(size)
            if not chunk:
                raise OSError("upload disconnected")
            return chunk

    with pytest.raises(OSError, match="disconnected"):
        await _service(tmp_path).stage_upload_file(FailingUpload([b"partial"]), "notes.txt")
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_route_cancellation_before_enqueue_removes_the_staged_file(tmp_path):
    staged = tmp_path / "staged.txt"
    staged.write_bytes(b"upload")
    validating = asyncio.Event()

    async def validate(**kwargs):
        validating.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        _stage_create_and_enqueue_document(
            document_service=SimpleNamespace(validate_and_create_document=validate),
            document_processing_service=SimpleNamespace(
                stage_upload_file=AsyncMock(
                    return_value={"temp_file_path": str(staged), "file_size": 6}
                )
            ),
            file=SimpleNamespace(filename="notes.txt", content_type="text/plain"),
            conversation_id=uuid4(),
        )
    )
    await asyncio.wait_for(validating.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not staged.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("enqueue_fails", [False, True])
async def test_route_cancellation_waits_for_enqueue_ownership(tmp_path, enqueue_fails):
    staged = tmp_path / "staged.txt"
    staged.write_bytes(b"upload")
    enqueuing, release = asyncio.Event(), asyncio.Event()

    async def enqueue(*args):
        enqueuing.set()
        await release.wait()
        if enqueue_fails:
            raise OSError("broker unavailable")
        return {"task_id": "queued"}

    task = asyncio.create_task(
        _stage_create_and_enqueue_document(
            document_service=SimpleNamespace(
                validate_and_create_document=AsyncMock(return_value=SimpleNamespace(id=uuid4()))
            ),
            document_processing_service=SimpleNamespace(
                stage_upload_file=AsyncMock(
                    return_value={"temp_file_path": str(staged), "file_size": 6}
                ),
                start_processing_task=enqueue,
            ),
            file=SimpleNamespace(filename="notes.txt", content_type="text/plain"),
            conversation_id=uuid4(),
        )
    )
    try:
        await asyncio.wait_for(enqueuing.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "Cancellation must await the broker ownership decision"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert staged.exists() is (not enqueue_fails)


@pytest.mark.asyncio
async def test_cancelled_upload_removes_the_partial_file(tmp_path):
    reading = asyncio.Event()

    class WaitingUpload(_Upload):
        async def read(self, size):
            chunk = await super().read(size)
            if not chunk:
                reading.set()
                await asyncio.Event().wait()
            return chunk

    task = asyncio.create_task(
        _service(tmp_path).stage_upload_file(WaitingUpload([b"partial"]), "notes.txt")
    )
    await asyncio.wait_for(reading.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_the_disk_write(tmp_path):
    started, release = threading.Event(), threading.Event()
    service = _service(tmp_path)

    def blocked_write(path, chunk):
        started.set()
        release.wait(timeout=3)
        with path.open("ab") as file:
            file.write(chunk)

    service._append_staged_upload = blocked_write
    task = asyncio.create_task(service.stage_upload_file(_Upload([b"payload"]), "notes.txt"))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done(), "A second cancellation must not abandon the disk worker"
        assert len(list(tmp_path.iterdir())) == 1
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_repeated_route_cancellation_keeps_the_running_broker_input(tmp_path):
    staged = tmp_path / "staged.txt"
    staged.write_bytes(b"upload")
    started, release = threading.Event(), threading.Event()
    service = _service(tmp_path)

    def blocked_enqueue(*args):
        started.set()
        release.wait(timeout=3)
        return SimpleNamespace(id="queued")

    service._enqueue_processing_chain = blocked_enqueue
    service.stage_upload_file = AsyncMock(
        return_value={"temp_file_path": str(staged), "file_size": 6}
    )
    task = asyncio.create_task(
        _stage_create_and_enqueue_document(
            document_service=SimpleNamespace(
                validate_and_create_document=AsyncMock(return_value=SimpleNamespace(id=uuid4()))
            ),
            document_processing_service=service,
            file=SimpleNamespace(filename="notes.txt", content_type="text/plain"),
            conversation_id=uuid4(),
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 1)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done(), "A second cancellation must not abandon the broker worker"
        assert staged.exists(), "The broker still owns its staged input"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert staged.exists()
