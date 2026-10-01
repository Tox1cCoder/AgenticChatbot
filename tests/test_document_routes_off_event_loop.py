"""Document routes keep their blocking I/O off the event loop.

Every document handler is ``async def``, and so is every ``DocumentService``
method it awaits, but the repository underneath is synchronous: a database
round trip made from those coroutines blocked the loop, and with it every
other request and every in-flight stream. The access checks the handlers make
first were synchronous too, and so are the Celery result backend behind the
task-status route and the broker publish behind an upload.

Each test records the thread the blocking call ran on and compares it with the
loop's own thread.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from dependency_injector import providers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.documents import router as documents_router
from app.core.auth import get_current_user_id
from app.core.container import Container, setup_auto_injection
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService
from app.utils.exception_handler import register_exception_handlers
from tests.token_state_stub import stub_token_states


@pytest.fixture(autouse=True)
def _restore_wiring():
    setup_auto_injection(Container)
    yield
    setup_auto_injection(Container)


@pytest.fixture(autouse=True)
def _live_token_users(monkeypatch):
    stub_token_states(monkeypatch)


class _Threads:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def record(self, name: str) -> None:
        self.calls.append((name, threading.get_ident()))


def _document(conversation_id=None):
    return SimpleNamespace(
        id=uuid4(),
        conversation_id=conversation_id or uuid4(),
        filename="notes.txt",
        file_type="text/plain",
        status=1,
        upload_time=datetime.now(UTC),
    )


class _Repository:
    """The synchronous repository, recording where each call ran."""

    session_factory = None

    def __init__(self, threads: _Threads) -> None:
        self._threads = threads

    def _ran(self, name: str, result=None):
        self._threads.record(f"repository.{name}")
        return result

    def get_by_id(self, _document_id):
        return self._ran("get_by_id", _document())

    def update(self, _document_id, _data):
        return self._ran("update", _document())

    def delete(self, _document_id):
        return self._ran("delete", True)

    def get_by_conversation_id(self, conversation_id, _page, _page_size):
        return self._ran("get_by_conversation_id", ([_document(conversation_id)], 1))

    def get_by_conversation_and_filename_key(self, _conversation_id, _key):
        return self._ran("get_by_conversation_and_filename_key", None)

    def create(self, data):
        return self._ran("create", _document(data.conversation_id))

    def set_processing_task_id(self, _document_id, _task_id):
        return self._ran("set_processing_task_id", _document())


class _Index:
    def __init__(self, threads: _Threads) -> None:
        self._threads = threads

    def delete_document_index(self, _document_id):
        self._threads.record("index.delete_document_index")


class _Access:
    """Both validation utilities: one synchronous ownership query each."""

    threads: _Threads

    def __init__(self, _session_factory) -> None:
        pass

    def validate_document_access(self, _user_id, _document_id):
        self.threads.record("validate_document_access")

    def validate_conversation_access(self, _user_id, _conversation_id):
        self.threads.record("validate_conversation_access")


class _Signature:
    def __init__(self, threads: _Threads) -> None:
        self._threads = threads


class _Chain:
    def __init__(self, threads: _Threads) -> None:
        self._threads = threads

    def apply_async(self, **_kwargs):
        self._threads.record("celery.apply_async")
        return SimpleNamespace(id="task-1")


class _AsyncResult:
    def __init__(self, threads: _Threads) -> None:
        self._threads = threads

    @property
    def status(self):
        self._threads.record("celery.AsyncResult.status")
        return "SUCCESS"

    def ready(self):
        self._threads.record("celery.AsyncResult.ready")
        return True

    @property
    def result(self):
        return {"success": True}

    info = None


def _processing_service(threads: _Threads, tmp_path) -> DocumentProcessingService:
    service = DocumentProcessingService.__new__(DocumentProcessingService)
    service.settings = SimpleNamespace(max_file_size_mb=5, temp_storage_path=str(tmp_path))
    service.celery_app = SimpleNamespace(
        AsyncResult=lambda _task_id: _AsyncResult(threads),
        signature=lambda *_args, **_kwargs: _Signature(threads),
    )
    return service


@pytest.fixture
def routes(monkeypatch, tmp_path):
    """The real documents router and ``DocumentService`` over recording doubles."""
    import celery

    import app.api.documents as documents_api

    threads = _Threads()
    _Access.threads = threads
    monkeypatch.setattr(documents_api, "DocumentValidationUtils", _Access)
    monkeypatch.setattr(documents_api, "ConversationValidationUtils", _Access)
    monkeypatch.setattr(celery, "chain", lambda *_signatures: _Chain(threads))
    monkeypatch.chdir(tmp_path)

    processing = _processing_service(threads, tmp_path)
    document_service = DocumentService(
        document_repository=_Repository(threads),
        document_processing_service=processing,
        document_validation_utils=None,
        document_index_service=_Index(threads),
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(documents_router)
    app.dependency_overrides[get_current_user_id] = lambda: uuid4()

    @app.get("/loop-thread")
    async def loop_thread():
        return {"ident": threading.get_ident()}

    with (
        Container.document_service.override(providers.Object(document_service)),
        Container.document_processing_service.override(providers.Object(processing)),
        TestClient(app) as client,
    ):
        loop_ident = client.get("/loop-thread").json()["ident"]
        yield client, threads, loop_ident


def _assert_off_the_loop(threads: _Threads, loop_ident: int, *expected: str) -> None:
    names = [name for name, _ident in threads.calls]
    for name in expected:
        assert name in names, f"{name} never ran: {names}"
    on_loop = [name for name, ident in threads.calls if ident == loop_ident]
    assert not on_loop, f"blocking calls ran on the event loop: {on_loop}"


def test_reading_a_document_runs_its_queries_off_the_event_loop(routes):
    client, threads, loop_ident = routes

    response = client.get(f"/documents/{uuid4()}")

    assert response.status_code == 200
    _assert_off_the_loop(threads, loop_ident, "validate_document_access", "repository.get_by_id")


def test_listing_documents_runs_its_queries_off_the_event_loop(routes):
    client, threads, loop_ident = routes

    response = client.get(f"/documents/conversation/{uuid4()}")

    assert response.status_code == 200
    assert response.json()["data"]["total"] == 1
    _assert_off_the_loop(
        threads, loop_ident, "validate_conversation_access", "repository.get_by_conversation_id"
    )


def test_updating_a_document_runs_its_queries_off_the_event_loop(routes):
    client, threads, loop_ident = routes

    response = client.put(f"/documents/{uuid4()}", json={"filename": "renamed.txt"})

    assert response.status_code == 200
    _assert_off_the_loop(threads, loop_ident, "validate_document_access", "repository.update")


def test_deleting_a_document_runs_its_cleanup_off_the_event_loop(routes):
    client, threads, loop_ident = routes

    response = client.delete(f"/documents/{uuid4()}")

    assert response.status_code == 200
    _assert_off_the_loop(
        threads,
        loop_ident,
        "validate_document_access",
        "repository.get_by_id",
        "index.delete_document_index",
        "repository.delete",
    )


def test_an_upload_runs_its_queries_and_broker_publish_off_the_event_loop(routes):
    client, threads, loop_ident = routes

    response = client.post(
        "/documents/upload",
        data={"conversation_id": str(uuid4())},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 201, response.json()
    _assert_off_the_loop(
        threads,
        loop_ident,
        "validate_conversation_access",
        "repository.get_by_conversation_and_filename_key",
        "repository.create",
        "celery.apply_async",
        "repository.set_processing_task_id",
    )


def test_task_status_reads_the_result_backend_off_the_event_loop(routes, monkeypatch):
    import app.api.documents as documents_api

    client, threads, loop_ident = routes
    monkeypatch.setattr(
        documents_api, "_require_owned_task_document", lambda *_args: threads.record("owner")
    )

    response = client.get("/documents/task/task-1")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "SUCCESS"
    _assert_off_the_loop(
        threads, loop_ident, "celery.AsyncResult.status", "celery.AsyncResult.ready"
    )
