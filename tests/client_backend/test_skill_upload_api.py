"""Wire contract for the ZIP upload and installation routes.

What a browser depends on, end to end: multipart in, camelCase out, a coded
failure envelope it can branch on, and identical behavior under ``/skills`` and
the ``/api/skills`` compatibility alias.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from client_backend.api import skills as skills_api
from client_backend.api.skill_errors import (
    SKILL_ERROR_STATUS,
    register_skill_exception_handlers,
)
from client_backend.schemas.skill_installation import (
    SkillArchivePreview,
    SkillArchiveSummary,
    SkillInstallationOperationModel,
    SkillInstallationResult,
    SkillUploadRecord,
)

USER_ID = "user-a"
SOURCE_HASH = "a" * 64
SKILL_MD = "---\nname: demo\ndescription: Demo skill\n---\nBody"


def _valid_skill_zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("demo/SKILL.md", SKILL_MD)
    return buffer.getvalue()


def _upload_record(**overrides) -> SkillUploadRecord:
    now = datetime(2026, 7, 31, 10, 0, tzinfo=timezone.utc)
    payload = {
        "upload_id": "upload-a",
        "owner": USER_ID,
        "created_at": now,
        "expires_at": now + timedelta(seconds=1800),
        "archive": SkillArchiveSummary(
            filename="demo.zip",
            compressed_bytes=1024,
            expanded_bytes=2048,
            file_count=2,
        ),
        "preview": SkillArchivePreview(
            name="demo",
            source_hash=SOURCE_HASH,
            bundle_shape="nested",
            executable_assets={"bin": ["demo.py"], "scripts": [], "python_project": False},
            setup={"python_project": False, "confirmation_required": False},
        ),
    }
    payload.update(overrides)
    return SkillUploadRecord(**payload)


def _operation(**overrides) -> SkillInstallationOperationModel:
    now = datetime(2026, 7, 31, 10, 1, tzinfo=timezone.utc)
    payload = {
        "operation_id": "operationa",
        "upload_id": "upload-a",
        "owner": USER_ID,
        "created_at": now,
        "expires_at": now + timedelta(seconds=3600),
    }
    payload.update(overrides)
    return SkillInstallationOperationModel(**payload)


class _UploadServiceStub:
    def __init__(self) -> None:
        self.staged: list[tuple[str, str, bytes]] = []
        self.deleted: list[tuple[str, str]] = []
        self.error: Exception | None = None
        self.recovered: list[str] = []

    async def ensure_recovered(self, user_id: str) -> None:
        self.recovered.append(user_id)

    async def stage(self, *, user_id: str, filename: str, stream):
        if self.error is not None:
            raise self.error
        body = await stream.read()
        self.staged.append((user_id, filename, body))
        return _upload_record()

    def delete(self, user_id: str, upload_id: str) -> None:
        if self.error is not None:
            raise self.error
        self.deleted.append((user_id, upload_id))


class _InstallationServiceStub:
    def __init__(self) -> None:
        self.started: list[tuple[str, str, object]] = []
        self.cancelled: list[tuple[str, str]] = []
        self.error: Exception | None = None
        self.operation = _operation()
        self.recovered: list[str] = []

    async def ensure_recovered(self, user_id: str) -> None:
        self.recovered.append(user_id)

    async def start(self, user_id, upload_id, request):
        if self.error is not None:
            raise self.error
        self.started.append((user_id, upload_id, request))
        return self.operation

    def get_owned(self, user_id, operation_id):
        if self.error is not None:
            raise self.error
        return self.operation

    async def cancel(self, user_id, operation_id):
        if self.error is not None:
            raise self.error
        self.cancelled.append((user_id, operation_id))
        return self.operation.model_copy(update={"state": "cancelled"})


class _ApiEnv:
    def __init__(self, client: TestClient, uploads, installations) -> None:
        self.client = client
        self.uploads = uploads
        self.installations = installations
        self.auth_headers = {"Authorization": "Bearer local-session-token"}

    def stage(self, prefix: str = "") -> dict:
        response = self.client.post(
            f"{prefix}/skills/uploads",
            files={"file": ("demo.zip", _valid_skill_zip_bytes(), "application/zip")},
            headers=self.auth_headers,
        )
        assert response.status_code == 201, response.text
        return response.json()["data"]


@pytest.fixture()
def api_env(monkeypatch) -> _ApiEnv:
    uploads = _UploadServiceStub()
    installations = _InstallationServiceStub()
    api_globals = skills_api.stage_skill_upload.__globals__
    monkeypatch.setitem(api_globals, "get_skill_upload_service", lambda: uploads)
    monkeypatch.setitem(api_globals, "get_skill_installation_service", lambda: installations)

    app = FastAPI()
    app.include_router(skills_api.router)
    app.include_router(skills_api.router, prefix="/api")
    register_skill_exception_handlers(app)
    app.dependency_overrides[skills_api.require_local_session] = lambda: SimpleNamespace(
        user_id=USER_ID
    )
    with TestClient(app) as client:
        yield _ApiEnv(client, uploads, installations)


def _api_globals() -> dict:
    """The route module's globals, where its `except` clauses resolve names.

    Exceptions must be constructed from *this* namespace: another test in this
    directory evicts every ``client_backend`` module to prove an import boundary,
    after which a re-imported ``SkillUploadError`` is a different class and
    neither the route's ``except`` nor ``response_for_exception``'s ``isinstance``
    recognizes it.
    """
    return skills_api.stage_skill_upload.__globals__


def _upload_error(code: str, message: str = "boom", status_code: int = 400, **kwargs):
    return _api_globals()["SkillUploadError"](code, message, status_code, **kwargs)


def _operation_error(code: str, message: str = "boom", status_code: int = 400):
    return _api_globals()["SkillOperationError"](code, message, status_code=status_code)


def test_stage_upload_returns_201_camel_case_preview(api_env):
    response = api_env.client.post(
        "/skills/uploads",
        files={"file": ("demo.zip", _valid_skill_zip_bytes(), "application/zip")},
        headers=api_env.auth_headers,
    )

    assert response.status_code == 201
    data = response.json()["data"]
    assert data["uploadId"] == "upload-a"
    assert data["preview"]["sourceHash"] == SOURCE_HASH
    assert "source_hash" not in data["preview"]
    assert data["preview"]["executableAssets"]["pythonProject"] is False
    assert data["preview"]["setup"]["confirmationRequired"] is False
    assert data["archive"]["compressedBytes"] == 1024
    assert "owner" not in data


def test_stage_upload_forwards_the_filename_and_body(api_env):
    api_env.stage()

    user_id, filename, body = api_env.uploads.staged[0]
    assert user_id == USER_ID
    assert filename == "demo.zip"
    assert body == _valid_skill_zip_bytes()


def test_start_install_returns_202_and_status_url(api_env):
    upload = api_env.stage()

    response = api_env.client.post(
        f"/skills/uploads/{upload['uploadId']}/install",
        json={
            "expectedSourceHash": upload["preview"]["sourceHash"],
            "approveSetup": False,
        },
        headers=api_env.auth_headers,
    )

    assert response.status_code == 202
    data = response.json()["data"]
    assert data["statusUrl"].startswith("/skills/installations/")
    assert data["operationId"] == "operationa"
    assert data["state"] == "pending"


def test_start_install_accepts_documented_snake_case_aliases(api_env):
    response = api_env.client.post(
        "/skills/uploads/upload-a/install",
        json={
            "expected_source_hash": SOURCE_HASH,
            "approve_setup": True,
            "replace_source_hash": "b" * 64,
        },
        headers=api_env.auth_headers,
    )

    assert response.status_code == 202
    _, _, request = api_env.installations.started[0]
    assert request.expected_source_hash == SOURCE_HASH
    assert request.approve_setup is True
    assert request.replace_source_hash == "b" * 64


def test_start_install_rejects_unknown_fields(api_env):
    response = api_env.client.post(
        "/skills/uploads/upload-a/install",
        json={"expectedSourceHash": SOURCE_HASH, "elevate": True},
        headers=api_env.auth_headers,
    )

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["code"] == "SKILL_REQUEST_INVALID"
    # The default 422 body echoes rejected values; a skill request can carry a
    # filename or hash, so only field locations may appear.
    assert SOURCE_HASH not in response.text


def test_start_install_requires_the_expected_hash(api_env):
    response = api_env.client.post(
        "/skills/uploads/upload-a/install",
        json={"approveSetup": True},
        headers=api_env.auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["code"] == "SKILL_REQUEST_INVALID"


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        ("SKILL_ARCHIVE_TYPE_UNSUPPORTED", 415),
        ("SKILL_ARCHIVE_TOO_LARGE", 413),
        ("SKILL_ARCHIVE_TOO_MANY_FILES", 413),
        ("SKILL_UPLOAD_QUOTA_EXCEEDED", 413),
        ("SKILL_ARCHIVE_INVALID", 400),
        ("SKILL_ARCHIVE_PATH_UNSAFE", 400),
        ("SKILL_BUNDLE_INVALID", 400),
        ("SKILL_STORAGE_INSUFFICIENT", 507),
    ],
)
def test_upload_failures_map_to_documented_statuses(api_env, code, expected_status):
    api_env.uploads.error = _upload_error(code, "safe message", expected_status)

    response = api_env.client.post(
        "/skills/uploads",
        files={"file": ("demo.zip", b"PK", "application/zip")},
        headers=api_env.auth_headers,
    )

    assert response.status_code == expected_status
    body = response.json()
    assert body["success"] is False
    assert body["code"] == code
    assert body["message"] == "safe message"
    assert body["data"] is None
    assert set(body["error"]) == {"retryable"}


def test_retryable_flag_is_set_for_transient_failures(api_env):
    api_env.uploads.error = _upload_error(
        "SKILL_STORAGE_INSUFFICIENT",
        "Free some disk space.",
        507,
        retryable=True,
    )

    response = api_env.client.post(
        "/skills/uploads",
        files={"file": ("demo.zip", b"PK", "application/zip")},
        headers=api_env.auth_headers,
    )

    assert response.json()["error"]["retryable"] is True


def test_foreign_or_expired_upload_is_a_404(api_env):
    api_env.uploads.error = _upload_error(
        "SKILL_UPLOAD_NOT_FOUND",
        "The upload is unknown or has expired.",
        status_code=404,
    )

    response = api_env.client.delete("/skills/uploads/upload-a", headers=api_env.auth_headers)

    assert response.status_code == 404
    assert response.json()["code"] == "SKILL_UPLOAD_NOT_FOUND"


def test_cancelling_a_staged_upload_succeeds(api_env):
    response = api_env.client.delete("/skills/uploads/upload-a", headers=api_env.auth_headers)

    assert response.status_code == 200
    assert response.json()["data"] == {"uploadId": "upload-a", "state": "cancelled"}
    assert api_env.uploads.deleted == [(USER_ID, "upload-a")]


def test_consumed_upload_conflict_is_a_409(api_env):
    api_env.installations.error = _operation_error(
        "SKILL_UPLOAD_CONSUMED",
        "This upload already has a different installation request.",
        status_code=409,
    )

    response = api_env.client.post(
        "/skills/uploads/upload-a/install",
        json={"expectedSourceHash": SOURCE_HASH},
        headers=api_env.auth_headers,
    )

    assert response.status_code == 409
    assert response.json()["code"] == "SKILL_UPLOAD_CONSUMED"


def test_locked_installation_is_a_423(api_env):
    api_env.installations.error = _operation_error(
        "SKILL_INSTALL_LOCKED",
        "Another skill operation is in progress.",
        status_code=423,
    )

    response = api_env.client.post(
        "/skills/uploads/upload-a/install",
        json={"expectedSourceHash": SOURCE_HASH},
        headers=api_env.auth_headers,
    )

    assert response.status_code == 423
    assert response.json()["error"]["retryable"] is True


def test_polling_a_running_operation(api_env):
    api_env.installations.operation = _operation(state="running", phase="preparingRuntime")

    response = api_env.client.get("/skills/installations/operationa", headers=api_env.auth_headers)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["state"] == "running"
    assert data["phase"] == "preparingRuntime"
    assert data["result"] is None
    assert data["failure"] is None


def test_polling_a_failed_operation_is_http_200(api_env):
    """The operation resource was retrieved successfully; only its state failed."""
    api_env.installations.operation = _operation(
        state="failed",
        phase="preparingRuntime",
        failure={
            "code": "SKILL_SETUP_FAILED",
            "message": "The skill runtime could not be prepared.",
            "retryable": True,
        },
    )

    response = api_env.client.get("/skills/installations/operationa", headers=api_env.auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["failure"]["code"] == "SKILL_SETUP_FAILED"
    assert body["data"]["failure"]["retryable"] is True


def test_polling_a_succeeded_operation_returns_the_catalog(api_env):
    api_env.installations.operation = _operation(
        state="succeeded",
        phase="syncingCatalog",
        result=SkillInstallationResult(
            action="installed",
            name="demo",
            source_hash=SOURCE_HASH,
            runtime_status="ready",
            catalog={
                "deviceId": "device-123",
                "catalogGeneration": 43,
                "catalogSyncStatus": "synced",
                "skills": [],
                "totalCount": 0,
                "enabledCount": 0,
            },
        ),
    )

    response = api_env.client.get("/skills/installations/operationa", headers=api_env.auth_headers)

    data = response.json()["data"]
    assert data["result"]["action"] == "installed"
    assert data["result"]["sourceHash"] == SOURCE_HASH
    assert data["result"]["runtimeStatus"] == "ready"
    assert data["result"]["catalog"]["catalogSyncStatus"] == "synced"


def test_unknown_operation_is_a_404(api_env):
    api_env.installations.error = _operation_error(
        "SKILL_OPERATION_NOT_FOUND",
        "The installation is unknown or has expired.",
        status_code=404,
    )

    response = api_env.client.get("/skills/installations/nope", headers=api_env.auth_headers)

    assert response.status_code == 404
    assert response.json()["code"] == "SKILL_OPERATION_NOT_FOUND"


def test_cancelling_after_commit_is_a_409(api_env):
    api_env.installations.error = _operation_error(
        "SKILL_OPERATION_COMMITTED",
        "The installation has already been committed.",
        status_code=409,
    )

    response = api_env.client.delete(
        "/skills/installations/operationa", headers=api_env.auth_headers
    )

    assert response.status_code == 409
    assert response.json()["code"] == "SKILL_OPERATION_COMMITTED"


def test_cancelling_before_commit_succeeds(api_env):
    response = api_env.client.delete(
        "/skills/installations/operationa", headers=api_env.auth_headers
    )

    assert response.status_code == 200
    assert response.json()["data"]["state"] == "cancelled"
    assert api_env.installations.cancelled == [(USER_ID, "operationa")]


def test_recovery_runs_on_the_first_authenticated_skill_request(api_env):
    api_env.stage()

    assert api_env.uploads.recovered == [USER_ID]
    assert api_env.installations.recovered == [USER_ID]


@pytest.mark.parametrize("prefix", ["", "/api"])
def test_upload_routes_are_equivalent_under_both_prefixes(api_env, prefix):
    upload = api_env.stage(prefix)
    assert upload["uploadId"] == "upload-a"

    started = api_env.client.post(
        f"{prefix}/skills/uploads/upload-a/install",
        json={"expectedSourceHash": SOURCE_HASH},
        headers=api_env.auth_headers,
    )
    polled = api_env.client.get(
        f"{prefix}/skills/installations/operationa",
        headers=api_env.auth_headers,
    )

    assert started.status_code == 202
    assert polled.status_code == 200
    # statusUrl is always canonical, so a client following it does not need to
    # know which alias it came in on.
    assert started.json()["data"]["statusUrl"].startswith("/skills/installations/")


def test_missing_authentication_uses_the_skill_envelope():
    app = FastAPI()
    app.include_router(skills_api.router)
    register_skill_exception_handlers(app)

    with TestClient(app) as client:
        response = client.get("/skills/installations/operationa")

    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["code"] == "UNAUTHENTICATED"


def test_non_skill_routes_keep_the_default_error_shape():
    """Scoping matters: this feature must not reshape unrelated endpoints."""
    from fastapi import HTTPException

    app = FastAPI()
    register_skill_exception_handlers(app)

    @app.get("/conversations/{conversation_id}")
    async def _conversation(conversation_id: str):
        raise HTTPException(status_code=404, detail="Conversation not found")

    with TestClient(app) as client:
        response = client.get("/conversations/abc")

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}


def test_every_mapped_status_is_a_real_http_status():
    assert all(100 <= status < 600 for status in SKILL_ERROR_STATUS.values())
