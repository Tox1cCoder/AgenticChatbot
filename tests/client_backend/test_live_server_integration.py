from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

from client_backend.core.config import client_settings
from client_backend.main import app
from client_backend.services import local_mcp_manager as local_mcp_manager_module
from client_backend.services import local_skills_registry as local_skills_registry_module
from client_backend.services import runtime_bridge as runtime_bridge_module
from client_backend.services import server_api as server_api_module
from client_backend.services import upstream_auth as upstream_auth_module

_RUN_LIVE_SERVER_TESTS = os.getenv("RUN_LIVE_SERVER_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LIVE_SERVER_URL = os.getenv("LIVE_SERVER_TEST_URL", "http://127.0.0.1:8000").rstrip("/")
TEST_PASSWORD = "Passw0rd!234"

pytestmark = pytest.mark.skipif(
    not _RUN_LIVE_SERVER_TESTS,
    reason="set RUN_LIVE_SERVER_TESTS=1 to test against a running API server",
)


def _run(coro):
    return asyncio.run(coro)


def _reset_client_backend_state() -> None:
    _run(server_api_module.close_server_client())
    upstream_auth_module._auth_service = None

    if runtime_bridge_module._runtime_bridge is not None:
        _run(runtime_bridge_module.get_runtime_bridge().stop())
        runtime_bridge_module._runtime_bridge = None

    if local_mcp_manager_module._mcp_manager is not None:
        _run(local_mcp_manager_module.shutdown_mcp_manager())

    local_skills_registry_module._skills_registry = None


def _assert_api_success(response, *, expected_status: int | None = None) -> dict:
    assert response.status_code < 300, response.text
    if expected_status is not None:
        assert response.status_code == expected_status, response.text

    payload = response.json()
    assert payload["success"] is True, payload
    return payload


def _make_identity() -> dict[str, str]:
    suffix = uuid.uuid4().hex[:10]
    return {
        "username": f"u{suffix}",
        "email": f"client-live-{suffix}@example.com",
        "password": TEST_PASSWORD,
    }


def _provision_user_on_server(identity: dict[str, str]) -> dict[str, str]:
    response = requests.post(
        f"{LIVE_SERVER_URL}/auth/signup",
        json={
            "username": identity["username"],
            "email": identity["email"],
            "password": identity["password"],
        },
        timeout=120,
    )
    assert response.status_code < 300, response.text
    payload = response.json()
    assert payload["success"] is True, payload
    return payload["data"]


def _signup_and_login(
    client: TestClient,
    *,
    login_path: str = "/auth/login",
    signup_via_client: bool = True,
) -> dict[str, str]:
    identity = _make_identity()

    if signup_via_client:
        signup_response = client.post(
            "/auth/signup",
            json={
                "username": identity["username"],
                "email": identity["email"],
                "password": identity["password"],
            },
        )
        signup_payload = _assert_api_success(signup_response, expected_status=201)
        assert signup_payload["data"]["email"] == identity["email"]
    else:
        _provision_user_on_server(identity)

    login_response = client.post(
        login_path,
        json={"email": identity["email"], "password": identity["password"]},
    )
    login_payload = _assert_api_success(login_response)
    data = login_payload["data"]

    assert data["userId"]
    assert data["accessToken"]
    assert data["refreshToken"]
    assert data["tokenType"] == "bearer"
    assert data["serverUrl"] == LIVE_SERVER_URL
    assert data["deviceIdentifier"]
    assert data["localSessionToken"]

    return {
        **identity,
        "user_id": data["userId"],
        "access_token": data["accessToken"],
        "refresh_token": data["refreshToken"],
        "local_session_token": data["localSessionToken"],
        "device_id": data.get("deviceId"),
    }


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _wait_for_runtime_status(
    client: TestClient,
    token: str,
    expected_status: str,
    *,
    timeout_seconds: float = 15,
) -> dict:
    deadline = time.time() + timeout_seconds
    last_payload: dict | None = None

    while time.time() < deadline:
        response = client.get("/runtime/status", headers=_auth_headers(token))
        assert response.status_code < 300, response.text
        last_payload = response.json()
        if last_payload.get("status") == expected_status:
            return last_payload
        time.sleep(0.25)

    raise AssertionError(
        f"Runtime status did not reach '{expected_status}' within {timeout_seconds}s: "
        f"{last_payload}"
    )


@pytest.fixture
def live_client_backend(tmp_path):
    profile_root = tmp_path / "profile"
    profile_root.mkdir(parents=True, exist_ok=True)

    original_settings = {
        "server_api_base_url": client_settings.server_api_base_url,
        "server_api_timeout_seconds": client_settings.server_api_timeout_seconds,
        "profile_root": client_settings.profile_root,
        "workspace_roots": list(client_settings.workspace_roots),
        "skills_root": client_settings.skills_root,
        "mcp_config_path": client_settings.mcp_config_path,
        "local_session_secret": client_settings.local_session_secret,
        "max_reconnect_attempts": client_settings.max_reconnect_attempts,
        "reconnect_delay_seconds": client_settings.reconnect_delay_seconds,
    }

    client_settings.server_api_base_url = LIVE_SERVER_URL
    client_settings.server_api_timeout_seconds = 120
    client_settings.profile_root = str(profile_root)
    client_settings.workspace_roots = [str(Path.cwd())]
    client_settings.skills_root = str(tmp_path / "skills")
    client_settings.mcp_config_path = ""
    client_settings.local_session_secret = "integration-test-local-secret-32-bytes"
    client_settings.max_reconnect_attempts = 1
    client_settings.reconnect_delay_seconds = 0

    _reset_client_backend_state()
    try:
        yield
    finally:
        _reset_client_backend_state()
        for key, value in original_settings.items():
            setattr(client_settings, key, value)


def test_live_auth_refresh_restore_and_logout_flow(live_client_backend):
    with TestClient(app) as client:
        health_payload = client.get("/health").json()
        assert health_payload["status"] in {"healthy", "degraded"}

        session = _signup_and_login(client, login_path="/auth/login")

        session_info = client.get("/auth/session").json()
        assert session_info["authenticated"] is True
        assert session_info["userId"] == session["user_id"]
        assert session_info["deviceIdentifier"]

        refresh_response = client.post(
            "/auth/refresh",
            headers=_auth_headers(session["refresh_token"]),
        )
        refresh_payload = _assert_api_success(refresh_response)
        refreshed_access_token = refresh_payload["data"].get("accessToken") or refresh_payload[
            "data"
        ].get("access_token")
        assert refreshed_access_token
        assert refresh_payload["data"]["userId"] == session["user_id"]
        assert refresh_payload["data"]["serverUrl"] == LIVE_SERVER_URL
        assert refresh_payload["data"]["deviceIdentifier"]
        assert refresh_payload["data"]["localSessionToken"]

        user_payload = _assert_api_success(
            client.get(
                f"/api/users/{session['user_id']}",
                headers=_auth_headers(refreshed_access_token),
            )
        )
        assert user_payload["data"]["email"] == session["email"]

    _reset_client_backend_state()

    with TestClient(app) as restored_client:
        restored_response = restored_client.post(f"/auth/restore?user_id={session['user_id']}")
        restored_payload = _assert_api_success(restored_response)
        assert restored_payload["data"]["userId"] == session["user_id"]
        assert restored_payload["data"]["accessToken"]

        local_skills_payload = _assert_api_success(
            restored_client.get(
                "/skills",
                headers=_auth_headers(restored_payload["data"]["localSessionToken"]),
            )
        )
        assert "skills" in local_skills_payload["data"]

        logout_payload = _assert_api_success(restored_client.post("/auth/logout"))
        assert "logged out" in logout_payload["message"].lower()

        unauthorized_response = restored_client.get(
            "/providers",
            headers=_auth_headers(restored_payload["data"]["accessToken"]),
        )
        assert unauthorized_response.status_code == 401


def test_live_login_rejects_switching_active_user_without_logout(live_client_backend):
    with TestClient(app) as client:
        first_session = _signup_and_login(client, login_path="/auth/login")

        other_identity = _make_identity()
        _provision_user_on_server(other_identity)

        conflict_response = client.post(
            "/auth/login",
            json={
                "email": other_identity["email"],
                "password": other_identity["password"],
            },
        )

        assert conflict_response.status_code == 409, conflict_response.text
        conflict_payload = conflict_response.json()
        assert "log out" in str(conflict_payload["detail"]).lower()

        session_info = client.get("/auth/session").json()
        assert session_info["authenticated"] is True
        assert session_info["userId"] == first_session["user_id"]


def test_live_conversation_task_plan_and_alias_routes(live_client_backend):
    with TestClient(app) as client:
        session = _signup_and_login(
            client,
            login_path="/api/auth/login",
            signup_via_client=False,
        )
        headers = _auth_headers(session["access_token"])

        create_payload = _assert_api_success(
            client.post(
                "/conversations/",
                json={"title": "Client backend live integration"},
                headers=headers,
            ),
            expected_status=201,
        )
        conversation_id = create_payload["data"]["id"]

        list_payload = _assert_api_success(
            client.get("/api/conversations/?page=1&limit=20", headers=headers)
        )
        assert any(item["id"] == conversation_id for item in list_payload["data"]["items"])

        get_payload = _assert_api_success(
            client.get(f"/conversations/{conversation_id}", headers=headers)
        )
        assert get_payload["data"]["title"] == "Client backend live integration"

        messages_payload = _assert_api_success(
            client.get(
                f"/api/conversations/{conversation_id}/messages?page=1&limit=20",
                headers=headers,
            )
        )
        assert "items" in messages_payload["data"]
        assert "meta" in messages_payload["data"]

        planning_status_payload = _assert_api_success(
            client.get(
                f"/conversations/{conversation_id}/planning-status",
                headers=headers,
            )
        )
        assert planning_status_payload["data"]["totalTasks"] == 0

        task_plans_payload = _assert_api_success(
            client.post(
                f"/api/conversations/{conversation_id}/task-plans/manual",
                json={"taskDescriptions": ["Verify distributed client backend"]},
                headers=headers,
            )
        )
        assert len(task_plans_payload["data"]) == 1
        task_id = task_plans_payload["data"][0]["id"]

        fetched_task_plans = _assert_api_success(
            client.get(
                f"/conversations/{conversation_id}/task-plans?include_completed=true",
                headers=headers,
            )
        )
        assert fetched_task_plans["data"][0]["id"] == task_id

        updated_task_payload = _assert_api_success(
            client.patch(
                f"/api/task-plans/{task_id}",
                json={"status": "in_progress"},
                headers=headers,
            )
        )
        assert updated_task_payload["data"]["status"] == "in_progress"

        completed_task_payload = _assert_api_success(
            client.post(f"/task-plans/{task_id}/complete", headers=headers)
        )
        assert completed_task_payload["data"]["status"] == "completed"

        ai_conversations_payload = _assert_api_success(
            client.get("/api/ai/conversations?page=1&limit=20", headers=headers)
        )
        assert any(
            item["id"] == conversation_id for item in ai_conversations_payload["data"]["items"]
        )

        delete_payload = _assert_api_success(
            client.delete(f"/conversations/{conversation_id}", headers=headers)
        )
        assert "deleted" in delete_payload["message"].lower()


def test_live_proxy_and_local_management_endpoints(live_client_backend):
    with TestClient(app) as client:
        session = _signup_and_login(
            client,
            login_path="/auth/login",
            signup_via_client=False,
        )
        access_headers = _auth_headers(session["access_token"])
        local_headers = _auth_headers(session["local_session_token"])

        providers_payload = _assert_api_success(client.get("/providers", headers=access_headers))
        assert isinstance(providers_payload["data"], list)

        model_config_payload = _assert_api_success(
            client.get("/api/model-config/options", headers=access_headers)
        )
        assert "providers" in model_config_payload["data"]
        assert "agentConfig" in model_config_payload["data"]

        messages_payload = _assert_api_success(
            client.get("/api/messages?page=1&limit=5", headers=access_headers)
        )
        assert "items" in messages_payload["data"]
        assert "meta" in messages_payload["data"]

        skills_payload = _assert_api_success(client.get("/skills", headers=local_headers))
        assert "skills" in skills_payload["data"]
        assert "totalCount" in skills_payload["data"]

        mcp_payload = _assert_api_success(client.get("/api/mcp/servers", headers=local_headers))
        assert "servers" in mcp_payload["data"]
        assert "totalCount" in mcp_payload["data"]


def test_live_server_routes_do_not_expose_server_skills_api(live_client_backend):
    with TestClient(app) as client:
        session = _signup_and_login(
            client,
            login_path="/auth/login",
            signup_via_client=False,
        )

    other_user = _provision_user_on_server(_make_identity())
    access_headers = _auth_headers(session["access_token"])

    unauthenticated_skills = requests.get(f"{LIVE_SERVER_URL}/skills", timeout=120)
    assert unauthenticated_skills.status_code == 404, unauthenticated_skills.text

    unauthenticated_mcp = requests.get(f"{LIVE_SERVER_URL}/mcp/servers", timeout=120)
    assert unauthenticated_mcp.status_code in {401, 403}, unauthenticated_mcp.text

    own_user_payload = _assert_api_success(
        requests.get(
            f"{LIVE_SERVER_URL}/users/{session['user_id']}",
            headers=access_headers,
            timeout=120,
        )
    )
    assert own_user_payload["data"]["email"] == session["email"]

    other_user_response = requests.get(
        f"{LIVE_SERVER_URL}/users/{other_user['id']}",
        headers=access_headers,
        timeout=120,
    )
    assert other_user_response.status_code == 403, other_user_response.text

    authenticated_skills = requests.get(
        f"{LIVE_SERVER_URL}/skills",
        headers=access_headers,
        timeout=120,
    )
    assert authenticated_skills.status_code == 404, authenticated_skills.text

    mcp_payload = _assert_api_success(
        requests.get(f"{LIVE_SERVER_URL}/mcp/servers", headers=access_headers, timeout=120)
    )
    assert "servers" in mcp_payload["data"]


def test_live_document_upload_list_get_task_and_delete_flow(live_client_backend, tmp_path):
    upload_file = tmp_path / "client-backend-live.txt"
    upload_file.write_text("Distributed client backend integration test.\n", encoding="utf-8")

    with TestClient(app) as client:
        session = _signup_and_login(
            client,
            login_path="/api/auth/login",
            signup_via_client=False,
        )
        headers = _auth_headers(session["access_token"])

        conversation_payload = _assert_api_success(
            client.post(
                "/api/conversations/",
                json={"title": "Document upload integration"},
                headers=headers,
            ),
            expected_status=201,
        )
        conversation_id = conversation_payload["data"]["id"]

        with upload_file.open("rb") as file_handle:
            upload_response = client.post(
                "/documents/upload",
                headers=headers,
                files={"file": (upload_file.name, file_handle, "text/plain")},
                data={"conversation_id": conversation_id},
            )
        upload_payload = _assert_api_success(upload_response, expected_status=201)
        document = upload_payload["data"]["document"]
        processing = upload_payload["data"]["processing"]
        document_id = document["id"]

        assert document["filename"] == upload_file.name
        assert document["conversation_id"] == conversation_id
        assert processing["task_id"]

        list_payload = _assert_api_success(
            client.get(
                f"/api/documents/conversation/{conversation_id}?page=1&page_size=20",
                headers=headers,
            )
        )
        assert list_payload["data"]["total"] >= 1
        assert list_payload["data"]["page"] == 1
        assert list_payload["data"]["page_size"] == 20
        assert any(item["id"] == document_id for item in list_payload["data"]["documents"])

        get_payload = _assert_api_success(client.get(f"/documents/{document_id}", headers=headers))
        assert get_payload["data"]["id"] == document_id
        assert get_payload["data"]["filename"] == upload_file.name

        task_payload = _assert_api_success(
            client.get(f"/api/documents/task/{processing['task_id']}", headers=headers)
        )
        assert task_payload["data"]["task_id"] == processing["task_id"]

        delete_payload = _assert_api_success(
            client.delete(f"/documents/{document_id}", headers=headers)
        )
        assert str(delete_payload["data"]["deleted_document_id"]) == document_id

        delete_conversation_payload = _assert_api_success(
            client.delete(f"/api/conversations/{conversation_id}", headers=headers)
        )
        assert "deleted" in delete_conversation_payload["message"].lower()


def test_live_runtime_bridge_registers_with_server_and_disconnects_cleanly(live_client_backend):
    with TestClient(app) as client:
        session = _signup_and_login(
            client,
            login_path="/auth/login",
            signup_via_client=False,
        )

        local_headers = _auth_headers(session["local_session_token"])
        access_headers = _auth_headers(session["access_token"])

        connect_payload = client.post(
            "/runtime/connect?wait_for_connection=true&timeout_seconds=15",
            headers=local_headers,
        )
        assert connect_payload.status_code < 300, connect_payload.text

        runtime_payload = _wait_for_runtime_status(
            client,
            session["local_session_token"],
            "connected",
        )
        assert runtime_payload["device_info"]["device_id"]
        device_id = runtime_payload["device_info"]["device_id"]

        connected_response = requests.get(
            f"{LIVE_SERVER_URL}/device-runtime/connected-devices",
            headers=access_headers,
            timeout=60,
        )
        assert connected_response.status_code < 300, connected_response.text
        connected_payload = connected_response.json()
        assert any(
            item["device_id"] == device_id for item in connected_payload["connected_devices"]
        )

        devices_response = requests.get(
            f"{LIVE_SERVER_URL}/client-devices/me",
            headers=access_headers,
            timeout=60,
        )
        devices_payload = devices_response.json()
        assert devices_response.status_code < 300, devices_response.text
        assert any(
            item["id"] == device_id and item["status"] == "online" for item in devices_payload
        )

        disconnect_response = client.post("/runtime/disconnect", headers=local_headers)
        assert disconnect_response.status_code < 300, disconnect_response.text

        disconnected_payload = _wait_for_runtime_status(
            client,
            session["local_session_token"],
            "disconnected",
        )
        assert disconnected_payload["device_info"]["device_id"] is None

        connected_after_response = requests.get(
            f"{LIVE_SERVER_URL}/device-runtime/connected-devices",
            headers=access_headers,
            timeout=60,
        )
        assert connected_after_response.status_code < 300, connected_after_response.text
        connected_after_payload = connected_after_response.json()
        assert all(
            item["device_id"] != device_id for item in connected_after_payload["connected_devices"]
        )
