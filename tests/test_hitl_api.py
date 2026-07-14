"""HTTP-level tests for the per-user HITL settings API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.hitl import router
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.core.exceptions import CustomHTTPException
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.hitl_interrupt import HITLInterrupt, HITLInterruptStatus
from app.models.tool_approval_setting import ToolApprovalSetting
from app.models.user import User
from app.repositories.hitl_interrupt import HITLInterruptRepository


def _build_app(user_id):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id

    @app.exception_handler(CustomHTTPException)
    async def custom_http_exception_handler(_, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "code": exc.error_code, "message": exc.detail},
        )

    return app


def _create_interrupt(
    session_factory,
    user_id,
    *,
    status: HITLInterruptStatus,
    expires_at: datetime | None = None,
) -> str:
    conversation_id = uuid4()
    interrupt_id = f"interrupt-{uuid4()}"
    with session_factory() as session:
        session.add(
            Conversation(
                id=conversation_id,
                owner_id=user_id,
                title="HITL lifecycle test",
            )
        )
        session.add(
            HITLInterrupt(
                id=interrupt_id,
                conversation_id=conversation_id,
                user_id=user_id,
                thread_id=str(conversation_id),
                status=status,
                expires_at=expires_at or datetime.now(timezone.utc) + timedelta(minutes=5),
                action_requests_json=[],
                interrupt_metadata_json={},
            )
        )
        session.commit()
    return interrupt_id


@pytest.fixture
def api():
    db = Database(settings.database_url)
    sf = db.session
    user_id = uuid4()
    with sf() as s:
        s.add(User(id=user_id, username=f"u_{user_id.hex[:12]}",
                   email=f"{user_id.hex[:12]}@test.local", password_hash="x"))
        s.commit()
    client = TestClient(_build_app(user_id))
    try:
        yield client, user_id, sf
    finally:
        with sf() as s:
            s.execute(delete(HITLInterrupt).where(HITLInterrupt.user_id == user_id))
            s.execute(delete(Conversation).where(Conversation.owner_id == user_id))
            s.execute(delete(ToolApprovalSetting).where(ToolApprovalSetting.user_id == user_id))
            s.execute(delete(User).where(User.id == user_id))
            s.commit()


def test_post_then_get_roundtrips_server_and_tool_rules(api):
    client, _user_id, _sf = api
    resp = client.post("/hitl/settings", json={"items": [
        {"scopeType": "server", "scopeValue": "desktop_commander", "requireApproval": True},
        {
            "scopeType": "tool",
            "scopeValue": "desktop_commander::list_files",
            "requireApproval": False,
        },
    ]})
    assert resp.status_code == 200
    assert resp.json()["success"] is True

    data = client.get("/hitl/settings").json()["data"]
    servers = {s["scopeValue"]: s["requireApproval"] for s in data["servers"]}
    tools = {t["scopeValue"]: t["requireApproval"] for t in data["tools"]}
    assert servers == {"desktop_commander": True}
    assert tools == {"desktop_commander::list_files": False}
    assert "masterEnabled" in data


def test_delete_clears_rule(api):
    client, _user_id, _sf = api
    client.post("/hitl/settings", json={"items": [
        {"scopeType": "server", "scopeValue": "excel", "requireApproval": True},
    ]})
    resp = client.request("DELETE", "/hitl/settings",
                          params={"scope_type": "server", "scope_value": "excel"})
    assert resp.status_code == 200
    data = client.get("/hitl/settings").json()["data"]
    assert all(s["scopeValue"] != "excel" for s in data["servers"])


def test_rejects_invalid_scope_type(api):
    client, _user_id, _sf = api
    resp = client.post("/hitl/settings", json={"items": [
        {"scopeType": "garbage", "scopeValue": "ignored", "requireApproval": True},
    ]})
    assert resp.status_code == 422


def test_get_interrupt_state_returns_only_public_failed_lifecycle_fields(api):
    client, user_id, session_factory = api
    interrupt_id = _create_interrupt(
        session_factory,
        user_id,
        status=HITLInterruptStatus.FAILED,
    )

    response = client.get(f"/hitl/interrupts/{interrupt_id}")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "failed"
    assert set(response.json()["data"]) == {
        "interruptId",
        "conversationId",
        "status",
        "expiresAt",
        "updatedAt",
    }


def test_mark_failed_changes_only_a_resolving_interrupt(api):
    _client, user_id, session_factory = api
    resolving_id = _create_interrupt(
        session_factory,
        user_id,
        status=HITLInterruptStatus.RESOLVING,
    )
    pending_id = _create_interrupt(
        session_factory,
        user_id,
        status=HITLInterruptStatus.PENDING,
    )
    repository = HITLInterruptRepository(session_factory)

    assert repository.mark_failed(resolving_id, resolution_source="stream_error") is True
    assert repository.mark_failed(pending_id, resolution_source="stream_error") is False

    assert repository.get_by_id(resolving_id).status == HITLInterruptStatus.FAILED
    assert repository.get_by_id(pending_id).status == HITLInterruptStatus.PENDING


def test_get_interrupt_state_hides_foreign_expired_interrupt_without_expiring_it(api):
    client, user_id, session_factory = api
    foreign_user_id = uuid4()
    with session_factory() as session:
        session.add(
            User(
                id=foreign_user_id,
                username=f"u_{foreign_user_id.hex[:12]}",
                email=f"{foreign_user_id.hex[:12]}@test.local",
                password_hash="x",
            )
        )
        session.commit()

    try:
        interrupt_id = _create_interrupt(
            session_factory,
            foreign_user_id,
            status=HITLInterruptStatus.PENDING,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )

        response = client.get(f"/hitl/interrupts/{interrupt_id}")

        assert response.status_code == 404
        assert response.json()["code"] == "INTERRUPT_NOT_FOUND"
        assert HITLInterruptRepository(session_factory).get_by_id(interrupt_id).status == (
            HITLInterruptStatus.PENDING
        )
    finally:
        with session_factory() as session:
            session.execute(delete(HITLInterrupt).where(HITLInterrupt.user_id == foreign_user_id))
            session.execute(delete(Conversation).where(Conversation.owner_id == foreign_user_id))
            session.execute(delete(User).where(User.id == foreign_user_id))
            session.commit()


def test_get_interrupt_state_lazily_expires_owned_pending_interrupt(api):
    client, user_id, session_factory = api
    interrupt_id = _create_interrupt(
        session_factory,
        user_id,
        status=HITLInterruptStatus.PENDING,
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )

    response = client.get(f"/hitl/interrupts/{interrupt_id}")

    assert response.status_code == 200
    assert response.json()["data"] == {
        "interruptId": interrupt_id,
        "conversationId": str(
            HITLInterruptRepository(session_factory).get_by_id(interrupt_id).conversation_id
        ),
        "status": "expired",
        "expiresAt": response.json()["data"]["expiresAt"],
        "updatedAt": response.json()["data"]["updatedAt"],
    }
    assert HITLInterruptRepository(session_factory).get_by_id(interrupt_id).status == (
        HITLInterruptStatus.EXPIRED
    )
