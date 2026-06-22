"""HTTP-level tests for the per-user HITL settings API."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.hitl import router
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.database.database import Database
from app.models.tool_approval_setting import ToolApprovalSetting
from app.models.user import User


def _build_app(user_id):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    return app


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
            s.execute(delete(ToolApprovalSetting).where(ToolApprovalSetting.user_id == user_id))
            s.execute(delete(User).where(User.id == user_id))
            s.commit()


def test_post_then_get_roundtrips_server_and_tool_rules(api):
    client, _user_id, _sf = api
    resp = client.post("/hitl/settings", json={"items": [
        {"scopeType": "server", "scopeValue": "desktop_commander", "requireApproval": True},
        {"scopeType": "tool", "scopeValue": "desktop_commander::list_files", "requireApproval": False},
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
