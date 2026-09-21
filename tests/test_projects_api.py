"""HTTP contract for the project API, including cross-user access."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

from app.api.projects import router
from app.core.auth import get_current_user_id
from app.core.config import settings
from app.database.database import Database
from app.models.conversation import Conversation
from app.models.custom_agent import ConversationCustomAgent, CustomAgent
from app.models.project import Project, ProjectCustomAgent
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
    owner_id = uuid4()
    other_id = uuid4()
    conversation_id = uuid4()
    with sf() as s:
        for uid in (owner_id, other_id):
            s.add(
                User(
                    id=uid,
                    username=f"u_{uid.hex[:12]}",
                    email=f"{uid.hex[:12]}@test.local",
                    password_hash="x",
                )
            )
        s.add(Conversation(id=conversation_id, owner_id=owner_id, title="t"))
        s.commit()

    owner_client = TestClient(_build_app(owner_id))
    other_client = TestClient(_build_app(other_id))
    try:
        yield owner_client, other_client, owner_id, other_id, conversation_id
    finally:
        with sf() as s:
            s.execute(delete(ConversationCustomAgent))
            s.execute(delete(ProjectCustomAgent))
            for uid in (owner_id, other_id):
                s.execute(delete(Conversation).where(Conversation.owner_id == uid))
                s.execute(delete(Project).where(Project.owner_id == uid))
                s.execute(delete(CustomAgent).where(CustomAgent.owner_id == uid))
                s.execute(delete(User).where(User.id == uid))
            s.commit()


def test_crud_flow(api):
    owner, _other, _oid, _otid, _cid = api

    created = owner.post("/projects", json={"name": "Roadmap", "instructions": "Be brief."})
    assert created.status_code == 201, created.text
    project = created.json()["data"]
    assert project["instructions"] == "Be brief."
    assert project["conversationCount"] == 0

    listed = owner.get("/projects")
    assert listed.status_code == 200
    assert len(listed.json()["data"]) == 1

    patched = owner.patch(f"/projects/{project['id']}", json={"name": "Renamed"})
    assert patched.status_code == 200
    assert patched.json()["data"]["name"] == "Renamed"

    deleted = owner.delete(f"/projects/{project['id']}")
    assert deleted.status_code == 200
    assert owner.get("/projects").json()["data"] == []


def test_instructions_over_the_cap_are_rejected(api):
    owner, *_ = api

    response = owner.post("/projects", json={"name": "X", "instructions": "P" * 8001})

    assert response.status_code == 422


def test_attach_then_detach_a_conversation(api):
    owner, _other, _oid, _otid, conversation_id = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    attached = owner.put(f"/projects/{project_id}/conversations/{conversation_id}")
    assert attached.status_code == 200
    assert owner.get(f"/projects/{project_id}").json()["data"]["conversationCount"] == 1

    detached = owner.delete(f"/projects/{project_id}/conversations/{conversation_id}")
    assert detached.status_code == 200
    assert owner.get(f"/projects/{project_id}").json()["data"]["conversationCount"] == 0


def test_detaching_from_the_wrong_project_is_404(api):
    owner, _other, _oid, _otid, conversation_id = api
    first = owner.post("/projects", json={"name": "First"}).json()["data"]["id"]
    second = owner.post("/projects", json={"name": "Second"}).json()["data"]["id"]
    owner.put(f"/projects/{first}/conversations/{conversation_id}")

    response = owner.delete(f"/projects/{second}/conversations/{conversation_id}")

    assert response.status_code == 404


def test_missing_project_is_404(api):
    owner, *_ = api

    assert owner.get(f"/projects/{uuid4()}").status_code == 404


def test_another_user_cannot_read_the_project(api):
    """Reverse direction: verify the block, not just the happy path."""
    owner, other, *_ = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    assert other.get(f"/projects/{project_id}").status_code == 403


def test_another_user_cannot_attach_to_the_project(api):
    owner, other, _oid, _otid, conversation_id = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    assert (
        other.put(f"/projects/{project_id}/conversations/{conversation_id}").status_code == 403
    )


def test_another_user_cannot_attach_the_owners_conversation_to_their_own_project(api):
    owner, other, _oid, _otid, conversation_id = api
    their_project = other.post("/projects", json={"name": "Theirs"}).json()["data"]["id"]

    response = other.put(f"/projects/{their_project}/conversations/{conversation_id}")

    assert response.status_code == 403


def test_another_user_cannot_delete_the_project(api):
    owner, other, *_ = api
    project_id = owner.post("/projects", json={"name": "Roadmap"}).json()["data"]["id"]

    assert other.delete(f"/projects/{project_id}").status_code == 403
