import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core import auth
from app.services.jwt_service import JwtService


def test_current_user_lookup_does_not_block_the_event_loop(monkeypatch):
    """The user lookup is a blocking database read; run on the event loop it
    stalls every other in-flight request while it waits."""
    user_id = uuid4()
    lookups: list[bool] = []

    def get_by_id(requested_id):
        try:
            asyncio.get_running_loop()
            lookups.append(True)
        except RuntimeError:
            lookups.append(False)
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            id=requested_id,
            username="u",
            email="u@example.test",
            created_at=now,
            updated_at=now,
            deleted_at=None,
            avatar_url=None,
        )

    monkeypatch.setattr(auth, "get_user_service", lambda: SimpleNamespace(get_by_id=get_by_id))
    app = FastAPI()

    @app.get("/me")
    async def me(user=Depends(auth.get_current_user)):  # noqa: B008
        return {"id": str(user.id)}

    token = JwtService().create_access_token({"sub": str(user_id)})
    response = TestClient(app).get("/me", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert response.json() == {"id": str(user_id)}
    assert lookups == [False], "user lookup ran on the event loop thread"
