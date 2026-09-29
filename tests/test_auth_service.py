"""Login must not reveal which emails exist, nor admit a soft-deleted account."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

import app.services.auth_service as auth_module
from app.core.security import hash_password
from app.schemas.responses.token_response import LoginRequest
from app.services.auth_service import AuthService
from app.services.jwt_service import JwtService

PASSWORD = "correct horse battery"


class _Users:
    def __init__(self, user) -> None:
        self.user = user

    def get_by_email_with_password(self, email):
        return self.user


def _user(deleted_at=None):
    return SimpleNamespace(
        id=uuid4(), password_hash=hash_password(PASSWORD), deleted_at=deleted_at
    )


def _login(user, password=PASSWORD):
    service = AuthService(_Users(user), JwtService())
    return service.authenticate_user(LoginRequest(email="a@example.com", password=password))


def test_a_live_user_with_the_right_password_logs_in():
    user = _user()

    tokens = _login(user)

    assert tokens["user_id"] == str(user.id)


def test_an_unknown_email_still_runs_a_password_check(monkeypatch):
    checked: list[str] = []
    real_verify = auth_module.verify_password

    def spy(password, hashed):
        checked.append(hashed)
        return real_verify(password, hashed)

    monkeypatch.setattr(auth_module, "verify_password", spy)

    with pytest.raises(HTTPException) as caught:
        _login(None)

    assert caught.value.status_code == 401
    assert len(checked) == 1


def test_a_soft_deleted_user_cannot_log_in():
    with pytest.raises(HTTPException) as caught:
        _login(_user(deleted_at=datetime.now(timezone.utc)))

    assert caught.value.status_code == 401
    assert caught.value.detail == "Invalid email or password"


def test_a_wrong_password_is_refused():
    with pytest.raises(HTTPException) as caught:
        _login(_user(), password="wrong password")

    assert caught.value.status_code == 401
