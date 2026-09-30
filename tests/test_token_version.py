"""Revocation of issued tokens by user soft-delete and ``token_version``.

Before this check a soft-deleted user's access token (and refresh token) kept
working until it expired, because the auth dependencies never loaded the user.
"""

from __future__ import annotations

import asyncio
import threading
from uuid import UUID, uuid4

import jwt
import pytest
from dependency_injector import providers
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.auth import router as auth_router
from app.core import auth
from app.core.container import Container, setup_auto_injection
from app.core.exceptions import AuthenticationException
from app.core.security import token_version
from app.core.security.token_version import TokenState, TokenStateCache
from app.models.base import Base
from app.models.user import User
from app.repositories.user import UserRepository
from app.services.auth_service import AuthService
from app.services.jwt_service import JwtService
from app.utils.exception_handler import register_exception_handlers
from tests.token_state_stub import stub_token_states


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def _client() -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/whoami")
    async def whoami(user_id: UUID = Depends(auth.get_current_user_id)):  # noqa: B008
        return {"user_id": str(user_id)}

    @app.post("/refresh")
    async def refresh(user_id: UUID = Depends(auth.get_refresh_token_user_id)):  # noqa: B008
        return {"user_id": str(user_id)}

    return TestClient(app)


def _access(user_id: UUID, **claims) -> dict[str, str]:
    token = JwtService().create_access_token({"sub": str(user_id), **claims})
    return {"Authorization": f"Bearer {token}"}


def _refresh(user_id: UUID, **claims) -> dict[str, str]:
    token = JwtService().create_refresh_token({"sub": str(user_id), **claims})
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Access tokens
# ---------------------------------------------------------------------------


def test_a_soft_deleted_users_access_token_is_refused(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: TokenState(version=0, deleted=True)})

    response = _client().get("/whoami", headers=_access(user_id, ver=0))

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATED_USER_NOT_FOUND"


def test_a_missing_users_access_token_is_refused(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: None})

    response = _client().get("/whoami", headers=_access(user_id))

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATED_USER_NOT_FOUND"


def test_an_access_token_below_the_current_version_is_refused(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: TokenState(version=2, deleted=False)})

    response = _client().get("/whoami", headers=_access(user_id, ver=1))

    assert response.status_code == 401
    assert response.json()["code"] == "TOKEN_REVOKED"


def test_an_access_token_at_the_current_version_is_accepted(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: TokenState(version=2, deleted=False)})

    response = _client().get("/whoami", headers=_access(user_id, ver=2))

    assert response.status_code == 200
    assert response.json() == {"user_id": str(user_id)}


def test_a_token_issued_before_versions_counts_as_version_zero(monkeypatch):
    """Review Focus 4: tokens already issued carry no ``ver`` and must keep working."""
    user_id = uuid4()
    states = stub_token_states(monkeypatch, {user_id: TokenState(version=0, deleted=False)})

    assert _client().get("/whoami", headers=_access(user_id)).status_code == 200

    states[user_id] = TokenState(version=1, deleted=False)
    token_version.token_state_cache.invalidate(user_id)

    response = _client().get("/whoami", headers=_access(user_id))
    assert response.status_code == 401
    assert response.json()["code"] == "TOKEN_REVOKED"


@pytest.mark.parametrize("bad_version", ["1", 1.5, True, None])
def test_a_non_integer_version_claim_is_refused(monkeypatch, bad_version):
    user_id = uuid4()
    stub_token_states(monkeypatch)

    response = _client().get("/whoami", headers=_access(user_id, ver=bad_version))

    assert response.status_code == 401
    assert response.json()["code"] == "INVALID_CREDENTIALS"


def test_get_current_user_refuses_a_revoked_token(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: TokenState(version=3, deleted=False)})
    monkeypatch.setattr(auth, "get_user_service", lambda: pytest.fail("user was loaded"))
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/me")
    async def me(user=Depends(auth.get_current_user)):  # noqa: B008
        return {"id": str(user.id)}

    response = TestClient(app).get("/me", headers=_access(user_id, ver=2))

    assert response.status_code == 401
    assert response.json()["code"] == "TOKEN_REVOKED"


def test_a_cache_miss_loads_off_the_event_loop(monkeypatch):
    user_id = uuid4()
    loads: list[bool] = []

    def loader(requested_id: UUID) -> TokenState:
        try:
            asyncio.get_running_loop()
            loads.append(True)
        except RuntimeError:
            loads.append(False)
        return TokenState(version=0, deleted=False)

    monkeypatch.setattr(token_version, "token_state_cache", TokenStateCache(loader))
    client = _client()

    assert client.get("/whoami", headers=_access(user_id)).status_code == 200
    assert client.get("/whoami", headers=_access(user_id)).status_code == 200
    assert loads == [False], "the user lookup ran on the event loop, or was not cached"


# ---------------------------------------------------------------------------
# Refresh tokens
# ---------------------------------------------------------------------------


def test_a_soft_deleted_users_refresh_token_is_refused(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: TokenState(version=0, deleted=True)})

    response = _client().post("/refresh", headers=_refresh(user_id))

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATED_USER_NOT_FOUND"


def test_a_refresh_token_below_the_current_version_is_refused(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: TokenState(version=1, deleted=False)})

    response = _client().post("/refresh", headers=_refresh(user_id, ver=0))

    assert response.status_code == 401
    assert response.json()["code"] == "TOKEN_REVOKED"


def test_a_refresh_token_without_a_version_is_accepted_at_version_zero(monkeypatch):
    user_id = uuid4()
    stub_token_states(monkeypatch, {user_id: TokenState(version=0, deleted=False)})

    response = _client().post("/refresh", headers=_refresh(user_id))

    assert response.status_code == 200
    assert response.json() == {"user_id": str(user_id)}


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


def _counting_cache(states: dict, clock: FakeClock, **options) -> tuple[TokenStateCache, list]:
    loads: list[UUID] = []

    def loader(user_id: UUID) -> TokenState | None:
        loads.append(user_id)
        return states.get(user_id)

    return TokenStateCache(loader, clock=clock, **options), loads


def test_the_cache_serves_a_state_until_its_ttl_then_reloads():
    user_id = uuid4()
    clock = FakeClock()
    states = {user_id: TokenState(version=0, deleted=False)}
    cache, loads = _counting_cache(states, clock, ttl_seconds=30)

    assert cache.get(user_id) == TokenState(version=0, deleted=False)
    states[user_id] = TokenState(version=0, deleted=True)
    clock.now += 29.9
    assert cache.get(user_id).deleted is False, "served from cache inside the TTL"
    clock.now += 0.1
    assert cache.get(user_id).deleted is True, "reloaded once the TTL elapsed"
    assert loads == [user_id, user_id]


def test_the_cache_remembers_a_missing_user():
    user_id = uuid4()
    cache, loads = _counting_cache({}, FakeClock())

    assert cache.get(user_id) is None
    assert cache.get(user_id) is None
    assert loads == [user_id]


def test_invalidate_forces_the_next_read_to_load():
    user_id = uuid4()
    states = {user_id: TokenState(version=0, deleted=False)}
    cache, loads = _counting_cache(states, FakeClock())
    cache.get(user_id)
    states[user_id] = TokenState(version=1, deleted=False)

    cache.invalidate(user_id)

    assert cache.get(user_id).version == 1
    assert loads == [user_id, user_id]


def test_an_invalidation_during_a_load_is_not_overwritten_by_that_load():
    user_id = uuid4()
    loading = threading.Event()
    release = threading.Event()
    results = iter([TokenState(version=0, deleted=False), TokenState(version=1, deleted=False)])

    def loader(_user_id: UUID) -> TokenState:
        state = next(results)
        if state.version == 0:
            loading.set()
            release.wait(5)
        return state

    cache = TokenStateCache(loader, clock=FakeClock())
    reader = threading.Thread(target=cache.get, args=(user_id,))
    reader.start()
    assert loading.wait(5)
    cache.invalidate(user_id)
    release.set()
    reader.join(5)

    assert cache.get(user_id).version == 1, "the stale pre-invalidation read was cached"


def test_the_cache_is_bounded_and_evicts_the_least_recently_used():
    first, second, third = uuid4(), uuid4(), uuid4()
    live = TokenState(version=0, deleted=False)
    cache, loads = _counting_cache(
        {first: live, second: live, third: live}, FakeClock(), max_entries=2
    )

    cache.get(first)
    cache.get(second)
    cache.get(first)
    cache.get(third)
    cache.get(first)
    cache.get(second)

    assert loads == [first, second, third, second]


@pytest.mark.parametrize("options", [{"ttl_seconds": 0}, {"max_entries": 0}])
def test_the_cache_rejects_non_positive_bounds(options):
    with pytest.raises(ValueError):
        TokenStateCache(lambda _user_id: None, **options)


# ---------------------------------------------------------------------------
# Issuing and revoking
# ---------------------------------------------------------------------------


class _Users:
    def __init__(self, state: TokenState | None) -> None:
        self.state = state

    def get_token_state(self, user_id: UUID) -> TokenState | None:
        return self.state


def _claims(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False})


def test_refresh_issues_an_access_token_at_the_current_version():
    user_id = uuid4()
    service = AuthService(_Users(TokenState(version=4, deleted=False)), JwtService())

    claims = _claims(service.refresh_access_token(user_id))

    assert claims["sub"] == str(user_id)
    assert claims["ver"] == 4
    assert "type" not in claims


@pytest.mark.parametrize("state", [None, TokenState(version=0, deleted=True)])
def test_refresh_refuses_a_missing_or_deleted_user(state):
    service = AuthService(_Users(state), JwtService())

    with pytest.raises(AuthenticationException) as caught:
        service.refresh_access_token(uuid4())

    assert caught.value.status_code == 401


def test_the_refresh_route_issues_a_current_version_access_token(monkeypatch):
    user_id = uuid4()
    current = TokenState(version=2, deleted=False)
    stub_token_states(monkeypatch, {user_id: current})
    setup_auto_injection(Container)
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_router)
    try:
        with Container.user_service.override(providers.Object(_Users(current))):
            response = TestClient(app).post("/auth/refresh", headers=_refresh(user_id, ver=2))
    finally:
        setup_auto_injection(Container)

    assert response.status_code == 200
    claims = _claims(response.json()["data"]["access_token"])
    assert (claims["sub"], claims["ver"]) == (str(user_id), 2)


@pytest.fixture
def user_repository():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine, tables=[User.__table__])
    try:
        yield UserRepository(sessionmaker(bind=engine, expire_on_commit=False))
    finally:
        engine.dispose()


def _add_user(repository: UserRepository) -> UUID:
    user_id = uuid4()
    with repository.session_factory() as session:
        session.add(
            User(
                id=user_id,
                username=f"u{user_id.hex[:8]}",
                email=f"{user_id.hex}@x.test",
                password_hash="x",
            )
        )
        session.commit()
    return user_id


def test_a_new_user_starts_at_version_zero(user_repository):
    user_id = _add_user(user_repository)

    assert user_repository.get_token_state(user_id) == TokenState(version=0, deleted=False)
    assert user_repository.get_token_state(uuid4()) is None


def test_soft_deleting_a_user_bumps_the_version_and_drops_the_cached_state(
    monkeypatch, user_repository
):
    user_id = _add_user(user_repository)
    monkeypatch.setattr(
        token_version, "token_state_cache", TokenStateCache(user_repository.get_token_state)
    )
    assert token_version.token_state_cache.get(user_id) == TokenState(version=0, deleted=False)

    assert user_repository.delete(user_id) is True

    assert user_repository.get_token_state(user_id) == TokenState(version=1, deleted=True)
    assert token_version.token_state_cache.get(user_id).deleted is True
    assert user_repository.delete(user_id) is False, "a second delete is a no-op"
    assert user_repository.get_token_state(user_id).version == 1
