"""The signing and encryption keys: configured, stored, or generated exactly once.

Runs against a SQLite file per test, never a real application database. The
suite configures both keys in the environment (``tests/conftest.py``), so every
test here clears them first to reach the stored and generated paths. The
concurrent ``ON CONFLICT`` path is asserted against PostgreSQL in
``tests/test_alembic_full_chain_postgres.py``.
"""

from __future__ import annotations

import threading

import jwt
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text

from app.core import server_secrets
from app.core.config import settings
from app.core.server_secrets import (
    MODEL_ENCRYPTION_KEY,
    SIGNING_KEY,
    ServerSecretError,
    get_model_encryption_key,
    get_signing_key,
    resolve_server_secrets,
)
from app.models.server_secret import ServerSecret

_CONFIGURED_SIGNING_KEY = "configured-signing-key-" + "x" * 32
_STORED_SIGNING_KEY = "stored-signing-key-" + "y" * 32


@pytest.fixture
def database(tmp_path, monkeypatch):
    """A migrated-enough database, with nothing configured and nothing cached."""
    url = f"sqlite:///{(tmp_path / 'secrets.sqlite').as_posix()}"
    engine = create_engine(url)
    ServerSecret.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE model_providers (id INTEGER PRIMARY KEY, "
                "api_key_encrypted TEXT NOT NULL, deleted_at TEXT)"
            )
        )
    monkeypatch.setattr(settings, "database_url", url)
    monkeypatch.setattr(settings, "secret_key", "")
    monkeypatch.setattr(settings, "model_encryption_key", "")
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(server_secrets, "_resolved", {})
    monkeypatch.setattr(server_secrets, "_DEV_SECRET_KEY_PATH", tmp_path / ".dev_secret_key")
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def unreachable_url(tmp_path) -> str:
    """A database that cannot be opened: its directory does not exist."""
    return f"sqlite:///{(tmp_path / 'missing' / 'x.sqlite').as_posix()}"


def _rows(engine) -> dict[str, str]:
    with engine.connect() as connection:
        return dict(connection.execute(text("SELECT name, value FROM server_secrets")).all())


def _store(engine, name: str, value: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO server_secrets (name, value) VALUES (:name, :value)"),
            {"name": name, "value": value},
        )


def _add_provider(engine, *, deleted: bool = False) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO model_providers (api_key_encrypted, deleted_at) "
                "VALUES (:ciphertext, :deleted_at)"
            ),
            {
                "ciphertext": Fernet(Fernet.generate_key()).encrypt(b"sk-live").decode(),
                "deleted_at": "2026-09-01" if deleted else None,
            },
        )


def _forget_resolved_keys(monkeypatch) -> None:
    """What a restarted process sees: nothing cached."""
    monkeypatch.setattr(server_secrets, "_resolved", {})


# ---------------------------------------------------------------------------
# precedence
# ---------------------------------------------------------------------------


def test_a_configured_signing_key_wins_and_is_never_stored(database, monkeypatch):
    _store(database, SIGNING_KEY, _STORED_SIGNING_KEY)
    monkeypatch.setattr(settings, "secret_key", _CONFIGURED_SIGNING_KEY)

    assert get_signing_key() == _CONFIGURED_SIGNING_KEY
    assert _rows(database) == {SIGNING_KEY: _STORED_SIGNING_KEY}


def test_a_configured_encryption_key_wins_and_is_never_stored(database, monkeypatch):
    configured = Fernet.generate_key().decode()
    monkeypatch.setattr(settings, "model_encryption_key", configured)

    resolve_server_secrets()

    assert get_model_encryption_key() == configured
    assert MODEL_ENCRYPTION_KEY not in _rows(database)


def test_configured_keys_never_touch_the_database(database, monkeypatch, unreachable_url):
    monkeypatch.setattr(settings, "secret_key", _CONFIGURED_SIGNING_KEY)
    monkeypatch.setattr(settings, "model_encryption_key", Fernet.generate_key().decode())
    monkeypatch.setattr(settings, "database_url", unreachable_url)

    resolve_server_secrets()

    assert get_signing_key() == _CONFIGURED_SIGNING_KEY


def test_the_stored_row_is_used_when_nothing_is_configured(database):
    _store(database, SIGNING_KEY, _STORED_SIGNING_KEY)

    assert get_signing_key() == _STORED_SIGNING_KEY
    assert _rows(database) == {SIGNING_KEY: _STORED_SIGNING_KEY}


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def test_a_missing_signing_key_is_generated_and_stored_once(database, monkeypatch):
    first = get_signing_key()
    _forget_resolved_keys(monkeypatch)
    second = get_signing_key()

    assert len(first) >= 32
    assert first == second
    assert _rows(database) == {SIGNING_KEY: first}


def test_a_missing_encryption_key_is_generated_as_a_fernet_key(database, monkeypatch):
    generated = get_model_encryption_key()
    _forget_resolved_keys(monkeypatch)

    assert get_model_encryption_key() == generated
    assert _rows(database) == {MODEL_ENCRYPTION_KEY: generated}
    assert Fernet(generated).decrypt(Fernet(generated).encrypt(b"sk-live")) == b"sk-live"


def test_concurrent_resolvers_converge_on_the_first_stored_value(database):
    """Two replicas that both find no row both generate; only one value survives.

    The barrier holds each resolver between its read (nothing stored) and its
    insert, which is the window a plain INSERT would lose a key in.
    """
    both_read_nothing = threading.Barrier(2, timeout=10)
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def resolve(label: str) -> None:
        def candidate(_connection) -> str:
            both_read_nothing.wait()
            return f"candidate-from-{label}-" + "z" * 32

        try:
            results[label] = server_secrets._load_or_create(database, SIGNING_KEY, candidate)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test thread
            errors.append(exc)

    threads = [threading.Thread(target=resolve, args=(label,)) for label in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert results["a"] == results["b"]
    assert _rows(database) == {SIGNING_KEY: results["a"]}


def test_a_resolved_key_is_cached_for_the_process(database, monkeypatch, unreachable_url):
    first = get_signing_key()
    monkeypatch.setattr(settings, "database_url", unreachable_url)

    assert get_signing_key() == first


def test_startup_resolution_fills_the_cache_for_both_keys(database):
    resolve_server_secrets()

    assert set(server_secrets._resolved) == {SIGNING_KEY, MODEL_ENCRYPTION_KEY}
    assert set(_rows(database)) == {SIGNING_KEY, MODEL_ENCRYPTION_KEY}


# ---------------------------------------------------------------------------
# the development key file
# ---------------------------------------------------------------------------


def test_an_existing_dev_key_file_seeds_the_signing_key(database):
    """Sessions signed with the old per-checkout key stay valid after the switch."""
    server_secrets._DEV_SECRET_KEY_PATH.write_text("  legacy-dev-key-value\n", encoding="utf-8")

    assert get_signing_key() == "legacy-dev-key-value"
    assert _rows(database) == {SIGNING_KEY: "legacy-dev-key-value"}


def test_the_dev_key_file_is_ignored_outside_development(database, monkeypatch):
    """A developer's key copied into a deployment must not become its signing key."""
    monkeypatch.setattr(settings, "environment", "production")
    server_secrets._DEV_SECRET_KEY_PATH.write_text("legacy-dev-key-value", encoding="utf-8")

    assert get_signing_key() != "legacy-dev-key-value"


def test_no_dev_key_file_is_created(database):
    get_signing_key()

    assert not server_secrets._DEV_SECRET_KEY_PATH.exists()


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deleted", [False, True])
def test_a_new_encryption_key_is_refused_while_provider_keys_are_stored(database, deleted):
    """Generating would make every stored provider key permanently undecryptable."""
    _add_provider(database, deleted=deleted)

    with pytest.raises(ServerSecretError, match="set MODEL_ENCRYPTION_KEY to that key"):
        get_model_encryption_key()

    assert _rows(database) == {}


def test_the_stored_encryption_key_is_used_while_provider_keys_exist(database):
    stored = Fernet.generate_key().decode()
    _store(database, MODEL_ENCRYPTION_KEY, stored)
    _add_provider(database)

    assert get_model_encryption_key() == stored


def test_a_stored_encryption_key_that_is_not_a_fernet_key_is_refused(database):
    _store(database, MODEL_ENCRYPTION_KEY, "not-a-fernet-key")

    with pytest.raises(ServerSecretError, match="not a valid Fernet key"):
        get_model_encryption_key()


@pytest.mark.parametrize(
    ("accessor", "variable"),
    [(get_signing_key, "SECRET_KEY"), (get_model_encryption_key, "MODEL_ENCRYPTION_KEY")],
)
def test_an_unreachable_database_with_nothing_configured_fails_loudly(
    database, monkeypatch, unreachable_url, accessor, variable
):
    monkeypatch.setattr(settings, "database_url", unreachable_url)

    with pytest.raises(ServerSecretError, match=variable):
        accessor()

    assert server_secrets._resolved == {}, "a failure must not be cached as a key"


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def test_access_tokens_are_signed_with_the_stored_key(database):
    from app.services.jwt_service import JwtService

    _store(database, SIGNING_KEY, _STORED_SIGNING_KEY)
    service = JwtService()

    token = service.create_access_token({"sub": "user"})

    assert jwt.decode(token, _STORED_SIGNING_KEY, algorithms=["HS256"])["sub"] == "user"
    assert service.decode_token(token)["sub"] == "user"


def test_widget_tokens_are_signed_with_the_stored_key(database):
    from app.services.widget_tokens import WidgetTokenService

    _store(database, SIGNING_KEY, _STORED_SIGNING_KEY)
    service = WidgetTokenService()

    token, _expires_at = service.mint(widget_id="w", session_id="s", user_id="u")

    assert jwt.decode(token, _STORED_SIGNING_KEY, algorithms=["HS256"])["wid"] == "w"
    assert service.verify(token)["wid"] == "w"


def test_provider_keys_are_encrypted_with_the_stored_key(database):
    from types import SimpleNamespace

    from app.services.provider_service import ProviderService

    stored = Fernet.generate_key().decode()
    _store(database, MODEL_ENCRYPTION_KEY, stored)

    ciphertext = ProviderService(provider_repository=SimpleNamespace())._encrypt_key("sk-live")

    assert Fernet(stored).decrypt(ciphertext.encode()) == b"sk-live"


# ---------------------------------------------------------------------------
# process start
# ---------------------------------------------------------------------------


async def test_api_startup_resolves_the_keys_after_migrating_and_fails_closed(monkeypatch):
    from app import main

    calls: list[str] = []

    async def migrate() -> None:
        calls.append("migrate")

    async def ready() -> None:
        return None

    def resolve() -> None:
        calls.append("resolve")
        raise ServerSecretError("no key")

    monkeypatch.setattr(main, "_ensure_selector_event_loop", lambda: None)
    monkeypatch.setattr(main, "_verify_async_database_ready", ready)
    monkeypatch.setattr(main, "init_database_migrations", migrate)
    monkeypatch.setattr(main, "resolve_server_secrets", resolve)

    with pytest.raises(ServerSecretError):
        async with main.lifespan(main.app):
            pass

    assert calls == ["migrate", "resolve"]


@pytest.mark.parametrize("signal_name", ["worker_init", "worker_process_init"])
def test_celery_worker_start_resolves_the_keys(monkeypatch, signal_name):
    from celery import signals

    import app.workers.celery_app as celery_app_module

    calls: list[str] = []
    monkeypatch.setattr(
        celery_app_module, "resolve_server_secrets", lambda: calls.append("resolve")
    )

    getattr(signals, signal_name).send(sender=None)

    assert calls == ["resolve"]
