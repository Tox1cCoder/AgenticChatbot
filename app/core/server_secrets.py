"""The server's token signing key and its provider-key encryption key.

Each key comes from the first of:

1. The environment (``SECRET_KEY`` / ``MODEL_ENCRYPTION_KEY``). A configured value
   always wins and is never written to the database.
2. Its row in the ``server_secrets`` table.
3. A newly generated value, inserted with ``ON CONFLICT (name) DO NOTHING`` and then
   read back, so API replicas and Celery workers that start together all keep
   whichever insert committed first.

The API resolves both keys at startup and each Celery worker when it starts, so a
request only reads this process's cache. Anything else, such as a script, resolves
on first use. A key is never empty: with nothing configured and no usable database,
the accessors raise :class:`ServerSecretError`.

The trade-off was chosen deliberately: a generated key lives in the database, so a
database backup is enough to sign tokens and to decrypt stored provider API keys.
Configure both keys in the environment wherever that matters.

``app.core.config`` must never import this module. The client sidecar bundle ships
``config.py`` and carries no database code.
"""

from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Callable
from pathlib import Path

from cryptography.fernet import Fernet
from sqlalchemy import Connection, Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from app.core.config import settings

logger = logging.getLogger(__name__)

#: Row names in ``server_secrets``.
SIGNING_KEY = "signing_key"
MODEL_ENCRYPTION_KEY = "model_encryption_key"

_VARIABLE = {SIGNING_KEY: "SECRET_KEY", MODEL_ENCRYPTION_KEY: "MODEL_ENCRYPTION_KEY"}

#: Written by earlier development setups. It seeds the stored signing key once, so
#: the tokens it signed stay valid; nothing creates or reads it after that.
_DEV_SECRET_KEY_PATH = Path(__file__).resolve().parents[2] / ".dev_secret_key"

_lock = threading.Lock()
_resolved: dict[str, str] = {}


class ServerSecretError(RuntimeError):
    """A key could not be resolved, so nothing may be signed or encrypted."""


def get_signing_key() -> str:
    """The key that signs and verifies access, refresh and widget tokens."""
    return settings.secret_key or _cached(SIGNING_KEY)


def get_model_encryption_key() -> str:
    """The Fernet key that encrypts stored provider API keys."""
    return settings.model_encryption_key or _cached(MODEL_ENCRYPTION_KEY)


def resolve_server_secrets() -> None:
    """Resolve both keys now, so that no request waits on the database for one."""
    get_signing_key()
    get_model_encryption_key()


def _cached(name: str) -> str:
    value = _resolved.get(name)
    if value is not None:
        return value
    # Held across the database round trip: concurrent first callers in this
    # process wait for one resolution instead of racing their own.
    with _lock:
        value = _resolved.get(name)
        if value is None:
            value = _resolve_from_database(name)
            _resolved[name] = value
    return value


def _resolve_from_database(name: str) -> str:
    variable = _VARIABLE[name]
    new_value = _new_signing_key if name == SIGNING_KEY else _new_model_encryption_key
    # A throwaway engine rather than the application's pool: this runs once per
    # process, and a pooled connection opened in a Celery prefork parent must
    # not be inherited by the children it forks.
    engine: Engine | None = None
    try:
        engine = create_engine(settings.database_url, poolclass=NullPool)
        value = _load_or_create(engine, name, new_value)
    except SQLAlchemyError as exc:
        raise ServerSecretError(
            f"{variable} is not set, and the {name!r} key could not be read from or "
            f"stored in the server_secrets table ({type(exc).__name__}). Make sure the "
            f"database is reachable and migrated, or set {variable}."
        ) from exc
    finally:
        if engine is not None:
            engine.dispose()
    return _validated(name, value)


def _load_or_create(engine: Engine, name: str, new_value: Callable[[Connection], str]) -> str:
    """Return the stored ``name``, first storing ``new_value(connection)`` if absent.

    Concurrent callers converge. The insert does nothing when another caller's row
    exists (PostgreSQL waits for that row's transaction to finish first), and every
    caller returns what it reads back afterwards in a fresh transaction.
    """
    with engine.connect() as connection:
        stored = _read(connection, name)
        if stored is None:
            connection.execute(
                text(
                    "INSERT INTO server_secrets (name, value) VALUES (:name, :value) "
                    "ON CONFLICT (name) DO NOTHING"
                ),
                {"name": name, "value": new_value(connection)},
            )
            connection.commit()
            stored = _read(connection, name)
    if stored is None:
        raise ServerSecretError(f"the {name!r} row was gone right after it was stored")
    return stored


def _read(connection: Connection, name: str) -> str | None:
    return connection.execute(
        text("SELECT value FROM server_secrets WHERE name = :name"), {"name": name}
    ).scalar_one_or_none()


def _new_signing_key(_connection: Connection) -> str:
    seeded = _development_key_file()
    if seeded:
        logger.info(
            "Seeding the stored signing key from %s so that the tokens it signed stay "
            "valid. The file is not read again and can be deleted.",
            _DEV_SECRET_KEY_PATH.name,
        )
        return seeded
    return secrets.token_urlsafe(48)


def _development_key_file() -> str:
    # Only development ever wrote this file. Elsewhere it could only be a
    # developer's key copied into a deployment, which must not become its key.
    if settings.environment != "development":
        return ""
    try:
        return _DEV_SECRET_KEY_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        logger.warning(
            "Could not read %s (%s); generating a new signing key, so earlier "
            "development sessions end.",
            _DEV_SECRET_KEY_PATH.name,
            type(exc).__name__,
        )
        return ""


def _new_model_encryption_key(connection: Connection) -> str:
    # Soft-deleted rows count: their ciphertext is still stored.
    encrypted = connection.execute(
        text("SELECT count(*) FROM model_providers WHERE api_key_encrypted <> ''")
    ).scalar_one()
    if encrypted:
        raise ServerSecretError(
            f"MODEL_ENCRYPTION_KEY is not set and server_secrets holds no encryption key, "
            f"but {encrypted} model_providers rows hold encrypted API keys: provider API "
            "keys are already encrypted with a key that is not configured; set "
            "MODEL_ENCRYPTION_KEY to that key. A new key would leave them undecryptable, "
            "so none was generated."
        )
    return Fernet.generate_key().decode()


def _validated(name: str, value: str) -> str:
    variable = _VARIABLE[name]
    if not value:
        raise ServerSecretError(
            f"the stored {name!r} key in server_secrets is empty; set {variable}, or "
            "delete that row to have a new key generated"
        )
    if name == MODEL_ENCRYPTION_KEY:
        try:
            Fernet(value)
        except ValueError as exc:
            raise ServerSecretError(
                f"the stored {name!r} key in server_secrets is not a valid Fernet key; "
                f"set {variable} to the key that encrypted the provider API keys"
            ) from exc
    return value
