"""The sandbox account's name and credentials.

The password exists so the sidecar can start processes as the account; nobody
signs in with it. It is generated once during setup and kept DPAPI-encrypted
under the signed-in user, which the sandbox account itself cannot decrypt.
"""

from __future__ import annotations

import json
import os
import secrets
import string
from dataclasses import dataclass
from pathlib import Path

from client_backend.core.config import client_settings
from client_backend.core.security import decrypt_local_secret, encrypt_local_secret

SANDBOX_USERNAME = "KaniSandbox"
PASSWORD_SYMBOLS = "!#%*+-=?@^_"
_PASSWORD_LENGTH = 32


@dataclass(frozen=True)
class SandboxCredentials:
    username: str
    password: str


def generate_password() -> str:
    """A random password that meets Windows' complexity rules."""

    alphabet = string.ascii_letters + string.digits + PASSWORD_SYMBOLS
    required = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(PASSWORD_SYMBOLS),
    ]
    rest = [secrets.choice(alphabet) for _ in range(_PASSWORD_LENGTH - len(required))]
    characters = required + rest
    secrets.SystemRandom().shuffle(characters)
    return "".join(characters)


def credentials_path() -> Path:
    # Machine-level, not per chat user: there is one sandbox account per PC.
    return Path(client_settings.profile_root) / "sandbox" / "account.json"


def save_credentials(username: str, password: str) -> None:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "username": username,
        "password": encrypt_local_secret(password.encode("utf-8")),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def load_credentials() -> SandboxCredentials | None:
    path = credentials_path()
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    password = decrypt_local_secret(payload["password"]).decode("utf-8")
    return SandboxCredentials(str(payload["username"]), password)


def forget_credentials() -> None:
    credentials_path().unlink(missing_ok=True)
