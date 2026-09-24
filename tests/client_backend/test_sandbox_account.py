"""The sandbox account's credentials: generated strong, stored only encrypted."""

from __future__ import annotations

import string

import pytest

from client_backend.services.sandbox import account


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    return tmp_path


def test_generated_password_satisfies_windows_complexity_rules():
    password = account.generate_password()

    assert len(password) >= 24
    assert any(char in string.ascii_lowercase for char in password)
    assert any(char in string.ascii_uppercase for char in password)
    assert any(char in string.digits for char in password)
    assert any(char in account.PASSWORD_SYMBOLS for char in password)


def test_each_generated_password_is_different():
    assert account.generate_password() != account.generate_password()


def test_stored_credentials_round_trip(profile):
    account.save_credentials("KaniSandbox", "S3cret!pass-word-for-test")

    loaded = account.load_credentials()

    assert loaded == account.SandboxCredentials("KaniSandbox", "S3cret!pass-word-for-test")


def test_password_is_never_written_in_plain_text(profile):
    account.save_credentials("KaniSandbox", "S3cret!pass-word-for-test")

    written = b"".join(path.read_bytes() for path in profile.rglob("*") if path.is_file())

    assert written
    assert b"S3cret!pass-word-for-test" not in written


def test_no_credentials_before_setup(profile):
    assert account.load_credentials() is None
