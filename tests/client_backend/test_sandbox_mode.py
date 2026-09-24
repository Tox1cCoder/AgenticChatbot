"""The device sandbox-mode override: the file that lets the app flip the switch.

CLIENT_SANDBOX_MODE stays the default; a valid override file wins. A missing or
unreadable file must fall back to the configured mode rather than fail, so a
corrupt override can never leave Desktop Commander stuck.
"""

from __future__ import annotations

import pytest

from client_backend.services.sandbox import mode


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "profile_root", str(tmp_path))
    return tmp_path


def test_default_is_the_configured_mode_when_no_override(profile, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "sandbox_mode", "workspace")

    assert mode.read_sandbox_mode() == "workspace"


def test_write_then_read_round_trips(profile):
    mode.write_sandbox_mode("workspace")

    assert mode.read_sandbox_mode() == "workspace"


def test_a_device_override_beats_the_configured_default(profile, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "sandbox_mode", "workspace")
    mode.write_sandbox_mode("off")

    assert mode.read_sandbox_mode() == "off"


def test_an_unknown_or_corrupt_override_falls_back_to_the_default(profile, monkeypatch):
    from client_backend.core.config import client_settings

    monkeypatch.setattr(client_settings, "sandbox_mode", "off")
    path = mode.sandbox_mode_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"mode": "banana"} not json', encoding="utf-8")

    assert mode.read_sandbox_mode() == "off"


def test_writing_an_unknown_mode_is_refused(profile):
    with pytest.raises(ValueError, match="sandbox mode"):
        mode.write_sandbox_mode("banana")
