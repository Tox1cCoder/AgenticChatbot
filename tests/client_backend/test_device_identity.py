"""
T014 (client_invocation.md): per-installation device identity (FR-3).

`generate_device_identifier` must return a random identifier persisted in the
installation's config directory: two installations on one machine (two config
directories) are distinct devices; one installation keeps its identity across
restarts; nothing is derived from machine attributes.
"""

from __future__ import annotations

import json

from client_backend.core.security import (
    DEVICE_IDENTITY_FILENAME,
    generate_device_identifier,
)


def test_two_config_directories_yield_distinct_identifiers(tmp_path):
    install_a = tmp_path / "install-a"
    install_b = tmp_path / "install-b"

    identifier_a = generate_device_identifier(install_a)
    identifier_b = generate_device_identifier(install_b)

    assert identifier_a != identifier_b


def test_same_directory_yields_stable_identifier(tmp_path):
    first = generate_device_identifier(tmp_path)
    second = generate_device_identifier(tmp_path)

    assert first == second


def test_identity_file_created_on_first_use(tmp_path):
    identity_path = tmp_path / DEVICE_IDENTITY_FILENAME
    assert not identity_path.exists()

    identifier = generate_device_identifier(tmp_path)

    stored = json.loads(identity_path.read_text(encoding="utf-8"))
    assert stored["device_identifier"] == identifier


def test_corrupt_identity_file_is_regenerated(tmp_path):
    identity_path = tmp_path / DEVICE_IDENTITY_FILENAME
    identity_path.write_text("not-json{", encoding="utf-8")

    identifier = generate_device_identifier(tmp_path)

    assert identifier
    stored = json.loads(identity_path.read_text(encoding="utf-8"))
    assert stored["device_identifier"] == identifier


def test_identifier_is_not_machine_derived(tmp_path):
    """Random per-installation identity: a fresh directory must never
    reproduce another directory's identifier (the old machine-hash did)."""
    identifiers = {generate_device_identifier(tmp_path / f"install-{i}") for i in range(3)}

    assert len(identifiers) == 3
