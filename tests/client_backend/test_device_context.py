"""
T013 (client_invocation.md): the proxy always asserts its own device identity.

`add_device_context` must stamp every chat payload with the local bridge's
registered device id — overwriting any incoming `device_id`/`deviceId`
(stale UI state or a replay from another machine, the D2 bug signature) —
and strip both keys when the bridge is not connected so the server binds no
client tools instead of trusting a forwarded id.
"""

from __future__ import annotations

import logging

import pytest

from client_backend.api import common as common_api
from client_backend.api.common import add_device_context

OWN_DEVICE_ID = "device-own-123"
FOREIGN_DEVICE_ID = "device-foreign-456"


class _BridgeStub:
    def __init__(self, device_id: str | None):
        self.device_id = device_id

    def is_connected(self) -> bool:
        return self.device_id is not None

    def get_registered_device_id(self) -> str | None:
        return self.device_id


@pytest.fixture
def connected_bridge(monkeypatch):
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: _BridgeStub(OWN_DEVICE_ID))


@pytest.fixture
def disconnected_bridge(monkeypatch):
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: _BridgeStub(None))


def test_absent_device_id_is_injected(connected_bridge):
    result = add_device_context({"messages": []})

    assert result["device_id"] == OWN_DEVICE_ID
    assert "deviceId" not in result


def test_same_device_id_is_kept(connected_bridge, caplog):
    with caplog.at_level(logging.WARNING):
        result = add_device_context({"messages": [], "device_id": OWN_DEVICE_ID})

    assert result["device_id"] == OWN_DEVICE_ID
    assert not caplog.records


def test_foreign_device_id_is_overwritten_with_warning(connected_bridge, caplog):
    with caplog.at_level(logging.WARNING):
        result = add_device_context({"messages": [], "device_id": FOREIGN_DEVICE_ID})

    assert result["device_id"] == OWN_DEVICE_ID
    assert any(FOREIGN_DEVICE_ID in record.getMessage() for record in caplog.records)


def test_foreign_camel_case_key_is_replaced(connected_bridge, caplog):
    with caplog.at_level(logging.WARNING):
        result = add_device_context({"messages": [], "deviceId": FOREIGN_DEVICE_ID})

    assert result["device_id"] == OWN_DEVICE_ID
    assert "deviceId" not in result
    assert any(FOREIGN_DEVICE_ID in record.getMessage() for record in caplog.records)


def test_both_key_styles_are_normalized_to_own_id(connected_bridge):
    result = add_device_context({"device_id": FOREIGN_DEVICE_ID, "deviceId": FOREIGN_DEVICE_ID})

    assert result["device_id"] == OWN_DEVICE_ID
    assert "deviceId" not in result


def test_disconnected_bridge_strips_device_keys(disconnected_bridge):
    result = add_device_context(
        {"messages": [], "device_id": FOREIGN_DEVICE_ID, "deviceId": FOREIGN_DEVICE_ID}
    )

    assert "device_id" not in result
    assert "deviceId" not in result
    assert result["messages"] == []


def test_disconnected_bridge_without_keys_is_a_noop(disconnected_bridge):
    result = add_device_context({"messages": []})

    assert result == {"messages": []}


def test_original_payload_is_not_mutated(connected_bridge):
    payload = {"messages": [], "device_id": FOREIGN_DEVICE_ID}

    add_device_context(payload)

    assert payload["device_id"] == FOREIGN_DEVICE_ID
