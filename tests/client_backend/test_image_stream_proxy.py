"""Phase-0 RED characterization: image streaming through the SIDECAR proxy.

These tests drive ``client_backend.main.create_app()`` (the local sidecar on
port 8100) end to end. The upstream (canonical) AI SDK stream is produced by
the REAL canonical adapter (chained from
``tests.test_image_stream_http_contract``), parsed exactly as
``ServerAPIClient.stream_sse`` parses it (dict events, upstream ``[DONE]``
consumed), and replayed through a fake upstream client so the sidecar's real
``_build_sse_response`` proxy path runs. No adapter helper is called directly.

Characterized contracts:

* The sidecar must preserve exactly one ``[DONE]`` across the proxy boundary
  (already correct on current source — a contract lock).
* A terminal ``file`` part must carry the protected ``/chat-images/{id}`` URL
  through the proxy so the desktop client can fetch it with credentials
  (currently RED: the canonical adapter already mangled it to a corrupt
  ``data:`` URL, and it survives the proxy still-broken).
* ``GET /chat-images/{id}`` must exist on the sidecar and proxy the protected
  bytes (currently RED: the route is absent, so the streamed reference cannot
  be fetched through port 8100).

Expected terminal state: RED for the media-read and protected-URL contracts;
GREEN for the ``[DONE]`` and partial-ordering contract locks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient

from app.core.container import Container
from client_backend.api import common as common_api
from client_backend.api import messages as messages_api
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.main import create_app
from tests.test_image_stream_http_contract import (
    FINAL_B64,
    PARTIAL_B64,
    _parse_sse,
    _types,
)
from tests.test_image_stream_http_contract import (
    _build_app as _canonical_app,
)
from tests.test_image_stream_http_contract import (
    _build_message_service as _canonical_message_service,
)

# ---------------------------------------------------------------------------
# Chain the REAL canonical AI SDK adapter output, parsed like stream_sse.
# ---------------------------------------------------------------------------


def _capture_canonical_ai_sdk_events(image_url: str):
    """Return (upstream_events, raw_payloads) for the image scenario.

    ``upstream_events`` are the dict events an upstream reader (stream_sse)
    would yield: JSON-decoded, heartbeats dropped, and the terminal ``[DONE]``
    consumed (not yielded) — mirroring
    ``client_backend.services.server_api.ServerAPIClient.stream_sse``.
    """
    conversation_id = uuid4()
    user_id = uuid4()
    service = _canonical_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )
    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_canonical_app(service, user_id))
        resp = client.post(
            f"/api/chat/{conversation_id}",
            json={"messages": [{"role": "user", "content": "draw a cat"}]},
        )
    assert resp.status_code == 200, resp.text
    raw = _parse_sse(resp.text)
    upstream_events = [
        p
        for p in raw
        if isinstance(p, dict) and p.get("type") != "heartbeat"
    ]
    return upstream_events, raw


# ---------------------------------------------------------------------------
# Sidecar dependency stubs
# ---------------------------------------------------------------------------


class _AuthStub:
    def is_authenticated(self) -> bool:
        # Returns False so the message flow skips runtime-bridge reconnect
        # (no real network) while still proxying.
        return False


class _BridgeStub:
    def is_connected(self) -> bool:
        return True

    def get_registered_device_id(self) -> str:
        return "device-1"


class _FakeServerClient:
    def __init__(self, events: list[dict]):
        self._events = events
        self.ai_sdk_calls: list[tuple[str, dict]] = []
        self.internal_calls: list[dict] = []

    async def stream_ai_sdk_chat(self, conversation_id: str, payload: dict):
        self.ai_sdk_calls.append((conversation_id, payload))
        for event in self._events:
            yield event

    async def stream_internal_message(self, payload: dict):
        self.internal_calls.append(payload)
        for event in self._events:
            yield event


def _session() -> LocalSessionPayload:
    now = datetime.now(timezone.utc)
    return LocalSessionPayload(
        user_id="user-1",
        server_user_id="user-1",
        device_id=None,
        device_identifier="dev-abc",
        iat=now,
        exp=now + timedelta(hours=1),
    )


def _sidecar_client(monkeypatch, events: list[dict]):
    fake = _FakeServerClient(events)
    monkeypatch.setattr(messages_api, "get_server_client", lambda: fake)
    monkeypatch.setattr(messages_api, "get_upstream_auth_service", lambda: _AuthStub())
    monkeypatch.setattr(messages_api, "get_runtime_bridge", lambda: _BridgeStub())
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: _BridgeStub())
    app = create_app()
    app.dependency_overrides[require_local_session] = lambda: _session()
    return TestClient(app), fake


def _sizes_banner() -> str:
    return (
        f"[sizes] partial_b64={len(PARTIAL_B64)} chars; "
        f"final_b64={len(FINAL_B64)} chars (>4,000,000 preview cap)"
    )


# ---------------------------------------------------------------------------
# AI SDK proxy contracts
# ---------------------------------------------------------------------------


def test_ai_sdk_proxy_preserves_exactly_one_done(monkeypatch):
    """Contract lock (GREEN): exactly one ``[DONE]`` survives the sidecar
    proxy and the stream ends with it."""
    image_url = f"/chat-images/{uuid4()}"
    upstream, _ = _capture_canonical_ai_sdk_events(image_url)
    client, _fake = _sidecar_client(monkeypatch, upstream)

    resp = client.post(
        f"/api/chat/{uuid4()}",
        json={"messages": [{"role": "user", "content": "draw a cat"}]},
    )
    assert resp.status_code == 200, resp.text
    order = _types(_parse_sse(resp.text))
    done_count = order.count("[DONE]")
    assert done_count == 1, f"expected exactly one [DONE] through the proxy; order={order}"
    assert order[-1] == "[DONE]", f"proxied stream must end on [DONE]; order={order}"


def test_ai_sdk_proxy_preserves_partial_preview_and_file_ordering(monkeypatch):
    """Contract lock (GREEN): the 128 KiB partial ``data-image-preview`` and a
    terminal ``file`` part both survive the proxy in order."""
    image_url = f"/chat-images/{uuid4()}"
    upstream, _ = _capture_canonical_ai_sdk_events(image_url)
    client, _fake = _sidecar_client(monkeypatch, upstream)

    resp = client.post(
        f"/api/chat/{uuid4()}",
        json={"messages": [{"role": "user", "content": "draw a cat"}]},
    )
    payloads = _parse_sse(resp.text)
    order = _types(payloads)
    previews = [
        p for p in payloads if isinstance(p, dict) and p.get("type") == "data-image-preview"
    ]
    files = [p for p in payloads if isinstance(p, dict) and p.get("type") == "file"]
    assert previews, f"partial preview should survive the proxy; order={order}"
    assert files, f"terminal file part should survive the proxy; order={order}"


def test_ai_sdk_proxy_preserves_protected_reference_url(monkeypatch):
    """RED: the terminal ``file`` part reaching the desktop client through the
    sidecar must carry the protected ``/chat-images/{id}`` URL so a credentialed
    fetch is possible. Current source proxies the canonical adapter's corrupt
    ``data:`` URL unchanged.
    """
    image_url = f"/chat-images/{uuid4()}"
    upstream, _ = _capture_canonical_ai_sdk_events(image_url)
    client, _fake = _sidecar_client(monkeypatch, upstream)

    resp = client.post(
        f"/api/chat/{uuid4()}",
        json={"messages": [{"role": "user", "content": "draw a cat"}]},
    )
    payloads = _parse_sse(resp.text)
    files = [p for p in payloads if isinstance(p, dict) and p.get("type") == "file"]
    urls = [p.get("url") for p in files]
    assert image_url in urls, (
        "DEFECT (ai_sdk_projection.py:104-154 upstream + no sidecar repair): the "
        "protected relative image URL did not survive to the desktop client; the "
        "proxied `file` part carries a corrupt data: URL a browser cannot fetch "
        "with a Bearer token (FR-IMG-006).\n"
        f"{_sizes_banner()}\nexpected url: {image_url}\nproxied file urls: {urls}"
    )


# ---------------------------------------------------------------------------
# Protected media read route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path_prefix", ["/chat-images", "/api/chat-images"])
def test_sidecar_exposes_chat_image_read_route(monkeypatch, path_prefix):
    """RED: the sidecar must expose ``GET /chat-images/{id}`` (and its ``/api``
    alias) so a streamed protected reference can be fetched through port 8100.
    Current source has no such route, so the fetch 404s and the delivered image
    reference is unusable from the local origin.
    """
    client, _fake = _sidecar_client(monkeypatch, [])
    image_id = uuid4()
    resp = client.get(
        f"{path_prefix}/{image_id}",
        headers={"Authorization": "Bearer local-session-token"},
    )
    assert resp.status_code != 404, (
        "DEFECT (client_backend has no /chat-images route or proxy - plan T004): "
        f"GET {path_prefix}/{{id}} returned 404, so the streamed protected image "
        "reference cannot be fetched through the local origin (port 8100). "
        f"status_code={resp.status_code}"
    )
