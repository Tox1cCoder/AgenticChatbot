"""Phase-0 RED characterization: image streaming through the SIDECAR proxy.

These tests drive ``client_backend.main.create_app()`` (the local sidecar on
port 8100) end to end, through BOTH sidecar streaming routes:

* ``POST /messages/stream`` -> ``ServerAPIClient.stream_internal_message``
  (the internal Streamlit SSE proxy), and
* ``POST /api/chat/{conversation_id}`` -> ``ServerAPIClient.stream_ai_sdk_chat``
  (the AI SDK UI message stream proxy).

For the AI-SDK- and internal-route tests, the upstream (canonical) stream is
produced by the REAL canonical adapter (chained from
``tests.test_image_stream_http_contract``), parsed exactly as
``ServerAPIClient.stream_sse`` parses it (dict events, upstream ``[DONE]``
consumed), and replayed through a fake upstream client so the sidecar's real
``_build_sse_response`` proxy path runs. No adapter helper is called directly.

The ``[DONE]`` dedup lock is the one exception: it drives a genuine upstream
``[DONE]`` (plus trailing junk) through the REAL ``ServerAPIClient.stream_sse``
via a mocked httpx transport (not the fake upstream client), so the actual
break-on-``[DONE]`` logic in ``stream_sse`` runs, not just the sidecar's own
hardcoded trailing yield.

Characterized contracts:

* The sidecar must preserve exactly one ``[DONE]`` across the proxy boundary,
  even when a genuine upstream ``[DONE]`` (plus trailing junk after it) must
  be discarded by the real ``stream_sse`` dedup/break logic (already correct
  on current source — a contract lock).
* A terminal ``file`` part must carry the protected ``/chat-images/{id}`` URL
  through the proxy so the desktop client can fetch it with credentials
  (currently RED: the canonical adapter already mangled it to a corrupt
  ``data:`` URL, and it survives the proxy still-broken).
* An oversized FINAL image must surface an early reference-delivery
  ``image_preview`` through the internal-SSE proxy route too, not just the AI
  SDK route (currently RED: the canonical adapter already drops it before the
  sidecar ever sees it, and the streamed reference cannot then be fetched
  through the sidecar either).
* ``GET /chat-images/{id}`` must exist on the sidecar and proxy the protected
  bytes (currently RED: the route is absent, so the streamed reference cannot
  be fetched through port 8100).

Expected terminal state: RED for the media-read, protected-URL, and
internal-route oversized-final contracts; GREEN for the ``[DONE]`` dedup and
partial/file-ordering contract locks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient

from app.core.container import Container
from client_backend.api import common as common_api
from client_backend.api import messages as messages_api
from client_backend.core import auth as client_auth
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.main import create_app
from client_backend.services.server_api import ServerAPIClient
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


def _capture_canonical_internal_events(image_url: str):
    """Return (upstream_events, raw_payloads) for the image scenario, captured
    from the REAL canonical internal Streamlit SSE route
    (``POST /messages/stream``) instead of the AI SDK route.

    Mirrors ``_capture_canonical_ai_sdk_events`` exactly, so the same
    ``ImagePreviewPublisher`` drop defect is captured pre-proxy: the oversized
    FINAL image never produces an ``image_preview`` (status=final) event, even
    before the sidecar gets involved.
    """
    conversation_id = uuid4()
    user_id = uuid4()
    service = _canonical_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )
    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_canonical_app(service, user_id))
        resp = client.post(
            "/messages/stream",
            json={
                "conversation_id": str(conversation_id),
                "content": "draw a cat",
                "role": 1,
            },
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


class _FakeMediaResponse:
    """Minimal stand-in for the upstream ``httpx.Response`` of a media read.

    Exposes only the surface the sidecar media proxy consumes: a status code,
    a header mapping, and a chunked ``aiter_bytes`` body — so the proxy streams
    rather than buffering the whole payload.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ):
        self.status_code = status_code
        self.headers = headers if headers is not None else {"content-type": "image/png"}
        self._chunks = chunks if chunks is not None else [b"image-bytes"]

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeMediaStream:
    """Async context manager the fake upstream client returns for a media read.

    Records ``__aexit__`` so a test can prove the upstream stream is closed on
    cancellation or early error (no dangling upstream connection / no leak).
    """

    def __init__(self, response: _FakeMediaResponse):
        self._response = response
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _FakeMediaResponse:
        self.entered = True
        return self._response

    async def __aexit__(self, *exc) -> bool:
        self.exited = True
        return False


class _FakeServerClient:
    def __init__(self, events: list[dict]):
        self._events = events
        self.ai_sdk_calls: list[tuple[str, dict]] = []
        self.internal_calls: list[dict] = []
        self.media_calls: list[tuple[str, str]] = []
        self.last_media_stream: _FakeMediaStream | None = None

    async def stream_ai_sdk_chat(self, conversation_id: str, payload: dict):
        self.ai_sdk_calls.append((conversation_id, payload))
        for event in self._events:
            yield event

    async def stream_internal_message(self, payload: dict):
        self.internal_calls.append(payload)
        for event in self._events:
            yield event

    def stream_media(self, path: str, *, method: str = "GET", headers: dict | None = None):
        # Default: a proxied 200 read so route-existence checks see a non-404
        # without any real network. Dedicated media tests use _FakeMediaClient
        # for precise status/header/chunk control.
        self.media_calls.append((method, path))
        stream = _FakeMediaStream(_FakeMediaResponse())
        self.last_media_stream = stream
        return stream


class _FakeMediaClient:
    """Upstream client stub exposing only the media streaming seam, with full
    control over the upstream status, headers, and body chunks."""

    def __init__(self, response: _FakeMediaResponse):
        self._response = response
        self.requested: list[tuple[str, str]] = []
        self.stream: _FakeMediaStream | None = None

    def stream_media(self, path: str, *, method: str = "GET", headers: dict | None = None):
        self.requested.append((method, path))
        self.stream = _FakeMediaStream(self._response)
        return self.stream


class _UnauthedAuthStub:
    """Upstream auth service with no active session (rejects every session)."""

    def is_authenticated(self) -> bool:
        return False

    def get_current_user_id(self):
        return None

    def get_current_access_token(self):
        return None

    async def restore_session(self, _user_id: str) -> bool:
        return False


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


def _sidecar_app_with_server_client(monkeypatch, server_client) -> TestClient:
    """Build the real sidecar app with ``server_client`` wired in as the
    upstream client. Accepts either ``_FakeServerClient`` (raw dict replay) or
    a REAL ``ServerAPIClient`` (e.g. with a mocked httpx transport), so the
    real ``stream_sse`` dedup logic can be exercised when needed.
    """
    monkeypatch.setattr(messages_api, "get_server_client", lambda: server_client)
    monkeypatch.setattr(messages_api, "get_upstream_auth_service", lambda: _AuthStub())
    monkeypatch.setattr(messages_api, "get_runtime_bridge", lambda: _BridgeStub())
    monkeypatch.setattr(common_api, "get_runtime_bridge", lambda: _BridgeStub())
    monkeypatch.setattr(common_api, "get_server_client", lambda: server_client)
    app = create_app()
    app.dependency_overrides[require_local_session] = lambda: _session()
    return TestClient(app)


def _sidecar_client(monkeypatch, events: list[dict]):
    fake = _FakeServerClient(events)
    client = _sidecar_app_with_server_client(monkeypatch, fake)
    return client, fake


def _mock_sse_transport(sse_body: str) -> httpx.MockTransport:
    """An httpx transport that returns a fixed SSE body regardless of request,
    so ``ServerAPIClient.stream_sse`` runs its REAL line-parsing/dedup logic
    against deterministic bytes instead of a real network call."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode("utf-8"),
        )

    return httpx.MockTransport(handler)


def _real_server_client_with_sse_body(sse_body: str) -> ServerAPIClient:
    """A REAL ``ServerAPIClient`` whose HTTP transport is mocked, so
    ``stream_sse`` (the actual break-on-``[DONE]`` / dedup logic at
    ``server_api.py:337-341``) runs unmodified against a deterministic
    upstream SSE body. Unlike ``_FakeServerClient``, this does not bypass
    ``stream_sse`` — it only bypasses the network.
    """
    client = ServerAPIClient(base_url="http://fake-upstream.test")
    client._client = httpx.AsyncClient(
        base_url=client.base_url,
        transport=_mock_sse_transport(sse_body),
    )
    return client


def _sizes_banner() -> str:
    return (
        f"[sizes] partial_b64={len(PARTIAL_B64)} chars; "
        f"final_b64={len(FINAL_B64)} chars (>4,000,000 preview cap)"
    )


# ---------------------------------------------------------------------------
# AI SDK proxy contracts
# ---------------------------------------------------------------------------


def test_ai_sdk_proxy_dedupes_genuine_upstream_done(monkeypatch):
    """Contract lock (GREEN): the REAL ``ServerAPIClient.stream_sse`` breaks on
    the first upstream ``data: [DONE]`` line — it never yields ``[DONE]``
    itself and never reads past it — and the sidecar appends its own single
    trailing ``[DONE]`` for the AI SDK route. A client of the sidecar
    therefore sees exactly one ``[DONE]``, ending the stream, even when the
    upstream body carries a genuine ``[DONE]`` followed by more (buggy) data.

    Unlike a version of this test that replays a pre-filtered event list
    through ``_FakeServerClient`` (which never yields upstream ``[DONE]`` at
    all), this drives a real upstream SSE body through the REAL
    ``ServerAPIClient.stream_sse``, so the actual dedup/break logic is what is
    being locked, not just the sidecar's own hardcoded trailing yield.
    """
    sse_body = (
        'data: {"type": "data-image-preview", "data": {"status": "partial"}}\n\n'
        "data: [DONE]\n\n"
        # Anything upstream sends after its own [DONE] must never surface:
        # stream_sse must break on the first [DONE] rather than continuing to
        # read/yield further lines.
        'data: {"type": "data-image-preview", "data": {"status": "final"}}\n\n'
        "data: [DONE]\n\n"
    )
    server_client = _real_server_client_with_sse_body(sse_body)
    client = _sidecar_app_with_server_client(monkeypatch, server_client)

    resp = client.post(
        f"/api/chat/{uuid4()}",
        json={"messages": [{"role": "user", "content": "draw a cat"}]},
    )
    assert resp.status_code == 200, resp.text
    payloads = _parse_sse(resp.text)
    order = _types(payloads)

    done_count = order.count("[DONE]")
    assert done_count == 1, (
        "expected the real stream_sse break-on-[DONE] logic plus the "
        f"sidecar's own trailing yield to produce exactly one [DONE]; order={order}"
    )
    assert order[-1] == "[DONE]", f"proxied stream must end on [DONE]; order={order}"

    statuses = [
        (p.get("data") or {}).get("status")
        for p in payloads
        if isinstance(p, dict) and p.get("type") == "data-image-preview"
    ]
    assert statuses == ["partial"], (
        "the 'final' event sent after upstream's own [DONE] must never "
        f"surface: stream_sse must stop reading at the first [DONE]; "
        f"statuses={statuses} order={order}"
    )


def test_ai_sdk_proxy_preserves_partial_preview_and_file_ordering(monkeypatch):
    """Contract lock (GREEN): the 128 KiB partial ``data-image-preview`` and a
    terminal ``file`` part both survive the proxy, with the preview arriving
    strictly before the terminal file part in the proxied event order."""
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

    preview_index = order.index("data-image-preview")
    file_index = order.index("file")
    assert preview_index < file_index, (
        "expected the partial data-image-preview to precede the terminal file "
        f"part through the proxy; preview_index={preview_index} "
        f"file_index={file_index} order={order}"
    )


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
# Internal SSE proxy contracts
# ---------------------------------------------------------------------------


def test_internal_sse_proxy_delivers_oversized_final_image_early_by_reference(monkeypatch):
    """RED: driven end-to-end through the sidecar's internal-SSE proxy route
    (``POST /messages/stream`` -> ``ServerAPIClient.stream_internal_message``),
    the oversized FINAL image must still surface an early reference-delivery
    ``image_preview`` (status=final) before the terminal ``complete``. Current
    source drops it in the REAL canonical ``ImagePreviewPublisher`` before the
    sidecar ever sees it, so only the 128 KiB partial preview is proxied.

    Mirrors ``test_ai_sdk_proxy_preserves_protected_reference_url`` /
    ``tests.test_image_stream_http_contract.
    test_internal_sse_delivers_oversized_final_image_early_by_reference``, but
    exercises the sidecar's internal-SSE route specifically — the seam
    ``_FakeServerClient.stream_internal_message`` exists for, which no prior
    test in this module called.
    """
    image_url = f"/chat-images/{uuid4()}"
    upstream, _ = _capture_canonical_internal_events(image_url)
    client, fake = _sidecar_client(monkeypatch, upstream)

    resp = client.post(
        "/messages/stream",
        json={
            "conversation_id": str(uuid4()),
            "content": "draw a cat",
            "role": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    assert fake.internal_calls, (
        "sidecar's /messages/stream route must invoke "
        f"ServerAPIClient.stream_internal_message; internal_calls={fake.internal_calls}"
    )

    payloads = _parse_sse(resp.text)
    order = _types(payloads)
    previews = [p for p in payloads if isinstance(p, dict) and p.get("type") == "image_preview"]
    statuses = [p.get("status") for p in previews]

    assert any(s == "partial" for s in statuses), (
        "expected the 128 KiB partial preview to survive the internal-SSE proxy.\n"
        f"{_sizes_banner()}\nevent order: {order}"
    )

    # Companion of test_sidecar_exposes_chat_image_read_route: through this
    # same internal-SSE path, the streamed protected reference is now fetchable
    # because plan T004 added the sidecar /chat-images media proxy. The route
    # forwards the credentialed upstream read, so the reference no longer 404s
    # at the local origin.
    image_resp = client.get(
        image_url,
        headers={"Authorization": "Bearer local-session-token"},
    )
    assert image_resp.status_code != 404, (
        "The sidecar /chat-images media proxy (plan T004) must let a streamed "
        f"protected reference be fetched through port 8100; GET {image_url} "
        f"returned {image_resp.status_code}."
    )
    assert fake.media_calls == [("GET", image_url)], (
        "the sidecar media route must proxy the exact upstream chat-image path; "
        f"media_calls={fake.media_calls}"
    )

    # Characterized defect: oversized final is dropped upstream -> no early
    # reference event survives through the sidecar's internal-SSE proxy.
    assert any(s == "final" for s in statuses), (
        "DEFECT (emitter.py:70-78, exercised end-to-end through "
        "client_backend/api/messages.py:194-202 -> "
        "ServerAPIClient.stream_internal_message): the oversized FINAL image "
        "produced NO early reference-delivery `image_preview` event through the "
        "sidecar's internal-SSE proxy route either; it only arrives with the "
        "terminal `complete`, violating FR-IMG-002/FR-IMG-003.\n"
        f"{_sizes_banner()}\nimage_preview statuses seen: {statuses}\nevent order: {order}"
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


# ---------------------------------------------------------------------------
# Protected media read route — behavioural contracts (plan T004)
# ---------------------------------------------------------------------------


def _media_sidecar(monkeypatch, response: _FakeMediaResponse, *, authed: bool = True):
    """Build the sidecar wired to a fake upstream media client.

    When ``authed`` is True the local-session gate is satisfied so the route
    body runs; when False the real ``require_local_session`` gate stays in
    place (with an unauthenticated upstream auth service) so its 401 semantics
    can be exercised.
    """
    fake = _FakeMediaClient(response)
    monkeypatch.setattr(common_api, "get_server_client", lambda: fake)
    monkeypatch.setattr(client_auth, "get_upstream_auth_service", lambda: _UnauthedAuthStub())
    app = create_app()
    if authed:
        app.dependency_overrides[require_local_session] = lambda: _session()
    return TestClient(app), fake


def test_media_route_streams_owner_image_with_mime_and_cache_headers(monkeypatch):
    """Owner success: the route streams the upstream bytes, preserves the exact
    upstream content-type, forwards cache validators, and stamps hardening
    headers (nosniff + a restrictive CSP)."""
    response = _FakeMediaResponse(
        status_code=200,
        headers={
            "content-type": "image/webp",
            "etag": '"v1-abc"',
            "last-modified": "Wed, 23 Jul 2026 10:00:00 GMT",
            "cache-control": "private, max-age=60",
        },
        chunks=[b"WEBP-", b"chunk-", b"tail"],
    )
    client, fake = _media_sidecar(monkeypatch, response)
    image_id = uuid4()

    resp = client.get(f"/chat-images/{image_id}", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 200
    assert resp.content == b"WEBP-chunk-tail"
    assert resp.headers["content-type"] == "image/webp"
    assert resp.headers["etag"] == '"v1-abc"'
    assert resp.headers["last-modified"] == "Wed, 23 Jul 2026 10:00:00 GMT"
    assert resp.headers["cache-control"] == "private, max-age=60"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "content-security-policy" in resp.headers
    assert fake.requested == [("GET", f"/chat-images/{image_id}")]


def test_media_route_other_user_gets_404_without_existence_leak(monkeypatch):
    """Attacker direction: the canonical server scopes reads per user and 404s
    another user's image. The sidecar must forward the 404 and NEVER leak the
    upstream body / internal storage path (no existence oracle)."""
    response = _FakeMediaResponse(
        status_code=404,
        headers={"content-type": "application/json"},
        chunks=[b'{"detail":"/srv/app/var/storage/chat_images/other-user-secret.png missing"}'],
    )
    client, _fake = _media_sidecar(monkeypatch, response)

    resp = client.get(f"/chat-images/{uuid4()}", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 404
    body = resp.text
    assert "/srv/" not in body and "storage" not in body, (
        f"upstream internal path must not leak to the caller; body={body!r}"
    )
    assert resp.json()["detail"] == "Image not found"


def test_media_route_missing_token_is_401_and_never_contacts_upstream(monkeypatch):
    """A caller with no local session token is rejected by the sidecar gate
    BEFORE any upstream fetch — so an unauthenticated caller cannot even probe
    whether an image id exists."""
    fake = _FakeMediaClient(_FakeMediaResponse())
    monkeypatch.setattr(common_api, "get_server_client", lambda: fake)
    monkeypatch.setattr(client_auth, "get_upstream_auth_service", lambda: _UnauthedAuthStub())
    client = TestClient(create_app())

    resp = client.get(f"/chat-images/{uuid4()}")

    assert resp.status_code == 401
    assert fake.stream is None, "upstream must not be contacted for an unauthenticated read"


def test_media_route_invalid_bearer_is_401_and_never_contacts_upstream(monkeypatch):
    """A bogus (non-session, non-JWT) bearer with no active upstream session is
    rejected without contacting upstream."""
    fake = _FakeMediaClient(_FakeMediaResponse())
    monkeypatch.setattr(common_api, "get_server_client", lambda: fake)
    monkeypatch.setattr(client_auth, "get_upstream_auth_service", lambda: _UnauthedAuthStub())
    client = TestClient(create_app())

    resp = client.get(
        f"/chat-images/{uuid4()}",
        headers={"Authorization": "Bearer not-a-valid-session-token"},
    )

    assert resp.status_code == 401
    assert fake.stream is None, "upstream must not be contacted for an invalid session token"


async def test_media_route_cancellation_closes_upstream_stream(monkeypatch):
    """Consumer disconnect mid-stream: closing the response body iterator (what
    Starlette does when the client goes away) must close the upstream stream so
    no upstream connection is leaked."""
    response = _FakeMediaResponse(
        status_code=200,
        headers={"content-type": "image/png"},
        chunks=[b"a", b"b", b"c", b"d"],
    )
    fake = _FakeMediaClient(response)
    monkeypatch.setattr(common_api, "get_server_client", lambda: fake)

    streaming = await common_api.proxy_media_request(upstream_path="/chat-images/abc")
    body = streaming.body_iterator
    first = await body.__anext__()
    assert first == b"a"
    assert fake.stream.exited is False

    await body.aclose()  # simulate consumer disconnect before the body drains

    assert fake.stream.exited is True, "upstream media stream must be closed on cancellation"


def test_media_route_payload_at_configured_maximum_succeeds(monkeypatch):
    """A body whose declared length is exactly the configured maximum streams
    through unchanged — the size guard rejects only what EXCEEDS the max."""
    monkeypatch.setattr(common_api, "MAX_MEDIA_PROXY_BYTES", 8)
    body = b"01234567"  # exactly 8 bytes
    response = _FakeMediaResponse(
        status_code=200,
        headers={"content-type": "image/png", "content-length": "8"},
        chunks=[body],
    )
    client, _fake = _media_sidecar(monkeypatch, response)

    resp = client.get(f"/chat-images/{uuid4()}", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 200
    assert resp.content == body


def test_media_route_payload_over_configured_maximum_is_413(monkeypatch):
    """A declared length over the configured maximum is rejected with 413
    without draining the oversized body (the upstream stream is closed)."""
    monkeypatch.setattr(common_api, "MAX_MEDIA_PROXY_BYTES", 8)
    response = _FakeMediaResponse(
        status_code=200,
        headers={"content-type": "image/png", "content-length": "9"},
        chunks=[b"012345678"],
    )
    client, fake = _media_sidecar(monkeypatch, response)

    resp = client.get(f"/chat-images/{uuid4()}", headers={"Authorization": "Bearer x"})

    assert resp.status_code == 413
    assert fake.stream.exited is True, "oversized upstream stream must be closed, not drained"


async def test_media_route_undeclared_oversized_body_is_bounded_mid_stream(monkeypatch):
    """Closes the T004 review Minor: the size guard must not depend on the
    upstream DECLARING a content-length.

    A chunked/undeclared upstream response bypasses the pre-check entirely, so
    without a running byte counter the sidecar would relay an unbounded body.
    The proxy must stop reading once the cap is exceeded and close the upstream
    stream.
    """
    monkeypatch.setattr(common_api, "MAX_MEDIA_PROXY_BYTES", 8)
    response = _FakeMediaResponse(
        status_code=200,
        headers={"content-type": "image/png"},  # no content-length: chunked
        chunks=[b"0123", b"4567", b"89ab", b"cdef"],
    )
    fake = _FakeMediaClient(response)
    monkeypatch.setattr(common_api, "get_server_client", lambda: fake)

    streaming = await common_api.proxy_media_request(upstream_path="/chat-images/abc")

    relayed = bytearray()
    with pytest.raises(common_api.MediaTooLargeError):
        async for chunk in streaming.body_iterator:
            relayed.extend(chunk)

    assert len(relayed) <= 8, (
        "the sidecar relayed more than the configured maximum from an upstream "
        f"that declared no content-length; relayed={len(relayed)} bytes"
    )
    assert fake.stream.exited is True, "upstream stream must be closed when the cap trips"


async def test_media_route_undeclared_body_within_cap_streams_fully(monkeypatch):
    """The byte counter must not truncate a legitimate chunked body at the cap."""
    monkeypatch.setattr(common_api, "MAX_MEDIA_PROXY_BYTES", 8)
    response = _FakeMediaResponse(
        status_code=200,
        headers={"content-type": "image/png"},
        chunks=[b"0123", b"4567"],  # exactly 8 bytes, undeclared
    )
    fake = _FakeMediaClient(response)
    monkeypatch.setattr(common_api, "get_server_client", lambda: fake)

    streaming = await common_api.proxy_media_request(upstream_path="/chat-images/abc")
    relayed = b"".join([chunk async for chunk in streaming.body_iterator])

    assert relayed == b"01234567"
