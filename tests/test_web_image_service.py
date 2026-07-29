"""Security and latency contract for render-time web-image delivery."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from PIL import Image

from app.services.web_image_service import (
    FetchedWebImage,
    WebImageRejected,
    WebImageService,
    WebImageUpstreamFailure,
)


def _png_bytes(width: int = 20, height: int = 10) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), color="red").save(output, format="PNG")
    return output.getvalue()


def _record(url: str, expected_mime: str = "image/png"):
    return SimpleNamespace(upstream_url=url, expected_mime=expected_mime, provider="tavily")


def _resolver(*answers):
    async def resolve(_hostname):
        return list(answers or ("93.184.216.34",))

    return resolve


def _service(
    *,
    repository=None,
    resolver=None,
    responses=None,
    handler=None,
    max_bytes=1024 * 1024,
    max_pixels=1_000_000,
    max_redirects=3,
    transport_factory=None,
):
    response_queue = list(responses or [])
    seen_ips: list[str] = []

    async def queued_handler(request):
        if handler is not None:
            return await handler(request)
        return response_queue.pop(0)

    def make_transport(ip):
        seen_ips.append(ip)
        return httpx.MockTransport(queued_handler)

    service = WebImageService(
        repository or AsyncMock(),
        connect_timeout_seconds=0.1,
        read_timeout_seconds=0.1,
        max_redirects=max_redirects,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        resolver=resolver or _resolver(),
        transport_factory=transport_factory or make_transport,
    )
    return service, seen_ips


@pytest.mark.asyncio
async def test_register_persists_metadata_without_touching_network():
    repository = AsyncMock()
    repository.acreate.return_value = SimpleNamespace(id=uuid4())
    transport_factory = Mock(side_effect=AssertionError("network must not run"))
    service, _ = _service(repository=repository, transport_factory=transport_factory)
    conversation_id = uuid4()
    user_id = uuid4()

    result = await service.register(
        conversation_id=conversation_id,
        user_id=user_id,
        upstream_url="https://img.example/image.jpg",
        expected_mime="image/jpeg",
        provider="brave",
    )

    assert result is repository.acreate.return_value
    repository.acreate.assert_awaited_once()
    transport_factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "http://img.example/image.png",
        "https://127.0.0.1/image.png",
        "https://169.254.169.254/latest/meta-data",
        "https://[::1]/image.png",
    ),
)
async def test_fetch_rejects_insecure_or_non_public_targets(url):
    service, _ = _service(resolver=_resolver("127.0.0.1"))

    with pytest.raises(WebImageRejected) as error:
        await service.fetch(_record(url))

    assert error.value.reason in {"scheme", "private_address"}


@pytest.mark.asyncio
async def test_fetch_rejects_mixed_public_and_private_dns_answers():
    service, _ = _service(resolver=_resolver("93.184.216.34", "10.0.0.4"))

    with pytest.raises(WebImageRejected, match="private_address"):
        await service.fetch(_record("https://img.example/image.png"))


@pytest.mark.asyncio
async def test_redirect_target_is_resolved_and_private_target_is_rejected():
    calls = 0

    async def resolver(_hostname):
        nonlocal calls
        calls += 1
        return ["93.184.216.34"] if calls == 1 else ["127.0.0.1"]

    service, seen_ips = _service(
        resolver=resolver,
        responses=[httpx.Response(302, headers={"Location": "https://private.example/x"})],
    )

    with pytest.raises(WebImageRejected, match="private_address"):
        await service.fetch(_record("https://public.example/image.png"))

    assert seen_ips == ["93.184.216.34"]


@pytest.mark.asyncio
async def test_redirect_limit_is_enforced_per_hop():
    service, _ = _service(
        max_redirects=1,
        responses=[
            httpx.Response(302, headers={"Location": "https://img.example/two"}),
            httpx.Response(302, headers={"Location": "https://img.example/three"}),
        ],
    )

    with pytest.raises(WebImageRejected, match="redirect_limit"):
        await service.fetch(_record("https://img.example/one"))


@pytest.mark.asyncio
async def test_fetch_rejects_non_image_mime_before_rendering():
    service, _ = _service(
        responses=[httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"no")]
    )

    with pytest.raises(WebImageRejected, match="mime"):
        await service.fetch(_record("https://img.example/image.png"))


@pytest.mark.asyncio
async def test_fetch_rejects_header_that_disagrees_with_decoded_format():
    service, _ = _service(
        responses=[
            httpx.Response(
                200,
                headers={"Content-Type": "image/jpeg"},
                content=_png_bytes(),
            )
        ]
    )

    with pytest.raises(WebImageRejected, match="mime"):
        await service.fetch(_record("https://img.example/image.jpg", "image/jpeg"))


@pytest.mark.asyncio
async def test_fetch_stops_when_streamed_bytes_exceed_cap():
    service, _ = _service(
        max_bytes=8,
        responses=[
            httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=b"123456789",
            )
        ],
    )

    with pytest.raises(WebImageRejected, match="size"):
        await service.fetch(_record("https://img.example/image.png"))


@pytest.mark.asyncio
async def test_fetch_rejects_corrupt_image_bytes():
    service, _ = _service(
        responses=[
            httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=b"not-a-png",
            )
        ]
    )

    with pytest.raises(WebImageRejected, match="decode"):
        await service.fetch(_record("https://img.example/image.png"))


@pytest.mark.asyncio
async def test_fetch_rejects_decoded_dimensions_over_pixel_cap():
    service, _ = _service(
        max_pixels=100,
        responses=[
            httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=_png_bytes(20, 10),
            )
        ],
    )

    with pytest.raises(WebImageRejected, match="dimensions"):
        await service.fetch(_record("https://img.example/image.png"))


@pytest.mark.asyncio
async def test_fetch_maps_http_timeout_to_bounded_upstream_failure():
    async def timeout_handler(request):
        raise httpx.ReadTimeout("secret upstream detail", request=request)

    service, _ = _service(handler=timeout_handler)

    with pytest.raises(WebImageUpstreamFailure) as error:
        await service.fetch(_record("https://img.example/image.png"))

    assert error.value.reason == "timeout"
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_valid_png_returns_verified_bytes_dimensions_and_pinned_ip():
    content = _png_bytes(20, 10)
    service, seen_ips = _service(
        responses=[
            httpx.Response(
                200,
                headers={"Content-Type": "image/png"},
                content=content,
            )
        ]
    )

    result = await service.fetch(_record("https://img.example/image.png"))

    assert result == FetchedWebImage(
        content=content,
        media_type="image/png",
        width=20,
        height=10,
    )
    assert seen_ips == ["93.184.216.34"]
