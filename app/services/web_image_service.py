"""Secure render-time delivery for selected third-party images."""

from __future__ import annotations

import asyncio
import inspect
import ipaddress
import socket
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from urllib.parse import urljoin, urlsplit
from uuid import UUID, uuid4

import httpx
from PIL import Image, UnidentifiedImageError

WEB_IMAGE_URL_PREFIX = "/web-images/"
ALLOWED_WEB_IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


class WebImageRejected(Exception):
    """The selected upstream violates a bounded delivery policy."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class WebImageUpstreamFailure(Exception):
    """The selected upstream could not be reached successfully."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class FetchedWebImage:
    content: bytes
    media_type: str
    width: int
    height: int


class PinnedAsyncTransport(httpx.AsyncHTTPTransport):
    """Connect to a validated IP while retaining the original Host and SNI."""

    def __init__(self, verified_ip: str) -> None:
        self.verified_ip = verified_ip
        super().__init__(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_host = request.url.host
        request.url = request.url.copy_with(host=self.verified_ip)
        request.headers["Host"] = original_host
        request.extensions["sni_hostname"] = original_host
        return await super().handle_async_request(request)


class WebImageService:
    """Register opaque references and securely fetch them only when rendered."""

    def __init__(
        self,
        repository: Any,
        *,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_redirects: int,
        max_bytes: int,
        max_pixels: int,
        resolver: Any | None = None,
        transport_factory: Any | None = None,
    ) -> None:
        self.repository = repository
        self.connect_timeout_seconds = max(0.001, float(connect_timeout_seconds))
        self.read_timeout_seconds = max(0.001, float(read_timeout_seconds))
        self.max_redirects = max(0, int(max_redirects))
        self.max_bytes = max(1, int(max_bytes))
        self.max_pixels = max(1, int(max_pixels))
        self.resolver = resolver or self._resolve_hostname
        self.transport_factory = transport_factory or PinnedAsyncTransport

    async def register(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        upstream_url: str,
        expected_mime: str | None,
        provider: str,
    ) -> Any:
        """Persist metadata only; never contact the upstream here."""
        self._https_hostname(upstream_url)
        normalized_provider = str(provider or "other").strip().lower()[:32] or "other"
        return await self.repository.acreate(
            {
                "id": uuid4(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "upstream_url": upstream_url,
                "expected_mime": expected_mime,
                "provider": normalized_provider,
            }
        )

    async def fetch(self, record: Any) -> FetchedWebImage:
        current_url = self._record_value(record, "upstream_url")
        for redirect_count in range(self.max_redirects + 1):
            outcome = await self._fetch_once(current_url)
            if isinstance(outcome, FetchedWebImage):
                return outcome
            if redirect_count >= self.max_redirects:
                raise WebImageRejected("redirect_limit")
            current_url = urljoin(current_url, outcome)
        raise WebImageRejected("redirect_limit")

    async def _fetch_once(self, url: str) -> FetchedWebImage | str:
        hostname = self._https_hostname(url)
        verified_ip = await self._resolve_public_ip(hostname)
        transport = self.transport_factory(verified_ip)
        timeout = httpx.Timeout(
            connect=self.connect_timeout_seconds,
            read=self.read_timeout_seconds,
            write=self.read_timeout_seconds,
            pool=self.connect_timeout_seconds,
        )
        try:
            async with httpx.AsyncClient(
                transport=transport,
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client, client.stream(
                "GET",
                url,
                headers={
                    "Accept": "image/png,image/jpeg,image/webp,image/gif",
                    "User-Agent": "sample-chatbot-image-fetch/1.0",
                },
            ) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("Location")
                    if not location:
                        raise WebImageUpstreamFailure("status")
                    return location
                if not 200 <= response.status_code < 300:
                    raise WebImageUpstreamFailure("status")
                media_type = self._validated_media_type(response)
                content = await self._read_bounded(response)
        except (WebImageRejected, WebImageUpstreamFailure):
            raise
        except httpx.TimeoutException as exc:
            raise WebImageUpstreamFailure("timeout") from exc
        except httpx.HTTPError as exc:
            raise WebImageUpstreamFailure("network") from exc
        return self._validate_decoded_image(content, media_type)

    async def _resolve_public_ip(self, hostname: str) -> str:
        resolved = self.resolver(hostname)
        answers = await resolved if inspect.isawaitable(resolved) else resolved
        addresses = [str(value).strip() for value in answers or [] if str(value).strip()]
        if not addresses:
            raise WebImageRejected("dns")
        parsed = []
        for address in addresses:
            try:
                candidate = ipaddress.ip_address(address)
            except ValueError as exc:
                raise WebImageRejected("dns") from exc
            if not candidate.is_global:
                raise WebImageRejected("private_address")
            parsed.append(candidate.compressed)
        return parsed[0]

    @staticmethod
    async def _resolve_hostname(hostname: str) -> list[str]:
        loop = asyncio.get_running_loop()
        try:
            answers = await loop.getaddrinfo(
                hostname,
                443,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise WebImageRejected("dns") from exc
        return list(
            dict.fromkeys(
                sockaddr[0]
                for family, _type, _proto, _canonname, sockaddr in answers
                if family in {socket.AF_INET, socket.AF_INET6}
            )
        )

    @staticmethod
    def _https_hostname(url: str) -> str:
        parsed = urlsplit(str(url or "").strip())
        if parsed.scheme.lower() != "https":
            raise WebImageRejected("scheme")
        if not parsed.hostname or parsed.username or parsed.password:
            raise WebImageRejected("url")
        return parsed.hostname

    def _validated_media_type(self, response: httpx.Response) -> str:
        media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if media_type not in ALLOWED_WEB_IMAGE_MIME_TYPES:
            raise WebImageRejected("mime")
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    raise WebImageRejected("size")
            except ValueError:
                pass
        return media_type

    async def _read_bounded(self, response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.max_bytes:
                raise WebImageRejected("size")
            chunks.append(chunk)
        if not chunks:
            raise WebImageRejected("decode")
        return b"".join(chunks)

    def _validate_decoded_image(self, content: bytes, declared_mime: str) -> FetchedWebImage:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(BytesIO(content)) as image:
                    width, height = image.size
                    image_format = str(image.format or "").upper()
                    if width < 1 or height < 1 or width * height > self.max_pixels:
                        raise WebImageRejected("dimensions")
                    image.verify()
        except WebImageRejected:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise WebImageRejected("dimensions") from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise WebImageRejected("decode") from exc
        decoded_mime = _MIME_BY_FORMAT.get(image_format)
        if decoded_mime is None or decoded_mime != declared_mime:
            raise WebImageRejected("mime")
        return FetchedWebImage(
            content=content,
            media_type=decoded_mime,
            width=width,
            height=height,
        )

    @staticmethod
    def _record_value(record: Any, key: str) -> str:
        value = record.get(key) if isinstance(record, dict) else getattr(record, key, None)
        return str(value or "").strip()
