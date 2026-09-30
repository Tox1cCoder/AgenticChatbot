"""Refuse an upload by its declared length, before the body is read.

FastAPI parses a multipart form -- Starlette spooling every byte to disk --
before any dependency or the endpoint runs, so a size limit checked there comes
too late. A route class runs first. A body with no declared length (chunked)
could not be bounded here at all, so it is refused; browsers and HTTP clients
declare one for a file form.
"""

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request, Response
from fastapi.routing import APIRoute

# Room for the multipart boundary and part headers around the file bytes; the
# limit a route names is for the content, not the envelope.
MULTIPART_ENVELOPE_BYTES = 64 * 1024


def declared_size_route(
    *,
    max_bytes: Callable[[], int],
    refuse: Callable[[int], Response],
) -> type[APIRoute]:
    """Build a route class that answers ``refuse(411)`` or ``refuse(413)`` unread.

    ``max_bytes`` is read per request, so a test or a settings reload changes the
    limit without rebuilding the router.
    """

    class DeclaredSizeRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()

            async def bounded_handler(request: Request) -> Response:
                declared = request.headers.get("content-length", "")
                if not declared.isdigit():
                    return refuse(411)
                if int(declared) > int(max_bytes()) + MULTIPART_ENVELOPE_BYTES:
                    return refuse(413)
                return await handler(request)

            return bounded_handler

    return DeclaredSizeRoute
