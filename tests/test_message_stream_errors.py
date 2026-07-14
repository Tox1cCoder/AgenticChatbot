from __future__ import annotations

import json

import pytest

from app.api.messages import _internal_event_stream_response
from app.core.exceptions import CustomHTTPException
from app.services.event_streaming.events import make_event
from app.services.event_streaming.internal_sse import legacy_event_from_v3


class _ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


async def _sse_payloads(response) -> list[dict]:
    payloads = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        for line in chunk.splitlines():
            if line.startswith("data: "):
                payloads.append(json.loads(line[6:]))
    return payloads


def test_internal_v3_error_retains_canonical_status_and_code():
    event = make_event(
        "error",
        sequence=1,
        data={
            "error": "This interrupt has already been resolved.",
            "status_code": 409,
            "error_code": "INTERRUPT_ALREADY_RESOLVED",
        },
    )

    assert legacy_event_from_v3(event) == {
        "type": "error",
        "error": "This interrupt has already been resolved.",
        "status_code": 409,
        "error_code": "INTERRUPT_ALREADY_RESOLVED",
    }


@pytest.mark.asyncio
async def test_internal_stream_exception_retains_custom_http_metadata():
    async def source():
        raise CustomHTTPException(409, "Already resolved", "INTERRUPT_ALREADY_RESOLVED")
        yield  # pragma: no cover

    response = _internal_event_stream_response(lambda: source(), _ConnectedRequest())
    payloads = await _sse_payloads(response)

    assert payloads == [
        {
            "type": "error",
            "error": "Already resolved",
            "status_code": 409,
            "error_code": "INTERRUPT_ALREADY_RESOLVED",
        }
    ]
