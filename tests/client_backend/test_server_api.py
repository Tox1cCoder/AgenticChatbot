import pytest

from client_backend.services.server_api import ServerAPIClient


class _ServerAPIClientStub(ServerAPIClient):
    def __init__(self):
        super().__init__(base_url="http://example.test", timeout=5)
        self.stream_calls: list[tuple[str, dict]] = []

    async def stream_sse(self, path: str, json: dict | None = None, **kwargs):
        self.stream_calls.append((path, json or {}))
        yield {"type": "complete"}


@pytest.mark.asyncio
async def test_resume_ai_sdk_interrupt_uses_canonical_server_route():
    client = _ServerAPIClientStub()

    events = [event async for event in client.resume_ai_sdk_interrupt({"threadId": "thread-1"})]

    assert events == [{"type": "complete"}]
    assert client.stream_calls == [
        (
            "/ai/resume-interrupt",
            {"threadId": "thread-1"},
        )
    ]
