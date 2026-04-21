import pytest

from client_backend.services.server_api import ServerAPIClient, ServerAPIError


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


@pytest.mark.asyncio
async def test_stream_sse_reads_streaming_error_body_before_raising(monkeypatch):
    class FakeResponse:
        status_code = 422

        def __init__(self):
            self._read = False
            self._body = b'{"detail":"invalid payload"}'

        async def aread(self):
            self._read = True
            return self._body

        @property
        def content(self):
            if not self._read:
                raise RuntimeError("Attempted to access streaming response content without read()")
            return self._body

        @property
        def text(self):
            if not self._read:
                raise RuntimeError("Attempted to access streaming response text without read()")
            return self._body.decode("utf-8")

        def json(self):
            if not self._read:
                raise RuntimeError("Attempted to access streaming response json without read()")
            return {"detail": "invalid payload"}

        async def aiter_lines(self):
            if False:
                yield ""

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *args):
            return False

    client = ServerAPIClient(base_url="http://example.test", timeout=5)
    client._tokens = type("T", (), {"access_token": "fake"})()

    real_client = await client._get_client()
    monkeypatch.setattr(real_client, "stream", lambda *args, **kwargs: FakeStream())

    with pytest.raises(ServerAPIError) as exc_info:
        [event async for event in client.stream_sse("/messages/stream")]

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == {"detail": "invalid payload"}
