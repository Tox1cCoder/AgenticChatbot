from __future__ import annotations

import json
import re
from uuid import UUID, uuid4

import pytest

from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_result_read_tool import create_read_tool_result_tool
from app.services.tool_result_blob_service import ToolResultBlobService

CONVERSATION_ID = str(uuid4())
USER_ID = str(uuid4())
BLOB_ID = str(uuid4())


class FakeRepository:
    def __init__(self, record=None):
        self.record = record
        self.calls = []

    def get_for_user_and_conversation(self, blob_id, user_id, conversation_id):
        self.calls.append((str(blob_id), str(user_id), str(conversation_id)))
        return self.record


class FakeService:
    def __init__(self, text: str):
        self.text = text

    def read_text(self, record):
        return self.text


@pytest.fixture(autouse=True)
def _clean_context():
    clear_tool_context()
    yield
    clear_tool_context()


async def _invoke(tool, **kwargs) -> dict:
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id=USER_ID, agent_key="search"
    ):
        return json.loads(await tool.ainvoke(kwargs))


@pytest.mark.asyncio
async def test_returns_a_bounded_slice_and_next_offset():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(
        repository=repository, service=FakeService("0123456789")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, offset=0, limit=4)

    assert payload["content"] == "0123"
    assert payload["returned_chars"] == 4
    assert payload["total_chars"] == 10
    assert payload["next_offset"] == 4
    assert repository.calls == [(BLOB_ID, USER_ID, CONVERSATION_ID)]


@pytest.mark.asyncio
async def test_final_slice_reports_no_next_offset():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}), service=FakeService("0123456789")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, offset=8, limit=50)

    assert payload["content"] == "89"
    assert payload["next_offset"] is None


@pytest.mark.asyncio
async def test_limit_above_the_cap_is_clamped(monkeypatch):
    monkeypatch.setattr(
        "app.ai.tool_result_read_tool.settings.tool_result_read_max_chars", 5, raising=False
    )
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}), service=FakeService("a" * 100)
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, limit=10_000)

    assert payload["returned_chars"] == 5


@pytest.mark.asyncio
async def test_blob_outside_the_conversation_is_not_found():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record=None), service=FakeService("secret")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID)

    assert payload["status"] == "error"
    assert payload["error_type"] == "not_found"
    assert "secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_malformed_blob_id_is_not_found_without_touching_the_repository():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(repository=repository, service=FakeService("x"))

    payload = await _invoke(tool, blob_id="not-a-uuid")

    assert payload["error_type"] == "not_found"
    assert repository.calls == []


@pytest.mark.asyncio
async def test_missing_tool_context_is_not_found():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(repository=repository, service=FakeService("x"))

    payload = json.loads(await tool.ainvoke({"blob_id": BLOB_ID}))

    assert payload["error_type"] == "not_found"
    assert repository.calls == []


def test_tool_identity_is_internal():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(), service=FakeService("x")
    )

    assert tool.name == "read_tool_result"
    assert tool.metadata["tool_origin"] == "internal"
    assert tool.metadata["qualified_tool_id"] == "internal::read_tool_result"


class _RaisingService:
    def __init__(self, error: Exception):
        self.error = error

    def read_text(self, record):
        raise self.error


class _RaisingRepository:
    def get_for_user_and_conversation(self, blob_id, user_id, conversation_id):
        raise RuntimeError("connection pool exhausted")


@pytest.mark.parametrize(
    "error",
    [
        ValueError("blob-x has neither content nor storage_path"),
        OSError("legacy file is gone"),
    ],
)
@pytest.mark.asyncio
async def test_unreadable_record_returns_the_documented_not_found(error):
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}), service=_RaisingService(error)
    )

    payload = await _invoke(tool, blob_id=BLOB_ID)

    assert payload["status"] == "error"
    assert payload["error_type"] == "not_found"
    assert payload["retryable"] is False


@pytest.mark.asyncio
async def test_repository_fault_returns_a_retryable_error():
    tool = create_read_tool_result_tool(
        repository=_RaisingRepository(), service=FakeService("secret")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID)

    assert payload["status"] == "error"
    assert payload["error_type"] == "unavailable"
    assert payload["retryable"] is True
    assert "secret" not in json.dumps(payload)


class StoringRepository:
    """A repository that genuinely stores records, scoped like the real one."""

    def __init__(self):
        self.records: dict[str, dict] = {}

    def create(self, data):
        record = dict(data)
        self.records[str(record["id"])] = record
        return record

    def get_for_user_and_conversation(self, blob_id, user_id, conversation_id):
        record = self.records.get(str(blob_id))
        if record is None:
            return None
        if str(record["user_id"]) != str(user_id):
            return None
        if str(record["conversation_id"]) != str(conversation_id):
            return None
        return record


def _offloaded_payload() -> str:
    return json.dumps(
        {
            "results": [
                {
                    "index": index,
                    "title": f"Source {index}",
                    "url": f"https://e.example/{index}",
                    "content": f"MARKER-{index} " + "body sentence. " * 200 + f"TAIL-{index}",
                    "score": 0.9,
                }
                for index in range(1, 9)
            ],
            "total_results": 8,
            "answer": "Synthesized answer.",
            "provider": "tavily",
            "operation": "search",
            "query": "round trip",
        }
    )


@pytest.mark.asyncio
async def test_notice_blob_id_resolves_through_the_real_service_and_repository(tmp_path):
    # The seam the whole feature depends on: the id the model is shown must parse
    # and resolve back to the stored text through the production code paths.
    repository = StoringRepository()
    blob_service = ToolResultBlobService(
        repository, storage_root=tmp_path, threshold_chars=1000, preview_chars=4000
    )
    output_text = _offloaded_payload()

    offloaded = blob_service.offload_if_large(
        conversation_id=UUID(CONVERSATION_ID),
        user_id=UUID(USER_ID),
        tool_call_id="call-round-trip",
        tool_name="tavily_search",
        output_text=output_text,
    )
    notice_ids = re.findall(r"blob_id=([0-9a-fA-F-]{36})", offloaded["output"])
    assert notice_ids and notice_ids[0] == offloaded["blob_id"]
    assert "TAIL-8" not in offloaded["output"], "the tail must be missing from the preview"

    tool = create_read_tool_result_tool(repository=repository, service=blob_service)
    chunks: list[str] = []
    offset = 0
    for _ in range(20):
        payload = await _invoke(tool, blob_id=notice_ids[0], offset=offset)
        assert payload["total_chars"] == len(output_text)
        chunks.append(payload["content"])
        if payload["next_offset"] is None:
            break
        offset = payload["next_offset"]
    else:
        pytest.fail("read_tool_result never reported the end of the blob")

    assert "".join(chunks) == output_text
    assert "TAIL-8" in "".join(chunks)


@pytest.mark.asyncio
async def test_offloaded_blob_from_another_conversation_is_not_found(tmp_path):
    repository = StoringRepository()
    blob_service = ToolResultBlobService(
        repository, storage_root=tmp_path, threshold_chars=100, preview_chars=400
    )
    offloaded = blob_service.offload_if_large(
        conversation_id=uuid4(),
        user_id=UUID(USER_ID),
        tool_call_id="call-other",
        tool_name="tavily_search",
        output_text=_offloaded_payload(),
    )

    tool = create_read_tool_result_tool(repository=repository, service=blob_service)
    payload = await _invoke(tool, blob_id=offloaded["blob_id"])

    assert payload["error_type"] == "not_found"
