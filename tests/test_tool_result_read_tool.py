from __future__ import annotations

import json
from uuid import uuid4

import pytest

from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_result_read_tool import create_read_tool_result_tool

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
