from __future__ import annotations

import json
import re
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.ai.tool_context import clear_tool_context, tool_execution_context
from app.ai.tool_result_read_tool import ReadToolResultInput, create_read_tool_result_tool
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


async def _raw(tool, **kwargs) -> str:
    with tool_execution_context(
        conversation_id=CONVERSATION_ID, user_id=USER_ID, agent_key="search"
    ):
        return await tool.ainvoke(kwargs)


async def _invoke(tool, **kwargs) -> dict:
    return json.loads(await _raw(tool, **kwargs))


def _texts(payload: dict) -> str:
    return " ".join(item["text"] for item in payload["excerpts"])


@pytest.mark.asyncio
async def test_returns_the_passages_matching_the_objective():
    payload_text = json.dumps(
        {
            "results": [
                {"content": "Arctic tern migration spans pole to pole."},
                {"content": "The migration deadline is 30 June 2026 for every tenant."},
            ]
        }
    )
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(repository=repository, service=FakeService(payload_text))

    payload = await _invoke(tool, blob_id=BLOB_ID, objective="What is the migration deadline?")

    assert "30 June 2026" in _texts(payload)
    assert payload["objective"] == "What is the migration deadline?"
    assert repository.calls == [(BLOB_ID, USER_ID, CONVERSATION_ID)]


@pytest.mark.asyncio
async def test_the_response_carries_no_offset_cursor():
    """The reader answers a question; it does not hand back a place to resume
    from. A cursor is what turned one large result into a paging loop."""
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}),
        service=FakeService("The migration deadline is 30 June 2026."),
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, objective="What is the migration deadline?")

    assert "next_offset" not in payload
    assert "offset" not in payload
    assert "returned_chars" not in payload


def test_the_input_schema_has_no_offset_or_limit():
    fields = set(ReadToolResultInput.model_fields)

    assert "offset" not in fields
    assert "limit" not in fields
    assert fields == {"blob_id", "objective", "max_excerpts", "max_chars"}


def test_an_objective_is_required_at_the_schema_boundary():
    with pytest.raises(ValidationError):
        ReadToolResultInput(blob_id=BLOB_ID)
    with pytest.raises(ValidationError):
        ReadToolResultInput(blob_id=BLOB_ID, objective="x")


def test_a_blank_objective_is_rejected_rather_than_ranking_against_nothing():
    """A length that counts spaces is not a stated objective.

    Ranking against no terms returns the no-match explanation for a payload
    that may well hold the answer, and it does so at full retrieval cost.
    """
    for blank in ("   ", "  	  ", " . "):
        with pytest.raises(ValidationError):
            ReadToolResultInput(blob_id=BLOB_ID, objective=blank)


def test_a_stated_objective_keeps_its_words_but_loses_stray_whitespace():
    parsed = ReadToolResultInput(blob_id=BLOB_ID, objective="  What is   the deadline? ")

    assert parsed.objective == "What is the deadline?"


@pytest.mark.asyncio
async def test_the_result_reports_what_it_left_behind():
    payload_text = json.dumps({f"k{index}": f"deadline detail {index}" for index in range(30)})
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}), service=FakeService(payload_text)
    )

    payload = await _invoke(
        tool, blob_id=BLOB_ID, objective="What is the deadline?", max_excerpts=2
    )

    assert len(payload["excerpts"]) == 2
    assert payload["total_candidates"] == 30
    assert payload["omitted_candidates"] == 28


@pytest.mark.asyncio
async def test_caller_limits_are_clamped_to_the_configured_maxima(monkeypatch):
    monkeypatch.setattr(
        "app.ai.tool_result_read_tool.settings.tool_result_focus_max_excerpts", 2, raising=False
    )
    monkeypatch.setattr(
        "app.ai.tool_result_read_tool.settings.tool_result_focus_max_chars", 900, raising=False
    )
    payload_text = json.dumps(
        {f"k{index}": f"deadline paragraph {index}: " + ("evidence " * 80) for index in range(20)}
    )
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}), service=FakeService(payload_text)
    )

    raw = await _raw(
        tool,
        blob_id=BLOB_ID,
        objective="What is the deadline?",
        max_excerpts=20,
        max_chars=80_000,
    )

    assert len(json.loads(raw)["excerpts"]) <= 2
    assert len(raw) <= 900, "the returned string is what reaches model context"


@pytest.mark.asyncio
async def test_an_objective_nothing_matches_returns_a_bounded_explanation():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record={"id": BLOB_ID}),
        service=FakeService(json.dumps({"content": "Arctic tern migration. " * 100})),
    )

    payload = await _invoke(
        tool, blob_id=BLOB_ID, objective="quarterly revenue for the Osaka subsidiary"
    )

    assert payload["excerpts"] == []
    assert payload["note"]
    assert "Arctic tern" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_blob_outside_the_conversation_is_not_found():
    tool = create_read_tool_result_tool(
        repository=FakeRepository(record=None), service=FakeService("secret")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, objective="find the secret")

    assert payload["status"] == "error"
    assert payload["error_type"] == "not_found"
    assert "secret" not in json.dumps(payload["hint"])


@pytest.mark.asyncio
async def test_malformed_blob_id_is_not_found_without_touching_the_repository():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(repository=repository, service=FakeService("x"))

    payload = await _invoke(tool, blob_id="not-a-uuid", objective="find the value")

    assert payload["error_type"] == "not_found"
    assert repository.calls == []


@pytest.mark.asyncio
async def test_missing_tool_context_is_not_found():
    repository = FakeRepository(record={"id": BLOB_ID})
    tool = create_read_tool_result_tool(repository=repository, service=FakeService("x"))

    payload = json.loads(
        await tool.ainvoke({"blob_id": BLOB_ID, "objective": "find the value"})
    )

    assert payload["error_type"] == "not_found"
    assert repository.calls == []


def test_tool_identity_is_internal():
    tool = create_read_tool_result_tool(repository=FakeRepository(), service=FakeService("x"))

    assert tool.name == "read_tool_result"
    assert tool.metadata["tool_origin"] == "internal"
    assert tool.metadata["qualified_tool_id"] == "internal::read_tool_result"


def test_the_description_asks_for_an_objective_and_forbids_a_repeat_call():
    tool = create_read_tool_result_tool(repository=FakeRepository(), service=FakeService("x"))
    description = tool.description.lower()

    assert "objective" in description
    assert "do not call repeatedly" in description
    assert "offset" not in description
    assert "full text" not in description


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

    payload = await _invoke(tool, blob_id=BLOB_ID, objective="find the value")

    assert payload["status"] == "error"
    assert payload["error_type"] == "not_found"
    assert payload["retryable"] is False


@pytest.mark.asyncio
async def test_repository_fault_returns_a_retryable_error():
    tool = create_read_tool_result_tool(
        repository=_RaisingRepository(), service=FakeService("secret")
    )

    payload = await _invoke(tool, blob_id=BLOB_ID, objective="find the secret")

    assert payload["status"] == "error"
    assert payload["error_type"] == "unavailable"
    assert payload["retryable"] is True
    assert "secret" not in json.dumps(payload["hint"])


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
    # and resolve back to the stored text through the production code paths, and
    # one call must reach evidence the preview dropped.
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
    payload = await _invoke(
        tool,
        blob_id=notice_ids[0],
        objective="Find the TAIL-8 value",
    )

    assert "TAIL-8" in _texts(payload)
    assert "next_offset" not in payload


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
    payload = await _invoke(
        tool, blob_id=offloaded["blob_id"], objective="Find the TAIL-8 value"
    )

    assert payload["error_type"] == "not_found"


def test_di_resolution_reuses_the_process_container():
    """A per-call ``Container()`` builds a second engine and connection pool.

    ``tool_result_blob_service`` is a container Singleton, so resolving twice
    off the shared container returns the same object. A fresh declarative
    container each call re-instantiates it — and, behind it, ``Database`` —
    which is the leak this guards.
    """
    from app.ai.tool_result_read_tool import _resolve
    from app.core.container import get_container

    first_repository, first_service = _resolve(None, None)
    second_repository, second_service = _resolve(None, None)

    assert first_service is not None
    assert first_service is second_service
    assert first_service is get_container().tool_result_blob_service()
    assert type(first_repository) is type(second_repository)
