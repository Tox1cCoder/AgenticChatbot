from uuid import uuid4

from app.services.tool_result_blob_service import ToolResultBlobService


class FakeRepository:
    def __init__(self):
        self.created = []

    def create(self, data):
        record = {"id": uuid4(), **data}
        self.created.append(record)
        return record


def test_offload_if_large_returns_inline_for_small_output(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=20)

    result = service.offload_if_large(
        conversation_id=uuid4(),
        user_id=uuid4(),
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="short result",
    )

    assert result["output"] == "short result"
    assert result["blob_id"] is None
    assert repo.created == []


def test_offload_if_large_writes_full_output_and_returns_preview(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=10)
    conversation_id = uuid4()
    user_id = uuid4()

    result = service.offload_if_large(
        conversation_id=conversation_id,
        user_id=user_id,
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="abcdefghijklmnopqrstuvwxyz",
    )

    assert (
        result["output"] == "abcdefghij\n\n[Output offloaded: use blob_id to read the full result.]"
    )
    assert result["blob_id"]
    assert result["size_bytes"] == 26
    assert repo.created[0]["conversation_id"] == conversation_id
    assert repo.created[0]["user_id"] == user_id
    assert repo.created[0]["tool_call_id"] == "call-1"
    assert (tmp_path / repo.created[0]["storage_path"]).read_text(
        encoding="utf-8"
    ) == "abcdefghijklmnopqrstuvwxyz"
