from uuid import uuid4

import pytest

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


def test_offload_if_large_stores_content_in_db_and_creates_no_files(tmp_path):
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
    record = repo.created[0]
    assert record["conversation_id"] == conversation_id
    assert record["user_id"] == user_id
    assert record["tool_call_id"] == "call-1"
    assert record["content"] == "abcdefghijklmnopqrstuvwxyz"
    assert record["storage_path"] is None
    assert list(tmp_path.iterdir()) == [], "offload must not create files on disk"


def test_read_text_prefers_db_content(tmp_path):
    service = ToolResultBlobService(FakeRepository(), storage_root=tmp_path, threshold_chars=10)
    record = {"content": "full output", "storage_path": None}
    assert service.read_text(record) == "full output"


def test_read_text_falls_back_to_legacy_file(tmp_path):
    service = ToolResultBlobService(FakeRepository(), storage_root=tmp_path, threshold_chars=10)
    legacy_dir = tmp_path / "conv-1"
    legacy_dir.mkdir()
    (legacy_dir / "blob-1.txt").write_text("legacy payload", encoding="utf-8")
    record = {"id": "blob-1", "content": None, "storage_path": "conv-1/blob-1.txt"}
    assert service.read_text(record) == "legacy payload"


def test_read_text_raises_on_corrupt_record(tmp_path):
    service = ToolResultBlobService(FakeRepository(), storage_root=tmp_path, threshold_chars=10)
    record = {"id": "blob-x", "content": None, "storage_path": None}
    with pytest.raises(ValueError, match="blob-x"):
        service.read_text(record)
