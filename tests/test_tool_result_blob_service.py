import json
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


def test_offload_stores_content_and_notice_carries_the_blob_id(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(repo, storage_root=tmp_path, threshold_chars=10)

    result = service.offload_if_large(
        conversation_id=uuid4(),
        user_id=uuid4(),
        tool_call_id="call-1",
        tool_name="search_documents",
        output_text="abcdefghijklmnopqrstuvwxyz",
    )

    assert result["output"].startswith("abcdefghij")
    assert f"blob_id={result['blob_id']}" in result["output"]
    assert "26 chars" in result["output"]
    assert "read_tool_result" in result["output"]
    assert result["size_bytes"] == 26
    record = repo.created[0]
    assert record["content"] == "abcdefghijklmnopqrstuvwxyz"
    assert record["storage_path"] is None
    assert list(tmp_path.iterdir()) == [], "offload must not create files on disk"


def test_offload_notice_names_the_omitted_array_and_dropped_results(tmp_path):
    repo = FakeRepository()
    service = ToolResultBlobService(
        repo, storage_root=tmp_path, threshold_chars=100, preview_chars=600
    )
    payload = json.dumps(
        {
            "results": [
                {"title": f"T{i}", "url": f"https://e.example/{i}", "content": "c" * 900}
                for i in range(6)
            ],
            "total_results": 6,
            "answer": "",
            "provider": "tavily",
            "images": [{"url": "https://cdn.example/a.jpg"} for _ in range(24)],
        }
    )

    result = service.offload_if_large(
        conversation_id=uuid4(),
        user_id=uuid4(),
        tool_call_id="call-2",
        tool_name="tavily_search",
        output_text=payload,
    )

    assert "images (24 entries)" in result["output"]
    assert "further results" in result["output"]
    assert "https://cdn.example/a.jpg" not in result["output"]


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
