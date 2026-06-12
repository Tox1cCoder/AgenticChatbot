"""Service for offloading large tool outputs to durable blob storage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class ToolResultBlobService:
    """Persist full tool outputs out-of-band when they exceed a size threshold.

    New blobs store their content in the ``tool_result_blobs`` Postgres table.
    ``storage_root`` is retained only to read legacy records whose payload
    still lives on disk under ``storage_path``.
    """

    def __init__(
        self,
        repository,
        *,
        storage_root: str | Path,
        threshold_chars: int,
        preview_chars: int | None = None,
    ):
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.threshold_chars = max(1, int(threshold_chars))
        self.preview_chars = max(1, int(preview_chars or threshold_chars))

    def offload_if_large(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        tool_call_id: str | None,
        tool_name: str,
        output_text: str,
    ) -> dict[str, object]:
        if len(output_text) <= self.threshold_chars:
            return {
                "output": output_text,
                "blob_id": None,
                "size_bytes": len(output_text.encode("utf-8")),
            }

        encoded = output_text.encode("utf-8")
        record = self.repository.create(
            {
                "id": uuid4(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "content": output_text,
                "storage_path": None,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
                "content_type": "text/plain",
            }
        )
        preview = output_text[: self.preview_chars].rstrip()
        record_id = record["id"] if isinstance(record, dict) else record.id
        return {
            "output": f"{preview}\n\n[Output offloaded: use blob_id to read the full result.]",
            "blob_id": str(record_id),
            "size_bytes": len(encoded),
        }

    def read_text(self, record: Any) -> str:
        content = record["content"] if isinstance(record, dict) else record.content
        if content is not None:
            return content
        storage_path = record["storage_path"] if isinstance(record, dict) else record.storage_path
        if not storage_path:
            record_id = record["id"] if isinstance(record, dict) else record.id
            raise ValueError(
                f"Tool result blob {record_id} has neither content nor storage_path; "
                "the record is corrupt and cannot be read."
            )
        return (self.storage_root / storage_path).read_text(encoding="utf-8")
