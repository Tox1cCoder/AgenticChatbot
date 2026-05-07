"""Service for offloading large tool outputs to durable blob storage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


class ToolResultBlobService:
    """Persist full tool outputs out-of-band when they exceed a size threshold."""

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

        blob_id = uuid4()
        relative_path = Path(str(conversation_id)) / f"{blob_id}.txt"
        absolute_path = self.storage_root / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_text(output_text, encoding="utf-8")
        encoded = output_text.encode("utf-8")
        record = self.repository.create(
            {
                "id": blob_id,
                "conversation_id": conversation_id,
                "user_id": user_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "storage_path": str(relative_path),
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
        storage_path = (
            record["storage_path"] if isinstance(record, dict) else record.storage_path
        )
        absolute_path = self.storage_root / storage_path
        return absolute_path.read_text(encoding="utf-8")
