"""Service for offloading large tool outputs to durable blob storage."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.services.tool_result_preview import ToolResultPreview, build_tool_result_preview


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
        answer_share: float = 0.25,
        min_result_content_chars: int = 200,
    ):
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.threshold_chars = max(1, int(threshold_chars))
        self.preview_chars = max(1, int(preview_chars or threshold_chars))
        self.answer_share = min(0.9, max(0.0, float(answer_share)))
        self.min_result_content_chars = max(1, int(min_result_content_chars))

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
        preview = build_tool_result_preview(
            output_text,
            budget_chars=self.preview_chars,
            answer_share=self.answer_share,
            min_result_content_chars=self.min_result_content_chars,
        )
        record_id = record["id"] if isinstance(record, dict) else record.id
        return {
            "output": f"{preview.text.rstrip()}\n\n{_notice(record_id, len(output_text), preview)}",
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


def _notice(blob_id: Any, total_chars: int, preview: ToolResultPreview) -> str:
    """Describe what the preview omitted and how to read the rest.

    Every loss is named, including the ones inside a kept result. A notice that
    listed two omitted arrays while silently stripping the page text out of each
    result told the model its omissions were trivial, which is worse than
    telling it nothing.
    """

    omissions = [f"{key} ({count} entries)" for key, count in preview.omitted_arrays]
    omissions.extend(preview.omitted_keys)
    if preview.omitted_results:
        omissions.append(f"{preview.omitted_results} further results")
    if preview.omitted_result_keys:
        omissions.append(f"{', '.join(preview.omitted_result_keys)} inside each result")
    detail = f" Omitted: {'; '.join(omissions)}." if omissions else ""
    shortened = (
        f" Shortened: {', '.join(preview.shortened_keys)}." if preview.shortened_keys else ""
    )
    return (
        f"[Output offloaded: {total_chars} chars stored as blob_id={blob_id}.{detail}"
        f"{shortened}"
        f' Call read_tool_result(blob_id="{blob_id}") to read the full text —'
        " do not re-run the tool.]"
    )
