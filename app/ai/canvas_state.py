"""Durable state helpers for the conversation-scoped canvas artifact."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

CANVAS_ARTIFACT_ID = "canvas:main"
CANVAS_EDIT_DENIED_TOOL_NAMES = frozenset({"widget_create", "widget_update"})
CanvasUpdateStatus = Literal["updated", "unchanged", "failed"]


@dataclass(frozen=True)
class CanvasArtifactSnapshot:
    artifact_id: str
    revision: int
    content: str
    language: str
    title: str
    message_id: str
    sequence: int
    is_latest_assistant: bool = False

    def descriptor(self) -> dict[str, Any]:
        """Return bounded routing context without executable artifact source."""
        return {
            "artifact_id": self.artifact_id,
            "revision": self.revision,
            "title": self.title,
            "message_id": self.message_id,
            "is_latest_assistant": self.is_latest_assistant,
        }

    def with_latest_assistant(self, value: bool) -> CanvasArtifactSnapshot:
        return replace(self, is_latest_assistant=value)

    def to_artifact(
        self,
        *,
        content: str | None = None,
        language: str | None = None,
        title: str | None = None,
        revision: int | None = None,
        operation: Literal["create", "update"] = "update",
    ) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "revision": revision if revision is not None else self.revision,
            "operation": operation,
            "content": self.content if content is None else content,
            "language": self.language if language is None else language,
            "title": self.title if title is None else title,
        }

    def update_status(
        self,
        status: CanvasUpdateStatus,
        *,
        revision: int | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": status,
            "artifact_id": self.artifact_id,
            "base_revision": self.revision,
            "revision": self.revision if revision is None else revision,
        }
        if reason:
            payload["reason"] = reason
        return payload


def canvas_snapshot_from_message(message: Any) -> CanvasArtifactSnapshot | None:
    """Validate and normalize a persisted assistant canvas message."""
    if getattr(message, "deleted_at", None) is not None:
        return None

    metadata = getattr(message, "message_metadata", None)
    if not isinstance(metadata, dict):
        return None
    artifact = metadata.get("canvas_artifact")
    if not isinstance(artifact, dict):
        return None

    content = artifact.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    artifact_id = artifact.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        artifact_id = CANVAS_ARTIFACT_ID

    raw_revision = artifact.get("revision", 1)
    revision = raw_revision if isinstance(raw_revision, int) and raw_revision > 0 else 1

    language = artifact.get("language")
    if not isinstance(language, str) or not language.strip():
        language = "html"

    title = artifact.get("title")
    if not isinstance(title, str) or not title.strip():
        title = "Canvas"

    message_id = getattr(message, "id", None)
    raw_sequence = getattr(message, "sequence", None)
    if message_id is None or not isinstance(raw_sequence, int):
        return None

    return CanvasArtifactSnapshot(
        artifact_id=artifact_id.strip(),
        revision=revision,
        content=content,
        language=language.strip(),
        title=title.strip(),
        message_id=str(message_id),
        sequence=raw_sequence,
    )


__all__ = [
    "CANVAS_ARTIFACT_ID",
    "CANVAS_EDIT_DENIED_TOOL_NAMES",
    "CanvasArtifactSnapshot",
    "canvas_snapshot_from_message",
]
