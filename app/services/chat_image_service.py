"""Content-addressed storage for chat image bytes (serve-by-reference)."""

from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
from uuid import UUID, uuid4

CHAT_IMAGE_URL_PREFIX = "/chat-images/"

_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


class ChatImageStorageService:
    """Persist chat image bytes on content-addressed disk, return references.

    Files live at ``storage_root/<sha[:2]>/<sha>.<ext>``; identical content is
    written once and shared. A DB row per reference records ownership for the
    per-user read endpoint.
    """

    def __init__(self, repository, *, storage_root: str | Path, max_bytes: int):
        self.repository = repository
        self.storage_root = Path(storage_root)
        self.max_bytes = max(1, int(max_bytes))

    def store(
        self,
        *,
        conversation_id: UUID,
        user_id: UUID,
        mime: str,
        data_b64: str,
        name: str,
    ) -> dict[str, str]:
        raw = self._decode(data_b64)
        if len(raw) > self.max_bytes:
            raise ValueError(f"chat image {len(raw)} bytes exceeds cap {self.max_bytes}")

        sha = hashlib.sha256(raw).hexdigest()
        content_type = mime if mime.startswith("image/") else "image/png"

        # Idempotent at the ownership-row level on (user_id, sha256): a resumed
        # run (or any retry) that re-persists identical content reuses the
        # existing row instead of inserting a duplicate. The bytes are already
        # content-addressed on disk, so this keeps the DB row model consistent
        # with the deduplicated files. Scoped to the owner so the per-user read
        # endpoint's ownership guarantees are preserved.
        #
        # NOTE: query-then-insert with a TOCTOU window and NO backing DB unique
        # constraint/index on (user_id, sha256). Correct for the sequential
        # resume/retry scenario; two concurrent identical persists (multi-worker)
        # could still double-insert. A partial unique index
        # `(user_id, sha256) WHERE deleted_at IS NULL` + conflict handling would
        # close that window if it ever matters.
        existing = self._existing_reference_for(user_id=user_id, sha=sha, name=name)
        if existing is not None:
            return existing

        rel_path = self._write_content_addressed(sha, content_type, raw)

        record = self.repository.create(
            {
                "id": uuid4(),
                "conversation_id": conversation_id,
                "user_id": user_id,
                "sha256": sha,
                "size_bytes": len(raw),
                "content_type": content_type,
                "storage_path": rel_path,
            }
        )
        image_id = record["id"] if isinstance(record, dict) else record.id
        return {
            "name": name or "image",
            "mime": content_type,
            "image_id": str(image_id),
            "url": f"{CHAT_IMAGE_URL_PREFIX}{image_id}",
            "content_hash": sha,
        }

    def _existing_reference_for(
        self, *, user_id: UUID, sha: str, name: str
    ) -> dict[str, str] | None:
        existing = self.repository.get_by_user_and_sha(user_id, sha)
        if existing is None:
            return None
        image_id = existing["id"] if isinstance(existing, dict) else existing.id
        content_type = (
            existing["content_type"] if isinstance(existing, dict) else existing.content_type
        )
        return {
            "name": name or "image",
            "mime": content_type,
            "image_id": str(image_id),
            "url": f"{CHAT_IMAGE_URL_PREFIX}{image_id}",
            "content_hash": sha,
        }

    def load_data_url(self, image_id: UUID, user_id: UUID) -> str | None:
        record = self.repository.get_for_user(image_id, user_id)
        if record is None:
            return None
        raw = self.read_bytes(record)
        b64 = base64.b64encode(raw).decode("ascii")
        content_type = record["content_type"] if isinstance(record, dict) else record.content_type
        return f"data:{content_type};base64,{b64}"

    def read_bytes(self, record) -> bytes:
        storage_path = record["storage_path"] if isinstance(record, dict) else record.storage_path
        return (self.storage_root / storage_path).read_bytes()

    def _decode(self, data_b64: str) -> bytes:
        payload = data_b64.split(",", 1)[1] if data_b64.startswith("data:") else data_b64
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as err:
            raise ValueError("invalid base64 image data") from err

    def _write_content_addressed(self, sha: str, content_type: str, raw: bytes) -> str:
        ext = _EXT_BY_MIME.get(content_type, "bin")
        rel = Path(sha[:2]) / f"{sha}.{ext}"
        dest = self.storage_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(raw)
        return str(rel).replace("\\", "/")
