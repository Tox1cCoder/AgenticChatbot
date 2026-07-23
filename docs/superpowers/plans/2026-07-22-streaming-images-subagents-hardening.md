# Streaming (Subagents + Images) Production Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden subagent and image streaming for production: move chat image bytes out of the DB JSON column into content-addressed storage served by reference (P1), bound the subagent event-sink queue against slow-client backpressure (P2), collapse the double canonical↔legacy stream-translation seam (P3), and fix silent-drop / schema-drift papercuts (P4).

**Architecture:** Four independent phases, each independently shippable and testable. P1 mirrors the existing `ToolResultBlob` serve-by-reference precedent (Postgres row + per-user `GET` endpoint) but stores bytes on content-addressed disk (S3-swappable behind one interface). Image bytes are externalized at message-persistence time; the model still receives inline bytes for the *current* turn, and a `ContextVar`-installed loader (mirroring the existing `image_preview_emitter` pattern) re-hydrates historical images from storage on demand. P2/P3/P4 are localized changes to the event-streaming layer.

**Tech Stack:** Python 3, FastAPI, SQLAlchemy + Alembic (Postgres/JSONB), pydantic-settings, LangGraph/LangChain, pytest (+ `pytest.mark.asyncio`), `dependency-injector` container.

## Global Constraints

- Python runtime: the app runs on `.venv` and must stay v3-stream capable (see project memory `runtime-interpreter-divergence`). Do not touch langchain/langgraph pins.
- Graph state must stay **msgpack-serializable** for LangGraph HITL checkpointing — never place services, sinks, or callables in graph state; use the token-in-registry / `ContextVar` patterns already established.
- DB is Postgres; UUID PKs use `sqlalchemy.dialects.postgresql.UUID(as_uuid=True)`; metadata column is `message_metadata` (JSONB) on `app/models/message.py:37`.
- Settings are `pydantic-settings` fields on `Settings` in `app/core/config.py` with `Field(default=..., description=...)`; env var = upper-cased field name; positive-int fields must be added to the `_positive_int` validator list.
- Ruff baseline is ~161 pre-existing warnings (project memory `test-and-lint-baseline`); do not add new warnings. Functions ≤100 lines, ≤5 positional params.
- One logical change per commit, imperative subject ≤72 chars.
- Backward compatibility: existing messages store inline base64 in `metadata["attachments"][*].data` and `metadata["images"][*].data`/`b64_data`. Read paths MUST keep working for old inline rows; only new writes externalize. No forced data migration.

---

# Phase P1 — Externalize chat image bytes (serve-by-reference)

**Reference shape (the new contract).** A persisted attachment/image reference is:
```python
{"name": str, "mime": str, "image_id": str, "url": f"/chat-images/{image_id}"}
```
No `data`/`b64_data` key on new writes. `url` is what the frontend renders (browser is authenticated); `image_id` is what the model-rehydration loader resolves to bytes.

## Task P1.1: `ChatImage` model + Alembic migration

**Files:**
- Create: `app/models/chat_image.py`
- Modify: `app/models/__init__.py` (register model import if the package eagerly imports models — check first)
- Create: `app/alembic/versions/<newrev>_add_chat_images_table.py`
- Test: `tests/test_chat_image_model.py`

**Interfaces:**
- Produces: `ChatImage` ORM (`__tablename__ = "chat_images"`) with columns `id: UUID pk`, `conversation_id: UUID fk→conversations.id (index)`, `user_id: UUID fk→users.id (index)`, `sha256: str(64)`, `size_bytes: int`, `content_type: str(128)`, `storage_path: str(1024)`, `created_at: datetime`, `deleted_at: datetime|None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_image_model.py
from app.models.chat_image import ChatImage


def test_chat_image_table_and_columns():
    assert ChatImage.__tablename__ == "chat_images"
    cols = ChatImage.__table__.columns
    for name in (
        "id", "conversation_id", "user_id", "sha256",
        "size_bytes", "content_type", "storage_path", "created_at", "deleted_at",
    ):
        assert name in cols, f"missing column {name}"
    assert cols["storage_path"].nullable is False
    assert cols["deleted_at"].nullable is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_chat_image_model.py -v`
Expected: FAIL — `ModuleNotFoundError: app.models.chat_image`.

- [ ] **Step 3: Write the model** (mirror `app/models/tool_result_blob.py`)

```python
# app/models/chat_image.py
"""Model for chat image bytes offloaded out of message metadata."""

import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.models.base import Base


class ChatImage(Base):
    """A single stored chat image (user attachment or generated image).

    Bytes live on content-addressed disk under ``storage_path``; the message
    metadata only carries an ``image_id`` reference. Rows are per-reference so
    multiple messages may point at the same content-addressed file.
    """

    __tablename__ = "chat_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    sha256 = Column(String(64), nullable=False, index=True)
    size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(128), nullable=False, default="image/png")
    storage_path = Column(String(1024), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    conversation = relationship("Conversation", backref="chat_images")
    user = relationship("User", backref="chat_images")

    def __repr__(self) -> str:
        return f"<ChatImage(id={self.id}, sha256='{self.sha256[:12]}', size={self.size_bytes})>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_chat_image_model.py -v`
Expected: PASS.

- [ ] **Step 5: Create the Alembic migration**

First find the current head: `.venv/Scripts/python -m alembic -c alembic.ini heads`. Use that value for `down_revision`. Template mirrors `app/alembic/versions/q0r1s2t3u4v5_add_tool_result_blobs.py`:

```python
"""add_chat_images

Revision ID: <newrev>
Revises: <current_head>
Create Date: 2026-07-22 00:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "<newrev>"
down_revision: str | Sequence[str] | None = "<current_head>"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_images",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", UUID(as_uuid=True),
                  sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=128), nullable=False,
                  server_default="image/png"),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_chat_images_conversation_id", "chat_images", ["conversation_id"])
    op.create_index("ix_chat_images_user_id", "chat_images", ["user_id"])
    op.create_index("ix_chat_images_sha256", "chat_images", ["sha256"])


def downgrade() -> None:
    op.drop_index("ix_chat_images_sha256", table_name="chat_images")
    op.drop_index("ix_chat_images_user_id", table_name="chat_images")
    op.drop_index("ix_chat_images_conversation_id", table_name="chat_images")
    op.drop_table("chat_images")
```

- [ ] **Step 6: Verify migration applies on the full chain**

Run: `.venv/Scripts/python -m pytest tests/test_alembic_full_chain_postgres.py -v`
Expected: PASS (this test walks the whole revision chain against Postgres). If Postgres is unavailable locally, at minimum run `.venv/Scripts/python -m alembic -c alembic.ini upgrade head` against a scratch DB and confirm no error.

- [ ] **Step 7: Commit**

```bash
git add app/models/chat_image.py app/alembic/versions/*_add_chat_images_table.py tests/test_chat_image_model.py
git commit -m "feat: add chat_images table and model"
```

## Task P1.2: `ChatImageRepository`

**Files:**
- Create: `app/repositories/chat_image.py`
- Test: `tests/test_chat_image_repository.py`

**Interfaces:**
- Consumes: `ChatImage` (P1.1), a `session_factory` context manager.
- Produces:
  - `ChatImageRepository(session_factory)`
  - `.create(data: dict) -> ChatImage`
  - `.get_for_user(image_id: UUID, user_id: UUID) -> ChatImage | None` (filters `deleted_at IS NULL`)
  - `.find_active_by_sha_for_user(sha256: str, user_id: UUID) -> ChatImage | None` (dedup lookup)

- [ ] **Step 1: Write the failing test** (fake-session style, mirrors `tests/test_document_parse_artifact_repository.py`)

```python
# tests/test_chat_image_repository.py
from contextlib import contextmanager
from uuid import uuid4

from app.repositories.chat_image import ChatImageRepository


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, rows=None):
        self.added = []
        self.committed = 0
        self.refreshed = []
        self._rows = rows or []

    def add(self, obj):
        obj.id = obj.id or uuid4()
        self.added.append(obj)

    def commit(self):
        self.committed += 1

    def refresh(self, obj):
        self.refreshed.append(obj)

    def execute(self, _stmt):
        return _FakeQuery(self._rows)


def _factory(session):
    @contextmanager
    def _f():
        yield session
    return _f


def test_create_persists_and_commits():
    session = _FakeSession()
    repo = ChatImageRepository(_factory(session))
    row = repo.create(
        {
            "id": uuid4(),
            "conversation_id": uuid4(),
            "user_id": uuid4(),
            "sha256": "a" * 64,
            "size_bytes": 10,
            "content_type": "image/png",
            "storage_path": "ab/abc.png",
        }
    )
    assert session.committed == 1
    assert row in session.added


def test_get_for_user_returns_row():
    marker = object()
    repo = ChatImageRepository(_factory(_FakeSession(rows=[marker])))
    assert repo.get_for_user(uuid4(), uuid4()) is marker
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_chat_image_repository.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the repository** (mirror `app/repositories/tool_result_blob.py`)

```python
# app/repositories/chat_image.py
"""Repository for stored chat image records."""

from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.models.chat_image import ChatImage


class ChatImageRepository:
    """Persistence for ChatImage rows."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, data: dict[str, Any]) -> ChatImage:
        with self.session_factory() as db:
            record = ChatImage(**data)
            db.add(record)
            db.commit()
            db.refresh(record)
            return record

    def get_for_user(self, image_id: UUID, user_id: UUID) -> ChatImage | None:
        with self.session_factory() as db:
            statement = select(ChatImage).where(
                ChatImage.id == image_id,
                ChatImage.user_id == user_id,
                ChatImage.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()

    def find_active_by_sha_for_user(self, sha256: str, user_id: UUID) -> ChatImage | None:
        with self.session_factory() as db:
            statement = select(ChatImage).where(
                ChatImage.sha256 == sha256,
                ChatImage.user_id == user_id,
                ChatImage.deleted_at.is_(None),
            )
            return db.execute(statement).scalars().first()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_chat_image_repository.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/repositories/chat_image.py tests/test_chat_image_repository.py
git commit -m "feat: add ChatImageRepository"
```

## Task P1.3: config settings for chat-image storage

**Files:**
- Modify: `app/core/config.py` (add fields near `document_images_storage_path` ~`:867`; register the int field in the `_positive_int` validator list ~`:1496-1642`)
- Test: `tests/test_config_chat_image_settings.py`

**Interfaces:**
- Produces on `Settings`: `chat_images_storage_path: str` (default `"app/storage/chat_images"`), `chat_image_max_bytes: int` (default `10 * 1024 * 1024`), `chat_image_history_rehydrate_limit: int` (default `4` — max historical images re-sent to the model per request; `0` = unlimited).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_chat_image_settings.py
from app.core.config import get_settings


def test_chat_image_settings_defaults():
    s = get_settings()
    assert s.chat_images_storage_path == "app/storage/chat_images"
    assert s.chat_image_max_bytes == 10 * 1024 * 1024
    assert s.chat_image_history_rehydrate_limit == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config_chat_image_settings.py -v`
Expected: FAIL — `AttributeError: chat_images_storage_path`.

- [ ] **Step 3: Add the fields** (near line 867, after `document_images_storage_path`)

```python
    chat_images_storage_path: str = Field(
        default="app/storage/chat_images",
        description="Content-addressed storage root for externalized chat image bytes.",
    )
    chat_image_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum decoded byte size accepted when externalizing a chat image.",
    )
    chat_image_history_rehydrate_limit: int = Field(
        default=4,
        description=(
            "Max historical stored images re-sent to the model per request "
            "(most recent first). 0 disables the cap."
        ),
    )
```

Add `chat_image_max_bytes` to the existing `_positive_int` `@field_validator` field list, and `chat_image_history_rehydrate_limit` to the `_non_negative_int` list (0 is valid).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_config_chat_image_settings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/config.py tests/test_config_chat_image_settings.py
git commit -m "feat: add chat image storage settings"
```

## Task P1.4: `ChatImageStorageService` (write + read, content-addressed disk)

**Files:**
- Create: `app/services/chat_image_service.py`
- Test: `tests/test_chat_image_service.py`

**Interfaces:**
- Consumes: `ChatImageRepository` (P1.2), `storage_root: str|Path`, `max_bytes: int`.
- Produces:
  - `ChatImageStorageService(repository, *, storage_root, max_bytes)`
  - `.store(*, conversation_id: UUID, user_id: UUID, mime: str, data_b64: str, name: str) -> dict` returning a **reference dict** `{"name","mime","image_id","url"}`; raises `ValueError` on invalid base64 or oversize.
  - `.load_data_url(image_id: UUID, user_id: UUID) -> str | None` → `data:{mime};base64,{b64}` or `None` if missing.
  - `.read_bytes(record: ChatImage) -> bytes` (used by the HTTP endpoint).
  - module constant `CHAT_IMAGE_URL_PREFIX = "/chat-images/"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_image_service.py
import base64
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.chat_image_service import ChatImageStorageService


class _Repo:
    def __init__(self):
        self.rows = {}
        self.created = []

    def create(self, data):
        row = SimpleNamespace(**data)
        self.rows[data["id"]] = row
        self.created.append(data)
        return row

    def get_for_user(self, image_id, user_id):
        row = self.rows.get(image_id)
        if row and row.user_id == user_id:
            return row
        return None

    def find_active_by_sha_for_user(self, sha256, user_id):
        for row in self.rows.values():
            if row.sha256 == sha256 and row.user_id == user_id:
                return row
        return None


def _svc(tmp_path):
    return ChatImageStorageService(
        _Repo(), storage_root=str(tmp_path / "chat_images"), max_bytes=1024
    )


def test_store_writes_file_and_returns_reference(tmp_path):
    svc = _svc(tmp_path)
    raw = b"\x89PNG\r\n\x1a\n" + b"x" * 32
    b64 = base64.b64encode(raw).decode()
    conv, user = uuid4(), uuid4()
    ref = svc.store(conversation_id=conv, user_id=user, mime="image/png", data_b64=b64, name="a.png")

    assert set(ref) == {"name", "mime", "image_id", "url"}
    assert ref["mime"] == "image/png"
    assert ref["url"] == f"/chat-images/{ref['image_id']}"
    assert "data" not in ref and "base64" not in ref
    # bytes are on disk, not in the reference
    assert svc.load_data_url(uuid4().__class__(ref["image_id"]), user).startswith("data:image/png;base64,")


def test_store_rejects_oversize(tmp_path):
    svc = _svc(tmp_path)
    b64 = base64.b64encode(b"y" * 2048).decode()
    with pytest.raises(ValueError):
        svc.store(conversation_id=uuid4(), user_id=uuid4(), mime="image/png", data_b64=b64, name="big.png")


def test_store_rejects_invalid_base64(tmp_path):
    svc = _svc(tmp_path)
    with pytest.raises(ValueError):
        svc.store(conversation_id=uuid4(), user_id=uuid4(), mime="image/png", data_b64="!!!not-b64!!!", name="x")


def test_load_data_url_missing_returns_none(tmp_path):
    svc = _svc(tmp_path)
    assert svc.load_data_url(uuid4(), uuid4()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_chat_image_service.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the service**

```python
# app/services/chat_image_service.py
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
```

Note the `test_store_writes_file_and_returns_reference` helper reconstructs a UUID from `ref["image_id"]`; when implementing, if that expression reads awkwardly, replace it in the test with `from uuid import UUID` and `UUID(ref["image_id"])`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_chat_image_service.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add app/services/chat_image_service.py tests/test_chat_image_service.py
git commit -m "feat: add ChatImageStorageService"
```

## Task P1.5: DI wiring (container providers)

**Files:**
- Modify: `app/core/container.py` (near `document_image_repository` ~`:261` and the blob-service `Singleton` ~`:301-307`)
- Test: `tests/test_container_chat_image_wiring.py`

**Interfaces:**
- Consumes: `ChatImageRepository`, `ChatImageStorageService`, `settings`.
- Produces on `Container`: `.chat_image_repository()` (Factory), `.chat_image_service()` (Singleton).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_container_chat_image_wiring.py
from app.core.container import Container
from app.repositories.chat_image import ChatImageRepository
from app.services.chat_image_service import ChatImageStorageService


def test_container_provides_chat_image_components():
    c = Container()
    assert isinstance(c.chat_image_repository(), ChatImageRepository)
    assert isinstance(c.chat_image_service(), ChatImageStorageService)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_container_chat_image_wiring.py -v`
Expected: FAIL — `AttributeError: chat_image_repository`.

- [ ] **Step 3: Register providers** (follow the existing `document_image_repository` + blob-service patterns)

```python
    chat_image_repository = providers.Factory(
        ChatImageRepository,
        session_factory=db.provided.session,
    )

    chat_image_service = providers.Singleton(
        ChatImageStorageService,
        repository=chat_image_repository,
        storage_root=providers.Object(settings.chat_images_storage_path),
        max_bytes=providers.Object(settings.chat_image_max_bytes),
    )
```

Add the imports at the top of `container.py` next to the existing repository/service imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_container_chat_image_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/core/container.py tests/test_container_chat_image_wiring.py
git commit -m "feat: wire chat image repository and service into container"
```

## Task P1.6: `GET /chat-images/{image_id}` read endpoint

**Files:**
- Create: `app/api/chat_images.py`
- Modify: `app/main.py` (import router near `:30`, `include_router` near `:254-276`)
- Test: `tests/test_chat_images_api.py`

**Interfaces:**
- Consumes: `ChatImageRepository.get_for_user`, `ChatImageStorageService.read_bytes`, `get_current_user_id` (from `app.core.auth`).
- Produces: `GET /chat-images/{image_id}` → 200 `Response(content=bytes, media_type=content_type)`; 404 when the row is absent or not owned by the caller.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chat_images_api.py
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_images import router, _get_repository, _get_service
from app.core.auth import get_current_user_id


@pytest.fixture
def client_and_state():
    app = FastAPI()
    app.include_router(router)

    user_id = uuid4()
    image_id = uuid4()
    record = SimpleNamespace(id=image_id, user_id=user_id, content_type="image/png",
                             storage_path="ab/x.png")

    class _Repo:
        def get_for_user(self, iid, uid):
            return record if (iid == image_id and uid == user_id) else None

    class _Svc:
        def read_bytes(self, rec):
            return b"PNGBYTES"

    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[_get_repository] = lambda: _Repo()
    app.dependency_overrides[_get_service] = lambda: _Svc()
    return TestClient(app), image_id


def test_get_image_returns_bytes(client_and_state):
    client, image_id = client_and_state
    resp = client.get(f"/chat-images/{image_id}")
    assert resp.status_code == 200
    assert resp.content == b"PNGBYTES"
    assert resp.headers["content-type"].startswith("image/png")


def test_get_unknown_image_404(client_and_state):
    client, _ = client_and_state
    resp = client.get(f"/chat-images/{uuid4()}")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_chat_images_api.py -v`
Expected: FAIL — module `app.api.chat_images` not found.

- [ ] **Step 3: Write the endpoint** (mirror `app/api/tool_result_blobs.py`)

```python
# app/api/chat_images.py
"""Per-user read endpoint for externalized chat images."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.auth import get_current_user_id
from app.core.container import Container
from app.repositories.chat_image import ChatImageRepository
from app.services.chat_image_service import ChatImageStorageService

router = APIRouter(prefix="/chat-images", tags=["chat-images"])


def _get_repository() -> ChatImageRepository:
    return Container().chat_image_repository()


def _get_service() -> ChatImageStorageService:
    return Container().chat_image_service()


@router.get("/{image_id}")
async def read_chat_image(
    image_id: UUID,
    current_user_id: UUID = Depends(get_current_user_id),
    repository: ChatImageRepository = Depends(_get_repository),
    service: ChatImageStorageService = Depends(_get_service),
) -> Response:
    record = repository.get_for_user(image_id, current_user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    return Response(content=service.read_bytes(record), media_type=record.content_type)
```

- [ ] **Step 4: Register the router in `app/main.py`**

Add `from app.api.chat_images import router as chat_images_router` near the other router imports (~`:30`) and `app.include_router(chat_images_router)` in the include block (~`:254-276`).

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_chat_images_api.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/api/chat_images.py app/main.py tests/test_chat_images_api.py
git commit -m "feat: add GET /chat-images/{id} read endpoint"
```

## Task P1.7: externalize user attachments at persistence

**Files:**
- Modify: `app/services/message_service.py` (`create_message` ~`:848-859`; the service must hold `chat_image_service` — check its `__init__` and add the dependency via the container where `MessageService` is constructed)
- Test: `tests/test_message_service_attachment_externalization.py`

**Interfaces:**
- Consumes: `ChatImageStorageService.store` (P1.4).
- Produces: after `create_message`, the persisted user row's `message_metadata["attachments"]` contains reference dicts (`image_id`+`url`, no `data`); the workflow request passed to the graph for the *current* turn still carries the original inline base64 attachments.

**Design note (why here):** `create_message` (message_service.py:848) has both `user_id` and `conversation_id`, persists the user row at :859, and builds the workflow request separately at :877 from `message_create_data.attachments`. Externalize a **copy** for persistence only; leave `message_create_data.attachments` untouched so the current-turn model call still gets inline bytes (no storage round-trip on the hot path).

- [ ] **Step 1: Write the failing test** (unit around a small helper)

Add a private helper `_externalize_attachments_for_persist(self, message_create_data, user_id) -> list[dict] | None` and test it directly:

```python
# tests/test_message_service_attachment_externalization.py
import base64
from types import SimpleNamespace
from uuid import uuid4

from app.services.message_service import MessageService


def _make_service(store_fn):
    svc = MessageService.__new__(MessageService)  # bypass heavy __init__
    svc.chat_image_service = SimpleNamespace(store=store_fn)
    return svc


def test_externalize_replaces_base64_with_reference():
    calls = []

    def store(*, conversation_id, user_id, mime, data_b64, name):
        calls.append((mime, name))
        return {"name": name, "mime": mime, "image_id": "img-1", "url": "/chat-images/img-1"}

    svc = _make_service(store)
    b64 = base64.b64encode(b"PNG").decode()
    data = SimpleNamespace(
        conversation_id=uuid4(),
        attachments=[{"name": "a.png", "mime": "image/png", "data": b64}],
    )
    refs = svc._externalize_attachments_for_persist(data, uuid4())
    assert refs == [{"name": "a.png", "mime": "image/png", "image_id": "img-1", "url": "/chat-images/img-1"}]
    assert "data" not in refs[0]
    assert calls == [("image/png", "a.png")]


def test_externalize_passthrough_when_no_inline_data():
    svc = _make_service(lambda **_: {"unexpected": True})
    data = SimpleNamespace(
        conversation_id=uuid4(),
        attachments=[{"name": "u", "mime": "image/png", "url": "https://x/y.png"}],
    )
    refs = svc._externalize_attachments_for_persist(data, uuid4())
    # already a reference / remote URL — left as-is, store not called
    assert refs == [{"name": "u", "mime": "image/png", "url": "https://x/y.png"}]


def test_externalize_none_when_no_attachments():
    svc = _make_service(lambda **_: None)
    data = SimpleNamespace(conversation_id=uuid4(), attachments=None)
    assert svc._externalize_attachments_for_persist(data, uuid4()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_message_service_attachment_externalization.py -v`
Expected: FAIL — `AttributeError: _externalize_attachments_for_persist`.

- [ ] **Step 3: Implement the helper and call it in `create_message`**

```python
    def _externalize_attachments_for_persist(
        self, message_create_data, user_id
    ) -> list[dict] | None:
        attachments = getattr(message_create_data, "attachments", None)
        if not attachments:
            return None
        refs: list[dict] = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            inline_b64 = att.get("data") or att.get("base64")
            if not inline_b64:
                refs.append(att)  # already a reference or remote URL
                continue
            ref = self.chat_image_service.store(
                conversation_id=message_create_data.conversation_id,
                user_id=user_id,
                mime=att.get("mime") or att.get("mimeType") or "image/png",
                data_b64=inline_b64,
                name=att.get("name") or "image",
            )
            refs.append(ref)
        return refs
```

Then in `create_message`, when `message_create_data.role == MessageRole.user`, build a persistence-only copy so the DB row stores references while the workflow request keeps inline bytes:

```python
        message_entity = MessageFactory.create_from_schema_with_role(
            message_create_data, message_create_data.role
        )
        if message_create_data.role == MessageRole.user:
            refs = self._externalize_attachments_for_persist(message_create_data, user_id)
            if refs is not None and isinstance(message_entity.message_metadata, dict):
                message_entity.message_metadata["attachments"] = refs
        created_message = self.repository.create(message_entity)
```

Wrap the externalization in a `try/except` that logs and falls back to the original inline attachments on failure (storage must never block sending a message). Add `chat_image_service` to `MessageService.__init__` and pass `container.chat_image_service()` at construction.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_message_service_attachment_externalization.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Run the message-service suite for regressions**

Run: `.venv/Scripts/python -m pytest tests/ -k message_service -q`
Expected: PASS (no regressions from the `__init__` signature change).

- [ ] **Step 6: Commit**

```bash
git add app/services/message_service.py tests/test_message_service_attachment_externalization.py
git commit -m "feat: externalize user image attachments at persistence"
```

## Task P1.8: rehydrate historical images by reference (model side)

**Files:**
- Modify: `app/ai/image_context.py` (`build_multimodal_content` gains an optional loader; add a `ContextVar` + install/read helpers mirroring `app/ai/image_generation/emitter.py`)
- Modify: `app/ai/agents/base_agent.py:1161-1174` (no signature change — reads the installed loader)
- Modify: `app/ai/graph.py` (install the loader for the run, bound to the request `user_id`, mirroring `_build_image_preview_emitter` ~`:1527`/`:1577`)
- Test: `tests/test_image_context_reference_loader.py`

**Interfaces:**
- Consumes: `ChatImageStorageService.load_data_url` (P1.4).
- Produces:
  - `image_context.CHAT_IMAGE_LOADER: ContextVar[Callable[[str], str | None] | None]`
  - `image_context.use_chat_image_loader(loader)` (contextmanager) and `current_chat_image_loader()`.
  - `build_multimodal_content(text, attachments)` unchanged signature but now, per attachment, if it has `image_id` and no inline `data`, resolves via the installed loader → data URL; falls back to `normalize_image_attachment` for inline/remote (backward compat with old rows).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_image_context_reference_loader.py
from app.ai import image_context
from app.ai.image_context import build_multimodal_content, has_image_parts, use_chat_image_loader


def test_reference_attachment_resolved_via_loader():
    calls = []

    def loader(image_id):
        calls.append(image_id)
        return "data:image/png;base64,QUJD"

    att = {"name": "a.png", "mime": "image/png", "image_id": "img-9", "url": "/chat-images/img-9"}
    with use_chat_image_loader(loader):
        parts = build_multimodal_content("hi", [att])

    assert calls == ["img-9"]
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_reference_dropped_when_no_loader_installed():
    att = {"name": "a.png", "mime": "image/png", "image_id": "img-9", "url": "/chat-images/img-9"}
    parts = build_multimodal_content("hi", [att])
    # internal /chat-images url is NOT model-fetchable; without a loader it must not leak
    assert not has_image_parts(parts)


def test_inline_data_still_works_without_loader():
    att = {"name": "a.png", "mime": "image/png", "data": "QUJD"}
    parts = build_multimodal_content("", [att])
    assert has_image_parts(parts)
    assert parts[0]["image_url"]["url"] == "data:image/png;base64,QUJD"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_image_context_reference_loader.py -v`
Expected: FAIL — `use_chat_image_loader` not defined.

- [ ] **Step 3: Implement the ContextVar loader + reference resolution**

Add to `app/ai/image_context.py`:

```python
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

CHAT_IMAGE_LOADER: ContextVar[Callable[[str], str | None] | None] = ContextVar(
    "chat_image_loader", default=None
)


def current_chat_image_loader() -> Callable[[str], str | None] | None:
    return CHAT_IMAGE_LOADER.get()


@contextmanager
def use_chat_image_loader(loader: Callable[[str], str | None] | None) -> Iterator[None]:
    token = CHAT_IMAGE_LOADER.set(loader)
    try:
        yield
    finally:
        CHAT_IMAGE_LOADER.reset(token)


def _resolve_reference_url(attachment: dict) -> str | None:
    image_id = attachment.get("image_id")
    if not image_id or attachment.get("data") or attachment.get("base64"):
        return None
    loader = current_chat_image_loader()
    if loader is None:
        return None
    return loader(str(image_id))
```

Then update the `build_multimodal_content` loop so a reference is tried first:

```python
    for attachment in attachments or []:
        if isinstance(attachment, dict):
            ref_url = _resolve_reference_url(attachment)
            if ref_url:
                parts.append(image_url_part(ref_url))
                continue
            if attachment.get("image_id") and not (attachment.get("data") or attachment.get("base64")):
                # reference we could not resolve (no loader / missing) — drop, do not leak internal url
                continue
        normalized = normalize_image_attachment(attachment)
        if normalized is None:
            continue
        parts.append(image_url_part(normalized["url"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_image_context_reference_loader.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Install the loader for the run in `graph.py`**

Where the request executes with `user_id` in scope (mirror `_build_image_preview_emitter` install at `graph.py:1577`), wrap agent invocation in `use_chat_image_loader(loader)` where:

```python
    def _loader(image_id: str, *, _svc=self._chat_image_service, _uid=user_id):
        try:
            return _svc.load_data_url(UUID(image_id), _uid)
        except Exception:
            return None
```

Apply `chat_image_history_rehydrate_limit` here: build the loader only for the most-recent N referenced images (count references across history; beyond the cap, return `None` so older images are dropped from the model context). Log how many were dropped (`no silent caps` rule).

- [ ] **Step 6: Run the agent/history suites for regressions**

Run: `.venv/Scripts/python -m pytest tests/test_message_history_image_context.py tests/ -k "history or multimodal" -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/ai/image_context.py app/ai/agents/base_agent.py app/ai/graph.py tests/test_image_context_reference_loader.py
git commit -m "feat: rehydrate historical images by reference with a per-run loader"
```

## Task P1.9: frontend renders references; generated-image externalization

**Files:**
- Modify: `demo.py` (`_normalize_image_for_gallery` ~`:4891`, `render_agent_images` ~`:5498`, attachment thumbnails ~`:8972-9010`) to prefer `url` when present and only fall back to base64 `data` for legacy rows.
- Modify: `app/core/response_constants.py` (`build_bot_metadata` ~`:374`) to externalize generated `metadata["images"][*]` base64 into references, reusing `ChatImageStorageService`.
- Test: `tests/test_bot_metadata_image_externalization.py`; manual demo.py check.

**Interfaces:**
- Consumes: `ChatImageStorageService.store`.
- Produces: persisted `metadata["images"]` entries carry `url`+`image_id` (no `data`/`b64_data`) on new writes; demo renders via `url`.

- [ ] **Step 1: Write the failing test** for the generated-image externalization helper (extract a pure helper `externalize_metadata_images(images, *, store) -> list[dict]` in `response_constants.py`)

```python
# tests/test_bot_metadata_image_externalization.py
import base64

from app.core.response_constants import externalize_metadata_images


def test_generated_images_externalized():
    def store(*, mime, data_b64, name, **_):
        return {"name": name, "mime": mime, "image_id": "g1", "url": "/chat-images/g1"}

    imgs = [{"name": "gen.png", "mime_type": "image/png", "data": base64.b64encode(b"X").decode()}]
    out = externalize_metadata_images(imgs, store=store)
    assert out[0]["url"] == "/chat-images/g1"
    assert "data" not in out[0] and "b64_data" not in out[0]


def test_remote_images_left_untouched():
    out = externalize_metadata_images([{"url": "https://x/y.png", "mime_type": "image/png"}], store=None)
    assert out[0]["url"] == "https://x/y.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_bot_metadata_image_externalization.py -v`
Expected: FAIL — `externalize_metadata_images` not defined.

- [ ] **Step 3: Implement `externalize_metadata_images`** in `response_constants.py` (drop `data`/`b64_data`, add `url`+`image_id`; leave entries that already have `url` and no inline bytes untouched; on `store` failure keep the inline entry). Call it from `build_bot_metadata` where `metadata["images"]` is assigned (~`:403-404`), passing the container's `chat_image_service.store` bound with `conversation_id`/`user_id`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_bot_metadata_image_externalization.py -v`
Expected: PASS.

- [ ] **Step 5: Update demo.py rendering** — in `_normalize_image_for_gallery` and the attachment-thumbnail builder, use `entry["url"]` when present (prefix the API base URL for the auth'd `GET`), else decode legacy `data`. Verify manually: upload an image, send, reload the conversation — the thumbnail must render from the `/chat-images/{id}` endpoint, and the DB row's `message_metadata` must contain no base64.

- [ ] **Step 6: Commit**

```bash
git add demo.py app/core/response_constants.py tests/test_bot_metadata_image_externalization.py
git commit -m "feat: externalize generated images and render chat images by reference"
```

## Task P1.10: verification pass

- [ ] Run the focused suites: `.venv/Scripts/python -m pytest tests/ -k "chat_image or image_context or message_service or bot_metadata" -q` — all pass.
- [ ] Run ruff on touched files: `.venv/Scripts/python -m ruff check app/models/chat_image.py app/repositories/chat_image.py app/services/chat_image_service.py app/api/chat_images.py app/ai/image_context.py` — no NEW warnings.
- [ ] Manual end-to-end (demo.py): upload → send → confirm image renders live; reload conversation → confirm image renders from `/chat-images/{id}`; inspect the DB row and confirm `message_metadata` has references, not base64; send a follow-up turn and confirm the model still "sees" the prior image (rehydration loader works) up to the rehydrate limit.
- [ ] Update project memory: note the new `chat_images` storage-by-reference pipeline and the `chat_image_history_rehydrate_limit` policy.

---

# Phase P2 — Bound the subagent event-sink queue (backpressure)

**Problem:** `SubagentEventSink._queue = asyncio.Queue()` is unbounded ([subagents.py:27](app/services/event_streaming/subagents.py#L27)); `emit_event` uses `put_nowait` "because the queue is unbounded so put_nowait never fails" ([subagents.py:61-67](app/services/event_streaming/subagents.py#L61)). Image previews push up to ~3 MB base64 per event onto it. A slow SSE client + a fast multi-image run grows memory without bound.

**Approach:** Bound the queue with a **type-aware overflow policy**: lifecycle events (`subagent_start`/`subagent_end`/`subagent_tool_execution_end`) are lossless (must never be dropped — the frontend upsert keys off them); transient `image_preview` `partial` frames and `subagent_message_delta` frames are **coalesce/drop-oldest** (they are already replace-in-place by `item_id`+`seq`, so dropping a stale partial is correct). Emit a single `log`/counter when frames are dropped (`no silent caps`).

## Task P2.1: bounded queue with overflow policy

**Files:**
- Modify: `app/services/event_streaming/subagents.py` (`SubagentEventSink.__init__`, `emit_event`, add `_maxsize` + drop bookkeeping)
- Modify: `app/core/config.py` (add `subagent_event_queue_maxsize: int = Field(default=512, ...)`, register in `_positive_int`)
- Test: `tests/test_subagent_event_sink_backpressure.py`

**Interfaces:**
- Consumes: config `subagent_event_queue_maxsize`.
- Produces: `SubagentEventSink(maxsize: int = 0)`; when full, a transient (`image_preview` partial / `subagent_message_delta`) event evicts the oldest transient in the queue; a lossless event is always enqueued (growing past the soft cap if forced) and increments `dropped_transient_count`.

- [ ] **Step 1: Write failing tests** (drive out the policy):

```python
# tests/test_subagent_event_sink_backpressure.py
import pytest

from app.services.event_streaming.events import make_event
from app.services.event_streaming.subagents import SubagentEventSink


def _preview(seq):
    return make_event("image_preview", data={"status": "partial", "seq": seq})


@pytest.mark.asyncio
async def test_transient_frames_bounded_by_maxsize():
    sink = SubagentEventSink(maxsize=3)
    for i in range(10):
        sink.emit_event(_preview(i))
    assert sink._queue.qsize() <= 3
    assert sink.dropped_transient_count >= 7


@pytest.mark.asyncio
async def test_lifecycle_events_never_dropped():
    sink = SubagentEventSink(maxsize=2)
    for i in range(5):
        sink.emit_event(_preview(i))
    await sink.emit(  # lifecycle 'end' must survive
        "subagent_end", task_id="t1", agent_name="worker", status="completed"
    )
    drained = await sink.drain()
    assert any(e.type == "subagent_end" for e in drained)
```

- [ ] **Step 2: Run to verify they fail.**
Run: `.venv/Scripts/python -m pytest tests/test_subagent_event_sink_backpressure.py -v`
Expected: FAIL — `SubagentEventSink()` takes no `maxsize`; `dropped_transient_count` missing.

- [ ] **Step 3: Implement** the bounded queue + eviction. Add `_TRANSIENT_TYPES = {"image_preview", "subagent_message_delta"}`; in `emit_event`, if `qsize() >= maxsize > 0` and the incoming event is transient, evict the oldest transient (drain-and-requeue keeping lossless ones) or drop the incoming and bump `dropped_transient_count`; lossless events always enqueue. Keep `put_nowait` for lossless. Keep `emit` (async lifecycle) lossless.

- [ ] **Step 4: Run to verify pass.** Then run the existing streaming suite for regressions: `.venv/Scripts/python -m pytest tests/ -k "subagent and (event or stream)" -q`.

- [ ] **Step 5: Wire `maxsize`** from `settings.subagent_event_queue_maxsize` at every `SubagentEventSink()` construction site (grep for `SubagentEventSink(`). Commit.

```bash
git add app/services/event_streaming/subagents.py app/core/config.py tests/test_subagent_event_sink_backpressure.py
git commit -m "feat: bound subagent event-sink queue with transient-drop policy"
```

---

# Phase P3 — Collapse the double canonical↔legacy translation seam

**Problem:** The graph converts canonical `V3StreamEvent` → legacy public dicts (`app/services/event_streaming/graph_public_projection.py`), then `AIService` converts legacy dicts → canonical again (`app/services/ai_service.py:282-452` + `app/services/event_streaming/compat.py`). Both modules' docstrings mark themselves deletable. The round-trip is the source of the `_to_ai_request` drift risk and unknown-event drops.

**Approach:** Make `graph.execute_request_stream` / `resume_with_decisions_stream` yield canonical `V3StreamEvent` directly to the service, and have the service consume canonical events without re-coercion. Delete `graph_public_projection.py` and `compat.py` once no caller remains.

> **Detail pass required before executing P3.** This phase needs the full current source of `graph_public_projection.py`, `compat.py`, and `ai_service._map_workflow_stream` read in full to enumerate every legacy dict `type` and its canonical equivalent. Produce that mapping table first, then write per-event characterization tests (capture current wire output for a representative run) so the refactor is provably behavior-preserving.

## Task P3.1: characterization tests (safety net) — *detail at execution*
- [ ] Capture the current internal-SSE and AI-SDK-v6 wire output for a fixture run exercising: token, thinking, tool_start/end, subagent start/tool/end, image_preview, interrupt, complete, error. Snapshot them (`tests/test_stream_wire_characterization.py`).

## Task P3.2: emit canonical from the graph — *detail at execution*
- [ ] Replace `GraphPublicStreamProjector.map_event` calls in `graph.py` with direct `V3StreamEvent` yields; adjust `ai_service` to consume canonical events (drop the `compat.coerce_legacy_event_to_v3` fallback). Keep both SSE adapters (`internal_sse.py`, `ai_sdk_v6.py`) unchanged — they already consume canonical.

## Task P3.3: delete the legacy seam — *detail at execution*
- [ ] Delete `graph_public_projection.py` and `compat.py`; remove imports; run the full characterization snapshot suite to prove identical wire output.

---

# Phase P4 — Papercuts

## Task P4.1: user-facing error for dropped `blob:` / local-path images
**Files:** `app/ai/image_context.py:68-84`; ingest paths in `app/api/ai_sdk.py` / `demo.py`.
- [ ] Test: an attachment that normalizes to `None` (blob/local path) surfaces a structured warning to the caller instead of vanishing. Return a `(normalized, reason)` from a new `normalize_image_attachment_result` (keep `normalize_image_attachment` as a thin wrapper for callers that only want the value), collect `reason`s at ingest, and emit one user-visible notice ("1 image couldn't be attached: browser blob URLs aren't supported"). Commit.

## Task P4.2: schema drift contract-test
**Files:** `tests/test_workflow_request_schema_parity.py`.
- [ ] Test asserts the field sets of `app/schemas/workflow.py:WorkflowExecutionRequest` and `app/ai/schemas.py:AIWorkflowExecutionRequest` are identical (`set(A.model_fields) == set(B.model_fields)`), so `_to_ai_request` ([ai_service.py:145](app/services/ai_service.py#L145)) can never silently drop a newly-added field. This is a pure guard test; no production change. Commit.

## Task P4.3: resume-path subagent progress — decision, then optional wiring
**Files:** `app/ai/graph.py` `resume_with_decisions_stream` (~`:2416`).
- [ ] Decision gate: confirm with the maintainer whether live subagent progress on the resume path is wanted (currently intentionally `event_sink=None` per `docs/.../event_streaming.md:228`). If yes: mirror the `execute_request_stream` sink wiring (`graph.py:2601,2682`) into the resume path and add a test asserting a `subagent_start` reaches the stream on resume. If no: add a one-line code comment linking the design-doc rationale so it stops reading as an omission. Commit.

---

## Self-Review Notes

- **Spec coverage:** P1 covers ingest (P1.7) → store (P1.4) → reference contract → serve (P1.6) → model rehydration (P1.8) → frontend + generated images (P1.9). P2 covers the unbounded-queue risk. P3 covers the double-translation seam. P4 covers blob/local-path silent drop, `_to_ai_request` drift, and resume-path progress.
- **Backward compatibility:** every read path (P1.8, P1.9) keeps a legacy inline-`data` fallback; no forced data migration; old messages keep rendering.
- **Placeholder honesty:** P1 and P2 are execution-ready (full test + implementation code). **P3 is intentionally task-level** — it is a behavior-preserving refactor whose correct implementation depends on reading two files in full and building a characterization snapshot first; a detail pass (P3.1's mapping table) must precede coding. P4 tasks are small and specified to the file/line.
- **Type consistency:** the reference dict `{"name","mime","image_id","url"}` is used identically in P1.4, P1.7, P1.8, P1.9. `ChatImageStorageService.store(...)` / `.load_data_url(...)` / `.read_bytes(...)` signatures are consistent across P1.4/P1.6/P1.7/P1.8.
