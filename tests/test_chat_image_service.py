import base64
import hashlib
from types import SimpleNamespace
from uuid import UUID, uuid4

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


def _svc(tmp_path, max_bytes=1024):
    return ChatImageStorageService(
        _Repo(), storage_root=str(tmp_path / "chat_images"), max_bytes=max_bytes
    )


def test_store_writes_file_and_returns_reference(tmp_path):
    svc = _svc(tmp_path)
    raw = b"\x89PNG\r\n\x1a\n" + b"x" * 32
    b64 = base64.b64encode(raw).decode()
    conv, user = uuid4(), uuid4()
    ref = svc.store(
        conversation_id=conv, user_id=user, mime="image/png", data_b64=b64, name="a.png"
    )

    assert set(ref) == {"name", "mime", "image_id", "url", "content_hash"}
    assert ref["mime"] == "image/png"
    assert ref["url"] == f"/chat-images/{ref['image_id']}"
    assert ref["content_hash"] == hashlib.sha256(raw).hexdigest()
    assert "data" not in ref and "base64" not in ref
    data_url = svc.load_data_url(UUID(ref["image_id"]), user)
    assert data_url.startswith("data:image/png;base64,")


def test_store_rejects_oversize(tmp_path):
    svc = _svc(tmp_path)
    b64 = base64.b64encode(b"y" * 2048).decode()
    with pytest.raises(ValueError):
        svc.store(
            conversation_id=uuid4(), user_id=uuid4(), mime="image/png", data_b64=b64, name="big.png"
        )


def test_store_rejects_invalid_base64(tmp_path):
    svc = _svc(tmp_path)
    with pytest.raises(ValueError):
        svc.store(
            conversation_id=uuid4(),
            user_id=uuid4(),
            mime="image/png",
            data_b64="!!!not-b64!!!",
            name="x",
        )


def test_load_data_url_missing_returns_none(tmp_path):
    svc = _svc(tmp_path)
    assert svc.load_data_url(uuid4(), uuid4()) is None
