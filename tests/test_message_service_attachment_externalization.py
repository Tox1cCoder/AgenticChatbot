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
    assert refs == [
        {"name": "a.png", "mime": "image/png", "image_id": "img-1", "url": "/chat-images/img-1"}
    ]
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


def test_externalize_falls_back_to_inline_on_store_failure():
    def boom(**_):
        raise RuntimeError("storage down")

    svc = _make_service(boom)
    original = {"name": "a.png", "mime": "image/png", "data": "QUJD"}
    data = SimpleNamespace(conversation_id=uuid4(), attachments=[original])
    refs = svc._externalize_attachments_for_persist(data, uuid4())
    # storage failure must not lose the image — keep the inline attachment
    assert refs == [original]
