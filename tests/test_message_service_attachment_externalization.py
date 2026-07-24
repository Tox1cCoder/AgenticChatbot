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


def test_generated_images_reuse_early_stored_ref_without_restoring():
    """A generated image persisted early carries a ``stored_ref`` descriptor;
    terminal persistence reuses it and never calls the storage backend again."""
    called = []

    def store(**kwargs):
        called.append(kwargs)
        return {"image_id": "SHOULD-NOT-RUN", "url": "/chat-images/SHOULD-NOT-RUN"}

    svc = _make_service(store)
    metadata = {
        "images": [
            {
                "mime": "image/png",
                "data": base64.b64encode(b"final").decode(),
                "stored_ref": {
                    "image_id": "early-9",
                    "url": "/chat-images/early-9",
                    "mime": "image/png",
                },
            }
        ]
    }
    svc._externalize_generated_images(metadata, uuid4(), uuid4())

    assert called == []  # early descriptor reused — no re-decode/re-write
    img = metadata["images"][0]
    assert img["image_id"] == "early-9"
    assert img["url"] == "/chat-images/early-9"
    assert "data" not in img and "stored_ref" not in img
