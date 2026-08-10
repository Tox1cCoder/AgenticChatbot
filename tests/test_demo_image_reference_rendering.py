"""Frontend rendering resolves stored chat-image references to displayable src."""

import contextlib
from types import SimpleNamespace

import demo


def _patch_st(monkeypatch, auth_token="tok"):
    monkeypatch.setattr(demo, "st", SimpleNamespace(session_state={"auth_token": auth_token}))


def test_absolute_url_passthrough(monkeypatch):
    _patch_st(monkeypatch)
    out = demo._normalize_image_for_gallery({"url": "https://x/y.png"}, "img")
    assert out == {"src": "https://x/y.png", "name": "img"}


def test_relative_reference_resolved_to_data_uri(monkeypatch):
    _patch_st(monkeypatch)
    seen = {}

    def fake_fetch(url, token):
        seen["url"] = url
        seen["token"] = token
        return "data:image/png;base64,QUJD"

    monkeypatch.setattr(demo, "_fetch_protected_image_data_uri", fake_fetch)
    out = demo._normalize_image_for_gallery(
        {"url": "/chat-images/abc", "name": "shot"}, "img"
    )
    assert out == {"src": "data:image/png;base64,QUJD", "name": "shot"}
    assert seen == {"url": "/chat-images/abc", "token": "tok"}


def test_relative_reference_falls_back_to_inline_data(monkeypatch):
    _patch_st(monkeypatch)
    monkeypatch.setattr(demo, "_fetch_protected_image_data_uri", lambda *_: None)
    out = demo._normalize_image_for_gallery(
        {"url": "/chat-images/abc", "data": "QUJD", "mime": "image/png"}, "img"
    )
    assert out == {"src": "data:image/png;base64,QUJD", "name": "img"}


def test_protected_fetch_rejects_non_image_content_type(monkeypatch):
    class Response:
        headers = {"Content-Type": "text/html; charset=utf-8"}
        content = b"<html>not an image</html>"

        def raise_for_status(self):
            return None

    class Session:
        def get(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(demo, "get_http_session", lambda: Session())

    result = demo._fetch_protected_image_data_uri.__wrapped__(
        "/web-images/11111111-1111-4111-8111-111111111111",
        "token",
    )

    assert result is None


def test_b64_data_field_supported(monkeypatch):
    _patch_st(monkeypatch)
    out = demo._normalize_image_for_gallery(
        {"b64_data": "QUJD", "mime_type": "image/webp"}, "img"
    )
    assert out == {"src": "data:image/webp;base64,QUJD", "name": "img"}


def test_inline_rich_image_resolves_protected_reference_via_shared_helper(monkeypatch):
    """The inline rich-item renderer (finalize + history) must resolve a
    protected ``/chat-images/{id}`` reference through the same authenticated
    fetch helper the gallery uses, because a browser cannot attach the app
    Bearer token to a bare ``<img src>``. The emitted HTML must carry the
    authenticated ``data:`` URI, never the raw relative reference."""
    captured: dict[str, str] = {}
    st_stub = SimpleNamespace(
        session_state={"auth_token": "tok"},
        markdown=lambda html, **_kwargs: captured.__setitem__("html", html),
    )
    monkeypatch.setattr(demo, "st", st_stub)
    seen: dict[str, str] = {}

    def fake_fetch(url, token):
        seen["url"] = url
        seen["token"] = token
        return "data:image/png;base64,QUJD"

    monkeypatch.setattr(demo, "_fetch_protected_image_data_uri", fake_fetch)

    demo._render_inline_rich_item(
        {"type": "image", "payload": {"url": "/chat-images/abc"}},
        message_metadata={},
        message_key="m1",
        auto_mount=False,
    )

    assert seen == {"url": "/chat-images/abc", "token": "tok"}
    assert "data:image/png;base64,QUJD" in captured["html"]
    assert "/chat-images/abc" not in captured["html"], (
        "a bare protected relative reference must never reach the <img src>; "
        "the browser cannot authenticate it"
    )


def test_inline_web_image_uses_one_source_footer_not_title_or_alt(monkeypatch):
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state={"auth_token": "tok"},
            markdown=lambda html, **_kwargs: captured.__setitem__("html", html),
        ),
    )
    monkeypatch.setattr(
        demo,
        "_fetch_protected_image_data_uri",
        lambda *_: "data:image/jpeg;base64,QUJD",
    )

    demo._render_inline_rich_item(
        {
            "type": "image",
            "title": "Provider title that is not a caption",
            "alt_text": "Accessibility description",
            "payload": {
                "url": "/web-images/abc",
                "mime_type": "image/jpeg",
                "source_url": "https://publisher.example/story",
                "width": 640,
                "height": 360,
            },
        },
        message_metadata={},
        message_key="m1",
        auto_mount=False,
    )

    assert 'alt="Accessibility description"' in captured["html"]
    assert "Provider title that is not a caption" not in captured["html"]
    assert "publisher.example" in captured["html"]


def test_failed_protected_fetch_renders_no_fallback(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state={"auth_token": "tok"},
            markdown=lambda html, **_kwargs: rendered.append(html),
        ),
    )
    monkeypatch.setattr(demo, "_fetch_protected_image_data_uri", lambda *_: None)

    demo._render_inline_rich_item(
        {
            "type": "image",
            "alt_text": "Example",
            "payload": {
                "url": "/web-images/missing",
                "mime_type": "image/jpeg",
                "source_url": "https://publisher.example/story",
            },
        },
        message_metadata={},
        message_key="m1",
        auto_mount=False,
    )

    assert rendered == []


def test_inline_group_preserves_failed_protected_cell_in_place(monkeypatch):
    rendered: list[str] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state={"auth_token": "tok"},
            markdown=lambda html, **_kwargs: rendered.append(html),
        ),
    )

    def fetch(url, _token):
        if url.endswith("/1"):
            return "data:image/jpeg;base64,QUJD"
        return None

    monkeypatch.setattr(demo, "_fetch_protected_image_data_uri", fetch)
    demo._render_inline_rich_item(
        {
            "type": "image_group",
            "alt_text": "Two selected images",
            "payload": {
                "items": [
                    {"url": "/web-images/1", "mime_type": "image/jpeg"},
                    {"url": "/web-images/2", "mime_type": "image/jpeg"},
                ]
            },
        },
        message_metadata={},
        message_key="m1",
        auto_mount=False,
    )

    html = rendered[0]
    assert html.count('data-role="cell"') == 2
    assert html.count("<img") == 1
    assert 'data-state="failed"' in html
    assert "Visual unavailable" in html


def test_inline_group_loads_and_renders_every_provider_selected_cell(monkeypatch):
    rendered: list[str] = []
    fetched: list[str] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state={"auth_token": "tok"},
            markdown=lambda html, **_kwargs: rendered.append(html),
        ),
    )

    def fetch(url, _token):
        fetched.append(url)
        return f"data:image/jpeg;base64,CELL{url.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(demo, "_fetch_protected_image_data_uri", fetch)
    demo._render_inline_rich_item(
        {
            "type": "image_group",
            "alt_text": "Six selected images",
            "payload": {
                "items": [
                    {"url": f"/web-images/{index}", "mime_type": "image/jpeg"}
                    for index in range(1, 7)
                ]
            },
        },
        message_metadata={},
        message_key="m1",
        auto_mount=False,
    )

    html = rendered[0]
    assert fetched == [f"/web-images/{index}" for index in range(1, 7)]
    assert html.count("<img") == 6
    assert [html.index(f"CELL{index}") for index in range(1, 7)] == sorted(
        html.index(f"CELL{index}") for index in range(1, 7)
    )


class _PlaceholderStub:
    def __init__(self):
        self.emptied = 0

    def container(self):
        return contextlib.nullcontext()

    def empty(self):
        self.emptied += 1


def _preview_event(*, image_index, status, kind, url, seq):
    delivery = (
        {"kind": "reference", "image_id": f"id-{image_index}", "url": url}
        if kind == "reference"
        else {"kind": "inline", "data_url": url}
    )
    return {
        "type": "image_preview",
        "schema_version": 2,
        "item_id": f"image-preview-{image_index}",
        "image_index": image_index,
        "status": status,
        "seq": seq,
        "media_type": "image/png",
        "delivery": delivery,
    }


def test_streaming_preview_panel_replaces_partial_with_final_by_index(monkeypatch):
    """A schema-v2 partial preview is replaced in place by the FINAL delivery
    for the same image index (not appended alongside it)."""
    images: list[bytes] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state={"auth_token": "tok"},
            image=lambda raw, **_kwargs: images.append(raw),
        ),
    )
    monkeypatch.setattr(
        demo, "_fetch_protected_image_data_uri", lambda *_: "data:image/png;base64,QUJD"
    )

    panel = demo._StreamingImagePreviewPanel(_PlaceholderStub())
    panel.apply(
        _preview_event(
            image_index=0,
            status="partial",
            kind="inline",
            url="data:image/png;base64,cGFydA==",
            seq=1,
        )
    )
    panel.apply(
        _preview_event(
            image_index=0,
            status="final",
            kind="reference",
            url="/chat-images/id-0",
            seq=2,
        )
    )

    assert set(panel._by_index) == {0}
    assert panel._by_index[0]["status"] == "final"
    assert panel._by_index[0]["kind"] == "reference"


def test_streaming_preview_panel_keeps_final_reference_across_complete(monkeypatch):
    """At ``complete`` the panel keeps a FINAL-by-reference image visible and
    drops only transient partials — it must not blank the image and hope a
    later render re-fetches it."""
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state={"auth_token": "tok"},
            image=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(
        demo, "_fetch_protected_image_data_uri", lambda *_: "data:image/png;base64,QUJD"
    )

    placeholder = _PlaceholderStub()
    panel = demo._StreamingImagePreviewPanel(placeholder)
    panel.apply(
        _preview_event(
            image_index=0, status="final", kind="reference", url="/chat-images/id-0", seq=2
        )
    )
    panel.apply(
        _preview_event(
            image_index=1,
            status="partial",
            kind="inline",
            url="data:image/png;base64,cGFydA==",
            seq=1,
        )
    )

    panel.finalize()

    assert set(panel._by_index) == {0}, "final reference kept, transient partial dropped"
    assert placeholder.emptied == 0, "a kept final must not blank the placeholder"
