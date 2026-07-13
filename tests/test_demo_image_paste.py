from pathlib import Path

from tests.test_demo_plan_widget import _import_demo_with_ui_stubs


class Upload:
    def __init__(self, name: str, data: bytes, mime: str = "image/png"):
        self.name = name
        self.type = mime
        self._data = data

    def read(self):
        return self._data

    def seek(self, _pos):
        return None


def test_handle_new_image_attachments_has_no_four_image_limit(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []

    uploads = [Upload(f"{idx}.png", f"img-{idx}".encode("ascii")) for idx in range(6)]

    demo._handle_new_image_attachments(uploads)

    assert len(streamlit_stub.session_state.pending_image_attachments) == 6


def test_handle_new_image_attachments_preserves_same_image_selected_twice(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []

    uploads = [
        Upload("first-copy.png", b"same-image"),
        Upload("second-copy.png", b"same-image"),
    ]

    demo._handle_new_image_attachments(uploads)

    pending = streamlit_stub.session_state.pending_image_attachments
    assert [item["name"] for item in pending] == ["first-copy.png", "second-copy.png"]


def test_consume_chat_image_uploader_adds_all_files_and_advances_dropzone(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []
    streamlit_stub.session_state.chat_image_uploader_nonce = 4
    uploader_key = "chat_image_uploader_conv-1_4"
    streamlit_stub.session_state[uploader_key] = [
        Upload("first.png", b"first-image"),
        Upload("second.png", b"second-image"),
    ]

    demo._consume_chat_image_uploader(uploader_key)

    pending = streamlit_stub.session_state.pending_image_attachments
    assert [item["name"] for item in pending] == ["first.png", "second.png"]
    assert streamlit_stub.session_state.chat_image_uploader_nonce == 5


def test_handle_pasted_image_payload_reuses_pending_attachment_queue(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []

    demo._handle_pasted_image_payload(
        [
            {
                "name": "clipboard-1.png",
                "mime": "image/png",
                "data": "data:image/png;base64,YWJj",
            }
        ]
    )

    pending = streamlit_stub.session_state.pending_image_attachments
    assert len(pending) == 1
    assert pending[0]["name"] == "clipboard-1.png"
    assert pending[0]["mime"] == "image/png"
    assert pending[0]["data"] == "YWJj"


def test_handle_pasted_image_payload_does_not_replay_consumed_event(monkeypatch):
    demo, streamlit_stub = _import_demo_with_ui_stubs(monkeypatch)
    streamlit_stub.session_state.pending_image_attachments = []

    payload = {
        "eventId": "paste-1",
        "images": [
            {
                "name": "clipboard-1.png",
                "mime": "image/png",
                "data": "data:image/png;base64,YWJj",
            }
        ],
    }

    assert demo._handle_pasted_image_payload(payload) is True
    streamlit_stub.session_state.pending_image_attachments = []

    assert demo._handle_pasted_image_payload(payload) is False
    assert streamlit_stub.session_state.pending_image_attachments == []


def test_pending_image_preview_uses_compact_grid_helper():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert "_PENDING_IMAGE_PREVIEW_COLUMNS = 8" in source
    assert "def _render_pending_image_attachments(" in source
    assert "_render_pending_image_attachments()" in source
    assert "st.columns(min(len(st.session_state.pending_image_attachments), 4))" not in source


def test_demo_mounts_clipboard_capture_near_chat_input():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert "capture_pasted_images(" in source
    assert "_handle_pasted_image_payload(" in source


def test_demo_mounts_image_uploader_with_consume_callback():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    assert "_chat_image_uploader_key(conversation_id)" in source
    assert "on_change=_consume_chat_image_uploader" in source


def test_paste_mount_does_not_block_message_form_rendering():
    source = (Path(__file__).resolve().parents[1] / "demo.py").read_text(encoding="utf-8")

    paste_pos = source.index("pasted_payload = capture_pasted_images")
    form_pos = source.index('with st.form("message_form"')
    paste_block = source[paste_pos:form_pos]

    assert paste_pos < form_pos
    assert "if _handle_pasted_image_payload(pasted_payload):" in paste_block
    assert "st.rerun()" in paste_block
    assert "return" not in paste_block


def test_clipboard_component_filters_to_focused_message_textarea():
    html = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "ui"
        / "clipboard_image_capture"
        / "index.html"
    ).read_text(encoding="utf-8")

    assert 'placeholder === "Type your message..."' in html
    assert "clipboardData" in html
    assert "eventId" in html
    assert "images" in html
    assert "__chatImagePasteSetComponentValue" in html
    assert "setFrameHeight(0)" in html
    assert "streamlit:setComponentValue" in html
