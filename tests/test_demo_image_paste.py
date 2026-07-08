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
