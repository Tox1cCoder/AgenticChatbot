from pathlib import Path


def test_rollout_runbook_documents_safety_and_rollback() -> None:
    text = Path("docs/operations/web-research-rollout.md").read_text(encoding="utf-8")
    assert "no token means no image" in text.lower()
    assert "buffered" in text.lower()
    assert "rollback" in text.lower()
    assert "expiry" in text.lower()


def test_streamlit_handles_source_upserts_on_send_and_resume() -> None:
    demo = Path("demo.py").read_text(encoding="utf-8")

    assert demo.count('event_type == "sources"') >= 2
    assert "stream_web_sources" in demo
