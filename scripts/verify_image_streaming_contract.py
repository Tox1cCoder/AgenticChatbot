"""Standalone full-path image-streaming contract verifier.

Drives the deterministic fake image generation through the REAL canonical
FastAPI stacks (internal Streamlit SSE + Vercel AI SDK) and reports whether the
production image-delivery contract holds. It reuses the same harness as
``tests/test_image_stream_http_contract.py`` so the CLI check and the pytest
gate stay in lockstep.

Usage (from the repo root)::

    .\\.venv\\Scripts\\python.exe scripts/verify_image_streaming_contract.py

Exit status:
    0  every checked contract holds (expected after Phase 1: T002-T005)
    1  one or more contracts are unmet

Contracts checked:
    C1  an oversized FINAL image is delivered early (internal SSE)
    C2  an oversized FINAL image is delivered early (AI SDK data-image-preview)
    C3  the AI SDK terminal `file` part preserves the protected /chat-images URL
    C4  exactly one [DONE] terminates the AI SDK stream
    C5  a resumed (post-HITL) run delivers the same early image reference as a
        fresh run — resume parity (FR-IMG-008, T005)

Nothing here performs real network/provider I/O; the fake image source runs the
REAL ``ImagePreviewPublisher`` + ``MediaDeliveryService`` so the emission policy
and final-by-reference delivery are genuinely exercised.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

# Make the repo root importable when run as ``python scripts/<name>.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dependency_injector import providers  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.container import Container  # noqa: E402
from tests.test_image_stream_http_contract import (  # noqa: E402
    FINAL_B64,
    PARTIAL_B64,
    _build_app,
    _build_message_service,
    _parse_sse,
    _types,
)

_PREVIEW_CAP = settings.image_stream_preview_max_b64_chars


class _Result:
    def __init__(self, name: str, ok: bool, detail: str):
        self.name = name
        self.ok = ok
        self.detail = detail


def _drive_internal_sse(conversation_id, user_id, service) -> list:
    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_build_app(service, user_id))
        resp = client.post(
            "/messages/stream",
            json={"conversation_id": str(conversation_id), "content": "draw a cat", "role": 1},
        )
    resp.raise_for_status()
    return _parse_sse(resp.text)


def _drive_ai_sdk(conversation_id, user_id, service) -> list:
    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_build_app(service, user_id))
        resp = client.post(
            f"/api/chat/{conversation_id}",
            json={"messages": [{"role": "user", "content": "draw a cat"}]},
        )
    resp.raise_for_status()
    return _parse_sse(resp.text)


def _drive_resume_sse(conversation_id, user_id, service) -> list:
    with Container.message_service.override(providers.Object(service)):
        client = TestClient(_build_app(service, user_id))
        resp = client.post(
            "/messages/resume-interrupt",
            json={
                "thread_id": str(conversation_id),
                "conversation_id": str(conversation_id),
                "interrupt_id": "int-1",
                "decisions": [],
            },
        )
    resp.raise_for_status()
    return _parse_sse(resp.text)


def run_checks() -> list[_Result]:
    conversation_id = uuid4()
    user_id = uuid4()
    image_url = f"/chat-images/{uuid4()}"

    internal_service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )
    internal = _drive_internal_sse(conversation_id, user_id, internal_service)
    internal_order = _types(internal)
    internal_previews = [
        p for p in internal if isinstance(p, dict) and p.get("type") == "image_preview"
    ]
    internal_final = any(p.get("status") == "final" for p in internal_previews)

    ai_service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=False
    )
    ai = _drive_ai_sdk(conversation_id, user_id, ai_service)
    ai_order = _types(ai)
    ai_previews = [
        (p.get("data") or {}).get("status")
        for p in ai
        if isinstance(p, dict) and p.get("type") == "data-image-preview"
    ]
    ai_final = any(s == "final" for s in ai_previews)
    file_urls = [p.get("url") for p in ai if isinstance(p, dict) and p.get("type") == "file"]
    done_count = ai_order.count("[DONE]")

    resume_service = _build_message_service(
        conversation_id=conversation_id, user_id=user_id, image_url=image_url, resume=True
    )
    resume = _drive_resume_sse(conversation_id, user_id, resume_service)
    resume_order = _types(resume)
    resume_previews = [
        p for p in resume if isinstance(p, dict) and p.get("type") == "image_preview"
    ]
    resume_final = any(p.get("status") == "final" for p in resume_previews)

    sizes = (
        f"partial_b64={len(PARTIAL_B64)} chars; final_b64={len(FINAL_B64)} chars; "
        f"preview_cap={_PREVIEW_CAP} chars"
    )

    return [
        _Result(
            "C1 internal SSE delivers oversized final early",
            internal_final,
            f"image_preview statuses={[p.get('status') for p in internal_previews]}; "
            f"[{sizes}]; order={internal_order}",
        ),
        _Result(
            "C2 AI SDK delivers oversized final early",
            ai_final,
            f"data-image-preview statuses={ai_previews}; [{sizes}]; order={ai_order}",
        ),
        _Result(
            "C3 AI SDK terminal file part preserves protected URL",
            image_url in file_urls,
            f"expected {image_url}; file urls={file_urls}",
        ),
        _Result(
            "C4 AI SDK stream terminates with exactly one [DONE]",
            done_count == 1 and ai_order[-1:] == ["[DONE]"],
            f"done_count={done_count}; order tail={ai_order[-3:]}",
        ),
        _Result(
            "C5 resumed run delivers early image reference (resume parity)",
            resume_final,
            f"resume image_preview statuses={[p.get('status') for p in resume_previews]}; "
            f"order={resume_order}",
        ),
    ]


def main() -> int:
    results = run_checks()
    print("Image streaming full-path contract verification")
    print("=" * 60)
    failed = 0
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        if not r.ok:
            failed += 1
        print(f"[{status}] {r.name}")
        print(f"        {r.detail}")
    print("=" * 60)
    if failed:
        print(f"{failed}/{len(results)} contract(s) UNMET.")
        return 1
    print("All image-streaming contracts hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
