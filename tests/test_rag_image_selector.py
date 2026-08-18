"""Task 11: RAGImageSelector — bounded, post-retrieval image selection.

Tests pin:
  * ``select`` deduplicates by image id, ranks by visual-intent overlap with
    the query, and enforces ``max_images`` / ``max_bytes`` / ``max_pixels`` /
    ``max_vision_tokens`` before any image is attached to a model turn.
  * Oversized images are resized/re-encoded under Pillow rather than dropped
    outright, unless even a single image cannot fit any budget.
  * Missing or undecodable files are skipped rather than raising.
"""

from __future__ import annotations

import io
import random
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from app.services.rag_image_selector import ImageCandidate, RAGImageSelector, SelectedImage


def _write_png(path: Path, *, size: tuple[int, int], seed: int) -> None:
    """Write a deterministic, poorly-compressible PNG so encoded byte size is
    predictable and roughly proportional to width * height * 3 (raw pixels)."""
    rng = random.Random(seed)
    width, height = size
    pixel_data = bytes(rng.getrandbits(8) for _ in range(width * height * 3))
    Image.frombytes("RGB", size, pixel_data).save(path, format="PNG", compress_level=0)


@pytest.fixture
def selector() -> RAGImageSelector:
    return RAGImageSelector(
        max_images=2,
        max_bytes=5_000_000,
        max_pixels=5_000_000,
        max_vision_tokens=50_000,
    )


def image_candidates() -> list[ImageCandidate]:
    """Deterministic factory: three unique images plus one duplicate id.

    One candidate shares an ``image_id`` with another (same underlying image
    re-linked to a second chunk) to exercise dedup; two captions mention
    "chart" to exercise visual-intent ranking against an unrelated caption.
    """
    base = Path(tempfile.mkdtemp(prefix="rag_image_selector_test_"))
    duplicate_id = uuid4()

    chart_path = base / "chart.png"
    _write_png(chart_path, size=(100, 100), seed=1)
    chart_duplicate_path = base / "chart-duplicate.png"
    _write_png(chart_duplicate_path, size=(100, 100), seed=1)
    diagram_path = base / "diagram.png"
    _write_png(diagram_path, size=(100, 100), seed=2)
    unrelated_path = base / "unrelated.png"
    _write_png(unrelated_path, size=(100, 100), seed=3)

    return [
        ImageCandidate(
            image_id=duplicate_id,
            image_path=str(chart_path),
            mime_type="image/png",
            page_number=1,
            caption="Revenue chart comparing quarters",
        ),
        ImageCandidate(
            image_id=duplicate_id,
            image_path=str(chart_duplicate_path),
            mime_type="image/png",
            page_number=1,
            caption="Revenue chart comparing quarters",
        ),
        ImageCandidate(
            image_id=uuid4(),
            image_path=str(diagram_path),
            mime_type="image/png",
            page_number=2,
            caption="Architecture diagram with charts",
        ),
        ImageCandidate(
            image_id=uuid4(),
            image_path=str(unrelated_path),
            mime_type="image/png",
            page_number=3,
            caption="Unrelated cover page photo",
        ),
    ]


def test_selector_enforces_count_bytes_pixels_and_dedup(selector):
    selected = selector.select("compare these charts", image_candidates())

    assert len(selected) <= selector.max_images
    assert sum(item.byte_count for item in selected) <= selector.max_bytes
    assert sum(item.pixel_count for item in selected) <= selector.max_pixels
    assert sum(item.estimated_vision_tokens for item in selected) <= selector.max_vision_tokens

    image_ids = [item.image_id for item in selected]
    assert len(image_ids) == len(set(image_ids)), "duplicate image id must not appear twice"
    for item in selected:
        assert isinstance(item, SelectedImage)
        assert item.data, "selected image must carry loaded bytes"


def test_selector_prefers_visual_intent_matches_when_capped(selector):
    """With only 3 unique images and a cap of 2, both chart-related images win
    over the unrelated cover photo."""
    selected = selector.select("compare these charts", image_candidates())

    assert len(selected) == 2
    captions = {item.caption for item in selected}
    assert "Unrelated cover page photo" not in captions


def test_selector_returns_original_order_for_empty_query(selector):
    candidates = image_candidates()
    # Drop the duplicate so ordering is unambiguous.
    unique_candidates = [candidates[0], candidates[2], candidates[3]]

    selected = selector.select("", unique_candidates)

    assert [item.image_id for item in selected][:2] == [
        unique_candidates[0].image_id,
        unique_candidates[1].image_id,
    ]


def test_selector_stops_before_max_images_when_byte_budget_exhausted():
    candidates = image_candidates()
    one_file_size = Path(candidates[0].image_path).stat().st_size
    tight = RAGImageSelector(
        max_images=5,
        max_bytes=int(one_file_size * 1.5),
        max_pixels=10_000_000,
        max_vision_tokens=1_000_000,
    )

    selected = tight.select("compare these charts", candidates)

    assert 0 < len(selected) < tight.max_images
    assert sum(item.byte_count for item in selected) <= tight.max_bytes


def test_selector_resizes_oversized_image_to_fit_pixel_budget():
    base = Path(tempfile.mkdtemp(prefix="rag_image_selector_resize_test_"))
    large_path = base / "large.png"
    _write_png(large_path, size=(400, 400), seed=7)

    small_budget = RAGImageSelector(
        max_images=1,
        max_bytes=10_000_000,
        max_pixels=20_000,
        max_vision_tokens=1_000_000,
    )
    candidate = ImageCandidate(
        image_id=uuid4(),
        image_path=str(large_path),
        mime_type="image/png",
        page_number=1,
        caption="Large chart",
    )

    selected = small_budget.select("chart", [candidate])

    assert len(selected) == 1
    item = selected[0]
    assert item.pixel_count <= small_budget.max_pixels
    assert item.mime_type == "image/jpeg"
    with Image.open(io.BytesIO(item.data)) as decoded:
        assert decoded.width * decoded.height == item.pixel_count


def test_selector_skips_missing_file_without_raising(selector):
    missing = ImageCandidate(
        image_id=uuid4(),
        image_path="/no/such/path/missing.png",
        mime_type="image/png",
        page_number=1,
        caption="chart that is missing on disk",
    )
    real = image_candidates()[0]

    selected = selector.select("chart", [missing, real])

    assert [item.image_id for item in selected] == [real.image_id]
