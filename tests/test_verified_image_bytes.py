"""Hand-off of verified image bytes from verification to registration.

Verification decodes and validates the bytes; registration happens later, in
``message_service``, after the graph has finished and every ContextVar scope
around the tool call has closed. This store bridges that gap without holding
the whole turn's discarded candidates: only approved images are remembered, a
reader takes its entry, and a total-byte budget evicts whatever nobody claimed.
"""

from __future__ import annotations

import pytest

from app.services.verified_image_bytes import (
    forget_conversation_bytes,
    remember_verified_bytes,
    take_verified_bytes,
)
from app.services.web_image_service import FetchedWebImage

CONVERSATION = "conv-a"
URL = "https://cdn.example/team.jpg"


def _image(size: int = 16) -> FetchedWebImage:
    return FetchedWebImage(
        content=b"x" * size, media_type="image/jpeg", width=995, height=565
    )


@pytest.fixture(autouse=True)
def _clean():
    forget_conversation_bytes(CONVERSATION)
    forget_conversation_bytes("conv-b")
    yield
    forget_conversation_bytes(CONVERSATION)
    forget_conversation_bytes("conv-b")


def test_remembered_bytes_come_back_once():
    remember_verified_bytes(CONVERSATION, URL, _image())

    assert take_verified_bytes(CONVERSATION, URL) == _image()
    assert take_verified_bytes(CONVERSATION, URL) is None, "taking must free the entry"


def test_another_conversation_cannot_take_them():
    remember_verified_bytes(CONVERSATION, URL, _image())

    assert take_verified_bytes("conv-b", URL) is None
    assert take_verified_bytes(CONVERSATION, URL) is not None


def test_an_unknown_url_is_a_miss_not_an_error():
    assert take_verified_bytes(CONVERSATION, "https://cdn.example/other.jpg") is None


def test_missing_conversation_id_never_shares_a_bucket():
    remember_verified_bytes(None, URL, _image())

    assert take_verified_bytes(CONVERSATION, URL) is None


def test_the_byte_budget_evicts_the_oldest_entries():
    """Nothing guarantees a reader arrives: an answer can drop the image.

    Without a budget those entries accumulate for the process's lifetime, one
    turn's worth of megabyte thumbnails at a time.
    """
    remember_verified_bytes(CONVERSATION, "https://cdn.example/1.jpg", _image(400), budget=1000)
    remember_verified_bytes(CONVERSATION, "https://cdn.example/2.jpg", _image(400), budget=1000)
    remember_verified_bytes(CONVERSATION, "https://cdn.example/3.jpg", _image(400), budget=1000)

    assert take_verified_bytes(CONVERSATION, "https://cdn.example/1.jpg") is None
    assert take_verified_bytes(CONVERSATION, "https://cdn.example/3.jpg") is not None


def test_an_entry_larger_than_the_whole_budget_is_not_stored():
    remember_verified_bytes(CONVERSATION, URL, _image(4096), budget=64)

    assert take_verified_bytes(CONVERSATION, URL) is None


def test_forgetting_a_conversation_drops_only_its_own_entries():
    remember_verified_bytes(CONVERSATION, URL, _image())
    remember_verified_bytes("conv-b", URL, _image())

    forget_conversation_bytes(CONVERSATION)

    assert take_verified_bytes(CONVERSATION, URL) is None
    assert take_verified_bytes("conv-b", URL) is not None
