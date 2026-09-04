"""Bounds on the focused-web evidence settings.

Each of these caps what reaches model context. A setting that silently accepted
0 or a million would defeat the boundary it exists to enforce, so the range is
part of the contract rather than a comment.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings

_EVIDENCE_BOUNDS = {
    "web_search_max_results": (8, 1, 20),
    "web_search_result_max_chars": (24_000, 2_000, 100_000),
    "web_open_max_urls": (4, 1, 10),
    "web_open_max_excerpts": (8, 1, 20),
    "web_open_max_chars": (18_000, 2_000, 80_000),
    "tool_result_focus_max_excerpts": (8, 1, 20),
    "tool_result_focus_max_chars": (16_000, 2_000, 80_000),
}


@pytest.mark.parametrize(("name", "bounds"), sorted(_EVIDENCE_BOUNDS.items()))
def test_evidence_setting_has_the_declared_default(name, bounds):
    default, _, _ = bounds

    assert getattr(settings, name) == default


@pytest.mark.parametrize(("name", "bounds"), sorted(_EVIDENCE_BOUNDS.items()))
def test_evidence_setting_accepts_both_ends_of_its_range(name, bounds):
    _, minimum, maximum = bounds

    assert getattr(Settings(**{name: minimum}), name) == minimum
    assert getattr(Settings(**{name: maximum}), name) == maximum


@pytest.mark.parametrize(("name", "bounds"), sorted(_EVIDENCE_BOUNDS.items()))
def test_evidence_setting_rejects_values_outside_its_range(name, bounds):
    _, minimum, maximum = bounds

    with pytest.raises(ValidationError):
        Settings(**{name: minimum - 1})
    with pytest.raises(ValidationError):
        Settings(**{name: maximum + 1})
