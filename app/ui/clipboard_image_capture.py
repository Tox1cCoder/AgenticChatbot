from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).with_name("clipboard_image_capture")
_capture_component = components.declare_component(
    "clipboard_image_capture",
    path=str(_COMPONENT_DIR),
)


def capture_pasted_images(*, key: str) -> dict[str, Any] | list[dict[str, Any]]:
    value = _capture_component(key=key, default={})
    return value if isinstance(value, (dict, list)) else {}
