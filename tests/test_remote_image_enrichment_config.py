"""Configuration and import contracts for provider-native image enrichment."""

from __future__ import annotations

import ast
from pathlib import Path

from app.core.config import Settings

_ROOT = Path(__file__).resolve().parents[1]
_RETIRED_SETTINGS = (
    "vision_image_" + "verification_enabled",
    "image_" + "verification_model",
    "image_" + "verification_media_resolution",
    "image_" + "verification_thinking_level",
    "image_" + "verification_confidence_threshold",
    "image_" + "verification_max_candidates",
    "image_" + "verification_timeout_seconds",
    "image_" + "verification_thumbnail_timeout_seconds",
    "verified_image_" + "cache_max_bytes",
)
_RETIRED_MODULES = (
    ("ai", "image_" + "verification_flow.py"),
    ("ai", "visual_" + "verifier.py"),
    ("services", "thumbnail_" + "batch.py"),
    ("services", "verified_image_" + "bytes.py"),
)


def test_remote_image_enrichment_replaces_retired_settings() -> None:
    fields = Settings.model_fields

    assert fields["remote_image_enrichment_enabled"].default is True
    assert not set(_RETIRED_SETTINGS) & set(fields)


def test_retired_remote_image_modules_are_absent_from_application() -> None:
    retired_names = {filename.removesuffix(".py") for _, filename in _RETIRED_MODULES}
    imported_retired_modules: list[str] = []

    for package, filename in _RETIRED_MODULES:
        assert not (_ROOT / "app" / package / filename).exists()

    for path in (_ROOT / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name.rsplit(".", 1)[-1] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                if node.module:
                    imported.add(node.module.rsplit(".", 1)[-1])
            else:
                continue
            for name in sorted(imported & retired_names):
                imported_retired_modules.append(f"{path.relative_to(_ROOT)}: {name}")

    assert imported_retired_modules == []
