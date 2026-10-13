"""Source folders the development servers watch for auto-reload."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Only the packages the servers import. uvicorn's default watches the working
# directory, which also covers skills/ and the scratch scripts agents write at the
# repo root; one such *.py restarts the server and kills the turn that wrote it.
RELOAD_PACKAGES = ("app", "client_backend", "shared")


def reload_dirs() -> list[str]:
    """Absolute paths of the source packages present in this checkout or bundle."""
    return [str(REPO_ROOT / name) for name in RELOAD_PACKAGES if (REPO_ROOT / name).is_dir()]
