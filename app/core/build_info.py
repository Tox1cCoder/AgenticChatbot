"""Build identity for health diagnostics.

Answers "which build is this process running?" so a stale process is
immediately distinguishable from current source. Resolution order:

1. ``BUILD_SHA`` — stamped by the deployment pipeline. Authoritative.
2. The git checkout's ``HEAD`` commit — the development fallback.
3. ``"unknown"`` — a packaged install with neither. A missing build identity is
   a degraded diagnostic, never a health failure, so nothing here raises.

Cached: the answer cannot change within a process lifetime, and health probes
are hot. Tests call ``resolve_build_info.cache_clear()``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

UNKNOWN_BUILD: dict[str, str] = {"build_sha": "unknown", "build_source": "unknown"}


def _sha_from_git(repo_root: Path) -> str | None:
    """Read ``HEAD`` without shelling out to git (may be absent at runtime)."""
    git_dir = repo_root / ".git"
    try:
        if git_dir.is_file():
            # Worktree/submodule: ".git" is a file pointing at the real gitdir.
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git_dir = Path(pointer.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (repo_root / git_dir).resolve()

        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None

        ref = head.split(":", 1)[1].strip()
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip() or None

        # Packed refs: the loose ref file does not exist after `git gc`.
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith(("#", "^")):
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip() == ref:
                    return parts[0].strip()
    except OSError:
        return None
    return None


@lru_cache(maxsize=1)
def resolve_build_info() -> dict[str, str]:
    """Return ``{"build_sha": ..., "build_source": "env"|"git"|"unknown"}``."""
    stamped = (os.getenv("BUILD_SHA") or "").strip()
    if stamped:
        return {"build_sha": stamped, "build_source": "env"}

    sha = _sha_from_git(_REPO_ROOT)
    if sha:
        return {"build_sha": sha, "build_source": "git"}

    return dict(UNKNOWN_BUILD)
