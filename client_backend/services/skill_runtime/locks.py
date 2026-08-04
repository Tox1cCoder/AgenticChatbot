"""Two-level mutual exclusion for skill mutations.

A skill install rewrites shared on-disk state (the bundle directory, its prepared
runtime, the registry view). Two concurrent installs of the same skill would
interleave those steps and leave a bundle whose runtime belongs to the other
version, so every mutation runs under a named lock.

Two levels are required, not one:

* an in-process ``asyncio.Lock`` serializes coroutines in this event loop, which
  a file lock alone cannot do -- most OS file locks are per-process, so the same
  process would happily re-enter its own lock; and
* a ``filelock.FileLock`` serializes separate sidecar processes sharing one
  profile directory (a second launch, or the CLI running beside the app).

Waiting is always bounded: an installation that cannot get the lock reports a
retryable ``SKILL_INSTALL_LOCKED`` instead of hanging the request.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from weakref import WeakKeyDictionary

import filelock

from client_backend.core.config import client_settings
from client_backend.core.paths import get_skill_locks_root

_LOCK_SUFFIX = ".lock"
SKILLS_MUTATION_SCOPE = "skills-mutation"

# One asyncio.Lock per resolved lock file, per event loop. The loop dimension is
# required: an asyncio.Lock binds to the loop that first awaits it and raises if
# reused from another, which a module-level cache would otherwise do the moment a
# second loop runs in this process (the Windows launcher's asyncio.run, a CLI
# invocation, or successive test loops). Weak keys let a finished loop's locks go.
_process_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Path, asyncio.Lock]] = (
    WeakKeyDictionary()
)


def _process_lock_for(path: Path) -> asyncio.Lock:
    """Return this loop's in-process lock for one resolved lock file."""
    loop = asyncio.get_running_loop()
    locks = _process_locks.setdefault(loop, {})
    return locks.setdefault(path, asyncio.Lock())


class SkillLockTimeoutError(Exception):
    """Raised when a skill mutation lock cannot be acquired in time."""

    def __init__(self, scope: str) -> None:
        super().__init__(f"another skill operation is already holding '{scope}'")
        self.scope = scope


def _lock_path(user_id: str, scope: str) -> Path:
    """Resolve one scope's lock file inside the user's profile lock directory.

    The filename is a digest of the scope rather than the scope text: a scope
    carries a skill name that came from archive front matter, and pasting that
    into a filename would reintroduce every path and reserved-name problem the
    archive validator exists to prevent.
    """
    root = get_skill_locks_root(user_id)
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
    return root / f"{digest}{_LOCK_SUFFIX}"


def _resolve_timeout(timeout_seconds: float | None) -> float:
    if timeout_seconds is not None:
        return float(timeout_seconds)
    return float(client_settings.skill_install_lock_timeout_seconds)


@contextlib.asynccontextmanager
async def profile_lock(
    user_id: str,
    scope: str,
    timeout_seconds: float | None = None,
) -> AsyncIterator[None]:
    """Hold the named mutation lock for one user profile.

    Args:
        user_id: Profile owner; validated as a single path component.
        scope: Logical resource, e.g. ``"uploads"`` or ``"skill:google-calendar"``.
        timeout_seconds: Override for the configured wait bound.

    Raises:
        SkillLockTimeoutError: The lock was held for longer than the bound.
    """
    deadline = _resolve_timeout(timeout_seconds)
    path = _lock_path(user_id, scope)
    path.parent.mkdir(parents=True, exist_ok=True)

    process_lock = _process_lock_for(path)
    try:
        await asyncio.wait_for(process_lock.acquire(), timeout=deadline)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise SkillLockTimeoutError(scope) from exc

    # thread_local=False is required, not a preference. filelock's default keys
    # its recursion counter by thread, while `asyncio.to_thread` is free to run
    # the acquire and the release on two different pool threads -- and then the
    # release no-ops and the OS lock is held until the process exits, so every
    # later skill operation for this profile fails as locked. Exclusion here is
    # per profile scope, coordinated by the asyncio lock above, never per thread.
    file_lock = filelock.FileLock(str(path), thread_local=False)
    try:
        try:
            # FileLock.acquire blocks the thread, so it must never run on the
            # event loop: doing so would stall every other request for the whole
            # timeout instead of just this one.
            await asyncio.to_thread(file_lock.acquire, timeout=deadline)
        except filelock.Timeout as exc:
            raise SkillLockTimeoutError(scope) from exc
        try:
            yield
        finally:
            await asyncio.to_thread(file_lock.release)
    finally:
        process_lock.release()
