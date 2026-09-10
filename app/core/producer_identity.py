"""Which process is producing a generation, and whether it still exists.

A generation row is written by the worker that streams the turn, and that
worker can die mid-turn -- a crash, a hard kill, or, in development, uvicorn's
reloader replacing the child on every code change. The row is then abandoned
in an active status, and because ``uq_generations_active_per_conversation``
admits one active row per conversation, the conversation cannot start another
turn until something terminalizes it.

Deciding that requires naming the producer. Nothing already recorded can:
``build_sha`` is identical across every worker of one build, and the worker
count deliberately lives in the launch command rather than in a setting (see
``_verify_async_database_ready``), so "there is only one worker" is not a fact
this module may assume.

Liveness is therefore three-valued, and the third value is the important one.
``False`` licenses one process to terminalize another's row, so it is returned
only where this host can prove the producer is gone. Another host, a token
this build cannot parse, or a platform that cannot answer all yield ``None``
-- unknown -- and the caller must leave such a row alone.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = [
    "ProducerToken",
    "current_producer_token",
    "is_producer_alive",
    "parse_producer_token",
]

# The token is ``hostname:pid:started_at``. ``started_at`` is what makes a
# recycled pid distinguishable from the process that originally held it;
# without it a new process inheriting the pid would read as the dead producer
# still running, and its conversation would stay blocked permanently.
_SEPARATOR = ":"

# ``started_at`` when this platform cannot report a process start time. The
# token stays usable as an identity; liveness for it is unknown rather than
# guessed.
UNKNOWN_START = 0


@dataclass(frozen=True)
class ProducerToken:
    """The parsed parts of a producer token."""

    hostname: str
    pid: int
    started_at: int


def _psutil() -> Any | None:
    """Return ``psutil``, or ``None`` where it is not installed.

    Imported through a function so the absence of an optional dependency
    degrades to "liveness unknown" instead of breaking startup. ``os.kill(pid,
    0)`` is deliberately not used as a fallback: on Windows CPython maps a
    signal other than the CTRL events onto ``TerminateProcess``, so the POSIX
    idiom for "does this process exist" would kill the process it asked about.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is pinned in requirements
        return None
    return psutil


def _process_started_at(pid: int) -> int:
    psutil = _psutil()
    if psutil is None:
        return UNKNOWN_START
    try:
        return int(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 - any failure here means "unknown"
        return UNKNOWN_START


@lru_cache(maxsize=1)
def current_producer_token() -> str:
    """This process's identity, computed once.

    Cached because it is an identity rather than a measurement: a second read
    returning a different token would make this process unable to recognise
    its own rows at shutdown.
    """
    pid = os.getpid()
    return _SEPARATOR.join((socket.gethostname(), str(pid), str(_process_started_at(pid))))


def parse_producer_token(token: str | None) -> ProducerToken | None:
    """Parse a token, or ``None`` if it is absent or not in this format.

    Split from the right so a hostname containing the separator cannot shift
    the numeric fields.
    """
    if not token:
        return None
    hostname, _, remainder = str(token).partition(_SEPARATOR)
    pid_text, separator, started_text = remainder.rpartition(_SEPARATOR)
    if not hostname or not separator:
        return None
    try:
        pid = int(pid_text)
        started_at = int(started_text)
    except ValueError:
        return None
    if pid <= 0:
        return None
    return ProducerToken(hostname=hostname, pid=pid, started_at=started_at)


def is_producer_alive(token: str | None) -> bool | None:
    """Whether the process named by ``token`` is running.

    ``True`` it is, ``False`` it provably is not, ``None`` this host cannot
    tell. Callers must treat ``None`` as "leave the row alone": the cost of a
    wrong ``False`` is terminalizing a turn that is still streaming.
    """
    parsed = parse_producer_token(token)
    if parsed is None:
        return None
    if parsed.hostname != socket.gethostname():
        return None
    if parsed.started_at == UNKNOWN_START:
        return None

    psutil = _psutil()
    if psutil is None:
        return None
    if not psutil.pid_exists(parsed.pid):
        return False
    started_at = _process_started_at(parsed.pid)
    if started_at == UNKNOWN_START:
        return None
    return started_at == parsed.started_at
