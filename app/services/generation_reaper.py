"""Reclaiming generations whose producing worker no longer exists.

``uq_generations_active_per_conversation`` admits one active row per
conversation. That is what stops a conversation running two turns at once, and
its cost is that a row left active by a worker that died blocks that
conversation permanently: the next question's INSERT raises
``UniqueViolation``, the API reports ``conversation_turn_conflict``, and no
retry can clear it because no worker exists to finish the turn. In development
this is routine rather than rare -- uvicorn's reloader replaces the serving
child on every code change, and any turn streaming at that moment is
abandoned.

Two halves, because a graceful exit and a crash are different facts:

* :func:`terminalize_own_generations` runs on shutdown, where the process
  knows it is leaving and can name its own rows with certainty.
* :func:`reclaim_orphaned_generations` runs on startup and cleans up after the
  exits that had no chance to do that -- a kill, a crash, a power loss.

The startup half is scoped to producers this host can prove are gone. "Fail
everything active at startup" would be simpler and is wrong: it is correct only
if there is exactly one worker, which is an assumption this application
deliberately does not encode (see the comment on ``resolve_connection_budget``
in ``app.main``), and under ``--workers N`` it would terminalize a peer's
streaming turn on every restart.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.core.producer_identity import current_producer_token, is_producer_alive

logger = logging.getLogger(__name__)

__all__ = [
    "TERMINAL_REASON_LOST",
    "TERMINAL_REASON_SHUTDOWN",
    "reclaim_orphaned_generations",
    "terminalize_own_generations",
]

# Recorded on the row, and distinct on purpose. "The worker went away without
# finishing" and "the worker was shutting down" are different diagnoses, and
# collapsing them would make a routine reload indistinguishable from a crash.
TERMINAL_REASON_LOST = "producer_lost"
TERMINAL_REASON_SHUTDOWN = "producer_shutdown"


class _Reclaimable(Protocol):
    """The repository surface the reaper needs, and nothing more."""

    async def aget_active_producer_tokens(self) -> list[str | None]: ...

    async def afail_active_by_producer(
        self, producer_tokens: list[str], *, terminal_reason: str
    ) -> int: ...


async def reclaim_orphaned_generations(repository: _Reclaimable) -> int:
    """Terminalize active generations whose producer is provably gone.

    Returns how many rows were reclaimed. Only ``False`` from
    :func:`is_producer_alive` licenses a write; ``None`` -- another host, a
    token this build cannot parse, a row written before the column existed --
    leaves the row alone, because the cost of being wrong is terminalizing a
    turn that is still streaming.
    """
    tokens = await repository.aget_active_producer_tokens()
    if not tokens:
        return 0

    dead: list[str] = []
    unknown = 0
    for token in tokens:
        liveness = is_producer_alive(token)
        if liveness is False and token:
            dead.append(token)
        elif liveness is None:
            unknown += 1

    if unknown:
        # Worth saying out loud: these conversations stay blocked, and no
        # later startup will clear them either. An operator seeing this has
        # something to act on; silence would look like success.
        logger.warning(
            "%s active generation(s) name a producer this host cannot verify "
            "and were left untouched",
            unknown,
        )

    if not dead:
        return 0

    reclaimed = await repository.afail_active_by_producer(
        dead, terminal_reason=TERMINAL_REASON_LOST
    )
    logger.warning(
        "Reclaimed %s generation(s) abandoned by %s departed worker(s); "
        "their conversations can start a new turn again",
        reclaimed,
        len(dead),
    )
    return reclaimed


async def terminalize_own_generations(repository: _Reclaimable) -> int:
    """Fail this process's still-active generations as it shuts down.

    Called from the lifespan shutdown path so the ordinary case -- a reload or
    a Ctrl-C while a turn is streaming -- is resolved by the process that knows
    the answer, instead of being left for a later startup to infer.
    """
    terminalized = await repository.afail_active_by_producer(
        [current_producer_token()], terminal_reason=TERMINAL_REASON_SHUTDOWN
    )
    if terminalized:
        logger.warning("Terminalized %s generation(s) still active at shutdown", terminalized)
    return terminalized
