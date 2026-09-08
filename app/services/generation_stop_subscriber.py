"""The receiving half of a distributed Stop.

Without this, ``GenerationControlBus.publish_stop`` broadcast into a void: the
row transitioned, the signal was published, and no process was listening. A
Stop landing on a worker other than the streaming one changed the database and
nothing else.

What this subscriber buys is *latency*, not correctness. The durable
transition happens before the publication, so the authority is always the row
and a worker that never receives a signal still stops when its own status check
comes round (``MessageService`` polls it at tool and model boundaries). What
this adds is interrupting a worker that is blocked inside a provider call right
now, where no check point is coming.

Broadcast, not addressed: the publisher does not know which worker owns the
generation, and each subscriber simply asks its own registry whether the id is
one of its own. Addressing it would need a worker registry the lifecycle
deliberately does not keep — one more thing to go stale while a turn is running.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.generation_control_bus import StopSignal
from app.services.generation_registry import get_generation_registry

logger = logging.getLogger(__name__)

__all__ = ["handle_stop_signal", "install_generation_stop_subscriber"]


async def handle_stop_signal(signal: StopSignal) -> bool:
    """Cancel the named generation if this process owns it.

    Returns whether a local entry was found. ``False`` is the ordinary case for
    every worker but one, and is not a failure: the broadcast reaches all of
    them and only the owner acts.

    Never raises. This runs inside the subscriber loop, which is how every
    *future* Stop arrives; letting one exception escape would disable
    cancellation for the whole process until it restarts.
    """
    try:
        return get_generation_registry().request_cancel(signal.generation_id)
    except Exception as exc:  # noqa: BLE001 - the loop must survive
        logger.warning(
            "Could not apply a stop signal for %s: %s",
            signal.generation_id,
            type(exc).__name__,
        )
        return False


async def install_generation_stop_subscriber(bus: Any) -> bool:
    """Attach this process's registry to the control bus.

    Called once at startup. Returns whether the subscription was established;
    ``False`` leaves Stop working through the durable status alone, which is
    slower but not wrong, so a Redis that is down must not prevent the
    application from serving requests.
    """
    if bus is None:
        return False
    try:
        await bus.subscribe(handle_stop_signal)
    except Exception as exc:  # noqa: BLE001 - degrades to the durable path
        logger.warning(
            "Generation stop signals will not be delivered to this worker: %s",
            type(exc).__name__,
        )
        return False
    logger.info("Subscribed this worker to generation stop signals")
    return True
