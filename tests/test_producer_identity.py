"""Whether the process that owns a generation is still running.

The lifecycle row records which process is producing a turn so that a later
process can tell an abandoned turn from a live one. Nothing else can: the
build sha is shared by every worker of the same build, and the worker count
belongs to the launch command rather than to any setting this application
reads (see the comment on ``resolve_connection_budget`` in ``app.main``).

The distinction that matters here is three-valued. "Dead" licenses another
process to terminalize someone else's row, so it must be reserved for the case
where this host can actually prove the producer is gone; anything else --
another host, a token this build cannot parse, a platform where liveness is
unavailable -- is "unknown", and unknown must leave the row alone.
"""

from __future__ import annotations

import os
import socket

from app.core.producer_identity import (
    current_producer_token,
    is_producer_alive,
    parse_producer_token,
)

# ----------------------------------------------------------------------
# the token
# ----------------------------------------------------------------------


def test_the_token_identifies_this_host_and_this_process():
    parsed = parse_producer_token(current_producer_token())

    assert parsed is not None
    assert parsed.hostname == socket.gethostname()
    assert parsed.pid == os.getpid()


def test_the_token_is_stable_within_one_process():
    """It is the process's identity, so a second read must not invent a new one."""
    assert current_producer_token() == current_producer_token()


def test_the_token_records_the_process_start_time():
    """Without it a recycled pid reads as the original process, forever.

    The reaper would then decide a dead producer is alive and leave its
    conversation blocked for good.
    """
    parsed = parse_producer_token(current_producer_token())

    assert parsed is not None
    assert parsed.started_at > 0


# ----------------------------------------------------------------------
# liveness
# ----------------------------------------------------------------------


def test_this_process_is_alive():
    assert is_producer_alive(current_producer_token()) is True


def test_a_pid_that_is_not_running_is_dead():
    """The case the reaper exists for: the producer's process is gone."""
    token = f"{socket.gethostname()}:{_unused_pid()}:1"

    assert is_producer_alive(token) is False


def test_a_recycled_pid_is_dead_rather_than_the_process_that_replaced_it():
    """Same pid, different start time -- the original producer is gone."""
    parsed = parse_producer_token(current_producer_token())
    assert parsed is not None
    token = f"{parsed.hostname}:{parsed.pid}:{parsed.started_at - 10_000}"

    assert is_producer_alive(token) is False


def test_a_producer_on_another_host_is_unknown_not_dead():
    """This host cannot see that process, and guessing would reap a live turn."""
    assert is_producer_alive(f"not-{socket.gethostname()}:1:1") is None


def test_a_token_this_build_cannot_parse_is_unknown():
    for token in ("", "garbage", "host:notapid:1", "host:1"):
        assert is_producer_alive(token) is None, token


def test_a_missing_token_is_unknown():
    """Rows written before the column existed name no producer."""
    assert is_producer_alive(None) is None


def _unused_pid() -> int:
    """A pid that is not running, found by asking rather than assuming."""
    import psutil

    for candidate in range(2**22, 2**22 + 5_000):
        if not psutil.pid_exists(candidate):
            return candidate
    raise AssertionError("no unused pid found in the probed range")
