"""Tests for the model-usage domain types and async-safe context binding."""

import asyncio
import sys
import threading
from uuid import UUID, uuid4

import pytest

from app.usage import (
    NormalizedUsage,
    UsageContext,
    UsageOperation,
    begin_usage_operation,
    bind_usage_context,
    current_usage_context,
    current_usage_operation,
)


def test_normalized_usage_preserves_unknown_instead_of_zero():
    usage = NormalizedUsage(input_tokens=None, output_tokens=0)
    assert usage.input_tokens is None
    assert usage.output_tokens == 0


@pytest.mark.asyncio
async def test_usage_context_isolated_between_concurrent_users():
    async def read_bound(user_id: UUID):
        with bind_usage_context(UsageContext(user_id=user_id)):
            await asyncio.sleep(0)
            return current_usage_context().user_id

    first, second = uuid4(), uuid4()
    assert await asyncio.gather(read_bound(first), read_bound(second)) == [first, second]


@pytest.mark.parametrize("bool_value", [True, False])
@pytest.mark.parametrize("field_name", ["input_tokens", "generated_images"])
def test_normalized_usage_rejects_boolean_token_counts(field_name, bool_value):
    with pytest.raises(TypeError):
        NormalizedUsage(**{field_name: bool_value})


@pytest.mark.parametrize(
    "field_name",
    [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "input_text_tokens",
        "input_image_tokens",
        "output_text_tokens",
        "output_image_tokens",
        "generated_images",
    ],
)
def test_normalized_usage_rejects_negative_token_counts(field_name):
    with pytest.raises(ValueError):
        NormalizedUsage(**{field_name: -1})


def test_normalized_usage_rejects_none_generated_images():
    # generated_images is declared `int = 0`, not `int | None` — unlike the
    # other numeric fields, None is not a valid "unknown" for it.
    with pytest.raises(TypeError):
        NormalizedUsage(generated_images=None)


def test_child_context_shares_parent_operation_allocator():
    with begin_usage_operation() as operation:
        parent = UsageContext(operation="parent")
        with bind_usage_context(parent):
            assert current_usage_operation() is operation
            first_attempt = current_usage_operation().allocate_attempt()

            child = parent.child(operation="child")
            with bind_usage_context(child):
                assert current_usage_operation() is operation
                assert current_usage_context().operation == "child"
                second_attempt = current_usage_operation().allocate_attempt()

    assert (first_attempt, second_attempt) == (1, 2)


def test_distinct_operations_receive_distinct_ids():
    with begin_usage_operation() as first:
        first_id = first.operation_id
    with begin_usage_operation() as second:
        second_id = second.operation_id

    assert first_id != second_id


def test_concurrent_attempts_never_reuse_operation_attempt_pair():
    operation = UsageOperation()
    pairs: list[tuple[UUID, int]] = []
    pairs_lock = threading.Lock()
    thread_count = 32
    allocations_per_thread = 200

    def allocate_many() -> None:
        for _ in range(allocations_per_thread):
            attempt = operation.allocate_attempt()
            with pairs_lock:
                pairs.append((operation.operation_id, attempt))

    # A tiny switch interval forces CPython to check the GIL-release request
    # far more often, maximizing the chance that concurrent threads actually
    # interleave inside allocate_attempt's read-modify-write — without this,
    # the race window is narrow enough that even thousands of iterations can
    # pass without ever exposing an unlocked implementation (verified: see
    # the fix section of task-1-report.md).
    original_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=allocate_many) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(original_interval)

    total = thread_count * allocations_per_thread
    assert len(pairs) == len(set(pairs)) == total
    assert sorted(attempt for _, attempt in pairs) == list(range(1, total + 1))


def test_allocate_attempt_serializes_concurrent_callers():
    # Deterministic mutual-exclusion proof, independent of GIL scheduling
    # luck: hold the operation's lock directly from the test thread, prove a
    # concurrent allocate_attempt() call blocks while it's held, then prove
    # it completes as soon as the lock is released.
    operation = UsageOperation()
    attempts: list[int] = []

    operation._lock.acquire()
    try:
        worker = threading.Thread(target=lambda: attempts.append(operation.allocate_attempt()))
        worker.start()
        worker.join(timeout=0.2)
        assert worker.is_alive(), "allocate_attempt should block while the lock is held"
    finally:
        operation._lock.release()

    worker.join(timeout=1)
    assert not worker.is_alive()
    assert attempts == [1]
