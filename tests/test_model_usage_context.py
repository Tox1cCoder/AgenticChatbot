"""Tests for the model-usage domain types and async-safe context binding."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_child_context_shares_parent_operation_allocator():
    with begin_usage_operation() as operation:
        parent = UsageContext(operation="parent")
        with bind_usage_context(parent):
            child = parent.child(operation="child")
            with bind_usage_context(child):
                assert current_usage_operation() is operation
                assert current_usage_context().operation == "child"


def test_distinct_operations_receive_distinct_ids():
    with begin_usage_operation() as first:
        first_id = first.operation_id
    with begin_usage_operation() as second:
        second_id = second.operation_id

    assert first_id != second_id


def test_concurrent_attempts_never_reuse_operation_attempt_pair():
    operations = [UsageOperation(), UsageOperation()]
    pairs: list[tuple[UUID, int]] = []
    lock = threading.Lock()

    def allocate(operation: UsageOperation) -> None:
        attempt = operation.allocate_attempt()
        with lock:
            pairs.append((operation.operation_id, attempt))

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(allocate, operations[i % 2]) for i in range(100)]
        for future in futures:
            future.result()

    assert len(pairs) == len(set(pairs)) == 100
    for operation in operations:
        attempts = sorted(attempt for op_id, attempt in pairs if op_id == operation.operation_id)
        assert attempts == list(range(1, len(attempts) + 1))
