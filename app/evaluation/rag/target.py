"""Authenticated, scope-aware RAG target adapters for experiments."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from .contracts import RAGEvaluationInput, RAGEvaluationOutput
from .corpus import EvaluationScope
from .metrics import output_from_mapping


def build_http_target(
    endpoint: str | None = None, token: str | None = None, timeout_seconds: float = 120.0
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Build the production-facing target without leaking credentials into traces."""
    target_endpoint = endpoint or os.getenv("RAG_EVALUATION_TARGET_URL")
    bearer_token = token or os.getenv("RAG_EVALUATION_BEARER_TOKEN")
    if not target_endpoint or not bearer_token:
        raise RuntimeError("RAG_EVALUATION_TARGET_URL and RAG_EVALUATION_BEARER_TOKEN are required")

    def target(inputs: Mapping[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            target_endpoint,
            json=dict(inputs),
            headers={"Authorization": f"Bearer {bearer_token}"},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return dict(payload.get("output", payload))

    return target


def load_local_target(factory_path: str | None) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """Load an explicit local RAG target; never synthesize answers from gold data."""
    if not factory_path:
        raise RuntimeError(
            "offline evaluation requires --target MODULE:FUNCTION for a local RAG target; "
            "the repository has no configured runtime service target"
        )
    module_name, separator, function_name = factory_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--target must be MODULE:FUNCTION")
    target = getattr(importlib.import_module(module_name), function_name)
    if not callable(target):
        raise ValueError("--target must resolve to a callable RAG target")
    return target


def scoped_target(
    target: Callable[[Mapping[str, Any]], Mapping[str, Any]], scope: EvaluationScope
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Bind experiment calls to the tenant and conversation generated for the corpus."""

    def invoke(inputs: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(inputs)
        payload["user_id"] = scope.user_id
        payload["conversation_id"] = scope.conversation_id
        return dict(target(payload))

    return invoke


def evaluate_target_call(
    target: Callable[[Mapping[str, Any]], Mapping[str, Any]], evaluation_input: RAGEvaluationInput
) -> RAGEvaluationOutput:
    """Invoke a target and normalize its observable response to the typed contract."""
    return output_from_mapping(
        target(
            {
                "question": evaluation_input.question,
                "user_id": evaluation_input.user_id,
                "conversation_id": evaluation_input.conversation_id,
            }
        )
    )
