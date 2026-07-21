"""Container contract tests for the model-usage recorder wiring (Task 6).

Covers the three providers added to ``app.core.container.Container``:

* ``model_usage_repository`` -- a plain ``providers.Factory`` (a repository per
  resolution, matching every other repository provider in the container).
* ``model_usage_metrics`` -- wraps the existing module-level
  ``app.observability.model_usage.model_usage_metrics`` singleton (D7: reused,
  never a second ``ModelUsageMetrics()``, which would split the Prometheus
  registry).
* ``model_usage_recorder`` -- a ``providers.Singleton`` wired to the
  repository and metrics above.

``model_usage_service`` (``ModelUsageService`` / ``IModelUsageService``) is
deferred to Task 12/13 and does not exist yet -- intentionally not asserted
here (see controller decision D6).
"""

from __future__ import annotations

from dependency_injector import providers

from app.core.container import container
from app.observability.model_usage import model_usage_metrics as module_level_metrics
from app.repositories.model_usage import ModelUsageRepository
from app.usage.recorder import ModelUsageRecorder


def test_model_usage_repository_is_a_factory():
    assert isinstance(container.model_usage_repository, providers.Factory)


def test_model_usage_metrics_returns_the_same_instance_on_repeated_resolution():
    first = container.model_usage_metrics()
    second = container.model_usage_metrics()
    assert first is second


def test_model_usage_metrics_is_the_module_level_singleton():
    assert container.model_usage_metrics() is module_level_metrics


def test_model_usage_recorder_returns_the_same_instance_on_repeated_resolution():
    first = container.model_usage_recorder()
    second = container.model_usage_recorder()
    assert first is second


def test_model_usage_recorder_is_wired_to_a_model_usage_repository():
    recorder = container.model_usage_recorder()
    assert isinstance(recorder, ModelUsageRecorder)
    assert isinstance(recorder._repository, ModelUsageRepository)
