import sys

import app.core.config as config_module
from app.core.config import Settings


def test_redis_password_is_accepted_and_injected_into_urls():
    settings = Settings(
        secret_key="test-secret",
        environment="development",
        redis_password="sample-chatbot-dev",
        redis_url="redis://localhost:6379/0",
        celery_broker_url="redis://localhost:6379/0",
        celery_result_backend="redis://localhost:6379/1",
    )

    host = "127.0.0.1" if sys.platform == "win32" else "localhost"
    assert settings.redis_url == f"redis://:sample-chatbot-dev@{host}:6379/0"
    assert settings.celery_broker_url == f"redis://:sample-chatbot-dev@{host}:6379/0"
    assert settings.celery_result_backend == f"redis://:sample-chatbot-dev@{host}:6379/1"


def test_existing_redis_url_password_is_not_overwritten():
    settings = Settings(
        secret_key="test-secret",
        environment="development",
        redis_password="sample-chatbot-dev",
        redis_url="redis://:already-set@localhost:6379/0",
    )

    expected = (
        "redis://:already-set@127.0.0.1:6379/0"
        if sys.platform == "win32"
        else "redis://:already-set@localhost:6379/0"
    )
    assert settings.redis_url == expected


def test_windows_localhost_redis_urls_are_normalized(monkeypatch):
    monkeypatch.setattr(config_module.sys, "platform", "win32")

    settings = Settings(
        secret_key="test-secret",
        environment="development",
        redis_password="sample-chatbot-dev",
        redis_url="redis://localhost:6379/0",
        celery_broker_url="redis://localhost:6379/0",
        celery_result_backend="redis://localhost:6379/1",
    )

    assert settings.redis_url == "redis://:sample-chatbot-dev@127.0.0.1:6379/0"
    assert settings.celery_broker_url == "redis://:sample-chatbot-dev@127.0.0.1:6379/0"
    assert settings.celery_result_backend == "redis://:sample-chatbot-dev@127.0.0.1:6379/1"


def test_planning_max_iterations_defaults_to_disabled():
    settings = Settings(secret_key="test-secret", environment="development")

    assert settings.planning_max_iterations == 0


def test_planning_max_iterations_accepts_zero_to_disable_budget():
    settings = Settings(
        secret_key="test-secret",
        environment="development",
        planning_max_iterations=0,
    )

    assert settings.planning_max_iterations == 0
