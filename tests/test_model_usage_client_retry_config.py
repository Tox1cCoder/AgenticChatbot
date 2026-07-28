"""SDK-internal provider retries are disabled at every client-construction site.

The application owns retry boundaries (the recorder executes each provider call
exactly once). To keep the ledger's "attempt" a true application-controlled
provider attempt -- not a hidden SDK re-request -- every supported client is
constructed with SDK retries off:

* LangChain ``ChatGoogleGenerativeAI`` / ``ChatOpenAI`` -> ``max_retries=0``
* raw ``openai.OpenAI`` / ``openai.AsyncOpenAI`` -> ``max_retries=0``
* ``google.genai.Client`` -> ``http_options.retry_options.attempts == 1``
  (that SDK counts ``attempts`` inclusive of the original request, so 1 = no
  retry)

Each test monkeypatches the constructor and asserts the exact setting; no real
client is built and no network/credentials are needed.
"""

from __future__ import annotations

import openai
from google import genai

from app.usage.types import UsageContext  # noqa: F401  (keeps app.usage importable in isolation)


class _Capture:
    """Records the kwargs the constructor was called with; returns a sentinel."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return object()


def _assert_genai_no_retry(kwargs: dict) -> None:
    http_options = kwargs["http_options"]
    assert http_options.retry_options.attempts == 1


# --- LangChain model factory ---------------------------------------------


def test_model_factory_gemini_disables_retries(monkeypatch):
    from app.ai import model_factory

    capture = _Capture()
    monkeypatch.setattr(model_factory, "ChatGoogleGenerativeAI", capture)
    model_factory.ModelFactory.create_model(provider="gemini", model="gemini-3-flash", api_key="k")
    assert capture.kwargs["max_retries"] == 0


def test_model_factory_openai_disables_retries(monkeypatch):
    from app.ai import model_factory

    capture = _Capture()
    monkeypatch.setattr(model_factory, "ChatOpenAI", capture)
    model_factory.ModelFactory.create_model(provider="openai", model="gpt-5", api_key="k")
    assert capture.kwargs["max_retries"] == 0


# --- agent_config --------------------------------------------------------


def test_agent_config_langchain_model_disables_retries(monkeypatch):
    from app.ai import agent_config

    capture = _Capture()
    monkeypatch.setattr(agent_config, "ReasoningNormalizedChatGoogleGenerativeAI", capture)
    agent_config.create_langchain_model("chat", api_key_override="k")
    assert capture.kwargs["max_retries"] == 0


def test_agent_config_gemini_client_disables_retries(monkeypatch):
    from app.ai import agent_config

    capture = _Capture()
    monkeypatch.setattr(agent_config.genai, "Client", capture)
    agent_config.create_gemini_client(api_key_override="k")
    _assert_genai_no_retry(capture.kwargs)


# --- router --------------------------------------------------------------


def test_router_gemini_client_disables_retries(monkeypatch):
    from app.ai.agents import router

    capture = _Capture()
    monkeypatch.setattr(router.genai, "Client", capture)
    monkeypatch.setattr(router.settings, "gemini_api_key", "k")
    router.Router()
    _assert_genai_no_retry(capture.kwargs)


# --- rag embedding service ----------------------------------------------


def test_rag_embedding_service_gemini_client_disables_retries(monkeypatch):
    from app.services import rag_embedding_service

    capture = _Capture()
    monkeypatch.setattr(rag_embedding_service.genai, "Client", capture)
    rag_embedding_service.GeminiRAGEmbeddingService(api_key="k")
    _assert_genai_no_retry(capture.kwargs)


# --- document processing service -----------------------------------------


def test_document_processing_service_gemini_client_disables_retries(monkeypatch):
    from types import SimpleNamespace

    from app.services import document_processing_service
    from app.services.document_processing_service import DocumentProcessingService

    capture = _Capture()
    monkeypatch.setattr(document_processing_service.genai, "Client", capture)
    service = DocumentProcessingService.__new__(DocumentProcessingService)
    service.settings = SimpleNamespace(gemini_api_key="k")
    service.gemini_client = None
    service._init_gemini()
    _assert_genai_no_retry(capture.kwargs)


# --- provider service ----------------------------------------------------


def test_provider_service_openai_async_client_disables_retries(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app.services.provider_service import ProviderService

    capture = _Capture()

    class _AsyncClient:
        def __init__(self, **kwargs):
            capture.kwargs = kwargs

            async def _list():
                return SimpleNamespace(data=[])

            self.models = SimpleNamespace(list=_list)

    monkeypatch.setattr(openai, "AsyncOpenAI", _AsyncClient)
    service = ProviderService.__new__(ProviderService)
    asyncio.run(service._fetch_openai_models("k"))
    assert capture.kwargs["max_retries"] == 0


def test_provider_service_openai_sync_client_disables_retries(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app.services.provider_service import ProviderService

    capture = _Capture()

    class _SyncClient:
        def __init__(self, **kwargs):
            capture.kwargs = kwargs
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    # Force the synchronous branch by removing AsyncOpenAI.
    monkeypatch.setattr(openai, "AsyncOpenAI", None)
    monkeypatch.setattr(openai, "OpenAI", _SyncClient)
    service = ProviderService.__new__(ProviderService)
    asyncio.run(service._fetch_openai_models("k"))
    assert capture.kwargs["max_retries"] == 0


def test_provider_service_gemini_client_disables_retries(monkeypatch):
    from app.services.provider_service import ProviderService

    capture = _Capture()

    class _Client:
        def __init__(self, **kwargs):
            capture.kwargs = kwargs

        class _Models:
            @staticmethod
            def list(*args, **kwargs):
                return []

        models = _Models()

    monkeypatch.setattr(genai, "Client", _Client)
    service = ProviderService.__new__(ProviderService)
    service._fetch_gemini_models_sync("k")
    _assert_genai_no_retry(capture.kwargs)


# --- OpenAI image provider -----------------------------------------------


def test_openai_image_provider_disables_retries_with_key(monkeypatch):
    from app.ai.image_generation import openai_provider

    capture = _Capture()
    monkeypatch.setattr(openai, "AsyncOpenAI", capture)
    openai_provider.OpenAIImageProvider(api_key="k")
    assert capture.kwargs["max_retries"] == 0


def test_openai_image_provider_disables_retries_without_key(monkeypatch):
    from app.ai.image_generation import openai_provider

    capture = _Capture()
    monkeypatch.setattr(openai, "AsyncOpenAI", capture)
    openai_provider.OpenAIImageProvider()
    assert capture.kwargs["max_retries"] == 0
