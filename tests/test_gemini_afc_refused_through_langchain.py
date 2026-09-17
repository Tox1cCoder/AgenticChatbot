"""AFC must also be refused on the LangChain-mediated Gemini path.

``test_gemini_afc_always_refused`` covers the calls this codebase makes into
the SDK itself. It does not cover the far busier path: every agent turn runs
through ``ChatGoogleGenerativeAI``, and ``langchain-google-genai`` never sets
``automatic_function_calling`` at all -- ``grep automatic_function_calling`` over
the installed package returns nothing. The SDK therefore applies its own
default, which is *enabled*.

That is the notice that kept being reported after the direct-SDK paths were
fixed. Its wording names ``AsyncModels.generate_content``, which is the async
client -- and the only caller of that in this stack is
``ChatGoogleGenerativeAI._agenerate``.

The oracle here is ``_extra_utils.should_disable_afc``: the SDK's own predicate,
the one it branches on to decide whether to log the notice and whether to run
its own function-calling loop. Asserting through it rather than on our field
shape means an SDK that changes how it reads the config fails this test instead
of quietly re-enabling AFC in production.
"""

from __future__ import annotations

import pytest
from google.genai import _extra_utils, types
from langchain_core.messages import HumanMessage

from app.ai.agent_config import create_langchain_model
from app.ai.model_factory import ModelFactory

MESSAGES = [HumanMessage(content="hi")]


def _afc_refused(model, **invoke_kwargs) -> bool:
    """Whether the SDK would skip AFC for what this model is about to send."""
    request = model._prepare_request(MESSAGES, **invoke_kwargs)
    return _extra_utils.should_disable_afc(request["config"])


@pytest.mark.parametrize("agent_type", ["chat", "planning", "rag", "search"])
def test_agent_models_refuse_afc(agent_type):
    model = create_langchain_model(agent_type, api_key_override="test-key")

    assert _afc_refused(model), f"{agent_type} would hand the SDK its AFC default"


def test_runtime_created_models_refuse_afc():
    """``ModelFactory`` builds Gemini models too, and on a different line."""
    model = ModelFactory._create_gemini_model(
        model="gemini-3-flash-preview", api_key="test-key", temperature=1.0
    )

    assert _afc_refused(model)


def test_refusal_survives_tool_binding():
    """Binding tools goes through ``bind``, which must not drop the refusal."""
    from app.ai.planning_tools import create_write_todos_tool

    model = create_langchain_model("planning", api_key_override="test-key")
    bound = ModelFactory.bind_tools_to_model(model, [create_write_todos_tool()])

    assert _afc_refused(bound.bound if hasattr(bound, "bound") else bound)


def test_an_explicit_afc_choice_is_not_overridden():
    """The refusal is a default, not a lock -- same rule as the direct path."""
    model = create_langchain_model("planning", api_key_override="test-key")

    assert not _afc_refused(
        model,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
    )


def test_langchain_itself_still_sets_nothing():
    """The premise. If this ever fails, the override can be deleted."""
    import inspect

    from langchain_google_genai import chat_models

    assert "automatic_function_calling" not in inspect.getsource(chat_models)
