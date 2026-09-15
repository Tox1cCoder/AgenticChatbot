"""Automatic function calling must be refused on every direct-SDK path.

``google-genai`` enables AFC by default and will execute any Python callable in
``tools`` itself, bypassing this product's own tool loop -- authorization,
approval, mutation receipts, artifacts and the execution budget all live there.
It also prints a notice on every ``generate_content``/``generate_content_stream``
call while enabled, which is the console noise that keeps being reported.

``build_gemini_generate_config`` set ``disable=True``, but only *after* an early
``return None`` for an empty config. A call with thinking off, code execution
off and no extras therefore returned ``None`` and got the SDK default -- and
those are exactly the tool-free calls the notice fires on.
"""

from __future__ import annotations

import pytest

from app.ai.agent_config import build_gemini_generate_config


def _afc_disabled(config) -> bool:
    assert config is not None, "a config that says nothing leaves the SDK default in place"
    afc = config.automatic_function_calling
    assert afc is not None, "automatic_function_calling was never set"
    return bool(afc.disable)


@pytest.mark.parametrize(
    ("include_thinking", "enable_code_execution"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_afc_is_refused_for_every_combination(include_thinking, enable_code_execution):
    config = build_gemini_generate_config(
        "gemini-3-flash-preview",
        include_thinking=include_thinking,
        enable_code_execution=enable_code_execution,
    )

    assert _afc_disabled(config)


def test_the_bare_config_is_the_one_that_used_to_slip_through():
    """No thinking, no code execution, no extras -- previously ``None``."""
    config = build_gemini_generate_config(
        "gemini-3-flash-preview", include_thinking=False, enable_code_execution=False
    )

    assert _afc_disabled(config)


def test_extras_still_reach_the_config():
    config = build_gemini_generate_config(
        "gemini-3-flash-preview",
        include_thinking=False,
        enable_code_execution=False,
        system_instruction="be helpful",
    )

    assert config.system_instruction == "be helpful"
    assert _afc_disabled(config)


def test_an_explicit_afc_choice_is_not_overridden():
    """The refusal is a default, not a lock."""
    from google.genai import types

    config = build_gemini_generate_config(
        "gemini-3-flash-preview",
        include_thinking=False,
        enable_code_execution=False,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False),
    )

    assert config.automatic_function_calling.disable is False


def test_the_image_generation_path_also_refuses_afc():
    """The other direct-SDK caller, which binds no tools at all."""
    import inspect

    from app.ai.image_generation import gemini

    source = inspect.getsource(gemini)
    assert "AutomaticFunctionCallingConfig" in source
