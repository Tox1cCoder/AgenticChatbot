# Reasoning Selector Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate primary-agent and custom-agent Streamlit reasoning selectors with model-compatible provider-native levels for exact models and safe family aliases, while keeping unknown models default-only.

**Architecture:** Extend the backend reasoning registry, which already enriches provider catalogs and validates saved values, rather than duplicating compatibility logic in Streamlit. Both the Models page and custom-agent editor consume the enriched catalog descriptor. Add one Gemini transport helper so named Gemini 2.5 levels are converted to the documented `thinking_budget` used by the current generateContent client; Gemini 3 and OpenAI values remain unchanged.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, Streamlit, LangChain Google GenAI/OpenAI, pytest, Ruff

---

### Task 1: Expand the authoritative model compatibility registry

**Files:**
- Modify: `app/ai/reasoning_controls.py`
- Modify: `tests/test_reasoning_controls.py`
- Modify: `tests/test_provider_reasoning_metadata.py`

- [ ] **Step 1: Write failing exact-family and alias tests**

Add these cases to `tests/test_reasoning_controls.py`:

```python
import pytest


@pytest.mark.parametrize(
    ("model", "levels", "parameter"),
    [
        ("gemini-2.5-pro", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-2.5-flash", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-2.5-flash-lite", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-pro-latest", ("low", "medium", "high"), "thinking_level"),
        ("gemini-flash-latest", ("low", "medium", "high"), "thinking_budget"),
        ("gemini-flash-lite-latest", ("low", "medium", "high"), "thinking_budget"),
    ],
)
def test_gemini_compatible_families_expose_native_levels(model, levels, parameter):
    control = resolve_reasoning_control("gemini", model, supports_reasoning=True)
    assert control.levels == levels
    assert control.parameter_name == parameter


@pytest.mark.parametrize("model", ["o1", "o3", "o3-mini", "o4-mini"])
def test_openai_o_series_exposes_reasoning_effort(model):
    control = resolve_reasoning_control("openai", model, supports_reasoning=True)
    assert control.parameter_name == "reasoning.effort"
    assert control.levels == ("low", "medium", "high")


def test_unknown_reasoning_model_remains_provider_default_only():
    control = resolve_reasoning_control("gemini", "gemini-unknown-latest", supports_reasoning=True)
    assert control.supported is True
    assert control.levels == ()
    assert control.parameter_name is None
```

Add a provider catalog assertion to `tests/test_provider_reasoning_metadata.py` that `_normalize_gemini_model("gemini-pro-latest", ...)` exposes `low`, `medium`, and `high` through both API response models.

- [ ] **Step 2: Run the tests and verify the intended failures**

Run:

```powershell
& 'C:\Users\ADMIN\miniconda3\envs\agents\python.exe' -m pytest tests/test_reasoning_controls.py tests/test_provider_reasoning_metadata.py -q
```

Expected: the new Gemini 2.5, Gemini alias, and OpenAI o-series cases fail because their descriptors currently have no levels.

- [ ] **Step 3: Add model-specific registries without a universal fallback**

In `app/ai/reasoning_controls.py`, add a budget-backed Gemini rule table before the existing Gemini 3 rules:

```python
_GEMINI_BUDGET_RULES = (
    (("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"),
     ("low", "medium", "high"), None),
    (("gemini-flash-latest", "gemini-flash-lite-latest"),
     ("low", "medium", "high"), None),
)
```

Add `gemini-pro-latest` to a separate Gemini thinking-level rule with the common documented Pro-family subset `low`, `medium`, `high`. In `resolve_reasoning_control`, match `_GEMINI_BUDGET_RULES` first and return `parameter_name="thinking_budget"`; then match the existing Gemini 3 table and return `parameter_name="thinking_level"`.

Prepend OpenAI rules for `o1`, `o3`, `o3-mini`, and `o4-mini` with `low`, `medium`, and `high`. Put any Pro-only rule before its broader family so `_matches` cannot broaden it accidentally. Do not add a provider-wide fallback: unknown model IDs retain an empty level tuple.

- [ ] **Step 4: Run registry and catalog tests**

Run the Step 2 command again.

Expected: all tests pass, including the unknown-model default-only guard.

- [ ] **Step 5: Commit registry coverage**

```powershell
git add app/ai/reasoning_controls.py tests/test_reasoning_controls.py tests/test_provider_reasoning_metadata.py
git commit -m "feat: expand reasoning model compatibility"
```

### Task 2: Send Gemini 2.5 selections through the compatible transport parameter

**Files:**
- Modify: `app/ai/reasoning_controls.py`
- Modify: `app/ai/agent_config.py`
- Modify: `tests/test_reasoning_controls.py`
- Modify: `tests/test_runtime_model_overrides.py`

- [ ] **Step 1: Write failing transport and runtime tests**

Add to `tests/test_reasoning_controls.py`:

```python
from app.ai.reasoning_controls import gemini_reasoning_kwargs


def test_gemini_25_named_level_uses_documented_budget():
    assert gemini_reasoning_kwargs("gemini-2.5-flash", "medium") == {
        "thinking_budget": 8192
    }


def test_gemini_3_named_level_is_unchanged():
    assert gemini_reasoning_kwargs("gemini-3.6-flash", "high") == {
        "thinking_level": "high"
    }
```

Add to `tests/test_runtime_model_overrides.py` a Gemini 2.5 case that invokes `BaseAgent._create_langchain_model_from_runtime` with `reasoning_effort="medium"` and asserts the patched `create_langchain_model` receives the unchanged logical override. Add a focused `create_langchain_model` constructor test that patches `ChatGoogleGenerativeAI` and asserts it receives `thinking_budget=8192` and does not receive `thinking_level`.

- [ ] **Step 2: Run the new tests and confirm the helper/transport failures**

Run:

```powershell
& 'C:\Users\ADMIN\miniconda3\envs\agents\python.exe' -m pytest tests/test_reasoning_controls.py tests/test_runtime_model_overrides.py -q
```

Expected: import or assertion failures show that Gemini 2.5 explicit selections are not yet converted into a budget.

- [ ] **Step 3: Implement one validated Gemini transport helper**

Add to `app/ai/reasoning_controls.py`:

```python
_GEMINI_LEVEL_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}


def gemini_reasoning_kwargs(model: str, value: str) -> dict[str, Any]:
    native_value = validate_reasoning_effort("gemini", model, value, supports_reasoning=True)
    if native_value is None:
        return {}
    control = resolve_reasoning_control("gemini", model, supports_reasoning=True)
    if control.parameter_name == "thinking_budget":
        return {"thinking_budget": _GEMINI_LEVEL_BUDGETS[native_value]}
    if control.parameter_name == "thinking_level":
        return {"thinking_level": native_value}
    return {}
```

In `app/ai/agent_config.py`, when `thinking_level_override` is present, call `gemini_reasoning_kwargs(model_name, thinking_level_override)` and merge its single returned parameter into `model_kwargs`. Only use the existing settings-based default branch when no explicit override exists. This guarantees that an explicit Gemini 2.5 selection replaces, rather than coexists with, the default thinking budget.

- [ ] **Step 4: Run the focused runtime tests**

Run the Step 2 command again.

Expected: all tests pass and no Gemini constructor receives both thinking parameters.

- [ ] **Step 5: Commit transport support**

```powershell
git add app/ai/reasoning_controls.py app/ai/agent_config.py tests/test_reasoning_controls.py tests/test_runtime_model_overrides.py
git commit -m "fix: apply compatible Gemini reasoning controls"
```

### Task 3: Add reasoning controls to custom-agent create and edit

**Files:**
- Modify: `demo.py`
- Modify: `tests/test_demo_custom_agents.py`
- Modify: `tests/test_custom_agents_service.py`

- [ ] **Step 1: Write failing custom-agent catalog and payload tests**

Add pure helper tests to `tests/test_demo_custom_agents.py`:

```python
def test_custom_agent_reasoning_options_follow_selected_model(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    options = {
        "providers": [
            {
                "provider_type": "gemini",
                "models": [
                    {
                        "id": "gemini-3.6-flash",
                        "reasoning_control": {
                            "display_label": "Thinking level",
                            "levels": ["minimal", "low", "medium", "high"],
                        },
                    }
                ],
            }
        ]
    }
    assert demo._custom_agent_reasoning_options(
        options, "gemini", "gemini-3.6-flash"
    ) == ("Thinking level", [None, "minimal", "low", "medium", "high"])


def test_custom_agent_unknown_model_is_default_only(monkeypatch):
    demo = _import_demo_with_ui_stubs(monkeypatch)
    assert demo._custom_agent_reasoning_options(
        {"providers": []}, "gemini", "custom-model"
    ) == ("Reasoning", [None])
```

Add an AST/source contract test for `render_custom_agents_manager` asserting
that both the edit body and create body contain `"reasoning_effort"`, and that
the custom-agent creation model selector is rendered before
`st.form("create_custom_agent_form"...)` so a model change can rerun and refresh
the compatible list.

Add service tests to `tests/test_custom_agents_service.py` that create an agent
with `reasoning_effort="high"`, update it to `"low"`, and explicitly update it
to `None`. Assert the fake `validate_provider_model` receives the value and the
stored/read model preserves or clears it respectively.

- [ ] **Step 2: Run the tests and verify the UI failures**

```powershell
& 'C:\Users\ADMIN\miniconda3\envs\agents\python.exe' -m pytest tests/test_demo_custom_agents.py tests/test_custom_agents_service.py -q
```

Expected: helper/import and UI source-contract failures show that Streamlit
does not expose or submit custom-agent reasoning yet. Existing backend service
plumbing may already satisfy some persistence assertions.

- [ ] **Step 3: Preserve full provider snapshots for custom-agent controls**

Add these helpers near `_provider_model_options` in `demo.py`:

```python
def _custom_agent_provider_map(options: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("provider_type") or entry.get("providerType") or "").strip(): entry
        for entry in options.get("providers") or []
        if isinstance(entry, dict)
    }


def _custom_agent_reasoning_options(
    options: dict[str, Any], provider_type: str, model_id: str
) -> tuple[str, list[str | None]]:
    provider = _custom_agent_provider_map(options).get(provider_type, {})
    return _model_reasoning_options(provider, model_id)
```

Keep `_provider_model_options` for existing model-ID callers, but derive both
helpers from the same `providers` payload returned by `/custom-agents/options`.
Do not add model IDs or level names to `demo.py`.

- [ ] **Step 4: Add a model-aware reasoning selector to existing-agent editing**

After `edit_model` is rendered, compute the label/options with
`_custom_agent_reasoning_options(options, agent["providerType"], edit_model)`.
Initialize `ca_edit_reasoning_<id>` from
`agent["reasoningEffort"]`/`agent["reasoning_effort"]`; if the stored value is
not compatible with the currently typed model, set it to `None` before creating
the widget. Render:

```python
edit_reasoning = st.selectbox(
    reasoning_label,
    reasoning_options,
    key=f"ca_edit_reasoning_{agent['id']}",
    format_func=lambda value: "Provider default" if value is None else str(value),
    help="Values are specific to this provider model.",
)
```

Include `"reasoning_effort": edit_reasoning` in the update body. Sending an
explicit `None` is required so an existing override can be cleared.

- [ ] **Step 5: Add live model and reasoning selectors to custom-agent creation**

Keep the Provider selector outside the form. Move the catalog Model selector
outside the form as well, give it key `ca_new_model`, and normalize its session
value before rendering whenever Provider changes. Immediately render the
reasoning selector from the chosen provider/model:

```python
reasoning_label, reasoning_options = _custom_agent_reasoning_options(
    options, provider_type, model
)
if st.session_state.get("ca_new_reasoning") not in reasoning_options:
    st.session_state.ca_new_reasoning = None
reasoning_effort = st.selectbox(
    reasoning_label,
    reasoning_options,
    key="ca_new_reasoning",
    format_func=lambda value: "Provider default" if value is None else str(value),
)
```

Leave name, description, prompt, temperature, tools, skills, and the Create
submit button inside the existing form. Include
`"reasoning_effort": reasoning_effort` in the create body.

- [ ] **Step 6: Run custom-agent tests**

Run the Step 2 command again.

Expected: all tests pass; create, edit, and clear operations use the selected
provider-native value.

- [ ] **Step 7: Commit custom-agent reasoning editing**

```powershell
git add demo.py tests/test_demo_custom_agents.py tests/test_custom_agents_service.py
git commit -m "feat: edit custom agent reasoning controls"
```

### Task 4: Verify the Streamlit contract and refresh the running services

**Files:**
- Modify: `tests/test_demo_model_reasoning_controls.py`
- Modify only if the contract test fails: `demo.py`

- [ ] **Step 1: Add an end-to-end selector descriptor contract test**

Extend `tests/test_demo_model_reasoning_controls.py` by loading the existing
Streamlit helper plus the real catalog normalizer and API response model:

```python
from unittest.mock import MagicMock

from app.api.model_config import ProviderOptionsSnapshot
from app.services.provider_service import ProviderService


service = ProviderService(provider_repository=MagicMock())
raw = service._normalize_gemini_model(
    "gemini-pro-latest", "Gemini Pro Latest", ["generateContent"], True
)
provider = ProviderOptionsSnapshot.model_validate(
    {
        "provider_type": "gemini",
        "configured": True,
        "key_source": "env",
        "sync_status": "ready",
        "models": [raw],
    }
).model_dump(by_alias=True)
assert helper(provider, "gemini-pro-latest") == (
    "Thinking level",
    [None, "low", "medium", "high"],
)
```

This spans registry enrichment, API alias serialization, and Streamlit
extraction rather than hand-authoring the levels.

- [ ] **Step 2: Run the contract test**

Run:

```powershell
& 'C:\Users\ADMIN\miniconda3\envs\agents\python.exe' -m pytest tests/test_demo_model_reasoning_controls.py tests/test_provider_reasoning_metadata.py -q
```

Expected after Tasks 1-3: PASS without adding a second compatibility table to
`demo.py`. If it fails because `_model_reasoning_options` drops the serialized
descriptor, first add a failing helper-only assertion for that exact shape,
then change only `_model_reasoning_options`; do not hardcode model names in
Streamlit.

- [ ] **Step 3: Run formatting and the complete focused suite**

```powershell
& 'C:\Users\ADMIN\miniconda3\envs\agents\python.exe' -m ruff check app/ai/reasoning_controls.py app/ai/agent_config.py demo.py tests/test_reasoning_controls.py tests/test_provider_reasoning_metadata.py tests/test_runtime_model_overrides.py tests/test_demo_model_reasoning_controls.py tests/test_demo_custom_agents.py tests/test_custom_agents_service.py
& 'C:\Users\ADMIN\miniconda3\envs\agents\python.exe' -m ruff format --check app/ai/reasoning_controls.py app/ai/agent_config.py demo.py tests/test_reasoning_controls.py tests/test_provider_reasoning_metadata.py tests/test_runtime_model_overrides.py tests/test_demo_model_reasoning_controls.py tests/test_demo_custom_agents.py tests/test_custom_agents_service.py
& 'C:\Users\ADMIN\miniconda3\envs\agents\python.exe' -m pytest tests/test_reasoning_controls.py tests/test_provider_reasoning_metadata.py tests/test_model_config_reasoning.py tests/test_runtime_model_overrides.py tests/test_demo_model_reasoning_controls.py tests/test_demo_custom_agents.py tests/test_custom_agents_service.py tests/test_demo_usage_dashboard.py -q
```

Expected: Ruff exits zero and all selected tests pass.

- [ ] **Step 4: Commit the UI contract test**

```powershell
git add tests/test_demo_model_reasoning_controls.py demo.py
git commit -m "test: cover compatible reasoning selector options"
```

- [ ] **Step 5: Restart only the two verified local app processes**

Resolve listeners for ports 8000 and 8501, confirm their command lines are respectively `python -m app.main` and `python -m streamlit run demo.py`, stop those exact process IDs, and relaunch the same commands hidden from the repository root. Do not stop unrelated Python processes.

- [ ] **Step 6: Verify service health and catalog enrichment**

Poll `http://127.0.0.1:8000/health` and `http://127.0.0.1:8501/_stcore/health` until both return HTTP 200. Instantiate `ProviderService` against the local database without printing credentials and assert that `gemini-3.6-flash`, `gemini-pro-latest`, and one OpenAI o-series model each expose more than one reasoning choice. The Streamlit user can then click `Reload snapshot` if their existing browser session still holds the previous cache.

- [ ] **Step 7: Inspect the final repository state**

Run:

```powershell
git diff --check
git status --short
git log --oneline -6
```

Expected: no uncommitted files and focused documentation, registry, transport, and contract-test commits at the branch tip.
