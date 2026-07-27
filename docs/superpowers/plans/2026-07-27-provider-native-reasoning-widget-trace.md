# Provider-Native Reasoning, Widget State, and Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model-aware provider-native reasoning controls, preserve Gemini final thought summaries, and make widget state a native object so large dynamic widgets succeed on the first tool call.

**Architecture:** A focused `reasoning_controls` module owns documented model capability resolution and validation. Provider catalogs expose that descriptor, every runtime/configuration path uses the same validator, and persisted primary-agent configuration stores the exact native value or `null` for provider default. Thinking extraction and widget object state are repaired at shared backend boundaries so Streamlit and AI SDK clients inherit the same behavior.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLAlchemy/Alembic, LangChain Google/OpenAI integrations, FastMCP, Streamlit, pytest.

---

## File map

- Create `app/ai/reasoning_controls.py`: immutable descriptor, official capability registry, resolution, serialization, and exact validation.
- Create `tests/test_reasoning_controls.py`: model-family matrices, aliases, unknown models, and validation behavior.
- Modify `app/services/provider_service.py`: enrich every normalized model entry with the shared descriptor.
- Modify `app/api/providers.py` and `app/api/model_config.py`: expose the same typed descriptor from both public catalog endpoints.
- Create `tests/test_provider_reasoning_metadata.py`: provider normalization and endpoint-schema contract tests.
- Modify `app/models/agent_model_config.py`, `app/repositories/agent_model_config.py`, and create `app/alembic/versions/e8f9a0b1c2d3_add_agent_reasoning_effort.py`: persist nullable native reasoning values.
- Modify `app/services/model_config_service.py`: validate, persist, read, and revalidate native values; omit incompatible values after fallback.
- Create `tests/test_model_config_reasoning.py`: persistence, patch validation, read repair, request override, and fallback behavior.
- Modify `app/schemas/custom_agent.py`, `app/services/custom_agent_service.py`, `app/ai/planning_subagents.py`, and `app/ai/agents/planning_agent.py`: route all other reasoning-bearing contracts through the shared capability policy and update compact planning guidance.
- Modify `app/ai/agents/base_agent.py` and `tests/test_runtime_model_overrides.py`: remove lossy mappings and pass validated native values unchanged.
- Modify `demo.py` and create `tests/test_demo_model_reasoning_controls.py`: dynamic per-model provider-native selector and payload persistence.
- Modify `app/ai/utils.py`, `app/ai/agents/base_agent.py`, and create `tests/test_thinking_summary_extraction.py`: extract final Gemini thinking blocks without exposing signatures.
- Modify `app/ai/mcp_servers/widgets_server.py`, `app/ai/prompts.py`, and `tests/test_widget_runtime.py`: advertise native object state, retain internal legacy-string tolerance, and shorten guidance.

### Task 1: Shared provider-native capability resolver

**Files:**
- Create: `app/ai/reasoning_controls.py`
- Create: `tests/test_reasoning_controls.py`

- [ ] **Step 1: Write failing capability and validation tests**

```python
from app.ai.reasoning_controls import resolve_reasoning_control, validate_reasoning_effort


def test_gemini_36_flash_uses_native_levels():
    control = resolve_reasoning_control("gemini", "gemini-3.6-flash", supports_reasoning=True)
    assert control.parameter_name == "thinking_level"
    assert control.levels == ("minimal", "low", "medium", "high")
    assert control.default_level == "medium"


def test_gemini_31_pro_does_not_offer_minimal():
    control = resolve_reasoning_control("gemini", "gemini-3.1-pro-preview")
    assert control.levels == ("low", "medium", "high")


def test_openai_56_retains_max():
    control = resolve_reasoning_control("openai", "gpt-5.6-sol")
    assert control.parameter_name == "reasoning.effort"
    assert control.levels == ("none", "low", "medium", "high", "xhigh", "max")


def test_openai_pro_rule_precedes_base_family():
    control = resolve_reasoning_control("openai", "gpt-5.4-pro")
    assert control.levels == ("medium", "high", "xhigh")


def test_unknown_reasoning_model_is_provider_default_only():
    control = resolve_reasoning_control("openai", "gpt-future", supports_reasoning=True)
    assert control.supported is True
    assert control.parameter_name is None
    assert control.levels == ()
    assert validate_reasoning_effort("openai", "gpt-future", None, supports_reasoning=True) is None


def test_invalid_native_value_is_not_remapped():
    try:
        validate_reasoning_effort("gemini", "gemini-3.1-pro-preview", "minimal")
    except ValueError as exc:
        assert "low, medium, high" in str(exc)
    else:
        raise AssertionError("expected model-specific validation failure")
```

- [ ] **Step 2: Run the new tests and confirm the missing-module failure**

Run: `python -m pytest tests/test_reasoning_controls.py -q`

Expected: collection fails with `ModuleNotFoundError: app.ai.reasoning_controls`.

- [ ] **Step 3: Implement the immutable descriptor and ordered official registry**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReasoningControl:
    supported: bool
    parameter_name: str | None
    display_label: str
    levels: tuple[str, ...] = ()
    default_level: str | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["levels"] = list(self.levels)
        return value


_GEMINI_RULES = (
    (("gemini-3.6-flash",), ("minimal", "low", "medium", "high"), "medium"),
    (("gemini-3.5-flash-lite",), ("minimal", "low", "medium", "high"), "minimal"),
    (("gemini-3.5-flash",), ("minimal", "low", "medium", "high"), "medium"),
    (("gemini-3.1-pro-preview",), ("low", "medium", "high"), "high"),
    (("gemini-3.1-flash-lite-image",), ("minimal", "high"), "minimal"),
    (("gemini-3-flash-preview",), ("minimal", "low", "medium", "high"), "high"),
    (("gemini-3-pro-preview",), ("low", "high"), "high"),
)

_OPENAI_RULES = (
    (("gpt-5.4-pro", "gpt-5.2-pro"), ("medium", "high", "xhigh"), "medium"),
    (("gpt-5-pro",), ("high",), "high"),
    (("gpt-5.6",), ("none", "low", "medium", "high", "xhigh", "max"), "medium"),
    (("gpt-5.5",), ("none", "low", "medium", "high", "xhigh"), "medium"),
    (("gpt-5.4", "gpt-5.2"), ("none", "low", "medium", "high", "xhigh"), "none"),
    (("gpt-5.1",), ("none", "low", "medium", "high"), "none"),
    (("gpt-5",), ("minimal", "low", "medium", "high"), None),
)


def _match(model: str, prefixes: tuple[str, ...]) -> bool:
    return any(model == prefix or model.startswith(f"{prefix}-") for prefix in prefixes)


def resolve_reasoning_control(
    provider: str, model: str, *, supports_reasoning: bool | None = None
) -> ReasoningControl:
    provider_key = str(provider or "").strip().lower()
    model_key = str(model or "").strip().lower()
    rules = _GEMINI_RULES if provider_key == "gemini" else _OPENAI_RULES if provider_key == "openai" else ()
    label = "Thinking level" if provider_key == "gemini" else "Reasoning effort"
    parameter = "thinking_level" if provider_key == "gemini" else "reasoning.effort"
    for prefixes, levels, default in rules:
        if _match(model_key, prefixes):
            return ReasoningControl(True, parameter, label, levels, default, "official_registry")
    return ReasoningControl(bool(supports_reasoning), None, label, source="provider_api" if supports_reasoning else "unknown")


def validate_reasoning_effort(
    provider: str,
    model: str,
    value: object,
    *,
    supports_reasoning: bool | None = None,
) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise ValueError("Reasoning control must be a string or null for Provider default.")
    native = value.strip().lower()
    control = resolve_reasoning_control(provider, model, supports_reasoning=supports_reasoning)
    if native not in control.levels:
        accepted = ", ".join(control.levels) if control.levels else "Provider default only"
        raise ValueError(f"{control.display_label} '{native}' is unsupported for {provider}:{model}. Accepted: {accepted}.")
    return native
```

- [ ] **Step 4: Run resolver tests**

Run: `python -m pytest tests/test_reasoning_controls.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the shared resolver**

```powershell
git add app/ai/reasoning_controls.py tests/test_reasoning_controls.py
git commit -m "feat: resolve provider-native reasoning controls"
```

### Task 2: Enrich both provider catalog endpoints

**Files:**
- Modify: `app/services/provider_service.py:636-707`
- Modify: `app/api/providers.py:60-74`
- Modify: `app/api/model_config.py:62-75`
- Create: `tests/test_provider_reasoning_metadata.py`

- [ ] **Step 1: Add failing provider normalization and schema tests**

```python
from unittest.mock import MagicMock

from app.api.model_config import ProviderModelOption as ConfigModelOption
from app.api.providers import ProviderModelOption as ProviderModelOption
from app.services.provider_service import ProviderService


def test_catalog_entries_share_reasoning_descriptor():
    service = ProviderService(MagicMock())
    entry = service._normalize_openai_model("gpt-5.6-sol")
    assert entry["reasoning_control"]["levels"][-1] == "max"
    assert ConfigModelOption.model_validate(entry).reasoning_control.levels[-1] == "max"
    assert ProviderModelOption.model_validate(entry).reasoning_control.levels[-1] == "max"


def test_gemini_api_thinking_boolean_is_preserved_for_unknown_model():
    service = ProviderService(MagicMock())
    entry = service._normalize_gemini_model(
        "gemini-future", "Future", ["generateContent"], True
    )
    assert entry["supports_reasoning"] is True
    assert entry["reasoning_control"]["supported"] is True
    assert entry["reasoning_control"]["levels"] == []


def test_cached_legacy_catalog_is_enriched_without_provider_resync():
    service = ProviderService(MagicMock())
    catalog = service._normalize_catalog_metadata({"catalog": {"models": [{"id": "gpt-5.6-sol", "provider_type": "openai", "supports_reasoning": True}]}})
    assert catalog["models"][0]["reasoning_control"]["levels"][-1] == "max"
```

- [ ] **Step 2: Run the tests and confirm descriptor fields are missing**

Run: `python -m pytest tests/test_provider_reasoning_metadata.py -q`

Expected: failures mention missing `reasoning_control`.

- [ ] **Step 3: Add the descriptor to catalog normalization and both API schemas**

Import `resolve_reasoning_control` in `provider_service.py`, compute it after `supports_reasoning`, and add:

```python
"reasoning_control": resolve_reasoning_control(
    "openai", model_id, supports_reasoning=supports_reasoning
).to_dict(),
```

Use the same code with `"gemini"` in `_normalize_gemini_model`.

Add `_enrich_model_reasoning_control(model)` and map it across models in
`_normalize_catalog_metadata`. This upgrades cached legacy entries in memory on
every status read without waiting for the six-hour provider resync TTL.

Define the same nested schema in each API module:

```python
class ReasoningControlOption(CamelModel):
    supported: bool = False
    parameter_name: str | None = None
    display_label: str
    levels: list[str] = Field(default_factory=list)
    default_level: str | None = None
    source: str = "unknown"


class ProviderModelOption(CamelModel):
    # existing fields remain unchanged
    reasoning_control: ReasoningControlOption = Field(
        default_factory=lambda: ReasoningControlOption(display_label="Reasoning")
    )
```

- [ ] **Step 4: Run catalog tests**

Run: `python -m pytest tests/test_provider_reasoning_metadata.py tests/test_provider_model_context_metadata.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit catalog enrichment**

```powershell
git add app/services/provider_service.py app/api/providers.py app/api/model_config.py tests/test_provider_reasoning_metadata.py
git commit -m "feat: expose model reasoning capabilities"
```

### Task 3: Persist and validate primary-agent reasoning selections

**Files:**
- Create: `app/alembic/versions/e8f9a0b1c2d3_add_agent_reasoning_effort.py`
- Modify: `app/models/agent_model_config.py:32-38`
- Modify: `app/repositories/agent_model_config.py:44-91`
- Modify: `app/api/model_config.py:31-48,89-97`
- Modify: `app/services/model_config_service.py:67-73,215-434,563-784,809-857`
- Create: `tests/test_model_config_reasoning.py`

- [ ] **Step 1: Add failing service tests for save, read, and model-specific rejection**

```python
import pytest


@pytest.mark.asyncio
async def test_patch_persists_native_reasoning(service, repository):
    result = await service.patch_configs(
        USER_ID,
        {"chat": {"provider": "gemini", "model": "gemini-3.6-flash", "reasoning_effort": "medium"}},
    )
    assert repository.upserts[-1]["reasoning_effort"] == "medium"
    assert result["chat"]["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_patch_rejects_level_not_supported_by_model(service):
    with pytest.raises(ValueError, match="Accepted: low, medium, high"):
        await service.patch_configs(
            USER_ID,
            {"chat": {"provider": "gemini", "model": "gemini-3.1-pro-preview", "reasoning_effort": "minimal"}},
        )


@pytest.mark.asyncio
async def test_patch_null_clears_saved_effort(service, repository):
    repository.rows = [row(reasoning_effort="high", provider_type="gemini", model="gemini-3.6-flash")]
    result = await service.patch_configs(USER_ID, {"chat": {"reasoning_effort": None}})
    assert repository.upserts[-1]["reasoning_effort"] is None
    assert result["chat"]["reasoning_effort"] is None


def test_saved_incompatible_value_is_omitted_with_warning(service, repository):
    repository.rows = [row(reasoning_effort="max", provider_type="openai", model="gpt-5.4")]
    config = service.get_effective_model_config(USER_ID)["chat"]
    assert config["reasoning_effort"] is None
    assert any("unsupported" in warning.lower() for warning in config["warnings"])
```

- [ ] **Step 2: Run the tests and confirm persistence/validation failures**

Run: `python -m pytest tests/test_model_config_reasoning.py -q`

Expected: failures show the repository and effective config lack `reasoning_effort`.

- [ ] **Step 3: Add the nullable database column and repository plumbing**

Migration body:

```python
revision = "e8f9a0b1c2d3"
down_revision = "d7e8f9a0b1c2"


def upgrade() -> None:
    op.add_column("agent_model_configs", sa.Column("reasoning_effort", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_model_configs", "reasoning_effort")
```

Add `reasoning_effort = Column(Text, nullable=True)` to `AgentModelConfig`. Add a nullable keyword to repository `upsert`, assign it on update, and pass it into new entities.

- [ ] **Step 4: Add API fields and service validation**

Add `reasoning_effort: str | None = None` to `AgentModelConfigPatch` and `AgentModelConfigSnapshot`. Replace `_normalize_reasoning_effort` with the shared `validate_reasoning_effort` call after provider/model resolution. Include `reasoning_effort` in defaults, effective configuration, validated patch output, repository upsert, and runtime config.

Change `patch_model_config` to serialize each patch with
`model_dump(exclude_unset=True)` rather than `exclude_none=True`. An explicitly
sent JSON `null` must reach the service so `Provider default` can clear a saved
value, while an omitted field must retain the previous selection.

When reading an existing row, use:

```python
try:
    native_effort = validate_reasoning_effort(
        provider,
        resolved_model,
        getattr(row, "reasoning_effort", None),
        supports_reasoning=bool(model_metadata and model_metadata.get("supports_reasoning")),
    )
except ValueError as exc:
    native_effort = None
    effective[agent_key]["warnings"].append(str(exc) + " Using Provider default.")
```

- [ ] **Step 5: Run service and migration tests**

Run: `python -m pytest tests/test_model_config_reasoning.py tests/test_runtime_model_overrides.py -q`

Run: `python -m alembic heads`

Expected: tests pass and the only head is `e8f9a0b1c2d3`.

- [ ] **Step 6: Commit persistence**

```powershell
git add app/alembic/versions/e8f9a0b1c2d3_add_agent_reasoning_effort.py app/models/agent_model_config.py app/repositories/agent_model_config.py app/api/model_config.py app/services/model_config_service.py tests/test_model_config_reasoning.py
git commit -m "feat: persist agent reasoning selections"
```

### Task 4: Use exact native values in every runtime path

**Files:**
- Modify: `app/ai/agents/base_agent.py:93-142,755-815`
- Modify: `tests/test_runtime_model_overrides.py:53-216`
- Modify: `app/schemas/custom_agent.py:137-170`
- Modify: `app/services/custom_agent_service.py:177-244,430-450`
- Modify: `tests/test_custom_agents_service.py`
- Modify: `app/ai/planning_subagents.py:54-99`
- Modify: `app/ai/agents/planning_agent.py:270-299`
- Modify: `tests/test_planning_subagents.py`

- [ ] **Step 1: Replace mapping expectations with exact-value and rejection tests**

```python
def test_openai_max_reaches_model_factory_unchanged(monkeypatch):
    captured = install_fake_model_factory(monkeypatch)
    BaseAgent._create_langchain_model_from_runtime(_FakeAgent(), _runtime_config(model="gpt-5.6-sol", reasoning_effort="max"))
    assert captured["reasoning"] == {"effort": "max"}


def test_gemini_medium_reaches_model_factory_unchanged(monkeypatch):
    captured = install_fake_gemini_factory(monkeypatch)
    BaseAgent._create_langchain_model_from_runtime(
        _FakeAgent(model_name="gemini-3.6-flash"),
        _runtime_config(provider="gemini", model="gemini-3.6-flash", reasoning_effort="medium", api_key="key"),
    )
    assert captured["thinking_level_override"] == "medium"
```

Add custom-agent tests asserting `minimal` is rejected for Gemini 3.1 Pro and `medium` is accepted. Replace the planning Pydantic fixed-enum test with model-aware validation tests executed when the override is resolved with its provider/model.

- [ ] **Step 2: Run focused tests and confirm old remaps/fixed enums fail**

Run: `python -m pytest tests/test_runtime_model_overrides.py tests/test_custom_agents_service.py tests/test_planning_subagents.py -q`

Expected: new exact-value and model-aware tests fail against the old generic mapping.

- [ ] **Step 3: Remove lossy mapping helpers from BaseAgent**

Delete `_REASONING_EFFORT_LEVELS`, `_normalize_reasoning_effort`, `_gemini_thinking_level_from_effort`, and `_openai_effort_from_reasoning_effort`. Runtime resolution has already validated the native value, so construct providers directly:

```python
native_effort = getattr(runtime_config, "reasoning_effort", None)
if runtime_config.provider == "gemini" and native_effort:
    thinking_level_override = native_effort

if runtime_config.provider == "openai" and native_effort:
    openai_kwargs["reasoning"] = {"effort": native_effort}
```

- [ ] **Step 4: Centralize custom-agent and planning validation**

Extend `ModelConfigService.validate_provider_model` to accept `reasoning_effort` and return the validated native value. `CustomAgentService` calls it on create and whenever provider, model, or reasoning effort changes, storing the returned value.

Change `SubagentModelOverride.reasoning_effort` to `str | None` with only whitespace normalization in Pydantic. Validate it in the existing planning runtime resolution where the final provider/model and catalog metadata are available. Update the planning prompt to say only:

```text
`reasoning_effort` is optional and separate from `model`. Use only a native value advertised for that exact model; otherwise omit it for Provider default.
```

- [ ] **Step 5: Revalidate effort after provider fallback**

After the final runtime provider/model is selected, call `validate_reasoning_effort`. If it fails because fallback changed compatibility, append the exception text plus `Using Provider default.` and set the value to `None`. Do not catch invalid explicit values before fallback selection; invalid values for the requested model must remain a 400-level validation error.

- [ ] **Step 6: Run cross-path tests**

Run: `python -m pytest tests/test_runtime_model_overrides.py tests/test_custom_agents_service.py tests/test_planning_subagents.py tests/test_graph_planning_subagents.py -q`

Expected: all tests pass with no mapping assertions remaining.

- [ ] **Step 7: Commit runtime consumers**

```powershell
git add app/ai/agents/base_agent.py app/schemas/custom_agent.py app/services/custom_agent_service.py app/ai/planning_subagents.py app/ai/agents/planning_agent.py tests/test_runtime_model_overrides.py tests/test_custom_agents_service.py tests/test_planning_subagents.py tests/test_graph_planning_subagents.py
git commit -m "fix: enforce native reasoning values end to end"
```

### Task 5: Add the dynamic Models-page selector

**Files:**
- Modify: `demo.py:3786-3826,11092-11233`
- Create: `tests/test_demo_model_reasoning_controls.py`

- [ ] **Step 1: Add failing pure-helper tests**

```python
def test_reasoning_options_use_selected_catalog_model(demo_module):
    provider = {"models": [{"id": "gemini-3.6-flash", "reasoningControl": {"displayLabel": "Thinking level", "levels": ["minimal", "low", "medium", "high"]}}]}
    label, options = demo_module._model_reasoning_options(provider, "gemini-3.6-flash")
    assert label == "Thinking level"
    assert options == [("Provider default", None), ("minimal", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high")]


def test_unknown_model_only_offers_provider_default(demo_module):
    assert demo_module._model_reasoning_options({"models": []}, "custom-model")[1] == [("Provider default", None)]


def test_reasoning_payload_uses_native_value_or_null(demo_module):
    assert demo_module._reasoning_option_value(("high", "high")) == "high"
    assert demo_module._reasoning_option_value(("Provider default", None)) is None
```

- [ ] **Step 2: Run the UI helper tests and confirm the helper is missing**

Run: `python -m pytest tests/test_demo_model_reasoning_controls.py -q`

Expected: failures mention `_model_reasoning_options`.

- [ ] **Step 3: Implement a pure catalog helper and form-state sync**

```python
def _model_reasoning_options(provider_snapshot: dict[str, Any], model_id: str) -> tuple[str, list[tuple[str, str | None]]]:
    label = "Reasoning"
    levels: list[str] = []
    for model in _provider_models(provider_snapshot):
        if str(model.get("id") or "").strip() != model_id:
            continue
        control = model.get("reasoningControl") or model.get("reasoning_control") or {}
        label = str(control.get("displayLabel") or control.get("display_label") or label)
        levels = [str(value) for value in control.get("levels") or [] if str(value).strip()]
        break
    return label, [("Provider default", None), *((value, value) for value in levels)]
```

Sync `cfg.reasoningEffort`/`cfg.reasoning_effort` into
`model_cfg_reasoning_<agent>`. Move each catalog model selector outside the save
form, alongside the existing provider selector, so changing a model triggers a
Streamlit rerun before reasoning options are computed. Render the reasoning
selectbox next to temperature inside the form using the selected model's exact
options. Use the tuple label for display and persist its native value or `None`
in every patch payload.

- [ ] **Step 4: Run UI tests**

Run: `python -m pytest tests/test_demo_model_reasoning_controls.py tests/test_demo_usage_dashboard.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the Models-page control**

```powershell
git add demo.py tests/test_demo_model_reasoning_controls.py
git commit -m "feat: configure native reasoning per agent"
```

### Task 6: Preserve final Gemini thought summaries

**Files:**
- Modify: `app/ai/utils.py:154-204`
- Modify: `app/ai/agents/base_agent.py:1542-1553`
- Create: `tests/test_thinking_summary_extraction.py`

- [ ] **Step 1: Add failing extraction tests**

```python
from app.ai.utils import extract_public_thinking_summary


def test_extracts_gemini_thinking_block():
    content = [{"type": "thinking", "thinking": "Checking the constraints"}, {"type": "text", "text": "Answer"}]
    assert extract_public_thinking_summary(content) == "Checking the constraints"


def test_ignores_signature_only_thinking_block():
    assert extract_public_thinking_summary([{"type": "thinking", "signature": "encrypted"}]) is None


def test_extracts_openai_summary_without_answer_text():
    content = [{"type": "reasoning", "summary": [{"type": "text", "text": "Compared options"}]}, {"type": "text", "text": "Answer"}]
    assert extract_public_thinking_summary(content) == "Compared options"
```

- [ ] **Step 2: Run extraction tests and confirm the helper is missing**

Run: `python -m pytest tests/test_thinking_summary_extraction.py -q`

Expected: import fails for `extract_public_thinking_summary`.

- [ ] **Step 3: Implement one public-summary extractor and use it in BaseAgent**

Implement a recursive text collector limited to `thinking`, `reasoning`, `summary`, and `text` fields inside blocks whose type is `thinking` or `reasoning`. Explicitly ignore `signature`, `thought_signature`, and ordinary `text` blocks. Keep `extract_openai_reasoning_summary` as a compatibility wrapper over the new helper filtered to reasoning blocks.

In `invoke_model_with_history`, set:

```python
thinking = getattr(response, "thinking", None) or extract_public_thinking_summary(response.content, block_types={"thinking"})
reasoning_summary = extract_public_thinking_summary(response.content, block_types={"reasoning"})
```

- [ ] **Step 4: Run extraction and stream/history parity tests**

Run: `python -m pytest tests/test_thinking_summary_extraction.py tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/test_message_history_pipeline.py -q`

Expected: all tests pass; no-summary content returns `None`.

- [ ] **Step 5: Commit trace extraction**

```powershell
git add app/ai/utils.py app/ai/agents/base_agent.py tests/test_thinking_summary_extraction.py
git commit -m "fix: persist final provider thought summaries"
```

### Task 7: Make widget state a native object and trim prompt guidance

**Files:**
- Modify: `app/ai/mcp_servers/widgets_server.py:11-166`
- Modify: `app/ai/prompts.py:81-89`
- Modify: `tests/test_widget_runtime.py:909-1075`
- Modify: `tests/test_widget_docs_html_only.py`

- [ ] **Step 1: Add failing schema and large-state tests**

```python
def test_widget_create_schema_uses_object_state():
    parameter = inspect.signature(widgets_server.widget_create).parameters["initial_state"]
    assert parameter.annotation == dict[str, Any]


@pytest.mark.asyncio
async def test_widget_create_accepts_large_quote_heavy_object(monkeypatch):
    html = "<!doctype html><script>const state = " + json.dumps({"label": 'He said "hello"', "rows": list(range(500))}) + ";</script>"
    result = await widgets_server.widget_create("session", {"html": html, "height": 620})
    assert json.loads(result)["state"]["html"] == html
```

Retain one direct-call test passing a legacy JSON string to `_coerce_widget_state`, not to the public tool annotation.

- [ ] **Step 2: Run widget tests and confirm the public annotation is still `str`**

Run: `python -m pytest tests/test_widget_runtime.py tests/test_widget_docs_html_only.py -q`

Expected: schema/object-state tests fail.

- [ ] **Step 3: Implement object-first state coercion**

```python
def _coerce_widget_state(raw: Any, *, field: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        parsed = _parse_legacy_widget_state(raw, field=field)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{field} must be an object.")


async def widget_create(session_id: str, initial_state: dict[str, Any], title: str = "") -> str:
    state = _coerce_widget_state(initial_state, field="initial_state")
    validate_html_widget_state(state)
    # existing store.create and serialization stay unchanged


async def widget_update(widget_id: str, state: dict[str, Any], version: int = 0) -> str:
    new_state = _coerce_widget_state(state, field="state")
    # existing lookup, validation, update, and serialization stay unchanged
```

Replace the two JSON-string prompt bullets with one line:

```text
- Pass widget `initial_state` / `state` as one native object containing self-contained `html` and numeric `height`; do not wrap it in Markdown or extra prose
```

Keep the creative interaction guidance but remove the long escaped-JSON repair wording from the tool docstring.

- [ ] **Step 4: Run all widget contract tests**

Run: `python -m pytest tests/test_widget_runtime.py tests/test_widget_contract.py tests/test_widget_docs_html_only.py tests/test_widgets_api.py -q`

Expected: all tests pass, including legacy artifact parsing in the widgets API.

- [ ] **Step 5: Commit widget object state**

```powershell
git add app/ai/mcp_servers/widgets_server.py app/ai/prompts.py tests/test_widget_runtime.py tests/test_widget_docs_html_only.py
git commit -m "fix: accept native widget state objects"
```

### Task 8: Integrated verification and final commit

**Files:**
- Modify only files required by failures directly caused by Tasks 1-7.

- [ ] **Step 1: Run formatting and static checks on changed Python files**

Run: `python -m ruff check app/ai/reasoning_controls.py app/services/provider_service.py app/services/model_config_service.py app/ai/agents/base_agent.py app/ai/mcp_servers/widgets_server.py app/ai/utils.py app/services/custom_agent_service.py app/ai/planning_subagents.py demo.py tests/test_reasoning_controls.py tests/test_provider_reasoning_metadata.py tests/test_model_config_reasoning.py tests/test_demo_model_reasoning_controls.py tests/test_thinking_summary_extraction.py`

Expected: exit code 0.

- [ ] **Step 2: Run the focused regression suite**

Run: `python -m pytest tests/test_reasoning_controls.py tests/test_provider_reasoning_metadata.py tests/test_provider_model_context_metadata.py tests/test_model_config_reasoning.py tests/test_runtime_model_overrides.py tests/test_custom_agents_service.py tests/test_planning_subagents.py tests/test_graph_planning_subagents.py tests/test_demo_model_reasoning_controls.py tests/test_thinking_summary_extraction.py tests/test_internal_sse_stream_contract.py tests/test_ai_sdk_v6_stream_contract.py tests/test_message_history_pipeline.py tests/test_widget_runtime.py tests/test_widget_contract.py tests/test_widget_docs_html_only.py tests/test_widgets_api.py -q`

Expected: all selected tests pass.

- [ ] **Step 3: Validate the migration graph**

Run: `python -m alembic heads`

Expected: exactly `e8f9a0b1c2d3 (head)`.

Run: `python -m alembic upgrade head`

Expected: migration succeeds or reports the database is already at head.

- [ ] **Step 4: Run the broader test suite and classify unrelated failures**

Run: `python -m pytest -q`

Expected: all tests pass. If a pre-existing unrelated test fails, rerun it from commit `d0e8de9` or compare its failure to the recorded baseline before classifying it as unrelated; do not hide new failures.

- [ ] **Step 5: Inspect the final diff and commit any verification-only adjustment**

Run: `git diff --check`

Run: `git status --short`

Expected: only intended application, migration, test, and plan files are changed.

If verification required a directly related adjustment, stage only the exact
files shown by `git status --short`, inspect `git diff --cached`, and commit them
with `git commit -m "test: complete reasoning and widget regressions"`. If no
adjustment was required, do not create an empty verification commit.

- [ ] **Step 6: Record the final commit range**

Run: `git log --oneline d0e8de9..HEAD`

Expected: documentation plus focused feature/fix commits for capability resolution, catalog exposure, persistence, runtime validation, UI, trace extraction, and widget state.
