# Markdown Currency Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render unescaped dollar-price ranges and lists correctly in Streamlit without changing stored/AI SDK content or breaking protected Markdown and LaTeX.

**Architecture:** A pure helper in `app.ui.stream_markdown` will split Markdown into protected and renderable spans, then escape only recognised, unambiguous dollar-price runs in the renderable spans. `demo.py` applies it immediately before assistant/user message Markdown rendering, including live rich-response slots. A single suffix is appended when final system prompts are assembled, including Agentic RAG's manual assembly path.

**Tech Stack:** Python 3.10, `re`, pytest, Streamlit.

## Global Constraints

- Do not mutate AI SDK payloads, message schemas, or persisted message content.
- Preserve fenced code, inline code, existing `\$` escapes, inline LaTeX, and display LaTeX byte-for-byte.
- Escape only unambiguous dollar-price ranges and comma/conjunction lists.
- Use exactly one concise currency-output instruction per final system prompt.

---

### Task 1: Add a pure Markdown currency display normalizer

**Files:**

- Modify: `app/ui/stream_markdown.py`
- Test: `tests/test_demo_stream_rendering.py`

**Interfaces:**

- Produces: `escape_markdown_currency(text: str) -> str`, a pure function usable without Streamlit.
- Produces: `normalize_stream_markdown_text(content: str) -> str`, retaining its existing entity/marker behavior and also applying `escape_markdown_currency`.

- [ ] **Step 1: Write the failing tests**

```python
def test_escape_markdown_currency_escapes_ranges_and_price_lists():
    from app.ui.stream_markdown import escape_markdown_currency

    assert escape_markdown_currency("Budget: $150–$160; sale: $5, $10, and $20.") == (
        "Budget: \$150–\$160; sale: \$5, \$10, and \$20."
    )


def test_escape_markdown_currency_keeps_code_escapes_and_latex_unchanged():
    from app.ui.stream_markdown import escape_markdown_currency

    raw = "`$150–$160`\n```text\n$150–$160\n```\n\$150\n$x^2$\n$$\nx + y\n$$"

    assert escape_markdown_currency(raw) == raw
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_stream_rendering.py -q`

Expected: FAIL during collection with `ImportError` because `escape_markdown_currency` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def escape_markdown_currency(text: str) -> str:
    """Escape recognised dollar-price runs outside Markdown protected spans."""
    if not isinstance(text, str) or not text:
        return ""
    # Split fenced code blocks, inline-code spans, escaped dollars, and LaTeX
    # into protected spans. Apply price-run replacement only between them.
```

The recognised run must require at least two dollar-price tokens joined by a range dash, comma, or English conjunction (`and`/`or`); do not transform a lone `$150` or a complete `$150 + 20$` math expression.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_stream_rendering.py -q`

Expected: PASS, including the existing entity and rich-marker tests.

- [ ] **Step 5: Commit**

```bash
git add app/ui/stream_markdown.py tests/test_demo_stream_rendering.py
git commit -m "fix: escape currency in streamed markdown"
```

### Task 2: Integrate rendering and add the compact prompt convention

**Files:**

- Modify: `demo.py:53,7424-7460,7528-7535`
- Modify: `app/ai/prompts.py`
- Modify: `app/ai/agents/base_agent.py:35,1641-1750`
- Modify: `app/ai/agents/rag_agent.py:30-40,957-970`
- Modify: `tests/test_demo_stream_rendering.py`
- Modify: `tests/test_base_agent_dynamic_handoff.py`
- Modify: `tests/test_rag_agent.py`

**Interfaces:**

- Consumes: `escape_markdown_currency(text: str) -> str` from Task 1.
- Produces: `MARKDOWN_CURRENCY_GUIDANCE`, a single suffix stating `In Markdown, write dollar prices as \$150; reserve $...$ for LaTeX.`
- Produces: all final assistant system-prompt paths containing `MARKDOWN_CURRENCY_GUIDANCE` once.

- [ ] **Step 1: Write the failing tests**

```python
def test_base_system_prompt_adds_currency_markdown_guidance_once():
    prompt = _agent()._build_system_prompt(persona=None, has_tool_context=False)

    assert prompt.count("write dollar prices as") == 1
```

Extend the existing Agentic-RAG captured-system-message test to assert the same count. Add a rendering test that stubs `demo.st.markdown`, invokes `render_message_bubble` with `$150–$160`, and verifies the received string is `\$150–\$160`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_demo_stream_rendering.py tests/test_base_agent_dynamic_handoff.py tests/test_rag_agent.py -q`

Expected: FAIL because persisted rendering still passes raw currency and no shared prompt guidance exists.

- [ ] **Step 3: Apply the rendering and prompt wiring**

```python
# demo.py
st.markdown(escape_markdown_currency(content_text))
st.markdown(escape_markdown_currency(segment.text))

# app/ai/prompts.py
MARKDOWN_CURRENCY_GUIDANCE = (
    "\n\nMarkdown: write dollar prices as \$150; reserve $...$ for LaTeX."
)

# BaseAgent and the Agentic RAG manual path
system_prompt = f"{system_prompt}{MARKDOWN_CURRENCY_GUIDANCE}"
```

Ensure the live path calls only `normalize_stream_markdown_text`, which includes Task 1's currency normalizer; do not double-escape it in `demo.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_stream_rendering.py tests/test_base_agent_dynamic_handoff.py tests/test_rag_agent.py -q`

Expected: PASS; the prompt assertions must prove exactly one suffix per path.

- [ ] **Step 5: Run focused lint and regression verification**

Run: `ruff check app/ui/stream_markdown.py app/ai/prompts.py app/ai/agents/base_agent.py app/ai/agents/rag_agent.py demo.py tests/test_demo_stream_rendering.py tests/test_base_agent_dynamic_handoff.py tests/test_rag_agent.py`

Run: `pytest tests/test_demo_stream_rendering.py tests/test_base_agent_dynamic_handoff.py tests/test_rag_agent.py tests/test_prompts_media_capability.py -q`

Expected: both commands exit 0.

- [ ] **Step 6: Commit**

```bash
git add demo.py app/ai/prompts.py app/ai/agents/base_agent.py app/ai/agents/rag_agent.py tests/test_demo_stream_rendering.py tests/test_base_agent_dynamic_handoff.py tests/test_rag_agent.py
git commit -m "fix: preserve dollar prices in markdown"
```

