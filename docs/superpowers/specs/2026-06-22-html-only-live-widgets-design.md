# HTML-Only Live Widgets Design

## Goal

Live widgets should produce bespoke, interactive iframe experiences instead of falling back to rigid structured tables or chart schemas. For conceptual prompts such as "giai thich dao dong dieu hoa", the expected widget is a self-contained HTML micro-app: animated visual state, parameter controls, live readouts, and a diagram or graph that helps the user understand the concept.

## Product Direction

- Live widgets have one supported type: `html`.
- Remove structured widget types: `table`, `chart`, `dashboard`, `form`, and `list`.
- Remove widget quality grading and `quality_guidance`.
- Keep minimal contract validation only: widget state must be a JSON object, HTML content must be non-empty, and iframe height must be numeric and bounded.
- Keep the widget runtime, connection, WebSocket, inline placement, and action plumbing where they still apply to HTML widgets.

## HTML Widget State

The widget state is:

```json
{
  "html": "<!doctype html>...",
  "height": 620,
  "caption": "Optional short caption"
}
```

No state-field aliases are part of the new contract. New widgets must use `html`.

## Generation Guidance

Prompt and tool guidance should steer the model toward micro-apps, not fixed schemas.

For conceptual explanations involving motion, changing variables, systems, physics, math, processes, or "show how it works", the assistant should create an HTML widget with:

- animation or manipulable visual state
- sliders or controls for key parameters
- live numeric readouts
- canvas, SVG, or DOM-based diagrams/graphs when useful
- labels and captions in the user's language
- responsive inline CSS and vanilla JavaScript
- no external dependencies, auth assumptions, or cross-window requirements

Example guidance for harmonic oscillation:

- animate the oscillator position `x(t)`
- draw a time graph of displacement
- expose sliders for amplitude, angular frequency, and phase
- include pause/reset controls
- show live values for time and displacement
- label the experience in Vietnamese when the user asks in Vietnamese

## Code Changes

- `app/ai/mcp_servers/widgets_server.py`
  - Restrict `widget_create` to exactly `widget_type="html"`.
  - Restrict `widget_update` through the existing record type.
  - Replace structured-widget docstring examples with HTML micro-app guidance.
  - Remove `quality_guidance` behavior.

- `app/services/widget_quality.py`
  - Delete this module.
  - Move minimal HTML contract validation into the widget server or a small contract helper.
  - Move action template helpers to the widget action endpoint module if actions remain supported.

- `app/ai/prompts.py`
  - Replace table/chart/dashboard/form/list widget guidance with HTML-only micro-app guidance.
  - Include the harmonic-oscillation example pattern.
  - Keep inline rich marker placement rules.

- `demo.py`
  - Remove structured live-widget render branches and support code for chart/table/dashboard/form/list widgets.
  - Keep only sandboxed iframe rendering for HTML widgets.

- Documentation and tests
  - Update README and live-widget frontend docs to describe HTML-only widgets.
  - Delete or rewrite tests that assert structured widget support.
  - Add tests that reject structured widget creation and accept a self-contained HTML simulation widget.

## Error Handling

Invalid live widgets should fail with clear, model-readable errors:

- unsupported widget type
- invalid JSON object
- missing HTML content
- non-numeric height
- height outside the accepted iframe range

These errors are contract validation, not editorial scoring.

## Testing

Targeted tests:

- prompt text mentions HTML-only widgets and micro-app expectations
- `widget_create` rejects `table`, `chart`, `dashboard`, `form`, and `list`
- `widget_create` accepts valid `html`
- `widget_create` rejects empty HTML and invalid height
- Streamlit live-widget component contains only the HTML iframe renderer path
- widget action endpoint continues to render assistant messages if actions remain supported

Broader verification:

```powershell
python -m pytest tests/test_widget_runtime.py tests/test_widgets_api.py tests/test_widget_actions_api.py tests/test_demo_meaningful_widgets.py tests/test_demo_plan_widget.py -q
```

The exact test set may change as structured-widget tests are removed or rewritten.
