---
name: AutoCAD TNF Layout
description: |
  Draw TNF grids, foundations, columns, and beams in AutoCAD using MCP tool calls.
  This file is written as an "agentic" Claude skill: it defines tool behavior, strict
  action formats, and required agent constraints.
---

# AutoCAD TNF Layout Skill (Agentic)

## Overview

This file defines a Claude agentic skill that drives AutoCAD through the MCP framework.
The agent must plan actions using the current AutoCAD context and execute them as strict JSON.

## Agentic behavior (must be enforced)

1. Only calling `read_autocad_documents` and reading its output when you are confused and want more information on how to use tools correctly.
2. Call `get_autocad_context(shortHex)` before planning any actions.
3. Plan actions and generate strict action JSON (see "Action format" below).
4. Call `apply_autocad_actions(shortHex, actionsJson)` to execute actions.
5. If actions create or modify objects, call `get_autocad_context` again before using new handles.
6. Repeat until the user's request is fully satisfied.

> **Critical constraints (required for safe execution):**
>
> - Only return valid JSON for tool outputs (no markdown, comments, or prose).
> - Never invent or guess `shortHex` or AutoCAD handles.
> - Use handles only from `context.selected`.
> - Ensure `actionsJson` is a JSON string (not a nested object).

## Tool definitions (Claude tool spec)

### Tool: `read_autocad_documents`
- **Description:** Returns AutoCAD tool documentation and the strict JSON action format.
- **Input:** none
- **Output:** text documentation string.

### Tool: `get_autocad_context`
- **Description:** Returns the current AutoCAD drawing context for a session.
- **Input:**
  - `shortHex` (string): session identifier from the AutoCAD plugin UI.
  - `timeoutSec` (integer, optional, default=30): request timeout.
- **Output:** JSON object describing selected handles, layers, and drawing state.

### Tool: `apply_autocad_actions`
- **Description:** Executes a list of actions in AutoCAD.
- **Input:**
  - `shortHex` (string): session identifier.
  - `actionsJson` (string): valid JSON text describing actions.
  - `timeoutSec` (integer, optional, default=60): request timeout.
  - `jobId` (string, optional): client-provided identifier.
- **Output:** JSON response from AutoCAD.

---

## Action format (strict JSON)

Choose exactly one of the following valid output shapes for `actionsJson`:

### A) Single action object
```json
{ "method": "create|modify|remove|tool", "params": { ... } }
```

### B) Array of actions
```json
[
  { "method": "...", "params": { ... } },
  { "method": "...", "params": { ... } }
]
```

### C) Wrapper object
```json
{ "actions": [ { "method": "...", "params": { ... } } ] }
```

**Global rules (must be enforced):**

- Output must be valid JSON only (no markdown, no comments, no prose).
- `method` must be one of: `create`, `modify`, `remove`, `tool`.
- `params` must be an object.
- Do not output `filter`.
- Do not invent handles; only use handles from `context.selected`.
- Coordinates are WCS: `+X` is right, `+Y` is up.
- After create/modify operations that generate new handles, call `get_autocad_context` again before using those handles.

---

## Supported create geometry

Supported shapes:

- `polyline`
- `line`
- `circle`
- `text`
- `dim`
- `arrayrect` (only when explicitly requested)

Use `method: "tool"` only for TNF-specific tools.

---

## TNF tools (tool calls)

### `TNF_DRAW_GRID_SYSTEM`
- **Purpose:** Draw a TNF grid system.
- **Args (order):**
  1. x spans as a CSV string
  2. y spans as a CSV string
- **Example:**
```json
{ "method":"tool", "params": { "name":"TNF_DRAW_GRID_SYSTEM", "args":["4000,6000,4000","5000,5000"] } }
```

### `KIS`
- **Purpose:** Draw TNF foundations on selected grid intersections.
- **Args (order):**
  1. grid handles
  2. type
  3. size
- **Rules:**
  - Use `"Nf"` for type unless explicitly requested otherwise.
  - Size format is `"X,Y"` as a CSV string.
- **Example:**
```json
{ "method":"tool", "params": { "name":"KIS", "args":[["1A2B","1A2C"],"Nf","4000,4000"] } }
```

### `HSRGT`
- **Purpose:** Draw foundation columns/rectangles from selected foundations.
- **Args (order):**
  1. foundation handles
  2. x
  3. y
- **Example:**
```json
{ "method":"tool", "params": { "name":"HSRGT", "args":[["2BC1","2BC2"],300,300] } }
```

### `KI`
- **Purpose:** Apply slope/chamfer to selected DD foundations.
- **Args (order):**
  1. dd handles
  2. slope size
- **Example:**
```json
{ "method":"tool", "params": { "name":"KI", "args":[["1A2B","1A2C"],300] } }
```

---

## Layout input format

The tool `extract_layout_from_sketch` returns data like this:

```json
{
  "grids": {
    "vertical": [...],
    "horizontal": [...]
  },
  "tmp_grids": {
    "vertical": [...],
    "horizontal": [...]
  },
  "foundations": {
    "F1": { "X": ..., "Y": ... },
    "F2": { "X": ..., "Y": ... }
  },
  "layout": [
    [...],
    ...
  ]
}
```

Interpretation:

- `grids.vertical`: X positions of base grids, left to right
- `grids.horizontal`: Y positions of base grids, bottom to top
- `tmp_grids`: temporary foundation-only grids that must not remain in the final result
- `foundations`: foundation type sizes
- `layout`: matrix of foundation types at each grid intersection

---

## Drawing workflow (recommended)

1. Read documentation
2. Get context
3. Draw base TNF grids
4. Get context again
5. Place foundations grouped by foundation type
6. Get context again
7. Rename foundation texts
8. Remove temporary grids
9. Create outer beams
10. Get context again and verify completion

---

## Important execution rules

- Always inspect current context before acting.
- Never guess handles.
- Never skip `read_autocad_documents` or `get_autocad_context`.
- Never output prose when the tool expects action JSON.
- Use strict JSON only.
- Re-check progress after each major action group.
- Stop only when the user's request is complete.

---

## Common failure causes

- Not reading `read_autocad_documents` first
- Inventing handles
- Using invalid JSON
- Passing an object instead of a JSON string to `actionsJson`
- Using wrong TNF tool argument order
- Drawing beam on the wrong layer
- Forgetting to remove `tmp_grids`
- Not re-reading context after object creation
- Using temporary grids as final structural grids
