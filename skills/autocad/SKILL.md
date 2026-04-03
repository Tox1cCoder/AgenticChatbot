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

Use `method: "create"` for creating new layers or geometry.

General rules:
- `params` must be an object.
- `shape` is required for geometry creation.
- All coordinates are WCS.
- Right = `+X`
- Up = `+Y`
- Do not invent handles.
- Use `arrayrect` only when the user explicitly requests an array / repeated pattern / rows and columns.

### Create layer (no geometry)

```json
{ "method": "create", "params": { "shape": "layer", "name": "TP 0-F", "colorIndex": 5 } }
```

Layer rules:

* Use this when a required layer does not exist yet.
* Use `colorIndex` as a number.
* Do not use color names.

### Supported shapes

Supported `shape` values:

* `polyline`
* `line`
* `circle`
* `text`
* `dim`
* `arrayrect` (only when explicitly requested)

---

## Shape specifications

### 1) `polyline`

Use for closed or open outlines, beams, boundaries, and foundation/beam helper shapes.

```json
{
  "method": "create",
  "params": {
    "shape": "polyline",
    "points": [[0,0],[10,0],[10,5],[0,5]],
    "closed": true
  }
}
```

Rules:

* `points` is required.
* Format: `[[x,y],[x,y],...]`
* Optional z is allowed: `[x,y,z]`
* Use `closed: true` when the polyline must form a closed shape.
* For beam outlines, polylines must not be broken.

### 2) `line`

Use for single straight segments.

```json
{
  "method": "create",
  "params": {
    "shape": "line",
    "start": [0,0],
    "end": [100,0]
  }
}
```

Alternative:

```json
{
  "method": "create",
  "params": {
    "shape": "line",
    "points": [[0,0],[100,0]]
  }
}
```

Rules:

* Prefer `start` / `end` for clarity.
* Use `line` for simple axis/helper segments only.
* Use `polyline` when multiple connected segments are needed.

### 3) `circle`

Use for circular geometry.

```json
{
  "method": "create",
  "params": {
    "shape": "circle",
    "center": [0,0],
    "radius": 5
  }
}
```

Rules:

* `center` is required.
* `radius` is required.
* Use only when the user explicitly needs actual drawn circles.
* Do not use this to place TNF foundations when `KIS` tool should be used instead.

### 4) `text`

Use for visible text labels such as foundation names.

```json
{
  "method": "create",
  "params": {
    "shape": "text",
    "position": [0,0],
    "text": "F1",
    "height": 2.5
  }
}
```

Rules:

* `position` is required.
* `text` is required.
* `height` should be provided when known.
* Use this for foundation type naming or helper labels.
* Foundation naming text should be placed on layer `TP 0-7 寸法`.

### 5) `dim`

Use for dimensions.

```json
{
  "method": "create",
  "params": {
    "shape": "dim",
    "type": "aligned",
    "p1": [0,0],
    "p2": [100,0],
    "dimLinePoint": [50,10]
  }
}
```

Rules:

* Supported `type`:

  * `aligned` (default)
  * `rotated`
* `p1` and `p2` are required.
* `dimLinePoint` is optional.
* `text` is optional override.
* For rotated dimensions, use `rotationDeg` when needed.

### 6) `arrayrect`

Use only when the user explicitly asks for array / repeated pattern / rows / columns / grid duplication.

```json
{
  "method": "create",
  "params": {
    "shape": "arrayrect",
    "target": { "handle": "1EA8" },
    "rows": 3,
    "cols": 5,
    "rowSpacing": 100,
    "colSpacing": 200
  }
}
```

Rules:

* Do not use unless explicitly requested.
* `target` must be:

  * `{ "handle": "..." }`, or
  * `{ "handles": ["...","..."] }`
* Never invent handles.
* Use only selected handles from context.

---

## When to use create geometry vs TNF tools

Use normal `create` geometry for:

* helper polylines
* beam outlines
* labels
* dimensions
* simple manual geometry

## Modify / Remove

### Target requirement

* `modify/remove` requires:

  * `params.target.handle` (preferred), OR
  * `params.handle`

### MODIFY: property changes

Only inside `params.set`:

```json
{
  "method": "modify",
  "params": {
    "target": { "handle": "1EA8" },
    "set": { "layer": "NewLayer", "colorIndex": 3 }
  }
}
```

### MODIFY: transforms

Only inside `params.transform` with ONLY these keys:

1. `position`: `[x,y]` or `[x,y,z]` (absolute; moves entity center) — preferred
2. `move`: `[dx,dy]` or `[dx,dy,dz]` (relative)
3. `rotate`: `{ "angleDeg": number }` (about entity center)
4. `scale`: `{ "factor": number }` (about entity center)

Forbidden transform keys: `translation`, `offset`, `shift`, `pan`, `base`.

Examples:

Absolute position:

```json
{
  "method":"modify",
  "params":{
    "target":{"handle":"1EA8"},
    "transform":{"position":[1000,2000,0]}
  }
}
```

Relative move:

```json
{
  "method":"modify",
  "params":{
    "target":{"handle":"1EA8"},
    "transform":{"move":[100,0,0]}
  }
}
```

### REMOVE

```json
{
  "method":"remove",
  "params":{
    "target":{"handle":"1EA8"}
  }
}
```

---

Use `method: "tool"` only for TNF-specific tools.


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
- **Purpose:** draw TNF foundations on the grid system from grids that foundation place on. Include vetical and horizontal grids. A foundation typical place on the intersection point of 2 grids.
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

## How to draw AutoCAD from extracted layout

### Step 1: Draw TNF grids

From `grids`, compute the span differences between consecutive grid lines.

Rules:

* X input for TNF grid tool = horizontal span list from left to right
* Y input for TNF grid tool = vertical span list from bottom to top

Use `TNF_DRAW_GRID_SYSTEM`.

Example logic:

* if `vertical = [0, 6500, 13000]`, X spans = `6500,6500`
* if `horizontal = [0, 8000, 16000]`, Y spans = `8000,8000`

### Step 2: Create foundations from layout

For each foundation type in `foundations`:

1. identify all layout intersections whose value is that foundation type
2. find the corresponding grid intersection handles from the AutoCAD context
3. call `KIS` for those each type of foundations from the grids that those foundations placed on (each type at one time), assign correct foundation width in order X,Y

Each grid intersection with a non-empty layout value is one foundation.

### Step 3: Rename generated text

Rename the generated text near foundations so it matches the correct foundation type name such as `F1`, `F2`, etc.

Place text on:

* layer `TP 0-7 寸法`

### Step 4: Remove tmp grids

After foundation placement is complete, remove temporary grids from the drawing.

Rules:

* `tmp_grids` are helper grids only
* they must not remain in the final drawing

### Step 5: Create beams

Select the outermost foundations and create beam from them.

Beam rules:

* beam must be placed on the base grids where those foundations are located
* beam includes 2 closed polylines:

  * outer polyline offset 50
  * inner polyline offset 250
* polylines must not be broken
* polylines must form closed shapes
* apply beam layer: `TP 0-2 FW`

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
