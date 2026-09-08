# Provider and Streaming Regression Repair

## Problem

Two recent changes broke specialist chat behavior:

- OpenAI `gpt-5.6-luna` tool calls use Chat Completions with reasoning enabled. The provider rejects that combination, causing unnecessary Gemini fallback and sometimes an empty terminal response.
- The stream privacy filter classifies nested specialist chunks by `langgraph_node`. LangGraph reports those chunks as the inner node `model`; the public parent specialist is represented by the first namespace segment. Legitimate deltas are therefore discarded, and both clients fall back to the completed response.

The reverted tool-description clamp is intentionally excluded because the captured provider error proves schema length was not the cause.

## Design

### OpenAI transport

Centralize runtime reasoning configuration in `ModelFactory`. OpenAI runtime models with effective reasoning enabled will use the Responses API, which supports reasoning with function tools. Explicit `none` remains valid without silently upgrading reasoning. The same construction path will serve specialists, routing, and configured fallbacks so runtime settings cannot diverge between legacy and routing-v2 paths.

Provider fallback remains a single recovery attempt for genuine failures. Its warning will include a safe exception type and message so future provider incompatibilities are diagnosable without exposing credentials or request payloads.

### Streaming privacy

Classify a message source using both its inner node and graph namespace:

- A direct public specialist node is public.
- An inner `model` node is public only when the first namespace component names a registered public specialist.
- Router, grader, and other internal namespaces remain private.
- Planning worker deltas continue through the existing attributed subagent channel before the public/private check.
- Unattributed scripted/legacy chunks retain their existing compatibility behavior.

Apply the same predicate to the v3 translator and tuple fallback projector so AI SDK and Streamlit receive identical incremental output.

### Validation and cleanup

Regression tests will use a real nested `create_agent` graph with a streaming fake model, asserting that deltas arrive before completion and that internal graph output stays private. Model tests will assert correct OpenAI transport/reasoning parameters and fallback diagnostics. A focused workflow test will cover terminal response generation after provider recovery.

Cleanup is restricted to touched modules: remove redundant docstrings/comments, consolidate duplicated model configuration, and keep security rationale where it explains a non-obvious boundary. Existing unrelated working-tree changes remain untouched.

