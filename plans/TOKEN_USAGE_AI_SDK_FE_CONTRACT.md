# Token usage frontend contract

This is the frontend contract for authenticated, user-scoped token analytics. It is
additive to the existing AI SDK v6 contract: no new stream event is required, and the
terminal sequence remains `data-assistant-message`, `text-end`, `finish-step`, `finish`,
then `[DONE]` when an assistant message is available.

## HTTP API

Send `Authorization: Bearer <JWT>` on both requests. The server always derives the user
from that JWT; clients must not send a user ID.

- `GET /usage/dashboard` accepts `from`, `to`, `bucket`, `timezone`, and optional
  `conversationId`.
- `GET /usage/conversations/{conversationId}` accepts `from`, `to`, `bucket`, and
  `timezone`. The path conversation must belong to the authenticated user.

`from` and `to` must either both be omitted or both be RFC 3339 timestamps with an
explicit numeric offset. `from` is inclusive and `to` is exclusive. `bucket` is `hour`
or `day` (default `day`), and `timezone` is an IANA zone (default `UTC`). Hour ranges
must use local top-of-hour boundaries and cannot exceed 31 days. Day ranges must use
local-midnight boundaries. No range can exceed 730 days. The optional dashboard
`conversationId` is a UUID. These camelCase spellings are the public query aliases:
use `conversationId`, never `conversation_id`, and never send `userId` or `user_id`.
When the range is omitted, the dashboard defaults to 30 local calendar days and a
conversation defaults to the retained 730 days. Both defaults end at the next local
midnight in the requested timezone.

Responses are `200`. Missing or invalid JWT is `401`; a disabled usage UI or a
conversation outside the user's ownership is `404`; malformed UUIDs, timestamps,
timezones, alignment, or ranges are `422`. Treat other non-2xx responses as errors and
use the existing API error envelope.

## Exact JSON examples

<!-- example:usage-dashboard-response -->
```json
{
  "success": true,
  "message": "Usage dashboard retrieved",
  "data": {
    "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3},
    "outcomes": [
      {"key": "success", "totals": {"inputTokens": 1000, "outputTokens": 250, "totalTokens": 1250, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 2}},
      {"key": "error", "totals": {"inputTokens": 200, "outputTokens": 50, "totalTokens": 250, "reasoningTokens": 0, "cachedInputTokens": 0, "generatedImages": 0, "requestCount": 1}}
    ],
    "series": [{"start": "2026-07-01T00:00:00Z", "end": "2026-07-02T00:00:00Z", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "byProvider": [{"key": "provider-a", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "byModel": [{"key": "model-a", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "byOperation": [{"key": "chat", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "byAgent": [{"key": "assistant-a", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "topConversations": [{"conversationId": "8b93ef00-9b1c-4c25-a605-e588d70f8ae0", "title": "Example conversation", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "coverage": {"providerReportedRequests": 2, "mixedRequests": 0, "locallyEstimatedRequests": 1, "unavailableRequests": 0, "requestsWithKnownTotal": 3, "totalRequests": 3, "knownTotalRatio": 1.0},
    "range": {"from": "2026-07-01T00:00:00Z", "to": "2026-07-02T00:00:00Z", "bucket": "day", "timezone": "UTC"},
    "generatedAt": "2026-07-02T00:00:01Z"
  },
  "error": null
}
```

<!-- example:conversation-usage-response -->
```json
{
  "success": true,
  "message": "Conversation usage retrieved",
  "data": {
    "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3},
    "byProvider": [{"key": "provider-a", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "byModel": [{"key": "model-a", "totals": {"inputTokens": 1200, "outputTokens": 300, "totalTokens": 1500, "reasoningTokens": 25, "cachedInputTokens": 100, "generatedImages": 1, "requestCount": 3}}],
    "coverage": {"providerReportedRequests": 2, "mixedRequests": 0, "locallyEstimatedRequests": 1, "unavailableRequests": 0, "requestsWithKnownTotal": 3, "totalRequests": 3, "knownTotalRatio": 1.0},
    "latestContextWindow": {"provider": "provider-a", "model": "model-a", "context_window_tokens": 128000, "max_input_tokens": 128000, "max_output_tokens": 16384, "limit_type": "shared_context", "source": "provider_api", "known": true, "input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500, "usage_source": "provider_reported", "used_tokens": 1500, "used_token_source": "provider_reported_total", "input_usage_ratio": 0.009375, "output_usage_ratio": 0.00234375, "usage_ratio": 0.01171875, "usage_ratio_basis": "shared_context_total", "display_state": "ok"},
    "range": {"from": "2026-07-01T00:00:00Z", "to": "2026-07-02T00:00:00Z", "bucket": "day", "timezone": "UTC"},
    "generatedAt": "2026-07-02T00:00:01Z"
  },
  "error": null
}
```

<!-- example:context-window-metadata -->
```json
{
  "provider": "provider-a",
  "model": "model-a",
  "context_window_tokens": 128000,
  "max_input_tokens": 128000,
  "max_output_tokens": 16384,
  "limit_type": "shared_context",
  "source": "provider_api",
  "known": true,
  "input_tokens": 1200,
  "output_tokens": 300,
  "total_tokens": 1500,
  "usage_source": "provider_reported",
  "used_tokens": 1500,
  "used_token_source": "provider_reported_total",
  "input_usage_ratio": 0.009375,
  "output_usage_ratio": 0.00234375,
  "usage_ratio": 0.01171875,
  "usage_ratio_basis": "shared_context_total",
  "display_state": "ok"
}
```

The endpoint examples are executed against the actual Pydantic response envelopes in
tests. They are not parallel handwritten validation schemas.

## TypeScript types

API aggregate fields are camelCase. `context_window` is persisted AI SDK metadata and
intentionally keeps its snake_case keys.

```ts
type UsageTotals = { inputTokens: number; outputTokens: number; totalTokens: number; reasoningTokens: number; cachedInputTokens: number; generatedImages: number; requestCount: number };
type UsageBreakdownItem = { key: string; totals: UsageTotals };
type UsageSeriesPoint = { start: string; end: string; totals: UsageTotals };
type UsageCoverage = { providerReportedRequests: number; mixedRequests: number; locallyEstimatedRequests: number; unavailableRequests: number; requestsWithKnownTotal: number; totalRequests: number; knownTotalRatio: number };
type UsageRange = { from: string; to: string; bucket: "hour" | "day"; timezone: string };
type ConversationUsageItem = { conversationId: string; title: string | null; totals: UsageTotals };

type ContextWindowMetadata = {
  provider: string; model: string;
  context_window_tokens: number | null; max_input_tokens: number | null; max_output_tokens: number | null;
  limit_type: "shared_context" | "separate_io" | "unknown";
  source: "provider_api" | "registry" | "heuristic" | "unknown"; known: boolean;
  input_tokens?: number | null; output_tokens?: number | null; total_tokens?: number | null;
  usage_source?: "provider_reported" | "mixed_reported_estimated" | "locally_estimated" | "unavailable" | null;
  used_tokens?: number | null;
  used_token_source?: "provider_reported_total" | "provider_reported_split" | "estimated_total" | "unknown" | null;
  input_usage_ratio?: number | null; output_usage_ratio?: number | null; usage_ratio?: number | null;
  usage_ratio_basis?: "shared_context_total" | "most_constrained_io_limit" | null;
  display_state?: "unknown" | "ok" | "warn" | "danger" | null;
};

type UsageDashboard = {
  totals: UsageTotals; outcomes: UsageBreakdownItem[]; series: UsageSeriesPoint[];
  byProvider: UsageBreakdownItem[]; byModel: UsageBreakdownItem[];
  byOperation: UsageBreakdownItem[]; byAgent: UsageBreakdownItem[];
  topConversations: ConversationUsageItem[]; coverage: UsageCoverage;
  range: UsageRange; generatedAt: string;
};
type ConversationUsage = {
  totals: UsageTotals; byProvider: UsageBreakdownItem[]; byModel: UsageBreakdownItem[];
  coverage: UsageCoverage; latestContextWindow: ContextWindowMetadata | null;
  range: UsageRange; generatedAt: string;
};
type ApiResponse<T> = { success: boolean; message: string; data: T | null; error?: Record<string, unknown> | null; code?: string };
```

The context shape appears after completion at
`data-assistant-message.data.message.metadata.context_window` and in history at
`messages[].metadata.context_window`. It is additive. Do not require a usage event, and
do not copy dashboard totals, series, rankings, or top conversations into message
metadata. Ignore all unknown future fields in API objects and message metadata.

## Rendering and refresh

- Plot `series[].totals.inputTokens` and `outputTokens` as a stacked input/output area
  or bar chart. Derive each outcome rate as its `requestCount / totals.requestCount`.
- Render ranked provider/model/operation/agent lists from their matching arrays and top
  conversations from `topConversations`. Arrays are already bounded and ordered.
- Fetch the dashboard on view entry and every filter change. Refetch a conversation
  summary after the AI SDK `finish` event. Never poll during generation.
- For `shared_context`, the gauge's raw ratio is `used_tokens / context_window_tokens`.
  For `separate_io`, calculate both available ratios and show their maximum (the most
  constrained limit). With an unknown denominator, show counts but no percentage.
  Clamp only the visual fill to `min(max(raw ratio, 0), 1)`; display and retain the
  uncapped raw ratio.
- **Loading:** retain the layout and show a progress/skeleton state.
- **Empty:** show zero totals and an explicit no-usage message; empty arrays are valid.
- **Error:** keep chat usable, show a retry action, and do not substitute fabricated data.

Clients must ignore unknown future fields everywhere for forward compatibility.
