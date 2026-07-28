# Demo Refactor, Conversation Search, and Stream Status Design

## Goal

Refactor `demo.py` while keeping it as the single Streamlit entry-point file, remove only
redundant or obsolete internal paths that can be proven behaviorally unnecessary, and fix
conversation search, the persistent loading row, and collapsing live chat status without
changing supported application behavior.

## Constraints

- `demo.py` remains one file; no Streamlit UI logic is moved into new modules.
- Existing supported API routes, stored conversation formats, authentication behavior,
  message rendering, HITL behavior, and streaming event semantics remain compatible.
- Compatibility code is removed only when repository evidence and characterization tests
  show that it is redundant or unreachable. Read compatibility for persisted historical data
  remains when removing it would change visible behavior.
- Search is scoped to conversations owned by the authenticated user.
- Search covers every non-deleted conversation and every message in those conversations, not
  only the currently loaded manager page or the three messages used for previews.

## Root Causes

### Incomplete and low-quality search

The Manage Conversations dialog lazily loads one page of at most 100 conversations. Its local
filter searches only those loaded rows and only the three recent messages included for each
preview. Conversations outside that page and matches in older messages are invisible. Results
retain load order rather than relevance order.

### First conversation element replaced by loading state

The initial synchronous fetch conditionally inserts
`st.status("Loading conversations...")` as the dialog's first element. On the next dialog rerun,
that status is omitted because the first page is cached, shifting the positions of all later
Streamlit elements. The first real conversation expander/statistics element can therefore be
reconciled against the prior loading component and appear to have been replaced by the Loading
conversations label.

### Collapsing live status and trace

The send and resume loops repeatedly update the label of a collapsible `st.status` without
reasserting its expanded state. The live trace is also rebuilt inside a placeholder and its
`stream_trace_expanded` state is forcibly changed as event types transition from agent selection
to thinking, tools, and answer tokens. These updates replace component state and can make the
user reopen the element repeatedly.

## Chosen Architecture

### Single-file internal organization

`demo.py` remains the application file. Refactoring will establish explicit sections for
constants and state, pure normalization/view helpers, API access, streaming state and event
projection, reusable UI primitives, and top-level page/dialog renderers. Repeated send/resume
status-update policy and conversation-manager transformations will be centralized in small
helpers inside the same file.

Existing helper names imported by tests or used as stable seams remain available. Large embedded
HTML, CSS, or JavaScript strings stay intact unless an exact duplicate can be removed safely.

### Server-side conversation search

The canonical conversation list route gains optional search parameters rather than adding a
separate route:

- `search`: normalized query text; blank input behaves exactly like the current list endpoint.
- `include=messages` and `latestMessages=3` continue to control preview data only.
- Pagination metadata describes the matched result set.

The repository performs the search in PostgreSQL across conversation titles and message content,
restricted by owner and excluding soft-deleted conversations. A conversation is returned once
even when several messages match. Ordering is deterministic and relevance-oriented:

1. exact normalized title match;
2. title prefix match;
3. title substring match;
4. message-content match;
5. most recently updated conversation as the tie-breaker.

Search matching is case-insensitive. The initial implementation uses SQL expressions supported by
the current PostgreSQL/SQLAlchemy stack and does not add an external search service or migration.

The Streamlit dialog requests the first search page from the server whenever a nonblank search is
submitted and displays the returned ranked results. Empty search keeps the existing lazy paging
and Load more behavior. Search results use the same preview cards and actions as ordinary manager
rows.

### Conversation manager loading state

The dialog creates the same dedicated loading placeholder at the same position on every render.
During an initial or load-more fetch, a transient spinner is rendered inside that placeholder;
after the fetch, only the placeholder's contents are cleared. Keeping the structural slot stable
prevents later widgets from shifting identity between reruns, while clearing its contents ensures
that no Loading conversations label remains visible and the first conversation retains its real
title and statistics. API failures leave the cache in a retryable state and show the existing
empty/error feedback rather than a misleading successful loading row.

### Stable live expansion

A single helper owns stream status updates for both initial sends and HITL resumes. Running status
updates explicitly preserve `expanded=True`. Live trace state remains expanded throughout active
thinking/tool work and is not forcibly collapsed when answer tokens begin. Completion and error
labels retain the current wording and state values.

The renderer continues projecting the same thinking, tool, subagent, image, token, interrupt,
complete, and error events. Only component expansion policy changes: a label transition must not
require a user click to reveal the live content again.

## Data Flow

### Search

1. The user enters a query in Manage Conversations.
2. Streamlit normalizes the text and calls the paginated conversation endpoint with `search` and
   preview include parameters.
3. The API passes the query through the service to the repository.
4. The repository filters all owned conversations through title/message matches, ranks distinct
   conversations, and paginates the matched set.
5. The existing response schemas serialize conversation previews and pagination metadata.
6. Streamlit renders ranked cards and fetches subsequent matched pages only when requested.

### Streaming status

1. The send or resume path creates one expanded status container.
2. Each stream event updates transient trace/response state as it does today.
3. Status label updates go through the shared helper, which explicitly keeps running content open.
4. Live trace redraws use an expanded default for the duration of the stream.
5. Terminal reconciliation persists the same metadata and reruns through the existing message
   renderer.

## Error Handling

- A blank search query follows the existing unfiltered pagination path.
- Failed search requests do not discard already cached unfiltered conversations.
- Invalid pagination and ownership rules retain existing validation and exception handling.
- Search does not return messages or conversations owned by another user.
- Duplicate message matches cannot duplicate a conversation row.
- Stream errors and interrupts keep their existing messages and terminal handling; the expansion
  helper changes presentation state only.

## Refactor and Cleanup Method

Before moving or removing code, characterization tests will capture the behavior of affected
helpers and static UI seams. Cleanup candidates will be classified as:

- duplicate implementation: consolidate behind one helper;
- unreachable internal code: remove after a test or static call-site check proves no supported
  path reaches it;
- compatibility read path: retain when historical persisted data or a supported client can still
  exercise it;
- operational fallback: retain when it is a documented availability feature rather than dead
  code.

This classification resolves the apparent conflict between removing legacy/fallback code and
preserving behavior: supported compatibility is behavior and is not deleted merely because its
name contains `legacy` or `fallback`.

## Testing

Tests are written before production changes and must fail for the intended reason.

- Repository/service/API tests prove title and full-message matches across conversations outside
  the first page, deterministic ranking, deduplication, pagination metadata, ownership isolation,
  soft-delete exclusion, and unchanged blank-search behavior.
- Streamlit tests prove query parameters are encoded correctly, search results are server-backed,
  unfiltered lazy paging remains available, the loading placeholder exists at a stable render
  position, and its transient contents are cleared after loading without replacing the first
  conversation expander/statistics element.
- Streaming UI tests prove every running label update explicitly remains expanded in both send and
  resume flows and token transitions do not force the trace closed.
- Characterization tests cover helpers affected by consolidation before code is rearranged.
- Focused pytest and Ruff checks run after each red-green-refactor cycle, followed by the relevant
  demo/API regression set and compilation of `demo.py`.
- When the local services are available, browser acceptance verifies the dialog and a real streamed
  status transition visually.

## Non-Goals

- Splitting `demo.py` into modules.
- Replacing Streamlit or changing the overall layout.
- Adding fuzzy, semantic, vector, or external full-text search infrastructure.
- Removing compatibility required to display existing persisted conversations.
- Changing conversation, message, streaming, HITL, or rich-response wire formats.

## Acceptance Criteria

- `demo.py` remains a single file and its supported behavior is preserved.
- Manage Conversations preserves the first conversation title/statistics element across reruns
  and shows no stale Loading conversations label after the fetch completes.
- A search can find any owned conversation by title or any of its messages, regardless of manager
  pagination or message age.
- Exact and title matches rank above message-only matches, with stable pagination.
- Live ChatAgent/Working/Thinking/tool status transitions remain visibly expanded without another
  click.
- Duplicate or unreachable internal code identified by the scoped audit is removed or consolidated,
  while required compatibility paths remain documented and tested.
- Focused and relevant regression tests, formatting, linting, and local browser acceptance pass.
