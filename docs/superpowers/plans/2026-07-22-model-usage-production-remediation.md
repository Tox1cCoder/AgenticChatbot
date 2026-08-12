# Model Usage Production Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct model-usage lifecycle, gauge, image, embedding, and model-limit behavior; remove superseded code; publish the frontend contract; and produce a verified sidecar bundle.

**Architecture:** Usage events and rollups remain the analytics authority, while validated metadata on the latest visible assistant message becomes the conversation gauge authority. Image paths enrich normalized usage before persistence or message save, explicit model-limit types survive catalog round trips, and conversation deletion cascades through both usage tables so reconciliation cannot resurrect aggregates.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2, PostgreSQL JSONB, Alembic, pytest/pytest-asyncio, Ruff, PowerShell sidecar packaging.

---

## File Structure

- `app/models/model_usage.py`: ORM foreign-key lifecycle contract.
- `app/alembic/versions/c6d7e8f9a0b1_cascade_model_usage_events_conversation.py`: reversible FK migration from `SET NULL` to `CASCADE`.
- `app/repositories/model_usage.py`: bounded assistant-message context lookup and ledger/rollup maintenance.
- `app/services/model_usage_service.py`: response assembly and context metadata validation.
- `app/usage/recorder.py`: response-aware normalized-usage transformation at the immutable write boundary.
- `app/ai/agents/base_agent.py`: default identity usage transform and recorder integration.
- `app/ai/agents/image_generator_agent.py`: inline-image count transform and dedicated-image terminal usage merge.
- `app/ai/model_context.py`: shared versus separate-I/O provider metadata normalization.
- `app/services/provider_service.py`: complete catalog context metadata serialization.
- `app/services/rag_embedding_service.py`: honest unavailable usage for non-text embeddings.
- `tests/test_model_usage_schema.py`: ORM FK contract.
- `tests/test_model_usage_conversation_cascade_migration.py`: migration source contract.
- `tests/test_alembic_full_chain_postgres.py`: current-head and live schema contract.
- `tests/integration/test_model_usage_repository_postgres.py`: live deletion/reconciliation and message-selection behavior.
- `tests/test_model_usage_service.py`: service gauge authority, validation, and SQL boundedness.
- `tests/test_model_usage_recorder.py`: transform timing and failure containment.
- `tests/test_model_usage_image_generation.py`: inline and dedicated image accounting.
- `tests/test_image_generation_providers.py`: structured image result compatibility updates.
- `tests/test_model_context_metadata.py`: explicit limit-type normalization.
- `tests/test_provider_model_context_metadata.py`: catalog round-trip coverage.
- `tests/test_rag_embedding_service.py`: image-only unavailable usage.
- `tests/test_model_usage_ai_sdk_contract.py`: executable Markdown/TypeScript frontend contract.
- `tests/test_model_usage_docs.py`: deployment-head and deletion documentation contract.
- `token_usage.md`: corrected source-plan lifecycle and completion notes.
- `docs/operations/model-usage-analytics.md`: deployment, rollback, deletion, and reconciliation runbook.
- `docs/operations/model-usage-cleanup-audit.md`: evidence and dispositions for dead/legacy/fallback audit candidates.
- `README.md`: current migration head and usage documentation links.
- `plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md`: focused frontend implementation contract.
- `plans/AI_SDK_FE_CONTRACT.md`: prominent link to the focused usage contract.

### Task 1: Make conversation deletion irreversible to usage reconciliation

**Files:**
- Create: `app/alembic/versions/c6d7e8f9a0b1_cascade_model_usage_events_conversation.py`
- Create: `tests/test_model_usage_conversation_cascade_migration.py`
- Modify: `app/models/model_usage.py:119-131`
- Modify: `tests/test_model_usage_schema.py:120-128`
- Modify: `tests/integration/test_model_usage_repository_postgres.py`
- Modify: `tests/test_alembic_full_chain_postgres.py:20-35,296-340`
- Modify: `tests/test_model_usage_docs.py:95-150`
- Modify: `token_usage.md:60-75,345-365`
- Modify: `docs/operations/model-usage-analytics.md:15-50,240-260,285-320`
- Modify: `README.md:490-540`

- [ ] **Step 1: Write failing ORM and migration contract tests**

Change the ORM assertion to require cascade and create a migration test that imports the new revision and checks both directions:

```python
def test_model_usage_events_foreign_key_delete_behavior():
    table = ModelUsageEvent.__table__
    assert next(iter(table.c.user_id.foreign_keys)).ondelete == "CASCADE"
    assert next(iter(table.c.conversation_id.foreign_keys)).ondelete == "CASCADE"
    assert next(iter(table.c.request_message_id.foreign_keys)).ondelete == "SET NULL"
    assert next(iter(table.c.document_id.foreign_keys)).ondelete == "SET NULL"


def test_conversation_fk_cascade_migration_contract():
    module = import_module(
        "app.alembic.versions.c6d7e8f9a0b1_cascade_model_usage_events_conversation"
    )
    assert module.revision == "c6d7e8f9a0b1"
    assert module.down_revision == "b5c6d7e8f9a0"
    upgrade = inspect.getsource(module.upgrade)
    downgrade = inspect.getsource(module.downgrade)
    assert 'ondelete="CASCADE"' in upgrade
    assert 'ondelete="SET NULL"' in downgrade
```

- [ ] **Step 2: Write the failing live deletion/reconciliation regression**

Add an integration test that records one event, confirms its rollup, deletes the conversation, verifies both rows are gone, reconciles the affected minute, and verifies they remain gone:

```python
def test_conversation_delete_removes_events_and_rollups_without_reconciliation_resurrection(
    repository, tenant_factory, session_factory
):
    user_id, conversation_id, _ = tenant_factory()
    started = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    repository.record_event(
        _command(user_id=user_id, conversation_id=conversation_id, started_at=started)
    )
    with session_factory.begin() as session:
        session.execute(delete(Conversation).where(Conversation.id == conversation_id))
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(ModelUsageEvent).where(
                ModelUsageEvent.user_id == user_id
            )
        ) == 0
        assert session.scalar(
            select(func.count()).select_from(ModelUsageMinute).where(
                ModelUsageMinute.user_id == user_id
            )
        ) == 0
    repository.reconcile_minute_range(
        start_inclusive=started,
        end_exclusive=started + timedelta(minutes=1),
    )
    with session_factory() as session:
        assert session.scalar(
            select(func.count()).select_from(ModelUsageEvent).where(
                ModelUsageEvent.user_id == user_id
            )
        ) == 0
        assert session.scalar(
            select(func.count()).select_from(ModelUsageMinute).where(
                ModelUsageMinute.user_id == user_id
            )
        ) == 0
```

- [ ] **Step 3: Run the tests to verify RED**

Run:

```powershell
python -m pytest tests/test_model_usage_schema.py tests/test_model_usage_conversation_cascade_migration.py -q
```

Expected: the ORM assertion fails and the migration import is missing. If `TEST_DATABASE_URL` is set, the live regression also fails because the event FK currently nulls its conversation reference.

- [ ] **Step 4: Implement the ORM and reversible migration**

Set the ORM FK to `ondelete="CASCADE"`. Create the migration with these operations:

```python
revision: str = "c6d7e8f9a0b1"
down_revision: str | None = "b5c6d7e8f9a0"


def _replace_conversation_fk(*, ondelete: str) -> None:
    op.drop_constraint(
        "fk_model_usage_events_conversation",
        "model_usage_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "fk_model_usage_events_conversation",
        "model_usage_events",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete=ondelete,
    )


def upgrade() -> None:
    _replace_conversation_fk(ondelete="CASCADE")


def downgrade() -> None:
    _replace_conversation_fk(ondelete="SET NULL")
```

- [ ] **Step 5: Update current-head, full-chain, source-plan, and runbook contracts**

Set `_HEAD = "c6d7e8f9a0b1"` in the full-chain test and assert the inspected `fk_model_usage_events_conversation` has `options["ondelete"] == "CASCADE"`. Update the README and runbook migration chain to name the new current head. Correct `token_usage.md` so raw-event and minute-rollup conversation FKs both say cascade, and document that hard conversation deletion removes raw events plus rollups before reconciliation.

- [ ] **Step 6: Run focused migration tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_usage_schema.py tests/test_model_usage_conversation_cascade_migration.py tests/test_model_usage_migration.py tests/test_model_usage_docs.py -q
python -m alembic heads
```

Expected: all tests pass and Alembic prints exactly `c6d7e8f9a0b1 (head)`. Live PostgreSQL tests may skip only when `TEST_DATABASE_URL` is absent.

- [ ] **Step 7: Commit the deletion fix**

```powershell
git add app/models/model_usage.py app/alembic/versions/c6d7e8f9a0b1_cascade_model_usage_events_conversation.py tests/test_model_usage_schema.py tests/test_model_usage_conversation_cascade_migration.py tests/integration/test_model_usage_repository_postgres.py tests/test_alembic_full_chain_postgres.py tests/test_model_usage_docs.py token_usage.md docs/operations/model-usage-analytics.md README.md
git commit -m "fix: cascade conversation usage deletion"
```

### Task 2: Make assistant-message metadata the context-gauge authority

**Files:**
- Modify: `app/repositories/model_usage.py:1-35,622-648`
- Modify: `app/services/model_usage_service.py:1-30,85-110,370-390`
- Modify: `tests/test_model_usage_service.py:210-320,780-940`
- Modify: `tests/integration/test_model_usage_repository_postgres.py:440-505`

- [ ] **Step 1: Write failing repository and service tests**

Replace fake-repository event methods with `get_latest_conversation_context_window`. Cover valid metadata, no metadata, malformed metadata, and a later helper ledger event:

```python
def test_conversation_uses_latest_assistant_message_context_not_later_helper_event():
    repository = PopulatedUsageRepository()
    repository.latest_context_window = {
            "provider": "gemini",
            "model": "gemini-3-pro-image",
            "context_window_tokens": None,
            "max_input_tokens": 65536,
            "max_output_tokens": 32768,
            "limit_type": "separate_io",
            "source": "registry",
            "known": True,
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "usage_source": "provider_reported",
            "input_usage_ratio": 100 / 65536,
            "output_usage_ratio": 200 / 32768,
            "usage_ratio": 200 / 32768,
            "usage_ratio_basis": "most_constrained_io_limit",
            "display_state": "ok",
    }
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )
    user_id = uuid4()
    conversation_id = uuid4()
    result = service.get_conversation_usage(
        user_id=user_id,
        conversation_id=conversation_id,
        query=ConversationUsageQuery(),
    )
    assert result.latest_context_window.model == "gemini-3-pro-image"
    assert ("latest_context", {"user_id": user_id, "conversation_id": conversation_id}) \
        in repository.calls


def test_invalid_latest_assistant_context_is_ignored():
    repository = PopulatedUsageRepository()
    repository.latest_context_window = {"provider": "openai"}
    service = ModelUsageService(
        repository=repository,
        conversation_repository=OwningConversationRepository(),
    )
    result = service.get_conversation_usage(
        user_id=uuid4(),
        conversation_id=uuid4(),
        query=ConversationUsageQuery(),
    )
    assert result.latest_context_window is None
```

The live repository test must insert user and assistant messages with different sequences, soft-delete the newest assistant message, add a later raw helper event, and assert the method returns only the newest non-deleted assistant message's `context_window`. A second tenant's message must never be returned.

- [ ] **Step 2: Run focused gauge tests to verify RED**

Run:

```powershell
python -m pytest tests/test_model_usage_service.py -k "latest or context" -q
```

Expected: failures mention the missing `get_latest_conversation_context_window` method and the service's obsolete raw-event lookup.

- [ ] **Step 3: Implement the bounded JSONB lookup**

Import `Message` and `MessageRole`, remove `get_latest_conversation_event`, and add:

```python
def get_latest_conversation_context_window(
    self, *, user_id: UUID, conversation_id: UUID
) -> dict[str, Any] | None:
    if user_id is None:
        raise ValueError(
            "get_latest_conversation_context_window requires a non-null user_id"
        )
    statement = (
        select(Message.message_metadata["context_window"])
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Conversation.owner_id == user_id,
            Conversation.id == conversation_id,
            Conversation.deleted_at.is_(None),
            Message.deleted_at.is_(None),
            Message.sender == MessageRole.assistant.value,
            Message.message_metadata.op("?")("context_window"),
        )
        .order_by(Message.sequence.desc(), Message.id.desc())
        .limit(1)
    )
    with self.session_factory() as session:
        value = session.execute(statement).scalar_one_or_none()
    return dict(value) if isinstance(value, dict) else None
```

- [ ] **Step 4: Validate stored metadata in the service and remove event reconstruction**

Remove imports of `build_context_window_usage`, `resolve_model_context_window`, and `NormalizedUsage`. Query the new repository method and validate without exposing stored content in logs:

```python
latest = self.repository.get_latest_conversation_context_window(
    user_id=user_id,
    conversation_id=conversation_id,
)


@staticmethod
def _latest_context_window(raw: Any | None) -> ContextWindowMetadata | None:
    if raw is None:
        return None
    try:
        return ContextWindowMetadata.model_validate(raw)
    except ValidationError:
        logger.warning("Ignoring invalid persisted context-window metadata")
        return None
```

- [ ] **Step 5: Run service and live repository tests to verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_usage_service.py tests/integration/test_model_usage_repository_postgres.py -q
```

Expected: unit tests pass; PostgreSQL tests pass when configured or report only the existing environment skip.

- [ ] **Step 6: Commit the gauge authority change**

```powershell
git add app/repositories/model_usage.py app/services/model_usage_service.py tests/test_model_usage_service.py tests/integration/test_model_usage_repository_postgres.py
git commit -m "fix: source conversation gauge from assistant messages"
```

### Task 3: Record inline generated-image counts before immutable persistence

**Files:**
- Modify: `app/usage/recorder.py:45-55,119-270`
- Modify: `app/ai/agents/base_agent.py:1030-1085`
- Modify: `app/ai/agents/image_generator_agent.py:1-35`
- Modify: `tests/test_model_usage_recorder.py`
- Modify: `tests/test_model_usage_image_generation.py`

- [ ] **Step 1: Write failing recorder transform tests**

Add async and sync tests proving the transform sees the provider response and resolved usage before persistence, plus a containment test proving a transform exception never converts a successful provider call into an application failure:

```python
async def test_async_attempt_transforms_usage_before_persistence():
    repository = FakeRepository()
    recorder = _recorder(repository)
    call = CountingCall(result={"usage_metadata": {"input_tokens": 5}})
    returned = await recorder.record_one_async_attempt(
        call=call,
        provider="gemini",
        model="gemini-image",
        operation=UsageOperation(),
        usage_transform=lambda seen, usage: replace(usage, generated_images=2),
    )
    assert returned is call.result
    assert repository.commands[0].usage.generated_images == 2


async def test_usage_transform_failure_keeps_original_usage_and_response(
):
    repository = FakeRepository()
    recorder = _recorder(repository)
    call = CountingCall(result={"usage_metadata": {"input_tokens": 5}})
    returned = await recorder.record_one_async_attempt(
        call=call,
        provider="gemini",
        model="gemini-image",
        operation=UsageOperation(),
        usage_transform=lambda _response, _usage: (_ for _ in ()).throw(ValueError()),
    )
    assert returned is call.result
    assert repository.commands[0].usage.generated_images == 0
```

- [ ] **Step 2: Run recorder tests to verify RED**

Run:

```powershell
python -m pytest tests/test_model_usage_recorder.py -k "transform" -q
```

Expected: the recorder rejects the new `usage_transform` keyword.

- [ ] **Step 3: Add the recorder transform boundary**

Define and use this callback for both async and sync wrappers:

```python
UsageTransform = Callable[[Any, NormalizedUsage], NormalizedUsage]


@staticmethod
def _transform_usage(
    response: Any,
    usage: NormalizedUsage,
    usage_transform: UsageTransform | None,
) -> NormalizedUsage:
    if usage_transform is None:
        return usage
    try:
        return usage_transform(response, usage)
    except Exception:
        logger.exception("model usage transform failed; preserving normalized usage")
        return usage
```

Pass `usage_transform` through `record_one_async_attempt` and `record_one_sync_attempt`, applying it after `_resolve_usage` and before `_build_command`.

- [ ] **Step 4: Add the base identity hook and image override**

In `BaseAgent`, pass `self._transform_recorded_usage` to the recorder and define:

```python
def _transform_recorded_usage(
    self, response: Any, usage: NormalizedUsage
) -> NormalizedUsage:
    return usage
```

In `ImageGeneratorAgent`, count response content blocks before the recorder persists:

```python
def _transform_recorded_usage(
    self, response: Any, usage: NormalizedUsage
) -> NormalizedUsage:
    images = extract_inline_images_from_content(getattr(response, "content", None))
    if not images:
        return usage
    return replace(usage, generated_images=len(images))
```

- [ ] **Step 5: Add and run the inline-image ledger regression**

The image-agent test must provide a raw model response with two inline image blocks and provider token metadata, invoke through the recorder boundary, and assert one successful event with `generated_images == 2` and the original token fields intact.

Run:

```powershell
python -m pytest tests/test_model_usage_recorder.py tests/test_model_usage_image_generation.py -q
```

Expected: all recorder and image usage tests pass.

- [ ] **Step 6: Commit inline image accounting**

```powershell
git add app/usage/recorder.py app/ai/agents/base_agent.py app/ai/agents/image_generator_agent.py tests/test_model_usage_recorder.py tests/test_model_usage_image_generation.py
git commit -m "fix: count inline images before usage persistence"
```

### Task 4: Merge dedicated image usage into the visible assistant response

**Files:**
- Modify: `app/ai/agents/image_generator_agent.py:1-40,175-210,270-390`
- Modify: `tests/test_model_usage_image_generation.py`
- Modify: `tests/test_image_generation_providers.py:300-375`

- [ ] **Step 1: Write the failing dedicated-image gauge test**

Exercise the agent with prompt-enhancement metadata followed by a successful dedicated image stream. Assert the terminal metadata names the image model and its separate-I/O ratios, not the prompt or acknowledgement model:

```python
assert response.metadata["context_window"] == {
    "provider": "gemini",
    "model": "gemini-3-pro-image",
    "context_window_tokens": None,
    "max_input_tokens": 65536,
    "max_output_tokens": 32768,
    "limit_type": "separate_io",
    "source": "registry",
    "known": True,
    "input_tokens": 120,
    "output_tokens": 320,
    "total_tokens": 440,
    "usage_source": "provider_reported",
    "used_tokens": 440,
    "used_token_source": "provider_reported_total",
    "input_usage_ratio": 120 / 65536,
    "output_usage_ratio": 320 / 32768,
    "usage_ratio": 320 / 32768,
    "usage_ratio_basis": "most_constrained_io_limit",
    "display_state": "ok",
}
```

- [ ] **Step 2: Run the dedicated-image test to verify RED**

Run:

```powershell
python -m pytest tests/test_model_usage_image_generation.py -k "context_window" -q
```

Expected: response metadata still contains the prompt model's gauge or has no terminal image usage.

- [ ] **Step 3: Introduce a structured terminal outcome**

Add an immutable internal type and replace tuple returns:

```python
@dataclass(frozen=True)
class ImageGenerationOutcome:
    images: list[dict[str, Any]]
    narrative: str
    usage: NormalizedUsage


terminal_usage = NormalizedUsage(source="unavailable")
# On ImageUsage:
terminal_usage = event.usage
# At exhaustion:
return ImageGenerationOutcome(
    images=images,
    narrative=narrative.strip(),
    usage=replace(terminal_usage, generated_images=len(images)),
)
```

Return an empty `ImageGenerationOutcome` when no provider is available. Update internal tests and callers to use `.images` and `.narrative`; do not retain tuple-unpacking compatibility code.

- [ ] **Step 4: Replace only the response gauge after successful image generation**

After `_generate_images` returns, build the image model's context metadata before the response is returned for persistence:

```python
context_window = resolve_model_context_window(
    image_provider_family(self.model_name), self.model_name
).to_dict()
context_window.update(build_context_window_usage(context_window, outcome.usage))
response.metadata = response.metadata or {}
response.metadata["context_window"] = context_window
```

Keep image attachment and narrative behavior unchanged. Do not let `_generate_user_facing_response` metadata or its later usage event overwrite this object.

- [ ] **Step 5: Run image suites to verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_usage_image_generation.py tests/test_image_generation_providers.py -q
```

Expected: all image provider, streaming, cancellation, accounting, and gauge tests pass.

- [ ] **Step 6: Commit dedicated image gauge behavior**

```powershell
git add app/ai/agents/image_generator_agent.py tests/test_model_usage_image_generation.py tests/test_image_generation_providers.py
git commit -m "fix: persist terminal image context usage"
```

### Task 5: Preserve separate input/output model limits through catalog metadata

**Files:**
- Modify: `app/ai/model_context.py:155-245`
- Modify: `app/services/provider_service.py:605-615`
- Modify: `tests/test_model_context_metadata.py:120-205`
- Modify: `tests/test_provider_model_context_metadata.py:330-410`

- [ ] **Step 1: Write failing explicit-limit tests**

Add a direct normalizer test and a catalog round-trip test:

```python
def test_normalize_explicit_separate_io_keeps_independent_limits():
    result = normalize_context_window_metadata(
        "gemini",
        "gemini-3-pro-image",
        {
            "context_window_tokens": None,
            "max_input_tokens": 65536,
            "max_output_tokens": 32768,
            "limit_type": "separate_io",
        },
    )
    assert result is not None
    assert result.limit_type == "separate_io"
    assert result.context_window_tokens is None
    assert result.max_input_tokens == 65536
    assert result.max_output_tokens == 32768


def test_provider_catalog_round_trip_preserves_separate_io(service):
    original = resolve_model_context_window("gemini", "gemini-3-pro-image")
    fields = service._context_window_fields(original)
    resolved = service._resolve_catalog_context_window(
        "gemini", "gemini-3-pro-image", fields
    )
    assert resolved == original
```

Also retain a legacy test proving metadata without `limit_type` still maps `max_input_tokens` to a shared context window.

- [ ] **Step 2: Run metadata tests to verify RED**

Run:

```powershell
python -m pytest tests/test_model_context_metadata.py tests/test_provider_model_context_metadata.py -k "limit or context_window" -q
```

Expected: `limit_type` is absent from catalog fields and separate-I/O returns as shared context.

- [ ] **Step 3: Separate shared-window and input-limit parsing**

Split accepted keys and branch on an explicit limit type:

```python
_SHARED_CONTEXT_KEYS = ("context_window_tokens", "contextWindowTokens")
_INPUT_LIMIT_KEYS = (
    "max_input_tokens",
    "maxInputTokens",
    "input_token_limit",
    "inputTokenLimit",
)

limit_type = raw_metadata.get("limit_type", raw_metadata.get("limitType"))
max_output = _extract_first(raw_metadata, _OUTPUT_KEYS)
if limit_type == "separate_io":
    max_input = _extract_first(raw_metadata, _INPUT_LIMIT_KEYS)
    if max_input is None and max_output is None:
        return None
    return ModelContextWindow(
        provider=provider,
        model=model_id,
        context_window_tokens=None,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        source="provider_api",
        known=True,
        limit_type="separate_io",
    )
```

For explicit `shared_context` and omitted legacy types, preserve the existing largest-context-candidate behavior. Reject unsupported explicit limit-type strings by returning `None` so registry resolution can supply a trusted value.

- [ ] **Step 4: Serialize the limit type in provider catalog entries**

Add this field to `_context_window_fields`:

```python
"limit_type": context_window.limit_type,
```

Extend `_CONTEXT_FIELD_KEYS` in the provider tests so recommended-model flagging must retain it.

- [ ] **Step 5: Run model metadata suites to verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_context_metadata.py tests/test_provider_model_context_metadata.py -q
```

Expected: all shared, separate-I/O, unknown, provider precedence, and schema tests pass.

- [ ] **Step 6: Commit limit metadata preservation**

```powershell
git add app/ai/model_context.py app/services/provider_service.py tests/test_model_context_metadata.py tests/test_provider_model_context_metadata.py
git commit -m "fix: preserve separate model token limits"
```

### Task 6: Represent image-only embedding usage as unavailable

**Files:**
- Modify: `app/services/rag_embedding_service.py:297-313`
- Modify: `tests/test_rag_embedding_service.py:360-405`

- [ ] **Step 1: Write the failing image-only embedding usage test**

Use the service's image embedding path with a response that has no provider usage:

```python
def test_image_embedding_without_provider_usage_is_unavailable(monkeypatch):
    service, client, repository = _build_recording_service(monkeypatch)
    client.models.embed_content.return_value = _make_response([0.1, 0.2])
    service.embed_image(
        b"image-bytes",
        mime_type="image/png",
        usage_context=UsageContext(user_id=uuid4(), operation="document_index"),
    )
    usage = repository.commands[0].usage
    assert usage.source == "unavailable"
    assert usage.input_tokens is None
```

- [ ] **Step 2: Run the embedding test to verify RED**

Run:

```powershell
python -m pytest tests/test_rag_embedding_service.py -k "image_embedding_without_provider_usage" -q
```

Expected: usage is `locally_estimated` with a misleading `input_tokens == 0`.

- [ ] **Step 3: Return unavailable when no text was estimable**

Track whether any string content was seen:

```python
items = contents if isinstance(contents, list) else [contents]
texts = [item for item in items if isinstance(item, str)]
if not texts:
    return NormalizedUsage(source="unavailable")
total = sum(
    counter.count_text(provider="gemini", model=self.model_name, text=text).tokens
    for text in texts
)
return NormalizedUsage(input_tokens=total, source="locally_estimated")
```

- [ ] **Step 4: Run the full embedding suite to verify GREEN**

Run:

```powershell
python -m pytest tests/test_rag_embedding_service.py -q
```

Expected: image-only usage is unavailable and existing text batch estimates remain positive.

- [ ] **Step 5: Commit honest embedding usage**

```powershell
git add app/services/rag_embedding_service.py tests/test_rag_embedding_service.py
git commit -m "fix: mark image embedding token usage unavailable"
```

### Task 7: Publish the frontend usage contract and source-of-truth links

**Files:**
- Modify: `plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md`
- Modify: `plans/AI_SDK_FE_CONTRACT.md:1-25,686-756`
- Modify: `tests/test_model_usage_ai_sdk_contract.py`
- Modify: `README.md:340-365`

- [ ] **Step 1: Write failing executable contract checks**

Add `UsageCapabilities` to the schema imports, parse a new `usage-capabilities-response` example, and require the authority and implementation phrases:

```python
def test_capability_example_matches_real_response_schema() -> None:
    parsed = ApiResponse[UsageCapabilities].model_validate(
        _json_example("usage-capabilities-response")
    )
    assert parsed.data == UsageCapabilities(enabled=True)


def test_contract_covers_capability_gating_and_message_gauge_authority() -> None:
    contract = _contract()
    for phrase in (
        "GET /usage/capabilities",
        "type UsageCapabilities",
        "latest non-deleted assistant message",
        "helper calls cannot replace",
        "No new SSE event",
        "AbortController",
        "separate_io",
        "unknown denominator",
        "Frontend acceptance checklist",
    ):
        assert phrase in contract
```

- [ ] **Step 2: Run contract tests to verify RED**

Run:

```powershell
python -m pytest tests/test_model_usage_ai_sdk_contract.py -q
```

Expected: capability example and new implementation guidance are missing.

- [ ] **Step 3: Update the focused contract**

Add the authenticated capability endpoint before dashboard fetching:

```json
<!-- example:usage-capabilities-response -->
{
  "success": true,
  "message": "Usage capabilities retrieved",
  "data": {"enabled": true},
  "error": null
}
```

Add `type UsageCapabilities = { enabled: boolean };`, capability gating, `AbortController` cancellation for filter changes, stale-response suppression, refresh-after-`finish`, and explicit no-polling behavior. State that `latestContextWindow` comes from the latest non-deleted assistant message with valid metadata; later prompt-enhancement, acknowledgement, suggestion, or helper calls cannot replace it. Include shared-context, separate-I/O, and unknown-denominator rendering examples and a frontend acceptance checklist. State verbatim that no new SSE event is introduced.

- [ ] **Step 4: Link from the main AI SDK contract and README**

Near the main contract's endpoint overview and context-window section add:

```markdown
For dashboard analytics, capability discovery, refresh behavior, and the complete
context-gauge UI contract, use
[`TOKEN_USAGE_AI_SDK_FE_CONTRACT.md`](../../../plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md).
```

Keep the detailed usage contract in one file; do not duplicate all TypeScript types in the main AI SDK contract.

- [ ] **Step 5: Run contract and API suites to verify GREEN**

Run:

```powershell
python -m pytest tests/test_model_usage_ai_sdk_contract.py tests/test_model_usage_api.py tests/test_ai_sdk_context_window.py tests/test_ai_sdk_v6_stream_contract.py -q
```

Expected: Markdown examples validate against actual Pydantic schemas and existing AI SDK stream ordering remains unchanged.

- [ ] **Step 6: Commit frontend documentation**

```powershell
git add plans/TOKEN_USAGE_AI_SDK_FE_CONTRACT.md plans/AI_SDK_FE_CONTRACT.md tests/test_model_usage_ai_sdk_contract.py README.md
git commit -m "docs: publish AI SDK usage frontend contract"
```

### Task 8: Remove superseded code and audit legacy/fallback paths

**Files:**
- Create: `docs/operations/model-usage-cleanup-audit.md`
- Modify: files identified in Tasks 2-7 only when the audit proves a symbol is unused or superseded

- [ ] **Step 1: Run static and reference audits**

Run:

```powershell
python -m ruff check app tests
rg -n "get_latest_conversation_event|_latest_context_window|tuple\[list\[dict\], str\]" app tests
rg -n -i "deprecated|legacy|fallback|compat(ibility)?|obsolete" app tests scripts README.md docs plans -g '*.py' -g '*.md'
rg -n "_CONTEXT_KEYS|_SHARED_CONTEXT_KEYS|_INPUT_LIMIT_KEYS" app tests
```

Expected: Ruff reports no unused imports or locals; the superseded raw-event selector and image tuple signature have no matches; context-key helpers each have active call sites. The broad legacy scan is classified rather than deleted indiscriminately.

- [ ] **Step 2: Remove concrete superseded code**

Confirm and remove these implementations and their obsolete tests:

- `ModelUsageRepository.get_latest_conversation_event` and its raw-event ordering tests.
- The event-to-context reconstruction imports and logic in `ModelUsageService`.
- Tuple-return compatibility in `_generate_images` and `_consume_image_stream`.
- Imports, fake-repository fields, comments, and assertions used only by those paths.

Do not retain aliases or wrappers for these internal methods because repository-wide reference search proves there are no external API contracts for them.

- [ ] **Step 3: Document retained compatibility and fallback paths with evidence**

Create the cleanup audit with this disposition table:

```markdown
| Candidate | Disposition | Evidence |
| --- | --- | --- |
| Raw-event latest gauge selector | Removed | Superseded by owner-bounded assistant-message metadata lookup; no remaining callers. |
| Image generation tuple result | Removed | Internal callers migrated to `ImageGenerationOutcome`; no public wire shape depended on it. |
| Provider retry/fallback runtime | Retained | Active production resilience path with dedicated tests and user-visible configuration. |
| Image acknowledgement text fallback | Retained | Required to return a usable response when the auxiliary model fails. |
| `gemini-3-pro-image-preview` registry alias | Retained | Persisted-history compatibility for the deprecated provider model identifier. |
| Sidecar `/api` aliases and single-upload wrapper | Retained | Documented external compatibility contracts outside model-usage scope. |
```

Record that `vulture` was not added as a production dependency; Ruff plus exact reference scans are the available reproducible static evidence.

- [ ] **Step 4: Treat warnings as errors across the affected subsystem**

Run:

```powershell
python -W error -m pytest tests/test_model_usage_recorder.py tests/test_model_usage_service.py tests/test_model_usage_image_generation.py tests/test_model_context_metadata.py tests/test_provider_model_context_metadata.py tests/test_rag_embedding_service.py tests/test_model_usage_ai_sdk_contract.py -q
```

Expected: all tests pass with no deprecation or resource warnings. Any warning introduced or exposed by changed code is fixed at its source; unrelated third-party warnings are reported with exact origin rather than suppressed globally.

- [ ] **Step 5: Commit cleanup evidence and removals**

```powershell
git add app tests docs/operations/model-usage-cleanup-audit.md
git diff --cached --check
git commit -m "refactor: remove superseded usage paths"
```

### Task 9: Run release verification and build the sidecar

**Files:**
- Modify only files required to fix verification failures caused by this change
- Produce: `dist/client-backend-bundle/`
- Produce: `dist/client-backend-bundle.zip`

- [ ] **Step 1: Run the focused production test matrix**

Run:

```powershell
python -m pytest tests/test_model_usage_schema.py tests/test_model_usage_conversation_cascade_migration.py tests/test_model_usage_migration.py tests/test_model_usage_timestamp_index_migration.py tests/test_model_usage_recorder.py tests/test_model_usage_service.py tests/test_model_usage_image_generation.py tests/test_image_generation_providers.py tests/test_model_context_metadata.py tests/test_provider_model_context_metadata.py tests/test_rag_embedding_service.py tests/test_model_usage_api.py tests/test_model_usage_ai_sdk_contract.py tests/test_model_usage_docs.py tests/test_ai_sdk_context_window.py tests/test_ai_sdk_v6_stream_contract.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run live PostgreSQL coverage when configured**

Run:

```powershell
python -m pytest tests/integration/test_model_usage_repository_postgres.py tests/test_alembic_full_chain_postgres.py -q
```

Expected: all tests pass when `TEST_DATABASE_URL` is configured. If absent, record exact skip counts and do not describe live database coverage as passed.

- [ ] **Step 3: Run the full supported test suite**

Run:

```powershell
python -W error -m pytest -q
```

Expected: all non-environmental tests pass. Investigate every new failure with the systematic-debugging workflow; do not weaken assertions or add global warning filters.

- [ ] **Step 4: Run formatting, lint, migration, and diff gates**

Run:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m alembic heads
git diff --check
git status --short
```

Expected: Ruff and diff checks are clean, Alembic reports only `c6d7e8f9a0b1 (head)`, and status contains only intentional plan/progress artifacts if any.

- [ ] **Step 5: Build the requested sidecar bundle**

Run the documented command exactly:

```powershell
pwsh -File scripts/build-client-backend-bundle.ps1
```

Expected: exit code 0 and the build reports completion.

- [ ] **Step 6: Verify sidecar artifacts and archive contents**

Run:

```powershell
if (-not (Test-Path 'dist/client-backend-bundle' -PathType Container)) { throw 'bundle directory missing' }
if (-not (Test-Path 'dist/client-backend-bundle.zip' -PathType Leaf)) { throw 'bundle zip missing' }
$archive = Get-Item 'dist/client-backend-bundle.zip'
if ($archive.Length -le 0) { throw 'bundle zip is empty' }
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($archive.FullName)
try {
    if ($zip.Entries.Count -eq 0) { throw 'bundle zip has no entries' }
    $zip.Entries | Select-Object -First 20 -ExpandProperty FullName
} finally {
    $zip.Dispose()
}
```

Expected: the directory exists, the zip is non-empty, and archive entries are listed.

- [ ] **Step 7: Review and commit final verification-driven corrections**

```powershell
git diff --check
git status --short
git add --all
git commit -m "chore: finalize model usage production remediation"
```

Do not commit generated bundle artifacts if they are ignored by the repository. Report test counts, skips, migration head, cleanup dispositions, and absolute artifact paths in the handoff.
