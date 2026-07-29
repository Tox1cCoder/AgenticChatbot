# Rich Image Injection Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make selected web images relevant, naturally placed, singly captioned, and resilient to blocked or expired upstream media without adding model evaluation, a labeled dataset, or answer-path availability checks.

**Architecture:** Keep the existing rich-item registry and marker placement contract. Brave is the focused image-discovery provider; Tavily remains web research and can return source-bound images when explicitly requested. Candidate handling is deterministic. Selected remote images are converted during message persistence into authenticated, user-owned `/web-images/{id}` references, but the upstream bytes are fetched only when a client renders the image. The renderer owns one structured footer and a complete loading/loaded/failed figure state.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, SQLAlchemy + Alembic, HTTPX 0.28, Pillow, Prometheus client, Streamlit, AI SDK v6 streaming, pytest/pytest-asyncio.

## Global Constraints

- Approved design: `docs/superpowers/specs/2026-07-29-rich-image-injection-reliability-design.md`.
- Do not add a relevance dataset, vision-model call, learned reranker, caption-model call, or pre-persistence image availability probe.
- Do not change `tavily_search_include_images` from its current `True` default in this rollout. A caller that omits `include_images` must retain configured behavior; an explicit `False` must suppress images for that call only.
- Image failure must never fail assistant completion or message persistence. If reference registration fails, remove that rich item and its marker and persist the text answer.
- The media route is not an open proxy. It accepts only an opaque ID created from a selected provider result and checks user ownership before any upstream request.
- New upstream display URLs are HTTPS-only. Existing historical `http://` rich items remain readable by old renderers but must not be registered as new web-image references.
- Every redirect hop must be independently validated and DNS-pinned. Merely resolving a hostname and then asking a normal client to resolve it again is not sufficient because it permits DNS rebinding.
- Remote bytes are fetched on media request, not during answer generation. Therefore media timeout does not affect text time-to-first-token or assistant completion latency.
- `payload.url` remains the preferred display URL for backward compatibility. `payload.source_url` remains the publisher page. Never overwrite it with the direct image URL.
- `alt_text` is accessibility text, not a visible caption. Web-image footers show source attribution only; an explicit `payload.caption` is reserved for trustworthy captions such as document figures or generated assets.
- Do not log full source/image URLs, query text, image bytes, user IDs, conversation IDs, or opaque media IDs in rich-image metrics/logs.
- Keep current selected-only terminal streaming semantics: image candidates are not transient rich-item upserts.
- Use async repository transport on FastAPI/message persistence paths; do not add sync DB I/O to an async request path.
- One logical change per commit, imperative subject at most 72 characters. Preserve unrelated user changes in a dirty worktree.

## File and Interface Map

| Area | Files | Responsibility |
|---|---|---|
| Public contract | `app/core/rich_response.py`, `app/ai/prompts.py` | Image payload fields, protected URL validation, and single-caption prompt rules. |
| Provider adapters | `app/ai/mcp_servers/tavily_server.py`, `app/ai/mcp_servers/brave_image_search_server.py` | Per-call Tavily image toggle and normalized provider metadata. |
| Candidate selection | `app/ai/tool_execution.py` | Preferred display URL, deterministic rejection/dedup, bounded candidates. |
| Persistence | `app/models/web_image_reference.py`, `app/repositories/web_image_reference.py`, Alembic | User-owned opaque references to selected upstream images. |
| Secure delivery | `app/services/web_image_service.py`, `app/api/web_images.py` | DNS-pinned HTTPS fetch, redirect/MIME/size/dimension enforcement. |
| Message boundary | `app/services/message_service.py`, `app/core/container.py` | Convert selected remote URLs to protected references without failing the answer. |
| Streaming | `app/services/event_streaming/ai_sdk_projection.py` | Recognize protected web-image URLs and preserve selected-only file parts. |
| Demo UI | `app/ui/rich_response.py`, `demo.py` | One footer, reserved layout, full-figure failure replacement. |
| Operations/docs | `app/observability/rich_images.py`, `app/api/health.py`, `README.md`, `.env.example` | Bounded telemetry and FE integration contract. |

---

## Task 1: Make the public contract single-caption and backward compatible

**Files:**
- Modify: `app/core/rich_response.py`
- Modify: `app/ai/prompts.py`
- Test: `tests/test_rich_response_contract.py`
- Test: `tests/test_rich_response_prompt_inventory.py`

**Interfaces:**
- `ImagePayload` adds `width: int | None`, `height: int | None`, and `caption: str | None`.
- Image `payload.url` accepts absolute HTTP(S) history URLs plus the protected chat/web-image route prefixes, including the existing `/api/...` compatibility forms.
- `payload.source_url` remains absolute HTTP(S).
- Prompt inventory tells the model to place the marker but never manufacture a Markdown caption.

- [ ] **Step 1: Write failing schema tests**

Add tests equivalent to:

```python
def test_image_payload_accepts_optional_display_metadata_and_protected_url():
    item = validate_public_rich_item(
        {
            "id": "image:web:1",
            "type": "image",
            "display_policy": "inline_only",
            "alt_text": "A red panda in a tree",
            "payload": {
                "url": "/web-images/55d170b5-b0f0-44fc-9155-af8af484513d",
                "mime_type": "image/jpeg",
                "source_url": "https://publisher.example/story",
                "width": 640,
                "height": 360,
                "caption": "Publisher-supplied figure caption",
            },
        }
    )
    assert item.payload.width == 640
    assert item.payload.height == 360
    assert item.payload.caption == "Publisher-supplied figure caption"


def test_image_payload_rejects_unknown_relative_url():
    with pytest.raises(ValidationError, match="protected image url"):
        validate_public_rich_item(
            {
                "id": "image:web:1",
                "type": "image",
                "display_policy": "inline_only",
                "alt_text": "Example",
                "payload": {
                    "url": "/proxy?url=https://internal.example",
                    "mime_type": "image/png",
                },
            }
        )
```

Add prompt assertions:

```python
def test_inventory_assigns_caption_ownership_to_renderer():
    prompt = build_rich_response_guidance(
        candidates=[_image_candidate("image:tool:c1:0")],
        enabled=True,
        capability=True,
    )
    assert "Do not write a Markdown caption" in prompt
    assert "caption immediately after" not in prompt
```

- [ ] **Step 2: Run tests and verify the contract fails first**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_rich_response_contract.py tests/test_rich_response_prompt_inventory.py -v
```

Expected: FAIL because `width`, `height`, and `caption` are forbidden, `/web-images/...` is rejected, and the old prompt still requests a caption.

- [ ] **Step 3: Implement the narrow URL and metadata contract**

In `app/core/rich_response.py`, add constrained fields and keep source-page validation absolute:

```python
PROTECTED_IMAGE_URL_PREFIXES: tuple[str, ...] = (
    "/chat-images/",
    "/api/chat-images/",
    "/web-images/",
    "/api/web-images/",
)


def _validate_image_url(url: str | None) -> None:
    if url is None:
        return
    if url.startswith(PROTECTED_IMAGE_URL_PREFIXES):
        return
    if url.startswith("/"):
        raise ValueError(f"unsupported protected image url: {url!r}")
    _validate_url_scheme(url)


class ImagePayload(PublicPayload):
    url: str | None = None
    data: str | None = None
    mime_type: str
    source_url: str | None = None
    description: str | None = None
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    caption: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _has_exactly_one_source(self) -> "ImagePayload":
        if (self.url is None) == (self.data is None):
            raise ValueError("image payload requires exactly one of url or data")
        if self.mime_type not in ALLOWED_IMAGE_MIME_TYPES:
            raise ValueError(f"unsupported image mime_type {self.mime_type!r}")
        _validate_image_url(self.url)
        _validate_url_scheme(self.source_url)
        return self
```

- [ ] **Step 4: Remove both caption-generation instructions**

Update `_INVENTORY_FOOTER` in `app/core/rich_response.py` and `INLINE_RICH_RESPONSE_SUFFIX` in `app/ai/prompts.py` to use the same rule:

```text
Place each selected item with its exact marker on a separate line near the
supporting paragraph. Do not write a Markdown caption after an image marker;
the renderer owns the single structured figure footer. Use surrounding prose
only when it materially helps the answer.
```

- [ ] **Step 5: Run the focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add app/core/rich_response.py app/ai/prompts.py tests/test_rich_response_contract.py tests/test_rich_response_prompt_inventory.py
git commit -m "fix: assign rich image captions to renderer"
```

---

## Task 2: Normalize provider images and select them deterministically

**Files:**
- Modify: `app/ai/mcp_servers/tavily_server.py`
- Modify: `app/ai/mcp_servers/brave_image_search_server.py`
- Modify: `app/ai/tool_execution.py`
- Modify: `app/core/config.py`
- Test: `tests/test_tavily_server.py`
- Test: `tests/test_brave_image_search_server.py`
- Test: `tests/test_tool_execution_rendering.py`
- Create: `tests/test_rich_image_config.py`

**Interfaces:**
- `tavily_search(..., include_images: bool | None = None)`; `None` resolves to `settings.tavily_search_include_images`.
- Tavily returns one normalized `images` list containing top-level and result-bound images.
- Brave continues returning original and thumbnail URLs, but candidate selection makes the Brave thumbnail the display upstream.
- `build_image_candidates_from_tool_result` rejects non-HTTPS remote display URLs, exact duplicates, and known-small images, and returns at most `rich_image_candidate_max_count`.

- [ ] **Step 1: Add failing Tavily behavior tests**

Cover all three toggle states and parent provenance:

```python
def test_tavily_search_omitted_include_images_uses_config(monkeypatch):
    monkeypatch.setattr(settings, "tavily_search_include_images", True)
    client = _RecordingClient(_search_response())
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    json.loads(tavily_server.tavily_search("query"))
    assert client.search_kwargs["include_images"] is True


def test_tavily_search_explicit_false_suppresses_images(monkeypatch):
    client = _RecordingClient(_search_response())
    monkeypatch.setattr(tavily_server, "_make_client", lambda: client)
    payload = json.loads(tavily_server.tavily_search("query", include_images=False))
    assert client.search_kwargs["include_images"] is False
    assert client.search_kwargs["include_image_descriptions"] is False
    assert payload["images"] == []


def test_tavily_result_images_keep_parent_page_provenance():
    normalized = tavily_server._normalize_search_response(
        query="mars rover",
        response={
            "images": ["https://cdn.example/query.jpg"],
            "results": [{
                "url": "https://publisher.example/rover",
                "title": "Rover story",
                "score": 0.91,
                "images": [{"url": "https://cdn.example/rover.jpg", "description": "Rover"}],
            }],
        },
        include_images=True,
    )
    result_image = next(x for x in normalized["images"] if x["url"].endswith("rover.jpg"))
    assert result_image["source_url"] == "https://publisher.example/rover"
    assert result_image["result_rank"] == 0
    assert result_image["provider"] == "tavily"
```

- [ ] **Step 2: Add failing deterministic-selection tests**

```python
def test_brave_candidate_prefers_thumbnail_and_keeps_original_in_provenance():
    candidates = build_image_candidates_from_tool_result(
        json.dumps({"images": [{
            "url": "https://origin.example/full.jpg",
            "thumbnail_url": "https://imgs.search.brave.com/proxy.jpg",
            "source_url": "https://publisher.example/page",
            "provider": "brave",
            "width": 1200,
            "height": 800,
            "description": "A mountain lake",
        }]}),
        tool_call_id="call-1",
        tool_name="brave_image_search",
    )
    assert candidates[0]["payload"]["url"] == "https://imgs.search.brave.com/proxy.jpg"
    assert candidates[0]["provenance"]["original_image_url"] == (
        "https://origin.example/full.jpg"
    )


def test_candidates_reject_insecure_duplicate_and_known_tiny_images(monkeypatch):
    monkeypatch.setattr(settings, "rich_image_min_width_px", 320)
    monkeypatch.setattr(settings, "rich_image_min_height_px", 180)
    payload = {"images": [
        {"url": "http://img.example/a.jpg", "width": 800, "height": 600},
        {"url": "https://img.example/a.jpg", "width": 100, "height": 100},
        {"url": "https://img.example/b.jpg", "width": 800, "height": 600},
        {"url": "https://img.example/b.jpg", "width": 800, "height": 600},
    ]}
    candidates = build_image_candidates_from_tool_result(
        json.dumps(payload), tool_call_id="call-1", tool_name="brave_image_search"
    )
    assert [x["payload"]["url"] for x in candidates] == ["https://img.example/b.jpg"]
```

- [ ] **Step 3: Run focused tests and verify failure**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py tests/test_brave_image_search_server.py tests/test_tool_execution_rendering.py tests/test_rich_image_config.py -v
```

Expected: FAIL on the missing Tavily parameter/provenance and current original-URL candidate behavior.

- [ ] **Step 4: Add deterministic settings**

In `app/core/config.py` near the existing rich-item settings:

```python
rich_image_candidate_max_count: int = Field(
    default=8,
    description="Maximum normalized web-image candidates retained from one tool result.",
)
rich_image_min_width_px: int = Field(
    default=320,
    description="Reject provider images with a known width below this value.",
)
rich_image_min_height_px: int = Field(
    default=180,
    description="Reject provider images with a known height below this value.",
)
```

Add all three to the existing positive-integer validator and test zero/negative rejection.

- [ ] **Step 5: Implement Tavily toggle resolution and normalization**

Use `include_images: bool | None = None`, resolve it once, and pass the resolved value to both the SDK call and normalizer. The normalizer must:

1. Normalize string and object image entries.
2. Add `provider="tavily"`.
3. Add `source_url`, `source_title`, `source_domain`, `result_rank`, and `result_score` for `results[].images`.
4. Mark a top-level item without a publisher page as `query_level=True`.
5. Deduplicate by normalized image URL while preferring a result-bound record over a top-level record.
6. Return no normalized images when the resolved toggle is false, even if a test double/provider sends some.

The signature is:

```python
def _normalize_search_response(
    *, query: str, response: Any, include_images: bool
) -> dict[str, Any]:
```

- [ ] **Step 6: Prefer Brave thumbnails and filter candidates**

In `build_image_candidates_from_tool_result`, derive the display upstream before schema construction:

```python
provider = str(image.get("provider") or "").lower()
original_url = str(image.get("url") or "").strip()
thumbnail_url = str(image.get("thumbnail_url") or "").strip()
display_url = thumbnail_url if provider == "brave" and thumbnail_url else original_url
```

Reject remote `display_url` unless `urlsplit(display_url).scheme == "https"`; reject known dimensions below settings; deduplicate on `display_url`; stop at the configured cap. Put `width`/`height` in the public payload and keep provider-specific fields in provenance:

```python
provenance.update(
    {
        "provider": provider or ("tavily" if tool_name == "tavily_search" else "unknown"),
        "original_image_url": original_url or None,
        "thumbnail_url": thumbnail_url or None,
        "source_domain": image.get("source_domain"),
        "result_rank": image.get("result_rank"),
        "result_score": image.get("result_score"),
    }
)
```

Do not copy Tavily `description` into `payload.caption`; it is descriptive metadata, not a publisher caption.

- [ ] **Step 7: Make routing intent explicit without changing the global default**

Update the `tavily_search` docstring/tool description to say ordinary text research should pass `include_images=False`, while source-bound visual research passes `True`. Keep Brave's strict SafeSearch default and document it as the primary focused visual search. Do not introduce an automatic serial Brave→Tavily retry in either provider adapter.

- [ ] **Step 8: Run focused tests and commit**

Run the Step 3 command. Expected: PASS.

```powershell
git add app/ai/mcp_servers/tavily_server.py app/ai/mcp_servers/brave_image_search_server.py app/ai/tool_execution.py app/core/config.py tests/test_tavily_server.py tests/test_brave_image_search_server.py tests/test_tool_execution_rendering.py tests/test_rich_image_config.py
git commit -m "feat: normalize and filter web image candidates"
```

---

## Task 3: Add user-owned opaque web-image references

**Files:**
- Create: `app/models/web_image_reference.py`
- Modify: `app/models/__init__.py`
- Create: `app/repositories/web_image_reference.py`
- Create: `app/alembic/versions/f9a0b1c2d3e4_add_web_image_references.py`
- Modify: `tests/test_alembic_full_chain_postgres.py`
- Modify: `README.md`
- Test: `tests/test_web_image_reference_model.py`
- Test: `tests/test_web_image_reference_repository.py`

**Interfaces:**
- Table `web_image_references`: `id`, `conversation_id`, `user_id`, `upstream_url`, `expected_mime`, `provider`, `created_at`, `deleted_at`.
- `WebImageReferenceRepository.acreate(data)` and `.aget_for_user(image_id, user_id)` use `RepositorySessionMixin._arun`.
- No image bytes are stored in this table.

- [ ] **Step 1: Write failing model/repository tests**

```python
def test_web_image_reference_columns_are_bounded_and_owned():
    columns = WebImageReference.__table__.columns
    assert WebImageReference.__tablename__ == "web_image_references"
    assert columns["conversation_id"].nullable is False
    assert columns["user_id"].nullable is False
    assert columns["upstream_url"].type.length == 4096
    assert columns["provider"].type.length == 32
    assert columns["deleted_at"].nullable is True


@pytest.mark.asyncio
async def test_aget_for_user_uses_async_transport_and_filters_owner():
    expected = SimpleNamespace(id=uuid4())
    repo = WebImageReferenceRepository(_sync_factory_that_must_not_run(), _async_factory([expected]))
    assert await repo.aget_for_user(expected.id, uuid4()) is expected
```

Repository fake-session assertions must inspect the compiled statement or recorded criteria to confirm `id`, `user_id`, and `deleted_at IS NULL` are all present.

- [ ] **Step 2: Run tests and verify failure**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_web_image_reference_model.py tests/test_web_image_reference_repository.py -v
```

Expected: FAIL because the model and repository do not exist.

- [ ] **Step 3: Implement the model**

```python
class WebImageReference(Base):
    __tablename__ = "web_image_references"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    upstream_url = Column(String(4096), nullable=False)
    expected_mime = Column(String(128), nullable=True)
    provider = Column(String(32), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
```

Import it from `app/models/__init__.py` so metadata discovery includes the table.

- [ ] **Step 4: Implement async repository methods**

Extend `RepositorySessionMixin`. Use sync-style closures executed through `_arun`:

```python
async def acreate(self, data: dict[str, Any]) -> WebImageReference:
    def work(db: Session) -> WebImageReference:
        record = WebImageReference(**data)
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    return await self._arun(work)


async def aget_for_user(
    self, image_id: UUID, user_id: UUID
) -> WebImageReference | None:
    def work(db: Session) -> WebImageReference | None:
        statement = select(WebImageReference).where(
            WebImageReference.id == image_id,
            WebImageReference.user_id == user_id,
            WebImageReference.deleted_at.is_(None),
        )
        return db.execute(statement).scalars().first()
    return await self._arun(work)
```

- [ ] **Step 5: Add the exact Alembic revision and head checks**

Create revision `f9a0b1c2d3e4`, `down_revision = "e8f9a0b1c2d3"`, with the seven columns and indexes `ix_web_image_references_conversation_id` and `ix_web_image_references_user_id`. Downgrade drops indexes then table.

Update:

- `_HEAD = "f9a0b1c2d3e4"` and `_PREVIOUS_HEAD = "e8f9a0b1c2d3"` in `tests/test_alembic_full_chain_postgres.py`.
- README migration text to `currently f9a0b1c2d3e4`.
- Full-chain assertions to confirm the table exists at head and is absent after downgrade to `e8f9a0b1c2d3`.

- [ ] **Step 6: Run tests**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_web_image_reference_model.py tests/test_web_image_reference_repository.py -v
.venv\Scripts\python.exe -m pytest tests/test_alembic_full_chain_postgres.py -v
```

Expected: PASS. If the integration database is unavailable, record that environmental skip/failure; the model/repository unit tests and `alembic heads` must still pass.

- [ ] **Step 7: Commit**

```powershell
git add app/models/web_image_reference.py app/models/__init__.py app/repositories/web_image_reference.py app/alembic/versions/f9a0b1c2d3e4_add_web_image_references.py tests/test_web_image_reference_model.py tests/test_web_image_reference_repository.py tests/test_alembic_full_chain_postgres.py README.md
git commit -m "feat: persist owned web image references"
```

---

## Task 4: Build the DNS-pinned, bounded image fetch service

**Files:**
- Create: `app/services/web_image_service.py`
- Modify: `app/core/config.py`
- Test: `tests/test_web_image_service.py`
- Modify: `tests/test_rich_image_config.py`

**Interfaces:**
- `WebImageService.register(...) -> WebImageReference` persists only metadata and performs no network call.
- `WebImageService.fetch(record) -> FetchedWebImage` performs client-time secure retrieval.
- `FetchedWebImage(content: bytes, media_type: str, width: int, height: int)`.
- Typed failures: `WebImageRejected(reason)` and `WebImageUpstreamFailure(reason)`; reasons are bounded enums/strings safe for telemetry.

- [ ] **Step 1: Add failing latency and registration tests**

```python
@pytest.mark.asyncio
async def test_register_does_not_touch_network():
    repository = AsyncMock()
    transport_factory = Mock(side_effect=AssertionError("network must not run"))
    service = _service(repository=repository, transport_factory=transport_factory)
    await service.register(
        conversation_id=uuid4(), user_id=uuid4(),
        upstream_url="https://img.example/a.jpg",
        expected_mime="image/jpeg", provider="brave",
    )
    repository.acreate.assert_awaited_once()
    transport_factory.assert_not_called()
```

This is the regression test for answer latency: registration must be a DB write only.

- [ ] **Step 2: Add failing SSRF, redirect, and content-bound tests**

Use injected `resolver(hostname) -> list[str]` and `transport_factory(ip)` fakes. Cover:

- `http://` rejected before resolution.
- literal/private, loopback, link-local, multicast, reserved, and unspecified IPv4/IPv6 rejected.
- all DNS answers must be public; a mixed public/private response is rejected.
- the selected public IP is passed to a pinned transport.
- redirect target is re-resolved and re-pinned; redirect to private host is rejected.
- redirect cap produces `redirect_limit`.
- `Content-Length` over cap and streamed bytes over cap produce `size`.
- missing/non-raster MIME produces `mime`.
- decoded format mismatch, decompression-bomb/oversized dimensions, and corrupt bytes produce bounded rejection reasons.
- connect/read timeout produces `timeout`.
- a valid JPEG returns bytes, verified dimensions, and `image/jpeg`.

Representative tests:

```python
@pytest.mark.asyncio
async def test_redirect_to_private_address_is_rejected(png_bytes):
    resolver = AsyncMock(side_effect=[["93.184.216.34"], ["127.0.0.1"]])
    service = _service(
        resolver=resolver,
        responses=[_response(302, headers={"Location": "https://localhost/secret"})],
    )
    with pytest.raises(WebImageRejected, match="private_address"):
        await service.fetch(_record("https://public.example/image.png"))


@pytest.mark.asyncio
async def test_stream_stops_once_byte_cap_is_exceeded():
    service = _service(
        max_bytes=8,
        responses=[_streaming_response("image/png", [b"123456", b"789"])],
    )
    with pytest.raises(WebImageRejected, match="size"):
        await service.fetch(_record("https://img.example/image.png"))
```

- [ ] **Step 3: Run tests and verify failure**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_web_image_service.py tests/test_rich_image_config.py -v
```

Expected: FAIL because the service/settings do not exist.

- [ ] **Step 4: Add bounded delivery settings**

```python
web_image_fetch_connect_timeout_seconds: float = Field(default=2.0, gt=0)
web_image_fetch_read_timeout_seconds: float = Field(default=5.0, gt=0)
web_image_fetch_max_redirects: int = Field(default=3, ge=0, le=5)
web_image_fetch_max_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
web_image_fetch_max_pixels: int = Field(default=25_000_000, gt=0)
```

Use the established descriptive `Field` style and validators. Add matching `.env.example` entries with comments explaining that the fetch is render-time, not answer-time.

- [ ] **Step 5: Implement IP validation and DNS pinning**

Use `ipaddress.ip_address(value).is_global` and reject the whole hostname if any resolved address is non-global. The transport must connect to the already-validated IP while preserving HTTP `Host` and TLS SNI:

```python
class PinnedAsyncTransport(httpx.AsyncHTTPTransport):
    def __init__(self, verified_ip: str) -> None:
        self.verified_ip = verified_ip
        super().__init__(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        original_host = request.url.host
        request.url = request.url.copy_with(host=self.verified_ip)
        request.headers["Host"] = original_host
        request.extensions["sni_hostname"] = original_host
        return await super().handle_async_request(request)
```

The default resolver uses `asyncio.get_running_loop().getaddrinfo`. Tests inject deterministic answers. Do not honor `HTTP_PROXY`/`HTTPS_PROXY`; construct the client with `trust_env=False` so an environment proxy cannot bypass pinning.

- [ ] **Step 6: Implement manual redirect and bounded streaming**

For each hop:

1. Validate exact scheme `https` and hostname.
2. Resolve and validate every answer.
3. create a new pinned transport/client for that hop with `follow_redirects=False` and explicit connect/read timeouts.
4. Resolve `Location` with `urljoin`, then repeat validation.
5. Validate status, `Content-Type`, and `Content-Length` before iterating.
6. Accumulate `aiter_bytes()` only until `max_bytes`; abort immediately above it.

Accept only `image/png`, `image/jpeg`, `image/webp`, and `image/gif`. Use Pillow on `BytesIO(content)` to `verify()`, reopen to read dimensions/format, reject `width * height > max_pixels`, and map Pillow decompression warnings/errors to `dimensions` or `decode` without exposing URLs.

- [ ] **Step 7: Run focused tests and commit**

Run the Step 3 command. Expected: PASS.

```powershell
git add app/services/web_image_service.py app/core/config.py .env.example tests/test_web_image_service.py tests/test_rich_image_config.py
git commit -m "feat: securely fetch selected web images"
```

---

## Task 5: Expose the authenticated media route and bounded telemetry

**Files:**
- Create: `app/observability/rich_images.py`
- Create: `app/api/web_images.py`
- Modify: `app/api/__init__.py`
- Modify: `app/main.py`
- Modify: `app/api/health.py`
- Modify: `app/core/container.py`
- Test: `tests/test_web_images_api.py`
- Test: `tests/test_rich_image_metrics.py`
- Test: `tests/test_container_web_image_wiring.py`

**Interfaces:**
- Authenticated `GET /web-images/{image_id}`.
- `RichImageMetrics` exposes bounded counters/histograms only.
- `GET /metrics/rich-images` returns Prometheus text.

- [ ] **Step 1: Write failing API tests**

Build a small FastAPI test app with dependency overrides. Cover:

```python
def test_web_image_requires_auth(client):
    assert client.get(f"/web-images/{uuid4()}").status_code == 401


def test_web_image_hides_other_users_reference(authed_client, repository):
    repository.aget_for_user.return_value = None
    response = authed_client.get(f"/web-images/{uuid4()}")
    assert response.status_code == 404


def test_web_image_success_sets_safe_headers(authed_client, repository, service):
    repository.aget_for_user.return_value = _record()
    service.fetch.return_value = FetchedWebImage(
        content=PNG_BYTES, media_type="image/png", width=20, height=10
    )
    response = authed_client.get(f"/web-images/{uuid4()}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, max-age=300"
```

Assert upstream timeout maps to 504 and policy/upstream/decode failures map to 502 with a generic body. No response body may contain an upstream URL.

- [ ] **Step 2: Write failing metrics tests**

The permitted labels are fixed sets:

- discovery provider: `brave`, `tavily`, `other`;
- selection outcome: `selected`, `rejected`, `omitted`;
- fetch outcome: `success`, `timeout`, `status`, `mime`, `size`, `dimensions`, `decode`, `ssrf`, `other`.

Test that rendered metrics include counts/duration and never contain a sample URL, UUID, query, hostname, or user identifier.

- [ ] **Step 3: Run tests and verify failure**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_web_images_api.py tests/test_rich_image_metrics.py tests/test_container_web_image_wiring.py -v
```

Expected: FAIL because route, metrics, and wiring do not exist.

- [ ] **Step 4: Implement route behavior**

Mirror `app/api/chat_images.py`, but await the repository and fetch service:

```python
@router.get("/{image_id}")
async def read_web_image(
    image_id: UUID,
    current_user_id: UUID = Depends(get_current_user_id),
    repository: WebImageReferenceRepository = Depends(_get_repository),
    service: WebImageService = Depends(_get_service),
) -> Response:
    record = await repository.aget_for_user(image_id, current_user_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        image = await service.fetch(record)
    except WebImageUpstreamFailure as exc:
        raise HTTPException(status_code=504 if exc.reason == "timeout" else 502,
                            detail="Visual unavailable") from exc
    except WebImageRejected as exc:
        raise HTTPException(status_code=502, detail="Visual unavailable") from exc
    return Response(
        content=image.content,
        media_type=image.media_type,
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'",
        },
    )
```

- [ ] **Step 5: Wire repository, service, router, and metrics**

- Container repository receives both `db.provided.session` and `db.provided.async_session`.
- Service singleton receives repository and all five settings from Task 4.
- Export/mount `web_images_router` next to `chat_images_router`.
- `create_health_router` accepts an optional `rich_image_metrics` dependency and exposes `/metrics/rich-images` just like existing metrics endpoints.
- Record fetch duration and a bounded outcome in `finally`; do not add high-cardinality labels.

- [ ] **Step 6: Run focused tests and commit**

Run Step 3. Expected: PASS.

```powershell
git add app/observability/rich_images.py app/api/web_images.py app/api/__init__.py app/main.py app/api/health.py app/core/container.py tests/test_web_images_api.py tests/test_rich_image_metrics.py tests/test_container_web_image_wiring.py
git commit -m "feat: serve protected web images"
```

---

## Task 6: Externalize selected remote images without risking the answer

**Files:**
- Modify: `app/services/message_service.py`
- Modify: `app/services/event_streaming/ai_sdk_projection.py`
- Modify: `app/core/container.py`
- Test: `tests/test_message_service_web_image_externalization.py`
- Test: `tests/test_rich_response_streaming.py`
- Test: `tests/test_ai_sdk_v6_stream_contract.py`

**Interfaces:**
- Async helper `_externalize_remote_rich_images(content, metadata, conversation_id, user_id) -> tuple[str, dict]`.
- Every newly persisted selected absolute remote image becomes `/web-images/{id}`.
- Failed registration removes the rich item and every matching marker, then persists the rest of the response.
- AI SDK projection recognizes `/web-images/` as protected and emits each selected image once.

- [ ] **Step 1: Write failing success and no-network tests**

```python
@pytest.mark.asyncio
async def test_selected_remote_image_becomes_protected_reference():
    service = AsyncMock()
    service.register.return_value = SimpleNamespace(id=IMAGE_ID)
    message_service = _message_service(web_image_service=service)
    content, metadata = await message_service._externalize_remote_rich_images(
        "Intro\n\n<!--rich:image:tool:c1:0-->",
        _metadata_with_remote_image(), CONVERSATION_ID, USER_ID,
    )
    item = metadata["rich_items"][0]
    assert item["payload"]["url"] == f"/web-images/{IMAGE_ID}"
    service.register.assert_awaited_once_with(
        conversation_id=CONVERSATION_ID,
        user_id=USER_ID,
        upstream_url="https://img.example/a.jpg",
        expected_mime="image/jpeg",
        provider="tavily",
    )
    assert "<!--rich:image:tool:c1:0-->" in content
    service.fetch.assert_not_called()
```

- [ ] **Step 2: Write failing graceful-degradation tests**

```python
@pytest.mark.asyncio
async def test_registration_failure_drops_only_visual_and_marker(caplog):
    service = AsyncMock()
    service.register.side_effect = RuntimeError("db unavailable")
    message_service = _message_service(web_image_service=service)
    content, metadata = await message_service._externalize_remote_rich_images(
        "Before\n\n<!--rich:image:tool:c1:0-->\n\nAfter",
        _metadata_with_remote_image(), CONVERSATION_ID, USER_ID,
    )
    assert content == "Before\n\nAfter"
    assert metadata["rich_items"] == []
    assert "db unavailable" not in caplog.text
    assert "code=web_image_reference_failed" in caplog.text
```

Also cover multiple selected images, data URLs, `/chat-images/`, already externalized `/web-images/`, missing user ID, non-HTTPS URL, resume workflow, and ordinary completed workflow.

- [ ] **Step 3: Run tests and verify failure**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_message_service_web_image_externalization.py tests/test_rich_response_streaming.py tests/test_ai_sdk_v6_stream_contract.py -v
```

Expected: FAIL because the helper/service dependency and protected prefix support do not exist.

- [ ] **Step 4: Implement failure-safe externalization**

Add `web_image_service=None` to `MessageService.__init__` and container wiring. For each image rich item:

1. Skip inline data and protected relative URLs.
2. Reject non-HTTPS remote URLs by treating them as a registration failure.
3. Read provider from bounded provenance.
4. `await web_image_service.register(...)`.
5. Replace `payload.url` with `/web-images/{id}`. Keep the provider provenance created in Task 2 unchanged; do not add another duplicate URL field.
6. On any exception, remove the item and strip its exact marker with the existing rich marker parser/escaping helper.
7. Re-run rich-reference warnings so removed markers/items cannot leave stale warnings.

Do not mutate the provider result/candidate shared object in place; deep-copy `metadata["rich_items"]` before edits.

- [ ] **Step 5: Call the helper at both persistence boundaries**

In `resume_workflow` and `_persist_completed_workflow_response`, call it after `build_bot_metadata`/generated-image externalization and before `_acreate_bot_response_message`:

```python
bot_response_content, bot_metadata = await self._externalize_remote_rich_images(
    bot_response_content, bot_metadata, conversation_id, user_id
)
```

This awaits only DB reference creation; it never calls the upstream network.

- [ ] **Step 6: Extend protected URL projection and dedup tests**

Update `_is_protected_relative_image_url` to use `PROTECTED_IMAGE_URL_PREFIXES`. Assert AI SDK terminal output contains:

- one rich item at the marker;
- at most one matching file part for that URL;
- no raw upstream display URL in `payload.url` or a file part (provider provenance remains metadata);
- no image candidate in transient upserts.

Do not remove terminal file parts in this backend change; older AI SDK consumers may rely on them. The FE contract in Task 8 makes rich items authoritative and requires matching file-part deduplication.

- [ ] **Step 7: Run focused tests and commit**

Run Step 3. Expected: PASS.

```powershell
git add app/services/message_service.py app/services/event_streaming/ai_sdk_projection.py app/core/container.py tests/test_message_service_web_image_externalization.py tests/test_rich_response_streaming.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "feat: externalize selected web images"
```

---

## Task 7: Give the Streamlit demo one natural footer and full failure states

**Files:**
- Modify: `app/ui/rich_response.py`
- Modify: `demo.py`
- Test: `tests/test_demo_rich_response.py`
- Modify: `tests/test_demo_image_reference_rendering.py`

**Interfaces:**
- `build_inline_image_html(src, *, alt_text, caption, source_url, width, height)`.
- `build_inline_image_unavailable_html(*, source_url)` for a server-side protected-fetch failure.
- Both paths replace the whole figure, not only the `<img>`.

- [ ] **Step 1: Replace old caption expectations with failing ownership/state tests**

```python
def test_web_image_footer_is_source_not_alt_text():
    out = build_inline_image_html(
        "data:image/png;base64,QUJD",
        alt_text="A generated description",
        caption=None,
        source_url="https://publisher.example/story",
        width=800,
        height=450,
    )
    assert 'alt="A generated description"' in out
    assert "A generated description</figcaption>" not in out
    assert "publisher.example" in out
    assert "https://publisher.example/story" in out


def test_image_error_replaces_complete_figure():
    out = build_inline_image_html(
        "https://img.example/a.jpg", alt_text="Example", caption="Trusted caption",
        source_url="https://publisher.example/page", width=640, height=360,
    )
    assert "this.style.display='none'" not in out
    assert "Visual unavailable" in out
    assert "replaceChildren" in out


def test_loading_state_reserves_known_aspect_ratio():
    out = build_inline_image_html(
        "https://img.example/a.jpg", alt_text="Example", caption=None,
        source_url=None, width=640, height=360,
    )
    assert "aspect-ratio:640 / 360" in out
    assert 'data-state="loading"' in out
```

Also assert all URLs/text are HTML/JavaScript-safe, failed markup has no caption, and no source link appears when `source_url` is absent/invalid.

- [ ] **Step 2: Run tests and verify failure**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_demo_rich_response.py tests/test_demo_image_reference_rendering.py -v
```

Expected: FAIL because the old helper uses `title/alt` as a visible caption and hides only the image.

- [ ] **Step 3: Implement the pure three-state figure markup**

Use a unique DOM id derived from a hash of `src` rather than inserting the URL into JavaScript. The initial figure contains a quiet skeleton plus the image. `onload` removes the skeleton and sets `data-state="loaded"`. `onerror` calls a small inline function that replaces the figure children with prebuilt, escaped fallback markup:

```html
<div role="status" class="rich-image-unavailable">
  <span>Visual unavailable</span>
  <a href="https://publisher.example/page" target="_blank" rel="noopener noreferrer">Open source</a>
</div>
```

The fallback includes no caption and no large fixed-height region. Use a reserved `aspect-ratio` only while loading. Preserve natural width capped at 480px and the existing `img-thumb` lightbox class.

- [ ] **Step 4: Update demo resolution and rendering**

- Rename `_fetch_chat_image_data_uri` to `_fetch_protected_image_data_uri` and allow only paths beginning with `PROTECTED_IMAGE_URL_PREFIXES`; never concatenate arbitrary relative paths.
- Keep Bearer-token server-side fetch for both `/chat-images/` and `/web-images/`.
- If protected fetch returns `None`, render `build_inline_image_unavailable_html(source_url=payload.get("source_url"))`.
- Pass `payload.caption`, `item.alt_text`, `payload.source_url`, `payload.width`, and `payload.height` separately.
- Never derive visible caption from `item.title` or `alt_text`.
- Keep legacy gallery behavior unchanged for messages without `rich_items_version == 1`.

- [ ] **Step 5: Run focused tests and commit**

Run Step 2. Expected: PASS.

```powershell
git add app/ui/rich_response.py demo.py tests/test_demo_rich_response.py tests/test_demo_image_reference_rendering.py
git commit -m "fix: render resilient single-footer rich images"
```

---

## Task 8: Publish the frontend contract, telemetry, and rollout checks

**Files:**
- Modify: `README.md`
- Create: `docs/frontend/rich-image-rendering.md`
- Modify: `tests/test_article_image_flow.py`
- Modify: `tests/test_rich_response_metadata.py`
- Modify: `tests/test_ai_sdk_v6_stream_contract.py`

**Interfaces:**
- FE teams receive an implementation-level protected fetch and dedup contract.
- End-to-end tests prove image failure cannot fail the answer.
- No dataset/evaluation stage is introduced.

- [ ] **Step 1: Write failing integration assertions**

Add a narrow end-to-end fixture that constructs a tool result, selects the candidate, persists a protected reference, and projects terminal AI SDK output. Assert:

1. the assistant content and marker persist;
2. metadata has one selected image and no candidate dump;
3. preferred URL is `/web-images/{id}`;
4. publisher page remains in `payload.source_url`;
5. no generated Markdown caption follows the marker;
6. registration failure produces a complete text answer with neither item nor marker;
7. media-route timeout happens only in the later route test and does not change the stored assistant message.

- [ ] **Step 2: Run the integration subset and verify failure**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_article_image_flow.py tests/test_rich_response_metadata.py tests/test_ai_sdk_v6_stream_contract.py -v
```

Expected: FAIL until the complete contract is wired.

- [ ] **Step 3: Document the browser frontend algorithm**

`docs/frontend/rich-image-rendering.md` must provide executable TypeScript-shaped guidance with no framework-specific dependency:

```typescript
async function loadProtectedImage(url: string, token: string): Promise<string> {
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
    credentials: "same-origin",
  });
  if (!response.ok) throw new Error("visual_unavailable");
  const blob = await response.blob();
  if (!blob.type.startsWith("image/")) throw new Error("visual_unavailable");
  return URL.createObjectURL(blob);
}
```

Require cleanup with `URL.revokeObjectURL`, and define:

- render marker-resolved `rich_items` in marker order;
- v1 rich-item images are authoritative;
- deduplicate terminal `file` parts whose URL matches a rich-item `payload.url`;
- never put a protected URL directly in `<img src>` because Bearer auth cannot be attached;
- loading reserves `width / height` when available;
- failed replaces the entire figure with `Visual unavailable`, optional `Open source`, and optional Retry;
- alt text is accessibility-only; visible footer is optional structured caption plus source attribution;
- client network failure must remain local to the visual.

- [ ] **Step 4: Document provider and latency policy in README**

Update the rich-response section with:

- Brave primary for focused visual discovery; Tavily for web research/source-bound images and fallback.
- `include_images=None` preserves config default; explicit false is per-call and intentionally returns no images for that call.
- no model/image evaluation or labeled dataset requirement.
- reference creation is persistence-time DB-only; media bytes are render-time.
- metrics endpoint and bounded labels.
- `INLINE_RICH_RESPONSE_ENABLED` remains rollback switch.

Do not claim that Brave is universally “better” than Tavily: document that they solve different retrieval jobs and that Brave's thumbnail metadata makes it the preferred focused-image adapter.

- [ ] **Step 5: Run integration tests and the complete regression suite**

Focused tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_article_image_flow.py tests/test_rich_response_metadata.py tests/test_ai_sdk_v6_stream_contract.py -v
```

Full feature regression:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_tavily_server.py tests/test_brave_image_search_server.py tests/test_tool_execution_rendering.py tests/test_rich_image_config.py tests/test_rich_response_contract.py tests/test_rich_response_prompt_inventory.py tests/test_rich_response_metadata.py tests/test_rich_placement.py tests/test_demo_rich_response.py tests/test_demo_image_reference_rendering.py tests/test_rich_response_streaming.py tests/test_ai_sdk_v6_stream_contract.py tests/test_article_image_flow.py tests/test_web_image_reference_model.py tests/test_web_image_reference_repository.py tests/test_web_image_service.py tests/test_web_images_api.py tests/test_rich_image_metrics.py tests/test_message_service_web_image_externalization.py -v
```

Expected: PASS.

- [ ] **Step 6: Run static and migration verification**

```powershell
.venv\Scripts\python.exe -m ruff check app/core/rich_response.py app/ai/prompts.py app/ai/mcp_servers/tavily_server.py app/ai/mcp_servers/brave_image_search_server.py app/ai/tool_execution.py app/models/web_image_reference.py app/repositories/web_image_reference.py app/services/web_image_service.py app/services/message_service.py app/api/web_images.py app/api/health.py app/observability/rich_images.py app/services/event_streaming/ai_sdk_projection.py app/ui/rich_response.py demo.py
.venv\Scripts\python.exe -m alembic -c alembic.ini heads
git diff --check
```

Expected: no new Ruff errors in touched files; one Alembic head `f9a0b1c2d3e4`; `git diff --check` emits no output.

- [ ] **Step 7: Commit documentation and integration coverage**

```powershell
git add README.md docs/frontend/rich-image-rendering.md tests/test_article_image_flow.py tests/test_rich_response_metadata.py tests/test_ai_sdk_v6_stream_contract.py
git commit -m "docs: publish rich image frontend contract"
```

---

## Production Acceptance Checklist

- [ ] A text-only Tavily call with `include_images=False` returns no images, while an omitted value still follows deployment config.
- [ ] Focused Brave results display the Brave thumbnail through `/web-images/{id}` and retain the publisher page/original image only as metadata.
- [ ] A web image produces one visible footer, never an agent Markdown caption plus renderer caption.
- [ ] `alt_text` is present on `<img>` but is not automatically visible.
- [ ] A blocked, expired, oversized, corrupt, or non-image upstream response changes only the figure to `Visual unavailable`.
- [ ] A reference-registration failure persists a valid text answer with no dangling marker.
- [ ] No upstream request occurs before assistant message persistence.
- [ ] Private/loopback/link-local/reserved redirect targets cannot be reached, including DNS-rebinding cases.
- [ ] Browser clients authenticate protected media with `fetch` + object URL and revoke object URLs.
- [ ] AI SDK clients do not render both a marker image and matching file-part gallery image.
- [ ] Metrics contain bounded outcomes and durations, never URLs, queries, IDs, hostnames, or bytes.
- [ ] No evaluation dataset, vision reranker, or added model call exists in the final diff.
