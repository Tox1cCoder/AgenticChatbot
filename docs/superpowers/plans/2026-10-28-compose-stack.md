# One-command Docker Stack Implementation Plan

> **For agentic workers:** Execute inline using superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Start every local chatbot service with `docker compose up -d --build`.

**Architecture:** Build one non-root Linux image from the project's declared
dependencies, with CPU document parsing and Node.js. Compose coordinates
health-gated services and named storage; the demo shares loopback networking
and the launch-token profile with the sidecar.

**Tech Stack:** Docker Compose, Python 3.12, PostgreSQL 17, Redis 7.4,
Qdrant, Celery, MinerU 3.4.4, Streamlit, Node.js 22.

## Global constraints

- Preserve unrelated work already present in the checkout.
- Keep the existing Redis-only workflow and existing containers/data intact.
- Retain sidecar launch-token and Host/Origin checks.
- All default services must start without a GPU or a mandatory dotenv file.
- Provider-backed requests require configured provider credentials.

## Task 1: Build and orchestrate the full stack

**Files:** Create `Dockerfile`, `.dockerignore`, and `compose.yaml`.
Modify `pyproject.toml` to request SQLAlchemy's asyncio extra: the fresh
Linux import reproduced a missing `greenlet` error with SQLAlchemy 2.1.

**Interfaces:** Consume `pyproject.toml`, `demo_requirements.txt`, existing
FastAPI/Celery entrypoints, and MinerU's `--api-url` CLI interface. Produce a
default Compose application with persistent named volumes.

- [x] Demonstrate the missing behavior with `docker compose config --services`.
- [x] Add the CPU runtime image and exclude credentials, uploads, local
  environments, vendored binaries, and git state from the build context.
- [x] Define infrastructure, API, queue workers, scheduler, MinerU, sidecar,
  and demo with dependency health checks and persistent data paths.
- [x] Run `docker compose config --quiet` and `docker compose build`.
- [x] Start an isolated test project using unused host ports, inspect service
  health, and exercise real HTTP and Celery interfaces. Resolve startup failures
  from their logs and repeat only affected checks.

## Task 2: Document and verify the supported workflow

**Files:** Modify `README.md` and `.env.example` where container settings
require explanation. Update this plan's completed checkboxes.

**Interfaces:** Consume the verified Compose commands and published endpoints;
produce a short quick-start above the existing native-development instructions.

- [x] Document the single startup command, UI/API URLs, optional credentials,
  container workspace behavior, first model download, and data persistence.
- [x] Verify configuration against an empty environment as well as the user's
  dotenv configuration without printing secrets.
- [x] Run `git diff --check`, review only task-related diffs, and stop isolated
  test containers without deleting volumes or touching existing services.
- [x] Report changed files, validation evidence, and any remaining limitations.


## Verification results (2026-10-28)

- `docker compose config --quiet` passes with the user's dotenv inputs.
- An isolated project with no dotenv files resolves all 11 services and starts
  successfully. Its first startup reproduced a missing SQLAlchemy `greenlet`
  dependency; requesting the asyncio extra fixes the same import in a fresh image.
- `docker compose up -d --build --wait` successfully builds the shared image and
  brings up the entire isolated application. All 10 healthchecked services are
  healthy; the single Beat scheduler is running.
- Tests use ports 18000/18100/18501, exercising published port overrides,
  browser Origins, and the separate browser WebSocket address. Host HTTP probes
  pass for API aggregate health, sidecar health, and Streamlit.
- Signup/login through the demo's real `SidecarSession`, authenticated
  conversation listing, browser widget connection minting, and authenticated
  WebSocket state sync pass. Missing launch tokens return 401; an untrusted Host
  returns 400. The demo reads the token from its read-only profile volume.
- The default Gemini embedding adapter requires a configured key even for
  conversation endpoints. A deliberately nonfunctional key is set only in the
  temporary test override for configuration-dependent endpoint tests; no model
  inference or real provider API call is tested. README documents the requirement.
- Celery inspect confirms parse, index, summary, and default maintenance queues
  have consumers. Beat starts with its persisted schedule file.
- A real DOCX parses from the parse worker through the persistent MinerU API,
  and the returned Markdown contains the input text. CPU PyTorch is confirmed.
  PDF OCR/model inference is not tested and downloads weights on first use.
- The sidecar's persistent workspace is writable and allowed by the existing
  file-tool guard, while launch-token credential paths remain protected.
- Qdrant is pinned to 1.20.0 to match the resolved 1.20 client; aggregate
  `/health/all` reports all dependencies healthy.
- The image runs as UID 10001 and excludes dotenv files and host-vendored MCP
  binaries. Scoped `git diff --check` and the legacy Redis-only Compose
  validation pass. Unrelated checkout edits remain intact.
- Isolated test containers are stopped without deleting named volumes. Existing
  native-development Redis and Qdrant containers are preserved.
