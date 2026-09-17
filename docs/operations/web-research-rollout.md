# Web research rollout

Canonical web research is controlled by `WEB_RESEARCH_ENABLED`. Keep the flag
enabled only after the focused web matrix, non-live suite, and deterministic
evaluation pass.

The answer model receives only MIME-validated, dimension-checked image bytes,
downscaled to a 512px preview for selection; the full-size rendition is what
gets published. Candidate metadata is private. A web image is published only
when the model selects its `I#` token; no token means no image. Web-answer text
is buffered until a citation resolves to a source admitted in that turn.

Image source pages hold registry capacity of their own, on top of the text
source budget: a turn admits up to 5 text sources (8 agentic) plus up to 4
image pages (6 for `visual_intent="gallery"`). They shared one budget until
2026-09-16, and because text results are admitted first and a search returns
its full quota, every image was discarded before the model saw it.

Monitor bounded operation outcomes together with the existing rich-image,
model, and routing telemetry. Do not add
queries, URLs, titles, user IDs, tenant IDs, or exception text as metric labels.
Scrape `/metrics/web-research`. The existing ten-minute expiry cleanup worker releases
expired pending references; keep its cadence shorter than the configured TTL
and alert when it stops running.

Release check:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_web_research_contracts.py `
  tests/test_web_source_registry.py `
  tests/test_web_research_policy.py `
  tests/test_web_research_providers.py `
  tests/test_web_research_service.py `
  tests/test_web_research_images.py `
  tests/test_web_image_capacity_and_resolution.py `
  tests/test_web_research_tool_session.py `
  tests/test_web_research_model_context.py `
  tests/test_web_grounding.py `
  tests/test_web_source_streaming.py `
  tests/test_web_tool_output_privacy.py `
  tests/test_web_research_output_policy.py `
  tests/test_required_web_streaming.py `
  tests/test_web_research_continuation.py `
  tests/test_web_research_worker_remap.py `
  tests/test_web_research_active_path_inventory.py `
  tests/test_web_research_config.py `
  tests/test_web_research_metrics.py `
  tests/test_web_research_evaluation.py `
  -q -p no:cacheprovider

# Supplemental deterministic contract scorer; not an end-to-end substitute.
.venv\Scripts\python.exe scripts/evaluate_web_research.py `
  --cases eval/web_research/cases.json `
  --output output/audits/web-research-eval.json
```

Rollback by disabling `WEB_RESEARCH_ENABLED`; raw Tavily and Brave tools remain
excluded from ordinary agent binding.
