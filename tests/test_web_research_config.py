from app.core.config import Settings


def test_web_research_has_structural_limits_without_search_call_ceiling() -> None:
    fields = Settings.model_fields

    assert fields["web_research_max_candidate_pool"].default == 8
    assert fields["web_research_max_download_bytes"].default == 20 * 1024 * 1024
    assert fields["web_research_max_model_image_bytes"].default == 8 * 1024 * 1024
    assert fields["web_research_image_concurrency"].default == 3
    assert fields["web_research_pending_image_ttl_seconds"].default == 900
    assert "research_max_search_calls_per_turn" not in fields
    assert "web_research_total_deadline_seconds" not in fields
