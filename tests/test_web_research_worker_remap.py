from app.ai.workflow.contracts import WorkerResult
from app.ai.workflow.planning_execution import (
    PlanningLimits,
    build_planning_outcome,
    render_worker_results,
)


def test_planning_preserves_two_workers_distinct_public_web_records() -> None:
    results = [
        WorkerResult(
            dispatch_id="dispatch",
            task_id=f"task-{index}",
            position=index,
            agent_id="search_agent",
            status="completed",
            content=f"Worker {index} [1](https://example.test/{index}).",
            evidence=(
                {
                    "source_id": "S1",
                    "url": f"https://example.test/{index}",
                    "title": f"Source {index}",
                },
            ),
        )
        for index in (1, 2)
    ]

    outcome = build_planning_outcome(content="Synthesis", results=results)

    assert [source["url"] for source in outcome.provenance.web_sources] == [
        "https://example.test/1",
        "https://example.test/2",
    ]
    assert [source["source_id"] for source in outcome.provenance.web_sources] == ["S1", "S2"]
    assert outcome.response.metadata["web_sources_version"] == 1


def test_planning_does_not_publish_or_relay_worker_image_markers() -> None:
    image = {
        "id": "image:web:worker-one",
        "type": "image",
        "payload": {"url": "/web-images/worker-one"},
    }
    result = WorkerResult(
        dispatch_id="dispatch",
        task_id="task-1",
        position=1,
        agent_id="search_agent",
        status="completed",
        content="Worker finding <!--rich:image:web:worker-one-->",
        images=(image,),
    )

    rendered = render_worker_results(
        "research",
        [result],
        PlanningLimits(
            max_tasks=8,
            max_concurrency=4,
            objective_max_chars=4000,
            parent_context_max_chars=12000,
        ),
    )
    outcome = build_planning_outcome(content="Synthesis", results=[result])

    assert "rich:image" not in rendered
    assert outcome.provenance.rich_items == ()
    assert "_rich_item_candidates" not in outcome.response.metadata
