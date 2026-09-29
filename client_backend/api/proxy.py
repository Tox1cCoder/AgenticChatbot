"""
Compatibility proxy routes for server-owned API surfaces.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from client_backend.api.common import proxy_server_request, rewrite_widget_ws_url
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.services.runtime_bridge import get_runtime_bridge

router = APIRouter(tags=["proxy"])

# Multi-method paths are declared with one stacked decorator per method rather
# than ``api_route(methods=[...])``: FastAPI derives one operation id per route,
# so a multi-method route published the same id for every method and the OpenAPI
# build warned about each one. Paths and handlers are unchanged.


def _params_with_active_device(request: Request):
    params = [
        (key, value)
        for key, value in request.query_params.multi_items()
        if key not in {"deviceId", "device_id"}
    ]
    device_id = get_runtime_bridge().get_registered_device_id()
    if device_id:
        params.append(("deviceId", device_id))
    return params


@router.get("/usage/capabilities")
async def proxy_usage_capabilities(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/usage/capabilities")


@router.get("/usage/dashboard")
async def proxy_usage_dashboard(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/usage/dashboard")


@router.get("/usage/conversations/{conversation_id}")
async def proxy_conversation_usage(
    conversation_id: UUID,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/usage/conversations/{conversation_id}",
    )


@router.get("/users/{user_id}")
async def proxy_user(
    user_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path=f"/users/{user_id}")


@router.get("/providers")
@router.post("/providers")
async def proxy_providers(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/providers")


@router.get("/hitl/settings")
@router.post("/hitl/settings")
@router.delete("/hitl/settings")
async def proxy_hitl_settings(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path="/hitl/settings",
        params_override=_params_with_active_device(request),
    )


@router.get("/hitl/interrupts/{interrupt_id}")
async def proxy_hitl_interrupt(
    interrupt_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/hitl/interrupts/{interrupt_id}",
    )


@router.get("/providers/{provider_type}")
@router.delete("/providers/{provider_type}")
async def proxy_provider(
    provider_type: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path=f"/providers/{provider_type}")


@router.post("/providers/{provider_type}/validate")
async def proxy_provider_validate(
    provider_type: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/providers/{provider_type}/validate",
    )


@router.get("/providers/{provider_type}/models")
async def proxy_provider_models(
    provider_type: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/providers/{provider_type}/models",
    )


@router.get("/model-config")
@router.patch("/model-config")
async def proxy_model_config(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/model-config")


@router.get("/model-config/options")
async def proxy_model_config_options(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/model-config/options")


@router.post("/model-config/reset")
async def proxy_model_config_reset(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/model-config/reset")


@router.get("/custom-agents")
@router.post("/custom-agents")
async def proxy_custom_agents(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path="/custom-agents",
        params_override=_params_with_active_device(request),
    )


@router.get("/custom-agents/options")
async def proxy_custom_agents_options(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path="/custom-agents/options",
        params_override=_params_with_active_device(request),
    )


@router.get("/custom-agents/{custom_agent_id}")
@router.patch("/custom-agents/{custom_agent_id}")
@router.delete("/custom-agents/{custom_agent_id}")
async def proxy_custom_agent(
    custom_agent_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/custom-agents/{custom_agent_id}",
        params_override=_params_with_active_device(request),
    )


@router.get("/ai/custom-agents")
@router.post("/ai/custom-agents")
async def proxy_ai_custom_agents(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path="/ai/custom-agents",
        params_override=_params_with_active_device(request),
    )


@router.get("/ai/custom-agents/options")
async def proxy_ai_custom_agents_options(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path="/ai/custom-agents/options",
        params_override=_params_with_active_device(request),
    )


@router.get("/ai/custom-agents/{custom_agent_id}")
@router.patch("/ai/custom-agents/{custom_agent_id}")
@router.delete("/ai/custom-agents/{custom_agent_id}")
async def proxy_ai_custom_agent(
    custom_agent_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/ai/custom-agents/{custom_agent_id}",
        params_override=_params_with_active_device(request),
    )


@router.get("/conversations/{conversation_id}/custom-agents")
@router.put("/conversations/{conversation_id}/custom-agents")
async def proxy_conversation_custom_agents(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/conversations/{conversation_id}/custom-agents",
    )


@router.get("/ai/conversations/{conversation_id}/custom-agents")
@router.put("/ai/conversations/{conversation_id}/custom-agents")
async def proxy_ai_conversation_custom_agents(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/ai/conversations/{conversation_id}/custom-agents",
    )


@router.get("/messages/{message_id}/feedbacks")
@router.post("/messages/{message_id}/feedbacks")
async def proxy_message_feedbacks(
    message_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path=f"/messages/{message_id}/feedbacks")


@router.get("/messages/{message_id}/feedbacks/stats")
async def proxy_message_feedback_stats(
    message_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/messages/{message_id}/feedbacks/stats",
    )


@router.get("/messages/{message_id}/feedbacks/user")
async def proxy_message_feedback_user(
    message_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/messages/{message_id}/feedbacks/user",
    )


@router.put("/messages/{message_id}/feedbacks/{feedback_id}")
async def proxy_message_feedback_update(
    message_id: str,
    feedback_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/messages/{message_id}/feedbacks/{feedback_id}",
    )


@router.get("/conversations/{conversation_id}/task-plans")
@router.post("/conversations/{conversation_id}/task-plans")
async def proxy_conversation_task_plans(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/conversations/{conversation_id}/task-plans",
    )


@router.post("/conversations/{conversation_id}/task-plans/manual")
async def proxy_conversation_task_plans_manual(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/conversations/{conversation_id}/task-plans/manual",
    )


@router.get("/conversations/{conversation_id}/planning-status")
async def proxy_conversation_planning_status(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/conversations/{conversation_id}/planning-status",
    )


@router.get("/task-plans/{task_id}")
@router.patch("/task-plans/{task_id}")
@router.delete("/task-plans/{task_id}")
async def proxy_task_plan(
    task_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path=f"/task-plans/{task_id}")


@router.post("/task-plans/{task_id}/complete")
async def proxy_task_plan_complete(
    task_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path=f"/task-plans/{task_id}/complete")


@router.get("/ai/conversations")
@router.post("/ai/conversations")
async def proxy_ai_conversations(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/ai/conversations")


@router.get("/ai/conversations/{conversation_id}")
@router.patch("/ai/conversations/{conversation_id}")
@router.delete("/ai/conversations/{conversation_id}")
async def proxy_ai_conversation(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path=f"/ai/conversations/{conversation_id}")


@router.get("/ai/conversations/{conversation_id}/messages")
async def proxy_ai_conversation_messages(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/ai/conversations/{conversation_id}/messages",
    )


# ── Widget connection proxy ───────────────────────────────────────────────
# Phase 1: proxy the widget connection minting endpoint so the frontend
# can obtain a signed widget token through client_backend.
# The returned ws_url points to the canonical server — no local WebSocket
# relay in this phase.


@router.post("/widgets/{widget_id}/connection")
async def proxy_widget_connection(
    widget_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/widgets/{widget_id}/connection",
        json_transform=rewrite_widget_ws_url,
    )


@router.post("/widgets/{widget_id}/actions/{action_key}")
async def proxy_widget_action(
    widget_id: str,
    action_key: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/widgets/{widget_id}/actions/{action_key}",
    )
