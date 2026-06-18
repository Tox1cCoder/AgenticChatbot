"""
Compatibility proxy routes for server-owned API surfaces.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from client_backend.api.common import proxy_server_request, rewrite_widget_ws_url
from client_backend.core.auth import require_local_session
from client_backend.core.security import LocalSessionPayload
from client_backend.services.runtime_bridge import get_runtime_bridge

router = APIRouter(tags=["proxy"])


def _params_with_active_device(request: Request):
    params = list(request.query_params.multi_items())
    if any(key == "deviceId" for key, _value in params):
        return params
    device_id = get_runtime_bridge().get_registered_device_id()
    if device_id:
        params.append(("deviceId", device_id))
    return params


@router.get("/users/{user_id}")
async def proxy_user(
    user_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path=f"/users/{user_id}")


@router.api_route("/providers", methods=["GET", "POST"])
async def proxy_providers(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/providers")


@router.api_route("/providers/{provider_type}", methods=["GET", "DELETE"])
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


@router.api_route("/model-config", methods=["GET", "PATCH"])
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


@router.api_route("/custom-agents", methods=["GET", "POST"])
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


@router.api_route("/custom-agents/{custom_agent_id}", methods=["GET", "PATCH", "DELETE"])
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


@router.api_route("/ai/custom-agents", methods=["GET", "POST"])
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


@router.api_route("/ai/custom-agents/{custom_agent_id}", methods=["GET", "PATCH", "DELETE"])
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


@router.api_route("/conversations/{conversation_id}/custom-agents", methods=["GET", "PUT"])
async def proxy_conversation_custom_agents(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/conversations/{conversation_id}/custom-agents",
    )


@router.api_route("/ai/conversations/{conversation_id}/custom-agents", methods=["GET", "PUT"])
async def proxy_ai_conversation_custom_agents(
    conversation_id: str,
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(
        request,
        upstream_path=f"/ai/conversations/{conversation_id}/custom-agents",
    )


@router.api_route("/messages/{message_id}/feedbacks", methods=["GET", "POST"])
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


@router.api_route("/conversations/{conversation_id}/task-plans", methods=["GET", "POST"])
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


@router.api_route("/task-plans/{task_id}", methods=["GET", "PATCH", "DELETE"])
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


@router.api_route("/ai/conversations", methods=["GET", "POST"])
async def proxy_ai_conversations(
    request: Request,
    _session: LocalSessionPayload = Depends(require_local_session),
) -> Response:
    return await proxy_server_request(request, upstream_path="/ai/conversations")


@router.api_route("/ai/conversations/{conversation_id}", methods=["GET", "PATCH", "DELETE"])
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
