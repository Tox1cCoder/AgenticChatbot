"""Projects OpenAPI must describe the wire contract consumed by the frontend."""

from fastapi import FastAPI

from app.api.projects import router as projects_router
from client_backend.main import create_app as create_sidecar_app


def test_server_projects_openapi_describes_errors_and_non_nullable_patch_name():
    app = FastAPI()
    app.include_router(projects_router)
    schema = app.openapi()

    read_responses = schema["paths"]["/projects/{project_id}"]["get"]["responses"]
    assert {"401", "403", "404", "422"} <= read_responses.keys()
    assert read_responses["404"]["content"]["application/json"]["schema"]

    patch_name = schema["components"]["schemas"]["ProjectUpdate"]["properties"]["name"]
    assert patch_name["type"] == "string"
    assert "anyOf" not in patch_name
    assert "default" not in patch_name


def test_sidecar_projects_openapi_describes_request_and_response_shapes():
    schema = create_sidecar_app().openapi()
    paths = schema["paths"]

    list_operation = paths["/projects"]["get"]
    assert {"page", "limit", "search"} <= {
        parameter["name"] for parameter in list_operation["parameters"]
    }
    list_schema = list_operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert list_schema["$ref"].endswith("PaginatedApiResponse_ProjectRead_")

    create_schema = paths["/projects"]["post"]["requestBody"]["content"]["application/json"][
        "schema"
    ]
    assert "name" in create_schema["required"]
    assert create_schema["properties"]["name"]["maxLength"] == 255

    agents_schema = paths["/projects/{project_id}/custom-agents"]["put"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert "customAgentIds" in agents_schema["properties"]

    patch_schema = paths["/projects/{project_id}"]["patch"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert "name" in patch_schema["properties"]
