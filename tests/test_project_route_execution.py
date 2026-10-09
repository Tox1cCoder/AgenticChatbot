"""Synchronous project service calls must not run on the HTTP event loop."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.projects import router
from app.core.auth import get_current_user_id
from app.repositories.utils.pagination import Paginator


def test_project_list_service_runs_outside_event_loop():
    running_loop = []

    def list_projects(*_args, **_kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            running_loop.append(False)
        else:
            running_loop.append(True)
        return Paginator.create([], 0, 1, 10)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user_id] = uuid4
    route = next(route for route in router.routes if route.path == "/projects")
    dependency = next(
        dep.call for dep in route.dependant.dependencies if dep.name == "project_service"
    )
    app.dependency_overrides[dependency] = lambda: SimpleNamespace(list_projects=list_projects)
    with TestClient(app) as client:
        response = client.get("/projects")
    assert response.status_code == 200
    assert running_loop == [False], "project database work blocks the HTTP event loop"
