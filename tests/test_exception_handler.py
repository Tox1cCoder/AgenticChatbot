from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import CustomHTTPException
from app.schemas.responses import ApiResponse
from app.utils.exception_handler import register_exception_handlers


def test_custom_http_error_envelope_retains_domain_code():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/conflict")
    async def conflict():
        raise CustomHTTPException(409, "Already resolved", "INTERRUPT_ALREADY_RESOLVED")

    response = TestClient(app).get("/conflict")

    assert response.json() == {
        "success": False,
        "code": "INTERRUPT_ALREADY_RESOLVED",
        "message": "Already resolved",
    }


def test_custom_http_error_without_domain_code_omits_code():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/no-domain-code")
    async def no_domain_code():
        raise CustomHTTPException(409, "No domain code")

    response = TestClient(app).get("/no-domain-code")

    assert response.json() == {
        "success": False,
        "message": "No domain code",
    }


def test_code_less_response_model_omits_code():
    app = FastAPI()

    @app.get("/success", response_model=ApiResponse)
    async def success():
        return ApiResponse(success=True, message="OK")

    response = TestClient(app).get("/success")

    assert response.json() == {
        "success": True,
        "message": "OK",
        "data": None,
        "error": None,
    }
