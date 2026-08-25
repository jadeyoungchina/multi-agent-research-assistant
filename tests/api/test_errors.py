from typing import Annotated

import pytest
from fastapi import Query

from app.domain.errors import DocumentError, ProviderError, RetrievalError, WorkflowError


def _install_error_route(client, exception: Exception) -> None:
    async def raise_error() -> None:
        raise exception

    client.app.add_api_route("/test-error", raise_error, methods=["GET"])


def test_unknown_route_uses_stable_error_shape(client) -> None:
    response = client.get("/missing")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Resource not found",
            "request_id": response.headers["X-Request-ID"],
            "details": {},
        }
    }


@pytest.mark.parametrize(
    ("exception", "status_code"),
    [
        (DocumentError("invalid_document", "document rejected"), 400),
        (RetrievalError("invalid_query", "query rejected"), 400),
        (WorkflowError("run_not_found", "research run does not exist"), 404),
        (ProviderError("provider_timeout", "provider unavailable"), 502),
        (ProviderError("model_not_found", "provider model unavailable"), 502),
    ],
)
def test_domain_errors_use_stable_envelope(client, exception, status_code: int) -> None:
    _install_error_route(client, exception)

    response = client.get("/test-error")

    assert response.status_code == status_code
    assert response.json() == {
        "error": {
            "code": exception.code,
            "message": str(exception),
            "request_id": response.headers["X-Request-ID"],
            "details": {},
        }
    }


def test_request_validation_omits_raw_input(client) -> None:
    async def validated(count: Annotated[int, Query(gt=0)]) -> None:
        return None

    client.app.add_api_route("/validated", validated, methods=["GET"])

    response = client.get("/validated?count=not-a-number")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert response.json()["error"]["message"] == "Request validation failed"
    assert "not-a-number" not in response.text


def test_unexpected_error_does_not_leak_exception_text(client) -> None:
    _install_error_route(client, RuntimeError("sentinel-secret-value"))

    response = client.get("/test-error")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "Internal server error",
            "request_id": response.headers["X-Request-ID"],
            "details": {},
        }
    }
    assert "sentinel-secret-value" not in response.text
