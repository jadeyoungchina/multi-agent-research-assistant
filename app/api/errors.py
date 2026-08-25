import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.schemas import ApiErrorBody, ApiErrorResponse
from app.domain.errors import (
    DocumentError,
    DomainError,
    ProviderError,
    RetrievalError,
    WorkflowError,
)
from app.observability import install_redaction_filter, log_event


logger = logging.getLogger(__name__)
install_redaction_filter(logger)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body = ApiErrorResponse(
        error=ApiErrorBody(
            code=code,
            message=message,
            request_id=_request_id(request),
            details=details or {},
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _domain_status(exception: DomainError) -> int:
    if isinstance(exception, ProviderError):
        return 502
    if exception.code.endswith("_not_found"):
        return 404
    if "too_large" in exception.code:
        return 413
    if isinstance(exception, (DocumentError, RetrievalError, WorkflowError)):
        return 400
    return 400


async def domain_error_handler(
    request: Request, exception: DomainError
) -> JSONResponse:
    return error_response(
        request,
        _domain_status(exception),
        exception.code,
        str(exception),
    )


async def validation_error_handler(
    request: Request, exception: RequestValidationError
) -> JSONResponse:
    details = {
        "errors": [
            {
                "location": list(error.get("loc", ())),
                "message": error.get("msg", "Invalid value"),
                "type": error.get("type", "value_error"),
            }
            for error in exception.errors()
        ]
    }
    return error_response(
        request,
        422,
        "validation_error",
        "Request validation failed",
        details,
    )


async def http_error_handler(
    request: Request, exception: HTTPException
) -> JSONResponse:
    if exception.status_code == 404:
        return error_response(request, 404, "not_found", "Resource not found")
    if isinstance(exception.detail, dict):
        code = str(exception.detail.get("code", "http_error"))
        message = str(exception.detail.get("message", "Request failed"))
    else:
        code = "http_error"
        message = str(exception.detail)
    return error_response(request, exception.status_code, code, message)


async def unexpected_error_handler(
    request: Request, exception: Exception
) -> JSONResponse:
    log_event(
        logger,
        logging.ERROR,
        "unexpected_request_error",
        request_id=_request_id(request),
        error_type=type(exception).__name__,
    )
    return error_response(
        request,
        500,
        "internal_error",
        "Internal server error",
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, domain_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
