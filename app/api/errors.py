import logging
import re
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
_VALIDATION_TYPE_PATTERN = re.compile(r"[a-z0-9_.-]{1,64}\Z")
_VALIDATION_MESSAGES = {
    "missing": "Field required",
    "int_parsing": "Invalid integer",
    "float_parsing": "Invalid number",
    "bool_parsing": "Invalid boolean",
    "json_invalid": "Invalid JSON",
    "string_too_short": "Value is too short",
    "string_too_long": "Value is too long",
    "greater_than": "Value is too small",
    "greater_than_equal": "Value is too small",
    "less_than": "Value is too large",
    "less_than_equal": "Value is too large",
}
_VALIDATION_LOCATION_ROOTS = frozenset({"body", "query", "path", "header", "cookie"})


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


def _safe_validation_type(value: object) -> str:
    candidate = str(value).casefold()
    if _VALIDATION_TYPE_PATTERN.fullmatch(candidate):
        return candidate
    return "value_error"


def _safe_validation_message(error_type: str) -> str:
    return _VALIDATION_MESSAGES.get(error_type, "Invalid value")


def _safe_validation_location(value: object) -> list[str | int]:
    if not isinstance(value, (list, tuple)):
        return []
    sanitized: list[str | int] = []
    for index, part in enumerate(value):
        if index == 0 and part in _VALIDATION_LOCATION_ROOTS:
            sanitized.append(str(part))
        elif isinstance(part, int) and not isinstance(part, bool):
            sanitized.append(part)
        elif part == "[key]":
            sanitized.append("key")
        else:
            sanitized.append("field")
    return sanitized


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
    safe_errors = []
    for error in exception.errors():
        error_type = _safe_validation_type(error.get("type", "value_error"))
        safe_errors.append(
            {
                "location": _safe_validation_location(error.get("loc", ())),
                "message": _safe_validation_message(error_type),
                "type": error_type,
            }
        )
    details = {
        "errors": safe_errors
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
