import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import unexpected_error_handler
from app.observability import install_redaction_filter, log_event


REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
        "style-src 'self'; script-src 'self'"
    ),
}
logger = logging.getLogger(__name__)
install_redaction_filter(logger)


def safe_request_id(value: str | None) -> str:
    if value is not None and REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = safe_request_id(Headers(scope=scope).get("X-Request-ID"))
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()
        response_started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception as exception:
            if response_started:
                raise
            request = Request(scope, receive=receive)
            response = await unexpected_error_handler(request, exception)
            await response(scope, receive, send_with_headers)
        finally:
            log_event(
                logger,
                logging.INFO,
                "request_completed",
                request_id=request_id,
                latency_ms=max(0, int((perf_counter() - started) * 1000)),
            )
