import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


STABLE_FIELDS = (
    "request_id",
    "run_id",
    "stage",
    "provider",
    "model",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "retries",
    "error_type",
)
_SENSITIVE_KEYS = ("api_key", "authorization", "token_value", "secret")
_REDACTED = "[REDACTED]"
_STANDARD_RECORD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).casefold()
    return any(fragment in normalized for fragment in _SENSITIVE_KEYS)


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _REDACTED if _is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.msg)
        record.args = redact(record.args)
        for key, value in list(record.__dict__.items()):
            if key in _STANDARD_RECORD_FIELDS:
                continue
            record.__dict__[key] = _REDACTED if _is_sensitive_key(key) else redact(value)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **{field: getattr(record, field, None) for field in STABLE_FIELDS},
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_FIELDS or key in payload:
                continue
            payload[key] = _REDACTED if _is_sensitive_key(key) else redact(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


def install_redaction_filter(logger: logging.Logger) -> None:
    if not any(isinstance(item, RedactionFilter) for item in logger.filters):
        logger.addFilter(RedactionFilter())


def configure_json_logging(logger: logging.Logger | None = None) -> logging.Logger:
    target = logger or logging.getLogger()
    target.setLevel(logging.INFO)
    install_redaction_filter(target)
    if not target.handlers:
        target.addHandler(logging.StreamHandler())
    for handler in target.handlers:
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(JsonFormatter())
        if not any(isinstance(item, RedactionFilter) for item in handler.filters):
            handler.addFilter(RedactionFilter())
    return target


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    safe_fields = {
        key: _REDACTED if _is_sensitive_key(key) else redact(value)
        for key, value in fields.items()
    }
    logger.log(level, event, extra=safe_fields)
