import json
import logging
from io import StringIO

from fastapi.testclient import TestClient

from app.main import create_app
from app.observability import (
    STABLE_FIELDS,
    JsonFormatter,
    RedactionFilter,
    configure_json_logging,
    log_event,
)


def _json_logger(name: str) -> tuple[logging.Logger, StringIO]:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.filters = [RedactionFilter()]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, stream


def test_redaction_filter_removes_sensitive_values_but_keeps_safe_metadata(
    caplog,
) -> None:
    logger = logging.getLogger("tests.safe_logging")
    logger.addFilter(RedactionFilter())

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_event(
            logger,
            logging.INFO,
            "provider_call",
            request_id="request-123",
            provider="fake",
            model="fake-chat",
            api_key="sentinel-api-key",
            context={"authorization": "sentinel-authorization", "safe": "visible"},
        )

    record = caplog.records[-1]
    assert "sentinel" not in caplog.text
    assert record.request_id == "request-123"
    assert record.provider == "fake"
    assert record.model == "fake-chat"
    assert record.api_key == "[REDACTED]"
    assert record.context == {"authorization": "[REDACTED]", "safe": "visible"}


def test_json_formatter_emits_stable_observability_fields() -> None:
    record = logging.LogRecord(
        name="tests.json",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="workflow_stage_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-123"
    record.run_id = "run-456"
    record.stage = "writer"
    record.latency_ms = 17

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "workflow_stage_completed"
    assert payload["request_id"] == "request-123"
    assert payload["run_id"] == "run-456"
    assert payload["stage"] == "writer"
    assert payload["latency_ms"] == 17
    assert set(
        (
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
    ) <= payload.keys()


def test_formatted_dictionary_message_is_recursively_redacted() -> None:
    logger, stream = _json_logger("tests.dictionary_message")

    logger.info(
        {
            "api_key": "sentinel-api-key",
            "nested": {"secret": "sentinel-secret"},
            "safe": "visible",
        }
    )

    output = stream.getvalue()
    payload = json.loads(output)
    assert "sentinel" not in output
    assert "visible" in payload["message"]
    assert "[REDACTED]" in payload["message"]


def test_formatted_positional_mapping_arguments_are_recursively_redacted() -> None:
    logger, stream = _json_logger("tests.positional_arguments")

    logger.info(
        "payload=%s auth=%s",
        {"token_value": "sentinel-token", "safe": "visible"},
        {"authorization": "sentinel-authorization"},
    )

    output = stream.getvalue()
    payload = json.loads(output)
    assert "sentinel" not in output
    assert "visible" in payload["message"]
    assert payload["message"].count("[REDACTED]") == 2


def test_json_logging_configures_existing_handler_once_at_info_level() -> None:
    logger = logging.getLogger("tests.configure_json")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger.handlers = [handler]
    logger.filters = []
    logger.propagate = False
    logger.setLevel(logging.WARNING)

    configure_json_logging(logger)
    configure_json_logging(logger)
    logger.info("configured", extra={"secret": "sentinel-secret"})

    assert len(logger.handlers) == 1
    payload = json.loads(stream.getvalue())
    assert payload["message"] == "configured"
    assert payload["secret"] == "[REDACTED]"
    assert "sentinel" not in stream.getvalue()
    assert set(STABLE_FIELDS) <= payload.keys()


def test_app_startup_emits_redacted_json_request_logs(application_container) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_filters = list(root.filters)
    original_level = root.level
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    root.handlers = [handler]
    root.filters = []
    root.setLevel(logging.WARNING)

    async def raise_error() -> None:
        raise RuntimeError("sentinel-secret-value")

    application = create_app(container=application_container)
    application.add_api_route("/logged-error", raise_error, methods=["GET"])
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            response = client.get("/logged-error")
    finally:
        root.handlers = original_handlers
        root.filters = original_filters
        root.setLevel(original_level)

    lines = [line for line in stream.getvalue().splitlines() if line]
    payloads = [json.loads(line) for line in lines]
    assert response.status_code == 500
    assert {payload["message"] for payload in payloads} >= {
        "unexpected_request_error",
        "request_completed",
    }
    assert all(set(STABLE_FIELDS) <= payload.keys() for payload in payloads)
    assert "sentinel-secret-value" not in stream.getvalue()
