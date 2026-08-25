import json
import logging

from app.observability import JsonFormatter, RedactionFilter, log_event


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
