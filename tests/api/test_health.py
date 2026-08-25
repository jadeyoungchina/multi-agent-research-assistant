from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import ApplicationServices
from app.bootstrap import (
    ApplicationContainer,
    build_container,
    configured_chat_identity,
)
from app.config import Settings
from app.main import create_app


def test_health_reports_only_safe_readiness(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "1.0.0",
        "database_ready": True,
        "providers": {"chat": True, "embedding": True},
    }
    assert "api_key" not in response.text.casefold()


def test_request_id_is_echoed_when_safe(client) -> None:
    response = client.get("/health", headers={"X-Request-ID": "demo-123"})

    assert response.headers["X-Request-ID"] == "demo-123"


@pytest.mark.parametrize(
    "request_id",
    ["", "contains space", "slash/value", "x" * 129],
)
def test_unsafe_request_id_is_replaced(client, request_id: str) -> None:
    response = client.get("/health", headers={"X-Request-ID": request_id})

    generated = response.headers["X-Request-ID"]
    assert generated != request_id
    assert len(generated) == 32
    assert generated.isalnum()


def test_security_headers_are_set_exactly(client) -> None:
    response = client.get("/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Content-Security-Policy"] == (
        "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
        "style-src 'self'; script-src 'self'"
    )


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("dashscope", ("dashscope", "qwen-custom")),
        ("openai", ("openai", "gpt-custom")),
        ("fake", ("fake", "fake-chat")),
    ],
)
def test_configured_chat_identity_uses_selected_chat_model(
    provider: str, expected: tuple[str, str]
) -> None:
    settings = Settings(
        chat_provider=provider,
        dashscope_chat_model="qwen-custom",
        openai_chat_model="gpt-custom",
    )

    assert configured_chat_identity(settings) == expected


def test_build_container_wraps_every_graph_node_and_records_chat_identity(
    tmp_path, monkeypatch
) -> None:
    wrapped_stages: list[str] = []

    def recording_wrapper(stage, handler, repository):
        del repository
        wrapped_stages.append(stage)
        return handler

    monkeypatch.setattr("app.bootstrap.checkpointed_node", recording_wrapper)
    settings = Settings(
        app_env="test",
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        database_path=tmp_path / "research.db",
        chat_provider="fake",
        embedding_provider="fake",
    )

    container = build_container(settings)

    assert set(wrapped_stages) == {
        "planner",
        "retriever",
        "researcher",
        "critic",
        "writer",
        "citation_validator",
    }
    assert container.services.research.provider_name == "fake"
    assert container.services.research.model == "fake-chat"
    container.close()


def test_lifespan_uses_injected_services_without_building_container(monkeypatch) -> None:
    services = ApplicationServices(
        documents=SimpleNamespace(),
        research=SimpleNamespace(list_incomplete_ids=lambda: []),
    )

    def fail_build():
        raise AssertionError("production container must not be built")

    monkeypatch.setattr("app.main.build_container", fail_build)
    app = create_app(services=services)

    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert test_client.app.state.services is services


def test_lifespan_closes_a_shared_provider_only_once(
    application_container: ApplicationContainer,
) -> None:
    shared = application_container.chat_provider
    application_container.embedding_provider = shared
    app = create_app(container=application_container)

    with TestClient(app):
        pass

    assert shared.close_count == 1


def test_database_readiness_closes_its_connection(
    application_container: ApplicationContainer,
) -> None:
    class RecordingConnection:
        def __init__(self) -> None:
            self.closed = False

        def execute(self, statement: str):
            assert statement == "SELECT 1"
            return self

        def fetchone(self):
            return (1,)

        def close(self) -> None:
            self.closed = True

    connection = RecordingConnection()
    application_container.database = SimpleNamespace(connect=lambda: connection)

    assert application_container.database_ready() is True
    assert connection.closed is True
