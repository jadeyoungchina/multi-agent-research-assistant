from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.bootstrap import build_container
from app.config import Settings
from app.main import create_app


@pytest.fixture
def fake_app_client(tmp_path) -> Iterator[TestClient]:
    """A complete local application with explicit offline providers."""
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path,
        upload_dir=tmp_path / "uploads",
        database_path=tmp_path / "research.db",
        chat_provider="fake",
        embedding_provider="fake",
        dashscope_api_key=None,
        openai_api_key=None,
        retrieval_min_similarity=-1.0,
    )
    with TestClient(create_app(container=build_container(settings))) as client:
        yield client


def test_fake_product_flow_uploads_runs_streams_and_reports(
    fake_app_client: TestClient,
) -> None:
    """Catch an offline composition that cannot turn uploaded evidence into a report."""
    upload = fake_app_client.post(
        "/api/documents",
        files=[
            (
                "files",
                (
                    "overview.md",
                    b"The project requires traceable citations.",
                    "text/markdown",
                ),
            )
        ],
    )
    assert upload.status_code == 201
    document_id = upload.json()["documents"][0]["id"]

    created = fake_app_client.post(
        "/api/research",
        json={
            "question": "What does the project require?",
            "document_ids": [document_id],
        },
    )
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    fake_app_client.app.state.run_tasks.wait(run_id, timeout=5)

    with fake_app_client.stream("GET", f"/api/research/{run_id}/events") as stream:
        event_payload = "".join(stream.iter_text())

    result = fake_app_client.get(f"/api/research/{run_id}").json()
    assert stream.status_code == 200
    assert "event: workflow" in event_payload
    assert '"stage":"citation_validator"' in event_payload
    assert result["run"]["status"] == "completed"
    assert result["report"]["citations"][0]["filename"] == "overview.md"
