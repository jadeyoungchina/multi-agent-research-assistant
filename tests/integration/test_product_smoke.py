from collections.abc import Iterator
import json

import pytest
from fastapi.testclient import TestClient

from app.bootstrap import build_container
from app.config import Settings
from app.main import create_app
from app.domain.runs import RunEvent


def _workflow_events(sse_payload: str) -> list[RunEvent]:
    """Decode complete workflow frames and reject malformed stream boundaries."""
    assert sse_payload.endswith("\n\n")
    frames = sse_payload[:-2].split("\n\n")
    events = []
    for frame in frames:
        lines = frame.splitlines()
        assert len(lines) == 3
        event_id, event_name, data = lines
        assert event_id.startswith("id: ")
        assert event_name == "event: workflow"
        assert data.startswith("data: ")
        events.append(RunEvent.model_validate(json.loads(data.removeprefix("data: "))))
    return events


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
    events = _workflow_events(event_payload)
    assert stream.status_code == 200
    assert events[-1].event_type == "finished"
    assert events[-1].payload["status"] == "completed"
    assert result["run"]["status"] == "completed"
    assert result["report"]["citations"][0]["filename"] == "overview.md"
