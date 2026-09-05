from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.dependencies import ApplicationServices
from app.main import create_app


def test_index_contains_complete_research_controls(client) -> None:
    response = client.get("/")

    assert response.status_code == 200
    for element_id in (
        "upload-form",
        "document-input",
        "document-list",
        "research-form",
        "question-input",
        "research-documents",
        "timeline",
        "report",
        "citation-dialog",
    ):
        assert f'id="{element_id}"' in response.text


def test_frontend_is_self_contained_and_avoids_html_injection() -> None:
    html = Path("app/static/index.html").read_text(encoding="utf-8")
    script = Path("app/static/app.js").read_text(encoding="utf-8")

    assert "https://" not in html and "http://" not in html
    assert "innerHTML" not in script
    assert "EventSource" in script


def test_dashboard_assets_cover_safe_research_lifecycle() -> None:
    script = Path("app/static/app.js").read_text(encoding="utf-8")
    stylesheet = Path("app/static/styles.css").read_text(encoding="utf-8")

    for function_name in (
        "refreshDocuments",
        "uploadDocuments",
        "startResearch",
        "openEventStream",
        "renderWorkflowEvent",
        "renderReport",
        "openCitation",
    ):
        assert f"function {function_name}" in script
    assert "evidence_sufficient === false" in script
    assert "activeEventSource.close()" in script
    assert "@media (max-width: 760px)" in stylesheet


def test_static_assets_are_served_when_the_server_starts_elsewhere(
    monkeypatch, tmp_path
) -> None:
    services = ApplicationServices(
        documents=SimpleNamespace(),
        research=SimpleNamespace(list_incomplete_ids=lambda: []),
    )
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app(services=services)) as client:
        response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "@media (max-width: 760px)" in response.text
