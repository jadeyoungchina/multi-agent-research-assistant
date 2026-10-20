"""Launch the real application with isolated, deterministic offline providers."""
import argparse
import json
import logging
from pathlib import Path
import socket
import sys
from time import monotonic, sleep
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.bootstrap import build_container
from app.config import Settings
from app.main import create_app

QUESTION = "What are the project's goals and evaluation limitations?"
CORPUS = ROOT / "examples/demo-corpus"


class DemoSettings(Settings):
    """Use explicit inputs only; never consume .env, environment or secret files."""

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        return (init_settings,)


def demo_settings(directory):
    directory = Path(directory).resolve()
    return DemoSettings(_env_file=None, chat_provider="fake", embedding_provider="fake",
                        data_dir=directory, upload_dir=directory / "uploads",
                        database_path=directory / "demo.db", retrieval_min_similarity=-1,
                        dashscope_api_key=None, openai_api_key=None, deepseek_api_key=None)


def demo_files():
    return [(CORPUS / "project-overview.md", "text/markdown"),
            (CORPUS / "evaluation-notes.txt", "text/plain")]


def check_demo(directory):
    """Exercise actual upload/run/SSE/report routes and reopen the saved database."""
    from fastapi.testclient import TestClient

    settings = demo_settings(directory)
    container = build_container(settings)
    with TestClient(create_app(container=container)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        response = client.post("/api/documents", files=[
            ("files", (path.name, path.read_bytes(), media_type)) for path, media_type in demo_files()])
        assert response.status_code == 201, "Demo uploads must succeed."
        documents = response.json()["documents"]
        assert len(documents) == 2 and all(d["status"] == "ready" for d in documents)
        created = client.post("/api/research", json={"question": QUESTION,
                              "document_ids": [d["id"] for d in documents]})
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        deadline = monotonic() + 30
        while True:
            response = client.get(f"/api/research/{run_id}")
            assert response.status_code == 200
            result = response.json()
            if result["run"]["status"] in {"completed", "failed"}:
                break
            if monotonic() > deadline:
                raise RuntimeError("Offline demo did not finish within 30 seconds.")
            sleep(.05)
        assert result["run"]["status"] == "completed", "Offline workflow must complete."
        assert result["run"]["provider"] == "fake" and result["run"]["total_tokens"] == 0
        report = result["report"]
        expected_names = {path.name for path, _ in demo_files()}
        assert {c["filename"] for c in report["citations"]} == expected_names
        assert all(c["excerpt"] and c["evidence_id"] in report["markdown"] for c in report["citations"])
        events = client.get(f"/api/research/{run_id}/events")
        assert events.status_code == 200 and "text/event-stream" in events.headers["content-type"]
        records = [json.loads(line[6:]) for line in events.text.splitlines() if line.startswith("data: ")]
        assert records and records[-1]["event_type"] == "finished"
        sequences = [r["sequence"] for r in records]
        assert sequences == sorted(set(sequences))
        stages = sorted({r["stage"] for r in records})
        assert {"planner", "retriever", "researcher", "critic", "writer", "citation_validator"} <= set(stages)
    reopened = build_container(settings)
    try:
        assert reopened.services.research.get_run(run_id).status == "completed"
        assert reopened.services.research.get_report(run_id).model_dump(mode="json") == report
    finally:
        reopened.close()
    receipt = {"mode": "deterministic_offline_fake", "documents": sorted(expected_names),
               "completed": True, "citation_count": len(report["citations"]),
               "stages": stages, "event_count": len(records), "persisted_report_reopened": True,
               "provider_tokens": 0, "real_model_requests": 0}
    (Path(directory) / "check.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check actual API/SSE/persistence offline and exit.")
    parser.add_argument("--port", type=int, default=8000, help="Loopback port; 0 selects a free port.")
    args = parser.parse_args()
    logging.disable(logging.INFO)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535.")
    directory = ROOT / "artifacts/demo" / uuid4().hex
    if args.check:
        check_demo(directory)
        return

    import uvicorn

    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error("Cannot bind the requested port; choose another with --port.")
        port = listener.getsockname()[1]
        container = build_container(demo_settings(directory))
        try:
            for path, media_type in demo_files():
                container.services.documents.ingest(path.name, media_type, path.read_bytes())
            print("OFFLINE FAKE DEMO: deterministic software demonstration; no model API calls.", flush=True)
            print(f"Open http://127.0.0.1:{port} and select both preloaded documents.", flush=True)
            print(f"Suggested question: {QUESTION}", flush=True)
            print(f"Isolated data: {directory.relative_to(ROOT).as_posix()}", flush=True)
            print("Press Ctrl+C to stop. Each invocation starts a new isolated session.", flush=True)
            app = create_app(container=container)
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                    log_level="warning", access_log=False))
            server.run(sockets=[listener])
        finally:
            container.close()


if __name__ == "__main__":
    main()
