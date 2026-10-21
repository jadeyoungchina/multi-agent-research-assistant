"""Exercise the documented Uvicorn entry with an isolated fake-provider profile."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import monotonic, sleep
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[3]


def main():
    session = ROOT / "artifacts/default-startup-check" / uuid4().hex
    session.mkdir(parents=True)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = dict(os.environ)
    environment.update({"CHAT_PROVIDER": "fake", "EMBEDDING_PROVIDER": "fake",
                        "DATA_DIR": str(session), "DATABASE_PATH": str(session / "app.db"),
                        "UPLOAD_DIR": str(session / "uploads"), "RETRIEVAL_MIN_SIMILARITY": "-1",
                        "DASHSCOPE_API_KEY": "",
                        "OPENAI_API_KEY": "",
                        "DEEPSEEK_API_KEY": "",
                        })
    command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)]
    with (session / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=10) as client:
                deadline = monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Default entry stopped before startup; inspect the isolated server log.")
                    try:
                        health = client.get("/health")
                        if health.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if monotonic() > deadline:
                        raise RuntimeError("Default entry did not become healthy within 30 seconds.")
                    sleep(.1)
                ui = client.get("/")
                assert ui.status_code == 200 and "script-src 'self'" in ui.headers["content-security-policy"]
                uploaded = client.post("/api/documents", files=[
                    ("files", ("startup-check.md", b"The note requires traceable citations and bounded research steps.", "text/markdown"))])
                assert uploaded.status_code == 201
                document_id = uploaded.json()["documents"][0]["id"]
                created = client.post("/api/research", json={"question": "What does the note require?",
                                      "document_ids": [document_id]})
                assert created.status_code == 202
                run_id = created.json()["run_id"]
                deadline = monotonic() + 30
                while True:
                    result = client.get(f"/api/research/{run_id}").json()
                    if result["run"]["status"] in {"completed", "failed"}:
                        break
                    if monotonic() > deadline:
                        raise RuntimeError("The default-entry research run did not finish.")
                    sleep(.1)
                assert result["run"]["status"] == "completed"
                assert result["run"]["provider"] == "fake" and result["run"]["total_tokens"] == 0
                assert {c["filename"] for c in result["report"]["citations"]} == {"startup-check.md"}
                events = client.get(f"/api/research/{run_id}/events")
                assert events.status_code == 200 and "text/event-stream" in events.headers["content-type"]
                records = [json.loads(line[6:]) for line in events.text.splitlines() if line.startswith("data: ")]
                assert records[-1]["event_type"] == "finished"
                stages = sorted({r["stage"] for r in records})
                assert {"planner", "retriever", "researcher", "critic", "writer", "citation_validator"} <= set(stages)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    record = {"kind": "post_freeze_default_entry_software_check",
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "product_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "entrypoint": "python -m uvicorn app.main:app --host 127.0.0.1 --port <ephemeral>",
              "profile": "isolated fake chat and embeddings; empty model credentials",
              "checks": {"health_http": 200, "ui_http": 200, "content_security_policy_preserved": True,
                         "upload_http": 201, "research_http": 202, "terminal_status": "completed",
                         "sse_http": 200, "workflow_stages": stages,
                         "citation_count": len(result["report"]["citations"])},
              "real_model_requests": 0,
              "bootstrap_sha256_lf": sha256((ROOT / "app/bootstrap.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "verification_tool_sha256_lf": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "interpretation": "Checks ordinary application startup and integration, not model quality or a new held-out experiment."}
    destination = ROOT / "docs/engineering/default-startup-check.json"
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Default entry PASS: health, real UI, upload, six workflow stages, report, citations and SSE; fake providers only.")


if __name__ == "__main__":
    main()
