"""Check the real browser workflow and capture an explicitly offline fake demo."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/portfolio"


def main():
    from playwright.sync_api import expect, sync_playwright

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-python", default=sys.executable, help="Python with the project dependencies installed.")
    parser.add_argument("--browser", choices=("chrome", "msedge", "chromium"), default="chrome")
    args = parser.parse_args()
    logs = ROOT / "artifacts/demo-browser"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "server-stdout.log"
    stderr_path = logs / "server-stderr.log"
    process = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen([args.app_python, "-X", "utf8", str(ROOT / "examples/run_demo.py"), "--port", "0"],
                                       cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Demo server stopped before becoming ready; inspect artifacts/demo-browser logs.")
                match = re.search(r"Open (http://127\.0\.0\.1:\d+)", stdout_path.read_text("utf-8"))
                if match:
                    url = match.group(1)
                    break
                if monotonic() > deadline:
                    raise RuntimeError("Demo server startup exceeded 30 seconds.")
                sleep(.1)

            with sync_playwright() as playwright:
                launch = {"headless": True}
                if args.browser != "chromium":
                    launch["channel"] = args.browser
                browser = playwright.chromium.launch(**launch)
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url, wait_until="networkidle")
                    page.locator("#research-documents input").nth(1).wait_for()
                    page.locator("#document-input").set_input_files([
                        str(ROOT / "examples/demo-corpus/project-overview.md"),
                        str(ROOT / "examples/demo-corpus/evaluation-notes.txt")])
                    page.locator("#upload-form button").click()
                    expect(page.locator("#status-message")).to_have_text("资料上传完成。")
                    assert page.locator(".document-item").count() == 2
                    for item in page.locator("#research-documents input").all():
                        item.check()
                    page.locator("#question-input").fill("What are the project's goals and evaluation limitations?")
                    page.locator("#research-submit").click()
                    page.locator("#report .citation-button").first.wait_for(timeout=30000)
                    expect(page.locator("#status-message")).to_contain_text("研究已完成")
                    stages = [name for name in ("planner", "retriever", "researcher", "critic", "writer", "citation_validator")
                              if name in page.locator("#timeline").inner_text()]
                    assert len(stages) == 6 and not errors
                    images = OUT / "images"
                    images.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(images / "offline-demo-report.png"), full_page=True)
                    page.locator("#report .citation-button").first.click()
                    page.wait_for_selector("#citation-dialog[open]")
                    content = page.locator("#citation-content").inner_text()
                    assert "文件：" in content and "片段：" in content
                    assert any(name in content for name in ("project-overview.md", "evaluation-notes.txt"))
                    assert page.locator(".citation-excerpt").inner_text().strip()
                    page.screenshot(path=str(images / "offline-demo-citation.png"))
                    page.locator("#citation-close").click()
                    assert not page.locator("#citation-dialog").is_visible()
                    # A reload checks that uploads remain available from the actual local database.
                    page.reload(wait_until="networkidle")
                    page.locator("#research-documents input").nth(1).wait_for()
                    assert page.locator(".document-item").count() == 2 and not errors
                    version = browser.version
                finally:
                    browser.close()
            inputs = ["examples/run_demo.py", "docs/portfolio/tools/capture_demo.py",
                      "examples/demo-corpus/project-overview.md", "examples/demo-corpus/evaluation-notes.txt"]
            record = {"kind": "offline_fake_software_demonstration", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                      "browser": args.browser, "browser_version": version,
                      "checks": {"ui_upload": True, "ready_documents": 2, "workflow_stages_seen": stages,
                                 "report_rendered": True, "citation_dialog_opened_and_closed": True,
                                 "source_excerpt_visible": True, "documents_survive_reload": True, "page_errors": 0},
                      "real_model_requests": 0,
                      "interpretation": "Verifies real UI/API/workflow integration using deterministic fake providers; not real-model quality evidence.",
                      "inputs": [{"path": name, "sha256_lf": sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()} for name in inputs],
                      "screenshots": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p.read_bytes()).hexdigest()}
                                      for p in sorted(images.glob("offline-demo-*.png"))]}
            (OUT / "demo-verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                                       encoding="utf-8", newline="\n")
            print("Browser demo PASS: upload, six workflow stages, report, citation dialog and persisted documents; zero page errors.")
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
