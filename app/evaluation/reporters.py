"""JSON, CSV, and Markdown views of one typed evaluation report."""

import csv
import json
from pathlib import Path

from .metrics import CaseScore
from .models import WorkflowVariant
from .runner import EvaluationReport, ReportMetrics
from .trace import EvaluationTrace


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(report: EvaluationReport) -> str:
    metadata = report.metadata
    variants = list(WorkflowVariant)
    lines = [
        "# Synthetic benchmark evaluation", "",
        "A single run does not establish a general conclusion. Repeat real-provider runs "
        "with the same dataset, corpus, configuration, and prompts before making quality claims.", "",
        "Fake results verify the software pipeline only. Fake token counts are zero because "
        "no model is called; they are not estimates of real token usage.", "",
        f"Run time (UTC): {metadata.run_at_utc.isoformat()}", "",
        f"Git commit: `{metadata.git_commit}` (dirty worktree: {metadata.git_dirty})", "",
        f"Corpus SHA-256: `{metadata.corpus_sha256}`", "",
        f"Dataset SHA-256: `{metadata.dataset_sha256 or 'not provided'}`", "",
        f"Prompt version: `{metadata.prompt_version}`", "",
        "## Four-variant comparison", "",
        "| Metric | " + " | ".join(variant.value for variant in variants) + " |",
        "| --- | " + " | ".join("---:" for _ in variants) + " |",
    ]
    for field in ReportMetrics.model_fields:
        values = []
        for variant in variants:
            metrics = report.by_variant.get(variant)
            value = getattr(metrics, field) if metrics else "not run"
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(f"| {field} | " + " | ".join(values) + " |")
    lines.extend([
        "", "Latency p50/p95 use nearest-rank percentiles and externally measured milliseconds. "
        "Failed runs remain in aggregates. Shared document ingestion is excluded from per-case "
        "latency and usage. Only multi_agent_rag is release-blocking in v1.0.", "",
        "## Failures", "",
    ])
    failures = [trace for trace in report.traces if trace.status == "failed"]
    if failures:
        lines.extend(["| Case | Variant | Error |", "| --- | --- | --- |"])
        lines.extend(
            f"| {_cell(trace.case_id)} | {trace.variant.value} | {_cell(trace.error_code)} |"
            for trace in failures
        )
    else:
        lines.append("No workflow failures.")
    lines.extend([
        "", "## Configuration", "", "```json",
        metadata.configuration.model_dump_json(indent=2), "```", "",
        "The corpus digest hashes sorted relative POSIX filenames and exact file bytes, "
        "each preceded by its unsigned 8-byte big-endian length. "
        "Timing and generated document IDs may differ between runs.", "",
    ])
    return "\n".join(lines)


def write_reports(report: EvaluationReport, output_dir: Path) -> list[Path]:
    """Write three complete views, preserving CSV quoting and one row per trace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    markdown_path = output_dir / "report.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    score_by_key = {(score.case_id, score.variant): score for score in report.scores}
    fieldnames = list(dict.fromkeys([*EvaluationTrace.model_fields, *CaseScore.model_fields]))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trace in report.traces:
            row = {**trace.model_dump(mode="json"), **score_by_key[(trace.case_id, trace.variant)].model_dump(mode="json")}
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return [json_path, csv_path, markdown_path]
