"""Publish immutable, allowlisted generation checkpoints without changing frozen inputs."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.evaluation.budget import BudgetLedger, cny
from app.evaluation.frozen import FREEZE_PATH, validate_freeze, write_json
from app.evaluation.test_judging import load_repeat

RUN = ROOT / "artifacts/evaluation/real-v1-test"
OUT = ROOT / "docs/evaluation/heldout-test"


def receipts(path, phase, repeat):
    if not path.is_file():
        raise ValueError("required ledger missing")
    rows = BudgetLedger(path).summary()["attempts"]
    return [{"id": r["id"], "repeat_index": repeat, "phase": phase,
             "provider": r["provider"], "model": r["model"], "kind": r["kind"], "role": r["role"],
             "created_at": r["created_at"], "request_sha256": r["request_hash"], "state": r["state"],
             "outcome": r["outcome"], "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"],
             "cached_input_tokens": r["cached_tokens"], "price": json.loads(r["price_json"]),
             "estimated_cost_cny": cny(r["charged"]) if r["state"] == "settled" and r["price_json"] != "null" else None}
            for r in rows]


def checkpoint(complete_repeat=None):
    record = validate_freeze(FREEZE_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    if complete_repeat is not None:
        report = load_repeat(record, RUN, complete_repeat)
        source, dest = RUN / f"repeat-{complete_repeat}", OUT / f"repeat-{complete_repeat}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "traces.jsonl").write_text("".join(t.model_dump_json() + "\n" for t in report.traces), encoding="utf-8", newline="\n")
        for name in ("repeat-manifest.json", "performance.json"):
            write_json(dest / name, json.loads((source / name).read_text("utf-8")))
        write_json(dest / "metadata.json", report.metadata.model_dump(mode="json"))
        phase_rows = receipts(source / "generation.sqlite", "answer_generation", complete_repeat) + receipts(source / "index.sqlite", "index_setup", complete_repeat)
        by_id = {r["id"]: r for r in phase_rows if r["phase"] == "answer_generation"}
        assigned = [i for t in report.traces for i in t.attempt_ids]
        if len(assigned) != len(set(assigned)) or set(assigned) != by_id.keys():
            raise ValueError("answer receipt attribution differs")
        for trace in report.traces:
            assigned_rows = [by_id[i] for i in trace.attempt_ids]
            if (len(assigned_rows) != trace.model_calls or sum(r["input_tokens"] or 0 for r in assigned_rows) != trace.prompt_tokens
                    or sum(r["output_tokens"] or 0 for r in assigned_rows) != trace.completion_tokens):
                raise ValueError("trace usage differs")
        (dest / "request-usage.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in phase_rows), encoding="utf-8", newline="\n")
        captures = [json.loads(line) for line in (source / "provider-responses.jsonl").read_text("utf-8").splitlines()]
        diagnostics = [{"case_id": t.case_id, "variant": t.variant.value, "trace_sha256": sha256(t.model_dump_json().encode()).hexdigest(),
                        "error_code": t.error_code, "captures": [{k: item[k] for k in ("attempt_id", "schema", "content", "finish_reason", "validation_issues")}
                        for item in captures if item["attempt_id"] in t.attempt_ids and (item["schema"] == "DraftReport" or item["validation_issues"])]}
                       for t in report.traces if t.status != "success"]
        write_json(dest / "failure-diagnostics.json", diagnostics)
        write_json(dest / "summary.json", {"freeze_id": record["freeze_id"], "repeat_index": complete_repeat,
                   "executed_units": len(report.traces), "successful_units": sum(t.status == "success" for t in report.traces),
                   "answer_attempts": len(by_id), "index_attempts": sum(r["phase"] == "index_setup" for r in phase_rows),
                   "generation_complete": True, "semantic_scores_published": False,
                   "publications": [{"path": p.name, "sha256": sha256(p.read_bytes()).hexdigest()} for p in sorted(dest.iterdir()) if p.name != "summary.json"]})
    repeats = []
    for repeat in range(3):
        directory = RUN / f"repeat-{repeat}"
        path = directory / "traces.jsonl"
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.exists() else []
        repeats.append({"repeat_index": repeat, "completed_units": len(rows), "successful_units": sum(r["status"] == "success" for r in rows),
                        "generation_complete": (directory / "results.json").is_file(),
                        "checkpoint_published": (OUT / f"repeat-{repeat}/summary.json").is_file()})
    status = {"schema_version": "frozen-heldout-progress-v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "freeze_id": record["freeze_id"], "planned_units": 480, "executed_units": sum(r["completed_units"] for r in repeats),
              "repeats": repeats, "human_scores": 0, "semantic_result_claim": "none; generation progress only"}
    write_json(OUT / "progress.json", status)
    lines = ["# Frozen held-out experiment: generation checkpoints", "",
             f"Progress recorded at **{status['recorded_at_utc']}** (UTC): **{status['executed_units']}/480** generation units durably completed.", "",
             "This page is a committed snapshot, not a live progress indicator. No semantic comparison is claimed at this stage.", "",
             f"Freeze ID: `{record['freeze_id']}`. [Frozen protocol](../frozen-experiment.md) · [Machine-readable progress](progress.json).", "",
             "The matrix is 40 questions × four workflows × three fresh serial repeats. All 16 test documents form the same candidate pool. Parameters, reference keys, review sample and statistics were frozen before generation.", "",
             "| Repeat | Completed / planned | Successful executions | Published generation packet |", "| --- | ---: | ---: | --- |"]
    for r in repeats:
        link = f"[Packet](repeat-{r['repeat_index']}/summary.json)" if r["checkpoint_published"] else "Pending"
        lines.append(f"| {r['repeat_index']} | {r['completed_units']}/160 | {r['successful_units']} | {link} |")
    lines += ["", "Execution success is a pipeline outcome, not correctness or citation entailment. Failed executions remain retained. Each completed generation packet contains exact answers/evidence, execution order, configuration, performance denominators, safe physical-request receipts and any failure diagnostics. Index setup and answer requests are explicitly separate.", "",
              "Blinded DeepSeek judging, the prespecified 80-answer AI source review and paired family analysis follow generation. Human scoring remains **0**. Account-mode judge prices are unverified; unknown usage or amounts must remain unknown. Judge aliases may drift. These public-document questions and AI references do not establish independent human validation.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    checkpoint(int(sys.argv[1]) if len(sys.argv) > 1 else None)
