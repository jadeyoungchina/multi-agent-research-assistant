"""Replay explicit Codex source adjudications for the retained v8 dev run."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.development import answer_blocks, load_scoring_cases, score_development
from app.evaluation.runner import EvaluationReport
from scripts.judge_development import decode


def main():
    root = ROOT / "artifacts/evaluation/real-v1-dev-v8"
    report = EvaluationReport.model_validate_json((root / "generation/results.json").read_text("utf-8"))
    if (report.metadata.git_commit != "c293dc02979af4b95a9d37f458480eaf66f3c56c"
            or report.metadata.git_dirty or report.metadata.prompt_version != "evaluation-v8"
            or len(report.traces) != 80 or not report.execution_complete):
        raise ValueError("this recipe applies only to the recorded clean final-rule run")
    cases = load_scoring_cases(ROOT / "benchmarks/real-v1")
    notes = json.loads((root / "source-notes-before-judge.json").read_text("utf-8"))
    decisions = {}
    for index, trace in enumerate(report.traces, 1):
        key = f"{trace.case_id}/{trace.variant.value}"
        if key not in notes:
            continue
        unit = f"answer-{index:03d}"
        raw = json.loads((root / f"judgments/{unit}.json").read_text("utf-8"))
        digest = sha256(trace.model_dump_json().encode()).hexdigest()
        if notes[key]["trace_sha256"] != digest or raw["trace_sha256"] != digest:
            raise ValueError("source review does not match the retained answer")
        judgment = decode(raw["attempts"][-1]["raw_response"]["text"])
        blocks = {block["index"]: block for block in answer_blocks(trace.answer)}
        for block in judgment.blocks:
            if block.requires_citation and not blocks[block.index]["citation_ids"]:
                block.support = "uncited"
                block.reason = "No attached citation; factual truth is assessed separately."

        def point(number, score, reason):
            next(p for p in judgment.key_points if p.index == number).score = score
            next(p for p in judgment.key_points if p.index == number).reason = reason

        def block(number, support, reason):
            verdict = next(b for b in judgment.blocks if b.index == number)
            verdict.requires_citation = True
            verdict.support, verdict.reason = support, reason

        if key == "REAL-DEV-006/baseline_llm":
            point(1, 2, "Explicitly says both defaults cannot be specified together.")
        elif key == "REAL-DEV-020/single_agent_rag":
            block(0, "supported", "Attached passages support classes, handlers and overrides; absence clause scoped.")
        elif key == "REAL-DEV-020/llm_rag":
            block(0, "supported", "Mixed paragraph describes documented class differences and handlers.")
        elif key == "REAL-DEV-020/multi_agent_rag":
            block(2, "uncited", "Mixed summary asserts inheritance and handlers without attached citations.")
        elif key == "REAL-DEV-019/multi_agent_rag":
            block(5, "partial", "Class guidance supported; named decorator absent from attached passage.")
        elif key == "REAL-DEV-001/multi_agent_rag":
            block(0, "uncited", "Mode-dependent validation behavior is a factual assertion without citation.")
            block(3, "partial", "Attached chunk shows config strictness and Field(strict=False), not strict=True.")
        elif key == "REAL-DEV-001/baseline_llm":
            point(1, 2, "Rejection because string is not integer faithfully paraphrases error.")
        elif key == "REAL-DEV-007/multi_agent_rag":
            block(1, "supported", "Summary scopes signature distinction and explicitly mentions alias workaround.")
            block(5, "partial", "AliasChoices statement lacks that source among this block's attached citations.")
        elif key == "REAL-DEV-007/baseline_llm":
            judgment.unsupported_assertion = True
        elif key == "REAL-DEV-007/single_agent_rag":
            point(0, 2, "validation_alias with populate_by_name accepts both names; locally verified.")
            block(0, "supported", "Attached fields passage covers runtime alias equivalence and populate_by_name.")
            block(2, "supported", "Attached fields passage distinguishes runtime aliases from editor hints.")

        # Preserve original wording for accepted verdicts. A source adjudication
        # is an explicit separate record, never a replacement of raw model text.
        original = decode(raw["attempts"][-1]["raw_response"]["text"])
        meaningful_change = judgment.model_dump(exclude={"blocks"}) != original.model_dump(exclude={"blocks"})
        meaningful_change |= any(
            (b.requires_citation, b.support) != (a.requires_citation, a.support)
            for a, b in zip(original.blocks, judgment.blocks)
        )
        corrected = raw["status"] != "valid_ai_judgment" or meaningful_change
        if not corrected:
            judgment = original
        else:
            judgment.explanation = "Codex source adjudication: " + notes[key]["reason"]
        score_development(cases[trace.case_id], trace, judgment)
        decisions[unit] = {
            "case_id": trace.case_id, "variant": trace.variant.value,
            "trace_sha256": digest, "reviewer_kind": "ai",
            "action": "corrected" if corrected else "accepted",
            "reason": notes[key]["reason"], "judgment": judgment.model_dump(),
            "source_checked_at_utc": notes[key]["checked_at_utc"],
            "before_original_judgment_seen": notes[key].get("before_original_judgment_seen", True),
        }
    result = {
        "schema_version": "codex-development-adjudications-v1", "reviewer_kind": "ai",
        "human_reviewed": False, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": sha256((ROOT / "benchmarks/real-v1/dev-scoring-v2.json").read_bytes()).hexdigest(),
        "adjudications": decisions, "execution_failure_audits": {},
        "runtime_check": json.loads((root / "alias-runtime-check.json").read_text("utf-8")),
    }
    (root / "adjudications.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
    print(f"Retained {len(decisions)} source reviews; corrected {sum(d['action']=='corrected' for d in decisions.values())}.")


if __name__ == "__main__":
    main()
