"""Replay explicit Codex source checks; never silently replace original judging."""

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
    root = ROOT / "artifacts/evaluation/real-v1-dev-full"
    suite = ROOT / "benchmarks/real-v1"
    report = EvaluationReport.model_validate_json((root / "generation/results.json").read_text("utf-8"))
    cases = load_scoring_cases(suite)
    selected = {"REAL-DEV-001", "REAL-DEV-007", "REAL-DEV-013", "REAL-DEV-019", "REAL-DEV-020"}
    decisions, failures = {}, {}
    for index, trace in enumerate(report.traces, 1):
        unit = f"answer-{index:03d}"
        raw = json.loads((root / f"judgments/{unit}.json").read_text("utf-8"))
        identity = sha256(trace.model_dump_json().encode()).hexdigest()
        if trace.status != "success":
            failures[unit] = {"case_id": trace.case_id, "variant": trace.variant.value,
                              "trace_sha256": identity, "error_code": trace.error_code,
                              "reason": {"REAL-DEV-020": "Writer repeated Cite:ref prefixes in ID fields after correction; both responses retain paid usage.",
                                         "REAL-DEV-019": "Researcher returned source passages in ID fields; rejecting unknown references is correct.",
                                         "REAL-DEV-004": "Output reached the prespecified 4096-token cap; retain as execution failure, not a semantic zero."}[trace.case_id]}
            continue
        if trace.case_id not in selected and raw["status"] != "judge_error":
            continue
        judgment = decode(raw["attempts"][-1]["raw_response"]["text"])
        original = judgment.model_dump()
        notes = ["Codex checked the answer against the pinned source and actual attached passages; human review remains false."]
        blocks = {block["index"]: block for block in answer_blocks(trace.answer)}
        for block in judgment.blocks:
            if block.requires_citation and not blocks[block.index]["citation_ids"] and block.support != "uncited":
                block.support = "uncited"
                block.reason = "No well-formed attached citation; factual correctness is assessed separately."
                notes.append("Factual blocks without exact attached tokens are uncited, including malformed brace-wrapped labels; no arbitrary source is credited.")
        if index == 50:
            judgment.key_points[1].score = 2
            judgment.key_points[1].reason = "ValidationError and expected-int/got-string faithfully paraphrase the pinned integer error."
            notes.append("Corrected a literal-error-wording penalty: both required facts are present in Chinese; exact English wording is not required.")
        elif index == 57:
            judgment.unsupported_assertion = False
            notes.append("Pinned dependencies/index lines 112 and 317 support the 0.95.0 history and editor/mypy benefits; non-Annotated examples establish the alternative syntax. Malformed citations affect citation scoring, not factual truth.")
        elif index == 67:
            notes.append("The block mixes a scoped measurement limitation with the actual cache mechanism; it is factual and uncited because its brace-wrapped token is malformed. No fabricated measurement is asserted.")
        elif index == 78:
            judgment.key_points[1].score = 1
            judgment.key_points[1].reason = "Core validation_alias distinction is right; claimed lack of any editor workaround contradicts fields guide."
            judgment.unsupported_assertion = True
            notes.append("The fields guide gives alias plus serialization_alias as an editor workaround. The broad no-standard-configuration claim is false; preserve partial citation verdicts and partial second-key-point credit.")
        elif index == 79:
            judgment.unsupported_assertion = True
            notes.append("The later AliasChoices example is correct and earns both key points. The first to_snake('first_name') example does not accept fname and is a misleading extra purported solution; retain an unsupported-extra flag.")
        elif index == 80:
            point_type = type(judgment.key_points[0])
            judgment.key_points = [
                point_type(index=0, score=2, reason="validation_alias with populate_by_name supports both described input names; AliasChoices is optional."),
                point_type(index=1, score=1, reason="Ignores validation_alias correctly, but wrongly says serialization_alias controls editor interpretation.")]
            judgment.unsupported_assertion = True
            notes.append("Removed the judge's invented third key point. Accept the documented runtime alternative, but give partial editor credit because serialization_alias controls output naming, not IDE signature selection. All three blocks lack well-formed citations.")
        elif trace.case_id == "REAL-DEV-013":
            notes.append("Pinned sub-dependencies guide supports one cached call per request and use_cache=False without a generator condition. Baseline invents scope parameters and denies this option. Uncited code and weakly attached passages remain penalized.")
        elif trace.case_id == "REAL-DEV-019":
            notes.append("Raise FastAPI's class and register Starlette's base class. A subclass handler does not catch base-class exceptions; optional import aliases do not earn or cost points.")
        elif trace.case_id == "REAL-DEV-020":
            notes.append("The whole-pool reference audit contains no production latency measurements. Baseline fabricates microsecond ranges; scoped RAG abstentions are appropriate and citation N/A.")
        score_development(cases[trace.case_id], trace, judgment)
        changed = judgment.model_dump() != original
        decisions[unit] = {"action": "corrected" if changed else "accepted", "reviewer_kind": "ai",
                           "trace_sha256": identity, "reason": " ".join(dict.fromkeys(notes)),
                           "judgment": judgment.model_dump()}
    packet = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "Codex", "human_reviewed": False,
              "policy_sha256": sha256((suite / "dev-scoring-v2.json").read_bytes()).hexdigest(),
              "selection": "All successful units for the five prespecified pilot IDs, plus every unresolved judge error; all execution failures audited separately.",
              "adjudications": decisions, "execution_failure_audits": failures}
    output = ROOT / "docs/evaluation/full-dev/adjudications.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {len(decisions)} explicit AI source checks; {sum(d['action']=='corrected' for d in decisions.values())} corrected; {len(failures)} execution failure audits. Human reviews: 0.")


if __name__ == "__main__":
    main()
