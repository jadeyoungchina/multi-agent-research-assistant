"""Replay explicit Codex source adjudications of the first blinded dev pilot.

This is an AI curation recipe, not automatic acceptance of a model verdict and
not human review. Raw DeepSeek judgments remain unchanged in the public packet.
The decisions below were checked against pinned source chunks and actual answers.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.runner import EvaluationReport
from app.evaluation.real_dataset import load_real_suite
from app.evaluation.semantic import SemanticJudgment, score_semantic_judgment
from scripts.judge_dev_pilot import decode_judgment


NOTES = {
    1: "The pinned handling-errors passage supports the three cited recommendations. Retain the reference's additional JSON-detail/import-alias requirements: the answer omits them. This reference is stricter than the question's wording; report that limitation, do not change gold after observing answers.",
    2: "Same source check as answer-001. The answer correctly identifies the raising and handler classes and the internal-Starlette rationale, but omits the reference's JSON-detail and import-alias extras.",
    3: "The cited handling-errors passage supports inheritance, JSON-able detail and the Starlette handler recommendation. The example import-alias key point is absent, so its original zero is retained.",
    4: "The recommendation that a FastAPI-subclass handler catches both base and subclass exceptions reverses inheritance. Naming Starlette as an alternative does not cure the later explicit wrong preference. Handler key point becomes zero. Baseline citation statuses are uncited, regardless of factual correctness; the factual-error flag remains true.",
    5: "Both statements are scoped acknowledgments that supplied evidence lacks measured latency, not external quantitative facts requiring citations. Use N/A citation statuses and retain correct abstention.",
    6: "Scoped corpus-absence statements are limitations, not claims that one cited chunk proves global absence. The factual Python-exception and inheritance/handler descriptions are supported by their attached source passages. Preserve the internal critic commentary as N/A process text and record it as a presentation defect. Accept the full contiguous citation group after checking its attachment; do not cut it off because long hashes exceed 160 characters.",
    7: "The corpus contains no production latency measurements. Invented microsecond ranges and benchmark percentages remain incorrect and unsupported. With no citations, factual spans use uncited; advice remains N/A.",
    8: "The first sentence is a scoped missing-evidence limitation without an attached citation, so N/A. The final contiguous citation group attaches to the subsequent description of the documents; it is not borrowed backward across the first sentence. No measurement is invented.",
    9: "The Chinese phrase '提示输入应为有效整数' faithfully translates the reference's valid-integer error, so key point 1 is fully met. Strict/lax statements are assessed in the question's int/string scope. The literal int example establishes that strict validation rejects rather than coerces this input; it need not repeat the prose verbatim.",
    10: "Baseline correctly states conversion and ValidationError but never describes the valid-integer error message, so retain the partial key-point score. Introductory question framing is N/A; all substantive uncited facts use uncited rather than supported/partial citation labels.",
    11: "The supplied strict-mode example and introductory paragraph jointly establish string-to-int coercion and the strict valid-integer error. Chinese paraphrases are faithful. Both attached citations bind to each full statement.",
    12: "The same two strict-mode chunks support the answer in the question's int/string scope. Both contiguous citations bind even when the second long hash extends beyond a raw 160-character window. The answer explicitly describes the valid-integer error.",
    13: "The answer contradicts its correct final use_cache=False instruction by inventing a necessary generator condition and a scope='call' mechanism. The conditions key point is zero under the rubric's contradicted rule. Pinned FastAPI sub-dependency docs and the installed matching 0.112.2 Depends signature have no such requirement. Its route-registration code is an API claim, not exempt illustration. Keep unsupported_assertion=true.",
    14: "The per-request cache passage supports the three prose claims. Both generated code examples lack an attached citation; they remain uncited even though the pinned document contains equivalent examples. ID validity alone must not produce perfect citation completeness.",
    15: "The source is pinned to FastAPI 0.112.2 in the corpus manifest, so the version context is established without repeating the version in every retrieved paragraph. The per-request cache statement is fully supported; the judge's version-only partial penalty is rejected.",
    16: "The two uncited summary sentences are factual repetitions and require citations, rather than becoming N/A because they are a summary. Preserve the partial verdict for the graph-recognition inference and full support for the attached cache/use_cache=False examples. Both key points remain correct.",
    17: "The answer explicitly demonstrates both runtime names, so the runtime-versus-editor key point is fully covered. A sentence about validation_alias has no citation of its own; do not borrow the next sentence's citation. Claims that serialization_alias controls initializer parameters are only partially supported, and 'likely applies' must not become certainty. Keep unsupported extra editor guidance visible despite correct core key points.",
    18: "The question permits a valid alternative to AliasChoices: alias='fname' plus populate_by_name=True is documented in fields.md and accepts both names in the installed matching Pydantic 2.8.2. Grant that key point full credit. Strip citation tokens out of judge-selected claim spans, not out of the actual answer. Merge a duplicated overlapping editor-override claim and keep its two actual attached citations. The erroneous serialization-alias/editor override stays partial and unsupported_assertion remains true.",
    19: "The prose method 'AliasChoices within validation_alias' is a sufficient faithful explanation; the question does not require a literal Field(...) code block. Grant the method key point full credit. Keep the unsupported static-analysis citation and partial/incorrect serialization_alias editor workaround; correct key points do not erase extra false assertions.",
    20: "Core AliasChoices and runtime/editor distinctions are correct; the Field import is explicitly noted. The final claim that native alias editor support requires separate fields/plugins conflicts with the pinned fields guide's alias workaround, so unsupported_assertion becomes true. Baseline citation metrics remain N/A.",
}


def main():
    root = ROOT / "artifacts/evaluation/real-v1-pilot"
    report = EvaluationReport.model_validate_json((root / "final/results.json").read_text("utf-8"))
    cases = {case.id: case for case in load_real_suite(
        ROOT / "benchmarks/real-v1/corpus-manifest.json", ROOT / "benchmarks/real-v1/dev.jsonl", "dev").cases}
    decisions = {}
    for index, trace in enumerate(report.traces, 1):
        unit = f"answer-{index:03d}"
        raw = json.loads((root / f"judge-final/{unit}.json").read_text("utf-8"))
        judgment = decode_judgment(raw["raw_response"]["text"])
        original = judgment.model_dump()
        # Citation labels and factual correctness are independent. Baseline facts
        # are uncited even when correct; non-factual limitations use N/A.
        if trace.variant.value == "baseline_llm":
            for claim in judgment.claims:
                claim.support = "uncited" if claim.requires_citation else "not_applicable"
        if index == 4:
            judgment.key_points[1].score = 0
            judgment.key_points[1].reason = "Wrongly recommends subclass handler and claims it catches base exceptions."
        elif index == 5:
            for claim in judgment.claims:
                claim.requires_citation, claim.support = False, "not_applicable"
        elif index == 6:
            for position in [0, 4, 6, 7, 8, 9]:
                judgment.claims[position].requires_citation = False
                judgment.claims[position].support = "not_applicable"
                judgment.claims[position].reason = "Scoped missing-evidence limitation; no measured fact asserted."
            marker = "证据不足："
            if marker in trace.answer:
                judgment.claims.append(type(judgment.claims[0])(
                    claim_text=marker + trace.answer.split(marker, 1)[1], requires_citation=False,
                    cited_evidence_ids=[], support="not_applicable",
                    reason="Internal critic process commentary; presentation defect, not a domain assertion."))
        elif index == 8:
            for position in [0, 2]:
                judgment.claims[position].requires_citation = False
                judgment.claims[position].support = "not_applicable"
            judgment.claims[0].cited_evidence_ids = []
        elif index == 9:
            judgment.key_points[1].score = 2
            judgment.key_points[1].reason = "Chinese valid-integer error phrase is a faithful translation."
            judgment.claims[3].support = "supported"
            judgment.claims[3].reason = "Int/string strict example explicitly rejects coercion in this question's scope."
        elif index == 10:
            judgment.claims[0].requires_citation, judgment.claims[0].support = False, "not_applicable"
        elif index == 12:
            judgment.claims[2].support = "supported"
            judgment.claims[2].reason = "Cited strict-mode prose and int example establish this scoped statement."
        elif index == 13:
            judgment.key_points[1].score = 0
            judgment.key_points[1].reason = "Invented generator prerequisite contradicts correct use_cache=False conclusion."
            judgment.claims[6].requires_citation, judgment.claims[6].support = True, "uncited"
            judgment.claims[6].claim_text = re.search(r"```python\n(.*?)\n\s*```", trace.answer, re.S).group(1).rstrip()
            judgment.claims[12].claim_text = trace.answer[trace.answer.index("**真正体现"):].splitlines()[0]
        elif index == 15:
            judgment.claims[0].support = "supported"
            judgment.claims[0].reason = "Version is established by pinned source identity; cache behavior explicit."
        elif index == 16:
            for position in [4, 5]:
                judgment.claims[position].requires_citation = True
                judgment.claims[position].support = "uncited"
        elif index == 17:
            judgment.key_points[1].score = 2
            judgment.key_points[1].reason = "Both runtime names are explicitly demonstrated; editor distinction stated."
            judgment.claims[3].cited_evidence_ids, judgment.claims[3].support = [], "uncited"
            judgment.claims[4].reason = "Alias is relevant; serialization_alias does not set initializer parameters."
            judgment.claims[6].support = "partial"
            judgment.claims[6].reason = "Source says likely; answer asserts certainty."
            judgment.unsupported_assertion = True
        elif index == 18:
            judgment.key_points[0].score = 2
            judgment.key_points[0].reason = "Documented alias plus populate_by_name alternative accepts both input names."
            for claim in judgment.claims:
                if "[[cite:" in claim.claim_text:
                    claim.claim_text = claim.claim_text.split("[[cite:", 1)[0].rstrip()
            judgment.claims[5].cited_evidence_ids = list(dict.fromkeys([
                *judgment.claims[5].cited_evidence_ids, *judgment.claims[6].cited_evidence_ids]))
            judgment.claims.pop(6)  # Same editor-override assertion, not a second atomic claim.
            for position in [0, 2, 3]:
                judgment.claims[position].support = "uncited"
            judgment.claims[5].reason = "Combined workaround exists; claimed editor override is actually serialization-only."
            judgment.claims[6].support = "supported"
            judgment.claims[6].reason = "Cited example explicitly works at runtime despite IDE warning."
        elif index == 19:
            judgment.key_points[0].score = 2
            judgment.key_points[0].reason = "Faithful prose method; literal Field(...) syntax not required by question."
            judgment.claims[3].reason = "Alias affects initialization; serialization_alias-only editor workaround is incorrect."
        elif index == 20:
            judgment.unsupported_assertion = True
        changed = judgment.model_dump() != original
        # Preserve the judge's original explanation separately in raw_response.
        if changed:
            judgment.explanation += " Codex source adjudication: " + NOTES[index]
        score_semantic_judgment(cases[trace.case_id], trace, judgment)
        decisions[unit] = {"action": "corrected" if changed else "accepted", "reviewer_kind": "ai",
                           "reason": NOTES[index], "trace_sha256": sha256(trace.model_dump_json().encode()).hexdigest(),
                           "judgment": judgment.model_dump()}
    packet = {"created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "Codex", "reviewer_kind": "ai",
              "human_reviewed": False, "generation_commit": report.metadata.git_commit,
              "original_judge_prompt_code_commit": report.metadata.git_commit,
              "adjudication_semantic_source_sha256": sha256((ROOT / "app/evaluation/semantic.py").read_bytes()).hexdigest(),
              "adjudications": decisions}
    path = ROOT / "docs/evaluation/dev-pilot/adjudications.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", "utf-8", newline="\n")
    print(f"Saved {len(decisions)} explicit AI source checks; {sum(item['action']=='corrected' for item in decisions.values())} corrected. Human scores: 0.")


if __name__ == "__main__":
    main()
