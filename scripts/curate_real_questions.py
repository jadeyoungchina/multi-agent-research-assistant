"""Import reviewed AI drafts with literal source locations and pending human status.

This is a deterministic conversion, not a semantic or human approval mechanism.
Curator corrections are kept in a separately reviewable overrides JSON file.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.real_dataset import RealBenchmarkCase, load_corpus_manifest, load_real_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, default=ROOT / "artifacts/evaluation/step-c-drafts")
    parser.add_argument("--overrides", type=Path, default=ROOT / "benchmarks/real-v1/curation-overrides.json")
    args = parser.parse_args()
    suite = ROOT / "benchmarks/real-v1"
    manifest = load_corpus_manifest(suite / "corpus-manifest.json")
    overrides = json.loads(args.overrides.read_text("utf-8")) if args.overrides.exists() else {}
    rows = {"dev": [], "test": []}
    for index, family in enumerate(dict.fromkeys(doc.family for doc in manifest.documents)):
        split = next(doc.split for doc in manifest.documents if doc.family == family)
        response = json.loads((args.draft_dir / f"{family}.response.json").read_text("utf-8"))
        raw = response["text"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        drafts = json.loads(raw)
        if len(drafts) != 5:
            raise ValueError("one candidate of each type required per family")
        for draft in drafts:
            correction = overrides.get(f"{family}/{draft['type']}", {})
            draft.update(correction)
            located = []
            for evidence in draft["evidence"]:
                text = (suite / "corpus" / evidence["source_file"]).read_text("utf-8")
                offset = text.find(evidence["quote"])
                if offset < 0:
                    raise ValueError(f"non-verbatim quote: {family}/{draft['type']}")
                first = text[:offset].count("\n") + 1
                last = first + evidence["quote"].count("\n")
                located.append(dict(source_file=evidence["source_file"], contains=evidence["quote"],
                                    start_line=first, end_line=last))
            insufficient = draft["type"] == "insufficient_corpus_evidence"
            points = draft["key_points"]
            if insufficient:
                points = (["The complete indexed split corpus does not establish the requested deployment-specific measurement; do not invent a number."]
                          if index % 2 else ["完整候选语料没有提供问题要求的部署特定实测结果；应说明证据不足，不能编造数值。"])
            identifier = f"REAL-{split.upper()}-{len(rows[split])+1:03d}"
            case = RealBenchmarkCase(
                id=identifier, split=split, language="zh" if index % 2 == 0 else "en",
                question_type=draft["type"], family=family, question_group=identifier,
                question=draft["question"], answer_key_points=points, expected_evidence=located,
                expected_sufficiency=not insufficient,
                abstention_policy="If the indexed corpus cannot support the requested conclusion, explicitly state that limitation and do not invent facts.",
                insufficiency_audit=(
                    f"AI curator inspected the full {split} pool, including performance-related paragraphs. "
                    "Qualitative performance guidance and the v1 construct() approximate speedup do not establish "
                    "the deployment-specific benchmark asked here. No matching measured workload, hardware, "
                    "or production result was found. This is an AI audit; human confirmation remains pending."
                ) if insufficient else None,
                provenance=dict(model="deepseek-flash", task_id=response["task_id"], ai_review="checked"),
            )
            rows[split].append(case)
    for split, cases in rows.items():
        (suite / f"{split}.jsonl").write_text("".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8", newline="\n")
        load_real_suite(suite / "corpus-manifest.json", suite / f"{split}.jsonl", split, require_design_counts=True)
    print(f"Imported {sum(map(len, rows.values()))} AI-authored candidates; human reviews remain pending.")


if __name__ == "__main__":
    main()
