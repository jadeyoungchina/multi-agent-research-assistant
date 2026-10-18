"""Publish the explicitly source-checked AI adjudication, preserving disagreements.

This publishes review provenance, not model-performance results or human
approval. DeepSeek feedback is advisory; Codex's final source checks and
corrections are the acceptance decision. Stale hashes invalidate acceptance.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.evaluation.budget import BudgetLedger
from app.evaluation.real_dataset import validate_real_benchmark
from app.evaluation.reference_review import reference_digest, validate_ai_reference_review
from app.retrieval.chunking import chunk_pages
from app.retrieval.loaders import load_document


ADJUDICATIONS = {
    "REAL-DEV-002": "The complete quoted exactness list explicitly says highest-to-lowest and places strict above lax. Multiline quotes are permitted and literal-validated. The question now explicitly requests input-type and Python/JSON distinctions.",
    "REAL-DEV-003": "The quoted UUID type error establishes the required Python instance type; the neighboring UUID-specific sentence explicitly allows JSON strings. Source statements are interpreted semantically, not by identical wording.",
    "REAL-DEV-004": "Added the explicit Union[int,str] declaration. The selected code shows the left-to-right int result; the documented exact-type rule establishes the smart str result for these primitive types. The nearby prose has a 123/456 typo. The reviewer's three-quote limit applies to its suggestions, not dataset gold units.",
    "REAL-DEV-007": "Replaced the weak cross-document question: AliasChoices input alternatives come from alias.md; VSCode initializer-signature limitations come from fields.md. fields.md contains no AliasChoices example. The question does not request the editor-signature workaround.",
    "REAL-DEV-009": "Consolidated overlapping gold and made simultaneous-parameter conditions explicit. Field(alias=...) is schematic prose, not a claim that an ellipsis is a valid alias. The reviewer's alleged required positional argument is not a documented restriction.",
    "REAL-DEV-011": "The source only gives explicit numeric defaults for skip and limit. Narrowed the question to ask those defaults, rather than requiring an unshown default for q.",
    "REAL-DEV-012": "Added the Depends declaration and outermost-dependency passages, supplementing the function-parameter and sub-dependency resolution passages.",
    "REAL-DEV-013": "Added the explicit per-request cache and passing-to-dependants sentence. The inline use_cache=False declaration is present without expanding external code includes.",
    "REAL-DEV-014": "The task concerns coexisting current forms, not a historical minimum-version requirement. Removed the unnecessary historical-version scoring point. A 0.95.0 introduction and 0.95.1 upgrade recommendation are different statements, not an API contradiction.",
    "REAL-DEV-017": "Expanded the status_code documentation and utility-exception passages so the declared-status and immediate-error paths are both grounded.",
    "REAL-DEV-018": "Added the logging statement, 500 client response and internal-information limitation as separate supported units.",
    "REAL-DEV-019": "Selected the complete JSON-able detail difference, Starlette handler recommendation, reason for catching internal exceptions and the renamed import.",
    "REAL-TEST-001": "Replaced the integer-coercion question, which was too similar to a dev question, with explicit-field tracking via __fields_set__. Type/language/family counts remain unchanged before any paid held-out run.",
    "REAL-TEST-002": "Added both versions' BaseModel inheritance statements and output-side guarantees. The output-model/output wording distinction does not change the shared claim.",
    "REAL-TEST-003": "Made arbitrary-class/ORM scope explicit. Split the GetterDict source into reachable semantic units without losing the dictionary-interface or getter_dict configuration facts.",
    "REAL-TEST-004": "Corrected model.dump to model_dump. The assertion and the statement that either API provides a field dictionary jointly support flat-model parity. The 180-character limit constrains reviewer suggestions, not gold excerpts.",
    "REAL-TEST-006": "Added the second-argument and signature-extra passages. The source explicitly says the second argument is always the field value, contrary to the initial reviewer's concern.",
    "REAL-TEST-007": "Selected the v1 default/always=True passages and the full v2 default-validation paragraph. always=True is literally present in the selected v1 text; the reviewer's claim that it is absent is rejected.",
    "REAL-TEST-008": "Selected the complete validator-order sentence and the single-PlainValidator condition; the shorter units fit the actual chunk boundaries.",
    "REAL-TEST-009": "Added the v2 class-method statement and recommended decorator order for type checking, matching the requested comparison.",
    "REAL-TEST-011": "Removed an unrequested dumps_kwargs point; the question only asks where encoder is passed.",
    "REAL-TEST-012": "Selected both API headings, both recursive-dump statements and both raw dict(model) statements.",
    "REAL-TEST-013": "Clarified that the when_used default refers to PlainSerializer/WrapSerializer, separately from the all-fields field_serializer argument.",
    "REAL-TEST-014": "Completed the RootModel recursive-dump statement and added the raw-value/non-recursive dict(model) passage.",
    "REAL-TEST-016": "Completed the config/model_config sentence instead of quoting fragments that stop before the comparison.",
    "REAL-TEST-017": "Reworked the cross-document question to require dataclass wrapping and the separate TypeAdapter reuse recommendation. No quantified performance improvement is inferred.",
    "REAL-TEST-018": "Selected the complete bytes/str return-type and backwards-compatibility explanation.",
    "REAL-TEST-019": "Narrowed direct adapter operations to validation and JSON-schema generation, the operations actually stated by the source.",
    "REAL-TEST-022": "Clarified the question and selected default jsonable_encoder/JSONResponse processing, output-model filtering and direct Response pass-through.",
    "REAL-TEST-024": "Selected separate reachable units for invalid data, app-code error and server error, plus pass-through and no automatic validation for direct Response. The key concerns validation, not a claim that filtering causes the error.",
    "REAL-TEST-026": "Completed the re-export origin and singular/plural BackgroundTask naming explanation.",
    "REAL-TEST-027": "Aligned cleanup wording with after request handling, immediately before shutdown. Background tasks run after returning a response.",
    "REAL-TEST-029": "Selected the explicit deprecated-alternative heading and lifespan-or-events warning; removed an unnecessary registration-possibility claim.",
    "REAL-TEST-031": "Removed the unrequested extra guarantee about token parameter type; the question only asks the missing-header behavior.",
    "REAL-TEST-032": "Added the endpoint-declaration, fixed form names and Bearer-header passages. Endpoint creation is not attributed to tokenUrl; token-response field details are outside the question's stated scope.",
    "REAL-TEST-033": "Made the enforced grant_type=password behavior explicit for OAuth2PasswordRequestFormStrict.",
    "REAL-TEST-034": "Made optional scope input and its parsed scopes output relationship explicit, with both full supporting sentences.",
    "REAL-TEST-036": "Selected both dependency-exit and background-task timing sentences.",
    "REAL-TEST-037": "Added allowed-origin setup and explicit middleware registration. Used separate reachable expose_headers and CORS-configuration units instead of a long HTML-containing passage that crosses a chunk boundary.",
    "REAL-TEST-038": "Selected API-labelled defaults for allow_methods, allow_headers and max_age rather than isolated numbers.",
    "REAL-TEST-039": "Selected the complete allow_credentials condition, including its restriction on wildcard origins.",
}


def main():
    root = ROOT / "benchmarks/real-v1"
    artifacts = ROOT / "artifacts/evaluation/ai-reference-review"
    dev, test = validate_real_benchmark(root)
    cases = [*dev.cases, *test.cases]
    tasks = []
    for item in json.loads((artifacts / "final/task-index.json").read_text("utf-8")):
        response = json.loads((artifacts / "final" / item["response_path"]).read_text("utf-8"))
        raw = response["text"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        reviews = json.loads(raw)
        if len(reviews) != len(item["references"]) or {r["id"] for r in reviews} != set(item["references"]):
            raise ValueError("delegated review is incomplete")
        prompt = (artifacts / "final" / f"{item['name']}.prompt.txt").read_text("utf-8")
        documents = [name for suite in (dev, test) for name in suite.sources if f"\nSOURCE_FILE: {name}\n" in prompt]
        tasks.append({**item, "documents": documents, "input_sha256": response["input_sha256"],
                      "attempt_ids": response["attempt_ids"], "feedback": reviews,
                      "response_text_sha256": sha256(response["text"].encode()).hexdigest()})
    settings = Settings(_env_file=None)
    unit_count = 0
    for suite in (dev, test):
        chunks = {name: chunk_pages("audit", name, load_document(src.content, name, src.media_type),
                                    settings.chunk_size, settings.chunk_overlap) for name, src in suite.sources.items()}
        for case in suite.cases:
            for evidence in case.expected_evidence:
                unit_count += 1
                if not any(evidence.contains.casefold() in chunk.content.casefold() for chunk in chunks[evidence.source_file]):
                    raise ValueError("gold evidence is unreachable with the declared chunk parameters")
    absence = {}
    for suite in (dev, test):
        split = suite.cases[0].split
        selected = [task for task in tasks if task["name"].startswith(f"absence-{split}-")]
        if sorted(name for task in selected for name in task["documents"]) != sorted(suite.sources):
            raise ValueError("absence review did not see every full source exactly once")
        for task in selected:
            if task["status"] != "validated" or any(row["found_answer"] for row in task["feedback"]):
                raise ValueError("negative label requires adjudication before acceptance")
        absence[split] = dict(covered_sources=list(suite.sources), task_ids=[t["task_id"] for t in selected],
                              conclusions={c.id: False for c in suite.cases if not c.expected_sufficiency})
    corrections = json.loads((root / "review/corrections.json").read_text("utf-8"))["changes"]
    changed = {entry["id"] for entry in corrections}
    reviews = []
    for case in cases:
        relevant = [task for task in tasks if case.id in task["references"]]
        note = ADJUDICATIONS.get(case.id, (
            "All complete candidate-pool shards reported no answer to the requested deployment-specific measurement. "
            "Codex checked every performance-related passage: qualitative advice, unrelated approximate speedups, "
            "configuration durations and timing-example code do not establish the requested benchmark."
        ) if not case.expected_sufficiency else "Question scope, reference points and literal supporting passages were checked; no unresolved substantive issue was found.")
        reviews.append(dict(id=case.id, reference_sha256=reference_digest(case),
                            status="accepted_for_ai_reviewed_pilot", corrected=case.id in changed,
                            deepseek_tasks=[t["task_id"] for t in relevant], codex_adjudication=note))
    # Include every physical audit call, including superseded responses and
    # rejected non-verbatim suggestions; resumed task responses do not add usage.
    identifiers = set()
    for path in artifacts.rglob("*.json"):
        payload = json.loads(path.read_text("utf-8"))
        if isinstance(payload, dict) and "input_sha256" in payload and "text" in payload:
            identifiers.update(payload["attempt_ids"])
    rows = [r for r in BudgetLedger(ROOT / "artifacts/evaluation/budget/ledger.sqlite").summary()["attempts"] if r["id"] in identifiers]
    report = dict(schema_version="ai-reference-review-v1", review_kind="ai_double_review",
                  reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                  reviewers=["DeepSeek Flash (fresh official CLI contexts)", "Codex (source adjudication)"],
                  corpus_manifest_sha256=sha256((root / "corpus-manifest.json").read_bytes()).hexdigest(),
                  dataset_sha256={name: sha256((root / name).read_bytes()).hexdigest() for name in ("dev.jsonl", "test.jsonl")},
                  accepted_count=len(reviews), corrected_count=len(changed), human_approved_count=sum(c.human_review.status == "approved" for c in cases),
                  source_checks=dict(located_gold_units=unit_count, all_gold_units_reachable=True,
                                     chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
                  scope="Reference-only AI adjudication before any paid held-out workflow run. Codex checks the final reference hash; DeepSeek feedback retains its own reviewed hash and may precede final Codex corrections.",
                  limitations=["AI authorship/review may retain correlated errors; no human approval or human judge calibration is claimed.",
                               "Negative questions currently emphasize absent deployment-performance measurements.",
                               "Reviewer suggestions can contain mistakes; rejected feedback and Codex dispositions are preserved.",
                               "This is not a workflow-quality result and does not establish admissions outcomes."],
                  usage=dict(physical_requests=len(rows), prompt_tokens=sum(r["input_tokens"] or 0 for r in rows),
                             completion_tokens=sum(r["output_tokens"] or 0 for r in rows),
                             cached_input_tokens=sum(r["cached_tokens"] or 0 for r in rows), total_cost_cny=None,
                             monetary_cap_enforced=False),
                  tasks=tasks, absence_audit=absence, reviews=reviews)
    (root / "review/final-audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    validate_ai_reference_review(root)
    print(f"Published {len(reviews)} AI-adjudicated references, {len(changed)} corrected cases; human approved={report['human_approved_count']}.")


if __name__ == "__main__":
    main()
