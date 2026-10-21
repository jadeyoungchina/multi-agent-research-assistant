"""Verify AI-review provenance separately from actual human approval."""

from hashlib import sha256
import json
from pathlib import Path

from .real_dataset import RealBenchmarkCase, validate_real_benchmark


def reference_digest(case: RealBenchmarkCase) -> str:
    """Bind the substantive reference, never promote a human-review flag."""
    return sha256(json.dumps(case.model_dump(mode="json", exclude={"human_review", "provenance"}),
                             ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_ai_reference_review(root: Path) -> dict:
    """Check completeness and staleness; this cannot establish semantic truth."""
    suites = validate_real_benchmark(root)
    report = json.loads((root / "review/final-audit.json").read_text("utf-8"))
    if report.get("schema_version") != "ai-reference-review-v1" or report.get("review_kind") != "ai_double_review":
        raise ValueError("unsupported AI reference review")
    if report.get("corpus_manifest_sha256") != sha256((root / "corpus-manifest.json").read_bytes()).hexdigest():
        raise ValueError("AI review corpus snapshot is stale")
    # Byte hashes preserve the originally audited artifact. Per-case reference
    # hashes below decide semantic staleness, so a later human-status update or
    # JSON whitespace change cannot erase an otherwise current AI review.
    cases = {case.id: case for suite in suites for case in suite.cases}
    reviews = report.get("reviews", [])
    if len(reviews) != len(cases) or {item["id"] for item in reviews} != set(cases):
        raise ValueError("AI review must cover every case exactly once")
    if report.get("accepted_count") != len(reviews):
        raise ValueError("AI review accepted count is inconsistent")
    for item in reviews:
        if item.get("reference_sha256") != reference_digest(cases[item["id"]]):
            raise ValueError("AI review case reference is stale")
        if item.get("status") != "accepted_for_ai_reviewed_pilot" or not item.get("codex_adjudication"):
            raise ValueError("AI reference review is incomplete")
        if not item.get("deepseek_tasks"):
            raise ValueError("AI review lacks delegated review provenance")
    for suite in suites:
        split = suite.cases[0].split
        audit = report.get("absence_audit", {}).get(split, {})
        if sorted(audit.get("covered_sources", [])) != sorted(suite.sources):
            raise ValueError("absence audit must cover the whole candidate pool")
        negatives = {case.id for case in suite.cases if not case.expected_sufficiency}
        conclusions = audit.get("conclusions", {})
        if set(conclusions) != negatives or any(value is not False for value in conclusions.values()):
            raise ValueError("insufficiency review is incomplete or contradicts the labels")
    return report
