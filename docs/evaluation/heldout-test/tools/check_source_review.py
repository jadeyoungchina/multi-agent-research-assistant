"""Offline review-binding regression checks over retained public-source traces."""
from copy import deepcopy
import json
import unittest

import source_review as review


class ReviewContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        record = review.validate_freeze(require_committed=False)
        cases = {c.id: c for c in review.load_real_suite(review.SUITE / "corpus-manifest.json", review.SUITE / "test.jsonl", "test").cases}
        for index, trace in enumerate(review.load_repeat(record, review.RUN, 0).traces, 1):
            original = json.loads((review.RUN / "judgments" / f"r0-answer-{index:03d}.json").read_text("utf-8"))
            case = cases[trace.case_id]
            if trace.status != "success" or trace.variant.value == "baseline_llm" or not case.expected_sufficiency or original["status"] != "valid_ai_judgment":
                continue
            judgment = review.TestJudgment.model_validate(original["judgment"])
            if not any(b.requires_citation and b.support in {"supported", "partial"} for b in judgment.blocks):
                continue
            cls.case, cls.trace, cls.original = case, trace, original
            source, text = next(iter(review.reference_context(case).items()))
            quote = text[:100]
            checks = [review.PointCheck(index=i, proofs=[review.SourceProof(source_file=source, quote=quote)], reason="Fixture tests membership binding, not semantic quality.") for i in range(len(case.answer_key_points))]
            blocks = {b["index"]: b for b in review.answer_blocks(trace.answer)}
            evidence = {e.id: e.text for e in trace.retrieved_evidence}
            proofs = []
            for block in judgment.blocks:
                if block.requires_citation and block.support in {"supported", "partial"}:
                    evidence_id = blocks[block.index]["citation_ids"][0]
                    proofs.append(review.CitationProof(block_index=block.index, evidence_id=evidence_id, quote=evidence[evidence_id][:100]))
            cls.audit = review.SourceAudit(judgment=judgment, key_point_checks=checks, citation_proofs=proofs, review_reason="Offline fixture; not a source adjudication.")
            return
        raise AssertionError("No retained successful cited fixture")

    def test_valid_membership_and_exhaustive_coverage(self):
        review.validate_audit(self.case, self.trace, self.audit.model_copy(deep=True))

    def test_fabricated_source_quote_rejected(self):
        audit = self.audit.model_copy(deep=True)
        audit.key_point_checks[0].proofs[0].quote = "This fabricated quote is absent from the pinned source 987654."
        with self.assertRaisesRegex(ValueError, "exact pinned source"):
            review.validate_audit(self.case, self.trace, audit)

    def test_missing_reference_point_check_rejected(self):
        audit = self.audit.model_copy(deep=True)
        audit.key_point_checks.pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            review.validate_audit(self.case, self.trace, audit)

    def test_unattached_citation_proof_rejected(self):
        audit = self.audit.model_copy(deep=True)
        audit.citation_proofs[0].evidence_id = "not-attached-to-this-block"
        with self.assertRaisesRegex(ValueError, "attached to the exact block"):
            review.validate_audit(self.case, self.trace, audit)

    def test_supported_block_requires_literal_passage_proof(self):
        audit = self.audit.model_copy(deep=True)
        audit.citation_proofs.clear()
        with self.assertRaisesRegex(ValueError, "lacks an attached passage quote"):
            review.validate_audit(self.case, self.trace, audit)

    def test_accepted_review_preserves_whole_original_including_reasons(self):
        updated = self.audit.judgment.model_copy(deep=True)
        updated.key_points[0].reason = "Different audit explanation; labels are identical."
        action, judgment = review.decision(self.original, updated)
        self.assertEqual(action, "accepted")
        self.assertEqual(judgment, self.original["judgment"])

    def test_changed_label_is_explicit_and_original_is_unchanged(self):
        previous = deepcopy(self.original)
        updated = self.audit.judgment.model_copy(deep=True)
        updated.key_points[0].score = 0 if updated.key_points[0].score else 2
        action, judgment = review.decision(self.original, updated)
        self.assertEqual(action, "corrected")
        self.assertNotEqual(judgment, self.original["judgment"])
        self.assertEqual(self.original, previous)

    def test_audit_prompt_hides_generation_metadata_and_original_labels(self):
        prompt = review.prompt_for(self.case, self.trace)
        payload = json.loads(prompt.split("\nData:\n", 1)[1])
        self.assertFalse({"variant", "provider", "model", "latency_ms", "cost", "original_judgment"} & payload.keys())
        self.assertEqual(payload["reference_key_points"], self.case.answer_key_points)


if __name__ == "__main__":
    unittest.main()
