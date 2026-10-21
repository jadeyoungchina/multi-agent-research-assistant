from pathlib import Path

import pytest

from app.evaluation.metrics import score_case
from app.evaluation.models import WorkflowVariant
from app.evaluation.real_dataset import validate_real_benchmark
from app.evaluation.semantic import SemanticJudgment, blinded_judge_payload, score_semantic_judgment
from app.evaluation.trace import EvaluationTrace, EvidenceSnapshot


@pytest.fixture
def inputs():
    dev, _ = validate_real_benchmark(Path("benchmarks/real-v1"))
    case = dev.cases[0].model_copy(update={"answer_key_points": ["The answer must preserve the condition."]})
    trace = EvaluationTrace(
        case_id=case.id, variant=WorkflowVariant.LLM_RAG, status="success",
        answer="A measured improvement of 90 percent. [[cite:e1]]",
        retrieved_evidence=[EvidenceSnapshot(id="e1", source_file="source.md", page_number=None,
                                             chunk_index=0, text="This guide provides no measured performance data.")],
        cited_evidence_ids=["e1"], latency_ms=9, prompt_tokens=10, completion_tokens=5,
        model_calls=1, critic_loops=0, provider="secret-system-label", model="secret-model-label",
    )
    judgment = SemanticJudgment(
        rubric_version="real-semantic-v1", key_points=[dict(index=0, score=0, reason="fabricated number")],
        claims=[dict(claim_text="A measured improvement of 90 percent.", requires_citation=True,
                     cited_evidence_ids=["e1"], support="unsupported", reason="The source contains no measurement.")],
        abstained=False, unsupported_assertion=True, explanation="Citation ID exists but does not entail the claim.",
    )
    return case, trace, judgment


def test_valid_id_does_not_establish_claim_support(inputs):
    case, trace, judgment = inputs
    assert score_case(case, trace).citation_id_validity == 1
    semantic = score_semantic_judgment(case, trace, judgment)
    assert semantic.citation_support_rate == semantic.citation_completeness == 0
    assert semantic.answer_correctness == 0
    assert semantic.unsupported_assertion is True


def test_judge_payload_is_blinded_by_explicit_allowlist(inputs):
    import json
    case, trace, _ = inputs
    payload = blinded_judge_payload(case, trace)
    encoded = json.dumps(payload)
    assert not any(label in encoded for label in (trace.variant.value, trace.provider, trace.model))
    assert not {"variant", "model", "provider", "latency_ms", "model_calls", "human_review", "family", "question_type"} & payload.keys()
    assert payload["reference_key_points"] == case.answer_key_points
    assert payload["answer_evidence"][0]["id"] == "e1"


@pytest.mark.parametrize("kind", ["missing_key", "duplicate_key", "invented_claim", "unknown_id", "unbound_id", "empty_cited_verdict"])
def test_invalid_judge_output_is_an_error_not_a_zero_quality_score(inputs, kind):
    case, trace, judgment = inputs
    payload = judgment.model_dump()
    if kind == "missing_key":
        payload["key_points"] = []
    elif kind == "duplicate_key":
        payload["key_points"] *= 2
    elif kind == "invented_claim":
        payload["claims"][0]["claim_text"] = "An invented span absent from the answer."
    elif kind == "unknown_id":
        payload["claims"][0]["cited_evidence_ids"] = ["unknown"]
    elif kind == "unbound_id":
        trace = trace.model_copy(update={"answer": "A measured improvement of 90 percent."})
    else:
        payload["claims"][0]["cited_evidence_ids"] = []
    with pytest.raises(ValueError):
        score_semantic_judgment(case, trace, SemanticJudgment.model_validate(payload))


def test_abstention_and_inapplicable_citations_keep_their_denominators(inputs):
    dev, _ = validate_real_benchmark(Path("benchmarks/real-v1"))
    case = next(item for item in dev.cases if not item.expected_sufficiency)
    _, trace, _ = inputs
    trace = trace.model_copy(update={"case_id": case.id, "answer": "The corpus cannot establish the requested measurement.",
                                     "retrieved_evidence": [], "cited_evidence_ids": []})
    judgment = SemanticJudgment(rubric_version="real-semantic-v1", key_points=[dict(index=0, score=2, reason="correct limitation")],
                                claims=[], abstained=True, unsupported_assertion=False, explanation="Appropriate abstention.")
    result = score_semantic_judgment(case, trace, judgment)
    assert result.correct_abstention is True and result.false_abstention is None
    assert result.citation_support_rate is result.citation_completeness is None


def test_closed_book_baseline_has_no_citation_metric(inputs):
    case, trace, judgment = inputs
    result = score_semantic_judgment(case, trace.model_copy(update={"variant": WorkflowVariant.BASELINE_LLM}), judgment)
    assert result.citation_support_rate is result.citation_completeness is None


def test_adjacent_citation_group_does_not_fail_due_to_long_hashes():
    from app.evaluation.semantic import _bound_citations
    ids = ["ev_" + digit * 64 for digit in "12345"]
    answer = "A supported claim. " + ", ".join(f"[[cite:{identifier}]]" for identifier in ids)
    assert _bound_citations(answer, "A supported claim.") == set(ids)
    assert not _bound_citations("A supported claim.\n\nAnother paragraph. [[cite:e1]]", "A supported claim.")
    assert _bound_citations("A supported claim. [[cite:e1]] Another claim. [[cite:e2]]", "A supported claim.") == {"e1"}
