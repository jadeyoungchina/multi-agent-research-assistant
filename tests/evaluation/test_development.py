from pathlib import Path

import pytest

from app.evaluation.development import DevelopmentJudgment, answer_blocks, load_scoring_cases, score_development
from app.evaluation.real_dataset import load_real_suite
from app.evaluation.trace import EvaluationTrace
from app.evaluation.models import WorkflowVariant


def test_clarifications_are_dev_only_and_preserve_the_original_pilot_references():
    root = Path("benchmarks/real-v1")
    original = {case.id: case for case in load_real_suite(root / "corpus-manifest.json", root / "dev.jsonl", "dev").cases}
    scoring = load_scoring_cases(root)
    assert len(scoring) == 20 and all(case.split == "dev" for case in scoring.values())
    assert len(original["REAL-DEV-019"].answer_key_points) == 3
    assert len(scoring["REAL-DEV-019"].answer_key_points) == 2
    for identifier in original:
        assert original[identifier].question == scoring[identifier].question
        assert original[identifier].expected_evidence == scoring[identifier].expected_evidence


def test_blocks_preserve_code_and_citation_groups_without_losing_uncited_prose():
    blocks = answer_blocks("# Heading\n\nAn uncited assertion.\n\n```python\na = 1\n\nprint(a)\n```\n\n[[cite:e1]] [[cite:e2]]\n\n- Another fact.")
    assert len(blocks) == 4
    assert blocks[1]["text"] == "An uncited assertion."
    assert "\n\nprint(a)" in blocks[2]["text"]
    assert blocks[2]["citation_ids"] == ["e1", "e2"]


def test_block_judge_cannot_omit_uncited_facts_or_turn_missing_citations_into_support():
    cases = load_scoring_cases(Path("benchmarks/real-v1"))
    case = cases["REAL-DEV-001"]
    trace = EvaluationTrace(case_id=case.id, variant=WorkflowVariant.BASELINE_LLM, status="success",
                            answer="# Answer\n\nAn uncited factual assertion.", latency_ms=0, prompt_tokens=0,
                            completion_tokens=0, model_calls=0, critic_loops=0, provider="fake", model="fake")
    judgment = DevelopmentJudgment(rubric_version="dev-block-v2",
        key_points=[dict(index=i, score=1, reason="partial") for i in range(len(case.answer_key_points))],
        blocks=[dict(index=0, requires_citation=False, support="not_applicable", reason="heading"),
                dict(index=1, requires_citation=True, support="uncited", reason="no citation")],
        abstained=False, unsupported_assertion=False, explanation="partial answer")
    assert score_development(case, trace, judgment).answer_correctness == 0.5
    assert score_development(case, trace, judgment).citation_completeness is None
    with pytest.raises(ValueError, match="incomplete"):
        score_development(case, trace, judgment.model_copy(update={"blocks": judgment.blocks[:1]}))
    judgment.blocks[1].support = "supported"
    with pytest.raises(ValueError, match="disagrees"):
        score_development(case, trace, judgment)
