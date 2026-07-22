from copy import deepcopy
from typing import Any, TypedDict


class WorkflowState(TypedDict, total=False):
    run_id: str
    question: str
    document_ids: list[str]
    plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    synthesis: dict[str, Any]
    critique: dict[str, Any]
    draft: dict[str, Any]
    report: dict[str, Any]
    iteration: int
    next_stage: str
    last_completed_stage: str | None
    provider_metrics: list[dict[str, Any]]
    errors: list[dict[str, str]]


def initial_state(run_id: str, question: str, document_ids: list[str]) -> WorkflowState:
    return WorkflowState(
        run_id=run_id,
        question=question,
        document_ids=list(document_ids),
        evidence=[],
        iteration=0,
        next_stage="planner",
        last_completed_stage=None,
        provider_metrics=[],
        errors=[],
    )


def merge_state(state: WorkflowState, updates: dict[str, Any]) -> WorkflowState:
    merged = deepcopy(dict(state))
    merged.update(deepcopy(updates))
    return WorkflowState(**merged)
