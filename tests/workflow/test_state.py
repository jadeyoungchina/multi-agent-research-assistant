import json

from app.workflow.state import initial_state, merge_state


def test_initial_state_starts_at_planner() -> None:
    state = initial_state("run1", "What changed?", ["doc1"])

    assert state["iteration"] == 0
    assert state["next_stage"] == "planner"
    assert state["evidence"] == []
    assert state["provider_metrics"] == []


def test_initial_state_copies_document_ids() -> None:
    document_ids = ["doc1"]

    state = initial_state("run1", "question", document_ids)
    document_ids.append("doc2")

    assert state["document_ids"] == ["doc1"]


def test_state_json_round_trip_preserves_writer_draft() -> None:
    state = initial_state("run1", "question", ["doc1"])
    updated = merge_state(
        state,
        {"next_stage": "citation_validator", "draft": {"title": "Draft"}},
    )

    assert json.loads(json.dumps(updated, ensure_ascii=False))["draft"]["title"] == "Draft"


def test_merge_state_copies_updates_without_mutating_original_state() -> None:
    state = initial_state("run1", "question", ["doc1"])
    updates = {"draft": {"title": "Draft"}}

    updated = merge_state(state, updates)
    updates["draft"]["title"] = "Changed"
    updated["draft"]["title"] = "Updated"

    assert state.get("draft") is None
    assert updated["draft"]["title"] == "Updated"
    assert updates["draft"]["title"] == "Changed"
