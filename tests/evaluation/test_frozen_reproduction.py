"""Protect historical analysis while allowing later application fixes."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "frozen_reproduction", ROOT / "docs/evaluation/heldout-test/tools/reproduce_frozen.py")
reproduction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduction)


def test_changed_current_answer_packet_cannot_be_hidden_by_old_checkout(tmp_path):
    for name in reproduction.public_paths():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    reproduction.validate_public_packet(tmp_path)
    trace = tmp_path / "docs/evaluation/heldout-test/repeat-0/traces.jsonl"
    with trace.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(RuntimeError, match="Published data differ.*traces.jsonl"):
        reproduction.validate_public_packet(tmp_path)


def test_archived_checkout_is_pinned_and_edited_verifiers_are_rejected(tmp_path):
    checkout = (tmp_path / "original-experiment").resolve()
    helper = checkout / "docs/evaluation/heldout-test/tools/verify_reviewed.py"
    notebook_path = checkout / "output/jupyter-notebook/real-evaluation-analysis.ipynb"
    original = None
    original_notebook = None
    try:
        assert reproduction.prepare_checkout(checkout) == checkout
        assert reproduction.git("rev-parse", "HEAD", directory=checkout) == reproduction.BASELINE
        original = helper.read_bytes()
        helper.write_bytes(original + b"\n# edited verifier\n")
        with pytest.raises(RuntimeError, match="unexpected edits"):
            reproduction.prepare_checkout(checkout)
        helper.write_bytes(original)
        original_notebook = notebook_path.read_bytes()
        notebook = json.loads(original_notebook)
        first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
        first_code["source"].append("\nraise RuntimeError('edited analysis source')\n")
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
        with pytest.raises(RuntimeError, match="notebook source"):
            reproduction.prepare_checkout(checkout)
    finally:
        if original is not None:
            helper.write_bytes(original)
        if original_notebook is not None:
            notebook_path.write_bytes(original_notebook)
        if checkout.exists():
            assert checkout.is_relative_to(tmp_path.resolve()) and checkout != ROOT
            subprocess.run(["git", "worktree", "remove", str(checkout)], cwd=ROOT, check=True)
