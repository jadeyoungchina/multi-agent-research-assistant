"""Verify or execute the published analysis against its original Git snapshot."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASELINE = "7e0cf6cdf167e112775b1376458b1e7ceff1af1a"
FREEZE_ID = "0fc19f9d47679eaf55ebeddd7c6857b36779b632db6d5dd644cfda6c6aeb3270"
PACKET = "docs/evaluation/heldout-test"
DEFAULT_CHECKOUT = ROOT / "artifacts/evaluation/frozen-reproduction" / BASELINE
REGENERATED = {"output/jupyter-notebook/real-evaluation-analysis.ipynb"} | {
    f"{PACKET}/figures/{name}.{extension}"
    for name in ("workflow-comparison", "paired-quality", "citations-and-failures")
    for extension in ("png", "svg")
} | {f"{PACKET}/figures/{name}" for name in
     ("analysis-tables.json", "analysis-provenance.json", "question-means.csv")}


def git(*args, directory=ROOT, raw=False):
    result = subprocess.check_output(["git", *args], cwd=directory)
    return result if raw else result.decode("utf-8").strip()


def public_paths():
    paths = git("ls-tree", "-r", "--name-only", BASELINE, "--", PACKET).splitlines()
    return ["benchmarks/real-v1/frozen-experiment.json"] + sorted(
        path for path in paths if Path(path).suffix in {".json", ".jsonl"}
        and not path.startswith(f"{PACKET}/figures/"))


def validate_public_packet(published_root=ROOT):
    """Do not silently verify an old copy if the current published data changed."""
    for name in public_paths():
        expected = git("show", f"{BASELINE}:{name}", raw=True)
        path = Path(published_root) / name
        if not path.is_file() or sha256(path.read_bytes()).digest() != sha256(expected).digest():
            raise RuntimeError(f"Published data differ from the baseline: {name}")
    record = json.loads((Path(published_root) / "benchmarks/real-v1/frozen-experiment.json").read_text("utf-8"))
    if record["freeze_id"] != FREEZE_ID:
        raise RuntimeError("Unexpected experiment identity.")


def prepare_checkout(destination=DEFAULT_CHECKOUT):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT.is_relative_to(destination):
        raise RuntimeError("The isolated checkout must not replace the project directory.")
    try:
        git("cat-file", "-e", f"{BASELINE}^{{commit}}")
    except subprocess.CalledProcessError:
        raise RuntimeError("The original commit is unavailable; use a checkout with full branch history.") from None
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "worktree", "add", "--quiet", "--detach",
                        str(destination), BASELINE], cwd=ROOT, check=True)
    if Path(git("rev-parse", "--show-toplevel", directory=destination)).resolve() != destination:
        raise RuntimeError("The reproduction path is not its own Git checkout.")
    if git("rev-parse", "HEAD", directory=destination) != BASELINE:
        raise RuntimeError("The reproduction checkout is at a different commit.")
    changed = set(git("diff", "--name-only", BASELINE, directory=destination).splitlines())
    untracked = git("ls-files", "--others", "--exclude-standard", directory=destination)
    if changed - REGENERATED or untracked:
        raise RuntimeError("The isolated checkout contains unexpected edits; existing files were left untouched.")
    # Check executable analysis helpers against committed bytes before launching them.
    helpers = git("ls-tree", "-r", "--name-only", BASELINE, "--", f"{PACKET}/tools").splitlines()
    for name in helpers:
        if name.endswith(".py"):
            original = git("show", f"{BASELINE}:{name}", raw=True).replace(b"\r\n", b"\n")
            if (destination / name).read_bytes().replace(b"\r\n", b"\n") != original:
                raise RuntimeError("An archived analysis helper differs from its committed version.")
    notebook_name = "output/jupyter-notebook/real-evaluation-analysis.ipynb"
    original_notebook = json.loads(git("show", f"{BASELINE}:{notebook_name}", raw=True))
    actual_notebook = json.loads((destination / notebook_name).read_text("utf-8"))

    def sources(notebook):
        return notebook["nbformat"], [
            (cell["cell_type"], "".join(cell["source"]).replace("\r\n", "\n"))
            for cell in notebook["cells"]]

    if sources(actual_notebook) != sources(original_notebook):
        raise RuntimeError("The archived notebook source differs from its committed version.")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", action="store_true", help="Execute all analysis cells and export figures in the original checkout.")
    args = parser.parse_args()
    validate_public_packet()
    checkout = prepare_checkout()
    print(f"Original experiment snapshot: {BASELINE}", flush=True)
    print(f"Isolated checkout: {checkout}", flush=True)
    helper = "run_analysis_notebook.py" if args.notebook else "verify_reviewed.py"
    subprocess.run([sys.executable, "-X", "utf8", str(checkout / PACKET / "tools" / helper)],
                   cwd=checkout, check=True)
    if args.notebook:
        print(f"Executed notebook: {checkout / 'output/jupyter-notebook/real-evaluation-analysis.ipynb'}")
    print("Frozen reproduction PASS; current product code and published result files were not replaced.")


if __name__ == "__main__":
    main()
