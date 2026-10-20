"""Execute the analysis with this Python interpreter and retain verified outputs."""
from pathlib import Path
import sys

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "output/jupyter-notebook/real-evaluation-analysis.ipynb"


def main():
    notebook = nbformat.read(PATH, as_version=4)
    nbformat.validate(notebook)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    images = [output for cell in code for output in cell.outputs
              if output.output_type == "display_data" and "image/png" in output.data]
    assert len(images) == 3, "All three figures must be embedded in the notebook."
    # Timing is local execution metadata, not part of the scientific record.
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbformat.validate(notebook)
    PATH.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8", newline="\n")
    print(f"Notebook PASS: {len(code)} code cells executed from a fresh kernel; no model calls.")


if __name__ == "__main__":
    main()
