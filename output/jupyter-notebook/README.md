# Offline held-out analysis

[Open the executed notebook](real-evaluation-analysis.ipynb) or follow the [reproduction guide](../../docs/evaluation/heldout-test/REPRODUCE.md). It reads only the committed result packet and makes no model calls. Install the project dependencies plus [analysis-requirements.txt](analysis-requirements.txt) first.

```bash
python docs/evaluation/heldout-test/tools/run_analysis_notebook.py
```

Seven code cells verify 480 traces, rebuild quality/citation/latency/request tables, and generate three figure sets. The notebook includes executed tables and embedded PNGs for reading on GitHub. Exact numerical outputs, input hashes, PNGs and SVGs are under [heldout-test/figures](../../docs/evaluation/heldout-test/figures). PDFs are exported locally under `artifacts/evaluation/heldout-analysis/figures/`; repository policy excludes tracked PDFs.

AI scoring and AI references have no independent human validation. Failures and unknown charges remain visible; the [report](../../docs/evaluation/heldout-test/RESULTS.md) and [seven failure examples](../../docs/evaluation/heldout-test/FAILURE-CASES.md) explain the results' practical limits.
