# Reproduce the published analysis without model calls

Use a full Git checkout of this branch and Python **3.11**. Git metadata is needed by the frozen-input verifier. No model credentials, DeepSeek login, private local artifacts, saved application database or new generation are needed. Dependency installation requires package downloads; the analysis itself uses only the public packet. No model API is invoked.

## Product development after the experiment

Product fixes after the experiment use a separate current code revision. The original
257-input experiment is preserved at commit **`7e0cf6cdf167e112775b1376458b1e7ceff1af1a`**.
The reproduction launcher checks that the current published result bytes still match
that snapshot, creates a detached local worktree at the original commit, checks its
helpers for edits, and runs the original verifier or notebook there. It neither fetches
models nor regenerates answers. Existing unexpected edits are rejected without reset.
Regenerated notebook outputs may differ by platform, but notebook cell sources and
Python analysis helpers must still match the original commit.

The original freeze ID stays `0fc19f9d47679eaf55ebeddd7c6857b36779b632db6d5dd644cfda6c6aeb3270`.
The direct frozen-run guard remains strict: a changed product tree cannot execute as
the old experiment. Use the commands below for historical evidence after product fixes.

Clone with automatic newline conversion disabled, especially on Windows:

```bash
git clone --config core.autocrlf=false --single-branch --branch codex/real-evaluation-foundation https://github.com/jadeyoungchina/multi-agent-research-assistant.git
cd multi-agent-research-assistant
```

The freeze binds exact bytes for all real-v1 inputs. With a global `core.autocrlf=true`, a default Windows checkout converts `benchmarks/real-v1/README.md` to CRLF and the verifier correctly rejects the altered bytes. Use a fresh directory with the command above rather than resetting existing work. A fresh checkout with this setting was verified without a local `.env` or any pre-existing artifacts, using the pinned analysis environment.

From the repository root, create and activate `.venv` as described in the [project README](../../../README.md), then install:

```bash
python -m pip install -r requirements/upstream-requirements.txt
python -m pip install -r requirements/project-requirements.txt
python -m pip install -r output/jupyter-notebook/analysis-requirements.txt
```

The optional analysis dependency file pins the five frozen scientific packages as well as the notebook and plotting tools. It lives outside the frozen implementation inputs. Original generation code, prompts, corpus, reference answers and the freeze record are retained in the pinned reproduction snapshot. Current product fixes are separately identified in Git.

## Verify saved scientific evidence

```bash
python docs/evaluation/heldout-test/tools/reproduce_frozen.py
```

Expected: **480 traces, 80 sampled + 17 additional AI reviews, 3960 physical receipts, 40 paired questions**, human scoring 0. This verifies the frozen inputs, recorded byte hashes, original-judge retention, source decisions, deterministic scores, available request prices and primary family-bootstrap statistic. Seven execution failures remain in primary quality as zeros; successful answers have no missing judgments. It does not independently prove semantic correctness or bill accuracy.

The reusable worktree is under `artifacts/evaluation/frozen-reproduction/<original-commit>/`.
The helper uses local Git history, so a shallow checkout missing the original commit must
first obtain full branch history. No model-provider credential is needed. The original
`verify_reviewed.py` remains available inside that worktree and validates the exact
original implementation; it deliberately refuses a modified current product tree.

## Execute the notebook from a fresh kernel

```bash
python docs/evaluation/heldout-test/tools/reproduce_frozen.py --notebook
```

The runner uses the same Python executable for the kernel, executes all **7 code cells** in order, fails on errors or missing embedded plots, closes the kernel, and prints the executed notebook path inside the isolated worktree. Inside the notebook, Python socket connections are disabled before verification and computation. Recomputed files are written inside that worktree; the current published result packet is read and checked, not overwritten.

You can also execute the current, saved [notebook](../../../output/jupyter-notebook/real-evaluation-analysis.ipynb) with
`python docs/evaluation/heldout-test/tools/run_analysis_notebook.py`. Its first cell selects the
same original source snapshot before importing the evaluator, and the runner saves the
executed notebook in the current checkout. Use a fresh kernel for interactive execution.

Recomputed outputs (paths below are relative to the original worktree; links show the committed baseline copies):

| Artifact | Purpose |
| --- | --- |
| [workflow-comparison.png](figures/workflow-comparison.png) | Quality over 40 question means, successful latency, physical answer attempts |
| [paired-quality.png](figures/paired-quality.png) | Family effects, 40 question differences and exploratory primary 95% interval |
| [citations-and-failures.png](figures/citations-and-failures.png) | Conditional citation means with individual denominators and all failed executions |
| `figures/*.svg` | Committed vector exports of the same three figures |
| `artifacts/evaluation/heldout-analysis/figures/*.pdf` | Locally generated vector PDFs; repository policy excludes tracked PDFs |
| [analysis-tables.json](figures/analysis-tables.json) | Exact plotted values, primary statistic, phase counts, unknown usage and conditional cost estimate |
| [question-means.csv](figures/question-means.csv) | 160 question-by-workflow means, each from three repeats |
| [analysis-provenance.json](figures/analysis-provenance.json) | Input SHA-256 hashes, freeze identity and plotting-helper code hash |

The helper recomputes correctness and citation metrics from the saved decisions and traces, latency from successful traces and attempts from trace request counts. It checks them against the published summary before plotting. Exact numerical tables should match; image/PDF/SVG bytes may vary with transitive rendering libraries and platforms. Saved plots were visually inspected for labels, units and clipping. The notebook preserves its executed outputs for GitHub readers.

## Interpret the output

Plain RAG has the highest observed primary quality (0.919). Multi-agent minus single-agent correctness is approximately **+5.7 percentage points**, with an exploratory eight-family bootstrap interval of about **[+1.4, +9.7] pp**, and median successful latency is about **5.8 times** single-agent. Do not infer universal architectural superiority. Citation structure checks are distinct from citation support.

The initial analysis is retained separately, including 15 judge errors. Source review followed that initial analysis and uses the same DeepSeek alias, with explicit Codex follow-ups; it is not independent human validation. Full charges remain unknown: 17 physical attempts have unknown token usage, and DeepSeek account prices are unverified. The 30/40-pair incremental answer-cost estimate excludes indexing and judging/review.

See [RESULTS.md](RESULTS.md) for the report, [FAILURE-CASES.md](FAILURE-CASES.md) for seven selected examples, and [SOURCE-REVIEW-METHOD.md](SOURCE-REVIEW-METHOD.md) for review limitations. No paid rerun is part of these reproduction commands.
