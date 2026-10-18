# Public-document benchmark: real-v1 candidate release

This is a custom, AI-authored question set over real public documentation.
All **60 references have completed delegated AI adjudication** by DeepSeek
and Codex; human reference reviews remain pending. It is **not real-user
traffic**. No held-out model-quality result is claimed by this release.

| Property | Current snapshot |
| --- | --- |
| Documents | 24 Markdown snapshots, 12 topic families |
| Sources | Pydantic v1.10.17 / v2.8.2; FastAPI 0.112.2 |
| Dev | 20 questions, 8 documents, 4 families |
| Test | 40 questions, 16 documents, 8 families |
| Languages | 30 Chinese, 30 English |
| Types | Five types, 12 questions each; 4 dev and 8 test per type |
| Authorship | DeepSeek Flash drafts; Codex source checks and corrections |
| Reference approval | 0 approved by a human; 60 pending |
| Delegated AI adjudication | 60/60 accepted for a dev pilot; 46 cases corrected |
| Intended domain | Documentation research for Python application engineering |

`corpus-manifest.json` pins each document to an immutable upstream SHA and
records the exact raw and normalized SHA-256, license, retrieval date, family,
and split. MIT license notices are included. `raw/` preserves upstream bytes;
`corpus/` prefixes six provenance lines and converts CRLF to LF. Original
line N maps to normalized line N + 6. `.gitattributes` preserves snapshot
bytes across Windows/Linux checkouts. Markdown includes, external code
examples, linked pages, and images are **not expanded**: only indexed text
is available to the workflows. Code behavior outside the corpus is not gold.

Each split searches its entire candidate pool, including distractors. The
question schema deliberately omits a per-question document filter. Gold
sources, key points, sufficiency labels, and review fields stay inside the
evaluator. `--limit` reduces questions, never the candidate corpus. Families
and every version of the same upstream path stay in one split; question
groups cannot cross splits. The split is by topic family, not by library:
the same libraries occur in dev and test, so this does not establish
generalization to unseen libraries or arbitrary research domains.

The 12 insufficient-evidence questions ask for deployment-specific measured
performance not established by the complete split corpus. Qualitative
guidance and the v1 `construct()` approximate speedup do not answer those
questions. This first release has limited negative-question diversity;
future datasets should add other missing-evidence patterns before broad
abstention claims. The AI whole-pool audit remains subject to human review.

Candidates were checked for literal source quotes, line ranges, split
membership, type/language balance, and distinct IDs. AI curation also
repaired four non-verbatim quotes, an irrelevant cross-document citation,
and several overbroad or underspecified references. Corrections are retained
in `curation-overrides.json`. The v1 model guide contains an apparent
`'123.45'`-to-int example inconsistency; that example was excluded from the
question, which instead uses the unambiguous `'123'` example. An original
source quote is evidence of what the document states, not proof that every
upstream example executes correctly.

Offline verification and a pipeline demonstration:

```powershell
.venv/Scripts/python.exe -c "from pathlib import Path; from app.evaluation.real_dataset import validate_real_benchmark; validate_real_benchmark(Path('benchmarks/real-v1'))"
.venv/Scripts/python.exe -m app.evaluation.cli --provider fake --real-manifest benchmarks/real-v1/corpus-manifest.json --split dev --output-dir artifacts/evaluation/real-v1-fake
```

Fake results check execution only. An AI-reviewed dev pilot uses
`--allow-ai-reviewed`, which validates the current per-reference and corpus
hashes against `review/final-audit.json`. Run metadata records AI review and
human approval separately. Human-reviewed dev runs are also supported;
`--allow-unreviewed` remains an explicit preliminary override. Paid test
execution is disabled until a frozen experiment runner is implemented.
Synthetic release thresholds are not applied to real-document quality.

Generate the local human review page with
`python scripts/review_real_dataset.py`. It opens no network connection and
changes no review status. The generated file is
`artifacts/evaluation/real-v1-human-review.html`; select each verdict after
reading the sources, enter the actual reviewer's name, and export the JSON.
Import that explicit export with
`python scripts/review_real_dataset.py --apply PATH_TO_EXPORTED_JSON`.
An import must match all three snapshot hashes, contain every case exactly
once, and attribute completed reviews to a reviewer and zoned timestamp.
Corrected references need a new dataset snapshot and review page.

The delegated review trail is in `review/`: initial findings, exact before/
after corrections, final task feedback and Codex adjudication. DeepSeek's
suggestions are advisory and can be wrong; final acceptance is the source-
checked Codex decision. All 139 final evidence units are reachable after
actual document normalization/chunking. See
[`ai-reference-review.md`](../../docs/evaluation/ai-reference-review.md) for
the source checks, disagreements, usage and limits. No test-workflow output
was used to curate these references.

`app/evaluation/semantic.py` defines the separate blinded semantic rubric
and validates per-key-point 0/1/2 judgments, verbatim claim spans, declared
and explicitly bound citations, and abstention judgments. Full claim support
and citation completeness have separate denominators; ID validity alone
does not earn a support score. Judge failures are errors, never automatic
quality zeros. Automatic checks cannot prove that a judge extracted every
atomic claim or interpreted every citation correctly; human calibration
and adjudication remain necessary. The conservative 160-character citation
binding check may reject ambiguous formatting. No model-judge ratings have
been generated in this milestone.

Recollection is explicit and networked:
`python scripts/collect_real_corpus.py`. It uses the fixed allowlist and
commits in that script; rerunning changes retrieval timestamps, so retain
the committed manifest for exact replay. Drafting prompts can be prepared
with `python scripts/draft_real_questions.py`; adding `--execute` makes
tracked DeepSeek CLI calls. These source documents are MIT licensed; the
project's separately licensed application code retains its existing terms.
