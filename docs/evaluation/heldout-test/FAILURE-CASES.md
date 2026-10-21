# Retained failure and uncertainty cases

These seven examples were selected after observing the held-out results. They illustrate failure mechanisms across all four workflows, not their population frequencies. The frozen prompts, references, answers and scores remain unchanged. Reviewers are AI systems; independent human validation remains **0**.

Unit `r0-answer-NNN` means the NNN-th line (1-based) of [repeat-0/traces.jsonl](repeat-0/traces.jsonl). The trace contains the complete answer, actually retrieved evidence, status and request IDs. Look up the same unit in [reviewed-analysis.json](reviewed-analysis.json) for the preserved initial judgment, and [source-adjudications.json](source-adjudications.json) for the explicit AI decision. Expected source passages and line ranges are in [test.jsonl](../../../benchmarks/real-v1/test.jsonl).

| Unit | Case | Workflow | Primary score | Observed issue |
| --- | --- | --- | ---: | --- |
| r0-answer-118 | REAL-TEST-005 | baseline_llm | 0 | Unsupported performance numbers despite an insufficiency disclaimer |
| r0-answer-145 | REAL-TEST-014 | single_agent_rag | 0.5 | Incorrect RootModel exception |
| r0-answer-143 | REAL-TEST-031 | llm_rag | 0 | Retrieval miss followed by scoped abstention |
| r0-answer-007 | REAL-TEST-004 | multi_agent_rag | 0.5 | Correct recursive contrast mixed with false generalizations |
| r0-answer-157 | REAL-TEST-038 | multi_agent_rag | 1/3 | Two requested defaults omitted/denied |
| r0-answer-043 | REAL-TEST-006 | single_agent_rag | 0 | Execution failure; detailed cause unresolved |
| r0-answer-138 | REAL-TEST-035 | multi_agent_rag | 0 | Execution failure with a malformed range citation in captured drafts |

## Unsupported numbers can survive a disclaimer

`r0-answer-118` correctly says there is no universal production throughput or memory figure, then introduces 5–10x speed and 2–5x QPS claims without the requested workload evidence. The frozen insufficiency question requires acknowledging missing measurements without inventing numbers. The source adjudication therefore assigns zero and records an unsupported assertion, rather than rewarding the opening disclaimer alone.

This suggests a future development check for substantive claims added after an abstention. It does not justify changing this held-out label or claiming those speed ranges were measured here.

## A valid citation can accompany the wrong API exception

`r0-answer-145` correctly describes raw nested values and the `root` key from `dict(RootModel)`, but says `model_dump()` produces the same wrapper. The [pinned serialization document](../../../benchmarks/real-v1/raw/pydantic-v2--concepts--serialization.md) distinguishes the direct root value returned by `model_dump()`. One of two reference points is satisfied; the result is 0.5. The adjudication retains the original judgment and explains why a second AI proposal's full credit was rejected.

A small local counterexample under the pinned Pydantic 2.8.2 runtime confirms the distinction:

```python
from pydantic import RootModel
value = RootModel[list[int]]([1, 2])
assert dict(value) == {"root": [1, 2]}
assert value.model_dump() == [1, 2]
```

## Scoped abstention can be reasonable for retrieved text and still miss the question

`r0-answer-143` says its supplied evidence does not describe a missing Authorization header. Its retrieved passages indeed omit the decisive rule, although the [whole candidate corpus](../../../benchmarks/real-v1/raw/fastapi--en--docs--tutorial--security--first-steps.md) contains the direct 401 response at line 281. The answer gets zero on the question and is counted as false abstention.

The reviewer separately records `unsupported_assertion=false`: a limitation scoped to supplied evidence is not a fabricated universal API claim. This distinction points toward diagnosing retrieval coverage rather than forcing a generator to answer without evidence.

## Longer answers can introduce unsupported generalizations

`r0-answer-007` correctly contrasts recursive dumping with non-recursive `dict(model)`, but treats a nested example as a flat-model difference and says the default dump is universally JSON compatible. The [pinned models example](../../../benchmarks/real-v1/raw/pydantic-v2--concepts--models.md) shows the same field dictionary for the flat case. Python-mode dumping can retain values such as `datetime`; recursive conversion alone does not establish JSON compatibility. The reviewed score is 0.5, with partially supported citation blocks and an unsupported-assertion flag.

All successful RAG answers in the experiment pass citation structure checks. This case demonstrates why structure checks and source support are reported separately.

## A document-level answer can remain incomplete despite multi-agent processing

`r0-answer-157` supplies `max_age=600` but says the exact method/header defaults are unspecified. The [pinned CORS source](../../../benchmarks/real-v1/raw/fastapi--en--docs--tutorial--cors.md) explicitly gives `allow_methods=['GET']` and `allow_headers=[]`. One of three required points is satisfied, so the score is 1/3. The answer's presence of citations and a limitations section does not make it complete.

This observation does not establish which internal stage caused the miss. The saved retrieval rounds and final evidence are the starting point for a future development-only investigation.

## Preserve the boundary between an error code and a proven cause

`r0-answer-043` ends with `invalid_citation`. Its final text contains reference tokens, but [failure-diagnostics.json](repeat-0/failure-diagnostics.json) has no diagnostic captures for this trace. The recorded failure and zero primary score are established; the specific internal rejection cause remains unresolved. The failure audit explicitly keeps that uncertainty.

By contrast, `r0-answer-138` has two captured drafts containing `[[cite:ref_0001-ref_0008]]`. The failure audit identifies the malformed range citation and the run retains `invalid_citation`. Captured schema validation succeeds, so schema validity should not be reported as successful citation validation. Both failed attempts remain available; no favorable retry replaced the execution.

## How to use these observations

Keep this benchmark as a fixed baseline. Form follow-up hypotheses about retrieval misses, exception handling, citation support and workflow overhead, then implement and evaluate them on development material and newly designed independent cases. These selected examples are not an additional test set, a claim of independent review, or evidence that a proposed repair already works.
