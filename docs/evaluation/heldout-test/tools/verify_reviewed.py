"""Recompute published trace identities, scores, primary statistics and phase costs offline."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from app.evaluation.budget import Price, cny
from app.evaluation.frozen import digest_json, validate_freeze
from app.evaluation.models import WorkflowVariant
from app.evaluation.real_dataset import load_real_suite
from app.evaluation.runner import execution_order
from app.evaluation.test_analysis import primary_comparison
from app.evaluation.test_judging import TestJudgment, decode, score_test
from app.evaluation.trace import EvaluationTrace
import source_review as review

OUT = ROOT / "docs/evaluation/heldout-test"


def verify():
    freeze = validate_freeze(require_committed=False)
    cases = load_real_suite(ROOT / "benchmarks/real-v1/corpus-manifest.json", ROOT / "benchmarks/real-v1/test.jsonl", "test").cases
    by_case = {c.id:c for c in cases}
    summary = json.loads((OUT / "reviewed-summary.json").read_text("utf-8"))
    analysis = json.loads((OUT / "reviewed-analysis.json").read_text("utf-8"))
    initial = json.loads((OUT / "initial-analysis.json").read_text("utf-8"))
    source = json.loads((OUT / "source-adjudications.json").read_text("utf-8"))
    reviews = json.loads((OUT / "source-review-records.json").read_text("utf-8"))
    assert {summary["freeze_id"],analysis["freeze_id"],initial["freeze_id"],source["freeze_id"]} == {freeze["freeze_id"]}
    assert summary["human_reference_reviews"] == summary["human_answer_scores"] == analysis["human_scores"] == 0
    for item in summary["publications"]:
        assert sha256((OUT/item["path"]).read_bytes()).hexdigest() == item["sha256"]
    traces = {}
    for repeat in range(3):
        directory = OUT/f"repeat-{repeat}"
        packet = json.loads((directory/'summary.json').read_text('utf-8'))
        for item in packet['publications']:
            assert sha256((directory/item['path']).read_bytes()).hexdigest() == item['sha256']
        rows = [EvaluationTrace.model_validate_json(line) for line in (directory/'traces.jsonl').read_text('utf-8').splitlines()]
        expected = execution_order(cases,[WorkflowVariant(v) for v in freeze['design']['matrix']['variants']],freeze['design']['matrix']['repeat_order_seeds'][repeat])
        assert [(t.case_id,t.variant) for t in rows] == expected and len(rows) == 160
        for index,trace in enumerate(rows,1): traces[f'r{repeat}-answer-{index:03d}'] = trace
    old = {x['unit']:x for x in initial['unit_scores']}
    current = {x['unit']:x for x in analysis['unit_scores']}
    assert old.keys() == current.keys() == traces.keys() and len(traces) == 480
    values, reviewed = {}, set()
    for unit,trace in traces.items():
        row = current[unit]
        original = row['original']
        assert original == old[unit]['original']
        digest = sha256(trace.model_dump_json().encode()).hexdigest()
        assert original['identity']['trace_sha256'] == digest and original['human_reviewed'] is False
        assert original['case_id'] == trace.case_id and original['variant'] == trace.variant.value
        if original['status'] == 'valid_ai_judgment':
            raw = decode(original['attempts'][-1]['raw_response']['text'])
            assert raw.model_dump() == original['judgment']
            assert score_test(by_case[trace.case_id],trace,raw).model_dump() == original['score']
        decision = source['adjudications'].get(unit)
        assert decision == row['source_adjudication']
        if decision:
            assert decision['trace_sha256'] == digest and decision['reviewer_kind'] == 'ai' and decision['human_reviewed'] is False
            assert decision['reason'] and decision['action'] in {'accepted','corrected'}
            if decision['action'] == 'accepted': assert decision['judgment'] == original['judgment']
            reviewed.add(unit)
        failure = source['execution_failure_audits'].get(unit)
        if failure:
            assert trace.status != 'success' and failure['trace_sha256'] == digest and failure['reviewer_kind'] == 'ai'
            assert failure['human_reviewed'] is False and failure['reason']
            reviewed.add(unit)
        if trace.status != 'success': value = 0.0
        elif decision or original['status'] == 'valid_ai_judgment':
            judgment = TestJudgment.model_validate(decision['judgment'] if decision else original['judgment'])
            value = score_test(by_case[trace.case_id],trace,judgment).answer_correctness
        else: value = None
        assert value == row['primary_quality']
        values[(trace.case_id,trace.variant.value,int(unit[1]))] = value
    sample_ids = {x['id'] for x in freeze['design']['source_review_sample']['metadata']}
    sample = {unit for unit,t in traces.items() if unit.startswith('r0-') and t.case_id in sample_ids}
    assert len(sample) == 80 and sample <= reviewed and len(reviewed) == 97
    assert set(source['adjudications']) <= traces.keys() and set(source['execution_failure_audits']) <= traces.keys()
    options = freeze['design']['statistics']
    result = primary_comparison(cases,values,replicates=options['bootstrap_replicates'],seed=options['bootstrap_seed'])
    assert result == analysis['primary'] == summary['primary']
    receipts = [json.loads(line) for line in (OUT/'all-request-usage.jsonl').read_text('utf-8').splitlines()]
    assert len({r['id'] for r in receipts}) == len(receipts) == summary['all_recorded_usage']['physical_attempts']
    by_id = {r['id']:r for r in receipts}
    assigned = set()
    for trace in traces.values():
        ids = trace.attempt_ids
        assert not assigned.intersection(ids) and len(set(ids)) == len(ids)
        assigned.update(ids)
        rows = [by_id[i] for i in ids]
        assert all(r['phase']=='answer_generation' for r in rows)
        assert len(rows) == trace.model_calls
        assert sum(r['input_tokens'] or 0 for r in rows) == trace.prompt_tokens
        assert sum(r['output_tokens'] or 0 for r in rows) == trace.completion_tokens
    assert assigned == {r['id'] for r in receipts if r['phase']=='answer_generation'}
    for row in receipts:
        if row['state']=='settled' and row['price'] is not None:
            assert None not in (row['input_tokens'],row['output_tokens'],row['cached_input_tokens'])
            assert cny(Price(**row['price']).cost(row['input_tokens'],row['output_tokens'],row['cached_input_tokens'])) == row['estimated_cost_cny']
        else: assert row['estimated_cost_cny'] is None
    for phase,usage in summary['phase_usage'].items():
        if phase == 'source_review': rows=[r for r in receipts if r['phase']==phase]
        else:
            repeat,name=phase.split('/')
            rows=[r for r in receipts if r['phase']==name and r['repeat_index']==int(repeat[-1])]
        assert len(rows)==usage['physical_attempts']
        assert sum(r['input_tokens'] or 0 for r in rows)==usage['input_tokens']
        assert sum(r['output_tokens'] or 0 for r in rows)==usage['output_tokens']
    assert len(reviews['records'])==97 and reviews['manifest']['identity']['freeze_id']==freeze['freeze_id']
    identity = reviews['manifest']['identity']
    assert identity['source_code_sha256'] == sha256((Path(__file__).parent/'source_review.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert identity['cli_sha256'] == freeze['judge_runtime']['cli_sha256']
    targets = {t['unit']:t for t in identity['targets']}
    expected = sample | {unit for unit,row in old.items() if row['original']['status']=='judge_error'} | {unit for unit,t in traces.items() if t.status!='success'}
    assert targets.keys() == expected == {entry['unit'] for entry in reviews['records']}
    for entry in reviews['records']:
        assert entry['target']['trace_sha256']==sha256(traces[entry['unit']].model_dump_json().encode()).hexdigest()
        assert entry['target'] == targets[entry['unit']] and entry['review_identity']==digest_json(identity)
        original_bytes=(json.dumps(old[entry['unit']]['original'],ensure_ascii=False,indent=2)+'\n').encode()
        assert sha256(original_bytes).hexdigest()==entry['target']['original_judgment_sha256']
        assert entry['human_reviewed'] is False
        if entry['status']=='valid_source_audit':
            audit=review.SourceAudit.model_validate(entry['audit'])
            assert audit==review.parse(entry['attempts'][-1]['raw_response']['text'],review.SourceAudit)
            review.validate_audit(by_case[traces[entry['unit']].case_id],traces[entry['unit']],audit)
        elif entry['status']=='valid_failure_audit':
            audit=review.FailureAudit.model_validate(entry['audit'])
            assert audit==review.parse(entry['attempts'][-1]['raw_response']['text'],review.FailureAudit)
            assert audit.observed_error_code==traces[entry['unit']].error_code
        for attempt in entry['attempts']:
            assert all(by_id[i]['phase']=='source_review' for i in attempt['raw_response']['attempt_ids'])
    assert sum(t.status!='success' for t in traces.values())==analysis['execution_failures']==7
    assert not analysis['successful_answers_missing_judgment']
    print(f"Reviewed packet PASS: {len(traces)} traces, 80 sampled + 17 additional AI reviews, {len(receipts)} physical receipts; original judges, hashes, scores and 40 paired questions recompute. Human scoring: 0.")


if __name__ == '__main__':
    verify()
