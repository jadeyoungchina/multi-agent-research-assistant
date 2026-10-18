"""Generate an offline human review page or apply an explicitly exported review.

Opening/generating the page never approves a reference. The reviewer must
choose a status and enter their name. Hashes prevent applying stale reviews.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evaluation.real_dataset import HumanReview, validate_real_benchmark


def review_payload(root: Path) -> dict:
    dev, test = validate_real_benchmark(root)
    cases = []
    docs = {doc.corpus_path.removeprefix("corpus/"): doc for doc in dev.manifest.documents}
    for case in [*dev.cases, *test.cases]:
        row = case.model_dump(mode="json")
        row["sources"] = []
        for evidence in case.expected_evidence:
            doc = docs[evidence.source_file]
            lines = (root / doc.corpus_path).read_text("utf-8").splitlines()
            start, end = max(doc.normalized_line_offset, evidence.start_line - 4), min(len(lines), evidence.end_line + 3)
            upstream_line = evidence.start_line - doc.normalized_line_offset
            row["sources"].append(dict(url=doc.source_url + f"#L{upstream_line}",
                                       location=f"{evidence.source_file}:{evidence.start_line}",
                                       quote=evidence.contains, context="\n".join(lines[start:end])))
        row["candidate_pool"] = [dict(filename=name, url=docs[name].source_url) for name in (dev.sources if case.split == "dev" else test.sources)]
        cases.append(row)
    return dict(schema_version="real-human-review-v1",
                hashes={name: sha256((root / name).read_bytes()).hexdigest()
                        for name in ("corpus-manifest.json", "dev.jsonl", "test.jsonl")}, cases=cases)


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实文档评测：参考答案审核</title>
<style>
body{font:16px/1.65 system-ui,sans-serif;color:#17242c;background:#f3f6f7;margin:0}
main{max-width:1050px;margin:auto;padding:30px}h1{font-size:27px}h2{font-size:20px}
header{position:sticky;top:0;background:#f3f6f7;padding:12px 0;border-bottom:1px solid #bccdd1;z-index:1}
article{background:white;border:1px solid #ccd9de;border-radius:10px;margin:22px 0;padding:22px}
button,select,input,textarea{font:inherit;padding:8px;border:1px solid #9dafb7;border-radius:5px}
button{background:#075967;color:white;cursor:pointer}textarea{box-sizing:border-box;width:100%;min-height:65px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f7;padding:12px;font:14px/1.5 ui-monospace,monospace}
.muted{color:#52656c}label{display:inline-block;margin:5px 12px 5px 0}a{color:#075967}
</style><main><h1>真实文档评测：参考答案审核</h1>
<p>60 道题由 AI 起草和核查，初始均为“待审核”。请对照原文确认问题可回答、参考要点正确且完整。无答案题请检查整个候选文档池。只有你的明确选择才会产生人工审核记录。</p>
<p class="muted">这是参考答案审核，不会运行模型，也不会自动推送 GitHub。页面刷新会清除本次未导出的选择；可随时导出当前进度。</p>
<header><label>审核人 <input id="reviewer" placeholder="你的姓名或明确身份"></label>
<label>查看 <select id="filter"><option value="all">全部 60 题</option><option value="dev">开发集 20 题</option><option value="test">测试集 40 题</option></select></label>
<button id="export">导出审核记录</button><span id="progress" class="muted"></span></header><div id="cases"></div>
<script>
const data=__DATA__, reviews=new Map(data.cases.map(c=>[c.id,{...c.human_review}]));
const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text)node.textContent=text;if(cls)node.className=cls;return node};
function progress(){document.querySelector('#progress').textContent=' 已通过 '+[...reviews.values()].filter(r=>r.status==='approved').length+'/60';}
function render(){
 const target=document.querySelector('#cases');target.replaceChildren();const filter=document.querySelector('#filter').value;
 for(const c of data.cases){if(filter!=='all'&&c.split!==filter)continue;
  const card=el('article');card.append(el('h2',c.id+' · '+c.question));
  card.append(el('p',c.split+' / '+c.language+' / '+c.question_type+' / '+c.family,'muted'));
  card.append(el('strong','参考要点'));const points=el('ul');for(const p of c.answer_key_points)points.append(el('li',p));card.append(points);
  card.append(el('p','预期语料充分性：'+(c.expected_sufficiency?'有充分证据':'证据不足')));
  for(const s of c.sources){const detail=el('details'),summary=el('summary',s.location+' · 原文及上下文');detail.append(summary);
   const link=el('a','查看固定版本原文');link.href=s.url;link.target='_blank';link.rel='noreferrer';detail.append(link,el('pre',s.context));card.append(detail);}
  if(c.insufficiency_audit)card.append(el('p','AI 的不足证据核查说明（仍需你确认）：'+c.insufficiency_audit));
  const pool=el('details');pool.append(el('summary','该题可检索的完整候选池（含干扰文档）'));
  for(const source of c.candidate_pool){const line=el('p'),link=el('a',source.filename);link.href=source.url;link.target='_blank';link.rel='noreferrer';line.append(link);pool.append(line);}card.append(pool);
  const row=reviews.get(c.id), label=el('label','审核结论 '), select=el('select');
  for(const [value,title] of [['pending','待审核'],['approved','确认通过'],['rejected','需要修改']]){const option=el('option',title);option.value=value;select.append(option);}select.value=row.status;
  select.onchange=()=>{row.status=select.value;row._changed=true;row.reviewed_at=select.value==='pending'?null:new Date().toISOString();progress();};label.append(select);
  const notes=el('textarea');notes.placeholder='纠正内容或审核说明（可留空）';notes.value=row.notes;notes.oninput=()=>{row.notes=notes.value;};card.append(label,notes);target.append(card);
 }progress();
}
document.querySelector('#filter').onchange=render;
document.querySelector('#export').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value.trim();if(!reviewer){alert('请填写实际审核人；未审核的题目可以保持待审核。');return;}
 const output={schema_version:data.schema_version,hashes:data.hashes,reviews:data.cases.map(c=>{const r=reviews.get(c.id);return {id:c.id,status:r.status,notes:r.notes,reviewed_at:r.reviewed_at,reviewer:r.status==='pending'?null:(r._changed?reviewer:r.reviewer)};})};
 const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='real-v1-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};render();
</script></main></html>'''


def apply_review(root: Path, review_path: Path) -> int:
    payload = review_payload(root)
    review = json.loads(review_path.read_text("utf-8"))
    if review.get("schema_version") != payload["schema_version"] or review.get("hashes") != payload["hashes"]:
        raise ValueError("review targets a different dataset snapshot")
    entries = review.get("reviews", [])
    known = {case["id"] for case in payload["cases"]}
    if len(entries) != len(known) or {entry["id"] for entry in entries} != known:
        raise ValueError("review must name every case exactly once; unreviewed cases stay pending")
    approvals = {entry["id"]: HumanReview.model_validate_json(json.dumps({k: v for k, v in entry.items() if k != "id"})) for entry in entries}
    # Parse and validate the whole import before changing either split.
    output = {}
    for split in ("dev", "test"):
        path = root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        for row in rows:
            if row["human_review"]["status"] != "pending" and approvals[row["id"]].status == "pending":
                continue  # Partial subsequent exports cannot erase an existing review.
            row["human_review"] = approvals[row["id"]].model_dump(mode="json")
        output[path] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    for path, text in output.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    validate_real_benchmark(root)
    return sum(entry.status == "approved" for entry in approvals.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "benchmarks/real-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation/real-v1-human-review.html")
    parser.add_argument("--apply", type=Path, help="apply a review actually exported by the human reviewer")
    args = parser.parse_args()
    if args.apply:
        print(f"Applied explicit review; {apply_review(args.root, args.apply)} approved entries.")
    else:
        data = json.dumps(review_payload(args.root), ensure_ascii=False).replace("<", "\\u003c")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
        print("Wrote human review page; no approval status changed.")


if __name__ == "__main__":
    main()
