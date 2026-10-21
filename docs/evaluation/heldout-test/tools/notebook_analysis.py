"""Offline tables and figures derived from the immutable held-out packet."""
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[4]
PACKET = ROOT / "docs/evaluation/heldout-test"
FIGURES = PACKET / "figures"
sys.path.insert(0, str(ROOT))

from app.evaluation.real_dataset import load_real_suite
from app.evaluation.test_judging import TestJudgment, score_test
from app.evaluation.trace import EvaluationTrace

VARIANTS = ("baseline_llm", "llm_rag", "single_agent_rag", "multi_agent_rag")
LABELS = ("Direct LLM", "Plain RAG", "Single-agent RAG", "Multi-agent RAG")
COLORS = ("#7b8794", "#007f73", "#3479b8", "#b66031")


def load_packet():
    summary = json.loads((PACKET / "reviewed-summary.json").read_text("utf-8"))
    analysis = json.loads((PACKET / "reviewed-analysis.json").read_text("utf-8"))
    cases = load_real_suite(ROOT / "benchmarks/real-v1/corpus-manifest.json",
                            ROOT / "benchmarks/real-v1/test.jsonl", "test").cases
    traces = {}
    for repeat in range(3):
        lines = (PACKET / f"repeat-{repeat}/traces.jsonl").read_text("utf-8").splitlines()
        for index, line in enumerate(lines, 1):
            traces[f"r{repeat}-answer-{index:03d}"] = EvaluationTrace.model_validate_json(line)
    receipts = [json.loads(line) for line in
                (PACKET / "all-request-usage.jsonl").read_text("utf-8").splitlines()]
    return summary, analysis, {case.id: case for case in cases}, traces, receipts


def recompute_tables(summary, analysis, cases, traces, receipts):
    """Rebuild plotted values from trace-bound decisions, not summary constants."""
    groups = defaultdict(list)
    semantic = defaultdict(list)
    for row in analysis["unit_scores"]:
        trace = traces[row["unit"]]
        if trace.status != "success":
            quality = 0.0
        else:
            decision = row["source_adjudication"]
            judgment = decision["judgment"] if decision else row["original"]["judgment"]
            score = score_test(cases[trace.case_id], trace, TestJudgment.model_validate(judgment))
            quality = score.answer_correctness
            semantic[trace.variant.value].append(score)
        assert quality == row["primary_quality"]
        groups[(trace.case_id, trace.variant.value)].append(quality)
    assert len(groups) == 160 and all(len(values) == 3 for values in groups.values())
    questions = [{"case_id": case_id, "variant": variant,
                  "quality": statistics.mean(values)}
                 for (case_id, variant), values in sorted(groups.items())]
    variants = []
    for variant in VARIANTS:
        selected = [trace for trace in traces.values() if trace.variant.value == variant]
        successful = [trace for trace in selected if trace.status == "success"]
        citation = {}
        for field in ("citation_support_rate", "citation_completeness"):
            values = [getattr(score, field) for score in semantic[variant]
                      if getattr(score, field) is not None]
            citation[field] = {"mean": statistics.mean(values) if values else None,
                               "denominator": len(values)}
            expected = summary["by_variant"][variant]["conditional_semantic_metrics"][field]
            assert citation[field]["denominator"] == expected["denominator"]
            assert (citation[field]["mean"] is None and expected["mean"] is None) or abs(citation[field]["mean"] - expected["mean"]) < 1e-12
        result = {"variant": variant, "quality_questions": 40,
                  "quality": statistics.mean(q["quality"] for q in questions if q["variant"] == variant),
                  "successful": len(successful), "failed": len(selected) - len(successful),
                  "median_success_seconds": statistics.median(t.latency_ms / 1000 for t in successful),
                  "answer_attempts": sum(t.model_calls for t in selected), **citation}
        expected = summary["by_variant"][variant]
        assert abs(result["quality"] - expected["mean_question_primary_quality"]) < 1e-12
        assert result["median_success_seconds"] == expected["median_success_latency_seconds"]
        assert result["answer_attempts"] == expected["answer_usage"]["physical_attempts"]
        variants.append(result)
    lookup = {(q["case_id"], q["variant"]): q["quality"] for q in questions}
    paired = [{"case_id": case_id,
               "difference": lookup[case_id, "multi_agent_rag"] - lookup[case_id, "single_agent_rag"]}
              for case_id in sorted(cases)]
    assert abs(statistics.mean(p["difference"] for p in paired) - summary["primary"]["mean_paired_correctness_difference"]) < 1e-12
    phases = Counter(receipt["phase"] for receipt in receipts)
    assert sum(phases.values()) == summary["all_recorded_usage"]["physical_attempts"]
    return {"freeze_id": summary["freeze_id"], "variants": variants,
            "question_means": questions, "paired_questions": paired,
            "primary": summary["primary"], "phase_attempts": dict(sorted(phases.items())),
            "recorded_usage": summary["all_recorded_usage"],
            "incremental_answer_cost": summary["incremental_answer_cost"],
            "human_answer_scores": summary["human_answer_scores"]}


def write_tables(tables, directory=FIGURES):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis-tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    with (directory / "question-means.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "variant", "quality"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(tables["question_means"])
    inputs = sorted(path for path in PACKET.rglob("*") if path.is_file()
                    and path.suffix in {".json", ".jsonl"}
                    and not path.is_relative_to(FIGURES))
    provenance = {"freeze_id": tables["freeze_id"],
                  "analysis_code_sha256": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                  "inputs": [{"path": path.relative_to(ROOT).as_posix(),
                              "sha256": sha256(path.read_bytes()).hexdigest()} for path in inputs],
                  "reproduction": "Numerical tables are reproducible. Figure bytes may vary across rendering-library/platform versions."}
    (directory / "analysis-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def render_figures(tables, directory=FIGURES):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "heldout-analysis-v1"})
    directory.mkdir(parents=True, exist_ok=True)
    rows = tables["variants"]
    figures = []

    def save(fig, name, caption):
        fig.text(.01, .015, caption, fontsize=9, color="#364152", va="bottom")
        for extension in ("png", "svg", "pdf"):
            target = directory if extension != "pdf" else ROOT / "artifacts/evaluation/heldout-analysis/figures"
            target.mkdir(parents=True, exist_ok=True)
            metadata = {"Date": None} if extension == "svg" else (
                {"CreationDate": None, "ModDate": None} if extension == "pdf" else {})
            path = target / f"{name}.{extension}"
            fig.savefig(path, dpi=180, metadata=metadata)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text("utf-8").splitlines()) + "\n",
                                encoding="utf-8", newline="\n")
        figures.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
    fig.subplots_adjust(left=.12, right=.97, bottom=.23, top=.78, wspace=.38)
    panels = (("quality", "Primary quality", "Mean correctness; failures = 0", 1.13),
              ("median_success_seconds", "Successful-run latency", "Median seconds (successful runs only)", 62),
              ("answer_attempts", "Answer-generation workload", "Physical API attempts; all 120 runs", 2550))
    for ax, (field, title, xlabel, xmax) in zip(axes, panels):
        bars = ax.barh(LABELS, [r[field] for r in rows], color=COLORS, height=.6)
        ax.invert_yaxis()
        ax.set(xlim=(0, xmax), title=title, xlabel=xlabel)
        ax.xaxis.grid(True, alpha=.15)
        ax.set_axisbelow(True)
        if ax is not axes[0]:
            ax.set_yticks(range(4), labels=[""] * 4)
        for bar, row in zip(bars, rows):
            value = row[field]
            label = f"{value:.3f}" if field == "quality" else (f"{value:.2f}" if field == "median_success_seconds" else str(value))
            ax.text(value + xmax * .02, bar.get_y() + bar.get_height()/2, label, va="center", fontsize=10)
    fig.suptitle("Frozen held-out comparison: quality, latency and workload", fontsize=15, y=.96)
    save(fig, "workflow-comparison", "40 questions x 3 repeats per workflow. AI-assisted scoring; human scores: 0.\nQuality uses 40 question means. Latency n = 120, 119, 119, 115; answer attempts include query embeddings and retries.")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6))
    fig.subplots_adjust(left=.18, right=.96, bottom=.2, top=.81, wspace=.45)
    families = tables["primary"]["by_family"]
    names = list(families)
    differences = [families[name]["mean_difference"] * 100 for name in names]
    axes[0].barh(names, differences, color=[COLORS[3] if v >= 0 else COLORS[2] for v in differences], height=.55)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#364152", linewidth=1)
    axes[0].set(xlim=(-8, 17), title="By document family (5 questions each)", xlabel="Multi-agent minus single-agent (percentage points)")
    for index, value in enumerate(differences):
        axes[0].text(value + (.5 if value >= 0 else -.5), index, f"{value:+.1f}", va="center", ha="left" if value >= 0 else "right")
    point = tables["primary"]["mean_paired_correctness_difference"] * 100
    lower, upper = [v * 100 for v in tables["primary"]["family_bootstrap_95_interval"]]
    ax = axes[1]
    ax.scatter([p["difference"] * 100 for p in tables["paired_questions"]],
               np.linspace(.82, 1.18, 40), color=COLORS[2], alpha=.6, s=25)
    ax.errorbar([point], [0], xerr=[[point-lower], [upper-point]], fmt="o", color=COLORS[3], capsize=6, markersize=8)
    ax.axvline(0, color="#364152", linewidth=1)
    ax.set(yticks=[0, 1], yticklabels=["Primary mean\n95% family bootstrap", "40 question pairs"],
           ylim=(-.45, 1.5), xlabel="Multi-agent minus single-agent (percentage points)", title=f"Primary difference: {point:+.1f} pp [{lower:.1f}, {upper:.1f}]")
    fig.suptitle("Prespecified paired comparison, with exploratory uncertainty", fontsize=15, y=.96)
    save(fig, "paired-quality", "Three repeats are averaged within each question. Families are resampled together; 10,000 draws, seed 20261023.\nEight families; 40/40 complete pairs. Display offsets separate overlapping points and have no statistical meaning.")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5))
    fig.subplots_adjust(left=.08, right=.96, bottom=.25, top=.8, wspace=.32)
    x = np.arange(3)
    for offset, (field, label, color) in zip((-.18, .18), (
            ("citation_support_rate", "Support", COLORS[1]),
            ("citation_completeness", "Completeness", COLORS[2]))):
        values = [row[field]["mean"] for row in rows[1:]]
        bars = axes[0].bar(x + offset, values, width=.34, color=color, label=label)
        for bar, row, value in zip(bars, rows[1:], values):
            axes[0].text(bar.get_x()+bar.get_width()/2, value+.02,
                         f"{value:.3f}\nn={row[field]['denominator']}", ha="center", fontsize=9)
    axes[0].set(xticks=x, xticklabels=LABELS[1:], ylim=(0, 1.13), title="Conditional citation quality", ylabel="Mean score (0-1)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    bars = axes[1].bar(LABELS, [row["failed"] for row in rows], color=COLORS)
    axes[1].set(ylim=(0, 6.5), title="Execution failures retained in primary quality", ylabel="Failed runs / 120 per workflow")
    axes[1].tick_params(axis="x", labelrotation=12)
    for bar, row in zip(bars, rows):
        axes[1].text(bar.get_x()+bar.get_width()/2, row["failed"]+.15, f"{row['failed']}/120", ha="center")
    fig.suptitle("Citation structure validity does not establish source support", fontsize=15, y=.96)
    save(fig, "citations-and-failures", "Citation means include only applicable, successfully scored answers; each denominator is shown. Direct LLM: N/A.\nAll successful RAG answers pass the citation structure check. Seven execution failures remain zero in primary quality.")
    return figures
