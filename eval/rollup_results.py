import json
from pathlib import Path
import numpy as np

EXPERIMENTS = [
    "exp_01_baseline",
    "exp_02_recursive",
    "exp_03_clause_aware",
    "exp_04_hybrid",
    "exp_05_hybrid_rerank",
]

# Example paid-tier rates (USD per 1M tokens) converted at ~₹85/USD;
# set to 0.0 if running strictly on a free tier.
USD_TO_INR = 85.0
INPUT_RATE_PER_1M_USD = 0.75   # e.g., Gemini Flash standard paid rate
OUTPUT_RATE_PER_1M_USD = 3.75


def extract_latency_p95(data: dict) -> str:
    # Check common keys in aggregate or per-question records
    agg = data.get("aggregate_latency", {})
    if isinstance(agg, dict) and "p95_s" in agg:
        return f"{agg['p95_s']:.2f}s"

    latencies = []
    for item in data.get("results", data.get("per_question", [])):
        val = item.get("latency_s") or item.get("latency")
        if isinstance(val, (int, float)):
            latencies.append(float(val))
    if latencies:
        return f"{np.percentile(latencies, 95):.2f}s"
    return "—"


def extract_cost_inr(data: dict) -> str:
    if "total_cost_inr" in data and isinstance(data["total_cost_inr"], (int, float)):
        return f"₹{data['total_cost_inr']:.2f}"

    in_tokens = 0
    out_tokens = 0
    for item in data.get("results", data.get("per_question", [])):
        in_tokens += item.get("input_tokens", 0) or 0
        out_tokens += item.get("output_tokens", 0) or 0

    if in_tokens == 0 and out_tokens == 0:
        usage = data.get("token_usage", {})
        in_tokens = usage.get("input_tokens", 0) or 0
        out_tokens = usage.get("output_tokens", 0) or 0

    if in_tokens == 0 and out_tokens == 0:
        return "—"

    cost_usd = (in_tokens / 1e6) * INPUT_RATE_PER_1M_USD + (out_tokens / 1e6) * OUTPUT_RATE_PER_1M_USD
    return f"₹{cost_usd * USD_TO_INR:.2f}"


def build_table(reports_dir: Path, out_file: Path) -> None:
    lines = [
        "# Master Results Table (Phase 4)",
        "",
        "| Experiment | Hit@3 | MRR | Faithfulness | p95 Latency | Cost per 60 Queries (₹) |",
        "|------------|------|-----|--------------|-------------|--------------------------|",
    ]

    for exp in EXPERIMENTS:
        report_path = reports_dir / f"{exp}.json"
        if not report_path.exists():
            lines.append(f"| {exp} | 0.00 | 0.0000 | 0.00 | — | — |")
            continue

        data = json.loads(report_path.read_text(encoding="utf-8"))
        ret = data.get("retrieval_metrics", data.get("aggregate_retrieval", {}))
        gen = data.get("generation_metrics", data.get("aggregate_generation", {}))

        hit_at_3 = ret.get("mean_hit_at_3", ret.get("hit_at_3", ret.get("Hit@3", 0.0)))
        mrr = ret.get("mrr", ret.get("MRR", ret.get("mean_reciprocal_rank", 0.0)))
        faithfulness = gen.get("mean_faithfulness", gen.get("faithfulness", gen.get("Faithfulness", 0.0)))
        latency_str = extract_latency_p95(data)
        cost_str = extract_cost_inr(data)

        lines.append(
            f"| {exp} | {hit_at_3:.2f} | {mrr:.4f} | {faithfulness:.2f} | {latency_str} | {cost_str} |"
        )

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    out_master = repo_root / "master_results.md"
    build_table(reports_dir, out_master)
    # Also write to reports/master_results.md if reports dir exists
    (reports_dir / "master_results.md").write_text(out_master.read_text(encoding="utf-8"), encoding="utf-8")
