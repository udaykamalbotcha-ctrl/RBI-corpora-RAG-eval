"""CI Quality & Regression Gate for RBI Regulatory RAG.

Validates benchmark results against production quality gates defined in
config/production.yaml. Exits with code 1 if any SLA is breached.
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def load_yaml_config(config_path: Path) -> dict:
    content = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(content)
    
    # Fallback minimal YAML parser for key-value structures
    cfg = {}
    current_section = None
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            parts = line.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip().split("#")[0].strip()
            if not val:
                current_section = key
                cfg[current_section] = {}
            else:
                try:
                    if "." in val:
                        parsed_val = float(val)
                    else:
                        parsed_val = int(val)
                except ValueError:
                    parsed_val = val.strip("\"'")
                if current_section:
                    cfg[current_section][key] = parsed_val
                else:
                    cfg[key] = parsed_val
    return cfg


def main():
    repo_root = Path(__file__).resolve().parent.parent
    config_file = repo_root / "config" / "production.yaml"
    report_file = repo_root / "reports" / "exp_02_recursive.json"

    if not config_file.exists():
        print(f"Error: Production config not found at {config_file}")
        sys.exit(1)

    if not report_file.exists():
        print(f"Error: Production benchmark report not found at {report_file}")
        sys.exit(1)

    config_data = load_yaml_config(config_file)
    gates = config_data.get("quality_gates", {})

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract metrics from standard report payload or summary
    summary = data.get("summary", {})
    if not summary:
        ret = data.get("retrieval_metrics", {})
        gen = data.get("generation_metrics", {})
        lat = data.get("aggregate_latency", {})
        summary = {
            "hit_at_3": ret.get("mean_hit_at_3", 0.0),
            "mrr": ret.get("mrr", 0.0),
            "faithfulness": gen.get("mean_faithfulness", 0.0),
            "p95_latency_s": lat.get("p95_s", 0.0),
        }

    failures = []
    min_hit_3 = gates.get("min_hit_at_3", 0.85)
    min_mrr = gates.get("min_mrr", 0.75)
    min_faith = gates.get("min_faithfulness", 0.84)
    max_lat = gates.get("max_p95_latency_s", 0.50)

    actual_hit_3 = summary.get("hit_at_3", 0.0)
    actual_mrr = summary.get("mrr", 0.0)
    actual_faith = summary.get("faithfulness", 0.0)
    actual_lat = summary.get("p95_latency_s", 0.0)

    if actual_hit_3 < min_hit_3:
        failures.append(f"Hit@3: {actual_hit_3:.2f} < threshold ({min_hit_3})")
    if actual_mrr < min_mrr:
        failures.append(f"MRR: {actual_mrr:.4f} < threshold ({min_mrr})")
    if actual_faith < min_faith:
        failures.append(f"Faithfulness: {actual_faith:.2f} < threshold ({min_faith})")
    if actual_lat > max_lat:
        failures.append(f"p95 Latency: {actual_lat:.2f}s > threshold ({max_lat}s)")

    print("=" * 65)
    print("REGULATORY RAG CI REGRESSION GATE")
    print("=" * 65)
    print(f"Target Architecture : {config_data.get('pipeline', {}).get('name', 'rbi-regulatory-rag-prod')}")
    print(f"Evaluated Report    : {report_file.name}")
    print(f"  * Hit@3           : {actual_hit_3:.4f}  (Min Required: {min_hit_3})")
    print(f"  * MRR             : {actual_mrr:.4f}  (Min Required: {min_mrr})")
    print(f"  * Faithfulness    : {actual_faith:.4f}  (Min Required: {min_faith})")
    print(f"  * p95 Latency     : {actual_lat:.4f}s (Max Allowed: {max_lat}s)")
    print("-" * 65)

    if failures:
        print("[FAIL] CI QUALITY GATE FAILED:")
        for err in failures:
            print(f"  - {err}")
        sys.exit(1)

    print("[PASS] CI QUALITY GATE PASSED — All regulatory RAG SLOs successfully met.")
    print("=" * 65)


if __name__ == "__main__":
    main()
