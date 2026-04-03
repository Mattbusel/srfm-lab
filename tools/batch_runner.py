"""
batch_runner.py — Run N strategy variants in parallel via Docker + lean backtest.

Usage:
    python tools/batch_runner.py strategies/larsa-v1 variants.json
    python tools/batch_runner.py strategies/larsa-v1 variants.json --workers 4 --metric SharpeRatio

variants.json format:
    [
        {"name": "v1-bh1.0",  "params": {"BH_FORM": 1.0, "BH_COLLAPSE": 0.35}},
        {"name": "v1-bh1.5",  "params": {"BH_FORM": 1.5, "BH_COLLAPSE": 0.40}},
        {"name": "v1-bh2.0",  "params": {"BH_FORM": 2.0, "BH_COLLAPSE": 0.45}}
    ]

Each variant:
    1. Copies the strategy directory to a temp location
    2. Patches the constants in main.py
    3. Runs `lean backtest` (via subprocess or Docker)
    4. Collects the result JSON

Final output: comparison table (calls compare.py logic internally).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─── Variant execution ────────────────────────────────────────────────────────

def patch_constants(src: str, params: Dict[str, float], dst: str):
    with open(src) as f:
        code = f.read()
    for name, value in params.items():
        pattern = rf"^({re.escape(name)}\s*=\s*)[^\n#]+"
        code, n = re.subn(pattern, rf"\g<1>{value!r}", code, flags=re.MULTILINE)
        if n == 0:
            print(f"  [WARN] Constant '{name}' not found in {src}", file=sys.stderr)
    with open(dst, "w") as f:
        f.write(code)


def run_variant(
    strategy_dir: str,
    variant_name: str,
    params: Dict[str, float],
    output_base: str,
    use_docker: bool = False,
) -> Tuple[str, Optional[str]]:
    """Run one variant. Returns (variant_name, result_json_path or None)."""
    out_dir = os.path.join(output_base, variant_name)
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_strategy = os.path.join(tmp, "strategy")
        shutil.copytree(strategy_dir, tmp_strategy)

        src_main = os.path.join(strategy_dir, "main.py")
        dst_main = os.path.join(tmp_strategy, "main.py")
        patch_constants(src_main, params, dst_main)

        cmd = ["lean", "backtest", tmp_strategy, "--output", out_dir]
        if use_docker:
            cmd.append("--docker")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"  [{variant_name}] FAILED\n{result.stderr[:500]}", file=sys.stderr)
            return variant_name, None

        # Find result JSON
        for fname in ["result.json", "backtest-results.json"]:
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                return variant_name, p
        jsons = list(Path(out_dir).glob("*.json"))
        return variant_name, str(jsons[0]) if jsons else None


# ─── Result loading (mirrors compare.py) ─────────────────────────────────────

METRIC_KEYS = {
    "Total Return":  ("TotalPerformance", "PortfolioStatistics", "TotalNetProfit"),
    "CAGR":          ("TotalPerformance", "PortfolioStatistics", "CompoundingAnnualReturn"),
    "Sharpe":        ("TotalPerformance", "PortfolioStatistics", "SharpeRatio"),
    "Sortino":       ("TotalPerformance", "PortfolioStatistics", "SortinoRatio"),
    "Max Drawdown":  ("TotalPerformance", "PortfolioStatistics", "Drawdown"),
    "Win Rate":      ("TotalPerformance", "PortfolioStatistics", "WinRate"),
}

ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"
ANSI_BOLD  = "\033[1m"


def _dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def load_metrics(result_json: str) -> Dict:
    try:
        with open(result_json) as f:
            data = json.load(f)
        row = {}
        for metric, key_path in METRIC_KEYS.items():
            val = _dig(data, *key_path)
            if isinstance(val, str):
                val = val.replace("%", "").strip()
                try:
                    val = float(val)
                except ValueError:
                    pass
            row[metric] = val
        return row
    except Exception:
        return {}


def print_results(rows: List[Dict]):
    if not rows:
        return
    cols = ["Name"] + list(METRIC_KEYS.keys())
    widths = {c: len(c) for c in cols}
    for r in rows:
        widths["Name"] = max(widths["Name"], len(r.get("Name", "")))
        for m in METRIC_KEYS:
            v = r.get(m)
            widths[m] = max(widths[m], len(f"{v:.4f}" if isinstance(v, float) else str(v or "—")))

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep    = "  ".join("-" * widths[c] for c in cols)
    print(f"\n{ANSI_BOLD}{header}{ANSI_RESET}")
    print(sep)
    for r in rows:
        parts = [r.get("Name", "?").ljust(widths["Name"])]
        for m in METRIC_KEYS:
            v = r.get(m)
            cell = (f"{v:.4f}" if isinstance(v, float) else str(v or "—")).ljust(widths[m])
            parts.append(cell)
        print("  ".join(parts))
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parallel SRFM batch runner")
    parser.add_argument("strategy",  help="Path to base strategy directory")
    parser.add_argument("variants",  help="Path to variants JSON file")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default: 2)")
    parser.add_argument("--metric",  default="Sharpe", help="Sort metric for final table")
    parser.add_argument("--docker",  action="store_true", help="Use Docker for LEAN backtest")
    args = parser.parse_args()

    with open(args.variants) as f:
        variants: List[Dict] = json.load(f)

    strategy_name = Path(args.strategy).name
    output_base = os.path.join("results", strategy_name, "batch")

    print(f"Running {len(variants)} variants with {args.workers} workers...\n")

    results: Dict[str, Optional[str]] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_variant,
                args.strategy,
                v["name"],
                v.get("params", {}),
                output_base,
                args.docker,
            ): v["name"]
            for v in variants
        }
        for fut in as_completed(futures):
            name, result_path = fut.result()
            status = "OK" if result_path else "FAILED"
            print(f"  [{status}] {name}")
            results[name] = result_path

    # Collect metrics
    rows = []
    for v in variants:
        name = v["name"]
        result_path = results.get(name)
        row = {"Name": name, "Params": json.dumps(v.get("params", {})), "ResultPath": result_path or ""}
        if result_path:
            row.update(load_metrics(result_path))
        rows.append(row)

    # Sort
    if args.metric in METRIC_KEYS:
        rows.sort(key=lambda r: r.get(args.metric) or float("-inf"), reverse=True)

    print_results(rows)

    # Save summary
    summary_path = os.path.join(output_base, "batch_summary.json")
    os.makedirs(output_base, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
