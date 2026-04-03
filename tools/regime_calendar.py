"""
regime_calendar.py — LARSA regime heatmap calendar.

Shows regime per month as an ASCII calendar grid. Runs arena_v2 with verbose=True
to collect per-bar regime data, aggregates to monthly mode, and displays.

Usage:
    python tools/regime_calendar.py --csv data/NDX_hourly_poly.csv --cf 0.005
    python tools/regime_calendar.py --synthetic --n-bars 5000
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np

# ── Arena import ──────────────────────────────────────────────────────────────
_ARENA_DIR = os.path.dirname(__file__)
sys.path.insert(0, _ARENA_DIR)
from arena_v2 import run_v2, load_ohlcv, generate_synthetic


# ── Helpers ───────────────────────────────────────────────────────────────────

REGIME_SHORT = {
    "BULL": "BUL",
    "BEAR": "BER",
    "SIDEWAYS": "SWY",
    "HIGH_VOLATILITY": "HVL",
    "NEUTRAL": "NEU",
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _parse_date(date_str: str):
    """Return (year, month) from a date string, or None."""
    import re
    # ISO format: 2021-03-15T...
    m = re.match(r"(\d{4})-(\d{2})", date_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    # bar_000123 — synthetic, no date
    return None


def _monthly_mode(bars_regime: list) -> str:
    """Most common regime string in a list."""
    if not bars_regime:
        return "---"
    c = Counter(bars_regime)
    return c.most_common(1)[0][0]


def _consecutive_runs(monthly: dict):
    """Find BEAR periods > 30 bars (hours) consecutive."""
    # monthly is {(year,month): regime_short}
    sorted_keys = sorted(monthly)
    runs = []
    current_regime = None
    current_start = None
    current_count = 0
    for ym in sorted_keys:
        r = monthly[ym]
        if r == current_regime:
            current_count += 1
        else:
            if current_regime == "BER" and current_count >= 2:
                runs.append((current_start, ym, current_count))
            current_regime = r
            current_start = ym
            current_count = 1
    if current_regime == "BER" and current_count >= 2:
        runs.append((current_start, sorted_keys[-1] if sorted_keys else None, current_count))
    return runs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LARSA Regime Calendar Heatmap")
    parser.add_argument("--csv", help="Price CSV file")
    parser.add_argument("--ticker", default="NDX")
    parser.add_argument("--cf", type=float, default=0.005)
    parser.add_argument("--bh-form", type=float, default=1.5)
    parser.add_argument("--bh-decay", type=float, default=0.95)
    parser.add_argument("--max-leverage", type=float, default=0.65)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-bars", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp", default="", help="Arena experiment flags e.g. ABCD")
    parser.add_argument("--out", default="results/regime_calendar.md")
    args = parser.parse_args()

    # Load bars
    if args.synthetic:
        bars = generate_synthetic(args.n_bars, args.seed)
        ticker = "SYNTH"
    elif args.csv:
        bars = load_ohlcv(args.csv)
        ticker = args.ticker
        print(f"  Loaded {len(bars)} bars from {args.csv}")
    else:
        parser.error("Provide --csv or --synthetic")

    cfg = {
        "cf": args.cf,
        "bh_form": args.bh_form,
        "bh_collapse": 1.0,
        "bh_decay": args.bh_decay,
    }

    print(f"  Running arena_v2 (verbose) cf={args.cf} ...")
    broker, bar_log = run_v2(bars, cfg, args.max_leverage, args.exp, verbose=True)

    if not bar_log:
        print("ERROR: bar_log is empty. arena_v2 returned no signal data.")
        sys.exit(1)

    print(f"  Got {len(bar_log)} bar records.")

    # ── Aggregate per-bar regime to monthly buckets ──────────────────────────
    # bar_log entries: {date, close, regime, bh_active, bh_mass, ...}
    # regime is a string like "BULL", "BEAR", "SIDEWAYS", "HIGH_VOLATILITY"

    monthly_regimes: dict = defaultdict(list)  # (year, month) -> [regime_str, ...]
    hourly_regimes: list = []  # (date_str, regime_short) for stats

    skipped = 0
    for entry in bar_log:
        date_str = entry.get("date", "")
        regime_raw = entry.get("regime", "SIDEWAYS")
        ym = _parse_date(date_str)
        short = REGIME_SHORT.get(regime_raw, regime_raw[:3].upper())
        hourly_regimes.append(short)
        if ym:
            monthly_regimes[ym].append(regime_raw)
        else:
            skipped += 1

    # If synthetic (no real dates), build fake year/month from bar index
    if skipped > 0 and len(monthly_regimes) == 0:
        for i, entry in enumerate(bar_log):
            regime_raw = entry.get("regime", "SIDEWAYS")
            year = 2018 + i // (12 * 20)
            month = (i // 20) % 12 + 1
            monthly_regimes[(year, month)].append(regime_raw)

    # Build monthly mode map
    monthly_mode: dict = {}
    for ym, regimes in monthly_regimes.items():
        monthly_mode[ym] = REGIME_SHORT.get(_monthly_mode(regimes), "---")

    # ── Render calendar ───────────────────────────────────────────────────────
    years = sorted(set(y for y, m in monthly_mode))
    lines = []
    lines.append(f"REGIME CALENDAR — {ticker} (cf={args.cf})")
    lines.append("=" * 60)
    header = "        " + "  ".join(f"{mn:>4}" for mn in MONTH_NAMES)
    lines.append(header)

    for year in years:
        row = f"  {year}  "
        for month in range(1, 13):
            cell = monthly_mode.get((year, month), "   ")
            row += f"  {cell:>3}"
        lines.append(row)

    lines.append("")
    lines.append("Legend: BUL=BULL  BER=BEAR  SWY=SIDEWAYS  HVL=HIGH_VOL  NEU=NEUTRAL")

    # ── Regime stats ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("REGIME STATS:")
    total = len(hourly_regimes)
    regime_counts = Counter(hourly_regimes)
    full_names = {"BUL": "BULL", "BER": "BEAR", "SWY": "SIDEWAYS", "HVL": "HIGH_VOL", "NEU": "NEUTRAL"}
    for short, full in full_names.items():
        cnt = regime_counts.get(short, 0)
        pct = 100.0 * cnt / total if total else 0.0
        lines.append(f"  {full:<12}: {cnt:>6,} hours ({pct:.1f}%)")

    # ── Bear periods ─────────────────────────────────────────────────────────
    bear_runs = _consecutive_runs(monthly_mode)
    if bear_runs:
        lines.append("")
        lines.append("BEAR PERIODS (>= 2 consecutive months):")
        for start, end, count in bear_runs:
            sy, sm = start
            ey, em = end if end else start
            lines.append(
                f"  {MONTH_NAMES[sm-1]} {sy} – {MONTH_NAMES[em-1]} {ey}: {count} months"
            )

    output = "\n".join(lines)
    print()
    print(output)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("```\n" + output + "\n```\n")
    print(f"\n  Saved to {args.out}")


if __name__ == "__main__":
    main()
