"""
regime_analyzer.py — Analyze market regimes across historical data.

Runs RegimeDetector on a price CSV and outputs a timeline of which regime
the market was in at each bar.  Useful for understanding:
  - What fraction of time is trending vs ranging vs crisis
  - When regime transitions occur relative to major market events
  - Whether SRFM regime labels align with intuition

Usage:
    python tools/regime_analyzer.py --csv data/ES_hourly.csv --ticker ES
    python tools/regime_analyzer.py --csv data/ES_hourly.csv --plot --save-csv
"""

import argparse
import csv
import os
import sys
from collections import Counter
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from srfm_core import MinkowskiClassifier, MarketRegime
from regime import RegimeDetector


# --- Data loading -------------------------------------------------------------

def load_csv(path: str) -> List[Tuple[str, float]]:
    bars = []
    with open(path) as f:
        reader = csv.DictReader(f)
        prev = None
        for row in reader:
            close = float(row.get("close") or row.get("Close") or row.get("CLOSE") or 0)
            date  = row.get("date") or row.get("Date") or row.get("DATE") or row.get("time") or ""
            if prev and prev > 0:
                ret = (close - prev) / prev
                bars.append((date, ret))
            prev = close
    return bars


# --- Analysis -----------------------------------------------------------------

REGIME_COLORS = {
    MarketRegime.TRENDING:  "blue",
    MarketRegime.RANGING:   "gray",
    MarketRegime.CRISIS:    "red",
    MarketRegime.RECOVERY:  "green",
}


def analyze(bars: List[Tuple[str, float]], window: int) -> List[Tuple[str, float, MarketRegime]]:
    mink     = MinkowskiClassifier()
    detector = RegimeDetector(window=window)
    records  = []
    for date, ret in bars:
        causal = mink.update(ret)
        regime = detector.update(ret, causal)
        records.append((date, ret, regime))
    return records


def print_summary(records: List[Tuple[str, float, MarketRegime]], ticker: str):
    regimes = [r for _, _, r in records]
    counts  = Counter(regimes)
    total   = len(regimes)

    print(f"\n{'-'*50}")
    print(f"  Regime Summary — {ticker}  ({total} bars)")
    print(f"{'-'*50}")
    for regime in MarketRegime:
        n   = counts.get(regime, 0)
        pct = 100.0 * n / total if total else 0.0
        bar = "█" * int(pct / 2)
        print(f"  {regime.value:<10} {n:>6} bars  {pct:>5.1f}%  {bar}")
    print(f"{'-'*50}\n")

    # Transition points
    transitions = []
    prev = None
    for date, _, regime in records:
        if regime != prev:
            transitions.append((date, regime))
            prev = regime

    print(f"  Regime transitions ({len(transitions)} total):")
    for date, regime in transitions[:20]:
        print(f"    {date:<20}  -> {regime.value}")
    if len(transitions) > 20:
        print(f"    ... and {len(transitions) - 20} more")
    print()


def save_csv(records: List[Tuple[str, float, MarketRegime]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "return", "regime"])
        for date, ret, regime in records:
            writer.writerow([date, f"{ret:.6f}", regime.value])
    print(f"CSV saved -> {path}")


def plot_regimes(records: List[Tuple[str, float, MarketRegime]], ticker: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] matplotlib not installed.")
        return

    n      = len(records)
    xs     = list(range(n))
    rets   = [r for _, r, _ in records]
    prices = []
    p = 100.0
    for ret in rets:
        p *= (1 + ret)
        prices.append(p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Price with regime shading
    ax1.plot(xs, prices, color="black", linewidth=0.8)
    prev_regime = None
    seg_start   = 0
    for i, (_, _, regime) in enumerate(records):
        if regime != prev_regime or i == n - 1:
            if prev_regime is not None:
                ax1.axvspan(seg_start, i, alpha=0.2, color=REGIME_COLORS.get(prev_regime, "white"))
            seg_start  = i
            prev_regime = regime

    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r.value)
        for r in MarketRegime
    ]
    ax1.legend(handles=patches, loc="upper left")
    ax1.set_title(f"Regime Analysis — {ticker}", fontweight="bold")
    ax1.set_ylabel("Normalised Price (100 = start)")
    ax1.grid(alpha=0.2)

    # Regime numeric (for seeing transitions)
    regime_map = {MarketRegime.TRENDING: 3, MarketRegime.RECOVERY: 2, MarketRegime.RANGING: 1, MarketRegime.CRISIS: 0}
    regime_vals = [regime_map[r] for _, _, r in records]
    ax2.fill_between(xs, regime_vals, alpha=0.5, color="steelblue")
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(["CRISIS", "RANGING", "RECOVERY", "TRENDING"])
    ax2.set_xlabel("Bar index")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out = f"results/regimes_{ticker}.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Plot saved -> {out}")
    plt.show()


# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SRFM regime analyzer")
    parser.add_argument("--csv",      required=True, help="Price CSV path")
    parser.add_argument("--ticker",   default="UNKNOWN")
    parser.add_argument("--window",   type=int, default=30, help="Detector lookback window")
    parser.add_argument("--plot",     action="store_true")
    parser.add_argument("--save-csv", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.csv}...", end=" ")
    bars = load_csv(args.csv)
    print(f"{len(bars)} bars")

    records = analyze(bars, args.window)
    print_summary(records, args.ticker)

    if args.save_csv:
        save_csv(records, f"results/regimes_{args.ticker}.csv")

    if args.plot:
        plot_regimes(records, args.ticker)


if __name__ == "__main__":
    main()
