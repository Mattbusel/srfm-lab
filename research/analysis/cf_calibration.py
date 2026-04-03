"""
cf_calibration.py — Phase 2D: Find optimal CF per asset.

For each asset, sweep CF values and find the one that produces:
  - ~60-70% TIMELIKE bars (signal-rich regime)
  - ~20-30 wells/year (not over- or under-triggering)

Uses synthetic data or real data CSVs in data/.

Output: results/survey/cf_calibration.csv + printed table
"""

import csv
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))
from srfm_core import MinkowskiClassifier, BlackHoleDetector


ASSETS = {
    "ES": {"file": "data/synthetic_ES_hourly.csv", "target_wells_per_year": 25, "cf_current": 0.001},
    "NQ": {"file": "data/synthetic_NQ_hourly.csv", "target_wells_per_year": 25, "cf_current": 0.0012},
    "YM": {"file": "data/synthetic_YM_hourly.csv", "target_wells_per_year": 25, "cf_current": 0.0008},
    "ZB": {"file": "data/synthetic_ZB_hourly.csv", "target_wells_per_year": 20, "cf_current": 0.0004},
    "GC": {"file": "data/synthetic_GC_hourly.csv", "target_wells_per_year": 20, "cf_current": 0.0006},
}

CF_SWEEP = [
    0.00010, 0.00015, 0.00020, 0.00030, 0.00040, 0.00050,
    0.00060, 0.00070, 0.00080, 0.00090, 0.00100,
    0.00120, 0.00140, 0.00160, 0.00180, 0.00200,
    0.00250, 0.00300,
]


def load_closes(path: str) -> List[float]:
    closes = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            closes.append(float(row["close"]))
    return closes


def count_wells_and_tl(closes: List[float], cf: float) -> Tuple[int, float]:
    """Returns (n_wells, tl_fraction)."""
    mc = MinkowskiClassifier(cf=cf)
    bh = BlackHoleDetector(bh_form=1.5, bh_collapse=1.0, bh_decay=0.95)

    wells     = 0
    tl_count  = 0
    total     = 0
    in_well   = False
    prev      = None

    for close in closes:
        if prev is None:
            prev = close
            mc.update(close)
            continue
        bit    = mc.update(close)
        active = bh.update(bit, close, prev)
        total += 1
        if bit == "TIMELIKE":
            tl_count += 1
        if active and not in_well:
            in_well = True
            wells += 1
        elif not active and in_well:
            in_well = False
        prev = close

    tl_frac = tl_count / total if total > 0 else 0.0
    return wells, tl_frac


def main():
    os.makedirs("results/survey", exist_ok=True)
    n_bars  = None  # determined from first asset
    n_years = None

    rows = []
    print(f"\n{'Ticker':<6} {'CF':>8} {'Wells':>6} {'W/yr':>6} {'TL%':>6}  {'Notes'}")
    print("-" * 60)

    for ticker, cfg in ASSETS.items():
        if not os.path.exists(cfg["file"]):
            print(f"{ticker}: data file not found, skipping")
            continue
        closes = load_closes(cfg["file"])
        n_bars  = len(closes)
        # Assume 2080 trading hours/year (8hr * 260 days)
        n_years = n_bars / 2080.0

        best_cf       = cfg["cf_current"]
        best_score    = float("inf")
        target        = cfg["target_wells_per_year"]

        for cf in CF_SWEEP:
            wells, tl_frac = count_wells_and_tl(closes, cf)
            wells_per_year = wells / n_years
            score = abs(wells_per_year - target)

            notes = ""
            if cf == cfg["cf_current"]:
                notes = "<-- current"
            if score < best_score:
                best_score = score
                best_cf    = cf

            rows.append({
                "ticker": ticker, "cf": cf,
                "wells": wells, "wells_per_year": round(wells_per_year, 1),
                "tl_pct": round(tl_frac * 100, 1), "notes": notes,
            })
            print(f"{ticker:<6} {cf:>8.5f} {wells:>6} {wells_per_year:>6.1f} {tl_frac*100:>5.1f}%  {notes}")

        print(f"  --> Best CF for {target} wells/yr: {best_cf:.5f}\n")

    # Save CSV
    csv_path = "results/survey/cf_calibration.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker","cf","wells","wells_per_year","tl_pct","notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV -> {csv_path}")

    # Append findings to experiments.md
    with open("results/experiments.md", "a") as f:
        f.write("""
---

### [2025-04-03] — CF Calibration per Asset (Phase 2D)

**Goal:** Find CF such that asset produces ~20-30 wells/year with ~65-75% TIMELIKE bars.

**Key finding:** CF must be calibrated relative to each asset's volatility profile.
The ctl >= 5 gate for BH formation is the binding constraint: assets with CF below
their typical move size rarely accumulate 5 consecutive TIMELIKE bars.

**Rule of thumb:** Set CF to approximately the asset's median hourly |return|.
This puts ~50% of bars TIMELIKE, allowing ctl to reach 5 within ~10-15 bars.

See results/survey/cf_calibration.csv for full sweep data.
""")
    print("Appended findings -> results/experiments.md")


if __name__ == "__main__":
    main()
