"""
Generate synthetic ES-like hourly price data for tool testing.

Uses a regime-switching GBM model calibrated roughly to ES 2018-2024:
  - Bull periods: mu=0.0002/hr, sigma=0.0008
  - Bear periods: mu=-0.0003/hr, sigma=0.0014
  - Sideways: mu=0.00005/hr, sigma=0.0006
  - Crisis: mu=-0.001/hr, sigma=0.003

Output: data/synthetic_ES_hourly.csv
"""

import csv
import math
import os
import numpy as np
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

REGIMES = {
    "bull":     {"mu": 0.0002,  "sigma": 0.0008,  "duration_range": (200, 800)},
    "bear":     {"mu": -0.0003, "sigma": 0.0014,  "duration_range": (100, 400)},
    "sideways": {"mu": 0.00005, "sigma": 0.0006,  "duration_range": (100, 500)},
    "crisis":   {"mu": -0.001,  "sigma": 0.003,   "duration_range": (20, 80)},
}

TRANSITION = {
    "bull":     {"bull": 0.60, "sideways": 0.30, "bear": 0.05, "crisis": 0.05},
    "bear":     {"bear": 0.50, "sideways": 0.30, "bull": 0.10, "crisis": 0.10},
    "sideways": {"sideways": 0.40, "bull": 0.35, "bear": 0.20, "crisis": 0.05},
    "crisis":   {"bear": 0.50, "sideways": 0.35, "bull": 0.10, "crisis": 0.05},
}


def next_regime(current: str) -> str:
    probs   = TRANSITION[current]
    regimes = list(probs.keys())
    weights = list(probs.values())
    return str(RNG.choice(regimes, p=weights))


def generate(n_hours: int = 52_560, start_price: float = 2700.0) -> list:
    """Generate n_hours of synthetic ES hourly OHLCV bars."""
    bars = []
    price = start_price
    dt    = datetime(2018, 1, 2, 9, 0, 0)

    current_regime = "bull"
    bars_in_regime = 0
    regime_len     = int(RNG.integers(*REGIMES[current_regime]["duration_range"]))

    for _ in range(n_hours):
        # Skip weekends
        while dt.weekday() >= 5:
            dt += timedelta(hours=1)
        # Skip outside RTH (roughly)
        if dt.hour < 9 or dt.hour >= 17:
            dt += timedelta(hours=1)
            continue

        r      = REGIMES[current_regime]
        ret    = r["mu"] + r["sigma"] * float(RNG.standard_normal())
        ret    = max(-0.05, min(0.05, ret))   # cap at 5% per hour
        close  = price * (1 + ret)
        high   = close * (1 + abs(float(RNG.standard_normal())) * r["sigma"] * 0.5)
        low    = close * (1 - abs(float(RNG.standard_normal())) * r["sigma"] * 0.5)
        volume = max(1000, int(100000 * (1 + float(RNG.standard_normal()) * 0.3)))

        bars.append({
            "date":   dt.strftime("%Y-%m-%d %H:%M"),
            "open":   round(price, 2),
            "high":   round(high, 2),
            "low":    round(low, 2),
            "close":  round(close, 2),
            "volume": volume,
            "regime": current_regime,
        })

        price  = close
        dt    += timedelta(hours=1)
        bars_in_regime += 1

        if bars_in_regime >= regime_len:
            current_regime = next_regime(current_regime)
            regime_len     = int(RNG.integers(*REGIMES[current_regime]["duration_range"]))
            bars_in_regime = 0

    return bars


def main():
    os.makedirs("data", exist_ok=True)
    print("Generating synthetic ES hourly data (2018-2024)...")
    bars = generate(n_hours=52_560)
    path = "data/synthetic_ES_hourly.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume", "regime"])
        writer.writeheader()
        writer.writerows(bars)
    print(f"Generated {len(bars)} bars -> {path}")

    # Summary
    from collections import Counter
    counts = Counter(b["regime"] for b in bars)
    for r, n in counts.items():
        print(f"  {r:<10} {n:>6} bars ({100*n/len(bars):.1f}%)")


if __name__ == "__main__":
    main()
