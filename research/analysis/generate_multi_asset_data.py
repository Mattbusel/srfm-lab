"""
Generate synthetic hourly price data for ES, NQ, YM, ZB, GC.

Uses correlated GBM with different volatility profiles per asset.
Correlation structure:
  ES/NQ/YM: rho=0.85 (equity index trio — highly correlated)
  ZB: rho=-0.30 with equities (bond futures — mild inverse)
  GC: rho=-0.10 with equities (gold — weak negative)

Output: data/synthetic_{TICKER}_hourly.csv for each asset
"""

import csv
import os
import numpy as np
from datetime import datetime, timedelta

RNG = np.random.default_rng(99)

ASSETS = {
    "ES": {"start_price": 2700.0,  "mu":  0.00015,  "sigma": 0.00080, "cf": 0.001},
    "NQ": {"start_price": 6500.0,  "mu":  0.00020,  "sigma": 0.00110, "cf": 0.0012},
    "YM": {"start_price": 24000.0, "mu":  0.00013,  "sigma": 0.00075, "cf": 0.0008},
    "ZB": {"start_price": 145.0,   "mu":  0.000050, "sigma": 0.00030, "cf": 0.0004},
    "GC": {"start_price": 1300.0,  "mu":  0.000060, "sigma": 0.00050, "cf": 0.0006},
}

# Cholesky: [ES, NQ, YM, ZB, GC]
CORR = np.array([
    [1.00,  0.85,  0.88, -0.30, -0.10],
    [0.85,  1.00,  0.82, -0.28, -0.08],
    [0.88,  0.82,  1.00, -0.25, -0.06],
    [-0.30,-0.28, -0.25,  1.00,  0.15],
    [-0.10,-0.08, -0.06,  0.15,  1.00],
])

REGIMES = {
    "bull":     {"mu_scale": 1.0,  "sig_scale": 1.0,  "dur": (200, 800)},
    "bear":     {"mu_scale": -2.0, "sig_scale": 1.6,  "dur": (100, 400)},
    "sideways": {"mu_scale": 0.3,  "sig_scale": 0.75, "dur": (100, 500)},
    "crisis":   {"mu_scale": -5.0, "sig_scale": 3.5,  "dur": (20, 80)},
}

TRANSITION = {
    "bull":     {"bull": 0.60, "sideways": 0.30, "bear": 0.05, "crisis": 0.05},
    "bear":     {"bear": 0.50, "sideways": 0.30, "bull": 0.10, "crisis": 0.10},
    "sideways": {"sideways": 0.40, "bull": 0.35, "bear": 0.20, "crisis": 0.05},
    "crisis":   {"bear": 0.50, "sideways": 0.35, "bull": 0.10, "crisis": 0.05},
}


def next_regime(current: str) -> str:
    probs = TRANSITION[current]
    return str(RNG.choice(list(probs.keys()), p=list(probs.values())))


def generate_all(n_hours: int = 52_560) -> dict:
    tickers = list(ASSETS.keys())
    n       = len(tickers)
    chol    = np.linalg.cholesky(CORR)

    prices  = {t: ASSETS[t]["start_price"] for t in tickers}
    bars    = {t: [] for t in tickers}

    dt             = datetime(2018, 1, 2, 9, 0, 0)
    current_regime = "bull"
    bars_in_regime = 0
    regime_len     = int(RNG.integers(*REGIMES[current_regime]["dur"]))

    hours_generated = 0
    while hours_generated < n_hours:
        while dt.weekday() >= 5:
            dt += timedelta(hours=1)
        if dt.hour < 9 or dt.hour >= 17:
            dt += timedelta(hours=1)
            continue

        reg   = REGIMES[current_regime]
        z_raw = RNG.standard_normal(n)
        z_cor = chol @ z_raw  # correlated shocks

        date_str = dt.strftime("%Y-%m-%d %H:%M")
        for i, ticker in enumerate(tickers):
            a   = ASSETS[ticker]
            mu  = a["mu"] * reg["mu_scale"]
            sig = a["sigma"] * reg["sig_scale"]
            ret = mu + sig * float(z_cor[i])
            ret = max(-0.05, min(0.05, ret))

            close  = prices[ticker] * (1 + ret)
            high   = close * (1 + abs(float(RNG.standard_normal())) * sig * 0.5)
            low    = close * (1 - abs(float(RNG.standard_normal())) * sig * 0.5)
            volume = max(1000, int(100000 * (1 + float(RNG.standard_normal()) * 0.3)))

            bars[ticker].append({
                "date":   date_str,
                "open":   round(prices[ticker], 4),
                "high":   round(high, 4),
                "low":    round(low, 4),
                "close":  round(close, 4),
                "volume": volume,
                "regime": current_regime,
            })
            prices[ticker] = close

        dt += timedelta(hours=1)
        hours_generated += 1
        bars_in_regime  += 1

        if bars_in_regime >= regime_len:
            current_regime = next_regime(current_regime)
            regime_len     = int(RNG.integers(*REGIMES[current_regime]["dur"]))
            bars_in_regime = 0

    return bars


def main():
    os.makedirs("data", exist_ok=True)
    print("Generating correlated multi-asset synthetic data (2018-2024)...")
    all_bars = generate_all()

    for ticker, bars in all_bars.items():
        path = f"data/synthetic_{ticker}_hourly.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date","open","high","low","close","volume","regime"])
            writer.writeheader()
            writer.writerows(bars)
        print(f"  {ticker}: {len(bars)} bars -> {path}")


if __name__ == "__main__":
    main()
