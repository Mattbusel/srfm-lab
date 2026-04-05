"""
onchain/metrics/hodl_waves.py
──────────────────────────────
HODL Wave Analysis — Coin Age Distribution.

Financial rationale
───────────────────
HODL Waves show what percentage of the circulating supply last moved within
each age band.  The key insight is that coin age acts as a proxy for holder
conviction:

  Short-Term Holders (STH, <155 days): recent buyers, more price-sensitive,
    more likely to sell on volatility.
  Long-Term Holders (LTH, >155 days): battle-hardened HODLers who have
    historically held through multiple corrections without selling.

Key patterns:
  Rising STH % (young coins growing) → accumulation phase ending, distribution
    beginning — historically precedes tops by 1–3 months.
  Rising LTH % (old coins growing) → accumulation — coins absorbed by
    conviction holders who are unlikely to sell.
  HODL Wave "inversion": when the 1–2 year band surges it often marks that
    coins from the prior cycle bottom have been accumulated and holders
    are now sitting on large unrealised profits (near top warning).

Age bands (standard CoinMetrics/Glassnode convention)
───────────────────────────────────────────────────────
  <1d | 1d–1w | 1w–1m | 1m–3m | 3m–6m | 6m–1y | 1y–2y | 2y+

Simulation model
────────────────
We simulate the age distribution using an exponential decay model:
  - Each day, a fraction of each age band's supply "ages" into the next band.
  - A fraction proportional to price volatility "resets" to the <1d band
    (simulating spending induced by price moves).
  - The result is a realistic time-varying age distribution that exhibits
    the empirical HODL wave patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Age band labels (in days: inclusive lower, exclusive upper)
AGE_BANDS: List[tuple[str, int, int]] = [
    ("<1d",    0,   1),
    ("1d-1w",  1,   7),
    ("1w-1m",  7,  30),
    ("1m-3m", 30,  90),
    ("3m-6m", 90, 180),
    ("6m-1y", 180, 365),
    ("1y-2y", 365, 730),
    ("2y+",   730, 99999),
]
BAND_NAMES = [b[0] for b in AGE_BANDS]

# Short-term / long-term threshold (days)
STH_THRESHOLD_DAYS = 155
LTH_THRESHOLD_DAYS = 155


@dataclass
class HODLWaveResult:
    symbol: str
    bands: Dict[str, float]          # {band_name: pct_of_supply}  sum ≈ 1.0
    sth_pct: float                   # short-term holder %  (<155d)
    lth_pct: float                   # long-term holder %   (≥155d)
    sth_30d_change: float            # 30d delta in STH % (+ = more young coins)
    lth_30d_change: float            # 30d delta in LTH %
    signal: float                    # [-1, +1]
    regime: str                      # "ACCUMULATION" | "DISTRIBUTION" | "NEUTRAL"
    computed_at: str


def _simulate_hodl_waves(price_series: pd.Series, n_bands: int = 8) -> pd.DataFrame:
    """Simulate HODL wave band percentages from daily price history.

    Model:
      state[t] = vector of length n_bands (fraction of supply in each age band)
      Each day:
        1. A "spending rate" sr[t] based on price volatility is computed.
           High volatility → more on-chain activity → more coins reset to <1d.
        2. Coins in each band flow forward (age) at a slow base rate.
        3. sr[t] fraction of all bands resets to the <1d band.

    The model is calibrated so that:
      - In low-volatility periods: old coins accumulate (LTH grows).
      - In high-volatility/bull market periods: STH surges (distribution).
    """
    rng = np.random.default_rng(seed=42)
    prices = price_series.values.astype(float)
    n = len(prices)

    # 30-day rolling volatility (annualised)
    log_ret = np.diff(np.log(prices + 1e-9))
    vol = np.zeros(n)
    for i in range(1, n):
        start = max(0, i - 30)
        vol[i] = np.std(log_ret[start:i]) * np.sqrt(365) if i > start else 0.3

    # Initialise: most supply is old (2y+ dominant at genesis/cold start)
    # Approximation: exponential decay across bands
    state = np.array([0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.45], dtype=float)
    state /= state.sum()

    # Ageing transition matrix (coins slowly move from younger to older bands)
    # Base daily transition prob from band i to band i+1
    base_aging_rates = np.array([1.0/1, 1.0/7, 1.0/23, 1.0/60, 1.0/90, 1.0/185, 1.0/365, 0.0])

    history = np.zeros((n, n_bands))
    history[0] = state.copy()

    for t in range(1, n):
        # Spending rate proportional to volatility (clipped)
        sr = np.clip(vol[t] * 0.15 + rng.normal(0, 0.005), 0.001, 0.08)

        new_state = state.copy()

        # Step 1: Ageing — fraction of each band ages into the next
        for i in range(n_bands - 1):
            aged = state[i] * base_aging_rates[i]
            new_state[i]     -= aged
            new_state[i + 1] += aged

        # Step 2: Spending — fraction resets to <1d band
        spent_total = 0.0
        for i in range(n_bands):
            # Older coins are less likely to be spent
            age_weight = 1.0 / (i + 1)
            spent = new_state[i] * sr * age_weight * 0.5
            new_state[i]  -= spent
            spent_total   += spent
        new_state[0] += spent_total  # reset to <1d

        # Normalise to ensure sum = 1
        total = new_state.sum()
        if total > 0:
            new_state /= total

        state = new_state
        history[t] = state.copy()

    return pd.DataFrame(history, index=price_series.index, columns=BAND_NAMES)


def compute_hodl_waves(
    price_series: pd.Series,
    symbol: str = "BTC-USD",
) -> HODLWaveResult:
    """Compute HODL Wave analysis for the given price history.

    Parameters
    ----------
    price_series:
        Daily close prices, DatetimeIndex, at least 180 bars.
    symbol:
        Ticker label.

    Returns
    -------
    HODLWaveResult with per-band percentages, STH/LTH metrics, and signal.
    """
    if price_series.empty or len(price_series) < 180:
        raise ValueError("Need at least 180 price bars for HODL wave analysis")

    price_series = price_series.dropna().sort_index()
    logger.info("HODL Waves: computing simulated age distribution (%d bars)", len(price_series))

    wave_df = _simulate_hodl_waves(price_series)

    # Compute STH / LTH percentages
    # STH bands: <1d, 1d-1w, 1w-1m, 1m-3m, 3m-6m  (all <180d approx)
    sth_bands = ["<1d", "1d-1w", "1w-1m", "1m-3m", "3m-6m"]
    lth_bands = ["6m-1y", "1y-2y", "2y+"]

    sth_series = wave_df[sth_bands].sum(axis=1)
    lth_series = wave_df[lth_bands].sum(axis=1)

    current_sth = float(sth_series.iloc[-1])
    current_lth = float(lth_series.iloc[-1])

    # 30-day rate of change
    sth_30d_change = float(sth_series.iloc[-1] - sth_series.iloc[-31]) if len(sth_series) > 30 else 0.0
    lth_30d_change = float(lth_series.iloc[-1] - lth_series.iloc[-31]) if len(lth_series) > 30 else 0.0

    current_bands = {band: round(float(wave_df[band].iloc[-1]), 4) for band in BAND_NAMES}

    signal, regime = _waves_to_signal(sth_30d_change, lth_30d_change, current_sth, current_lth)

    return HODLWaveResult(
        symbol=symbol,
        bands=current_bands,
        sth_pct=round(current_sth, 4),
        lth_pct=round(current_lth, 4),
        sth_30d_change=round(sth_30d_change, 4),
        lth_30d_change=round(lth_30d_change, 4),
        signal=signal,
        regime=regime,
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def _waves_to_signal(
    sth_change: float,
    lth_change: float,
    sth_pct: float,
    lth_pct: float,
) -> tuple[float, str]:
    """Derive signal and regime label from HODL wave dynamics.

    Logic:
      - Rising LTH + falling STH = coins moving to strong hands = ACCUMULATION = bullish
      - Rising STH + falling LTH = distribution phase = bearish
      - LTH > 65% = extreme accumulation (historic buy zone)
      - STH > 50% = peak speculation (historic sell zone)
    """
    # Base score from 30d dynamics
    score = lth_change * 5.0 - sth_change * 5.0  # net flow score

    # Absolute level adjustments
    if lth_pct > 0.65:
        score += 0.3   # extreme accumulation
    elif lth_pct > 0.55:
        score += 0.1
    if sth_pct > 0.50:
        score -= 0.3   # peak distribution
    elif sth_pct > 0.40:
        score -= 0.1

    signal = float(np.clip(score, -1.0, 1.0))

    if lth_change > 0.01 and sth_change < 0:
        regime = "ACCUMULATION"
    elif sth_change > 0.01 and lth_change < 0:
        regime = "DISTRIBUTION"
    else:
        regime = "NEUTRAL"

    return round(signal, 3), regime


def hodl_summary(result: HODLWaveResult) -> str:
    return (
        f"HODL Waves — STH={result.sth_pct:.1%} (30d Δ={result.sth_30d_change:+.1%}) "
        f"LTH={result.lth_pct:.1%} (30d Δ={result.lth_30d_change:+.1%}) "
        f"— {result.regime} signal={result.signal:+.2f}"
    )
