"""
correlation.py — BH-activation correlation matrix for Spacetime Arena.

For each instrument pair:
  - Jaccard similarity = days_both_active / days_either_active
  - Pearson correlation of binary BH-active time series

Finds minimum-correlation portfolio via greedy algorithm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    instruments: List[str]
    jaccard_matrix: np.ndarray      # NxN
    pearson_matrix: np.ndarray      # NxN
    optimal_portfolio: List[str]
    diversification_score: float    # 1 - mean_jaccard of portfolio pairs
    bh_active_series: Dict[str, pd.Series]   # binary series per instrument


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute_correlation(
    bh_active_series: Dict[str, pd.Series],
) -> CorrelationResult:
    """
    Compute BH-activation correlation matrices.

    Parameters
    ----------
    bh_active_series : {sym: pd.Series(bool, DatetimeIndex)} — daily BH active flags
    """
    syms = list(bh_active_series.keys())
    n    = len(syms)

    if n == 0:
        raise ValueError("No instruments provided")

    # Align all series to common index
    df = pd.DataFrame(bh_active_series).fillna(False).astype(float)
    df = df.sort_index()

    # Common-index boolean arrays
    arrays = {s: df[s].values for s in syms}

    jaccard = np.zeros((n, n))
    pearson = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = arrays[syms[i]]
            b = arrays[syms[j]]
            if i == j:
                jaccard[i, j] = 1.0
                pearson[i, j] = 1.0
                continue

            both   = float(np.sum(a * b))
            either = float(np.sum((a + b) > 0))
            jaccard[i, j] = both / either if either > 0 else 0.0

            std_a = float(np.std(a))
            std_b = float(np.std(b))
            if std_a > 0 and std_b > 0:
                pearson[i, j] = float(np.corrcoef(a, b)[0, 1])
            else:
                pearson[i, j] = 0.0

    optimal, div_score = _greedy_min_correlation_portfolio(syms, jaccard)

    return CorrelationResult(
        instruments=syms,
        jaccard_matrix=jaccard,
        pearson_matrix=pearson,
        optimal_portfolio=optimal,
        diversification_score=div_score,
        bh_active_series={s: df[s] for s in syms},
    )


def _greedy_min_correlation_portfolio(
    syms: List[str],
    jaccard: np.ndarray,
) -> Tuple[List[str], float]:
    """
    Greedy algorithm: start with the instrument that has the lowest total
    correlation to all others, then add instruments that minimally increase
    average portfolio BH-overlap.
    """
    n = len(syms)
    if n <= 1:
        return syms, 1.0

    # Total correlation per instrument
    total_corr = jaccard.sum(axis=1) - 1.0   # subtract self-correlation
    first_idx  = int(np.argmin(total_corr))

    portfolio = [first_idx]
    remaining = list(range(n))
    remaining.remove(first_idx)

    while remaining:
        best_idx = -1
        best_increase = float("inf")

        for idx in remaining:
            # Mean Jaccard to current portfolio
            increase = float(np.mean([jaccard[idx, p] for p in portfolio]))
            if increase < best_increase:
                best_increase = increase
                best_idx = idx

        portfolio.append(best_idx)
        remaining.remove(best_idx)

    # Compute portfolio diversification score = 1 - mean_jaccard of portfolio pairs
    portfolio_pairs = []
    for i in range(len(portfolio)):
        for j in range(i + 1, len(portfolio)):
            portfolio_pairs.append(jaccard[portfolio[i], portfolio[j]])

    div_score = 1.0 - float(np.mean(portfolio_pairs)) if portfolio_pairs else 1.0
    optimal_syms = [syms[i] for i in portfolio]

    return optimal_syms, div_score


# ---------------------------------------------------------------------------
# Run full correlation from BHEngine results
# ---------------------------------------------------------------------------

def run_correlation_from_bar_states(
    bar_states_per_sym: Dict[str, Any],  # {sym: List[BarState]}
) -> CorrelationResult:
    """
    Build correlation from BHEngine bar_states output.

    Parameters
    ----------
    bar_states_per_sym : {sym: List[BarState]} from BacktestResult.bar_states
    """
    series: Dict[str, pd.Series] = {}

    for sym, bar_states in bar_states_per_sym.items():
        if not bar_states:
            continue
        timestamps = [bs.timestamp for bs in bar_states]
        active     = [bs.bh_active_1d or bs.bh_active_1h for bs in bar_states]
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
        s   = pd.Series(active, index=idx, dtype=float)
        # Resample to daily: any hour active → day active
        s = s.resample("1D").max().fillna(0.0)
        series[sym] = s

    return compute_correlation(series)


# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------

def correlation_to_dict(result: CorrelationResult) -> Dict[str, Any]:
    return {
        "instruments":          result.instruments,
        "jaccard_matrix":       result.jaccard_matrix.tolist(),
        "pearson_matrix":       result.pearson_matrix.tolist(),
        "optimal_portfolio":    result.optimal_portfolio,
        "diversification_score": result.diversification_score,
    }
