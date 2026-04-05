"""
idea-engine/ingestion/statistical_filters/bootstrap_filter.py
──────────────────────────────────────────────────────────────
Politis-Romano stationary bootstrap for time-series PnL data,
with Benjamini-Hochberg FDR correction and effect-size gating.

Reference
─────────
  Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
  JASA, 89(428), 1303–1313.

  Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate:
  a practical and powerful approach to multiple testing. JRSS-B, 57(1), 289–300.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_BLOCK_LENGTH,
    BOOTSTRAP_N_RESAMPLES,
    MIN_EFFECT_SIZE,
)
from ..types import EffectSizeType, MinedPattern, PatternStatus

logger = logging.getLogger(__name__)


# ── Politis-Romano stationary bootstrap ──────────────────────────────────────

def _optimal_block_length(x: np.ndarray) -> int:
    """
    Estimate the optimal block length for stationary bootstrap using the
    Politis-Romano (1994) rule of thumb: b ≈ n^(1/3).

    For very short series we fall back to b=1 (i.i.d.).
    """
    n = len(x)
    if n < 10:
        return 1
    return max(1, int(round(n ** (1.0 / 3.0))))


def _stationary_bootstrap_resample(
    x: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw one stationary bootstrap resample of x.

    The block length is geometrically distributed with mean = block_length.
    """
    n = len(x)
    result = np.empty(n, dtype=x.dtype)
    idx    = 0
    p      = 1.0 / block_length  # geometric success probability

    while idx < n:
        start     = rng.integers(0, n)
        geo_len   = rng.geometric(p)
        needed    = min(geo_len, n - idx)
        for i in range(needed):
            result[idx] = x[(start + i) % n]
            idx += 1

    return result


def _bootstrap_test_mean(
    group: np.ndarray,
    baseline: np.ndarray,
    n_resamples:  int = BOOTSTRAP_N_RESAMPLES,
    block_length: Optional[int] = BOOTSTRAP_BLOCK_LENGTH,
    rng_seed:     int = 42,
) -> float:
    """
    Stationary bootstrap test of H0: mean(group) == mean(baseline).

    Returns a two-sided p-value.
    """
    if len(group) < 3 or len(baseline) < 3:
        return 1.0

    obs_diff = group.mean() - baseline.mean()
    bl_g     = block_length or _optimal_block_length(group)
    bl_b     = block_length or _optimal_block_length(baseline)
    rng      = np.random.default_rng(rng_seed)

    # Centre both arrays under H0
    pooled_mean = np.concatenate([group, baseline]).mean()
    g_centred   = group    - group.mean()    + pooled_mean
    b_centred   = baseline - baseline.mean() + pooled_mean

    count = 0
    for _ in range(n_resamples):
        g_r  = _stationary_bootstrap_resample(g_centred, bl_g, rng)
        b_r  = _stationary_bootstrap_resample(b_centred, bl_b, rng)
        diff = g_r.mean() - b_r.mean()
        if abs(diff) >= abs(obs_diff):
            count += 1

    return float(count / n_resamples)


# ── Effect size computation ───────────────────────────────────────────────────

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d pooled-variance effect size."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size ∈ [-1, 1]."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    dom = sum(1 if xi > xj else (-1 if xi < xj else 0) for xi in a for xj in b)
    return float(dom / (n1 * n2))


def compute_effect_size(
    group:    np.ndarray,
    baseline: np.ndarray,
    kind:     EffectSizeType = EffectSizeType.COHENS_D,
) -> float:
    if kind == EffectSizeType.CLIFFS_DELTA:
        return abs(cliffs_delta(group, baseline))
    return abs(cohens_d(group, baseline))


# ── BH multiple-testing correction ───────────────────────────────────────────

def benjamini_hochberg(p_values: List[float], alpha: float = BOOTSTRAP_ALPHA) -> List[bool]:
    """
    Benjamini-Hochberg FDR correction.

    Returns a list of bool (True = reject H0 at the given FDR level).
    """
    n = len(p_values)
    if n == 0:
        return []
    order   = sorted(range(n), key=lambda i: p_values[i])
    reject  = [False] * n
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / n:
            reject[idx] = True
    # BH monotonicity: once we stop at some rank, all earlier ranks also rejected
    # (guaranteed by sorted order and the threshold condition)
    return reject


# ── Main filter ───────────────────────────────────────────────────────────────

def filter_patterns(
    patterns:    List[MinedPattern],
    alpha:       float = BOOTSTRAP_ALPHA,
    min_effect:  float = MIN_EFFECT_SIZE,
    n_resamples: int   = BOOTSTRAP_N_RESAMPLES,
    block_length: Optional[int] = BOOTSTRAP_BLOCK_LENGTH,
) -> List[MinedPattern]:
    """
    Filter MinedPattern objects using the stationary bootstrap + BH correction.

    Algorithm
    ---------
    For each pattern that has raw_group and raw_baseline arrays:

    1. Compute a bootstrap p-value for mean(group) vs mean(baseline).
    2. Apply BH FDR correction across all patterns.
    3. Gate on minimum effect size.
    4. Mark surviving patterns as CONFIRMED, others as REJECTED.
    5. Set pattern.confidence to 1 - adjusted_p_value.

    Patterns that lack raw data (raw_group is None) fall back to their
    pre-computed p_value for the BH step.

    Parameters
    ----------
    patterns     : list of MinedPattern from any miner
    alpha        : FDR significance level
    min_effect   : minimum effect size (absolute) to survive
    n_resamples  : bootstrap resamples
    block_length : stationary bootstrap block length (None = auto)

    Returns
    -------
    The same list with status and confidence updated in place.
    (Also returns it for convenience.)
    """
    if not patterns:
        return patterns

    logger.info(
        "StatisticalFilter: evaluating %d pattern(s) at α=%.3f, min_effect=%.2f …",
        len(patterns), alpha, min_effect,
    )

    # ── Step 1: compute bootstrap p-values ──────────────────────────────────
    p_bootstrap: List[float] = []
    effect_sizes: List[float] = []

    for pat in patterns:
        # Use pre-existing p_value as fallback
        if (
            pat.raw_group is not None
            and pat.raw_baseline is not None
            and len(pat.raw_group) >= 3
            and len(pat.raw_baseline) >= 3
        ):
            g = pat.raw_group.dropna().values.astype(float)
            b = pat.raw_baseline.dropna().values.astype(float)
            p = _bootstrap_test_mean(g, b, n_resamples=n_resamples, block_length=block_length)
            ef_type = pat.effect_size_type
            ef = compute_effect_size(g, b, ef_type)
        else:
            p  = pat.p_value if pat.p_value is not None else 1.0
            ef = pat.effect_size if pat.effect_size is not None else 0.0

        p_bootstrap.append(float(p))
        effect_sizes.append(float(ef))

    # ── Step 2: BH correction ────────────────────────────────────────────────
    rejected_h0 = benjamini_hochberg(p_bootstrap, alpha=alpha)

    # ── Step 3: apply decisions ──────────────────────────────────────────────
    n_confirmed = 0
    n_rejected  = 0

    for pat, p, ef, rej_h0 in zip(patterns, p_bootstrap, effect_sizes, rejected_h0):
        # Update bootstrap p-value and effect size
        pat.p_value     = p
        pat.effect_size = ef

        # Decision gate: must pass BOTH statistical test AND effect size
        if rej_h0 and ef >= min_effect:
            pat.status     = PatternStatus.CONFIRMED
            pat.confidence = max(0.0, min(1.0, 1.0 - p))
            n_confirmed += 1
        else:
            pat.status     = PatternStatus.REJECTED
            pat.confidence = max(0.0, min(1.0, 1.0 - p))
            n_rejected  += 1

    logger.info(
        "StatisticalFilter: %d confirmed, %d rejected",
        n_confirmed, n_rejected,
    )
    return patterns


# ── Convenience: recompute effect sizes for a pattern list ────────────────────

def enrich_effect_sizes(patterns: List[MinedPattern]) -> List[MinedPattern]:
    """
    Recompute Cohen's d and Cliff's delta for all patterns with raw data,
    and store both in feature_dict for downstream inspection.
    """
    for pat in patterns:
        if pat.raw_group is None or pat.raw_baseline is None:
            continue
        g = pat.raw_group.dropna().values.astype(float)
        b = pat.raw_baseline.dropna().values.astype(float)
        if len(g) < 2 or len(b) < 2:
            continue
        d     = cohens_d(g, b)
        delta = cliffs_delta(g, b)
        pat.feature_dict["cohens_d"]     = round(float(d), 6)
        pat.feature_dict["cliffs_delta"] = round(float(delta), 6)
    return patterns
