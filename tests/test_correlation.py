"""
test_correlation.py — Tests for BH activation correlation analysis.

~400 LOC. Tests Jaccard similarity, correlation matrix, and portfolio diversification.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "lib"))


# ─────────────────────────────────────────────────────────────────────────────
# BH activation vector builder
# ─────────────────────────────────────────────────────────────────────────────

def _compute_bh_activation_series(
    closes: np.ndarray,
    cf: float = 0.001,
    bh_form: float = 1.2,
    bh_collapse: float = 0.8,
    bh_decay: float = 0.95,
) -> np.ndarray:
    """Return a boolean array: True where BH is active."""
    from srfm_core import MinkowskiClassifier, BlackHoleDetector
    mc  = MinkowskiClassifier(cf=cf)
    bh  = BlackHoleDetector(bh_form, bh_collapse, bh_decay)
    n   = len(closes)
    active = np.zeros(n, dtype=bool)
    mc.update(float(closes[0]))
    for i in range(1, n):
        bit = mc.update(float(closes[i]))
        act = bh.update(bit, float(closes[i]), float(closes[i-1]))
        active[i] = act
    return active


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two boolean activation vectors."""
    intersection = np.sum(a & b)
    union        = np.sum(a | b)
    return float(intersection / union) if union > 0 else 0.0


def compute_activation_correlation_matrix(
    activation_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity matrix from activation boolean arrays.
    Arrays are aligned to minimum length.
    """
    syms = list(activation_dict.keys())
    n    = len(syms)
    mat  = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = activation_dict[syms[i]]
            b = activation_dict[syms[j]]
            min_len = min(len(a), len(b))
            jac = jaccard_similarity(a[:min_len], b[:min_len])
            mat[i, j] = jac
            mat[j, i] = jac
    return pd.DataFrame(mat, index=syms, columns=syms)


def diversification_score(corr_matrix: pd.DataFrame) -> float:
    """
    Diversification score: 1 - average off-diagonal correlation.
    Range [0, 1]: 1 = fully uncorrelated, 0 = fully correlated.
    """
    n = len(corr_matrix)
    if n < 2:
        return 0.0
    off_diag = corr_matrix.values[~np.eye(n, dtype=bool)]
    return float(1.0 - np.mean(off_diag))


def optimal_portfolio_unique(
    corr_matrix: pd.DataFrame,
    n_select: int = 3,
) -> List[str]:
    """
    Select the n_select instruments with lowest average pairwise correlation.
    Uses greedy approach: start with least-correlated pair, then add lowest-avg-corr.
    """
    syms = list(corr_matrix.columns)
    if len(syms) <= n_select:
        return syms
    avg_corr = corr_matrix.values.copy()
    np.fill_diagonal(avg_corr, 0.0)
    row_avg = avg_corr.mean(axis=1)
    # Greedy: pick the two least correlated first
    off = corr_matrix.values.copy()
    np.fill_diagonal(off, np.inf)
    idx_min = int(np.unravel_index(np.argmin(off), off.shape)[0])
    idx_min2 = int(np.unravel_index(np.argmin(off), off.shape)[1])
    selected = list({syms[idx_min], syms[idx_min2]})
    remaining = [s for s in syms if s not in selected]
    while len(selected) < n_select and remaining:
        # Add instrument with lowest avg correlation to already selected
        best = min(remaining, key=lambda s: np.mean([corr_matrix.loc[s, sel] for sel in selected]))
        selected.append(best)
        remaining.remove(best)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Data generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_closes(n: int, drift: float, sigma: float, seed: int,
                 start: float = 4500.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    closes = np.empty(n)
    closes[0] = start
    for i in range(1, n):
        closes[i] = closes[i-1] * max(1e-3, 1.0 + drift + sigma * rng.standard_normal())
    return closes


def _make_correlated_closes(n: int, rho: float = 0.9,
                             sigma: float = 0.001, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Two price series with correlation rho."""
    rng = np.random.default_rng(seed)
    base = 4500.0
    common  = rng.standard_normal(n)
    idio1   = rng.standard_normal(n)
    idio2   = rng.standard_normal(n)
    ret1 = sigma * (rho * common + math.sqrt(1 - rho**2) * idio1)
    ret2 = sigma * (rho * common + math.sqrt(1 - rho**2) * idio2)
    c1, c2 = np.empty(n), np.empty(n)
    c1[0] = c2[0] = base
    for i in range(1, n):
        c1[i] = c1[i-1] * max(1e-3, 1.0 + ret1[i])
        c2[i] = c2[i-1] * max(1e-3, 1.0 + ret2[i])
    return c1, c2


# ─────────────────────────────────────────────────────────────────────────────
# Class TestBHCorrelation
# ─────────────────────────────────────────────────────────────────────────────

class TestBHCorrelation:

    def test_self_correlation_is_one(self):
        """Jaccard similarity of a series with itself should be 1.0."""
        closes = _make_closes(500, 0.0001, 0.0008, seed=1)
        act = _compute_bh_activation_series(closes)
        assert jaccard_similarity(act, act) == pytest.approx(1.0)

    def test_self_correlation_matrix_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1.0."""
        syms = ["ES", "NQ"]
        activations = {}
        for i, sym in enumerate(syms):
            closes = _make_closes(500, 0.0001, 0.0008, seed=i * 10)
            activations[sym] = _compute_bh_activation_series(closes)
        mat = compute_activation_correlation_matrix(activations)
        for sym in syms:
            assert mat.loc[sym, sym] == pytest.approx(1.0)

    def test_uncorrelated_instruments_low_jaccard(self):
        """Two statistically independent price series should have low Jaccard similarity."""
        c1 = _make_closes(500, 0.0001, 0.0008, seed=1)
        c2 = _make_closes(500, 0.0001, 0.0008, seed=9999)  # very different seed
        act1 = _compute_bh_activation_series(c1)
        act2 = _compute_bh_activation_series(c2)
        jac = jaccard_similarity(act1, act2)
        # Independent series should have lower Jaccard than 0.9
        assert jac < 0.9, f"Independent series Jaccard={jac:.3f} should be < 0.9"

    def test_correlated_instruments_higher_jaccard(self):
        """Highly correlated price series should have higher Jaccard than uncorrelated."""
        n = 800
        c_corr1, c_corr2 = _make_correlated_closes(n, rho=0.95, sigma=0.001, seed=42)
        c_indep1 = _make_closes(n, 0.0001, 0.001, seed=1)
        c_indep2 = _make_closes(n, 0.0001, 0.001, seed=9999)

        act_corr1 = _compute_bh_activation_series(c_corr1)
        act_corr2 = _compute_bh_activation_series(c_corr2)
        act_ind1  = _compute_bh_activation_series(c_indep1)
        act_ind2  = _compute_bh_activation_series(c_indep2)

        jac_corr  = jaccard_similarity(act_corr1, act_corr2)
        jac_indep = jaccard_similarity(act_ind1, act_ind2)
        assert jac_corr >= jac_indep, (
            f"Correlated Jaccard {jac_corr:.3f} should be >= independent {jac_indep:.3f}")

    def test_optimal_portfolio_unique_instruments(self):
        """Optimal portfolio selection should return unique instruments."""
        n_syms = 6
        activations = {}
        for i in range(n_syms):
            closes = _make_closes(500, 0.0001, 0.001, seed=i * 7)
            activations[f"SYM{i}"] = _compute_bh_activation_series(closes)
        mat = compute_activation_correlation_matrix(activations)
        selected = optimal_portfolio_unique(mat, n_select=3)
        assert len(selected) == 3
        assert len(set(selected)) == 3, "Selected instruments should be unique"
        for sym in selected:
            assert sym in activations, f"Selected {sym} not in activation dict"

    def test_diversification_score_between_0_and_1(self):
        """Diversification score should always be in [0, 1]."""
        n_syms = 5
        activations = {}
        for i in range(n_syms):
            closes = _make_closes(500, 0.0001, 0.001, seed=i * 13)
            activations[f"SYM{i}"] = _compute_bh_activation_series(closes)
        mat = compute_activation_correlation_matrix(activations)
        score = diversification_score(mat)
        assert 0.0 <= score <= 1.0, f"Diversification score {score} out of [0,1]"

    def test_fully_correlated_low_diversification(self):
        """Identical series → all Jaccard = 1.0 → diversification = 0."""
        closes = _make_closes(500, 0.0001, 0.0008, seed=42)
        act = _compute_bh_activation_series(closes)
        activations = {"A": act, "B": act.copy(), "C": act.copy()}
        mat = compute_activation_correlation_matrix(activations)
        score = diversification_score(mat)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_matrix_symmetric(self):
        """Correlation matrix should be symmetric: mat[i,j] == mat[j,i]."""
        syms = ["ES", "NQ", "BTC"]
        activations = {}
        for i, sym in enumerate(syms):
            closes = _make_closes(500, 0.0001, 0.001, seed=i * 5)
            activations[sym] = _compute_bh_activation_series(closes)
        mat = compute_activation_correlation_matrix(activations)
        for s1 in syms:
            for s2 in syms:
                assert mat.loc[s1, s2] == pytest.approx(mat.loc[s2, s1], abs=1e-10)

    def test_jaccard_bounded_0_to_1(self):
        """Jaccard similarity is always in [0, 1]."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = rng.random(100) > 0.5
            b = rng.random(100) > 0.5
            jac = jaccard_similarity(a, b)
            assert 0.0 <= jac <= 1.0, f"Jaccard {jac} out of [0,1]"

    def test_jaccard_empty_both_false(self):
        """If both arrays are all False (no activations), union=0 → returns 0."""
        a = np.zeros(100, dtype=bool)
        b = np.zeros(100, dtype=bool)
        jac = jaccard_similarity(a, b)
        assert jac == 0.0

    def test_no_activation_in_flat_market(self):
        """Flat price → no BH activations → activation series all False."""
        closes = np.full(500, 4500.0)
        act = _compute_bh_activation_series(closes)
        assert not np.any(act), "Flat market should produce no BH activations"

    def test_correlation_matrix_shape(self):
        """Matrix should be n×n for n instruments."""
        n_syms = 4
        activations = {}
        for i in range(n_syms):
            closes = _make_closes(300, 0.0001, 0.001, seed=i)
            activations[f"SYM{i}"] = _compute_bh_activation_series(closes)
        mat = compute_activation_correlation_matrix(activations)
        assert mat.shape == (n_syms, n_syms)

    def test_optimal_portfolio_n_select_respected(self):
        """optimal_portfolio_unique should return exactly n_select symbols."""
        n_syms = 8
        activations = {}
        for i in range(n_syms):
            closes = _make_closes(400, 0.0001, 0.001, seed=i * 11)
            activations[f"S{i}"] = _compute_bh_activation_series(closes)
        mat = compute_activation_correlation_matrix(activations)
        for n_sel in (2, 3, 4, 5):
            selected = optimal_portfolio_unique(mat, n_select=n_sel)
            assert len(selected) == n_sel

    def test_diversification_single_asset(self):
        """Single-asset portfolio: diversification_score returns 0."""
        closes = _make_closes(300, 0.0001, 0.001, seed=1)
        act = _compute_bh_activation_series(closes)
        mat = compute_activation_correlation_matrix({"A": act})
        score = diversification_score(mat)
        assert score == 0.0
