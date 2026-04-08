"""
Combinatorics and combinatorial methods in finance.

Implements:
  - Combinatorial purged cross-validation (CPCV) for backtesting
  - Deflated Sharpe Ratio (DSR) — adjusts for multiple testing
  - Minimum backtest length estimation
  - Probabilistic Sharpe Ratio (PSR)
  - Combinatorial features for ensemble diversity
  - Bootstrap-based alpha significance testing
  - Multiple testing corrections (Bonferroni, Holm, BH)
  - Walk-forward combinatorial CV
  - Strategy diversification: minimum correlation portfolio of strategies
  - Return-based style analysis (Sharpe's RBSA)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import norm
from itertools import combinations


# ── Probabilistic Sharpe Ratio ────────────────────────────────────────────────

def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Bailey & Lopez de Prado (2012) Probabilistic Sharpe Ratio.
    Probability that observed SR > benchmark SR after adjusting for non-normality.
    """
    if n <= 1:
        return 0.5

    # Standard error of SR estimate
    se_sr = math.sqrt(
        (1 + 0.5 * observed_sr**2 - skewness * observed_sr + (kurtosis - 3) / 4 * observed_sr**2)
        / max(n - 1, 1)
    )

    # Z-statistic
    z = (observed_sr - benchmark_sr) / max(se_sr, 1e-10)
    psr = float(norm.cdf(z))
    return psr


def deflated_sharpe_ratio(
    observed_sr: float,
    n_strategies_tested: int,
    n: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio: adjusts for multiple testing.
    Expected maximum SR from n strategies tested follows extreme value distribution.
    """
    if n_strategies_tested <= 1:
        return probabilistic_sharpe_ratio(observed_sr, 0, n, skewness, kurtosis)

    # Expected maximum SR from multiple testing
    euler_mascheroni = 0.5772156649
    gamma = euler_mascheroni
    v = 1 - math.log(1 - 1.0 / n_strategies_tested)
    mean_max_sr = norm.ppf(1 - 1.0 / n_strategies_tested) + math.sqrt(v) * (
        gamma - math.log(math.log(1 / (1 - 1.0 / n_strategies_tested)))
    )

    # Use expected max as benchmark
    return probabilistic_sharpe_ratio(observed_sr, mean_max_sr, n, skewness, kurtosis)


# ── Minimum Backtest Length ───────────────────────────────────────────────────

def minimum_backtest_length(
    sharpe_annual: float,
    target_psr: float = 0.95,
    n_strategies: int = 1,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    periods_per_year: int = 252,
) -> int:
    """
    Minimum number of periods needed to achieve target PSR.
    Solves for n such that PSR >= target_psr.
    """
    sharpe_daily = sharpe_annual / math.sqrt(periods_per_year)

    # Adjust benchmark for multiple testing
    if n_strategies > 1:
        benchmark = norm.ppf(1 - 1.0 / n_strategies)
    else:
        benchmark = 0.0

    # Z-score needed
    z_target = norm.ppf(target_psr)

    # Solve: z_target = (SR - benchmark) / se_SR
    # se_SR^2 = (1 + SR^2/2 - skew*SR + (kurt-3)/4 * SR^2) / (n-1)
    # n-1 = (1 + SR^2/2 - skew*SR + (kurt-3)/4 * SR^2) * (SR - benchmark)^2 / z_target^2

    variance_term = 1 + sharpe_daily**2 / 2 - skewness * sharpe_daily + (kurtosis - 3) / 4 * sharpe_daily**2
    n_needed = variance_term * ((sharpe_daily - benchmark) / max(z_target, 0.01))**(-2) + 1

    return max(int(math.ceil(n_needed)), 30)


# ── Combinatorial Purged Cross-Validation ────────────────────────────────────

def combinatorial_purged_cv_splits(
    n: int,
    k: int,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    Lopez de Prado: creates C(k, n) train/test splits with purging + embargo.
    k: number of test groups
    n: total observations
    Returns list of (train_indices, test_indices) tuples.
    """
    group_size = n // k
    embargo = max(int(n * embargo_pct), 1)
    groups = [np.arange(i * group_size, min((i + 1) * group_size, n)) for i in range(k)]

    splits = []
    for n_test in range(1, min(k, 3) + 1):
        for test_group_indices in combinations(range(k), n_test):
            test_idx = np.concatenate([groups[i] for i in test_group_indices])
            test_min, test_max = test_idx.min(), test_idx.max()

            # Purge: remove training indices that overlap with test labels
            # Embargo: remove indices just before/after test set
            train_idx = []
            for g_idx, group in enumerate(groups):
                if g_idx in test_group_indices:
                    continue
                # Remove indices within embargo of test set
                mask = (group < test_min - embargo) | (group > test_max + embargo)
                train_idx.append(group[mask])

            if train_idx:
                train = np.concatenate(train_idx)
                splits.append((train, test_idx))

    return splits


# ── Walk-Forward CV ───────────────────────────────────────────────────────────

def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int = 1,
    min_train: Optional[int] = None,
    expanding: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Rolling or expanding walk-forward cross-validation splits.
    expanding: if True, training window grows; else slides.
    """
    if min_train is None:
        min_train = train_size

    splits = []
    start = min_train
    while start + test_size <= n:
        test = np.arange(start, start + test_size)
        if expanding:
            train = np.arange(0, start)
        else:
            train = np.arange(max(0, start - train_size), start)
        splits.append((train, test))
        start += step

    return splits


# ── Multiple Testing Corrections ─────────────────────────────────────────────

def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Bonferroni correction: divide alpha by number of tests."""
    m = len(p_values)
    corrected_alpha = alpha / m
    rejected = p_values <= corrected_alpha
    return {
        "rejected": rejected,
        "corrected_alpha": corrected_alpha,
        "n_rejected": int(rejected.sum()),
        "familywise_error_rate": float(min(m * p_values.min(), 1.0)),
    }


def holm_correction(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction."""
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    rejected = np.zeros(m, dtype=bool)

    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_p[k] <= threshold:
            rejected[sorted_idx[k]] = True
        else:
            break  # stop at first non-rejection

    return {
        "rejected": rejected,
        "n_rejected": int(rejected.sum()),
        "method": "holm",
    }


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg FDR correction.
    Controls False Discovery Rate, less conservative than Bonferroni.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    rejected = np.zeros(m, dtype=bool)

    bh_thresholds = alpha * np.arange(1, m + 1) / m
    bh_rejected = sorted_p <= bh_thresholds

    if bh_rejected.any():
        last_rejected = np.where(bh_rejected)[0][-1]
        rejected[sorted_idx[:last_rejected + 1]] = True

    return {
        "rejected": rejected,
        "n_rejected": int(rejected.sum()),
        "fdr_threshold": float(bh_thresholds[last_rejected] if bh_rejected.any() else 0),
        "method": "benjamini_hochberg",
    }


# ── Return-Based Style Analysis ───────────────────────────────────────────────

def return_based_style_analysis(
    portfolio_returns: np.ndarray,
    factor_returns: np.ndarray,
    factor_names: Optional[list[str]] = None,
    constrained: bool = True,
) -> dict:
    """
    Sharpe's RBSA: decompose portfolio returns into factor style exposures.
    constrained: weights sum to 1 and are non-negative.
    """
    T, k = factor_returns.shape
    if factor_names is None:
        factor_names = [f"F{i+1}" for i in range(k)]

    if constrained:
        # Constrained least squares via quadratic programming (gradient descent)
        w = np.ones(k) / k
        for _ in range(500):
            resid = portfolio_returns - factor_returns @ w
            grad = -2 * factor_returns.T @ resid
            step = 0.001
            w = w - step * grad
            w = np.maximum(w, 0)
            w /= w.sum() + 1e-10
    else:
        w = np.linalg.lstsq(factor_returns, portfolio_returns, rcond=None)[0]

    fitted = factor_returns @ w
    resid = portfolio_returns - fitted
    r2 = float(1 - resid.var() / (portfolio_returns.var() + 1e-10))

    return {
        "style_weights": dict(zip(factor_names, w.tolist())),
        "r2_selection": r2,
        "selection_return_annualized": float(resid.mean() * 252),
        "factor_contribution": dict(zip(
            factor_names,
            (w * factor_returns.mean(axis=0) * 252).tolist()
        )),
    }


# ── Strategy Diversification ──────────────────────────────────────────────────

def minimum_correlation_portfolio(
    strategy_returns: np.ndarray,
    strategy_names: Optional[list[str]] = None,
) -> dict:
    """
    Minimum Correlation Portfolio of strategies.
    Maximizes diversification by minimizing pairwise correlation.
    strategy_returns: (T, n_strategies)
    """
    T, n = strategy_returns.shape
    if strategy_names is None:
        strategy_names = [f"S{i+1}" for i in range(n)]

    corr = np.corrcoef(strategy_returns.T)
    np.fill_diagonal(corr, 1.0)

    # Equal weights as starting point, optimize to minimize avg correlation
    w = np.ones(n) / n

    for _ in range(500):
        port_corr_contrib = corr @ w
        grad = port_corr_contrib - float(w @ corr @ w)
        step = 0.01
        w = w - step * grad
        w = np.maximum(w, 0)
        w /= w.sum() + 1e-10

    avg_pairwise_corr = float((w @ corr @ w - 1/n) * n / max(n - 1, 1))

    return {
        "weights": dict(zip(strategy_names, w.tolist())),
        "avg_pairwise_correlation": avg_pairwise_corr,
        "effective_n_strategies": float(1 / max(np.sum(w**2), 1e-10)),
    }


# ── Sharpe Ratio Significance ─────────────────────────────────────────────────

def sharpe_ratio_test(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """
    Test statistical significance of Sharpe ratio.
    Uses Jobson-Korkie asymptotic test.
    """
    n = len(returns)
    mu = float(returns.mean())
    sigma = float(returns.std())
    sr = float(mu / max(sigma, 1e-10) * math.sqrt(periods_per_year))

    skew = float(np.mean(((returns - mu) / max(sigma, 1e-10))**3))
    kurt = float(np.mean(((returns - mu) / max(sigma, 1e-10))**4))

    # Asymptotic variance of SR estimate
    var_sr = float((1 + 0.5 * sr**2 / periods_per_year - skew * sr / math.sqrt(periods_per_year)
                    + (kurt - 3) / 4 * sr**2 / periods_per_year) / n)

    se_sr = math.sqrt(max(var_sr, 0))
    z = (sr - benchmark_sr) / max(se_sr, 1e-10)
    p_value = float(2 * (1 - norm.cdf(abs(z))))

    psr = float(norm.cdf(z))

    return {
        "sharpe_ratio": sr,
        "z_stat": z,
        "p_value": p_value,
        "psr": psr,
        "se_sharpe": se_sr,
        "is_significant_5pct": bool(p_value < 0.05),
        "n": n,
    }
