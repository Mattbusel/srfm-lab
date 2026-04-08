"""
Optimal transport for comparing financial distributions.

Implements:
  - Wasserstein-1 (Earth Mover's Distance) via LP
  - Wasserstein-2 via Sinkhorn algorithm (regularized)
  - Sliced Wasserstein distance (scalable to high dimensions)
  - Distribution shift detection using OT distance
  - Portfolio rebalancing as OT problem
  - Barycenters of return distributions
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional


# ── 1D Wasserstein ────────────────────────────────────────────────────────────

def wasserstein1_1d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Wasserstein-1 distance between two 1D distributions.
    W1 = integral |F_u(x) - F_v(x)| dx = mean |sorted_u - sorted_v|.
    """
    su = np.sort(u)
    sv = np.sort(v)
    # If different sizes, interpolate
    n = max(len(su), len(sv))
    if len(su) != n:
        su = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(su)), su)
    if len(sv) != n:
        sv = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sv)), sv)
    return float(np.mean(np.abs(su - sv)))


def wasserstein2_1d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Wasserstein-2 distance between two 1D distributions.
    W2^2 = mean (sorted_u - sorted_v)^2.
    """
    su = np.sort(u)
    sv = np.sort(v)
    n = max(len(su), len(sv))
    if len(su) != n:
        su = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(su)), su)
    if len(sv) != n:
        sv = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sv)), sv)
    return float(math.sqrt(np.mean((su - sv) ** 2)))


# ── Sinkhorn algorithm ────────────────────────────────────────────────────────

def sinkhorn_distance(
    x: np.ndarray,
    y: np.ndarray,
    reg: float = 0.01,
    n_iter: int = 100,
    cost: str = "L2",
) -> float:
    """
    Regularized optimal transport via Sinkhorn-Knopp.
    Computes W_p^p between empirical measures on x and y.

    Parameters:
      x, y  : sample arrays shape (n,d) or (n,)
      reg   : entropy regularization (larger = faster, less accurate)
      cost  : 'L1' or 'L2' (default L2 squared)
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y
    n, m = len(x), len(y)

    # Cost matrix
    if cost == "L2":
        C = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    elif cost == "L1":
        C = np.sum(np.abs(x[:, None, :] - y[None, :, :]), axis=-1)
    else:
        raise ValueError(f"Unknown cost: {cost}")

    # Uniform marginals
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Gibbs kernel
    K = np.exp(-C / reg)

    u = np.ones(n)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-10)
        u = a / (K @ v + 1e-10)

    transport = np.diag(u) @ K @ np.diag(v)
    return float(np.sum(transport * C))


# ── Sliced Wasserstein ────────────────────────────────────────────────────────

def sliced_wasserstein(
    x: np.ndarray,
    y: np.ndarray,
    n_projections: int = 100,
    p: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Sliced Wasserstein distance: average 1D W_p over random projections.
    Scalable to high dimensions. Approximates the true W_p.
    """
    rng = rng or np.random.default_rng()
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    y = np.atleast_2d(y).T if y.ndim == 1 else y
    d = x.shape[1]

    # Random unit projections
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    distances = []
    for theta in directions:
        xp = x @ theta
        yp = y @ theta
        if p == 1:
            distances.append(wasserstein1_1d(xp, yp))
        else:
            distances.append(wasserstein2_1d(xp, yp))

    return float(np.mean(distances))


# ── Distribution shift detection ─────────────────────────────────────────────

def distribution_shift_score(
    returns_ref: np.ndarray,
    returns_new: np.ndarray,
    window: int = 60,
    method: str = "wasserstein1",
) -> np.ndarray:
    """
    Rolling distribution shift score between reference and new windows.
    Returns array of OT distances over time.
    """
    n = len(returns_new)
    scores = np.full(n, np.nan)

    for i in range(window, n):
        ref = returns_ref[-window:]
        new = returns_new[i - window: i]
        if method == "wasserstein1":
            scores[i] = wasserstein1_1d(ref, new)
        elif method == "wasserstein2":
            scores[i] = wasserstein2_1d(ref, new)
        elif method == "sinkhorn":
            scores[i] = sinkhorn_distance(ref, new)

    return scores


# ── Wasserstein barycenter ────────────────────────────────────────────────────

def wasserstein_barycenter_1d(
    distributions: list[np.ndarray],
    weights: Optional[np.ndarray] = None,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Wasserstein barycenter of 1D distributions (fixed support).
    Returns the barycenter as a sorted array.
    """
    k = len(distributions)
    if weights is None:
        weights = np.ones(k) / k

    # Interpolate all to same size
    n_out = max(len(d) for d in distributions)
    quantiles = np.linspace(0, 1, n_out)

    # Quantile function interpolation
    Q = np.zeros((k, n_out))
    for i, dist in enumerate(distributions):
        sd = np.sort(dist)
        src_q = np.linspace(0, 1, len(sd))
        Q[i] = np.interp(quantiles, src_q, sd)

    # Barycenter is weighted average of quantile functions
    bary_Q = np.average(Q, axis=0, weights=weights)
    return bary_Q


# ── Portfolio rebalancing as OT ───────────────────────────────────────────────

def portfolio_ot_cost(
    w_from: np.ndarray,
    w_to: np.ndarray,
    prices: np.ndarray,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Compute cost of rebalancing portfolio from w_from to w_to.
    Uses L1 OT (proportional transaction costs).

    Parameters:
      w_from, w_to : weight vectors (sum to 1)
      prices       : current prices (for dollar amount computation)
      transaction_cost : cost per unit of turnover (e.g., 0.001 = 10bps)

    Returns:
      total_cost, turnover, optimal_trades
    """
    delta_w = w_to - w_from
    turnover = float(np.sum(np.abs(delta_w)) / 2)  # one-way

    # OT cost: transport mass from overweight to underweight positions
    buys = np.maximum(delta_w, 0)
    sells = np.minimum(delta_w, 0)

    # Cost matrix: proportional to price ratio (for cross-asset trades)
    trade_cost = transaction_cost * np.sum(np.abs(delta_w))

    return {
        "turnover": turnover,
        "transaction_cost": float(trade_cost),
        "net_buys": float(buys.sum()),
        "net_sells": float(abs(sells.sum())),
        "trades": delta_w,
        "is_balanced": bool(abs(buys.sum() - abs(sells.sum())) < 1e-6),
    }


# ── Return distribution comparison ───────────────────────────────────────────

def compare_return_distributions(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> dict:
    """
    Comprehensive comparison of two return distributions using OT metrics.
    """
    w1 = wasserstein1_1d(strategy_returns, benchmark_returns)
    w2 = wasserstein2_1d(strategy_returns, benchmark_returns)

    # Also compute moments
    return {
        "wasserstein1": w1,
        "wasserstein2": w2,
        "mean_diff": float(strategy_returns.mean() - benchmark_returns.mean()),
        "std_diff": float(strategy_returns.std() - benchmark_returns.std()),
        "skew_diff": float(
            _skew(strategy_returns) - _skew(benchmark_returns)
        ),
        "kurt_diff": float(
            _kurt(strategy_returns) - _kurt(benchmark_returns)
        ),
        "distribution_distance": w2,
        "is_similar": bool(w2 < strategy_returns.std() * 0.1),
    }


def _skew(x: np.ndarray) -> float:
    mu, s = x.mean(), x.std()
    return float(np.mean(((x - mu) / (s + 1e-10)) ** 3)) if s > 0 else 0.0


def _kurt(x: np.ndarray) -> float:
    mu, s = x.mean(), x.std()
    return float(np.mean(((x - mu) / (s + 1e-10)) ** 4)) if s > 0 else 3.0
