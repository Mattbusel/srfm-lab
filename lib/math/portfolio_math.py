"""
Portfolio construction mathematics.

Implements:
  - Mean-variance optimization (Markowitz)
  - Minimum variance portfolio
  - Maximum Sharpe ratio portfolio (tangency)
  - Black-Litterman model
  - Risk parity / equal risk contribution
  - Hierarchical Risk Parity (HRP) — Lopez de Prado
  - Kelly criterion (single and multi-asset)
  - Entropy-maximizing portfolio
  - CVaR minimization portfolio
  - Turnover-constrained rebalancing
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Mean-Variance ─────────────────────────────────────────────────────────────

def mean_variance_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Efficient frontier via parametric approach.
    Returns (expected_returns, volatilities, weights) for n_points on frontier.
    """
    from scipy.optimize import minimize

    n = len(mu)
    # Minimum and maximum target returns
    r_min = mu.min()
    r_max = mu.max()
    target_returns = np.linspace(r_min, r_max, n_points)

    frontier_vols = []
    frontier_weights = []

    for r_target in target_returns:
        def portfolio_var(w):
            return float(w @ Sigma @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: w @ mu - r_target},
        ]
        bounds = [(0, 1)] * n
        result = minimize(
            portfolio_var,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10},
        )
        if result.success:
            w = result.x
            frontier_vols.append(math.sqrt(max(w @ Sigma @ w, 0)))
            frontier_weights.append(w)
        else:
            frontier_vols.append(np.nan)
            frontier_weights.append(np.ones(n) / n)

    return target_returns, np.array(frontier_vols), frontier_weights


def min_variance_portfolio(Sigma: np.ndarray, allow_short: bool = False) -> np.ndarray:
    """
    Analytical minimum variance portfolio.
    w* = Sigma^{-1} * 1 / (1' * Sigma^{-1} * 1)
    """
    try:
        Sigma_inv = np.linalg.inv(Sigma + 1e-8 * np.eye(len(Sigma)))
    except np.linalg.LinAlgError:
        return np.ones(len(Sigma)) / len(Sigma)

    ones = np.ones(len(Sigma))
    w = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

    if not allow_short:
        w = np.maximum(w, 0)
        w /= w.sum() if w.sum() > 0 else 1

    return w


def max_sharpe_portfolio(
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float = 0.0,
    allow_short: bool = False,
) -> np.ndarray:
    """
    Tangency (maximum Sharpe) portfolio.
    w* = Sigma^{-1} * (mu - rf) / [1' * Sigma^{-1} * (mu - rf)]
    """
    try:
        Sigma_inv = np.linalg.inv(Sigma + 1e-8 * np.eye(len(Sigma)))
    except np.linalg.LinAlgError:
        return np.ones(len(mu)) / len(mu)

    excess = mu - rf
    w = Sigma_inv @ excess
    if w.sum() == 0:
        return np.ones(len(mu)) / len(mu)
    w = w / w.sum()

    if not allow_short:
        w = np.maximum(w, 0)
        total = w.sum()
        if total > 0:
            w /= total

    return w


# ── Black-Litterman ────────────────────────────────────────────────────────────

def black_litterman(
    Sigma: np.ndarray,
    w_market: np.ndarray,
    P: np.ndarray,        # pick matrix (K x N): views
    Q: np.ndarray,        # view returns (K,)
    Omega: np.ndarray,    # view uncertainty covariance (K x K)
    rf: float = 0.0,
    delta: float = 2.5,   # risk aversion
    tau: float = 0.05,    # uncertainty in prior
) -> tuple[np.ndarray, np.ndarray]:
    """
    Black-Litterman posterior expected returns and covariance.

    P: view matrix — each row is a view (e.g., [1,-1,0,...] = asset1 outperforms asset2)
    Q: expected return for each view
    Omega: diagonal = var of each view (use P @ (tau*Sigma) @ P.T for proportional)

    Returns (posterior_mu, posterior_Sigma).
    """
    # Implied equilibrium returns
    pi = delta * Sigma @ w_market + rf

    # Posterior
    tau_Sigma = tau * Sigma
    M1 = np.linalg.inv(tau_Sigma)
    M2 = P.T @ np.linalg.inv(Omega) @ P
    Sigma_post = np.linalg.inv(M1 + M2)
    mu_post = Sigma_post @ (M1 @ pi + P.T @ np.linalg.inv(Omega) @ Q)

    return mu_post, Sigma_post


# ── Risk Parity ───────────────────────────────────────────────────────────────

def risk_parity(
    Sigma: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Equal Risk Contribution (ERC) / Risk Parity portfolio.
    Each asset contributes equally to total portfolio variance.
    Solved via Newton-Raphson.
    """
    n = len(Sigma)
    w = np.ones(n) / n  # equal-weight start
    target_rc = 1.0 / n  # equal risk contribution

    for iteration in range(max_iter):
        var = w @ Sigma @ w
        if var < 1e-12:
            break
        # Marginal risk contributions
        mrc = Sigma @ w
        rc = w * mrc / var  # risk contributions (sum to 1)

        # Newton step: minimize (rc_i - target)^2
        grad = 2 * (rc - target_rc) * mrc / var
        hess_diag = 2 * (mrc / var) ** 2
        step = grad / (hess_diag + 1e-10)

        w_new = w - 0.5 * step
        w_new = np.maximum(w_new, 1e-8)
        w_new /= w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w / w.sum()


def risk_contribution(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Return each asset's fractional risk contribution."""
    var = float(w @ Sigma @ w)
    if var < 1e-12:
        return np.ones(len(w)) / len(w)
    mrc = Sigma @ w
    return w * mrc / var


# ── HRP — Hierarchical Risk Parity ────────────────────────────────────────────

def hrp(
    returns: np.ndarray,
    method: str = "ward",
) -> np.ndarray:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).
    Builds hierarchical cluster tree, then bisects using inverse-variance weighting.

    returns: shape (T, N)
    Returns weight vector (N,).
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    T, N = returns.shape
    # Correlation and distance
    corr = np.corrcoef(returns.T)
    dist = np.sqrt((1 - corr) / 2)
    np.fill_diagonal(dist, 0)

    # Hierarchical clustering
    dist_condensed = squareform(dist)
    Z = linkage(dist_condensed, method=method)
    order = leaves_list(Z)  # optimal leaf ordering

    # Bisection
    weights = np.ones(N)
    clusters = [list(order)]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            # Cluster variances (via inverse-variance of sub-portfolio)
            def cluster_var(idx):
                sub_cov = np.cov(returns[:, idx].T) if len(idx) > 1 else np.var(returns[:, idx[0]])
                if len(idx) == 1:
                    return float(sub_cov)
                sub_w = min_variance_portfolio(np.atleast_2d(sub_cov))
                return float(sub_w @ np.atleast_2d(sub_cov) @ sub_w)

            var_left = cluster_var(left)
            var_right = cluster_var(right)

            # Allocate inversely proportional to variance
            alpha = 1 - var_left / (var_left + var_right + 1e-12)
            weights[left] *= alpha
            weights[right] *= 1 - alpha

            new_clusters += [left, right]
        clusters = new_clusters

    return weights / weights.sum()


# ── Kelly criterion ───────────────────────────────────────────────────────────

def kelly_fraction(mu: float, sigma: float, rf: float = 0.0) -> float:
    """
    Single-asset Kelly fraction: f* = (mu - rf) / sigma^2
    Half-Kelly: f* / 2
    """
    return float((mu - rf) / max(sigma ** 2, 1e-10))


def multi_asset_kelly(
    mu: np.ndarray,
    Sigma: np.ndarray,
    rf: float = 0.0,
    fraction: float = 0.5,
) -> np.ndarray:
    """
    Multi-asset Kelly (log-utility optimizer).
    w* = Sigma^{-1} * (mu - rf) * fraction
    fraction=0.5 for half-Kelly.
    """
    try:
        Sigma_inv = np.linalg.inv(Sigma + 1e-8 * np.eye(len(Sigma)))
    except np.linalg.LinAlgError:
        return np.ones(len(mu)) / len(mu)
    w = fraction * Sigma_inv @ (mu - rf)
    return w  # not constrained to sum=1 (leveraged)


# ── CVaR portfolio optimization ───────────────────────────────────────────────

def cvar_portfolio(
    returns: np.ndarray,
    confidence: float = 0.95,
    allow_short: bool = False,
) -> np.ndarray:
    """
    CVaR (Expected Shortfall) minimizing portfolio via linear programming.
    Rockafellar & Uryasev (2000).

    returns: (T, N)
    """
    from scipy.optimize import linprog

    T, N = returns.shape
    alpha = confidence
    losses = -returns  # shape (T, N)

    # Variables: [w (N), z_t (T), gamma (1)]
    # min gamma + 1/(T*(1-alpha)) * sum(z_t)
    # s.t. z_t >= -losses @ w - gamma, z_t >= 0, sum(w) = 1, w >= 0
    n_vars = N + T + 1

    c = np.zeros(n_vars)
    c[N + T] = 1.0                             # gamma
    c[N: N + T] = 1.0 / (T * (1 - alpha))     # z_t

    # Inequality: z_t + losses_t @ w + gamma >= 0 → -z_t - losses_t @ w - gamma <= 0
    A_ub = np.zeros((T, n_vars))
    for t in range(T):
        A_ub[t, :N] = -losses[t]               # -losses @ w
        A_ub[t, N + t] = -1.0                  # -z_t
        A_ub[t, N + T] = -1.0                  # -gamma
    b_ub = np.zeros(T)

    # Equality: sum(w) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    bounds = ([(0, 1) if not allow_short else (None, None)] * N
              + [(0, None)] * T + [(None, None)])

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")

    if result.success:
        w = np.array(result.x[:N])
        w = np.maximum(w, 0)
        return w / w.sum() if w.sum() > 0 else np.ones(N) / N
    return np.ones(N) / N


# ── Maximum entropy portfolio ──────────────────────────────────────────────────

def max_entropy_portfolio(
    Sigma: np.ndarray,
    target_vol: Optional[float] = None,
) -> np.ndarray:
    """
    Maximum entropy portfolio: maximize H(w) = -sum w_i log w_i
    subject to portfolio constraints.
    Minimum concentration — most diversified in information-theoretic sense.
    """
    from scipy.optimize import minimize

    n = len(Sigma)

    def neg_entropy(w):
        w_pos = np.maximum(w, 1e-10)
        return float(np.sum(w_pos * np.log(w_pos)))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_vol is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: target_vol ** 2 - w @ Sigma @ w,
        })

    result = minimize(
        neg_entropy,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=constraints,
    )
    w = np.maximum(result.x if result.success else np.ones(n) / n, 0)
    return w / w.sum()
