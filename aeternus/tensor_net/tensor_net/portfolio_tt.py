"""
portfolio_tt.py — Tensor-Train based portfolio optimisation for Project AETERNUS.

This module implements portfolio construction, risk estimation, and position sizing
using tensor decompositions.  All compute-intensive routines are JAX-compatible and
JIT-compilable where possible.

Key classes
-----------
* TTCovarianceEstimator   — low-rank TT covariance estimation from streaming returns
* TTMeanVarianceOptimiser — classical MV optimisation with TT-compressed covariance
* TTRiskParityPortfolio   — equal risk contribution via TT factor model
* TTFactorModel           — Fama-French style factor model with TT weight matrices
* TTBlackLitterman        — Black-Litterman allocation with TT prior covariance
* PortfolioBacktester     — vectorised back-test engine
* TransactionCostModel    — linear + quadratic + market-impact cost models
* PositionSizer           — Kelly / fixed-fraction / volatility-target sizers
* RiskBudgetAllocator     — risk-budget optimisation via gradient descent
* CorrelationShockModel   — stress-testing via tensor-based correlation shocks
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Optional deps
try:
    import jax.scipy as jsp
except ImportError:
    jsp = None  # type: ignore

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _safe_inv(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Element-wise safe reciprocal."""
    return jnp.where(jnp.abs(x) < eps, 0.0, 1.0 / (x + eps * jnp.sign(x + eps)))


def _chol_solve(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve A x = b where A is SPD, using Cholesky (JAX-compatible)."""
    L = jnp.linalg.cholesky(A + 1e-6 * jnp.eye(A.shape[0]))
    y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    return jax.scipy.linalg.solve_triangular(L.T, y, lower=False)


def _portfolio_vol(weights: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Annualised portfolio volatility (assumes daily returns, 252 trading days)."""
    return jnp.sqrt(252.0 * weights @ cov @ weights)


def _marginal_risk(weights: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Marginal risk contributions per asset."""
    port_var = weights @ cov @ weights
    return (cov @ weights) / jnp.sqrt(port_var + 1e-12)


def _risk_contributions(weights: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
    """Percentage risk contribution per asset."""
    rc = weights * _marginal_risk(weights, cov)
    return rc / (rc.sum() + 1e-12)


def _simplex_project(v: np.ndarray) -> np.ndarray:
    """Project vector *v* onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u > cssv / np.arange(1, n + 1))[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# TTCovarianceEstimator
# ---------------------------------------------------------------------------


@dataclass
class CovEstimatorConfig:
    """Configuration for TTCovarianceEstimator."""
    n_assets: int = 64
    tt_rank: int = 8
    half_life_days: float = 60.0      # exponential decay half-life
    shrinkage_intensity: float = 0.1  # Ledoit-Wolf-style shrinkage
    min_observations: int = 30
    regularisation_eps: float = 1e-5
    dtype: str = "float32"


class TTCovarianceEstimator:
    """
    Streaming, exponentially-weighted covariance estimator that stores the
    running outer-product accumulator in a low-rank TT-compressed form.

    The estimator maintains:

    * ``_S`` : (N, N) numpy array — current exponentially-weighted scatter matrix
    * ``_n`` : int — effective number of observations (decay-adjusted)
    * ``_mu`` : (N,) — exponentially-weighted mean

    Shrinkage is applied on retrieval via :meth:`covariance`.

    Parameters
    ----------
    config : CovEstimatorConfig
        Hyperparameters.
    """

    def __init__(self, config: CovEstimatorConfig | None = None) -> None:
        self.config = config or CovEstimatorConfig()
        N = self.config.n_assets
        self._S = np.zeros((N, N), dtype=np.float64)
        self._mu = np.zeros(N, dtype=np.float64)
        self._n: float = 0.0
        self._decay = math.pow(0.5, 1.0 / self.config.half_life_days)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, returns: np.ndarray) -> None:
        """
        Incorporate a new observation (or batch of observations).

        Parameters
        ----------
        returns : np.ndarray, shape (N,) or (T, N)
            Log-return vector(s).
        """
        r = np.atleast_2d(returns).astype(np.float64)
        lam = self._decay
        for t in range(r.shape[0]):
            x = r[t]
            self._S = lam * self._S + (1 - lam) * np.outer(x, x)
            self._mu = lam * self._mu + (1 - lam) * x
            self._n = lam * self._n + 1.0

    def reset(self) -> None:
        """Reset accumulator to zero state."""
        N = self.config.n_assets
        self._S = np.zeros((N, N), dtype=np.float64)
        self._mu = np.zeros(N, dtype=np.float64)
        self._n = 0.0

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def covariance(self) -> np.ndarray:
        """
        Return the current covariance matrix estimate (N, N).

        Applies bias correction, de-biases the mean-outer-product, and
        applies Ledoit-Wolf shrinkage toward a scaled identity.
        """
        if self._n < self.config.min_observations:
            warnings.warn(
                f"TTCovarianceEstimator: only {self._n:.1f} effective observations; "
                "returning scaled identity."
            )
            return np.eye(self.config.n_assets, dtype=np.float32)

        # Bias-corrected scatter
        S = self._S - np.outer(self._mu, self._mu)
        N = self.config.n_assets

        # Ledoit-Wolf shrinkage target: scaled identity
        mu_target = np.trace(S) / N
        alpha = self.config.shrinkage_intensity
        shrunk = (1 - alpha) * S + alpha * mu_target * np.eye(N)

        # Regularisation
        shrunk += self.config.regularisation_eps * np.eye(N)
        return shrunk.astype(np.float32)

    def correlation(self) -> np.ndarray:
        """Return normalised correlation matrix (N, N)."""
        cov = self.covariance()
        std = np.sqrt(np.diag(cov))
        std_outer = np.outer(std, std)
        return cov / (std_outer + 1e-12)

    def effective_observations(self) -> float:
        """Return the decay-adjusted effective number of observations."""
        return self._n

    def mean_returns(self) -> np.ndarray:
        """Return the exponentially-weighted mean return vector."""
        return self._mu.astype(np.float32)

    def annualised_volatilities(self) -> np.ndarray:
        """Return annualised asset volatilities (sqrt(252 * diag(cov)))."""
        cov = self.covariance()
        return np.sqrt(252.0 * np.diag(cov)).astype(np.float32)


# ---------------------------------------------------------------------------
# TTMeanVarianceOptimiser
# ---------------------------------------------------------------------------


@dataclass
class MVOptConfig:
    """Configuration for TTMeanVarianceOptimiser."""
    risk_aversion: float = 2.0       # lambda in mean-variance trade-off
    allow_short: bool = False        # if False, weights >= 0 (long-only)
    max_weight: float = 0.2         # max single-asset weight
    min_weight: float = 0.0         # min single-asset weight
    target_return: float | None = None
    n_iters: int = 500              # gradient descent iterations
    lr: float = 5e-3
    l2_reg: float = 1e-4


class TTMeanVarianceOptimiser:
    """
    Mean-variance optimiser that operates on the covariance output of
    :class:`TTCovarianceEstimator`.

    Solves::

        max_w  mu^T w  -  (lambda/2) w^T Sigma w
        s.t.   sum(w) = 1,  w_i in [min_weight, max_weight]

    via projected gradient descent using optax (Adam).

    Parameters
    ----------
    config : MVOptConfig
        Hyperparameters.
    n_assets : int
        Number of assets.
    """

    def __init__(self, config: MVOptConfig | None = None, n_assets: int = 64) -> None:
        self.config = config or MVOptConfig()
        self.n_assets = n_assets

    def optimise(
        self, mu: np.ndarray, cov: np.ndarray, initial_weights: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Run the optimisation and return portfolio weights (N,).

        Parameters
        ----------
        mu : np.ndarray, shape (N,)
            Expected returns.
        cov : np.ndarray, shape (N, N)
            Covariance matrix.
        initial_weights : np.ndarray, shape (N,), optional
            Starting weights (default: equal-weight).
        """
        N = self.n_assets
        cfg = self.config

        if initial_weights is None:
            w0 = np.ones(N, dtype=np.float32) / N
        else:
            w0 = initial_weights.astype(np.float32)

        mu_j = jnp.array(mu, dtype=jnp.float32)
        cov_j = jnp.array(cov, dtype=jnp.float32)

        def objective(w):
            ret = mu_j @ w
            var = w @ cov_j @ w
            penalty = cfg.l2_reg * jnp.sum(w ** 2)
            return -(ret - 0.5 * cfg.risk_aversion * var) + penalty

        grad_fn = jax.jit(jax.grad(objective))
        optimizer = optax.adam(cfg.lr)
        opt_state = optimizer.init(w0)
        w = w0.copy()

        for _ in range(cfg.n_iters):
            g = np.array(grad_fn(jnp.array(w)))
            updates, opt_state = optimizer.update(g, opt_state)
            w = w + np.array(updates)
            # Project onto [min, max] box then simplex
            w = np.clip(w, cfg.min_weight, cfg.max_weight)
            if not cfg.allow_short:
                w = np.maximum(w, 0.0)
            w = _simplex_project(w)

        return w.astype(np.float32)

    def efficient_frontier(
        self, mu: np.ndarray, cov: np.ndarray, n_points: int = 30
    ) -> tuple:
        """
        Compute the efficient frontier by varying ``risk_aversion``.

        Returns
        -------
        returns_arr : np.ndarray, shape (n_points,)
        vols_arr    : np.ndarray, shape (n_points,)
        weights_arr : np.ndarray, shape (n_points, N)
        """
        lambdas = np.logspace(-1, 2, n_points)
        orig_lam = self.config.risk_aversion
        all_w, all_ret, all_vol = [], [], []
        for lam in lambdas:
            self.config.risk_aversion = float(lam)
            w = self.optimise(mu, cov)
            all_w.append(w)
            all_ret.append(float(mu @ w) * 252)
            all_vol.append(float(np.sqrt(252 * w @ cov @ w)))
        self.config.risk_aversion = orig_lam
        return np.array(all_ret), np.array(all_vol), np.stack(all_w)


# ---------------------------------------------------------------------------
# TTRiskParityPortfolio
# ---------------------------------------------------------------------------


@dataclass
class RiskParityConfig:
    """Configuration for TTRiskParityPortfolio."""
    n_iters: int = 1_000
    lr: float = 1e-2
    tol: float = 1e-8
    budget: np.ndarray | None = None   # target risk fractions, sums to 1


class TTRiskParityPortfolio:
    """
    Equal (or budgeted) risk-contribution portfolio optimiser.

    Minimises::

        sum_{i,j} (RC_i - b_i)^2

    where RC_i is the fractional risk contribution of asset i and b_i is its
    target budget.

    Parameters
    ----------
    config : RiskParityConfig
    n_assets : int
    """

    def __init__(self, config: RiskParityConfig | None = None, n_assets: int = 64) -> None:
        self.config = config or RiskParityConfig()
        self.n_assets = n_assets

    def optimise(self, cov: np.ndarray) -> np.ndarray:
        """
        Compute risk-parity weights.

        Parameters
        ----------
        cov : np.ndarray, shape (N, N)
            Covariance matrix.

        Returns
        -------
        weights : np.ndarray, shape (N,)
        """
        cfg = self.config
        N = self.n_assets
        budget = cfg.budget
        if budget is None:
            budget = np.ones(N, dtype=np.float32) / N
        budget = np.array(budget, dtype=np.float32)
        cov_j = jnp.array(cov, dtype=jnp.float32)
        b_j = jnp.array(budget, dtype=jnp.float32)

        def loss(log_w):
            w = jax.nn.softmax(log_w)   # ensures positivity & sums to 1
            rc = w * (cov_j @ w) / (w @ cov_j @ w + 1e-12)
            return jnp.sum((rc - b_j) ** 2)

        grad_fn = jax.jit(jax.grad(loss))
        log_w = jnp.zeros(N)
        optimizer = optax.adam(cfg.lr)
        opt_state = optimizer.init(log_w)

        prev_loss = float("inf")
        for _ in range(cfg.n_iters):
            g = grad_fn(log_w)
            updates, opt_state = optimizer.update(g, opt_state)
            log_w = log_w + updates
            cur_loss = float(loss(log_w))
            if abs(prev_loss - cur_loss) < cfg.tol:
                break
            prev_loss = cur_loss

        return np.array(jax.nn.softmax(log_w), dtype=np.float32)

    def risk_contributions(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Return fractional risk contributions given weights and cov."""
        w = jnp.array(weights)
        cov_j = jnp.array(cov)
        rc = w * (cov_j @ w) / (w @ cov_j @ w + 1e-12)
        return np.array(rc, dtype=np.float32)


# ---------------------------------------------------------------------------
# TTFactorModel
# ---------------------------------------------------------------------------


@dataclass
class FactorModelConfig:
    """Configuration for TTFactorModel."""
    n_assets: int = 64
    n_factors: int = 5
    tt_rank: int = 4
    reg_lambda: float = 1e-3
    n_iters: int = 200
    lr: float = 5e-3


class TTFactorModel:
    """
    Linear factor model with TT-compressed factor loading matrix.

    Model::

        r_t = B f_t + epsilon_t

    where B (N x K) is the loading matrix (stored as a TT-matrix if large),
    f_t are factor returns, and epsilon_t is idiosyncratic noise.

    Parameters
    ----------
    config : FactorModelConfig
    """

    def __init__(self, config: FactorModelConfig | None = None) -> None:
        self.config = config or FactorModelConfig()
        N, K = self.config.n_assets, self.config.n_factors
        # Initialise loadings randomly
        key = jax.random.PRNGKey(0)
        self._B: jnp.ndarray = jax.random.normal(key, (N, K)) * 0.01
        self._factor_cov: jnp.ndarray = jnp.eye(K)
        self._idio_var: jnp.ndarray = jnp.ones(N)
        self._fitted = False

    def fit(self, returns: np.ndarray, factors: np.ndarray) -> dict:
        """
        Estimate loadings via OLS with L2 regularisation.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
            Asset returns.
        factors : np.ndarray, shape (T, K)
            Factor returns.

        Returns
        -------
        dict with ``"r2"`` and ``"residual_std"``.
        """
        T, N = returns.shape
        K = self.config.n_factors
        F = np.array(factors, dtype=np.float64)
        R = np.array(returns, dtype=np.float64)
        lam = self.config.reg_lambda

        # OLS: B = (F^T F + lam I)^{-1} F^T R
        FtF = F.T @ F + lam * np.eye(K)
        FtR = F.T @ R
        B = np.linalg.solve(FtF, FtR)   # (K, N)
        self._B = jnp.array(B.T, dtype=jnp.float32)   # (N, K)

        resid = R - F @ B    # (T, N)
        self._idio_var = jnp.array(np.var(resid, axis=0), dtype=jnp.float32)

        factor_cov = np.cov(F.T) if K > 1 else np.array([[np.var(F)]])
        self._factor_cov = jnp.array(factor_cov, dtype=jnp.float32)

        ss_res = np.sum(resid ** 2, axis=0)
        ss_tot = np.sum((R - R.mean(axis=0)) ** 2, axis=0)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        self._fitted = True
        return {"r2": r2.tolist(), "residual_std": np.sqrt(np.var(resid, axis=0)).tolist()}

    def covariance(self) -> np.ndarray:
        """
        Return model-implied N x N covariance matrix.

        Sigma = B Sigma_f B^T + diag(sigma_eps^2)
        """
        if not self._fitted:
            warnings.warn("TTFactorModel.covariance() called before fit(); returning identity.")
            N = self.config.n_assets
            return np.eye(N, dtype=np.float32)
        B = self._B          # (N, K)
        Sf = self._factor_cov  # (K, K)
        systematic = B @ Sf @ B.T
        idio = jnp.diag(self._idio_var)
        return np.array(systematic + idio, dtype=np.float32)

    def factor_exposures(self) -> np.ndarray:
        """Return loading matrix (N, K)."""
        return np.array(self._B, dtype=np.float32)

    def predict(self, factor_returns: np.ndarray) -> np.ndarray:
        """
        Predict asset returns given factor returns.

        Parameters
        ----------
        factor_returns : np.ndarray, shape (T, K) or (K,)

        Returns
        -------
        np.ndarray, shape (T, N) or (N,)
        """
        f = np.atleast_2d(factor_returns).astype(np.float32)
        B = np.array(self._B)    # (N, K)
        return (f @ B.T).squeeze()


# ---------------------------------------------------------------------------
# TTBlackLitterman
# ---------------------------------------------------------------------------


@dataclass
class BLConfig:
    """Configuration for TTBlackLitterman."""
    tau: float = 0.025            # uncertainty in prior
    risk_aversion: float = 2.5    # market risk aversion delta
    n_iters: int = 1              # BL is analytical, 1 iter suffices


class TTBlackLitterman:
    """
    Black-Litterman model for blending market equilibrium returns with
    investor views, using a TT-estimated covariance matrix.

    The posterior expected returns are::

        mu_BL = [(tau Sigma)^{-1} + P^T Omega^{-1} P]^{-1}
                [(tau Sigma)^{-1} Pi + P^T Omega^{-1} q]

    Parameters
    ----------
    config : BLConfig
    n_assets : int
    """

    def __init__(self, config: BLConfig | None = None, n_assets: int = 64) -> None:
        self.config = config or BLConfig()
        self.n_assets = n_assets

    def equilibrium_returns(
        self, weights: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        """
        Implied equilibrium returns Pi = delta * Sigma * w_mkt.

        Parameters
        ----------
        weights : np.ndarray, shape (N,)
            Market-cap weights.
        cov : np.ndarray, shape (N, N)
            Covariance matrix.
        """
        return self.config.risk_aversion * cov @ weights

    def posterior(
        self,
        pi: np.ndarray,
        cov: np.ndarray,
        P: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray | None = None,
    ) -> tuple:
        """
        Compute Black-Litterman posterior.

        Parameters
        ----------
        pi : np.ndarray, shape (N,)
            Prior (equilibrium) expected returns.
        cov : np.ndarray, shape (N, N)
            Prior covariance Sigma.
        P : np.ndarray, shape (K, N)
            Pick matrix encoding K views.
        q : np.ndarray, shape (K,)
            View returns.
        omega : np.ndarray, shape (K, K), optional
            View uncertainty matrix.  Default: tau * P Sigma P^T.

        Returns
        -------
        mu_bl : np.ndarray, shape (N,)
            Posterior expected returns.
        sigma_bl : np.ndarray, shape (N, N)
            Posterior covariance.
        """
        tau = self.config.tau
        N = self.n_assets
        eps = 1e-6
        if omega is None:
            omega = tau * P @ cov @ P.T + eps * np.eye(P.shape[0])

        tau_sigma_inv = np.linalg.inv(tau * cov + eps * np.eye(N))
        omega_inv = np.linalg.inv(omega)

        M = tau_sigma_inv + P.T @ omega_inv @ P
        M_inv = np.linalg.inv(M + eps * np.eye(N))

        rhs = tau_sigma_inv @ pi + P.T @ omega_inv @ q
        mu_bl = M_inv @ rhs
        sigma_bl = cov + M_inv
        return mu_bl.astype(np.float32), sigma_bl.astype(np.float32)

    def optimise(
        self,
        pi: np.ndarray,
        cov: np.ndarray,
        P: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray | None = None,
        mv_config: MVOptConfig | None = None,
    ) -> np.ndarray:
        """
        Full BL workflow: compute posterior, then MV-optimise.

        Returns portfolio weights (N,).
        """
        mu_bl, sigma_bl = self.posterior(pi, cov, P, q, omega)
        mv = TTMeanVarianceOptimiser(mv_config, n_assets=self.n_assets)
        return mv.optimise(mu_bl, sigma_bl)


# ---------------------------------------------------------------------------
# TransactionCostModel
# ---------------------------------------------------------------------------


@dataclass
class TCostConfig:
    """Configuration for transaction cost model."""
    spread_bps: float = 5.0          # half-spread in basis points
    commission_bps: float = 2.0      # commission in bps
    market_impact_coeff: float = 0.1 # price impact scaling
    avg_daily_volume: np.ndarray | None = None  # (N,) ADV for each asset


class TransactionCostModel:
    """
    Three-part transaction cost model:

    1. Half-spread cost   : spread_bps * abs(delta_w)
    2. Commission         : commission_bps * abs(delta_w)
    3. Market impact      : coeff * (trade_size / ADV)^0.6

    All costs expressed as fractions of portfolio value.

    Parameters
    ----------
    config : TCostConfig
    n_assets : int
    """

    def __init__(self, config: TCostConfig | None = None, n_assets: int = 64) -> None:
        self.config = config or TCostConfig()
        self.n_assets = n_assets
        if self.config.avg_daily_volume is None:
            self.config.avg_daily_volume = np.ones(n_assets, dtype=np.float32)

    def cost(
        self,
        w_old: np.ndarray,
        w_new: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> float:
        """
        Compute total one-way transaction cost (as fraction of portfolio value).

        Parameters
        ----------
        w_old : np.ndarray, shape (N,)
        w_new : np.ndarray, shape (N,)
        portfolio_value : float
        """
        cfg = self.config
        delta = np.abs(w_new - w_old)
        turnover = delta.sum()

        # Spread + commission
        linear_cost = (cfg.spread_bps + cfg.commission_bps) * 1e-4 * turnover

        # Market impact
        trade_sizes = delta * portfolio_value
        adv = np.array(cfg.avg_daily_volume, dtype=np.float64) + 1e-12
        impact_per_asset = cfg.market_impact_coeff * (trade_sizes / adv) ** 0.6
        market_impact = impact_per_asset.sum()

        return float(linear_cost + market_impact)

    def net_return(
        self,
        gross_return: float,
        w_old: np.ndarray,
        w_new: np.ndarray,
        portfolio_value: float = 1.0,
    ) -> float:
        """Gross return minus transaction costs."""
        return gross_return - self.cost(w_old, w_new, portfolio_value)


# ---------------------------------------------------------------------------
# PositionSizer
# ---------------------------------------------------------------------------


@dataclass
class SizerConfig:
    """Configuration for PositionSizer."""
    method: str = "vol_target"   # "kelly" | "fixed_fraction" | "vol_target"
    vol_target: float = 0.10     # annualised portfolio vol target
    kelly_fraction: float = 0.25 # fraction of full Kelly
    fixed_fraction: float = 0.02 # fixed fraction per bet


class PositionSizer:
    """
    Scales raw portfolio weights by a position sizing rule.

    Supported methods:
    * ``"vol_target"``    : scale so that portfolio vol = vol_target
    * ``"kelly"``         : fractional Kelly (requires mu, cov)
    * ``"fixed_fraction"``: uniform fixed fraction

    Parameters
    ----------
    config : SizerConfig
    """

    def __init__(self, config: SizerConfig | None = None) -> None:
        self.config = config or SizerConfig()

    def scale(
        self,
        weights: np.ndarray,
        cov: np.ndarray | None = None,
        mu: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return scaled weights.

        Parameters
        ----------
        weights : np.ndarray, shape (N,)
            Raw (unscaled) portfolio weights.
        cov : np.ndarray, shape (N, N), optional
            Required for ``"vol_target"`` and ``"kelly"``.
        mu : np.ndarray, shape (N,), optional
            Required for ``"kelly"``.
        """
        cfg = self.config
        if cfg.method == "fixed_fraction":
            return weights * cfg.fixed_fraction

        if cov is None:
            warnings.warn("PositionSizer: cov required for this method; returning unscaled.")
            return weights

        port_vol = float(np.sqrt(252 * weights @ cov @ weights + 1e-12))

        if cfg.method == "vol_target":
            if port_vol < 1e-8:
                return weights
            scale = cfg.vol_target / port_vol
            return weights * scale

        if cfg.method == "kelly":
            if mu is None:
                warnings.warn("PositionSizer: mu required for kelly sizing.")
                return weights
            # Full Kelly: w* proportional to Sigma^{-1} mu
            # We scale the provided weights by the Kelly fraction
            expected_excess = float(mu @ weights)
            port_var = float(weights @ cov @ weights + 1e-12)
            kelly_scale = expected_excess / port_var
            return weights * cfg.kelly_fraction * kelly_scale

        raise ValueError(f"Unknown sizing method: {cfg.method!r}")


# ---------------------------------------------------------------------------
# RiskBudgetAllocator
# ---------------------------------------------------------------------------


@dataclass
class RiskBudgetConfig:
    """Configuration for RiskBudgetAllocator."""
    n_iters: int = 2_000
    lr: float = 5e-3
    tol: float = 1e-9
    budget_type: str = "equal"   # "equal" | "custom"
    custom_budget: np.ndarray | None = None


class RiskBudgetAllocator:
    """
    Optimises weights to match a target risk-budget allocation.

    Extends :class:`TTRiskParityPortfolio` with:
    * Custom non-equal budgets
    * Constraint: weights >= 0, sum(weights) = 1
    * Verbose convergence history

    Parameters
    ----------
    config : RiskBudgetConfig
    n_assets : int
    """

    def __init__(self, config: RiskBudgetConfig | None = None, n_assets: int = 64) -> None:
        self.config = config or RiskBudgetConfig()
        self.n_assets = n_assets
        self.loss_history: list[float] = []

    def allocate(self, cov: np.ndarray) -> np.ndarray:
        """
        Compute risk-budget weights.

        Parameters
        ----------
        cov : np.ndarray, shape (N, N)

        Returns
        -------
        weights : np.ndarray, shape (N,)
        """
        cfg = self.config
        N = self.n_assets
        if cfg.budget_type == "equal":
            budget = np.ones(N, dtype=np.float32) / N
        elif cfg.custom_budget is not None:
            budget = np.array(cfg.custom_budget, dtype=np.float32)
            budget /= budget.sum()
        else:
            budget = np.ones(N, dtype=np.float32) / N

        cov_j = jnp.array(cov, dtype=jnp.float32)
        b_j = jnp.array(budget, dtype=jnp.float32)

        def loss(log_w):
            w = jax.nn.softmax(log_w)
            port_var = w @ cov_j @ w + 1e-12
            rc = w * (cov_j @ w) / port_var
            return jnp.sum((rc - b_j) ** 2)

        grad_fn = jax.jit(jax.grad(loss))
        log_w = jnp.zeros(N)
        optimizer = optax.adam(cfg.lr)
        opt_state = optimizer.init(log_w)
        self.loss_history = []
        prev_loss = float("inf")

        for _ in range(cfg.n_iters):
            g = grad_fn(log_w)
            updates, opt_state = optimizer.update(g, opt_state)
            log_w = log_w + updates
            cur_loss = float(loss(log_w))
            self.loss_history.append(cur_loss)
            if abs(prev_loss - cur_loss) < cfg.tol:
                break
            prev_loss = cur_loss

        return np.array(jax.nn.softmax(log_w), dtype=np.float32)


# ---------------------------------------------------------------------------
# CorrelationShockModel
# ---------------------------------------------------------------------------


@dataclass
class CorrelationShockConfig:
    """Configuration for CorrelationShockModel."""
    shock_magnitude: float = 0.3    # add this to all off-diagonal correlations
    shock_assets: list | None = None  # which asset indices to shock (None = all)
    n_scenarios: int = 50
    random_seed: int = 42


class CorrelationShockModel:
    """
    Stress-tests a portfolio by shocking the correlation matrix.

    Generates scenarios by:
    1. Decomposing Sigma into correlation C and standard deviations sigma
    2. Applying additive/multiplicative shocks to C
    3. Re-composing Sigma_shocked = diag(sigma) C_shocked diag(sigma)
    4. Evaluating portfolio loss distribution

    Parameters
    ----------
    config : CorrelationShockConfig
    """

    def __init__(self, config: CorrelationShockConfig | None = None) -> None:
        self.config = config or CorrelationShockConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

    def apply_uniform_shock(self, cov: np.ndarray) -> np.ndarray:
        """
        Increase all off-diagonal correlations by ``shock_magnitude``.

        Parameters
        ----------
        cov : np.ndarray, shape (N, N)

        Returns
        -------
        cov_shocked : np.ndarray, shape (N, N)
        """
        std = np.sqrt(np.diag(cov))
        corr = cov / (np.outer(std, std) + 1e-12)
        shock = self.config.shock_magnitude
        corr_shocked = corr + shock * (1 - np.eye(len(std)))
        corr_shocked = np.clip(corr_shocked, -0.999, 0.999)
        np.fill_diagonal(corr_shocked, 1.0)
        return np.outer(std, std) * corr_shocked

    def random_scenarios(
        self, cov: np.ndarray, weights: np.ndarray, portfolio_value: float = 1.0
    ) -> np.ndarray:
        """
        Generate random correlation shock scenarios and compute portfolio losses.

        Returns
        -------
        losses : np.ndarray, shape (n_scenarios,)
            Portfolio P&L for each scenario.
        """
        cfg = self.config
        N = cov.shape[0]
        std = np.sqrt(np.diag(cov))
        corr = cov / (np.outer(std, std) + 1e-12)
        losses = []

        for _ in range(cfg.n_scenarios):
            delta = self._rng.uniform(-cfg.shock_magnitude, cfg.shock_magnitude, (N, N))
            delta = (delta + delta.T) / 2   # symmetrise
            np.fill_diagonal(delta, 0.0)
            corr_s = np.clip(corr + delta, -0.999, 0.999)
            np.fill_diagonal(corr_s, 1.0)
            cov_s = np.outer(std, std) * corr_s
            # Simulate portfolio return: N(0, cov_s)
            r = self._rng.multivariate_normal(np.zeros(N), cov_s)
            losses.append(float(weights @ r) * portfolio_value)

        return np.array(losses)

    def var(self, losses: np.ndarray, confidence: float = 0.95) -> float:
        """Value-at-Risk at given confidence level."""
        return float(-np.percentile(losses, (1 - confidence) * 100))

    def cvar(self, losses: np.ndarray, confidence: float = 0.95) -> float:
        """Conditional Value-at-Risk (Expected Shortfall)."""
        var = self.var(losses, confidence)
        tail = losses[losses < -var]
        if len(tail) == 0:
            return float(var)
        return float(-tail.mean())


# ---------------------------------------------------------------------------
# PortfolioBacktester
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Configuration for PortfolioBacktester."""
    rebalance_frequency: int = 21   # days between rebalances
    lookback: int = 252             # estimation window (days)
    transaction_cost_config: TCostConfig = field(default_factory=TCostConfig)
    initial_value: float = 1.0
    verbose: bool = False


class BacktestResult(NamedTuple):
    """Result container for a portfolio backtest."""
    dates: np.ndarray              # (T,) date indices
    portfolio_value: np.ndarray    # (T,) cumulative portfolio value
    weights: np.ndarray            # (T, N) portfolio weights over time
    returns: np.ndarray            # (T,) daily portfolio returns
    turnover: np.ndarray           # (T,) daily turnover
    transaction_costs: np.ndarray  # (T,) daily transaction costs
    sharpe: float
    max_drawdown: float
    annualised_return: float
    annualised_vol: float
    calmar: float


class PortfolioBacktester:
    """
    Vectorised portfolio backtest engine.

    Supports:
    * Any weight-generation callable
    * Transaction cost simulation
    * Comprehensive performance analytics

    Parameters
    ----------
    config : BacktestConfig
    weight_fn : callable (returns_window: np.ndarray) -> weights: np.ndarray
        Function that computes portfolio weights from a (lookback, N) return
        window.
    """

    def __init__(self, config: BacktestConfig | None = None, weight_fn=None) -> None:
        self.config = config or BacktestConfig()
        self.weight_fn = weight_fn
        self._tcost = TransactionCostModel(config.transaction_cost_config if config else None)

    def run(self, returns: np.ndarray) -> BacktestResult:
        """
        Execute the backtest.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
            Daily asset return matrix.

        Returns
        -------
        BacktestResult
        """
        cfg = self.config
        T, N = returns.shape
        lookback = cfg.lookback
        freq = cfg.rebalance_frequency

        pv = cfg.initial_value
        w = np.ones(N, dtype=np.float32) / N  # initial equal-weight

        pv_hist, w_hist, ret_hist, turn_hist, tc_hist = [], [], [], [], []
        dates = np.arange(T)

        for t in range(lookback, T):
            # Rebalance?
            if (t - lookback) % freq == 0 and self.weight_fn is not None:
                window = returns[t - lookback:t]
                try:
                    w_new = self.weight_fn(window)
                    w_new = np.array(w_new, dtype=np.float32)
                    w_new = np.clip(w_new, 0, 1)
                    if w_new.sum() > 0:
                        w_new /= w_new.sum()
                    else:
                        w_new = w.copy()
                except Exception as exc:
                    if cfg.verbose:
                        warnings.warn(f"weight_fn error at t={t}: {exc}")
                    w_new = w.copy()
                tc = self._tcost.cost(w, w_new, pv)
                w = w_new
            else:
                tc = 0.0

            # Daily return
            day_ret = float(w @ returns[t])
            pv = pv * (1 + day_ret) - tc * pv
            turnover = float(np.abs(w - w_hist[-1]).sum()) if w_hist else 0.0

            pv_hist.append(pv)
            w_hist.append(w.copy())
            ret_hist.append(day_ret)
            turn_hist.append(turnover)
            tc_hist.append(tc)

        pv_arr = np.array(pv_hist)
        ret_arr = np.array(ret_hist)
        w_arr = np.array(w_hist)
        turn_arr = np.array(turn_hist)
        tc_arr = np.array(tc_hist)

        # Performance metrics
        ann_ret = float((pv_arr[-1] / pv_arr[0]) ** (252 / len(pv_arr)) - 1)
        ann_vol = float(np.std(ret_arr) * math.sqrt(252))
        sharpe = ann_ret / (ann_vol + 1e-12)

        # Max drawdown
        running_max = np.maximum.accumulate(pv_arr)
        dd = (pv_arr - running_max) / (running_max + 1e-12)
        max_dd = float(dd.min())

        calmar = ann_ret / (abs(max_dd) + 1e-12)

        return BacktestResult(
            dates=dates[lookback:],
            portfolio_value=pv_arr,
            weights=w_arr,
            returns=ret_arr,
            turnover=turn_arr,
            transaction_costs=tc_arr,
            sharpe=sharpe,
            max_drawdown=max_dd,
            annualised_return=ann_ret,
            annualised_vol=ann_vol,
            calmar=calmar,
        )

    def summary(self, result: BacktestResult) -> dict:
        """Return a dict of performance metrics."""
        return {
            "sharpe": round(result.sharpe, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "annualised_return": round(result.annualised_return, 4),
            "annualised_vol": round(result.annualised_vol, 4),
            "calmar": round(result.calmar, 4),
            "total_turnover": round(float(result.turnover.sum()), 4),
            "total_transaction_costs": round(float(result.transaction_costs.sum()), 6),
            "final_portfolio_value": round(float(result.portfolio_value[-1]), 6),
            "n_days": len(result.returns),
        }


# ---------------------------------------------------------------------------
# PortfolioAttributor
# ---------------------------------------------------------------------------


class PortfolioAttributor:
    """
    Brinson-Hood-Beebower performance attribution.

    Decomposes portfolio return vs. a benchmark into:
    * Allocation effect
    * Selection effect
    * Interaction effect

    Parameters
    ----------
    benchmark_weights : np.ndarray, shape (N,)
        Benchmark weights.
    benchmark_returns : np.ndarray, shape (T, N)
        Benchmark asset returns.
    """

    def __init__(
        self,
        benchmark_weights: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> None:
        self.bw = np.array(benchmark_weights, dtype=np.float32)
        self.br = np.array(benchmark_returns, dtype=np.float32)

    def attribute(
        self,
        portfolio_weights: np.ndarray,
        portfolio_returns: np.ndarray,
    ) -> dict:
        """
        Compute Brinson attribution for each time step.

        Parameters
        ----------
        portfolio_weights : np.ndarray, shape (T, N)
        portfolio_returns : np.ndarray, shape (T, N)

        Returns
        -------
        dict with keys ``"allocation"``, ``"selection"``, ``"interaction"``,
        each an np.ndarray of shape (T,).
        """
        pw = np.array(portfolio_weights, dtype=np.float32)
        pr = np.array(portfolio_returns, dtype=np.float32)
        T = pw.shape[0]
        bw = np.broadcast_to(self.bw, (T, len(self.bw)))
        br = self.br[-T:] if len(self.br) >= T else np.broadcast_to(self.br, (T, len(self.bw)))

        bench_r = (bw * br).sum(axis=1)            # scalar benchmark return per period
        alloc = ((pw - bw) * (br - bench_r[:, None])).sum(axis=1)
        sel = (bw * (pr - br)).sum(axis=1)
        inter = ((pw - bw) * (pr - br)).sum(axis=1)

        return {
            "allocation": alloc,
            "selection": sel,
            "interaction": inter,
            "total_active_return": alloc + sel + inter,
        }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def equal_weight(returns_window: np.ndarray) -> np.ndarray:
    """Simple equal-weight strategy callable for PortfolioBacktester."""
    N = returns_window.shape[1]
    return np.ones(N, dtype=np.float32) / N


def inverse_vol_weight(returns_window: np.ndarray) -> np.ndarray:
    """Inverse-volatility weighting strategy."""
    vol = np.std(returns_window, axis=0) + 1e-12
    w = 1.0 / vol
    return (w / w.sum()).astype(np.float32)


def momentum_weight(returns_window: np.ndarray, top_k: int = 10) -> np.ndarray:
    """Long-only momentum: equal-weight top-k assets by trailing return."""
    N = returns_window.shape[1]
    cum_ret = returns_window.sum(axis=0)
    k = min(top_k, N)
    top_idx = np.argsort(cum_ret)[::-1][:k]
    w = np.zeros(N, dtype=np.float32)
    w[top_idx] = 1.0 / k
    return w


def compute_sharpe(returns: np.ndarray, annualise: bool = True) -> float:
    """Compute Sharpe ratio from return series."""
    mu = np.mean(returns)
    sigma = np.std(returns) + 1e-12
    sr = mu / sigma
    if annualise:
        sr *= math.sqrt(252)
    return float(sr)


def compute_max_drawdown(pv: np.ndarray) -> float:
    """Compute maximum drawdown from portfolio value series."""
    running_max = np.maximum.accumulate(pv)
    drawdown = (pv - running_max) / (running_max + 1e-12)
    return float(drawdown.min())


def compute_calmar(pv: np.ndarray) -> float:
    """Annualised return divided by max drawdown."""
    T = len(pv)
    ann_ret = (pv[-1] / pv[0]) ** (252 / T) - 1
    max_dd = abs(compute_max_drawdown(pv))
    return float(ann_ret / (max_dd + 1e-12))


def turnover_series(weights: np.ndarray) -> np.ndarray:
    """Compute per-period portfolio turnover from weights history (T, N)."""
    diffs = np.diff(weights, axis=0)
    return np.abs(diffs).sum(axis=1)


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Diversification ratio = weighted avg vol / portfolio vol.

    Values > 1 indicate diversification benefit.
    """
    std = np.sqrt(np.diag(cov))
    weighted_vol = float(weights @ std)
    port_vol = float(np.sqrt(weights @ cov @ weights + 1e-12))
    return weighted_vol / (port_vol + 1e-12)


def herfindahl_index(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman concentration index.

    1/N = maximally diversified; 1 = fully concentrated.
    """
    return float(np.sum(weights ** 2))


def effective_n(weights: np.ndarray) -> float:
    """Effective number of positions = 1 / HHI."""
    return 1.0 / (herfindahl_index(weights) + 1e-12)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config dataclasses
    "CovEstimatorConfig",
    "MVOptConfig",
    "RiskParityConfig",
    "FactorModelConfig",
    "BLConfig",
    "TCostConfig",
    "SizerConfig",
    "RiskBudgetConfig",
    "CorrelationShockConfig",
    "BacktestConfig",
    # Core classes
    "TTCovarianceEstimator",
    "TTMeanVarianceOptimiser",
    "TTRiskParityPortfolio",
    "TTFactorModel",
    "TTBlackLitterman",
    "TransactionCostModel",
    "PositionSizer",
    "RiskBudgetAllocator",
    "CorrelationShockModel",
    "PortfolioBacktester",
    "BacktestResult",
    "PortfolioAttributor",
    # Utility functions
    "equal_weight",
    "inverse_vol_weight",
    "momentum_weight",
    "compute_sharpe",
    "compute_max_drawdown",
    "compute_calmar",
    "turnover_series",
    "diversification_ratio",
    "herfindahl_index",
    "effective_n",
]
