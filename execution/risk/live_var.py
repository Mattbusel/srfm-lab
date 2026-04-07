"""
execution/risk/live_var.py
==========================
Real-time Value-at-Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
for the SRFM Lab portfolio.

Three complementary methods are provided:
  ParametricVaR    -- delta-normal with EWMA covariance (RiskMetrics lambda=0.94)
  HistoricalVaR    -- full-revaluation simulation on rolling 252-day return window
  MonteCarloVaR    -- 10 000 correlated GBM paths via Cholesky decomposition

VaRMonitor wraps all three, computes a consensus estimate, performs Kupiec
backtesting, detects breaches, and persists metrics to the SQLite risk_metrics
table.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("execution.risk.live_var")

_DB_PATH = Path(__file__).parents[2] / "execution" / "live_trades.db"

# ---------------------------------------------------------------------------
# Confidence levels
# ---------------------------------------------------------------------------
CONF_95 = 0.95
CONF_99 = 0.99
EWMA_LAMBDA = 0.94          # RiskMetrics decay factor
HIST_WINDOW = 252           # trading days for historical simulation
MC_PATHS = 10_000           # number of Monte Carlo paths
MC_HORIZON = 1              # 1-day horizon
ANNUALIZATION = 252.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PositionSnapshot:
    """Snapshot of a single position at a point in time."""
    symbol: str
    qty: float                  # positive = long, negative = short
    entry_price: float
    current_price: float

    @property
    def notional(self) -> float:
        """Signed notional value in USD."""
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in USD."""
        return self.qty * (self.current_price - self.entry_price)


@dataclass
class PortfolioSnapshot:
    """Full portfolio state at a point in time."""
    positions: List[PositionSnapshot]
    equity: float               # total account equity in USD
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def symbols(self) -> List[str]:
        return [p.symbol for p in self.positions]

    @property
    def weights(self) -> np.ndarray:
        """Dollar-weighted position fractions (signed)."""
        notionals = np.array([p.notional for p in self.positions])
        if self.equity <= 0:
            return np.zeros(len(self.positions))
        return notionals / self.equity


@dataclass
class VaRResult:
    """Unified result container for all VaR methods."""
    var_95: float               # VaR at 95% confidence (positive number = loss)
    var_99: float               # VaR at 99% confidence
    cvar_95: float              # CVaR / ES at 95%
    cvar_99: float              # CVaR / ES at 99%
    method: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Parametric VaR (delta-normal, EWMA covariance)
# ---------------------------------------------------------------------------

class ParametricVaR:
    """
    Delta-normal VaR using an EWMA covariance matrix (RiskMetrics, lambda=0.94).

    The covariance matrix is updated incrementally as new daily returns arrive.
    Portfolio VaR is sqrt(w' Sigma w) * z_alpha * sqrt(horizon).
    """

    def __init__(self, ewma_lambda: float = EWMA_LAMBDA) -> None:
        self.lam = ewma_lambda
        # keyed by symbol; values are 1-D numpy arrays of returns
        self._returns: Dict[str, float] = {}
        # running EWMA covariance: symbol-pair -> cov
        self._cov: Optional[np.ndarray] = None
        self._symbols: List[str] = []
        self._vol: Dict[str, float] = {}     # per-symbol EWMA variance
        self._cross: Dict[Tuple[str, str], float] = {}  # off-diagonal EWMA cov

    # ------------------------------------------------------------------ #
    # Update                                                               #
    # ------------------------------------------------------------------ #

    def update(self, returns: Dict[str, float]) -> None:
        """
        Ingest one day's returns for all symbols.

        Parameters
        ----------
        returns : dict mapping symbol -> daily return (fractional, e.g. 0.01 = 1%)
        """
        for sym, r in returns.items():
            if sym not in self._vol:
                self._vol[sym] = r ** 2
            else:
                self._vol[sym] = self.lam * self._vol[sym] + (1 - self.lam) * r ** 2
            self._returns[sym] = r

        # Update off-diagonal covariances
        syms = sorted(returns.keys())
        for i, s1 in enumerate(syms):
            for j, s2 in enumerate(syms):
                if j <= i:
                    continue
                key = (s1, s2)
                r1, r2 = returns.get(s1, 0.0), returns.get(s2, 0.0)
                if key not in self._cross:
                    self._cross[key] = r1 * r2
                else:
                    self._cross[key] = self.lam * self._cross[key] + (1 - self.lam) * r1 * r2

    def _build_cov_matrix(self, symbols: List[str]) -> np.ndarray:
        """Construct the EWMA covariance matrix for the given symbol list."""
        n = len(symbols)
        cov = np.zeros((n, n))
        for i, s1 in enumerate(symbols):
            cov[i, i] = self._vol.get(s1, 1e-6)
            for j, s2 in enumerate(symbols):
                if j <= i:
                    continue
                key = (s1, s2) if s1 < s2 else (s2, s1)
                val = self._cross.get(key, 0.0)
                cov[i, j] = val
                cov[j, i] = val
        # Ensure positive semi-definiteness via diagonal nudge
        min_eig = np.linalg.eigvalsh(cov).min()
        if min_eig < 1e-10:
            cov += np.eye(n) * (abs(min_eig) + 1e-8)
        return cov

    # ------------------------------------------------------------------ #
    # VaR calculation                                                      #
    # ------------------------------------------------------------------ #

    def portfolio_var(
        self,
        snapshot: PortfolioSnapshot,
        horizon_days: int = 1,
    ) -> VaRResult:
        """Compute delta-normal portfolio VaR."""
        if not snapshot.positions:
            return VaRResult(0.0, 0.0, 0.0, 0.0, "parametric")

        syms = snapshot.symbols
        w = snapshot.weights
        cov = self._build_cov_matrix(syms)

        sigma_p = math.sqrt(float(w @ cov @ w) * horizon_days)
        z95 = stats.norm.ppf(CONF_95)
        z99 = stats.norm.ppf(CONF_99)

        # For normal distribution ES = phi(z) / (1-alpha) * sigma
        es95 = (stats.norm.pdf(z95) / (1 - CONF_95)) * sigma_p
        es99 = (stats.norm.pdf(z99) / (1 - CONF_99)) * sigma_p

        equity = snapshot.equity
        return VaRResult(
            var_95=abs(z95 * sigma_p * equity),
            var_99=abs(z99 * sigma_p * equity),
            cvar_95=abs(es95 * equity),
            cvar_99=abs(es99 * equity),
            method="parametric",
        )

    def marginal_var(
        self,
        snapshot: PortfolioSnapshot,
        confidence: float = CONF_99,
    ) -> Dict[str, float]:
        """
        Marginal VaR per position: dVaR/dw_i * w_i.

        Returns a dict mapping symbol -> marginal VaR contribution in USD.
        """
        if not snapshot.positions:
            return {}
        syms = snapshot.symbols
        w = snapshot.weights
        cov = self._build_cov_matrix(syms)
        sigma_p2 = float(w @ cov @ w)
        if sigma_p2 <= 0:
            return {s: 0.0 for s in syms}
        sigma_p = math.sqrt(sigma_p2)
        z = stats.norm.ppf(confidence)
        # Marginal contribution: (Sigma * w)_i / sigma_p * z * equity
        contrib = (cov @ w) / sigma_p * z * snapshot.equity
        return {s: float(contrib[i]) * w[i] for i, s in enumerate(syms)}

    def component_var(
        self,
        snapshot: PortfolioSnapshot,
        confidence: float = CONF_99,
    ) -> Dict[str, float]:
        """
        Component VaR: each position's contribution to total portfolio VaR.

        sum(component_var.values()) == portfolio_var (approximately).
        """
        if not snapshot.positions:
            return {}
        syms = snapshot.symbols
        w = snapshot.weights
        cov = self._build_cov_matrix(syms)
        sigma_p2 = float(w @ cov @ w)
        if sigma_p2 <= 0:
            return {s: 0.0 for s in syms}
        sigma_p = math.sqrt(sigma_p2)
        z = stats.norm.ppf(confidence)
        # Component VaR_i = (Sigma @ w)_i / sigma_p * w_i * z * equity
        # This satisfies: sum_i component_var_i == portfolio_var (Euler homogeneity)
        contrib = (cov @ w) / sigma_p  # length n
        result = {}
        for i, s in enumerate(syms):
            result[s] = float(contrib[i]) * w[i] * z * snapshot.equity
        return result

    def diversification_ratio(self, snapshot: PortfolioSnapshot) -> float:
        """
        Diversification ratio = weighted sum of individual vols / portfolio vol.

        Values > 1 indicate diversification benefit.
        """
        if not snapshot.positions:
            return 1.0
        syms = snapshot.symbols
        w = np.abs(snapshot.weights)
        ind_vols = np.array([math.sqrt(self._vol.get(s, 1e-6)) for s in syms])
        weighted_sum_vol = float(w @ ind_vols)
        cov = self._build_cov_matrix(syms)
        portfolio_vol = math.sqrt(max(float(snapshot.weights @ cov @ snapshot.weights), 1e-12))
        if portfolio_vol <= 0:
            return 1.0
        return weighted_sum_vol / portfolio_vol


# ---------------------------------------------------------------------------
# Historical VaR (full revaluation)
# ---------------------------------------------------------------------------

class HistoricalVaR:
    """
    Full-revaluation historical simulation.

    Maintains a rolling 252-day window of per-symbol returns.
    Weights scenarios by recency using EWMA (older scenarios get lower weight).
    Fat tails are handled naturally because actual historical returns are used.
    """

    def __init__(
        self,
        window: int = HIST_WINDOW,
        ewma_lambda: float = EWMA_LAMBDA,
    ) -> None:
        self.window = window
        self.lam = ewma_lambda
        # symbol -> deque of (date_str, return) in chronological order
        self._history: Dict[str, List[float]] = {}

    def update(self, returns: Dict[str, float]) -> None:
        """Append one day's returns to the rolling history."""
        for sym, r in returns.items():
            if sym not in self._history:
                self._history[sym] = []
            buf = self._history[sym]
            buf.append(r)
            if len(buf) > self.window:
                buf.pop(0)

    def _scenario_weights(self, n: int) -> np.ndarray:
        """
        EWMA scenario weights: most recent observation has highest weight.
        Returns normalised weights of length n.
        """
        # Weight for scenario t (0=oldest) = lambda^(n-1-t) * (1-lambda)
        ages = np.arange(n - 1, -1, -1, dtype=float)  # newest=0, oldest=n-1
        w = (1 - self.lam) * (self.lam ** ages)
        total_w = float(w.sum())
        w = w / total_w if total_w > 1e-300 else np.ones(n, dtype=float) / n
        return w

    def portfolio_var(
        self,
        snapshot: PortfolioSnapshot,
        horizon_days: int = 1,
    ) -> VaRResult:
        """Compute weighted historical portfolio VaR."""
        if not snapshot.positions:
            return VaRResult(0.0, 0.0, 0.0, 0.0, "historical")

        syms = snapshot.symbols
        w = snapshot.weights  # fraction of equity

        # Find min available history
        lengths = [len(self._history.get(s, [])) for s in syms]
        if not lengths or min(lengths) < 5:
            log.warning("Historical VaR: insufficient history for %s", syms)
            return VaRResult(0.0, 0.0, 0.0, 0.0, "historical")

        n = min(lengths)
        # Build return matrix: shape (n, n_assets)
        ret_matrix = np.zeros((n, len(syms)))
        for i, s in enumerate(syms):
            ret_matrix[:, i] = self._history[s][-n:]

        # Scale for horizon (sqrt-of-time approximation for historical)
        ret_matrix *= math.sqrt(horizon_days)

        # Portfolio scenario returns
        port_returns = ret_matrix @ w  # shape (n,)
        scenario_weights = self._scenario_weights(n)

        # Weighted quantile
        sorted_idx = np.argsort(port_returns)
        sorted_returns = port_returns[sorted_idx]
        sorted_weights = scenario_weights[sorted_idx]
        cum_weights = np.cumsum(sorted_weights)

        def weighted_quantile(q: float) -> float:
            # Find the left-tail quantile index for confidence level q.
            # We want the scenario at the (1-q) quantile of the loss distribution.
            # Subtract a tiny epsilon from target so that when cumulative weight
            # is exactly equal to (1-q) we land on that scenario rather than
            # the one immediately after it (floating-point subtraction drift).
            target = (1.0 - q) - 1e-12
            target = max(target, 0.0)
            idx = np.searchsorted(cum_weights, target, side="right")
            idx = min(idx, n - 1)
            return float(sorted_returns[idx])

        q95 = weighted_quantile(CONF_95)
        q99 = weighted_quantile(CONF_99)

        # CVaR: expected loss beyond the VaR threshold (weighted average of tail)
        def weighted_cvar(threshold_return: float) -> float:
            tail_mask = sorted_returns <= threshold_return
            if not tail_mask.any():
                return abs(threshold_return)
            tail_w = sorted_weights[tail_mask]
            tail_r = sorted_returns[tail_mask]
            if tail_w.sum() <= 0:
                return abs(threshold_return)
            return float(-np.average(tail_r, weights=tail_w))

        cvar95 = weighted_cvar(q95)
        cvar99 = weighted_cvar(q99)

        equity = snapshot.equity
        return VaRResult(
            var_95=abs(q95) * equity,
            var_99=abs(q99) * equity,
            cvar_95=cvar95 * equity,
            cvar_99=cvar99 * equity,
            method="historical",
        )


# ---------------------------------------------------------------------------
# Monte Carlo VaR (correlated GBM)
# ---------------------------------------------------------------------------

class MonteCarloVaR:
    """
    Monte Carlo VaR using correlated Geometric Brownian Motion paths.

    The covariance matrix is estimated from the same EWMA approach as
    ParametricVaR. Cholesky decomposition produces correlated shocks.
    10 000 paths are simulated over 1-day horizon.
    """

    def __init__(
        self,
        n_paths: int = MC_PATHS,
        ewma_lambda: float = EWMA_LAMBDA,
        seed: Optional[int] = None,
    ) -> None:
        self.n_paths = n_paths
        self.lam = ewma_lambda
        self._rng = np.random.default_rng(seed)
        # Same EWMA state as ParametricVaR
        self._vol: Dict[str, float] = {}
        self._cross: Dict[Tuple[str, str], float] = {}
        self._mu: Dict[str, float] = {}
        self._last_return: Dict[str, float] = {}

    def update(self, returns: Dict[str, float]) -> None:
        """Ingest one day's returns; update EWMA cov state."""
        for sym, r in returns.items():
            self._last_return[sym] = r
            # EWMA mean (drift)
            if sym not in self._mu:
                self._mu[sym] = r
            else:
                self._mu[sym] = self.lam * self._mu[sym] + (1 - self.lam) * r
            # EWMA variance
            if sym not in self._vol:
                self._vol[sym] = r ** 2
            else:
                self._vol[sym] = self.lam * self._vol[sym] + (1 - self.lam) * r ** 2

        syms = sorted(returns.keys())
        for i, s1 in enumerate(syms):
            for j, s2 in enumerate(syms):
                if j <= i:
                    continue
                key = (s1, s2)
                r1, r2 = returns.get(s1, 0.0), returns.get(s2, 0.0)
                if key not in self._cross:
                    self._cross[key] = r1 * r2
                else:
                    self._cross[key] = self.lam * self._cross[key] + (1 - self.lam) * r1 * r2

    def _build_cov_matrix(self, symbols: List[str]) -> np.ndarray:
        n = len(symbols)
        cov = np.zeros((n, n))
        for i, s1 in enumerate(symbols):
            cov[i, i] = self._vol.get(s1, 1e-6)
            for j, s2 in enumerate(symbols):
                if j <= i:
                    continue
                key = (s1, s2) if s1 < s2 else (s2, s1)
                val = self._cross.get(key, 0.0)
                cov[i, j] = val
                cov[j, i] = val
        min_eig = np.linalg.eigvalsh(cov).min()
        if min_eig < 1e-10:
            cov += np.eye(n) * (abs(min_eig) + 1e-8)
        return cov

    def portfolio_var(
        self,
        snapshot: PortfolioSnapshot,
        horizon_days: int = 1,
    ) -> VaRResult:
        """Simulate portfolio return distribution and extract VaR/CVaR."""
        if not snapshot.positions:
            return VaRResult(0.0, 0.0, 0.0, 0.0, "montecarlo")

        syms = snapshot.symbols
        n_assets = len(syms)
        w = snapshot.weights

        cov = self._build_cov_matrix(syms)
        mu = np.array([self._mu.get(s, 0.0) for s in syms])

        # Cholesky for correlated shocks
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: nearest PD via eigenvalue clipping
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-8, None)
            cov_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(cov_pd)

        # Draw standard normal shocks: shape (n_paths, n_assets)
        z = self._rng.standard_normal((self.n_paths, n_assets))
        # Correlated returns for horizon: r = mu * h + L @ z * sqrt(h)
        h = float(horizon_days)
        corr_ret = mu * h + (z @ L.T) * math.sqrt(h)  # (n_paths, n_assets)

        # Portfolio return per path
        port_ret = corr_ret @ w  # (n_paths,)

        # Sort for quantile extraction
        port_ret_sorted = np.sort(port_ret)
        n = len(port_ret_sorted)

        idx95 = int(math.floor((1 - CONF_95) * n))
        idx99 = int(math.floor((1 - CONF_99) * n))
        idx95 = max(0, min(idx95, n - 1))
        idx99 = max(0, min(idx99, n - 1))

        q95 = float(port_ret_sorted[idx95])
        q99 = float(port_ret_sorted[idx99])

        cvar95 = float(-port_ret_sorted[:idx95 + 1].mean()) if idx95 >= 0 else abs(q95)
        cvar99 = float(-port_ret_sorted[:idx99 + 1].mean()) if idx99 >= 0 else abs(q99)

        equity = snapshot.equity
        return VaRResult(
            var_95=abs(q95) * equity,
            var_99=abs(q99) * equity,
            cvar_95=cvar95 * equity,
            cvar_99=cvar99 * equity,
            method="montecarlo",
        )


# ---------------------------------------------------------------------------
# VaR Monitor -- orchestrates all three methods
# ---------------------------------------------------------------------------

_CREATE_RISK_METRICS_SQL = """
CREATE TABLE IF NOT EXISTS risk_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    method          TEXT    NOT NULL,
    var_95          REAL,
    var_99          REAL,
    cvar_95         REAL,
    cvar_99         REAL,
    consensus_var99 REAL,
    equity          REAL,
    n_positions     INTEGER,
    breach_flag     INTEGER DEFAULT 0
);
"""

_CREATE_KUPIEC_SQL = """
CREATE TABLE IF NOT EXISTS var_backtests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    method          TEXT    NOT NULL,
    confidence      REAL,
    n_obs           INTEGER,
    n_exceptions    INTEGER,
    kupiec_stat     REAL,
    kupiec_pvalue   REAL,
    pass_flag       INTEGER
);
"""


class VaRMonitor:
    """
    Runs ParametricVaR, HistoricalVaR, and MonteCarloVaR on each update.

    Consensus VaR is a weighted average:
        40% parametric + 30% historical + 30% montecarlo

    Breach detection compares today's actual portfolio P&L against the
    previous period's VaR estimate. Kupiec LR test is run when there are
    at least 60 daily observations.
    """

    CONSENSUS_WEIGHTS = {
        "parametric": 0.40,
        "historical":  0.30,
        "montecarlo":  0.30,
    }

    def __init__(
        self,
        db_path: Path = _DB_PATH,
        ewma_lambda: float = EWMA_LAMBDA,
        n_mc_paths: int = MC_PATHS,
        mc_seed: Optional[int] = None,
    ) -> None:
        self.db_path = db_path
        self.parametric = ParametricVaR(ewma_lambda=ewma_lambda)
        self.historical = HistoricalVaR(ewma_lambda=ewma_lambda)
        self.montecarlo = MonteCarloVaR(n_paths=n_mc_paths, ewma_lambda=ewma_lambda, seed=mc_seed)
        # Rolling history of (actual_pnl, prior_var99_consensus)
        self._breach_history: List[Tuple[float, float]] = []
        self._last_consensus_var99: float = 0.0
        self._ensure_tables()

    # ------------------------------------------------------------------ #
    # Schema                                                               #
    # ------------------------------------------------------------------ #

    def _ensure_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_RISK_METRICS_SQL)
            conn.execute(_CREATE_KUPIEC_SQL)
            conn.commit()

    # ------------------------------------------------------------------ #
    # Main update                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        returns: Dict[str, float],
        snapshot: PortfolioSnapshot,
        actual_daily_pnl: Optional[float] = None,
    ) -> Dict[str, VaRResult]:
        """
        Ingest today's returns and compute VaR for the current snapshot.

        Parameters
        ----------
        returns          : dict symbol -> daily return fraction
        snapshot         : current PortfolioSnapshot
        actual_daily_pnl : realized P&L for the prior period (for breach test)

        Returns
        -------
        dict with keys 'parametric', 'historical', 'montecarlo', 'consensus'
        """
        self.parametric.update(returns)
        self.historical.update(returns)
        self.montecarlo.update(returns)

        p_result = self.parametric.portfolio_var(snapshot)
        h_result = self.historical.portfolio_var(snapshot)
        m_result = self.montecarlo.portfolio_var(snapshot)

        consensus_var99 = (
            self.CONSENSUS_WEIGHTS["parametric"] * p_result.var_99
            + self.CONSENSUS_WEIGHTS["historical"] * h_result.var_99
            + self.CONSENSUS_WEIGHTS["montecarlo"] * m_result.var_99
        )
        consensus_cvar99 = (
            self.CONSENSUS_WEIGHTS["parametric"] * p_result.cvar_99
            + self.CONSENSUS_WEIGHTS["historical"] * h_result.cvar_99
            + self.CONSENSUS_WEIGHTS["montecarlo"] * m_result.cvar_99
        )

        consensus = VaRResult(
            var_95=(
                self.CONSENSUS_WEIGHTS["parametric"] * p_result.var_95
                + self.CONSENSUS_WEIGHTS["historical"] * h_result.var_95
                + self.CONSENSUS_WEIGHTS["montecarlo"] * m_result.var_95
            ),
            var_99=consensus_var99,
            cvar_95=(
                self.CONSENSUS_WEIGHTS["parametric"] * p_result.cvar_95
                + self.CONSENSUS_WEIGHTS["historical"] * h_result.cvar_95
                + self.CONSENSUS_WEIGHTS["montecarlo"] * m_result.cvar_95
            ),
            cvar_99=consensus_cvar99,
            method="consensus",
        )

        # Breach detection
        breach = False
        if actual_daily_pnl is not None and self._last_consensus_var99 > 0:
            loss = -actual_daily_pnl  # positive = loss
            breach = loss > self._last_consensus_var99
            self._breach_history.append((loss, self._last_consensus_var99))
            if breach:
                log.warning(
                    "VaR BREACH: actual loss %.2f > VaR99 %.2f",
                    loss, self._last_consensus_var99,
                )

        self._last_consensus_var99 = consensus_var99

        # Persist to database
        results = {
            "parametric": p_result,
            "historical": h_result,
            "montecarlo": m_result,
            "consensus": consensus,
        }
        self._persist(results, snapshot, breach)

        # Run Kupiec test periodically
        if len(self._breach_history) >= 60:
            self._run_kupiec(CONF_99, "consensus")

        return results

    # ------------------------------------------------------------------ #
    # Kupiec proportional-of-failures LR test                             #
    # ------------------------------------------------------------------ #

    def _run_kupiec(self, confidence: float, method: str) -> None:
        """
        Kupiec LR test: checks whether the observed exception rate matches
        the expected rate (1 - confidence).

        H0: p_hat == 1 - confidence
        """
        history = self._breach_history[-252:]  # use last year
        n = len(history)
        if n < 30:
            return
        n_exceptions = sum(1 for loss, var99 in history if loss > var99)
        p_hat = n_exceptions / n
        p_expected = 1.0 - confidence

        if p_hat <= 0:
            lr_stat = 0.0
        elif p_hat >= 1:
            lr_stat = 1e6
        else:
            try:
                lr_stat = -2 * (
                    n_exceptions * math.log(p_expected / p_hat)
                    + (n - n_exceptions) * math.log((1 - p_expected) / (1 - p_hat))
                )
            except (ValueError, ZeroDivisionError):
                lr_stat = 0.0

        p_value = float(stats.chi2.sf(lr_stat, df=1))
        passed = int(p_value > 0.05)

        log.info(
            "Kupiec test (%s, %.0f%%): n=%d excep=%d LR=%.3f p=%.4f %s",
            method, confidence * 100, n, n_exceptions, lr_stat, p_value,
            "PASS" if passed else "FAIL",
        )

        ts = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO var_backtests
                       (timestamp, method, confidence, n_obs, n_exceptions,
                        kupiec_stat, kupiec_pvalue, pass_flag)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (ts, method, confidence, n, n_exceptions, lr_stat, p_value, passed),
                )
                conn.commit()
        except sqlite3.Error as exc:
            log.error("Failed to persist Kupiec result: %s", exc)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _persist(
        self,
        results: Dict[str, VaRResult],
        snapshot: PortfolioSnapshot,
        breach: bool,
    ) -> None:
        ts = snapshot.timestamp.isoformat()
        rows = []
        consensus_var99 = results["consensus"].var_99
        for method, r in results.items():
            rows.append((
                ts, method, r.var_95, r.var_99, r.cvar_95, r.cvar_99,
                consensus_var99, snapshot.equity, len(snapshot.positions),
                int(breach),
            ))
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """INSERT INTO risk_metrics
                       (timestamp, method, var_95, var_99, cvar_95, cvar_99,
                        consensus_var99, equity, n_positions, breach_flag)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
                conn.commit()
        except sqlite3.Error as exc:
            log.error("Failed to persist risk metrics: %s", exc)

    # ------------------------------------------------------------------ #
    # Convenience: read latest metrics from DB                            #
    # ------------------------------------------------------------------ #

    def latest_metrics(self, n_rows: int = 4) -> pd.DataFrame:
        """Return the most recent VaR rows from risk_metrics table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(
                    f"""SELECT * FROM risk_metrics
                        ORDER BY id DESC LIMIT {n_rows}""",
                    conn,
                )
        except Exception as exc:
            log.error("Failed to read risk_metrics: %s", exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Utility: build PortfolioSnapshot from live_trades.db
# ---------------------------------------------------------------------------

def snapshot_from_db(
    db_path: Path = _DB_PATH,
    equity: float = 100_000.0,
) -> PortfolioSnapshot:
    """
    Construct a PortfolioSnapshot from open positions in live_trades.db.

    Positions are reconstructed by summing buy/sell fills per symbol.
    Current price is approximated from the last fill price (no live feed here).
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            df = pd.read_sql_query(
                """SELECT symbol, side, qty, price
                   FROM live_trades
                   ORDER BY fill_time ASC""",
                conn,
            )
    except Exception as exc:
        log.error("snapshot_from_db failed: %s", exc)
        return PortfolioSnapshot(positions=[], equity=equity)

    if df.empty:
        return PortfolioSnapshot(positions=[], equity=equity)

    positions = []
    for sym, grp in df.groupby("symbol"):
        net_qty = 0.0
        avg_entry = 0.0
        last_price = 0.0
        for _, row in grp.iterrows():
            side_sign = 1.0 if row["side"] in ("buy", "long") else -1.0
            net_qty += side_sign * row["qty"]
            last_price = row["price"]
        if abs(net_qty) < 1e-9:
            continue
        # Use last fill as entry (simplification; proper FIFO would differ)
        avg_entry = last_price
        positions.append(PositionSnapshot(
            symbol=str(sym),
            qty=net_qty,
            entry_price=avg_entry,
            current_price=last_price,
        ))

    return PortfolioSnapshot(positions=positions, equity=equity)
