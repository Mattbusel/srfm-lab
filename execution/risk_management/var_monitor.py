"""
var_monitor.py # Real-time VaR monitoring for SRFM.

Three VaR estimation methods:
  1. Historical simulation  # last 252 days, 99th percentile
  2. Parametric             # covariance matrix, normality assumption
  3. Monte Carlo            # 1,000 paths over 252-day covariance

Incremental covariance updates use EWMA with lambda=0.94 (RiskMetrics standard).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm  # type: ignore[import]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EWMA_LAMBDA: float = 0.94
HIST_SIM_WINDOW: int = 252          # trading days for historical simulation
MC_PATHS: int = 1_000               # Monte Carlo path count
ANNUALISATION_FACTOR: int = 252     # trading days per year


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VaREstimate:
    """Result from a single VaR computation."""
    method: str                     # historical | parametric | monte_carlo
    confidence: float               # e.g. 0.99
    horizon_days: int
    var: float                      # dollar VaR (positive = loss)
    cvar: float                     # Expected Shortfall (dollar)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ComponentVaR:
    """Per-symbol contribution to portfolio VaR."""
    symbol: str
    dollar_exposure: float
    component_var: float
    pct_of_total_var: float


@dataclass
class VaRBreach:
    """Emitted when portfolio VaR exceeds its limit."""
    portfolio_var: float
    limit: float
    confidence: float
    method: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# EWMA covariance tracker
# ---------------------------------------------------------------------------

class EWMACovarianceMatrix:
    """
    Maintains an incremental EWMA covariance matrix for N assets.

    Covariance update:
        sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t * r_t^T

    This is the RiskMetrics (1994) exponentially weighted covariance model.
    """

    def __init__(self, symbols: List[str], lam: float = EWMA_LAMBDA) -> None:
        self._symbols = list(symbols)
        self._index: Dict[str, int] = {s: i for i, s in enumerate(symbols)}
        self._lam = lam
        n = len(symbols)
        self._cov: np.ndarray = np.eye(n) * 1e-4   # small initial variance
        self._initialized: bool = False

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    @property
    def n(self) -> int:
        return len(self._symbols)

    def add_symbol(self, symbol: str) -> None:
        """Expand the covariance matrix to include a new symbol."""
        if symbol in self._index:
            return
        idx = len(self._symbols)
        self._symbols.append(symbol)
        self._index[symbol] = idx
        n = len(self._symbols)
        new_cov = np.zeros((n, n))
        new_cov[: n - 1, : n - 1] = self._cov
        new_cov[n - 1, n - 1] = 1e-4
        self._cov = new_cov

    def update(self, returns: Dict[str, float]) -> None:
        """
        Perform one EWMA update step given a dict of symbol returns.
        Missing symbols are treated as 0 return for this step.
        """
        n = self.n
        r_vec = np.zeros(n)
        for sym, ret in returns.items():
            if sym in self._index:
                r_vec[self._index[sym]] = ret

        outer = np.outer(r_vec, r_vec)
        self._cov = self._lam * self._cov + (1.0 - self._lam) * outer
        self._initialized = True

    def correlation_matrix(self) -> np.ndarray:
        """Convert covariance to correlation matrix."""
        std = np.sqrt(np.diag(self._cov))
        std = np.where(std < 1e-10, 1e-10, std)
        return self._cov / np.outer(std, std)

    def covariance_matrix(self) -> np.ndarray:
        return self._cov.copy()

    def volatility(self, symbol: str) -> float:
        """Daily volatility for a single symbol."""
        idx = self._index.get(symbol)
        if idx is None:
            return 0.0
        return math.sqrt(max(self._cov[idx, idx], 0.0))

    def is_initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# VaR computation functions
# ---------------------------------------------------------------------------

def historical_var(
    portfolio_returns: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> Tuple[float, float]:
    """
    Historical simulation VaR and CVaR.

    Args:
        portfolio_returns: Array of daily P&L (in dollars, negative = loss).
        confidence:        Confidence level (e.g. 0.99).
        horizon_days:      Scaling horizon (square root of time).

    Returns:
        (var, cvar) as positive dollar amounts representing losses.
    """
    if len(portfolio_returns) == 0:
        return 0.0, 0.0

    losses = -portfolio_returns  # flip sign so losses are positive
    losses_sorted = np.sort(losses)
    cutoff_idx = int(np.ceil(confidence * len(losses_sorted))) - 1
    cutoff_idx = max(0, min(cutoff_idx, len(losses_sorted) - 1))

    var = losses_sorted[cutoff_idx] * math.sqrt(horizon_days)
    tail_losses = losses_sorted[cutoff_idx:]
    cvar = tail_losses.mean() * math.sqrt(horizon_days) if len(tail_losses) > 0 else var

    return float(max(var, 0.0)), float(max(cvar, 0.0))


def parametric_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> Tuple[float, float]:
    """
    Parametric (variance-covariance) VaR and CVaR under normality assumption.

    Args:
        weights:         Array of portfolio weights (fractions summing to ~1).
        cov_matrix:      Daily covariance matrix of returns.
        portfolio_value: Total dollar value of portfolio.
        confidence:      Confidence level.
        horizon_days:    Scaling horizon.

    Returns:
        (var, cvar) as positive dollar amounts.
    """
    if weights.shape[0] == 0 or cov_matrix.shape[0] == 0:
        return 0.0, 0.0

    port_variance = float(weights @ cov_matrix @ weights)
    port_std = math.sqrt(max(port_variance, 0.0)) * math.sqrt(horizon_days)

    z = norm.ppf(confidence)
    var = z * port_std * portfolio_value

    # CVaR = phi(z) / (1 - confidence) * sigma * portfolio_value
    cvar = (norm.pdf(z) / (1.0 - confidence)) * port_std * portfolio_value

    return float(max(var, 0.0)), float(max(cvar, 0.0))


def monte_carlo_var(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.99,
    horizon_days: int = 1,
    n_paths: int = MC_PATHS,
    rng_seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Monte Carlo VaR and CVaR using Cholesky decomposition.

    Args:
        weights:         Portfolio weight vector.
        cov_matrix:      Daily covariance matrix.
        portfolio_value: Total dollar value.
        confidence:      Confidence level.
        horizon_days:    Number of days for multi-day VaR (Brownian bridge).
        n_paths:         Number of Monte Carlo paths.
        rng_seed:        Optional RNG seed for reproducibility.

    Returns:
        (var, cvar) as positive dollar amounts.
    """
    if weights.shape[0] == 0 or cov_matrix.shape[0] == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(rng_seed)

    try:
        chol = np.linalg.cholesky(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-10)
    except np.linalg.LinAlgError:
        logger.warning("Cholesky decomposition failed # using diagonal approximation")
        chol = np.diag(np.sqrt(np.maximum(np.diag(cov_matrix), 0.0)))

    # Simulate horizon_days cumulative returns
    z = rng.standard_normal((n_paths, len(weights), horizon_days))
    # Shape: (n_paths, n_assets, horizon_days)
    correlated = np.einsum("ij,klj->kli", chol, z)
    cumulative_returns = correlated.sum(axis=2)  # sum over days
    port_returns = cumulative_returns @ weights   # (n_paths,)

    losses = -port_returns * portfolio_value
    losses_sorted = np.sort(losses)
    cutoff_idx = int(np.ceil(confidence * n_paths)) - 1
    cutoff_idx = max(0, min(cutoff_idx, n_paths - 1))

    var = float(losses_sorted[cutoff_idx])
    tail = losses_sorted[cutoff_idx:]
    cvar = float(tail.mean()) if len(tail) > 0 else var

    return max(var, 0.0), max(cvar, 0.0)


# ---------------------------------------------------------------------------
# VaR Monitor
# ---------------------------------------------------------------------------

class VaRMonitor:
    """
    Maintains incremental position and return data, computes portfolio VaR
    using three methods, and exposes per-symbol component VaR.
    """

    def __init__(
        self,
        nav: float,
        ewma_lambda: float = EWMA_LAMBDA,
        hist_window: int = HIST_SIM_WINDOW,
    ) -> None:
        self._nav = nav
        self._positions: Dict[str, float] = {}      # symbol -> dollar_value
        self._return_history: Dict[str, Deque[float]] = {}
        self._portfolio_pnl_history: Deque[float] = deque(maxlen=hist_window)
        self._hist_window = hist_window
        self._cov = EWMACovarianceMatrix([], lam=ewma_lambda)

        # Breach state
        self._last_var_estimate: Optional[VaREstimate] = None

    # # Data ingestion ------------------------------------------------------

    def update_positions(self, positions: Dict[str, float]) -> None:
        """
        Update position book.  Values are dollar exposures
        (positive = long, negative = short).
        """
        self._positions = dict(positions)
        # Ensure covariance matrix has all symbols
        for sym in positions:
            if sym not in self._cov.symbols:
                self._cov.add_symbol(sym)
            if sym not in self._return_history:
                self._return_history[sym] = deque(maxlen=self._hist_window)

    def update_nav(self, nav: float) -> None:
        self._nav = nav

    def update_returns(self, symbol: str, ret: float) -> None:
        """
        Incremental single-symbol return update.
        Call once per bar per symbol.  Triggers EWMA covariance update.
        """
        if symbol not in self._cov.symbols:
            self._cov.add_symbol(symbol)
        if symbol not in self._return_history:
            self._return_history[symbol] = deque(maxlen=self._hist_window)

        self._return_history[symbol].append(ret)
        self._cov.update({symbol: ret})

    def update_returns_batch(self, returns: Dict[str, float]) -> None:
        """
        Update multiple symbols at once (one bar).  Also computes portfolio
        P&L for historical simulation.
        """
        for sym, ret in returns.items():
            self.update_returns(sym, ret)

        # Portfolio P&L for this bar
        port_pnl = sum(
            self._positions.get(sym, 0.0) * ret for sym, ret in returns.items()
        )
        self._portfolio_pnl_history.append(port_pnl)

    # # VaR computation -----------------------------------------------------

    def portfolio_var(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        method: str = "historical",
    ) -> float:
        """Dollar VaR of the current portfolio."""
        estimate = self._compute_var(confidence, horizon_days, method)
        self._last_var_estimate = estimate
        return estimate.var

    def portfolio_cvar(
        self,
        confidence: float = 0.99,
        horizon_days: int = 1,
        method: str = "historical",
    ) -> float:
        """Expected Shortfall (CVaR) of the current portfolio."""
        estimate = self._compute_var(confidence, horizon_days, method)
        return estimate.cvar

    def _compute_var(
        self,
        confidence: float,
        horizon_days: int,
        method: str,
    ) -> VaREstimate:
        if method == "historical":
            pnl_arr = np.array(list(self._portfolio_pnl_history))
            var, cvar = historical_var(pnl_arr, confidence, horizon_days)
        elif method == "parametric":
            weights, port_value = self._compute_weights()
            if weights is None:
                return VaREstimate("parametric", confidence, horizon_days, 0.0, 0.0)
            cov = self._cov.covariance_matrix()
            if cov.shape[0] != len(weights):
                return VaREstimate("parametric", confidence, horizon_days, 0.0, 0.0)
            var, cvar = parametric_var(weights, cov, port_value, confidence, horizon_days)
        elif method == "monte_carlo":
            weights, port_value = self._compute_weights()
            if weights is None:
                return VaREstimate("monte_carlo", confidence, horizon_days, 0.0, 0.0)
            cov = self._cov.covariance_matrix()
            if cov.shape[0] != len(weights):
                return VaREstimate("monte_carlo", confidence, horizon_days, 0.0, 0.0)
            var, cvar = monte_carlo_var(weights, cov, port_value, confidence, horizon_days)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return VaREstimate(method, confidence, horizon_days, var, cvar)

    def _compute_weights(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Compute weight vector aligned with the EWMA covariance matrix symbol order.
        Returns (weights_array, portfolio_dollar_value).
        """
        symbols = self._cov.symbols
        if not symbols:
            return None, 0.0

        port_value = sum(abs(v) for v in self._positions.values())
        if port_value <= 0:
            return None, 0.0

        weights = np.array([
            self._positions.get(s, 0.0) / port_value for s in symbols
        ])
        return weights, port_value

    def component_var(
        self,
        symbol: str,
        confidence: float = 0.99,
        horizon_days: int = 1,
    ) -> float:
        """
        Marginal component VaR for a single symbol using the parametric method.
        Component VaR = w_i * (Sigma * w)_i / sigma_p * z * portfolio_value
        """
        weights, port_value = self._compute_weights()
        if weights is None:
            return 0.0

        symbols = self._cov.symbols
        if symbol not in symbols:
            return 0.0

        cov = self._cov.covariance_matrix()
        port_variance = float(weights @ cov @ weights)
        port_std = math.sqrt(max(port_variance, 1e-20)) * math.sqrt(horizon_days)

        sym_idx = symbols.index(symbol)
        marginal = float((cov @ weights)[sym_idx])
        w_i = weights[sym_idx]

        z = norm.ppf(confidence)
        component = w_i * marginal / max(port_std ** 2, 1e-20) * z * port_value * port_std

        return max(component, 0.0)

    def component_var_all(
        self,
        confidence: float = 0.99,
    ) -> List[ComponentVaR]:
        """Component VaR breakdown for all positions."""
        results = []
        total_var = self.portfolio_var(confidence)

        for sym in list(self._positions.keys()):
            cv = self.component_var(sym, confidence)
            results.append(
                ComponentVaR(
                    symbol=sym,
                    dollar_exposure=self._positions[sym],
                    component_var=cv,
                    pct_of_total_var=cv / total_var if total_var > 0 else 0.0,
                )
            )
        results.sort(key=lambda x: x.component_var, reverse=True)
        return results

    def is_var_breach(
        self,
        limit: float,
        confidence: float = 0.99,
        method: str = "historical",
    ) -> bool:
        """True when portfolio VaR exceeds `limit` (in dollars)."""
        var = self.portfolio_var(confidence, method=method)
        return var > limit

    def var_utilization(self, limit: float, confidence: float = 0.99) -> float:
        """VaR as a fraction of the limit (0.0-1.0+)."""
        var = self.portfolio_var(confidence)
        if limit <= 0:
            return 0.0
        return var / limit

    def volatility(self, symbol: str) -> float:
        """Annualized volatility for a symbol from EWMA covariance."""
        return self._cov.volatility(symbol) * math.sqrt(ANNUALISATION_FACTOR)


# ---------------------------------------------------------------------------
# VaR limit manager
# ---------------------------------------------------------------------------

class VaRLimitManager:
    """
    Enforces per-symbol and portfolio-level VaR limits.
    Routes alerts via configurable callback.
    """

    def __init__(
        self,
        monitor: VaRMonitor,
        portfolio_var_limit: float,           # e.g. 0.02 * NAV
        confidence: float = 0.99,
        alert_fn: Optional[Callable[[VaRBreach], None]] = None,
    ) -> None:
        self._monitor = monitor
        self._portfolio_limit = portfolio_var_limit
        self._confidence = confidence
        self._symbol_limits: Dict[str, float] = {}
        self._alert_fn = alert_fn
        self._breaches: List[VaRBreach] = []

    def set_symbol_limit(self, symbol: str, limit: float) -> None:
        """Set per-symbol VaR limit."""
        self._symbol_limits[symbol] = limit

    def set_portfolio_limit(self, limit: float) -> None:
        self._portfolio_limit = limit

    def check_all(self, method: str = "historical") -> List[VaRBreach]:
        """
        Run all VaR limit checks.  Returns list of active breaches.
        """
        new_breaches: List[VaRBreach] = []

        # Portfolio-level check
        port_var = self._monitor.portfolio_var(self._confidence, method=method)
        if port_var > self._portfolio_limit:
            breach = VaRBreach(
                portfolio_var=port_var,
                limit=self._portfolio_limit,
                confidence=self._confidence,
                method=method,
                timestamp=datetime.now(timezone.utc),
            )
            new_breaches.append(breach)
            self._dispatch_alert(breach)

        # Per-symbol checks
        for sym, limit in self._symbol_limits.items():
            cv = self._monitor.component_var(sym, self._confidence)
            if cv > limit:
                breach = VaRBreach(
                    portfolio_var=cv,
                    limit=limit,
                    confidence=self._confidence,
                    method=f"component_{method}",
                    timestamp=datetime.now(timezone.utc),
                )
                new_breaches.append(breach)
                self._dispatch_alert(breach)

        self._breaches.extend(new_breaches)
        return new_breaches

    def _dispatch_alert(self, breach: VaRBreach) -> None:
        if self._alert_fn is not None:
            try:
                self._alert_fn(breach)
            except Exception:
                logger.exception("VaR alert function raised an exception")

    def recent_breaches(self, n: int = 20) -> List[VaRBreach]:
        return list(reversed(self._breaches[-n:]))

    def is_portfolio_breached(self, method: str = "historical") -> bool:
        return self._monitor.is_var_breach(
            self._portfolio_limit, self._confidence, method
        )

    def portfolio_var_utilization(self, method: str = "historical") -> float:
        return self._monitor.var_utilization(self._portfolio_limit, self._confidence)
