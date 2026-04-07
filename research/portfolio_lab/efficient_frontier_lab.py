"""
research/portfolio_lab/efficient_frontier_lab.py

Interactive efficient frontier analysis for SRFM-Lab.
Supports long-only SLSQP optimization, Monte Carlo simulation,
and bootstrap confidence bands.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PortfolioPoint:
    """Single portfolio on the efficient frontier."""

    weights: np.ndarray  # (n_assets,) allocation vector
    expected_return: float
    volatility: float
    sharpe: float
    label: Optional[str] = None

    def as_series(self, asset_labels: List[str]) -> pd.Series:
        """Return weights as a labelled Series."""
        s = pd.Series(self.weights, index=asset_labels, name=self.label or "portfolio")
        s["expected_return"] = self.expected_return
        s["volatility"] = self.volatility
        s["sharpe"] = self.sharpe
        return s


@dataclass
class FrontierResult:
    """Full efficient frontier with metadata."""

    weights: np.ndarray  # (n_points, n_assets)
    returns: np.ndarray  # (n_points,)
    volatilities: np.ndarray  # (n_points,)
    sharpes: np.ndarray  # (n_points,)
    max_sharpe_idx: int
    min_var_idx: int
    asset_labels: List[str]
    rf_rate: float = 0.0
    # bootstrap confidence bands (populated when compute_bootstrap=True)
    vol_lower: Optional[np.ndarray] = None  # 5th percentile band
    vol_upper: Optional[np.ndarray] = None  # 95th percentile band

    # ------------------------------------------------------------------
    def max_sharpe_point(self) -> PortfolioPoint:
        i = self.max_sharpe_idx
        return PortfolioPoint(
            weights=self.weights[i],
            expected_return=float(self.returns[i]),
            volatility=float(self.volatilities[i]),
            sharpe=float(self.sharpes[i]),
            label="max_sharpe",
        )

    def min_variance_point(self) -> PortfolioPoint:
        i = self.min_var_idx
        return PortfolioPoint(
            weights=self.weights[i],
            expected_return=float(self.returns[i]),
            volatility=float(self.volatilities[i]),
            sharpe=float(self.sharpes[i]),
            label="min_variance",
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert frontier to tidy DataFrame."""
        df = pd.DataFrame(self.weights, columns=self.asset_labels)
        df["expected_return"] = self.returns
        df["volatility"] = self.volatilities
        df["sharpe"] = self.sharpes
        if self.vol_lower is not None:
            df["vol_lower_5pct"] = self.vol_lower
        if self.vol_upper is not None:
            df["vol_upper_95pct"] = self.vol_upper
        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _port_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _port_vol(w: np.ndarray, cov: np.ndarray) -> float:
    var = float(w @ cov @ w)
    return float(np.sqrt(max(var, 0.0)))


def _port_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    vol = _port_vol(w, cov)
    if vol < 1e-12:
        return 0.0
    return float((_port_return(w, mu) - rf) / vol)


def _build_constraints(n_assets: int, target_return: Optional[float], mu: np.ndarray) -> list:
    """Build SLSQP constraint dicts."""
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w, r=target_return: _port_return(w, mu) - r}
        )
    return constraints


def _bounds(n_assets: int) -> tuple:
    """Long-only bounds."""
    return tuple((0.0, 1.0) for _ in range(n_assets))


def _initial_weights(n_assets: int) -> np.ndarray:
    """Equal weight starting point."""
    return np.ones(n_assets) / n_assets


def _minimize_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: Optional[float] = None,
    w0: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Minimize portfolio variance subject to optional return target."""
    n = len(mu)
    if w0 is None:
        w0 = _initial_weights(n)

    def neg_vol(w: np.ndarray) -> float:
        return float(w @ cov @ w)  # minimize variance directly

    result: OptimizeResult = minimize(
        neg_vol,
        w0,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_build_constraints(n, target_return, mu),
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )
    if result.success or result.fun < 1e-6:
        w = np.clip(result.x, 0.0, 1.0)
        w /= w.sum()
        return w
    return None


def _maximize_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.0,
    w0: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Maximize Sharpe ratio (long-only)."""
    n = len(mu)
    if w0 is None:
        w0 = _initial_weights(n)

    def neg_sharpe(w: np.ndarray) -> float:
        return -_port_sharpe(w, mu, cov, rf)

    result: OptimizeResult = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=_bounds(n),
        constraints=_build_constraints(n, None, mu),
        options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
    )
    if result.success:
        w = np.clip(result.x, 0.0, 1.0)
        w /= w.sum()
        return w
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EfficientFrontierLab:
    """
    Interactive efficient frontier analysis.

    Usage
    -----
    lab = EfficientFrontierLab()
    lab.add_asset("SPY", spy_returns)
    lab.add_asset("AGG", agg_returns)
    result = lab.compute_frontier(n_points=100)
    """

    def __init__(self, annualise: bool = True, trading_days: int = 252) -> None:
        self._symbols: List[str] = []
        self._labels: List[str] = []
        self._returns: List[pd.Series] = []
        self._annualise = annualise
        self._trading_days = trading_days

    # ------------------------------------------------------------------
    # Asset management
    # ------------------------------------------------------------------

    def add_asset(
        self,
        symbol: str,
        returns: pd.Series,
        label: Optional[str] = None,
    ) -> None:
        """Register an asset return series."""
        if symbol in self._symbols:
            raise ValueError(f"Asset '{symbol}' already added")
        self._symbols.append(symbol)
        self._labels.append(label if label is not None else symbol)
        self._returns.append(returns.copy())

    def _build_returns_matrix(self) -> pd.DataFrame:
        """Align all return series on a common index."""
        if len(self._returns) < 2:
            raise ValueError("Need at least 2 assets to build a frontier")
        df = pd.concat(self._returns, axis=1, keys=self._labels)
        df = df.dropna()
        if len(df) < 30:
            raise ValueError(f"Only {len(df)} aligned observations -- need at least 30")
        return df

    def _stats(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return annualised mean vector and covariance matrix."""
        scale = self._trading_days if self._annualise else 1
        mu = df.mean().values * scale
        cov = df.cov().values * scale
        return mu, cov

    # ------------------------------------------------------------------
    # Frontier computation
    # ------------------------------------------------------------------

    def compute_frontier(
        self,
        n_points: int = 100,
        rf_rate: float = 0.0,
        compute_bootstrap: bool = True,
        n_bootstrap: int = 500,
    ) -> FrontierResult:
        """
        Compute the mean-variance efficient frontier.

        Parameters
        ----------
        n_points : number of target-return grid points
        rf_rate : risk-free rate (annualised if annualise=True)
        compute_bootstrap : whether to add 5/95 pct confidence bands
        n_bootstrap : number of bootstrap resamples for confidence bands
        """
        df = self._build_returns_matrix()
        mu, cov = self._stats(df)
        n_assets = len(mu)

        # find feasible return range
        w_min = _minimize_variance(mu, cov)
        if w_min is None:
            raise RuntimeError("Min-variance optimisation failed")

        ret_min = _port_return(w_min, mu)
        ret_max = float(np.max(mu))  # max individual asset return

        target_returns = np.linspace(ret_min, ret_max, n_points)

        weights_list: List[np.ndarray] = []
        vols: List[float] = []
        rets: List[float] = []

        prev_w = w_min.copy()
        for target in target_returns:
            w = _minimize_variance(mu, cov, target_return=target, w0=prev_w)
            if w is None:
                # fallback to equal weight for failed points
                w = _initial_weights(n_assets)
            weights_list.append(w)
            vols.append(_port_vol(w, cov))
            rets.append(_port_return(w, mu))
            prev_w = w.copy()

        weights_arr = np.array(weights_list)
        vols_arr = np.array(vols)
        rets_arr = np.array(rets)
        sharpes_arr = np.array(
            [_port_sharpe(w, mu, cov, rf_rate) for w in weights_list]
        )

        max_sharpe_idx = int(np.argmax(sharpes_arr))
        min_var_idx = int(np.argmin(vols_arr))

        vol_lower: Optional[np.ndarray] = None
        vol_upper: Optional[np.ndarray] = None

        if compute_bootstrap:
            vol_lower, vol_upper = self._bootstrap_bands(
                df, target_returns, n_bootstrap
            )

        return FrontierResult(
            weights=weights_arr,
            returns=rets_arr,
            volatilities=vols_arr,
            sharpes=sharpes_arr,
            max_sharpe_idx=max_sharpe_idx,
            min_var_idx=min_var_idx,
            asset_labels=self._labels.copy(),
            rf_rate=rf_rate,
            vol_lower=vol_lower,
            vol_upper=vol_upper,
        )

    # ------------------------------------------------------------------
    # Special portfolio points
    # ------------------------------------------------------------------

    def find_max_sharpe(self, rf_rate: float = 0.0) -> PortfolioPoint:
        """Find the tangency (max Sharpe) portfolio."""
        df = self._build_returns_matrix()
        mu, cov = self._stats(df)

        # try multiple starting points for robustness
        best_w: Optional[np.ndarray] = None
        best_sharpe = -np.inf
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.dirichlet(np.ones(len(mu)))
            w = _maximize_sharpe(mu, cov, rf=rf_rate, w0=w0)
            if w is not None:
                s = _port_sharpe(w, mu, cov, rf_rate)
                if s > best_sharpe:
                    best_sharpe = s
                    best_w = w.copy()

        if best_w is None:
            raise RuntimeError("Max-Sharpe optimisation failed")

        return PortfolioPoint(
            weights=best_w,
            expected_return=_port_return(best_w, mu),
            volatility=_port_vol(best_w, cov),
            sharpe=best_sharpe,
            label="max_sharpe",
        )

    def find_min_variance(self) -> PortfolioPoint:
        """Find the global minimum variance portfolio."""
        df = self._build_returns_matrix()
        mu, cov = self._stats(df)
        w = _minimize_variance(mu, cov)
        if w is None:
            raise RuntimeError("Min-variance optimisation failed")
        return PortfolioPoint(
            weights=w,
            expected_return=_port_return(w, mu),
            volatility=_port_vol(w, cov),
            sharpe=_port_sharpe(w, mu, cov, 0.0),
            label="min_variance",
        )

    def find_risk_parity(self) -> PortfolioPoint:
        """
        Find the risk-parity portfolio: each asset contributes
        equally to total portfolio volatility.
        """
        df = self._build_returns_matrix()
        mu, cov = self._stats(df)
        n = len(mu)

        def risk_parity_objective(w: np.ndarray) -> float:
            sigma = _port_vol(w, cov)
            if sigma < 1e-12:
                return 0.0
            # marginal risk contribution
            mrc = (cov @ w) / sigma
            rc = w * mrc  # risk contributions
            target_rc = sigma / n
            return float(np.sum((rc - target_rc) ** 2))

        best_w: Optional[np.ndarray] = None
        best_obj = np.inf
        rng = np.random.default_rng(99)
        for _ in range(15):
            w0 = rng.dirichlet(np.ones(n))
            result = minimize(
                risk_parity_objective,
                w0,
                method="SLSQP",
                bounds=_bounds(n),
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
            )
            if result.fun < best_obj:
                best_obj = result.fun
                w = np.clip(result.x, 0.0, 1.0)
                w /= w.sum()
                best_w = w.copy()

        if best_w is None:
            best_w = _initial_weights(n)

        return PortfolioPoint(
            weights=best_w,
            expected_return=_port_return(best_w, mu),
            volatility=_port_vol(best_w, cov),
            sharpe=_port_sharpe(best_w, mu, cov, 0.0),
            label="risk_parity",
        )

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def random_portfolios(self, n: int = 10000, rf_rate: float = 0.0) -> pd.DataFrame:
        """
        Simulate n random (Dirichlet) portfolios.

        Returns a DataFrame with columns:
        [*asset_labels, expected_return, volatility, sharpe]
        """
        df = self._build_returns_matrix()
        mu, cov = self._stats(df)
        n_assets = len(mu)

        rng = np.random.default_rng(0)
        weights_mc = rng.dirichlet(np.ones(n_assets), size=n)

        port_returns = weights_mc @ mu
        # vectorised variance: sum_i sum_j w_i * w_j * cov_ij
        port_vars = np.einsum("ni,ij,nj->n", weights_mc, cov, weights_mc)
        port_vols = np.sqrt(np.maximum(port_vars, 0.0))
        port_sharpes = np.where(
            port_vols > 1e-12, (port_returns - rf_rate) / port_vols, 0.0
        )

        out = pd.DataFrame(weights_mc, columns=self._labels)
        out["expected_return"] = port_returns
        out["volatility"] = port_vols
        out["sharpe"] = port_sharpes
        return out

    # ------------------------------------------------------------------
    # Bootstrap confidence bands
    # ------------------------------------------------------------------

    def _bootstrap_bands(
        self,
        df: pd.DataFrame,
        target_returns: np.ndarray,
        n_bootstrap: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample returns 500x, recompute the frontier for each resample,
        return 5th and 95th percentile volatility bands.
        """
        n_obs = len(df)
        n_points = len(target_returns)
        all_vols = np.full((n_bootstrap, n_points), np.nan)

        rng = np.random.default_rng(7)
        scale = self._trading_days if self._annualise else 1

        for b in range(n_bootstrap):
            idx = rng.integers(0, n_obs, size=n_obs)
            sample = df.iloc[idx]
            mu_b = sample.mean().values * scale
            cov_b = sample.cov().values * scale

            # find feasible return range for this bootstrap sample
            w_min_b = _minimize_variance(mu_b, cov_b)
            if w_min_b is None:
                continue
            ret_min_b = _port_return(w_min_b, mu_b)
            ret_max_b = float(np.max(mu_b))

            prev_w = w_min_b.copy()
            for j, t in enumerate(target_returns):
                # clip target to feasible range for this sample
                t_clipped = float(np.clip(t, ret_min_b, ret_max_b))
                w = _minimize_variance(mu_b, cov_b, target_return=t_clipped, w0=prev_w)
                if w is not None:
                    all_vols[b, j] = _port_vol(w, cov_b)
                    prev_w = w.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vol_lower = np.nanpercentile(all_vols, 5, axis=0)
            vol_upper = np.nanpercentile(all_vols, 95, axis=0)

        return vol_lower, vol_upper

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        return len(self._symbols)

    @property
    def asset_labels(self) -> List[str]:
        return self._labels.copy()

    def summary(self) -> pd.DataFrame:
        """Individual asset annualised stats."""
        if not self._returns:
            return pd.DataFrame()
        df = self._build_returns_matrix()
        scale = self._trading_days if self._annualise else 1
        rows = []
        for lbl in self._labels:
            s = df[lbl]
            rows.append(
                {
                    "label": lbl,
                    "ann_return": s.mean() * scale,
                    "ann_vol": s.std() * np.sqrt(scale),
                    "sharpe": (s.mean() * scale) / (s.std() * np.sqrt(scale))
                    if s.std() > 0
                    else 0.0,
                    "n_obs": len(s),
                }
            )
        return pd.DataFrame(rows).set_index("label")
