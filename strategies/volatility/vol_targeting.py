"""
vol_targeting.py — Volatility targeting and risk parity strategies.

References:
  - Hurst, Ooi, Pedersen (2012): Risk parity portfolios
  - Asness, Frazzini, Pedersen (2012): Leverage aversion and risk parity
  - Choueifaty & Coignard (2008): Maximum diversification
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")


@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_leverage: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    weights_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"AvgLev={self.avg_leverage:.2f}")


def _stats(ec: np.ndarray, trades: list = None) -> dict:
    n = len(ec)
    tot = ec[-1] / ec[0] - 1
    cagr = (ec[-1] / ec[0]) ** (1 / max(1, n / 252)) - 1
    r = np.diff(ec) / (ec[:-1] + 1e-9)
    r = np.concatenate([[0], r])
    std = r.std()
    sh = r.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / (np.std(down) + 1e-9) * math.sqrt(252)
    pk = np.maximum.accumulate(ec)
    dd = (ec - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    wins = [x for x in (trades or []) if x > 0]
    losses = [x for x in (trades or []) if x <= 0]
    wr = len(wins) / len(trades) if trades else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(total_return=tot, cagr=cagr, sharpe=sh, sortino=sortino,
                max_drawdown=mdd, calmar=calmar, win_rate=wr, profit_factor=pf,
                n_trades=len(trades or []),
                avg_trade_return=float(np.mean(trades)) if trades else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. VolatilityTargeting
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityTargeting:
    """
    Volatility targeting: scale position size to maintain constant realized vol.

    The strategy adjusts leverage each period so that the portfolio's
    expected volatility equals target_vol.

    leverage = target_vol / realized_vol

    This produces risk-stable returns and naturally reduces exposure
    in high-vol regimes (risk-off) and increases in low-vol (risk-on).

    Parameters
    ----------
    target_vol  : target annualized volatility (default 0.10 = 10%)
    lookback    : window for vol estimation (default 21)
    max_lev     : maximum leverage (default 2.0)
    min_lev     : minimum leverage (default 0.0)
    vol_method  : "ewm" (exponential) or "rolling" (default "ewm")
    halflife    : halflife for EWM vol (default 21)
    """

    def __init__(
        self,
        target_vol: float = 0.10,
        lookback: int = 21,
        max_lev: float = 2.0,
        min_lev: float = 0.0,
        vol_method: str = "ewm",
        halflife: int = 21,
    ):
        self.target_vol = target_vol
        self.lookback = lookback
        self.max_lev = max_lev
        self.min_lev = min_lev
        self.vol_method = vol_method
        self.halflife = halflife

    def estimate_vol(self, returns: pd.Series) -> pd.Series:
        """Estimate annualized realized volatility."""
        if self.vol_method == "ewm":
            vol = returns.ewm(halflife=self.halflife, min_periods=5).std() * math.sqrt(252)
        else:
            vol = returns.rolling(self.lookback, min_periods=5).std() * math.sqrt(252)
        return vol.fillna(method="bfill").fillna(self.target_vol)

    def compute_leverage(self, returns: pd.Series) -> pd.Series:
        """
        Compute leverage series: target_vol / realized_vol, clamped to [min_lev, max_lev].
        """
        rv = self.estimate_vol(returns)
        leverage = (self.target_vol / (rv + 1e-9)).clip(self.min_lev, self.max_lev)
        return leverage

    def generate_weights(self, df: pd.DataFrame, base_signal: pd.Series = None) -> pd.Series:
        """
        Generate vol-targeted position series.

        Parameters
        ----------
        df          : OHLCV DataFrame
        base_signal : base directional signal (default: always +1 = long)
        """
        returns = df["close"].pct_change().fillna(0)
        leverage = self.compute_leverage(returns)
        if base_signal is None:
            base_signal = pd.Series(1.0, index=df.index)
        # Apply leverage to direction
        weights = base_signal * leverage
        return weights

    def backtest(
        self,
        df: pd.DataFrame,
        base_signal: pd.Series = None,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        weights = self.generate_weights(df, base_signal)
        close = df["close"].values
        w_vals = weights.values
        n = len(close)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        prev_w = 0.0

        for i in range(1, n):
            w = float(w_vals[i - 1]) if not np.isnan(w_vals[i - 1]) else 0.0
            bar_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
            cost = abs(w - prev_w) * commission_pct
            equity *= (1 + w * bar_ret - cost)
            if abs(w - prev_w) > 0.05:
                trades.append(w * bar_ret)
            ec[i] = equity
            prev_w = w

        s = _stats(ec, trades)
        avg_lev = float(weights.abs().mean())

        return BacktestResult(
            **{k: v for k, v in s.items() if k != "avg_trade_return"},
            avg_leverage=avg_lev,
            equity_curve=pd.Series(ec, index=df.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=df.index[1:]),
            weights_df=pd.DataFrame({"weight": weights}, index=df.index),
            params={"target_vol": self.target_vol, "lookback": self.lookback, "max_lev": self.max_lev},
        )

    def vol_statistics(self, df: pd.DataFrame) -> dict:
        """Return statistics about the vol-targeting leverage series."""
        returns = df["close"].pct_change().fillna(0)
        rv = self.estimate_vol(returns)
        leverage = self.compute_leverage(returns)
        return {
            "mean_rv": float(rv.mean()),
            "std_rv": float(rv.std()),
            "mean_leverage": float(leverage.mean()),
            "std_leverage": float(leverage.std()),
            "pct_at_max_lev": float((leverage == self.max_lev).mean()),
            "pct_at_min_lev": float((leverage == self.min_lev).mean()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RiskParity
# ─────────────────────────────────────────────────────────────────────────────

class RiskParity:
    """
    Risk Parity (Equal Risk Contribution) portfolio.

    Each asset contributes equally to total portfolio risk.
    The portfolio weights are solved such that:
        w_i * (Cov @ w)_i / (w.T @ Cov @ w) = 1/N for all i

    This is equivalent to minimizing the sum of squared differences
    from the equal risk contribution target.

    Parameters
    ----------
    assets      : list of asset column names
    target_vol  : target annualized portfolio volatility (default 0.10)
    lookback    : estimation window for covariance (default 63)
    rebal_freq  : rebalancing frequency in bars (default 21)
    max_lev     : maximum total leverage (default 2.0)
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        target_vol: float = 0.10,
        lookback: int = 63,
        rebal_freq: int = 21,
        max_lev: float = 2.0,
    ):
        self.assets = assets
        self.target_vol = target_vol
        self.lookback = lookback
        self.rebal_freq = rebal_freq
        self.max_lev = max_lev

    def _risk_parity_weights(self, cov: np.ndarray) -> np.ndarray:
        """
        Solve for equal risk contribution weights.

        Minimize: sum_i (w_i * MRC_i - budget_i)^2
        subject to: w >= 0, sum(w) = 1

        where MRC_i = (Cov @ w)_i = marginal risk contribution of asset i
        """
        n = cov.shape[0]
        budget = np.ones(n) / n  # equal risk budget

        def objective(w):
            w = np.abs(w)
            port_var = w @ cov @ w
            if port_var <= 0:
                return 1e9
            mrc = cov @ w
            rc = w * mrc / math.sqrt(port_var)
            return float(np.sum((rc - budget * math.sqrt(port_var)) ** 2))

        def gradient(w):
            w = np.abs(w)
            port_var = w @ cov @ w
            if port_var <= 0:
                return np.zeros(n)
            port_vol = math.sqrt(port_var)
            mrc = cov @ w
            rc = w * mrc / port_vol
            target_rc = budget * port_vol
            diff = rc - target_rc
            # Gradient w.r.t. w
            d_rc_d_w = (np.diag(mrc) + np.outer(w, cov.diagonal()) + cov * w[:, np.newaxis]) / port_vol
            d_rc_d_w -= np.outer(mrc, w) * (mrc @ w) / port_var ** 1.5
            grad = 2 * d_rc_d_w.T @ diff
            return grad

        # Initial guess: inverse volatility weights
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        w0 = (1.0 / vols) / (1.0 / vols).sum()

        bounds = [(0.001, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            objective, w0, jac=gradient, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if result.success:
            w = np.abs(result.x)
            return w / w.sum()
        return w0  # fallback to inverse vol

    def compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute risk parity weights from return covariance."""
        cov = returns.cov().values * 252  # annualize
        cov = np.maximum(cov, np.eye(len(cov)) * 1e-6)  # regularize
        return self._risk_parity_weights(cov)

    def risk_contributions(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Compute the risk contribution of each asset.
        RC_i = w_i * (Cov @ w)_i / sqrt(w.T @ Cov @ w)
        """
        port_var = weights @ cov @ weights
        port_vol = math.sqrt(max(1e-12, port_var))
        mrc = cov @ weights
        return weights * mrc / port_vol

    def generate_weights_df(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-series of risk parity weights.
        Rebalances every rebal_freq bars.
        """
        cols = self.assets if self.assets is not None else list(prices.columns)
        returns = prices[cols].pct_change().fillna(0)
        n_assets = len(cols)
        n_bars = len(prices)

        weights_df = pd.DataFrame(0.0, index=prices.index, columns=cols)
        w = np.ones(n_assets) / n_assets  # start equal weight

        for i in range(self.lookback, n_bars, self.rebal_freq):
            win = returns.iloc[max(0, i - self.lookback):i]
            if len(win) < n_assets + 2:
                continue
            try:
                w_new = self.compute_weights(win)
                # Scale to hit target_vol
                cov = win.cov().values * 252
                port_vol = math.sqrt(max(1e-12, w_new @ cov @ w_new))
                scale = min(self.max_lev, self.target_vol / (port_vol + 1e-9))
                w = w_new * scale
            except Exception:
                pass

            end_i = min(i + self.rebal_freq, n_bars)
            weights_df.iloc[i:end_i] = w

        return weights_df

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        cols = self.assets if self.assets is not None else list(prices.columns)
        weights_df = self.generate_weights_df(prices[cols])
        returns = prices[cols].pct_change().fillna(0)

        equity = initial_equity
        n = len(prices)
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        prev_w = np.zeros(len(cols))

        for i in range(1, n):
            w = weights_df.iloc[i].values
            r = returns.iloc[i].values
            port_ret = float(np.dot(w, r))
            turnover = np.abs(w - prev_w).sum() / 2
            port_ret -= turnover * commission_pct
            equity *= (1 + port_ret)
            ec[i] = equity
            if abs(port_ret) > 1e-9:
                trades.append(port_ret)
            prev_w = w

        s = _stats(ec, trades)
        avg_lev = float(weights_df.abs().sum(axis=1).mean())

        return BacktestResult(
            **{k: v for k, v in s.items() if k != "avg_trade_return"},
            avg_leverage=avg_lev,
            equity_curve=pd.Series(ec, index=prices.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=prices.index[1:]),
            weights_df=weights_df,
            params={"target_vol": self.target_vol, "lookback": self.lookback},
        )

    def equal_risk_contribution_check(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
    ) -> pd.DataFrame:
        """Verify risk contributions are equal."""
        rc = self.risk_contributions(weights, cov)
        total_rc = rc.sum()
        return pd.DataFrame({
            "weight": weights,
            "risk_contribution": rc,
            "risk_pct": rc / (total_rc + 1e-9),
            "target_risk_pct": 1.0 / len(weights),
            "deviation": rc / (total_rc + 1e-9) - 1.0 / len(weights),
        })


# ─────────────────────────────────────────────────────────────────────────────
# 3. MaxDiversification
# ─────────────────────────────────────────────────────────────────────────────

class MaxDiversification:
    """
    Maximum Diversification portfolio (Choueifaty & Coignard, 2008).

    Maximizes the Diversification Ratio:
        DR = (w.T @ sigma) / sqrt(w.T @ Cov @ w)

    where sigma is the vector of individual asset volatilities.

    This finds the portfolio with the highest ratio of weighted average vol
    to portfolio vol — maximum diversification.

    Parameters
    ----------
    assets          : list of asset names (columns in price DataFrame)
    lookback        : estimation window (default 63)
    rebal_freq      : rebalancing frequency (default 21)
    long_only       : long-only constraint (default True)
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        lookback: int = 63,
        rebal_freq: int = 21,
        long_only: bool = True,
    ):
        self.assets = assets
        self.lookback = lookback
        self.rebal_freq = rebal_freq
        self.long_only = long_only

    def _max_div_weights(self, cov: np.ndarray) -> np.ndarray:
        """
        Compute maximum diversification weights.

        Equivalent to minimum variance on the correlation matrix,
        then re-scaled by inverse volatility.
        """
        n = cov.shape[0]
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        corr = cov / (np.outer(vols, vols) + 1e-12)
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        def neg_dr(w):
            w = np.abs(w) if self.long_only else w
            port_var = w @ corr @ w
            if port_var <= 0:
                return 0.0
            return -float(w.sum() / math.sqrt(port_var))  # = -weighted avg vol / port vol (in corr space)

        def neg_dr_grad(w):
            w_use = np.abs(w) if self.long_only else w
            port_var = w_use @ corr @ w_use
            if port_var <= 0:
                return np.zeros(n)
            port_vol = math.sqrt(port_var)
            # gradient of DR
            num = w_use.sum()
            d_num = np.ones(n)
            d_denom = corr @ w_use / port_vol
            dr = num / port_vol
            grad = (d_num / port_vol - num * d_denom / port_var)
            return -grad  # negative because we minimize

        w0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n if self.long_only else [(-1.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}]

        result = minimize(
            neg_dr, w0, jac=neg_dr_grad, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if result.success:
            w = np.abs(result.x) if self.long_only else result.x
            return w / (np.sum(np.abs(w)) + 1e-12)

        # Fallback: minimum variance
        return self._min_var_weights(cov)

    def _min_var_weights(self, cov: np.ndarray) -> np.ndarray:
        """Minimum variance weights (analytical solution for long-only)."""
        n = cov.shape[0]
        try:
            cov_inv = np.linalg.pinv(cov)
            ones = np.ones(n)
            raw = cov_inv @ ones
            return raw / (raw.sum() + 1e-12)
        except Exception:
            return np.ones(n) / n

    def diversification_ratio(self, weights: np.ndarray, cov: np.ndarray) -> float:
        """
        Compute the diversification ratio of a portfolio.
        DR = (w.T @ sigma) / sqrt(w.T @ Cov @ w)
        """
        vols = np.sqrt(np.maximum(np.diag(cov), 1e-10))
        weighted_vol_sum = float(weights @ vols)
        port_vol = math.sqrt(max(1e-12, weights @ cov @ weights))
        return weighted_vol_sum / port_vol

    def generate_weights_df(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Generate time-series of max diversification weights."""
        cols = self.assets if self.assets is not None else list(prices.columns)
        returns = prices[cols].pct_change().fillna(0)
        n_bars = len(prices)
        n_assets = len(cols)

        weights_df = pd.DataFrame(0.0, index=prices.index, columns=cols)
        w = np.ones(n_assets) / n_assets

        for i in range(self.lookback, n_bars, self.rebal_freq):
            win = returns.iloc[max(0, i - self.lookback):i]
            if len(win) < n_assets + 2:
                continue
            cov = win.cov().values * 252
            cov = np.maximum(cov, np.eye(n_assets) * 1e-6)
            try:
                w = self._max_div_weights(cov)
            except Exception:
                pass
            end_i = min(i + self.rebal_freq, n_bars)
            weights_df.iloc[i:end_i] = w

        return weights_df

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        cols = self.assets if self.assets is not None else list(prices.columns)
        weights_df = self.generate_weights_df(prices[cols])
        returns = prices[cols].pct_change().fillna(0)

        equity = initial_equity
        n = len(prices)
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        prev_w = np.zeros(len(cols))

        for i in range(1, n):
            w = weights_df.iloc[i].values
            r = returns.iloc[i].values
            port_ret = float(np.dot(w, r))
            turnover = np.abs(w - prev_w).sum() / 2
            port_ret -= turnover * commission_pct
            equity *= (1 + port_ret)
            ec[i] = equity
            if abs(port_ret) > 1e-9:
                trades.append(port_ret)
            prev_w = w

        s = _stats(ec, trades)
        avg_lev = float(weights_df.abs().sum(axis=1).mean())

        return BacktestResult(
            **{k: v for k, v in s.items() if k != "avg_trade_return"},
            avg_leverage=avg_lev,
            equity_curve=pd.Series(ec, index=prices.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=prices.index[1:]),
            weights_df=weights_df,
            params={"lookback": self.lookback, "rebal_freq": self.rebal_freq},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1200
    idx = pd.date_range("2019-01-01", periods=n, freq="D")

    # Single-asset vol targeting
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": 1000}, index=idx)

    vt = VolatilityTargeting(target_vol=0.10, lookback=21, max_lev=2.0)
    res1 = vt.backtest(df)
    print("VolTargeting:", res1.summary())
    print("Vol stats:", vt.vol_statistics(df))

    # Multi-asset portfolio
    prices = pd.DataFrame({
        "A": 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)),
        "B": 80.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n)),
        "C": 50.0 * np.cumprod(1 + rng.normal(0.0002, 0.02, n)),
        "D": 200.0 * np.cumprod(1 + rng.normal(0.0001, 0.008, n)),
    }, index=idx)

    rp = RiskParity(target_vol=0.10, lookback=63, rebal_freq=21)
    res2 = rp.backtest(prices)
    print("RiskParity:", res2.summary())

    md = MaxDiversification(lookback=63, rebal_freq=21)
    res3 = md.backtest(prices)
    print("MaxDiversification:", res3.summary())
