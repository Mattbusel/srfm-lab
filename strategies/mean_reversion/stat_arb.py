"""
stat_arb.py — Statistical arbitrage strategies.

All strategies implement:
    generate_signals(df_a, df_b) -> pd.Series
    backtest(df_a, df_b) -> BacktestResult

References:
  - Engle & Granger (1987): Cointegration
  - Kalman (1960): Kalman filter
  - Johansen (1988): Multivariate cointegration
  - Ornstein & Uhlenbeck: mean-reverting SDE
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Shared result
# ─────────────────────────────────────────────────────────────────────────────

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
    avg_trade_return: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    spread_series: pd.Series = field(default_factory=pd.Series)
    z_score_series: pd.Series = field(default_factory=pd.Series)
    hedge_ratio_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%} "
                f"Trades={self.n_trades}")


def _stats(ec: np.ndarray, trades: list) -> dict:
    n = len(ec)
    tot = ec[-1] / ec[0] - 1
    n_yr = max(1, n / 252)
    cagr = (ec[-1] / ec[0]) ** (1 / n_yr) - 1
    r = np.diff(ec) / (ec[:-1] + 1e-9)
    r = np.concatenate([[0], r])
    std = r.std()
    sharpe = r.mean() / std * math.sqrt(252) if std > 0 else 0.0
    down = r[r < 0]
    sortino = r.mean() / (np.std(down) + 1e-9) * math.sqrt(252)
    pk = np.maximum.accumulate(ec)
    dd = (ec - pk) / (pk + 1e-9)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    wins = [x for x in trades if x > 0]
    losses = [x for x in trades if x <= 0]
    wr = len(wins) / len(trades) if trades else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")
    return dict(total_return=tot, cagr=cagr, sharpe=sharpe, sortino=sortino,
                max_drawdown=mdd, calmar=calmar, win_rate=wr, profit_factor=pf,
                n_trades=len(trades), avg_trade_return=float(np.mean(trades)) if trades else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# OLS hedge ratio / cointegration tests
# ─────────────────────────────────────────────────────────────────────────────

def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    OLS regression: y = alpha + beta * x + epsilon.
    Returns (alpha, beta).
    """
    n = len(x)
    xm = x.mean(); ym = y.mean()
    sxx = np.sum((x - xm) ** 2)
    sxy = np.sum((x - xm) * (y - ym))
    beta = sxy / (sxx + 1e-12)
    alpha = ym - beta * xm
    return float(alpha), float(beta)


def _rolling_ols(y: pd.Series, x: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """Rolling OLS. Returns (rolling_alpha, rolling_beta)."""
    n = len(y)
    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)
    y_arr = y.values
    x_arr = x.values

    for i in range(window, n + 1):
        y_w = y_arr[i - window:i]
        x_w = x_arr[i - window:i]
        a, b = _ols_hedge_ratio(y_w, x_w)
        alphas[i - 1] = a
        betas[i - 1] = b

    return pd.Series(alphas, index=y.index), pd.Series(betas, index=y.index)


def _adf_test_stat(series: np.ndarray, max_lag: int = 10) -> Tuple[float, float]:
    """
    Augmented Dickey-Fuller test statistic via OLS.
    H0: unit root (non-stationary). Returns (test_stat, p_value).
    """
    y = np.diff(series)
    x_lag = series[:-1]
    n = len(y)

    # Lag selection (default AIC)
    best_lag = 0
    best_aic = np.inf

    for lag in range(0, min(max_lag, n // 5)):
        if lag > 0:
            x_lags = np.column_stack([x_lag[lag:]] + [y[lag - k - 1: n - k - 1] for k in range(lag)])
            y_reg = y[lag:]
        else:
            x_lags = x_lag.reshape(-1, 1)
            y_reg = y

        x_lags = np.column_stack([np.ones(len(x_lags)), x_lags])
        try:
            coef, res, _, _ = np.linalg.lstsq(x_lags, y_reg, rcond=None)
            sigma2 = np.var(y_reg - x_lags @ coef)
            k = x_lags.shape[1]
            aic = len(y_reg) * np.log(sigma2 + 1e-12) + 2 * k
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
        except Exception:
            continue

    # Run final ADF with best lag
    lag = best_lag
    if lag > 0:
        x_mat = np.column_stack([np.ones(n - lag), x_lag[lag:]] +
                                 [y[lag - k - 1: n - k - 1] for k in range(lag)])
        y_reg = y[lag:]
    else:
        x_mat = np.column_stack([np.ones(n), x_lag])
        y_reg = y

    try:
        coef, _, _, _ = np.linalg.lstsq(x_mat, y_reg, rcond=None)
        residuals = y_reg - x_mat @ coef
        se2 = np.sum(residuals ** 2) / (len(y_reg) - len(coef))
        # Variance of beta coefficient
        xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
        se_beta = math.sqrt(max(0, se2 * xtx_inv[1, 1]))
        t_stat = coef[1] / (se_beta + 1e-12)
    except Exception:
        return 0.0, 0.5

    # Approximate p-value from t-distribution (not exact ADF critical values)
    p_val = float(scipy_stats.t.sf(abs(t_stat), df=max(1, len(y_reg) - len(coef))) * 2)
    return float(t_stat), p_val


def _cointegration_pvalue(y: np.ndarray, x: np.ndarray) -> float:
    """Engle-Granger cointegration p-value."""
    _, beta = _ols_hedge_ratio(y, x)
    spread = y - beta * x
    _, p_val = _adf_test_stat(spread)
    return p_val


# ─────────────────────────────────────────────────────────────────────────────
# 1. PairsTrading (Engle-Granger cointegration)
# ─────────────────────────────────────────────────────────────────────────────

class PairsTrading:
    """
    Classic pairs trading via Engle-Granger cointegration.

    1. Estimate hedge ratio via rolling OLS: A = alpha + beta * B + spread
    2. Normalize spread: z = (spread - mu) / sigma
    3. Long A / Short B when z < -entry_z
       Short A / Long B when z > +entry_z
       Exit when |z| < exit_z

    Parameters
    ----------
    lookback  : rolling window for hedge ratio and spread stats (default 60)
    entry_z   : z-score threshold to enter (default 2.0)
    exit_z    : z-score threshold to exit (default 0.5)
    stop_z    : stop-loss z-score (default 4.0)
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
    ):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def compute_spread(self, price_a: pd.Series, price_b: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute spread and z-score.
        Returns (spread, z_score, hedge_ratio).
        """
        alpha_series, beta_series = _rolling_ols(price_a, price_b, self.lookback)

        spread = price_a - beta_series * price_b - alpha_series
        spread_mean = spread.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        spread_std = spread.rolling(self.lookback, min_periods=self.lookback // 2).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-9)

        return spread, z_score, beta_series

    def generate_signals(self, price_a: pd.Series, price_b: pd.Series) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
            signal_a: +1 long A, -1 short A
            signal_b: opposite of signal_a (pairs trade)
        """
        _, z_score, _ = self.compute_spread(price_a, price_b)

        sig_a = pd.Series(0.0, index=price_a.index)
        position = 0

        for i in range(self.lookback, len(z_score)):
            z = z_score.iloc[i]
            if np.isnan(z):
                continue

            if position == 0:
                if z < -self.entry_z:
                    position = 1   # long A, short B
                elif z > self.entry_z:
                    position = -1  # short A, long B
            elif position == 1:
                if z > -self.exit_z or z > self.stop_z:
                    position = 0
            elif position == -1:
                if z < self.exit_z or z < -self.stop_z:
                    position = 0
            sig_a.iloc[i] = float(position)

        sig_a.iloc[:self.lookback] = np.nan
        return pd.DataFrame({"signal_a": sig_a, "signal_b": -sig_a})

    def backtest(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        """
        Pairs backtest: equal dollar allocation to each leg.
        P&L = 0.5 * (position_a * ret_a + position_b * ret_b).
        """
        spread, z_score, hedge_ratio = self.compute_spread(price_a, price_b)
        signals_df = self.generate_signals(price_a, price_b)
        sig_a = signals_df["signal_a"].values
        sig_b = signals_df["signal_b"].values

        ret_a = price_a.pct_change().fillna(0).values
        ret_b = price_b.pct_change().fillna(0).values
        n = len(price_a)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        prev_sig = 0.0

        for i in range(1, n):
            s_a = float(sig_a[i - 1]) if not np.isnan(sig_a[i - 1]) else 0.0
            s_b = float(sig_b[i - 1]) if not np.isnan(sig_b[i - 1]) else 0.0

            if s_a != prev_sig:
                cost = commission_pct * 2  # both legs
                if prev_sig != 0:
                    trades.append(-cost)  # closing trade cost
            prev_sig = s_a

            if s_a != 0 or s_b != 0:
                pair_ret = 0.5 * (s_a * ret_a[i] + s_b * ret_b[i])
                equity *= (1 + pair_ret)
                if abs(pair_ret) > 1e-8:
                    trades.append(pair_ret)

            ec[i] = equity

        s = _stats(ec, trades)
        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=price_a.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price_a.index[1:]),
            spread_series=spread,
            z_score_series=z_score,
            hedge_ratio_series=hedge_ratio,
            params={"lookback": self.lookback, "entry_z": self.entry_z, "exit_z": self.exit_z},
        )

    def cointegration_test(self, price_a: pd.Series, price_b: pd.Series) -> dict:
        """Run Engle-Granger cointegration test. Returns dict with test stat and p-value."""
        p_val = _cointegration_pvalue(price_a.values, price_b.values)
        is_cointegrated = p_val < 0.05
        return {
            "p_value": p_val,
            "is_cointegrated_5pct": is_cointegrated,
            "interpretation": "cointegrated" if is_cointegrated else "not cointegrated",
        }

    def half_life(self, price_a: pd.Series, price_b: pd.Series) -> float:
        """
        Estimate half-life of mean reversion from AR(1) fit of spread.
        half_life = -log(2) / log(lambda) where lambda = AR(1) coefficient.
        """
        spread, _, _ = self.compute_spread(price_a, price_b)
        spread_clean = spread.dropna()
        if len(spread_clean) < 20:
            return float("inf")
        y = spread_clean.diff().dropna().values
        x = spread_clean.shift(1).dropna().values
        _, phi = _ols_hedge_ratio(y, x)
        phi = float(np.clip(phi, -0.999, -0.001))  # AR(1) coefficient should be negative
        hl = -math.log(2) / math.log(1 + phi) if phi < 0 else float("inf")
        return hl


# ─────────────────────────────────────────────────────────────────────────────
# 2. KalmanPairsTrading
# ─────────────────────────────────────────────────────────────────────────────

class KalmanPairsTrading:
    """
    Kalman filter-based pairs trading with dynamic hedge ratio.

    The Kalman filter estimates the time-varying hedge ratio and intercept.
    This is more adaptive than rolling OLS and handles structural breaks better.

    State: [alpha, beta] — intercept and hedge ratio
    Observation: price_a = alpha + beta * price_b + noise

    Parameters
    ----------
    delta         : system noise (smaller = slower adaptation, default 1e-4)
    obs_noise_var : observation noise variance (default 1e-3)
    entry_z       : entry threshold (default 1.5)
    exit_z        : exit threshold (default 0.0)
    """

    def __init__(
        self,
        delta: float = 1e-4,
        obs_noise_var: float = 1e-3,
        entry_z: float = 1.5,
        exit_z: float = 0.0,
    ):
        self.delta = delta
        self.obs_noise_var = obs_noise_var
        self.entry_z = entry_z
        self.exit_z = exit_z

    def kalman_filter(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Run Kalman filter to estimate alpha, beta, spread, and z-score.

        Returns (alpha_series, beta_series, spread_series, z_score_series).
        """
        n = len(price_a)
        y_arr = price_a.values
        x_arr = price_b.values

        # Initialize state
        theta = np.zeros(2)   # [alpha, beta]
        P = np.ones((2, 2))   # state covariance
        R = self.obs_noise_var
        Q = self.delta / (1 - self.delta) * np.eye(2)  # process noise

        alphas = np.zeros(n)
        betas = np.zeros(n)
        spreads = np.zeros(n)
        e_series = np.zeros(n)   # innovations
        Q_series = np.zeros(n)   # innovation variance

        for i in range(n):
            # Observation matrix
            H = np.array([1.0, x_arr[i]])

            # Prediction step
            # (constant transition matrix = I)
            P = P + Q

            # Innovation
            yhat = float(H @ theta)
            e = y_arr[i] - yhat
            S = float(H @ P @ H) + R   # innovation variance

            # Kalman gain
            K = P @ H / (S + 1e-12)

            # Update
            theta = theta + K * e
            P = (np.eye(2) - np.outer(K, H)) @ P

            alphas[i] = theta[0]
            betas[i] = theta[1]
            spreads[i] = e
            e_series[i] = e
            Q_series[i] = S

        # Compute rolling z-score of spread
        spread_series = pd.Series(spreads, index=price_a.index)
        sqrt_Q = pd.Series(np.sqrt(np.maximum(Q_series, 1e-12)), index=price_a.index)
        z_score = spread_series / sqrt_Q.rolling(20, min_periods=5).mean()

        return (
            pd.Series(alphas, index=price_a.index),
            pd.Series(betas, index=price_a.index),
            spread_series,
            z_score,
        )

    def generate_signals(self, price_a: pd.Series, price_b: pd.Series) -> pd.DataFrame:
        _, _, _, z_score = self.kalman_filter(price_a, price_b)

        sig_a = pd.Series(0.0, index=price_a.index)
        position = 0

        for i in range(50, len(z_score)):
            z = z_score.iloc[i]
            if np.isnan(z):
                continue
            if position == 0:
                if z < -self.entry_z:
                    position = 1
                elif z > self.entry_z:
                    position = -1
            elif position == 1:
                if z > -self.exit_z:
                    position = 0
            elif position == -1:
                if z < self.exit_z:
                    position = 0
            sig_a.iloc[i] = float(position)

        sig_a.iloc[:50] = np.nan
        return pd.DataFrame({"signal_a": sig_a, "signal_b": -sig_a})

    def backtest(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        alpha_s, beta_s, spread, z_score = self.kalman_filter(price_a, price_b)
        signals_df = self.generate_signals(price_a, price_b)

        sig_a = signals_df["signal_a"].values
        ret_a = price_a.pct_change().fillna(0).values
        ret_b = price_b.pct_change().fillna(0).values
        n = len(price_a)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []

        for i in range(1, n):
            s_a = float(sig_a[i - 1]) if not np.isnan(sig_a[i - 1]) else 0.0
            s_b = -s_a

            if s_a != 0:
                pair_ret = 0.5 * (s_a * ret_a[i] + s_b * ret_b[i])
                equity *= (1 + pair_ret)
                trades.append(pair_ret)
            ec[i] = equity

        s = _stats(ec, trades)
        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=price_a.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=price_a.index[1:]),
            spread_series=spread,
            z_score_series=z_score,
            hedge_ratio_series=beta_s,
            params={"delta": self.delta, "entry_z": self.entry_z},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. JohansenPortfolio
# ─────────────────────────────────────────────────────────────────────────────

class JohansenPortfolio:
    """
    Johansen cointegration for multivariate portfolios.

    The Johansen procedure finds linear combinations of I(1) series
    that are stationary. These define the cointegrating vectors.

    The first eigenvector (with largest eigenvalue) defines the optimal
    stationary portfolio — this is the mean-reverting combination.

    Parameters
    ----------
    n_components : number of cointegrating vectors to use (default 1)
    lookback     : estimation window (default 252)
    entry_z      : entry z-score (default 2.0)
    exit_z       : exit z-score (default 0.5)
    """

    def __init__(
        self,
        n_components: int = 1,
        lookback: int = 252,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ):
        self.n_components = n_components
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def johansen_test(self, prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Johansen trace test for cointegration.

        Returns (eigenvalues, eigenvectors) — eigenvectors are columns,
        sorted by eigenvalue descending.

        This implements the eigenvalue decomposition of the reduced rank
        model via the companion form.
        """
        X = prices.values.copy()
        n, k = X.shape

        if n < k + 10:
            return np.ones(k) / k, np.eye(k)

        # Compute first differences
        dX = np.diff(X, axis=0)

        # Lagged levels
        X_lag1 = X[:-1]

        # Regress out lagged differences (use 1 lag)
        lag = 1
        if n > lag + k + 5:
            dX_lag = dX[:-1]
            X_lag1_adj = X_lag1[1:]
            dX_adj = dX[1:]
        else:
            dX_adj = dX
            X_lag1_adj = X_lag1
            dX_lag = None

        # Residuals of OLS regression of dX and X_lag1 on dX_lag
        def _ols_resid(Y, Z):
            if Z is None or len(Z) == 0:
                return Y
            Z_aug = np.column_stack([np.ones(len(Z)), Z])
            coef, _, _, _ = np.linalg.lstsq(Z_aug, Y, rcond=None)
            return Y - Z_aug @ coef

        R0 = _ols_resid(dX_adj, dX_lag if dX_lag is not None else None)
        R1 = _ols_resid(X_lag1_adj, dX_lag if dX_lag is not None else None)

        T = len(R0)
        S00 = R0.T @ R0 / T
        S11 = R1.T @ R1 / T
        S01 = R0.T @ R1 / T
        S10 = S01.T

        # Solve eigenvalue problem: S11^{-1/2} S10 S00^{-1} S01 S11^{-1/2}
        try:
            S11_inv = np.linalg.pinv(S11)
            S00_inv = np.linalg.pinv(S00)
            M = S11_inv @ S10 @ S00_inv @ S01
            eigenvalues, eigenvectors = np.linalg.eig(M)
            # Sort by eigenvalue descending
            idx = np.argsort(eigenvalues.real)[::-1]
            eigenvalues = eigenvalues.real[idx]
            eigenvectors = eigenvectors.real[:, idx]
        except Exception:
            eigenvalues = np.ones(k) / k
            eigenvectors = np.eye(k)

        return eigenvalues, eigenvectors

    def _compute_portfolio_series(
        self,
        prices: pd.DataFrame,
        cointegrating_vector: np.ndarray,
    ) -> pd.Series:
        """Compute the stationary portfolio series from a cointegrating vector."""
        return pd.Series(prices.values @ cointegrating_vector, index=prices.index)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate position weights for the cointegrating portfolio.
        Returns DataFrame of per-asset weights.
        """
        n = len(prices)
        k = prices.shape[1]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        portfolio_z = pd.Series(0.0, index=prices.index)

        step = max(1, self.lookback // 4)
        for i in range(self.lookback, n, step):
            window = prices.iloc[max(0, i - self.lookback):i]
            _, eigenvecs = self.johansen_test(window)
            coint_vec = eigenvecs[:, 0]  # first cointegrating vector

            # Compute portfolio value in this window
            port_vals = window.values @ coint_vec
            port_mean = port_vals.mean()
            port_std = port_vals.std()
            if port_std == 0:
                continue

            # Current portfolio value
            current_val = float(prices.iloc[i].values @ coint_vec)
            z = (current_val - port_mean) / port_std

            portfolio_z.iloc[i] = z

            if z < -self.entry_z:
                direction = 1.0
            elif z > self.entry_z:
                direction = -1.0
            else:
                direction = 0.0

            end_step = min(i + step, n)
            # Normalize the cointegrating vector
            norm = np.linalg.norm(coint_vec)
            if norm > 0:
                w = direction * coint_vec / norm / k
                weights.iloc[i:end_step] = w

        return weights

    def backtest(
        self,
        prices: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.001,
    ) -> BacktestResult:
        weights = self.generate_signals(prices)
        returns = prices.pct_change().fillna(0)

        equity = initial_equity
        n = len(prices)
        ec = np.full(n, initial_equity, dtype=float)
        trades = []

        for i in range(1, n):
            w = weights.iloc[i].values
            r = returns.iloc[i].values
            port_ret = float(np.dot(w, r))
            prev_w = weights.iloc[i - 1].values
            turnover = np.abs(w - prev_w).sum() / 2
            port_ret -= turnover * commission_pct
            equity *= (1 + port_ret)
            ec[i] = equity
            if abs(port_ret) > 1e-9:
                trades.append(port_ret)

        s = _stats(ec, trades)
        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=prices.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=prices.index[1:]),
            params={"lookback": self.lookback, "entry_z": self.entry_z},
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. OUMeanReversion
# ─────────────────────────────────────────────────────────────────────────────

class OUMeanReversion:
    """
    Ornstein-Uhlenbeck mean reversion strategy.

    The OU process: dX = theta * (mu - X) * dt + sigma * dW

    Parameters are estimated via MLE on the spread series.
    Trading signals derived from the standardized deviation from mean.

    Parameters
    ----------
    lookback         : estimation window (default 60)
    entry_threshold  : entry signal threshold in units of sigma_eq (default 1.5)
    exit_threshold   : exit signal threshold (default 0.0)
    min_half_life    : minimum acceptable half-life in bars (default 5)
    max_half_life    : maximum acceptable half-life (default 100)
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.0,
        min_half_life: float = 5.0,
        max_half_life: float = 100.0,
    ):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def fit_ou_params(self, spread: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Estimate OU parameters via OLS on discretized version.

        AR(1): X_t = a + b * X_{t-1} + epsilon

        theta = -log(b) / dt      (mean reversion speed)
        mu    = a / (1 - b)       (long-term mean)
        sigma = std(epsilon) / sqrt((1 - b^2) / (2 * theta))
        half_life = log(2) / theta

        Returns (theta, mu, sigma, half_life)
        """
        y = spread[1:]
        x = spread[:-1]
        _, b = _ols_hedge_ratio(y, x)
        b = float(np.clip(b, 0.001, 0.999))  # stability
        a = np.mean(y) - b * np.mean(x)

        # OU params from AR(1)
        mu = a / (1 - b + 1e-12)
        theta = -math.log(b) if b > 0 else 0.01
        theta = max(0.001, theta)
        half_life = math.log(2) / theta

        # Sigma of stationary distribution
        residuals = y - (a + b * x)
        sigma_eps = residuals.std()
        sigma_eq = sigma_eps / math.sqrt(max(1e-9, 1 - b**2))

        return theta, mu, sigma_eq, half_life

    def _compute_z_score(self, spread: pd.Series) -> pd.Series:
        """Rolling OU-adjusted z-score of the spread."""
        z_scores = pd.Series(np.nan, index=spread.index)

        for i in range(self.lookback, len(spread)):
            window = spread.iloc[i - self.lookback:i].values
            theta, mu, sigma_eq, hl = self.fit_ou_params(window)

            if hl < self.min_half_life or hl > self.max_half_life:
                continue
            if sigma_eq <= 0:
                continue

            current = float(spread.iloc[i])
            z_scores.iloc[i] = (current - mu) / sigma_eq

        return z_scores

    def generate_signals(self, spread: pd.Series) -> pd.Series:
        """Signal on the spread series: +1 = long spread (spread below mean)."""
        z_scores = self._compute_z_score(spread)
        signal = pd.Series(0.0, index=spread.index)
        position = 0

        for i in range(len(z_scores)):
            z = z_scores.iloc[i]
            if np.isnan(z):
                continue
            if position == 0:
                if z < -self.entry_threshold:
                    position = 1
                elif z > self.entry_threshold:
                    position = -1
            elif position == 1:
                if z > -self.exit_threshold:
                    position = 0
            elif position == -1:
                if z < self.exit_threshold:
                    position = 0
            signal.iloc[i] = float(position)

        signal.iloc[:self.lookback] = np.nan
        return signal

    def backtest(
        self,
        spread: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        """Backtest trading the spread directly."""
        signal = self.generate_signals(spread)
        spread_vals = spread.values
        sig_vals = signal.values
        n = len(spread)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        position = 0.0

        for i in range(1, n):
            s = float(sig_vals[i - 1]) if not np.isnan(sig_vals[i - 1]) else 0.0
            ret = (spread_vals[i] - spread_vals[i - 1]) / (abs(spread_vals[i - 1]) + 1e-9)

            if s != position:
                if position != 0 and len(trades) > 0:
                    trades[-1] -= commission_pct
                position = s

            if position != 0:
                pnl = position * ret
                equity *= (1 + pnl)
                trades.append(pnl)
            ec[i] = equity

        s = _stats(ec, trades)
        z_scores = self._compute_z_score(spread)

        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=spread.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=spread.index[1:]),
            spread_series=spread,
            z_score_series=z_scores,
            params={"lookback": self.lookback, "entry_threshold": self.entry_threshold},
        )

    def current_params(self, spread: pd.Series) -> dict:
        """Get most recent OU parameter estimates."""
        if len(spread) < self.lookback:
            return {}
        window = spread.iloc[-self.lookback:].values
        theta, mu, sigma_eq, hl = self.fit_ou_params(window)
        return {
            "theta": theta,
            "mu": mu,
            "sigma_eq": sigma_eq,
            "half_life_bars": hl,
            "half_life_days": hl,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ETFArbitrage
# ─────────────────────────────────────────────────────────────────────────────

class ETFArbitrage:
    """
    ETF premium/discount mean reversion strategy.

    ETFs trade at a premium or discount to their NAV due to creation/redemption
    frictions. When the premium/discount is extreme, it tends to mean-revert.

    Signal: buy ETF when discount > threshold (ETF price << NAV).
            sell ETF when premium > threshold (ETF price >> NAV).

    Parameters
    ----------
    entry_threshold : premium/discount threshold to enter (default 0.005 = 0.5%)
    exit_threshold  : threshold to exit (default 0.001 = 0.1%)
    smoothing       : EMA smoothing of premium series (default 5)
    """

    def __init__(
        self,
        entry_threshold: float = 0.005,
        exit_threshold: float = 0.001,
        smoothing: int = 5,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.smoothing = smoothing

    def compute_premium(self, etf_price: pd.Series, nav_series: pd.Series) -> pd.Series:
        """
        Compute ETF premium/discount.

        premium = (etf_price - nav) / nav
        Positive = ETF trading at premium to NAV.
        Negative = ETF trading at discount.
        """
        premium = (etf_price - nav_series) / (nav_series.abs() + 1e-9)
        if self.smoothing > 1:
            premium = premium.ewm(span=self.smoothing, adjust=False).mean()
        return premium

    def generate_signals(self, etf_price: pd.Series, nav_series: pd.Series) -> pd.Series:
        """
        Returns signal: +1 buy ETF (discount), -1 sell ETF (premium), 0 flat.
        """
        premium = self.compute_premium(etf_price, nav_series)
        signal = pd.Series(0.0, index=etf_price.index)
        position = 0

        for i in range(1, len(premium)):
            p = float(premium.iloc[i])
            if np.isnan(p):
                continue
            if position == 0:
                if p < -self.entry_threshold:  # discount
                    position = 1
                elif p > self.entry_threshold:  # premium
                    position = -1
            elif position == 1:  # long ETF, exit when premium normalizes
                if p > -self.exit_threshold:
                    position = 0
            elif position == -1:  # short ETF, exit when discount normalizes
                if p < self.exit_threshold:
                    position = 0
            signal.iloc[i] = float(position)

        return signal

    def backtest(
        self,
        etf_price: pd.Series,
        nav_series: pd.Series,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0002,
    ) -> BacktestResult:
        signal = self.generate_signals(etf_price, nav_series)
        premium = self.compute_premium(etf_price, nav_series)

        ec, trades = _signal_to_equity(etf_price.values, signal.values, initial_equity, commission_pct)
        s = _stats(ec, trades)

        return BacktestResult(
            **s,
            equity_curve=pd.Series(ec, index=etf_price.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=etf_price.index[1:]),
            spread_series=premium,
            z_score_series=(premium - premium.mean()) / (premium.std() + 1e-9),
            params={"entry_threshold": self.entry_threshold, "exit_threshold": self.exit_threshold},
        )

    def premium_statistics(self, etf_price: pd.Series, nav_series: pd.Series) -> dict:
        """Summary statistics of the premium series."""
        p = self.compute_premium(etf_price, nav_series)
        return {
            "mean_premium_bps": p.mean() * 10000,
            "std_premium_bps": p.std() * 10000,
            "max_premium_bps": p.max() * 10000,
            "min_premium_bps": p.min() * 10000,
            "pct_above_threshold": (p.abs() > self.entry_threshold).mean(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def _make_cointegrated_pair(n: int = 1000, seed: int = 42) -> Tuple[pd.Series, pd.Series]:
    """Generate cointegrated price pair for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    # Common trend
    common = np.cumsum(rng.normal(0, 0.5, n))
    noise_a = rng.normal(0, 0.2, n)
    noise_b = rng.normal(0, 0.2, n)
    # A and B cointegrated (both driven by common factor)
    log_a = common + 0.5 * noise_a
    log_b = common * 0.8 + 0.3 + 0.5 * noise_b
    # Add mean reversion in the spread
    spread = log_a - log_b
    mean_rev_spread = np.zeros(n)
    for i in range(1, n):
        mean_rev_spread[i] = mean_rev_spread[i - 1] * 0.95 + rng.normal(0, 0.3)
    log_a_final = common + mean_rev_spread
    log_b_final = common + rng.normal(0, 0.1, n)
    price_a = pd.Series(np.exp(log_a_final / 10 + 4.5), index=idx)  # ~90 price
    price_b = pd.Series(np.exp(log_b_final / 10 + 4.3), index=idx)  # ~74 price
    return price_a, price_b


if __name__ == "__main__":
    pa, pb = _make_cointegrated_pair(1000)

    pt = PairsTrading(lookback=60, entry_z=2.0, exit_z=0.5)
    print("Cointegration test:", pt.cointegration_test(pa, pb))
    print("Half-life:", pt.half_life(pa, pb), "bars")
    res = pt.backtest(pa, pb)
    print("PairsTrading:", res.summary())

    kpt = KalmanPairsTrading()
    res2 = kpt.backtest(pa, pb)
    print("KalmanPairs:", res2.summary())

    # OU on spread
    spread = pa - pb
    ou = OUMeanReversion(lookback=60, entry_threshold=1.5)
    res3 = ou.backtest(spread)
    print("OU MeanRev:", res3.summary())
    print("OU params:", ou.current_params(spread))
