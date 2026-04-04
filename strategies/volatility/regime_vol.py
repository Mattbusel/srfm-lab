"""
regime_vol.py — Volatility regime switching strategies.

Switch between strategies based on detected market vol regime.
GARCH-based vol forecasting and trading.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class BacktestResult:
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    n_trades: int = 0
    avg_regime: str = ""
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    regime_series: pd.Series = field(default_factory=pd.Series)
    params: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"Return={self.total_return:.2%} CAGR={self.cagr:.2%} "
                f"Sharpe={self.sharpe:.3f} MaxDD={self.max_drawdown:.2%}")


def _stats(ec: np.ndarray) -> dict:
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
    return dict(total_return=tot, cagr=cagr, sharpe=sh, sortino=sortino,
                max_drawdown=mdd, calmar=calmar)


# ─────────────────────────────────────────────────────────────────────────────
# Vol Regime Detector
# ─────────────────────────────────────────────────────────────────────────────

class VolRegimeDetector:
    """
    Classifies market into volatility regimes using rolling realized vol.

    Regimes:
        CALM     : realized_vol < low_threshold
        NORMAL   : low_threshold <= realized_vol < high_threshold
        HIGH_VOL : realized_vol >= high_threshold

    Parameters
    ----------
    low_threshold  : vol threshold for CALM regime (default 0.10 = 10%)
    high_threshold : vol threshold for HIGH_VOL regime (default 0.25 = 25%)
    lookback       : vol estimation window (default 21)
    method         : "rolling" or "ewm" (default "ewm")
    halflife       : halflife for EWM (default 10)
    """

    def __init__(
        self,
        low_threshold: float = 0.10,
        high_threshold: float = 0.25,
        lookback: int = 21,
        method: str = "ewm",
        halflife: int = 10,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.lookback = lookback
        self.method = method
        self.halflife = halflife

    def estimate_vol(self, returns: pd.Series) -> pd.Series:
        """Estimate annualized realized vol."""
        if self.method == "ewm":
            return returns.ewm(halflife=self.halflife, min_periods=5).std() * math.sqrt(252)
        return returns.rolling(self.lookback, min_periods=5).std() * math.sqrt(252)

    def classify(self, prices: pd.DataFrame) -> pd.Series:
        """
        Returns regime series: "CALM", "NORMAL", "HIGH_VOL".
        """
        returns = prices["close"].pct_change().fillna(0)
        vol = self.estimate_vol(returns).fillna(self.high_threshold)

        regime = pd.Series("NORMAL", index=prices.index)
        regime[vol < self.low_threshold] = "CALM"
        regime[vol >= self.high_threshold] = "HIGH_VOL"
        return regime

    def vol_series(self, prices: pd.DataFrame) -> pd.Series:
        """Return the realized vol series used for classification."""
        returns = prices["close"].pct_change().fillna(0)
        return self.estimate_vol(returns)

    def regime_statistics(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Statistics of returns in each regime.
        """
        returns = prices["close"].pct_change().fillna(0)
        regime = self.classify(prices)

        rows = []
        for reg in ["CALM", "NORMAL", "HIGH_VOL"]:
            mask = regime == reg
            r = returns[mask].dropna()
            if len(r) < 10:
                continue
            rows.append({
                "regime": reg,
                "n_days": len(r),
                "pct_days": len(r) / len(returns),
                "mean_daily_ret": float(r.mean()),
                "std_daily_ret": float(r.std()),
                "ann_sharpe": float(r.mean() / (r.std() + 1e-9) * math.sqrt(252)),
                "hit_rate": float((r > 0).mean()),
            })
        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 1. RegimeVolSwitching
# ─────────────────────────────────────────────────────────────────────────────

class RegimeVolSwitching:
    """
    Regime-switching strategy: switch between calm and storm strategies.

    In CALM regime: deploy the calm_strategy (e.g., trend following, full size)
    In HIGH_VOL regime: deploy the storm_strategy (e.g., reduce size, mean rev)
    In NORMAL regime: blend or use calm strategy at reduced size.

    Parameters
    ----------
    calm_strategy  : callable(df) -> pd.Series of signals, used in calm regime
    storm_strategy : callable(df) -> pd.Series of signals, used in high-vol
    regime_detector: VolRegimeDetector instance
    normal_weight  : weight on calm strategy in NORMAL regime (default 0.5)
    calm_weight    : weight on calm strategy in CALM (default 1.0)
    storm_weight   : weight on storm strategy in HIGH_VOL (default 1.0)
    """

    def __init__(
        self,
        calm_strategy: Optional[Callable] = None,
        storm_strategy: Optional[Callable] = None,
        regime_detector: Optional[VolRegimeDetector] = None,
        normal_weight: float = 0.5,
        calm_weight: float = 1.0,
        storm_weight: float = 1.0,
    ):
        self.calm_strategy = calm_strategy
        self.storm_strategy = storm_strategy
        self.regime_detector = regime_detector or VolRegimeDetector()
        self.normal_weight = normal_weight
        self.calm_weight = calm_weight
        self.storm_weight = storm_weight

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate combined signal based on detected regime.
        If no strategies provided, uses simple buy/hold in calm, neutral in storm.
        """
        regime = self.regime_detector.classify(df)

        # Get signals from each strategy
        if self.calm_strategy is not None:
            try:
                calm_sig = self.calm_strategy(df)
            except Exception:
                calm_sig = pd.Series(1.0, index=df.index)
        else:
            calm_sig = pd.Series(1.0, index=df.index)  # default: long

        if self.storm_strategy is not None:
            try:
                storm_sig = self.storm_strategy(df)
            except Exception:
                storm_sig = pd.Series(0.0, index=df.index)
        else:
            storm_sig = pd.Series(0.0, index=df.index)  # default: flat

        # Blend based on regime
        combined = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            reg = regime.iloc[i]
            if reg == "CALM":
                combined.iloc[i] = float(calm_sig.iloc[i]) * self.calm_weight
            elif reg == "HIGH_VOL":
                combined.iloc[i] = float(storm_sig.iloc[i]) * self.storm_weight
            else:  # NORMAL
                cs = float(calm_sig.iloc[i]) if not np.isnan(calm_sig.iloc[i]) else 0.0
                ss = float(storm_sig.iloc[i]) if not np.isnan(storm_sig.iloc[i]) else 0.0
                combined.iloc[i] = cs * self.normal_weight + ss * (1 - self.normal_weight)

        return combined

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signal = self.generate_signals(df)
        regime = self.regime_detector.classify(df)
        close = df["close"].values
        sig = signal.values
        n = len(close)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        pos = 0.0

        for i in range(1, n):
            s = float(sig[i - 1]) if not np.isnan(sig[i - 1]) else 0.0
            bar_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
            cost = abs(s - pos) * commission_pct
            equity *= (1 + s * bar_ret - cost)
            if abs(s - pos) > 0.05:
                trades.append(s * bar_ret)
            ec[i] = equity
            pos = s

        s_stats = _stats(ec)
        regime_counts = regime.value_counts(normalize=True)
        avg_regime = str(regime_counts.idxmax())

        return BacktestResult(
            **s_stats,
            n_trades=len(trades),
            avg_regime=avg_regime,
            equity_curve=pd.Series(ec, index=df.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=df.index[1:]),
            regime_series=regime,
            params={"calm_weight": self.calm_weight, "storm_weight": self.storm_weight,
                    "normal_weight": self.normal_weight},
        )

    def regime_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Show strategy performance sliced by regime."""
        signal = self.generate_signals(df)
        regime = self.regime_detector.classify(df)
        returns = df["close"].pct_change().fillna(0)
        strat_ret = signal.shift(1).fillna(0) * returns

        rows = []
        for reg in ["CALM", "NORMAL", "HIGH_VOL"]:
            mask = regime == reg
            r = strat_ret[mask].dropna()
            if len(r) < 5:
                continue
            rows.append({
                "regime": reg,
                "n_days": len(r),
                "total_return": float((1 + r).prod() - 1),
                "ann_sharpe": float(r.mean() / (r.std() + 1e-9) * math.sqrt(252)),
                "hit_rate": float((r > 0).mean()),
            })
        return pd.DataFrame(rows).set_index("regime") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 2. GARCHVolTrading
# ─────────────────────────────────────────────────────────────────────────────

class GARCHVolTrading:
    """
    Trade based on GARCH(p,q) volatility forecasts.

    Strategy logic:
    - Estimate GARCH model on rolling window
    - Generate 1-step ahead vol forecast
    - When forecast vol is high relative to recent average → reduce/short
    - When forecast vol is low → increase/long

    Full GARCH(p,q) implementation without external libraries.

    Parameters
    ----------
    p         : GARCH lag order for lagged variance terms (default 1)
    q         : ARCH lag order for lagged squared returns (default 1)
    threshold : GARCH vol forecast z-score to enter (default 1.0)
    lookback  : estimation window for GARCH (default 252)
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        threshold: float = 1.0,
        lookback: int = 252,
    ):
        self.p = p
        self.q = q
        self.threshold = threshold
        self.lookback = lookback
        self._last_params = None

    def fit_garch(
        self,
        returns: np.ndarray,
    ) -> Tuple[float, List[float], List[float], float]:
        """
        Fit GARCH(p,q) model via maximum likelihood (Gaussian innovations).

        Returns (omega, alpha_list, beta_list, log_likelihood).

        GARCH(1,1): sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

        Log-likelihood: -0.5 * sum(log(sigma_t^2) + eps_t^2 / sigma_t^2)
        """
        n = len(returns)
        mu = returns.mean()
        eps = returns - mu

        def garch_variance(omega, alphas, betas):
            """Compute variance series for GARCH(p,q)."""
            sigma2 = np.full(n, np.var(eps))
            for t in range(max(self.p, self.q), n):
                v = float(omega)
                for j, alpha in enumerate(alphas):
                    if t - j - 1 >= 0:
                        v += float(alpha) * float(eps[t - j - 1]) ** 2
                for j, beta in enumerate(betas):
                    if t - j - 1 >= 0:
                        v += float(beta) * sigma2[t - j - 1]
                sigma2[t] = max(1e-12, v)
            return sigma2

        def neg_log_likelihood(params):
            """Negative log-likelihood for GARCH."""
            omega = params[0]
            alphas = params[1:1 + self.q]
            betas = params[1 + self.q:1 + self.q + self.p]

            # Constraints: omega > 0, alpha >= 0, beta >= 0, sum(alpha+beta) < 1
            if omega <= 0 or any(a < 0 for a in alphas) or any(b < 0 for b in betas):
                return 1e9
            if sum(alphas) + sum(betas) >= 1.0:
                return 1e9

            sigma2 = garch_variance(omega, alphas, betas)
            ll = -0.5 * np.sum(np.log(sigma2 + 1e-12) + eps ** 2 / (sigma2 + 1e-12))
            return -ll

        # Initialize with method-of-moments
        var_init = float(np.var(eps))
        omega0 = var_init * 0.1
        alpha0 = [0.1] * self.q
        beta0 = [0.8] * self.p

        # Constrained optimization using gradient-free method (Nelder-Mead)
        from scipy.optimize import minimize as sp_minimize
        x0 = [omega0] + alpha0 + beta0
        bounds_list = [(1e-8, None)] + [(0, 0.5)] * self.q + [(0, 0.99)] * self.p

        try:
            result = sp_minimize(
                neg_log_likelihood, x0, method="L-BFGS-B",
                bounds=bounds_list,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            params = result.x
        except Exception:
            params = x0

        omega = float(params[0])
        alphas = list(float(params[1 + j]) for j in range(self.q))
        betas = list(float(params[1 + self.q + j]) for j in range(self.p))
        ll = float(-neg_log_likelihood(params))

        return omega, alphas, betas, ll

    def forecast_vol(
        self,
        returns: np.ndarray,
        omega: float,
        alphas: List[float],
        betas: List[float],
        h: int = 1,
    ) -> float:
        """
        h-step ahead GARCH variance forecast.

        For GARCH(1,1):
        sigma_t+h^2 = omega/(1-alpha-beta) + (alpha+beta)^(h-1) * (sigma_t+1^2 - LT_var)
        """
        n = len(returns)
        eps = returns - returns.mean()
        # Compute variance series
        sigma2 = np.full(n, float(np.var(eps)))
        for t in range(max(self.p, self.q), n):
            v = omega
            for j, alpha in enumerate(alphas):
                if t - j - 1 >= 0:
                    v += alpha * eps[t - j - 1] ** 2
            for j, beta in enumerate(betas):
                if t - j - 1 >= 0:
                    v += beta * sigma2[t - j - 1]
            sigma2[t] = max(1e-12, v)

        # Long-term variance
        persistence = sum(alphas) + sum(betas)
        if persistence >= 1.0:
            persistence = 0.999
        lt_var = omega / (1.0 - persistence)

        # h-step forecast
        current_var = sigma2[-1]
        forecast = lt_var + (persistence ** (h - 1)) * (current_var - lt_var)
        return float(math.sqrt(max(1e-12, forecast)) * math.sqrt(252))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on GARCH vol forecast vs. recent average.
        Short when forecast vol >> average vol, long when forecast vol << average.
        """
        close = df["close"]
        returns = close.pct_change().fillna(0).values
        n = len(returns)

        signals = pd.Series(np.nan, index=df.index)
        forecast_vols = pd.Series(np.nan, index=df.index)

        step = max(1, self.lookback // 10)  # refit every 10% of window

        cached_params = None
        for i in range(self.lookback, n, step):
            window = returns[i - self.lookback:i]
            try:
                omega, alphas, betas, _ = self.fit_garch(window)
                cached_params = (omega, alphas, betas)
            except Exception:
                if cached_params is None:
                    continue

            omega, alphas, betas = cached_params
            fvol = self.forecast_vol(window, omega, alphas, betas, h=1)

            end_i = min(i + step, n)
            for j in range(i, end_i):
                forecast_vols.iloc[j] = fvol

        # Rolling average of forecast vol
        avg_forecast = forecast_vols.rolling(self.lookback // 4, min_periods=10).mean()
        std_forecast = forecast_vols.rolling(self.lookback // 4, min_periods=10).std()
        z_vol = (forecast_vols - avg_forecast) / (std_forecast + 1e-9)

        for i in range(self.lookback, n):
            z = float(z_vol.iloc[i]) if not np.isnan(z_vol.iloc[i]) else 0.0
            if z > self.threshold:   # high vol forecast → reduce/go short
                signals.iloc[i] = -1.0
            elif z < -self.threshold:  # low vol forecast → go long
                signals.iloc[i] = 1.0
            else:
                signals.iloc[i] = 0.0

        self._last_params = cached_params
        return signals

    def backtest(
        self,
        df: pd.DataFrame,
        initial_equity: float = 1_000_000,
        commission_pct: float = 0.0001,
    ) -> BacktestResult:
        signal = self.generate_signals(df)
        close = df["close"].values
        sig = signal.values
        n = len(close)

        equity = initial_equity
        ec = np.full(n, initial_equity, dtype=float)
        trades = []
        pos = 0.0

        for i in range(1, n):
            s = float(sig[i - 1]) if not np.isnan(sig[i - 1]) else 0.0
            bar_ret = (close[i] - close[i - 1]) / (close[i - 1] + 1e-9)
            cost = abs(s - pos) * commission_pct
            equity *= (1 + s * bar_ret - cost)
            if abs(bar_ret * s) > 1e-9:
                trades.append(s * bar_ret)
            ec[i] = equity
            pos = s

        s_stats = _stats(ec)

        return BacktestResult(
            **s_stats,
            n_trades=len(trades),
            equity_curve=pd.Series(ec, index=df.index),
            returns=pd.Series(np.diff(ec) / (ec[:-1] + 1e-9), index=df.index[1:]),
            regime_series=signal,
            params={"p": self.p, "q": self.q, "threshold": self.threshold, "lookback": self.lookback},
        )

    def garch_statistics(self, df: pd.DataFrame) -> dict:
        """Fit GARCH model and return model statistics."""
        returns = df["close"].pct_change().dropna().values
        if len(returns) < self.lookback:
            return {}
        omega, alphas, betas, ll = self.fit_garch(returns[-self.lookback:])
        persistence = sum(alphas) + sum(betas)
        lt_vol = math.sqrt(omega / max(1e-12, 1.0 - persistence)) * math.sqrt(252) if persistence < 1 else float("inf")
        return {
            "omega": omega,
            "alpha": alphas,
            "beta": betas,
            "persistence": persistence,
            "long_term_vol": lt_vol,
            "log_likelihood": ll,
            "aic": -2 * ll + 2 * (1 + len(alphas) + len(betas)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    # GARCH-like data: vol clustering
    vol = np.full(n, 0.01)
    returns = np.zeros(n)
    for i in range(1, n):
        vol[i] = 0.00002 + 0.1 * returns[i-1]**2 + 0.85 * vol[i-1]
        returns[i] = rng.normal(0, math.sqrt(vol[i]))
    close = 100.0 * np.cumprod(1 + returns)
    df = pd.DataFrame({"open": close, "high": close*1.01, "low": close*0.99,
                       "close": close, "volume": 1000}, index=idx)

    # Regime detector
    rd = VolRegimeDetector(low_threshold=0.10, high_threshold=0.25)
    print("Regime stats:")
    print(rd.regime_statistics(df))

    # Regime switching
    def calm_strat(df):
        close = df["close"]
        return pd.Series(np.where(close > close.ewm(50).mean(), 1.0, 0.0), index=df.index)

    rvsw = RegimeVolSwitching(calm_strategy=calm_strat, regime_detector=rd)
    res1 = rvsw.backtest(df)
    print("\nRegime Switching:", res1.summary())

    # GARCH trading
    gvt = GARCHVolTrading(p=1, q=1, threshold=0.8, lookback=120)
    print("\nGARCH stats:", gvt.garch_statistics(df))
    res2 = gvt.backtest(df)
    print("GARCH Trading:", res2.summary())
