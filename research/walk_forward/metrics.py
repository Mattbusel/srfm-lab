"""
research/walk_forward/metrics.py
─────────────────────────────────
Comprehensive performance metrics for walk-forward and CPCV analysis.

All functions accept plain numpy arrays or pandas Series/DataFrames.
Return types are always scalars (float / tuple) unless stated otherwise.

Reference implementations:
  • Sharpe, Sortino, Calmar — standard CFA definitions
  • Deflated Sharpe Ratio — Bailey & López de Prado (2014)
  • Probability of Backtest Overfitting — López de Prado (2018) Ch.14
  • Hurst exponent — R/S analysis (Hurst, 1951)
  • Kelly criterion — Kelly (1956)
  • Omega ratio — Keating & Shadwick (2002)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helper types
# ─────────────────────────────────────────────────────────────────────────────

Array = np.ndarray | pd.Series

@dataclass
class UnderwaterPeriod:
    """Describes a contiguous drawdown period."""
    start_idx:  int
    end_idx:    int
    peak_value: float
    trough_value: float
    drawdown:   float          # fraction (negative)
    duration:   int            # number of bars
    recovery_idx: Optional[int] = None  # None if still underwater

    @property
    def recovered(self) -> bool:
        return self.recovery_idx is not None


# ─────────────────────────────────────────────────────────────────────────────
# Internal conversion
# ─────────────────────────────────────────────────────────────────────────────

def _to_array(x: object) -> np.ndarray:
    """Convert any array-like to a clean float64 numpy array."""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=np.float64, na_value=np.nan)
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
        raise ValueError("DataFrame with >1 column passed — expected 1-D input")
    return np.asarray(x, dtype=np.float64)


def _clean(arr: np.ndarray) -> np.ndarray:
    """Drop NaN / Inf values."""
    arr = arr[np.isfinite(arr)]
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Return / equity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _annualization_factor(periods_per_year: int) -> float:
    return float(periods_per_year) ** 0.5


def _mean_return(returns: np.ndarray) -> float:
    """Arithmetic mean return."""
    return float(np.mean(returns)) if len(returns) > 0 else 0.0


def _std_return(returns: np.ndarray, ddof: int = 1) -> float:
    """Standard deviation of returns."""
    return float(np.std(returns, ddof=ddof)) if len(returns) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(
    returns:          object,
    rf:               float = 0.0,
    annualize:        bool  = True,
    periods_per_year: int   = 252,
) -> float:
    """
    Compute the annualized Sharpe ratio.

    SR = mean(r - rf) / std(r - rf) * sqrt(periods_per_year)

    Parameters
    ----------
    returns          : 1-D array of period returns (not log returns).
    rf               : risk-free rate per period (default 0.0).
    annualize        : if True, multiply by sqrt(periods_per_year).
    periods_per_year : trading periods per year (252 for daily, 52 weekly).

    Returns
    -------
    float Sharpe ratio. Returns 0.0 if std is zero or input is empty.
    """
    r = _clean(_to_array(returns))
    if len(r) < 2:
        return 0.0

    excess = r - rf / periods_per_year  # convert annual rf to per-period
    mu     = float(np.mean(excess))
    sigma  = float(np.std(excess, ddof=1))

    if sigma < 1e-12:
        return 0.0

    sr = mu / sigma
    if annualize:
        sr *= math.sqrt(periods_per_year)
    return sr


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sortino Ratio
# ─────────────────────────────────────────────────────────────────────────────

def sortino_ratio(
    returns:          object,
    rf:               float = 0.0,
    mar:              float = 0.0,
    periods_per_year: int   = 252,
) -> float:
    """
    Compute the annualized Sortino ratio.

    SR_sortino = mean(r - MAR) / downside_deviation * sqrt(T)

    Parameters
    ----------
    returns : 1-D array of period returns.
    rf      : annual risk-free rate (converted to per-period internally).
    mar     : minimum acceptable return per period (default 0).

    Returns
    -------
    float Sortino ratio.
    """
    r = _clean(_to_array(returns))
    if len(r) < 2:
        return 0.0

    rf_per = rf / periods_per_year
    excess = r - max(rf_per, mar)
    mu     = float(np.mean(excess))

    downside = excess[excess < 0]
    if len(downside) == 0:
        # No losing periods → infinite Sortino
        return np.inf if mu > 0 else 0.0

    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    if downside_dev < 1e-12:
        return 0.0

    return mu / downside_dev * math.sqrt(periods_per_year)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Calmar Ratio
# ─────────────────────────────────────────────────────────────────────────────

def calmar_ratio(
    returns:          object,
    max_dd:           Optional[float] = None,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the Calmar ratio: annualized return / |max drawdown|.

    Parameters
    ----------
    returns : 1-D period returns.
    max_dd  : pre-computed max drawdown (as a negative fraction, e.g. -0.25).
              If None, it is computed from the equity curve of `returns`.
    periods_per_year : annualization factor.

    Returns
    -------
    float Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    r = _clean(_to_array(returns))
    if len(r) < 2:
        return 0.0

    ann_return = float(np.mean(r)) * periods_per_year

    if max_dd is None:
        equity = np.cumprod(1.0 + r)
        max_dd = float(max_drawdown(equity))

    if abs(max_dd) < 1e-12:
        return np.inf if ann_return > 0 else 0.0

    return ann_return / abs(max_dd)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Max Drawdown
# ─────────────────────────────────────────────────────────────────────────────

def max_drawdown(equity_curve: object) -> float:
    """
    Compute the maximum drawdown of an equity curve.

    Parameters
    ----------
    equity_curve : cumulative equity values (e.g. starting at 1.0 or 100.0).
                   Must be strictly positive.

    Returns
    -------
    float Maximum drawdown as a negative fraction (e.g. -0.25 for -25%).
    Returns 0.0 if no drawdown exists.

    Examples
    --------
    >>> max_drawdown(np.array([1.0, 1.2, 0.9, 1.1]))
    -0.25
    """
    eq = _clean(_to_array(equity_curve))
    if len(eq) < 2:
        return 0.0

    rolling_max = np.maximum.accumulate(eq)
    dd          = (eq - rolling_max) / (rolling_max + 1e-12)
    return float(np.min(dd))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Drawdown Series
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_series(equity_curve: object) -> pd.Series:
    """
    Compute the full drawdown fraction series from an equity curve.

    Parameters
    ----------
    equity_curve : 1-D array or Series of equity values.

    Returns
    -------
    pd.Series of drawdown fractions (≤ 0), same index as equity_curve if Series.
    """
    if isinstance(equity_curve, pd.Series):
        idx = equity_curve.index
        eq  = equity_curve.to_numpy(dtype=np.float64)
    else:
        eq  = _to_array(equity_curve)
        idx = pd.RangeIndex(len(eq))

    eq          = np.where(np.isfinite(eq), eq, np.nan)
    rolling_max = pd.Series(eq).cummax().to_numpy()
    dd          = np.where(rolling_max > 0, (eq - rolling_max) / rolling_max, 0.0)
    return pd.Series(dd, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Underwater Periods
# ─────────────────────────────────────────────────────────────────────────────

def underwater_periods(equity_curve: object) -> List[UnderwaterPeriod]:
    """
    Identify all contiguous drawdown periods (underwater periods).

    A new period begins when the equity curve falls below a previous peak
    and ends when it recovers (or at the end of the series).

    Parameters
    ----------
    equity_curve : 1-D equity values.

    Returns
    -------
    List[UnderwaterPeriod] sorted by start_idx.
    """
    eq = _clean(_to_array(equity_curve))
    if len(eq) < 2:
        return []

    dd_series = drawdown_series(eq).to_numpy()
    periods: List[UnderwaterPeriod] = []

    in_drawdown = False
    peak_val    = eq[0]
    peak_idx    = 0
    trough_val  = eq[0]
    trough_idx  = 0
    start_idx   = 0

    for i, (val, dd) in enumerate(zip(eq, dd_series)):
        if not in_drawdown:
            if dd < -1e-6:
                in_drawdown = True
                start_idx   = i
                peak_val    = eq[max(0, i - 1)]
                peak_idx    = max(0, i - 1)
                trough_val  = val
                trough_idx  = i
        else:
            if val < trough_val:
                trough_val = val
                trough_idx = i
            if dd >= -1e-6:
                # Recovered
                dd_frac = (trough_val - peak_val) / (peak_val + 1e-12)
                periods.append(UnderwaterPeriod(
                    start_idx    = start_idx,
                    end_idx      = i,
                    peak_value   = peak_val,
                    trough_value = trough_val,
                    drawdown     = dd_frac,
                    duration     = i - start_idx,
                    recovery_idx = i,
                ))
                in_drawdown = False
                peak_val    = val
                peak_idx    = i

    if in_drawdown:
        dd_frac = (trough_val - peak_val) / (peak_val + 1e-12)
        periods.append(UnderwaterPeriod(
            start_idx    = start_idx,
            end_idx      = len(eq) - 1,
            peak_value   = peak_val,
            trough_value = trough_val,
            drawdown     = dd_frac,
            duration     = len(eq) - 1 - start_idx,
            recovery_idx = None,
        ))

    return periods


# ─────────────────────────────────────────────────────────────────────────────
# 7. Profit Factor
# ─────────────────────────────────────────────────────────────────────────────

def profit_factor(pnl_series: object) -> float:
    """
    Compute profit factor = gross profit / gross loss.

    Parameters
    ----------
    pnl_series : per-trade P&L values.

    Returns
    -------
    float ≥ 0. Returns 0.0 if no profitable trades. Returns inf if no losses.
    """
    pnl = _clean(_to_array(pnl_series))
    if len(pnl) == 0:
        return 0.0

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss   = float(abs(pnl[pnl < 0].sum()))

    if gross_loss < 1e-12:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


# ─────────────────────────────────────────────────────────────────────────────
# 8. Payoff Ratio
# ─────────────────────────────────────────────────────────────────────────────

def payoff_ratio(pnl_series: object) -> float:
    """
    Compute payoff ratio = avg win / avg loss (absolute value).

    Parameters
    ----------
    pnl_series : per-trade P&L values.

    Returns
    -------
    float ≥ 0.
    """
    pnl = _clean(_to_array(pnl_series))
    if len(pnl) < 2:
        return 0.0

    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    avg_win  = float(np.mean(wins))
    avg_loss = float(abs(np.mean(losses)))

    if avg_loss < 1e-12:
        return np.inf

    return avg_win / avg_loss


# ─────────────────────────────────────────────────────────────────────────────
# 9. Win Rate
# ─────────────────────────────────────────────────────────────────────────────

def win_rate(pnl_series: object) -> float:
    """
    Compute the win rate (fraction of trades with positive P&L).

    Parameters
    ----------
    pnl_series : per-trade P&L values.

    Returns
    -------
    float in [0, 1].
    """
    pnl = _clean(_to_array(pnl_series))
    if len(pnl) == 0:
        return 0.0
    return float(np.mean(pnl > 0))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Expectancy
# ─────────────────────────────────────────────────────────────────────────────

def expectancy(pnl_series: object) -> float:
    """
    Compute average trade expectancy (mean P&L per trade).

    Parameters
    ----------
    pnl_series : per-trade P&L values.

    Returns
    -------
    float average P&L.
    """
    pnl = _clean(_to_array(pnl_series))
    if len(pnl) == 0:
        return 0.0
    return float(np.mean(pnl))


# ─────────────────────────────────────────────────────────────────────────────
# 11. Kelly Fraction
# ─────────────────────────────────────────────────────────────────────────────

def kelly_fraction(pnl_series: object) -> float:
    """
    Compute the Kelly criterion optimal fraction.

    Kelly = W - (1 - W) / R

    where W = win rate, R = avg win / avg loss (payoff ratio).

    Parameters
    ----------
    pnl_series : per-trade P&L values.

    Returns
    -------
    float in [-inf, 1] (negative = do not trade; 1 = all-in per Kelly).
    Clipped to [-1, 1] for safety.
    """
    pnl = _clean(_to_array(pnl_series))
    if len(pnl) < 5:
        return 0.0

    W = win_rate(pnl)
    R = payoff_ratio(pnl)

    if R < 1e-12:
        return -1.0

    k = W - (1.0 - W) / R
    return float(np.clip(k, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 12. Omega Ratio
# ─────────────────────────────────────────────────────────────────────────────

def omega_ratio(returns: object, threshold: float = 0.0) -> float:
    """
    Compute the Omega ratio.

    Ω(τ) = E[max(r - τ, 0)] / E[max(τ - r, 0)]

    Parameters
    ----------
    returns   : 1-D array of period returns.
    threshold : minimum acceptable return per period (default 0).

    Returns
    -------
    float ≥ 0. Returns inf if all returns > threshold.
    """
    r = _clean(_to_array(returns))
    if len(r) == 0:
        return 0.0

    gains  = float(np.mean(np.maximum(r - threshold, 0.0)))
    losses = float(np.mean(np.maximum(threshold - r, 0.0)))

    if losses < 1e-12:
        return np.inf if gains > 0 else 0.0

    return gains / losses


# ─────────────────────────────────────────────────────────────────────────────
# 13. Tail Ratio
# ─────────────────────────────────────────────────────────────────────────────

def tail_ratio(returns: object, q: float = 0.05) -> float:
    """
    Compute the tail ratio: |95th percentile| / |5th percentile|.

    Measures the ratio of the right tail magnitude to the left tail magnitude.
    Values > 1 indicate positive skew (right tail dominates).

    Parameters
    ----------
    returns : 1-D array of period returns.
    q       : quantile for left tail (default 0.05 = 5th percentile).

    Returns
    -------
    float tail ratio.
    """
    r = _clean(_to_array(returns))
    if len(r) < 10:
        return 1.0

    upper = abs(float(np.percentile(r, (1.0 - q) * 100)))
    lower = abs(float(np.percentile(r, q * 100)))

    if lower < 1e-12:
        return np.inf if upper > 0 else 1.0

    return upper / lower


# ─────────────────────────────────────────────────────────────────────────────
# 14. Value at Risk
# ─────────────────────────────────────────────────────────────────────────────

def value_at_risk(returns: object, confidence: float = 0.95) -> float:
    """
    Compute the historical Value at Risk (VaR).

    VaR is the return threshold such that losses exceed it with probability
    (1 - confidence). Returned as a positive number.

    Parameters
    ----------
    returns    : 1-D array of period returns.
    confidence : confidence level (e.g. 0.95 for 95% VaR).

    Returns
    -------
    float VaR (positive value representing a potential loss).

    Examples
    --------
    >>> value_at_risk(np.random.normal(0, 0.01, 1000), confidence=0.95)
    ~0.0165
    """
    r = _clean(_to_array(returns))
    if len(r) < 10:
        return 0.0
    return float(-np.percentile(r, (1.0 - confidence) * 100))


# ─────────────────────────────────────────────────────────────────────────────
# 15. Conditional VaR (Expected Shortfall)
# ─────────────────────────────────────────────────────────────────────────────

def conditional_var(returns: object, confidence: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR = mean of returns in the worst (1-confidence) tail.
    Returned as a positive number.

    Parameters
    ----------
    returns    : 1-D array of period returns.
    confidence : confidence level (e.g. 0.95).

    Returns
    -------
    float CVaR (positive value).
    """
    r = _clean(_to_array(returns))
    if len(r) < 10:
        return 0.0

    var_threshold = -value_at_risk(r, confidence)
    tail          = r[r <= var_threshold]

    if len(tail) == 0:
        return value_at_risk(r, confidence)

    return float(-np.mean(tail))


# ─────────────────────────────────────────────────────────────────────────────
# 16. Hurst Exponent
# ─────────────────────────────────────────────────────────────────────────────

def hurst_exponent(prices: object, max_lag: int = 100) -> float:
    """
    Estimate the Hurst exponent via Rescaled Range (R/S) analysis.

    H < 0.5 → mean-reverting
    H = 0.5 → random walk
    H > 0.5 → trending / persistent

    Parameters
    ----------
    prices  : 1-D price series (not returns).
    max_lag : maximum lag for R/S computation (default 100).

    Returns
    -------
    float Hurst exponent H ∈ (0, 1).

    References
    ----------
    Hurst, H.E. (1951). Long-term storage capacity of reservoirs.
    Transactions of the American Society of Civil Engineers.
    """
    p = _clean(_to_array(prices))
    n = len(p)
    if n < 20:
        return 0.5

    max_lag = min(max_lag, n // 2)
    lags    = range(2, max_lag)
    rs_vals: List[float] = []

    for lag in lags:
        # Sub-series of length lag starting at each valid position
        sub_rs: List[float] = []
        for start in range(0, n - lag, lag):
            chunk = p[start:start + lag]
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            devs   = np.cumsum(chunk - mean_c)
            r      = float(devs.max() - devs.min())
            s      = float(np.std(chunk, ddof=1))
            if s > 1e-12:
                sub_rs.append(r / s)

        if sub_rs:
            rs_vals.append(np.mean(sub_rs))
        else:
            rs_vals.append(np.nan)

    valid_lags = [l for l, v in zip(lags, rs_vals) if np.isfinite(v)]
    valid_rs   = [v for v in rs_vals if np.isfinite(v)]

    if len(valid_lags) < 4:
        return 0.5

    log_lags = np.log(valid_lags)
    log_rs   = np.log(np.clip(valid_rs, 1e-12, None))

    # Linear regression of log(R/S) ~ H * log(lag)
    slope, *_ = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(slope, 0.01, 0.99))


# ─────────────────────────────────────────────────────────────────────────────
# 17. Ljung-Box Test
# ─────────────────────────────────────────────────────────────────────────────

def ljung_box_test(series: object, lags: int = 20) -> Tuple[float, float]:
    """
    Perform the Ljung-Box test for autocorrelation.

    H0: no autocorrelation up to `lags` lags.

    Parameters
    ----------
    series : 1-D time series.
    lags   : number of lags to test.

    Returns
    -------
    (statistic, p_value) : Ljung-Box Q statistic and p-value.
    """
    x = _clean(_to_array(series))
    if len(x) < lags + 5:
        return (np.nan, np.nan)

    from statsmodels.stats.diagnostic import acorr_ljungbox
    try:
        result = acorr_ljungbox(x, lags=[lags], return_df=True)
        stat   = float(result["lb_stat"].iloc[-1])
        pval   = float(result["lb_pvalue"].iloc[-1])
        return (stat, pval)
    except Exception:
        # Fallback: manual Ljung-Box
        n    = len(x)
        mean = np.mean(x)
        c0   = np.var(x, ddof=0)
        if c0 < 1e-12:
            return (0.0, 1.0)
        q = 0.0
        for k in range(1, lags + 1):
            ck = np.mean((x[:n-k] - mean) * (x[k:] - mean))
            rk = ck / c0
            q += rk ** 2 / (n - k)
        q    *= n * (n + 2)
        pval  = float(1.0 - stats.chi2.cdf(q, df=lags))
        return (float(q), pval)


# ─────────────────────────────────────────────────────────────────────────────
# 18. Jarque-Bera Test
# ─────────────────────────────────────────────────────────────────────────────

def jarque_bera_test(series: object) -> Tuple[float, float]:
    """
    Perform the Jarque-Bera normality test.

    H0: the series follows a normal distribution (skewness=0, excess kurtosis=0).

    Parameters
    ----------
    series : 1-D array of values.

    Returns
    -------
    (statistic, p_value)
    """
    x = _clean(_to_array(series))
    if len(x) < 8:
        return (np.nan, np.nan)

    stat, pval = stats.jarque_bera(x)
    return (float(stat), float(pval))


# ─────────────────────────────────────────────────────────────────────────────
# 19. Augmented Dickey-Fuller Test
# ─────────────────────────────────────────────────────────────────────────────

def augmented_dickey_fuller(
    series: object,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Perform the Augmented Dickey-Fuller (ADF) unit root test.

    H0: series has a unit root (non-stationary).

    Parameters
    ----------
    series : 1-D array of values (e.g., price series).

    Returns
    -------
    (statistic, p_value, critical_values) where critical_values is a dict
    with keys '1%', '5%', '10%'.
    """
    x = _clean(_to_array(series))
    if len(x) < 20:
        return (np.nan, np.nan, {})

    from statsmodels.tsa.stattools import adfuller
    try:
        result = adfuller(x, autolag="AIC")
        return (
            float(result[0]),
            float(result[1]),
            {k: float(v) for k, v in result[4].items()},
        )
    except Exception as e:
        logger.warning("ADF test failed: %s", e)
        return (np.nan, np.nan, {})


# ─────────────────────────────────────────────────────────────────────────────
# 20. Newey-West Standard Error
# ─────────────────────────────────────────────────────────────────────────────

def newey_west_se(series: object, lags: int = 5) -> float:
    """
    Compute Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC)
    standard error for the mean of a series.

    Accounts for serial correlation in the standard error estimate, which is
    critical for financial return series exhibiting autocorrelation.

    Parameters
    ----------
    series : 1-D return or P&L series.
    lags   : bandwidth (number of lags), default 5.

    Returns
    -------
    float Newey-West standard error.
    """
    x = _clean(_to_array(series))
    n = len(x)
    if n < lags + 5:
        return float(np.std(x, ddof=1) / math.sqrt(max(1, n)))

    mean   = np.mean(x)
    demean = x - mean

    # Estimate long-run variance using Bartlett kernel
    gamma_0 = float(np.mean(demean ** 2))
    lrv     = gamma_0

    for lag in range(1, lags + 1):
        weight   = 1.0 - lag / (lags + 1)          # Bartlett weight
        gamma_l  = float(np.mean(demean[lag:] * demean[:n - lag]))
        lrv     += 2.0 * weight * gamma_l

    lrv = max(lrv, 1e-20)
    return float(math.sqrt(lrv / n))


# ─────────────────────────────────────────────────────────────────────────────
# 21. Bootstrap Confidence Interval
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray], float],
    data:      object,
    n_boot:    int   = 1000,
    ci:        float = 0.95,
    seed:      int   = 42,
) -> Tuple[float, float]:
    """
    Compute a bootstrap confidence interval for any scalar metric.

    Uses the percentile bootstrap method (not bias-corrected; adequate for
    most trading metric applications).

    Parameters
    ----------
    metric_fn : callable(array) → float. The metric to estimate.
    data      : 1-D data array.
    n_boot    : number of bootstrap resamples (default 1000).
    ci        : confidence level (default 0.95 for 95% CI).
    seed      : random seed for reproducibility.

    Returns
    -------
    (lower, upper) confidence interval bounds.

    Examples
    --------
    >>> ci = bootstrap_confidence_interval(sharpe_ratio, daily_returns, n_boot=2000)
    >>> print(f"Sharpe 95% CI: {ci[0]:.2f} – {ci[1]:.2f}")
    """
    rng = np.random.default_rng(seed)
    arr = _clean(_to_array(data))
    n   = len(arr)

    if n < 10:
        pt = metric_fn(arr)
        return (pt, pt)

    boot_stats: List[float] = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        try:
            stat = float(metric_fn(sample))
            if np.isfinite(stat):
                boot_stats.append(stat)
        except Exception:
            pass

    if len(boot_stats) < 10:
        pt = metric_fn(arr)
        return (pt, pt)

    alpha   = (1.0 - ci) / 2.0
    lower   = float(np.percentile(boot_stats, alpha * 100))
    upper   = float(np.percentile(boot_stats, (1.0 - alpha) * 100))
    return (lower, upper)


# ─────────────────────────────────────────────────────────────────────────────
# 22. Deflated Sharpe Ratio (Bailey-López de Prado 2014)
# ─────────────────────────────────────────────────────────────────────────────

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials:        int,
    skewness:        float,
    kurtosis:        float,
    n_obs:           int,
) -> float:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    The DSR adjusts the observed Sharpe for multiple-testing bias (the
    "inflation" of the Sharpe due to selection from many trial strategies).

    DSR = P[SR* > SR̂_max] where SR̂_max is the expected maximum Sharpe
    from N independent trials.

    Parameters
    ----------
    observed_sharpe : the observed (best) Sharpe ratio from N trials.
    n_trials        : total number of strategies/parameter combinations tried.
    skewness        : skewness of the return distribution.
    kurtosis        : excess kurtosis of the return distribution.
    n_obs           : number of observations used to compute the Sharpe.

    Returns
    -------
    float DSR in [0, 1]. Values > 0.95 suggest genuine edge.

    References
    ----------
    Bailey, D.H. & López de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
    Journal of Portfolio Management.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be ≥ 1, got {n_trials}")
    if n_obs < 5:
        return 0.0

    # Expected maximum Sharpe from N iid normal trials
    euler_mascheroni = 0.5772156649
    e_max = (
        (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )

    # Variance of Sharpe estimator (non-normal correction)
    # V[SR] ≈ (1 + 0.5*SR^2*(k4-1) - SR*k3) / (n-1)
    sr    = observed_sharpe
    k3    = skewness        # third cumulant / std^3
    k4    = kurtosis + 3.0  # total kurtosis (not excess)

    var_sr = (1.0 + 0.5 * sr ** 2 * (k4 - 1.0) - sr * k3) / max(n_obs - 1, 1)
    std_sr = math.sqrt(max(var_sr, 1e-12))

    # DSR = Φ((SR_observed - E[max]) / std_SR)
    z    = (sr - e_max) / std_sr
    dsr  = float(stats.norm.cdf(z))
    return float(np.clip(dsr, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# 23. Probability of Backtest Overfitting (PBO)
# ─────────────────────────────────────────────────────────────────────────────

def probability_of_backtest_overfitting(
    is_sharpes:  object,
    oos_sharpes: object,
) -> float:
    """
    Estimate the Probability of Backtest Overfitting (PBO).

    PBO measures the fraction of IS-best strategies that underperform in OOS.
    A PBO > 0.5 indicates systematic overfitting.

    Algorithm (López de Prado 2018, Ch.14):
    1. For each CPCV path, identify the IS-best parameter configuration.
    2. Check if it ranks below median in OOS performance.
    3. PBO = fraction of paths where IS winner is OOS loser.

    Parameters
    ----------
    is_sharpes  : array of in-sample Sharpe ratios (one per strategy/path).
    oos_sharpes : array of out-of-sample Sharpe ratios (same length).

    Returns
    -------
    float PBO in [0, 1]. Values near 0 → low overfitting; near 1 → high.

    Notes
    -----
    This simplified version pairs each IS Sharpe with the corresponding OOS
    Sharpe (aligned by strategy/path). The full CPCV version requires the
    combinatorial path structure from CPCVSplitter.
    """
    is_arr  = _clean(_to_array(is_sharpes))
    oos_arr = _clean(_to_array(oos_sharpes))

    if len(is_arr) != len(oos_arr):
        raise ValueError(
            f"is_sharpes length ({len(is_arr)}) ≠ oos_sharpes length ({len(oos_arr)})"
        )
    if len(is_arr) == 0:
        return np.nan

    # For each "path" (row), find IS winner index and check its OOS rank
    n         = len(is_arr)
    is_winner = int(np.argmax(is_arr))

    oos_median = float(np.median(oos_arr))
    oos_winner = oos_arr[is_winner]

    # Single-comparison version: fraction of bootstrap resamples where IS
    # winner's OOS < median OOS
    rng = np.random.default_rng(42)
    n_boot = 1000
    pbo_count = 0

    for _ in range(n_boot):
        idx        = rng.choice(n, size=n, replace=True)
        is_boot    = is_arr[idx]
        oos_boot   = oos_arr[idx]
        best_boot  = int(np.argmax(is_boot))
        oos_median_boot = float(np.median(oos_boot))
        if oos_boot[best_boot] < oos_median_boot:
            pbo_count += 1

    return pbo_count / n_boot


# ─────────────────────────────────────────────────────────────────────────────
# 24. CAGR from equity curve
# ─────────────────────────────────────────────────────────────────────────────

def cagr(
    equity_curve:     object,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR) from an equity curve.

    CAGR = (final / initial)^(periods_per_year / n_periods) - 1

    Parameters
    ----------
    equity_curve     : 1-D equity values (starting value can be any positive number).
    periods_per_year : trading periods per year (default 252 for daily).

    Returns
    -------
    float CAGR as a decimal (e.g. 0.25 for 25%).
    """
    eq = _clean(_to_array(equity_curve))
    if len(eq) < 2:
        return 0.0

    initial = float(eq[0])
    final   = float(eq[-1])
    n       = len(eq) - 1

    if initial <= 0 or final <= 0:
        return 0.0

    ratio = final / initial
    years = n / periods_per_year
    if years <= 0:
        return 0.0

    return float(ratio ** (1.0 / years) - 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 25. Full Performance Summary
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PerformanceStats:
    """Complete performance statistics for a trade series or equity curve."""
    sharpe:           float = 0.0
    sortino:          float = 0.0
    calmar:           float = 0.0
    cagr_ann:         float = 0.0
    max_dd:           float = 0.0
    profit_factor_:   float = 0.0
    payoff_ratio_:    float = 0.0
    win_rate_:        float = 0.0
    expectancy_:      float = 0.0
    kelly:            float = 0.0
    omega:            float = 0.0
    tail_ratio_:      float = 0.0
    var_95:           float = 0.0
    cvar_95:          float = 0.0
    n_trades:         int   = 0
    total_pnl:        float = 0.0
    hurst:            float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "sharpe":        self.sharpe,
            "sortino":       self.sortino,
            "calmar":        self.calmar,
            "cagr_ann":      self.cagr_ann,
            "max_drawdown":  self.max_dd,
            "profit_factor": self.profit_factor_,
            "payoff_ratio":  self.payoff_ratio_,
            "win_rate":      self.win_rate_,
            "expectancy":    self.expectancy_,
            "kelly":         self.kelly,
            "omega":         self.omega,
            "tail_ratio":    self.tail_ratio_,
            "var_95":        self.var_95,
            "cvar_95":       self.cvar_95,
            "n_trades":      float(self.n_trades),
            "total_pnl":     self.total_pnl,
            "hurst":         self.hurst,
        }


def compute_performance_stats(
    trades:           pd.DataFrame,
    starting_equity:  float = 100_000.0,
    periods_per_year: int   = 252,
    pnl_col:          str   = "pnl",
    dollar_pos_col:   str   = "dollar_pos",
) -> PerformanceStats:
    """
    Compute comprehensive performance statistics from a trade DataFrame.

    Parameters
    ----------
    trades           : DataFrame with at minimum a `pnl` column.
    starting_equity  : initial portfolio equity for equity curve construction.
    periods_per_year : annualization factor.
    pnl_col          : name of the P&L column.
    dollar_pos_col   : name of the dollar position column (for return calc).

    Returns
    -------
    PerformanceStats dataclass.
    """
    if trades is None or len(trades) == 0:
        return PerformanceStats()

    pnl_vals = trades[pnl_col].to_numpy(dtype=np.float64) if pnl_col in trades.columns else np.array([])
    pos_vals = trades[dollar_pos_col].to_numpy(dtype=np.float64) if dollar_pos_col in trades.columns else None

    if len(pnl_vals) == 0:
        return PerformanceStats()

    # Build equity curve from cumulative P&L
    equity_curve = starting_equity + np.cumsum(pnl_vals)
    equity_curve = np.concatenate([[starting_equity], equity_curve])

    # Compute returns as pnl / dollar_pos (or pnl / equity)
    if pos_vals is not None and np.all(np.abs(pos_vals) > 1e-6):
        returns = pnl_vals / pos_vals
    else:
        eq_before = equity_curve[:-1]
        returns   = np.where(eq_before > 0, pnl_vals / eq_before, 0.0)

    returns = np.where(np.isfinite(returns), returns, 0.0)

    max_dd_val = max_drawdown(equity_curve)
    hurst_val  = hurst_exponent(equity_curve) if len(equity_curve) >= 20 else 0.5

    return PerformanceStats(
        sharpe         = sharpe_ratio(returns, periods_per_year=periods_per_year),
        sortino        = sortino_ratio(returns, periods_per_year=periods_per_year),
        calmar         = calmar_ratio(returns, max_dd=max_dd_val, periods_per_year=periods_per_year),
        cagr_ann       = cagr(equity_curve, periods_per_year=periods_per_year),
        max_dd         = max_dd_val,
        profit_factor_ = profit_factor(pnl_vals),
        payoff_ratio_  = payoff_ratio(pnl_vals),
        win_rate_      = win_rate(pnl_vals),
        expectancy_    = expectancy(pnl_vals),
        kelly          = kelly_fraction(pnl_vals),
        omega          = omega_ratio(returns),
        tail_ratio_    = tail_ratio(returns),
        var_95         = value_at_risk(returns, 0.95),
        cvar_95        = conditional_var(returns, 0.95),
        n_trades       = len(trades),
        total_pnl      = float(pnl_vals.sum()),
        hurst          = hurst_val,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 26. Newey-West t-statistic
# ─────────────────────────────────────────────────────────────────────────────

def newey_west_t_stat(series: object, lags: int = 5) -> float:
    """
    Compute the Newey-West t-statistic for the mean of a series being zero.

    t = mean(series) / NW_standard_error

    Useful for testing whether mean returns are significantly different from 0
    while accounting for serial correlation.

    Parameters
    ----------
    series : 1-D return or P&L series.
    lags   : bandwidth for Newey-West SE.

    Returns
    -------
    float t-statistic.
    """
    x = _clean(_to_array(series))
    if len(x) < lags + 5:
        return 0.0

    mu = float(np.mean(x))
    se = newey_west_se(x, lags=lags)

    if se < 1e-12:
        return 0.0

    return mu / se
