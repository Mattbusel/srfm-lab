"""
research/signal_analytics/utils.py
====================================
Utility functions for the signal analytics framework.

Provides:
  - Trade preprocessing and validation
  - Return normalisation helpers
  - Panel data alignment utilities
  - Statistical helpers (Newey-West, DM test, etc.)
  - Synthetic data generators for testing
  - Common financial metrics (Sharpe, Sortino, Calmar, max drawdown)
  - Rolling statistics utilities
  - Factor standardisation helpers

Usage example
-------------
>>> from research.signal_analytics.utils import (
...     normalize_returns, compute_max_drawdown, validate_trades,
...     generate_synthetic_trades, sharpe_ratio,
... )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Trade validation
# ---------------------------------------------------------------------------

REQUIRED_TRADE_COLUMNS = {"pnl"}
OPTIONAL_TRADE_COLUMNS = {
    "exit_time", "sym", "entry_price", "exit_price",
    "dollar_pos", "hold_bars", "regime",
    "tf_score", "mass", "ATR", "ensemble_signal", "delta_score",
}


def validate_trades(
    trades: pd.DataFrame,
    required: Optional[set[str]] = None,
    warn_missing: Optional[set[str]] = None,
    raise_on_error: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate a trades DataFrame for signal analytics.

    Parameters
    ----------
    trades        : trade records DataFrame
    required      : columns that must be present (default: {'pnl'})
    warn_missing  : columns to warn about if missing (default: OPTIONAL_TRADE_COLUMNS)
    raise_on_error: raise ValueError if required columns are missing

    Returns
    -------
    (is_valid, list_of_issues)
    """
    required = required or REQUIRED_TRADE_COLUMNS
    warn_missing = warn_missing or OPTIONAL_TRADE_COLUMNS

    issues: list[str] = []
    missing_required = required - set(trades.columns)
    if missing_required:
        msg = f"Missing required columns: {missing_required}"
        issues.append(msg)
        if raise_on_error:
            raise ValueError(msg)

    missing_optional = warn_missing - set(trades.columns)
    if missing_optional:
        issues.append(f"Missing optional columns (reduced functionality): {missing_optional}")

    # Check for all-NaN columns
    for col in trades.columns:
        if trades[col].isna().all():
            issues.append(f"Column '{col}' is all-NaN")

    # Check for sufficient observations
    if len(trades) < 10:
        issues.append(f"Very few observations ({len(trades)}); results may not be reliable")

    return len([i for i in issues if "required" in i.lower()]) == 0, issues


# ---------------------------------------------------------------------------
# Return normalisation
# ---------------------------------------------------------------------------

def normalize_returns(
    trades: pd.DataFrame,
    return_col: str = "pnl",
    dollar_pos_col: str = "dollar_pos",
    method: str = "pos_size",
) -> pd.Series:
    """Normalise trade P&L into returns.

    Parameters
    ----------
    trades        : trade records
    return_col    : column containing raw P&L
    dollar_pos_col: column containing dollar position size
    method        : 'pos_size' (P&L / |pos|), 'log' (log(exit/entry)),
                    'simple' (exit/entry - 1), or 'raw' (no normalisation)

    Returns
    -------
    pd.Series of normalised returns
    """
    if method == "pos_size":
        if dollar_pos_col in trades.columns:
            pos = trades[dollar_pos_col].abs().replace(0, np.nan)
            return (trades[return_col] / pos).rename("return")
        return trades[return_col].rename("return")

    if method == "log":
        if "entry_price" in trades.columns and "exit_price" in trades.columns:
            entry = trades["entry_price"].replace(0, np.nan)
            exit_ = trades["exit_price"].replace(0, np.nan)
            return np.log(exit_ / entry).rename("return")
        return trades[return_col].rename("return")

    if method == "simple":
        if "entry_price" in trades.columns and "exit_price" in trades.columns:
            entry = trades["entry_price"].replace(0, np.nan)
            exit_ = trades["exit_price"].replace(0, np.nan)
            return (exit_ / entry - 1).rename("return")
        return trades[return_col].rename("return")

    # raw
    return trades[return_col].rename("return")


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free: float = 0.0,
    bars_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns      : return series (one value per period)
    risk_free    : risk-free rate per period (default 0)
    bars_per_year: number of periods per year
    ddof         : degrees of freedom for std

    Returns
    -------
    float annualised Sharpe
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free
    std = np.std(excess, ddof=ddof)
    if std == 0:
        return float("nan")
    return float(np.mean(excess) / std * np.sqrt(bars_per_year))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free: float = 0.0,
    bars_per_year: int = 252,
) -> float:
    """Annualised Sortino ratio (downside deviation in denominator).

    Parameters
    ----------
    returns      : return series
    risk_free    : risk-free rate per period
    bars_per_year: number of periods per year

    Returns
    -------
    float annualised Sortino
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    down_std = np.std(downside, ddof=1)
    if down_std == 0:
        return float("nan")
    return float(np.mean(excess) / down_std * np.sqrt(bars_per_year))


def calmar_ratio(
    returns: pd.Series | np.ndarray,
    bars_per_year: int = 252,
) -> float:
    """Calmar ratio = annualised return / |max drawdown|.

    Parameters
    ----------
    returns      : return series
    bars_per_year: periods per year

    Returns
    -------
    float Calmar ratio
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    ann_ret = float(np.mean(r) * bars_per_year)
    mdd = compute_max_drawdown(r)
    if mdd == 0:
        return float("inf")
    return ann_ret / abs(mdd)


def compute_max_drawdown(
    returns: pd.Series | np.ndarray,
    cumulative: bool = False,
) -> float:
    """Maximum drawdown of a return series.

    Parameters
    ----------
    returns    : return or cumulative return series
    cumulative : if True, input is already cumulative (wealth index)

    Returns
    -------
    float max drawdown (negative number)
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return 0.0

    if not cumulative:
        wealth = np.cumprod(1 + r)
    else:
        wealth = r

    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1
    return float(drawdown.min())


def hit_rate(
    returns: pd.Series | np.ndarray,
) -> float:
    """Fraction of positive returns (win rate).

    Parameters
    ----------
    returns : return series

    Returns
    -------
    float in [0, 1]
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    return float(np.mean(r > 0))


def profit_factor(
    returns: pd.Series | np.ndarray,
) -> float:
    """Profit factor = sum(gains) / |sum(losses)|.

    Parameters
    ----------
    returns : return series

    Returns
    -------
    float profit factor
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def performance_summary(
    returns: pd.Series | np.ndarray,
    bars_per_year: int = 252,
    risk_free: float = 0.0,
) -> Dict[str, float]:
    """Compute a comprehensive performance summary.

    Parameters
    ----------
    returns      : return series
    bars_per_year: periods per year
    risk_free    : risk-free rate per period

    Returns
    -------
    Dict with keys: sharpe, sortino, calmar, max_drawdown, hit_rate,
                    profit_factor, mean_return, std_return, skewness, kurtosis,
                    n_obs, total_return, ann_return
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {}

    return {
        "sharpe": sharpe_ratio(r, risk_free=risk_free, bars_per_year=bars_per_year),
        "sortino": sortino_ratio(r, risk_free=risk_free, bars_per_year=bars_per_year),
        "calmar": calmar_ratio(r, bars_per_year=bars_per_year),
        "max_drawdown": compute_max_drawdown(r),
        "hit_rate": hit_rate(r),
        "profit_factor": profit_factor(r),
        "mean_return": float(np.mean(r)),
        "std_return": float(np.std(r, ddof=1)),
        "skewness": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r)),
        "n_obs": len(r),
        "total_return": float(np.sum(r)),
        "ann_return": float(np.mean(r) * bars_per_year),
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def newey_west_se(
    series: np.ndarray,
    n_lags: int = 5,
) -> float:
    """Newey-West HAC standard error for a time-series mean estimate.

    Parameters
    ----------
    series : observations (should be zero-mean or the covariance is of the mean)
    n_lags : Bartlett kernel lag truncation

    Returns
    -------
    float standard error of the mean
    """
    n = len(series)
    if n < 2:
        return float("nan")
    demeaned = series - series.mean()
    s0 = np.dot(demeaned, demeaned) / n
    s_sum = s0
    for lag in range(1, n_lags + 1):
        w = 1 - lag / (n_lags + 1)
        cov_lag = np.dot(demeaned[lag:], demeaned[: n - lag]) / n
        s_sum += 2 * w * cov_lag
    var_mean = max(s_sum / n, 0.0)
    return float(np.sqrt(var_mean))


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: MSE(model_a) == MSE(model_b) vs H1: model_a is better.

    Parameters
    ----------
    errors_a : prediction errors from model A
    errors_b : prediction errors from model B
    h        : forecast horizon (used for HAC correction)

    Returns
    -------
    (DM_statistic, p_value)
    """
    e_a = np.asarray(errors_a, dtype=float)
    e_b = np.asarray(errors_b, dtype=float)
    if len(e_a) != len(e_b):
        raise ValueError("errors_a and errors_b must have the same length")

    d = e_a**2 - e_b**2
    n = len(d)
    se = newey_west_se(d, n_lags=h - 1) * np.sqrt(n)
    if se == 0:
        return float("nan"), float("nan")

    dm_stat = float(np.mean(d) / se)
    p_val = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return dm_stat, p_val


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    bars_per_year: int = 252,
) -> pd.Series:
    """Rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns      : return series
    window       : rolling window
    bars_per_year: periods per year

    Returns
    -------
    pd.Series of rolling Sharpe values
    """
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=1)
    sharpe = mu / sigma * np.sqrt(bars_per_year)
    return sharpe.rename("rolling_sharpe")


def rolling_drawdown(
    returns: pd.Series,
) -> pd.Series:
    """Rolling maximum drawdown at each point in time.

    Parameters
    ----------
    returns : return series

    Returns
    -------
    pd.Series of drawdown values (non-positive)
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.rename("drawdown")


# ---------------------------------------------------------------------------
# Panel alignment
# ---------------------------------------------------------------------------

def align_signal_and_returns(
    signal_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    dropna_thresh: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align signal and returns panels, dropping columns with too many NaNs.

    Parameters
    ----------
    signal_df     : DataFrame[time x assets] of signal values
    returns_df    : DataFrame[time x assets] of returns
    dropna_thresh : drop assets with more than this fraction of NaN rows

    Returns
    -------
    (signal_aligned, returns_aligned) — same index and columns
    """
    common_cols = signal_df.columns.intersection(returns_df.columns)
    common_idx = signal_df.index.intersection(returns_df.index)

    sig = signal_df.loc[common_idx, common_cols]
    ret = returns_df.loc[common_idx, common_cols]

    # Drop columns with too many NaNs
    n = len(common_idx)
    good_sig = sig.columns[sig.isna().mean() <= dropna_thresh]
    good_ret = ret.columns[ret.isna().mean() <= dropna_thresh]
    keep_cols = good_sig.intersection(good_ret)

    return sig[keep_cols], ret[keep_cols]


def winsorise_panel(
    df: pd.DataFrame,
    n_std: float = 5.0,
    axis: int = 0,
) -> pd.DataFrame:
    """Winsorise each column at +/- n_std standard deviations.

    Parameters
    ----------
    df    : DataFrame to winsorise
    n_std : number of standard deviations for clip bounds
    axis  : 0 = clip each column; 1 = clip each row cross-sectionally

    Returns
    -------
    pd.DataFrame with clipped values
    """
    result = df.copy()
    if axis == 0:
        for col in result.columns:
            mu = result[col].mean()
            sigma = result[col].std(ddof=1)
            result[col] = result[col].clip(mu - n_std * sigma, mu + n_std * sigma)
    else:
        for idx in result.index:
            row = result.loc[idx]
            mu = row.mean()
            sigma = row.std(ddof=1)
            result.loc[idx] = row.clip(mu - n_std * sigma, mu + n_std * sigma)
    return result


def cross_sectional_zscore(
    df: pd.DataFrame,
    clip_extreme: float = 3.0,
) -> pd.DataFrame:
    """Cross-sectional z-score normalisation (at each time step).

    Parameters
    ----------
    df           : DataFrame[time x assets]
    clip_extreme : clip z-scores at this magnitude (default 3.0)

    Returns
    -------
    pd.DataFrame of z-scored values
    """
    mu = df.mean(axis=1)
    sigma = df.std(axis=1, ddof=1)
    z = df.sub(mu, axis=0).div(sigma.replace(0, np.nan), axis=0)
    return z.clip(-clip_extreme, clip_extreme)


# ---------------------------------------------------------------------------
# Factor standardisation
# ---------------------------------------------------------------------------

def standardise_factors(
    factor_df: pd.DataFrame,
    method: str = "zscore",
    clip: float = 3.0,
) -> pd.DataFrame:
    """Standardise factor exposures using specified method.

    Parameters
    ----------
    factor_df : DataFrame[obs x factors]
    method    : 'zscore' (zero mean, unit std), 'rank' (percentile rank),
                'mad' (median absolute deviation), or 'minmax'
    clip      : clip z-scores after standardisation (zscore method)

    Returns
    -------
    pd.DataFrame of standardised factors
    """
    result = factor_df.copy()
    for col in result.columns:
        s = result[col]
        if method == "zscore":
            mu = s.mean()
            sigma = s.std(ddof=1)
            if sigma > 0:
                result[col] = ((s - mu) / sigma).clip(-clip, clip)
        elif method == "rank":
            result[col] = s.rank(pct=True)
        elif method == "mad":
            med = s.median()
            mad = (s - med).abs().median()
            if mad > 0:
                result[col] = (s - med) / (1.4826 * mad)  # scale MAD to ~std
        elif method == "minmax":
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                result[col] = (s - s_min) / (s_max - s_min)
    return result


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_synthetic_trades(
    n_trades: int = 500,
    n_symbols: int = 10,
    ic: float = 0.05,
    signal_vol: float = 1.0,
    return_vol: float = 0.02,
    seed: int = 42,
    include_bh_signals: bool = True,
) -> pd.DataFrame:
    """Generate synthetic trade records for testing signal analytics.

    Generates trades with a configurable information coefficient between
    ensemble_signal and normalised P&L.

    Parameters
    ----------
    n_trades          : number of synthetic trades
    n_symbols         : number of synthetic instruments
    ic                : target Spearman IC between signal and returns
    signal_vol        : signal standard deviation
    return_vol        : return standard deviation
    seed              : random seed for reproducibility
    include_bh_signals: if True, include tf_score, mass, ATR, delta_score

    Returns
    -------
    pd.DataFrame with columns matching BH trade schema
    """
    rng = np.random.default_rng(seed)

    # Generate correlated signal and return
    corr_matrix = np.array([[1.0, ic], [ic, 1.0]])
    L = np.linalg.cholesky(corr_matrix)
    z = rng.standard_normal((n_trades, 2))
    correlated = z @ L.T

    signal = correlated[:, 0] * signal_vol
    returns_norm = correlated[:, 1] * return_vol

    symbols = [f"SYM_{i:02d}" for i in range(n_symbols)]
    syms = rng.choice(symbols, size=n_trades)

    dollar_pos = rng.uniform(1000, 50000, size=n_trades)
    pnl = returns_norm * dollar_pos

    entry_price = rng.uniform(10, 1000, size=n_trades)
    exit_price = entry_price * (1 + returns_norm)

    hold_bars = rng.integers(1, 50, size=n_trades)

    exit_dates = pd.date_range("2022-01-01", periods=n_trades, freq="1h")

    regimes = rng.choice(["bull", "bear", "sideways"], size=n_trades, p=[0.4, 0.3, 0.3])

    df = pd.DataFrame({
        "exit_time": exit_dates,
        "sym": syms,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "dollar_pos": dollar_pos,
        "pnl": pnl,
        "hold_bars": hold_bars,
        "regime": regimes,
        "ensemble_signal": signal,
    })

    if include_bh_signals:
        tf_score = rng.integers(0, 8, size=n_trades).astype(float)
        mass = rng.uniform(0, 2, size=n_trades)
        atr = rng.uniform(0.001, 0.05, size=n_trades) * entry_price
        vol = rng.uniform(0.01, 0.05, size=n_trades)
        delta_score = tf_score * mass * atr / np.maximum(vol ** 2, 1e-8)

        df["tf_score"] = tf_score
        df["mass"] = mass
        df["ATR"] = atr
        df["vol"] = vol
        df["delta_score"] = delta_score

    return df


def generate_synthetic_panel(
    n_periods: int = 252,
    n_assets: int = 20,
    ic: float = 0.05,
    signal_persistence: float = 0.8,
    return_vol: float = 0.02,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic signal and return panels for testing.

    Parameters
    ----------
    n_periods          : number of time periods
    n_assets           : number of assets
    ic                 : target IC between signal and forward returns
    signal_persistence : AR(1) coefficient for signal time-series
    return_vol         : return volatility
    seed               : random seed

    Returns
    -------
    (signal_df, returns_df) — both DataFrame[time x assets]
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    assets = [f"ASSET_{i:02d}" for i in range(n_assets)]

    # AR(1) signal process
    sig = np.zeros((n_periods, n_assets))
    sig[0] = rng.standard_normal(n_assets)
    for t in range(1, n_periods):
        sig[t] = signal_persistence * sig[t - 1] + np.sqrt(1 - signal_persistence**2) * rng.standard_normal(n_assets)

    # Returns with embedded IC
    corr_matrix = np.array([[1.0, ic], [ic, 1.0]])
    L = np.linalg.cholesky(corr_matrix)
    ret = np.zeros((n_periods, n_assets))
    for a in range(n_assets):
        z = rng.standard_normal((n_periods, 2))
        c = z @ L.T
        ret[:, a] = c[:, 1] * return_vol

    signal_df = pd.DataFrame(sig, index=dates, columns=assets)
    returns_df = pd.DataFrame(ret, index=dates, columns=assets)
    return signal_df, returns_df


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def rolling_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling beta of asset returns to benchmark.

    Parameters
    ----------
    asset_returns     : asset return series
    benchmark_returns : benchmark return series
    window            : rolling window

    Returns
    -------
    pd.Series of rolling beta values
    """
    df = pd.concat({"asset": asset_returns, "bench": benchmark_returns}, axis=1).dropna()
    betas: list[float] = []
    for i in range(window - 1, len(df)):
        w = df.iloc[i - window + 1 : i + 1]
        cov = np.cov(w["asset"], w["bench"])
        var_b = np.var(w["bench"], ddof=1)
        betas.append(float(cov[0, 1] / var_b) if var_b > 0 else float("nan"))
    return pd.Series(betas, index=df.index[window - 1 :], name="rolling_beta")


def rolling_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 60,
    method: str = "pearson",
) -> pd.Series:
    """Rolling correlation between two series.

    Parameters
    ----------
    series_a : first series
    series_b : second series
    window   : rolling window
    method   : 'pearson' or 'spearman'

    Returns
    -------
    pd.Series of rolling correlation values
    """
    df = pd.concat({"a": series_a, "b": series_b}, axis=1).dropna()
    corr_vals: list[float] = []
    for i in range(window - 1, len(df)):
        w = df.iloc[i - window + 1 : i + 1]
        if method == "pearson":
            r, _ = stats.pearsonr(w["a"], w["b"])
        else:
            r, _ = stats.spearmanr(w["a"], w["b"])
        corr_vals.append(float(r))
    return pd.Series(corr_vals, index=df.index[window - 1 :], name="rolling_corr")


def ewm_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    halflife: int = 20,
) -> float:
    """Exponentially weighted IC (more recent observations weighted higher).

    Parameters
    ----------
    signal          : signal values indexed by time
    forward_returns : forward return values indexed by time
    halflife        : EWM half-life in periods

    Returns
    -------
    float EWM-weighted IC
    """
    df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
    if len(df) < 3:
        return float("nan")

    # EWM weights: w_t = decay^(n-t), where decay = 0.5^(1/halflife)
    n = len(df)
    decay = 0.5 ** (1 / halflife)
    exponents = np.arange(n - 1, -1, -1)
    weights = decay ** exponents
    weights /= weights.sum()

    sig_vals = df["sig"].values
    ret_vals = df["ret"].values

    # Weighted rank correlation approximation (weighted Spearman)
    sig_ranks = stats.rankdata(sig_vals)
    ret_ranks = stats.rankdata(ret_vals)

    # Weighted Pearson on ranks
    ws_mean = np.sum(weights * sig_ranks)
    wr_mean = np.sum(weights * ret_ranks)
    ws_var = np.sum(weights * (sig_ranks - ws_mean) ** 2)
    wr_var = np.sum(weights * (ret_ranks - wr_mean) ** 2)
    cov = np.sum(weights * (sig_ranks - ws_mean) * (ret_ranks - wr_mean))

    denom = np.sqrt(ws_var * wr_var)
    if denom == 0:
        return float("nan")
    return float(cov / denom)


# ---------------------------------------------------------------------------
# Regime detection helpers
# ---------------------------------------------------------------------------

def label_volatility_regimes(
    returns: pd.Series,
    window: int = 20,
    high_threshold: float = 0.75,
    low_threshold: float = 0.25,
) -> pd.Series:
    """Label each time period as 'high_vol', 'low_vol', or 'medium_vol'.

    Parameters
    ----------
    returns        : return series
    window         : rolling volatility window
    high_threshold : quantile above which is 'high_vol'
    low_threshold  : quantile below which is 'low_vol'

    Returns
    -------
    pd.Series of regime labels
    """
    rolling_vol = returns.rolling(window).std(ddof=1)
    high = rolling_vol.quantile(high_threshold)
    low = rolling_vol.quantile(low_threshold)

    regime = pd.Series("medium_vol", index=returns.index, name="regime")
    regime[rolling_vol >= high] = "high_vol"
    regime[rolling_vol <= low] = "low_vol"
    return regime


def label_trend_regimes(
    prices: pd.Series,
    fast: int = 20,
    slow: int = 60,
) -> pd.Series:
    """Label trend regime as 'uptrend', 'downtrend', or 'sideways'.

    Uses dual moving-average crossover: fast MA above slow MA = uptrend.

    Parameters
    ----------
    prices : price series
    fast   : fast MA window
    slow   : slow MA window

    Returns
    -------
    pd.Series of regime labels
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    regime = pd.Series("sideways", index=prices.index, name="regime")
    regime[fast_ma > slow_ma * 1.01] = "uptrend"
    regime[fast_ma < slow_ma * 0.99] = "downtrend"
    return regime


# ---------------------------------------------------------------------------
# Signal combination helpers
# ---------------------------------------------------------------------------

def combine_signals(
    signals: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]] = None,
    method: str = "weighted",
    normalise_first: bool = True,
) -> pd.Series:
    """Combine multiple signals into a composite signal.

    Parameters
    ----------
    signals         : dict of {signal_name -> pd.Series}
    weights         : dict of {signal_name -> weight}; default equal-weight
    method          : 'weighted' (weighted average), 'rank' (rank-average),
                      'vote' (sign vote), 'max_ic' (placeholder for dynamic)
    normalise_first : z-score each signal before combining

    Returns
    -------
    pd.Series of composite signal
    """
    # Align all signals
    df = pd.DataFrame(signals).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    if normalise_first:
        for col in df.columns:
            mu = df[col].mean()
            sigma = df[col].std(ddof=1)
            if sigma > 0:
                df[col] = (df[col] - mu) / sigma

    if method == "rank":
        ranked = df.rank(pct=True) * 2 - 1  # Scale to [-1, 1]
        if weights:
            w = pd.Series(weights)
            w = w / w.sum()
            composite = (ranked * w).sum(axis=1)
        else:
            composite = ranked.mean(axis=1)

    elif method == "vote":
        signs = df.apply(np.sign)
        if weights:
            w = pd.Series(weights)
            w = w / w.sum()
            composite = (signs * w).sum(axis=1)
        else:
            composite = signs.mean(axis=1)

    else:  # weighted
        if weights:
            w = pd.Series(weights)
            w = w / w.sum()
            composite = (df * w).sum(axis=1)
        else:
            composite = df.mean(axis=1)

    return composite.rename("composite_signal")


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------

def compute_forward_returns(
    prices: pd.DataFrame | pd.Series,
    horizons: Optional[List[int]] = None,
    method: str = "log",
) -> Dict[int, pd.DataFrame | pd.Series]:
    """Compute forward returns at multiple horizons.

    Parameters
    ----------
    prices   : prices (Series for single asset, DataFrame for panel)
    horizons : list of horizons (default [1, 5, 10, 20])
    method   : 'log' (log return), 'simple' (arithmetic return)

    Returns
    -------
    Dict[horizon -> forward_returns (same type as prices)]
    """
    if horizons is None:
        horizons = [1, 5, 10, 20]

    result: dict[int, Any] = {}
    for h in horizons:
        if method == "log":
            if isinstance(prices, pd.Series):
                fwd = np.log(prices.shift(-h) / prices)
            else:
                fwd = np.log(prices.shift(-h) / prices)
        else:
            if isinstance(prices, pd.Series):
                fwd = prices.shift(-h) / prices - 1
            else:
                fwd = prices.shift(-h) / prices - 1
        result[h] = fwd
    return result


# ---------------------------------------------------------------------------
# IC significance grid
# ---------------------------------------------------------------------------

def ic_significance_grid(
    trades: pd.DataFrame,
    signal_cols: List[str],
    return_col: str = "pnl",
    dollar_pos_col: str = "dollar_pos",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute IC and significance for all signal columns.

    Parameters
    ----------
    trades        : trade records
    signal_cols   : list of signal column names
    return_col    : P&L column
    dollar_pos_col: position column
    alpha         : significance level (e.g. 0.05 = 5%)

    Returns
    -------
    pd.DataFrame with IC, t-stat, p-value, significant_bool for each signal
    """
    df = trades.copy()
    if dollar_pos_col in df.columns:
        pos = df[dollar_pos_col].abs().replace(0, np.nan)
        df["_ret"] = df[return_col] / pos
    else:
        df["_ret"] = df[return_col]

    records: list[dict] = []
    for col in signal_cols:
        if col not in df.columns:
            records.append({"signal": col, "ic": float("nan"), "t_stat": float("nan"),
                            "p_value": float("nan"), "significant": False, "n_obs": 0})
            continue
        sub = df[[col, "_ret"]].dropna()
        n = len(sub)
        if n < 3:
            records.append({"signal": col, "ic": float("nan"), "t_stat": float("nan"),
                            "p_value": float("nan"), "significant": False, "n_obs": n})
            continue
        r, p = stats.spearmanr(sub[col], sub["_ret"])
        t = float(r) * np.sqrt(n - 2) / max(np.sqrt(max(1 - float(r)**2, 1e-10)), 1e-10)
        records.append({
            "signal": col,
            "ic": float(r),
            "t_stat": float(t),
            "p_value": float(p),
            "significant": float(p) < alpha,
            "n_obs": n,
        })

    return pd.DataFrame(records).set_index("signal")


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------

def save_results(
    data: pd.DataFrame | pd.Series | Dict,
    output_dir: str | Path,
    filename: str,
    format: str = "csv",
) -> Path:
    """Save analysis results to file.

    Parameters
    ----------
    data       : data to save
    output_dir : output directory
    filename   : filename (without extension if format is specified)
    format     : 'csv', 'parquet', or 'json'

    Returns
    -------
    Path — path where file was saved
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add extension if not present
    if not Path(filename).suffix:
        filename = f"{filename}.{format}"

    out_path = out_dir / filename

    if isinstance(data, dict):
        import json
        with open(out_path, "w") as f:
            json.dump(
                {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in data.items()},
                f, indent=2, default=str,
            )
    elif format == "parquet":
        if isinstance(data, pd.Series):
            data.to_frame().to_parquet(out_path)
        else:
            data.to_parquet(out_path)
    elif format == "json":
        if isinstance(data, pd.DataFrame):
            data.to_json(out_path, orient="records", indent=2)
        else:
            data.to_json(out_path)
    else:  # csv
        if isinstance(data, pd.Series):
            data.to_csv(out_path)
        else:
            data.to_csv(out_path)

    return out_path


def load_trades(
    path: str | Path,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load a trades CSV with sensible defaults.

    Parameters
    ----------
    path        : path to trades CSV
    parse_dates : attempt to parse exit_time as datetime

    Returns
    -------
    pd.DataFrame of trades
    """
    trades = pd.read_csv(path)
    if parse_dates and "exit_time" in trades.columns:
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    return trades
