"""
signal_research_suite.py -- Comprehensive signal research analytics.

IC analysis, quintile decomposition, Fama-MacBeth regression, factor decay,
turnover, capacity, alpha overlap, event/regime conditioning, signal combination,
and walk-forward evaluation.  All numpy/scipy, no pandas dependency.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy import linalg as sla
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# ===================================================================
# 1.  Information Coefficient (IC) calculations
# ===================================================================

def pearson_ic(signal: FloatArray, forward_returns: FloatArray) -> float:
    """Cross-sectional Pearson IC for a single period."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    if mask.sum() < 5:
        return np.nan
    return float(np.corrcoef(signal[mask], forward_returns[mask])[0, 1])


def spearman_ic(signal: FloatArray, forward_returns: FloatArray) -> float:
    """Cross-sectional Spearman rank IC for a single period."""
    mask = np.isfinite(signal) & np.isfinite(forward_returns)
    if mask.sum() < 5:
        return np.nan
    corr, _ = stats.spearmanr(signal[mask], forward_returns[mask])
    return float(corr)


def ic_series(
    signals: FloatArray,
    returns: FloatArray,
    method: str = "spearman",
) -> FloatArray:
    """Compute cross-sectional IC for each time period.

    Parameters
    ----------
    signals : (n_periods, n_assets)
    returns : (n_periods, n_assets) forward returns
    method  : 'pearson' or 'spearman'

    Returns
    -------
    (n_periods,) IC values
    """
    n_periods = signals.shape[0]
    ic_fn = spearman_ic if method == "spearman" else pearson_ic
    ics = np.empty(n_periods)
    for t in range(n_periods):
        ics[t] = ic_fn(signals[t], returns[t])
    return ics


def ic_summary(ics: FloatArray) -> Dict[str, float]:
    """Summary statistics of IC series."""
    valid = ics[np.isfinite(ics)]
    if len(valid) < 2:
        return {"mean_ic": np.nan, "std_ic": np.nan, "ic_ir": np.nan,
                "hit_rate": np.nan, "t_stat": np.nan, "p_value": np.nan}
    mean_ic = float(valid.mean())
    std_ic = float(valid.std(ddof=1))
    ic_ir = mean_ic / (std_ic + 1e-12)
    hit_rate = float((valid > 0).mean())
    t_stat = mean_ic / (std_ic / np.sqrt(len(valid)) + 1e-12)
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=len(valid) - 1)))
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "hit_rate": hit_rate,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_periods": len(valid),
    }


def rolling_ic(
    signals: FloatArray,
    returns: FloatArray,
    window: int = 63,
    method: str = "spearman",
) -> FloatArray:
    """Rolling window IC."""
    ics = ic_series(signals, returns, method)
    n = len(ics)
    result = np.full(n, np.nan)
    for i in range(window, n):
        chunk = ics[i - window : i]
        valid = chunk[np.isfinite(chunk)]
        if len(valid) > 0:
            result[i] = valid.mean()
    return result


def cumulative_ic(ics: FloatArray) -> FloatArray:
    """Cumulative sum of IC (like a wealth curve for IC)."""
    valid_ics = np.where(np.isfinite(ics), ics, 0.0)
    return np.cumsum(valid_ics)


# ===================================================================
# 2.  IC IR and significance testing
# ===================================================================

@dataclass
class ICIRResult:
    ic_ir: float
    mean_ic: float
    std_ic: float
    t_stat: float
    p_value: float
    n_obs: int
    is_significant_5pct: bool
    is_significant_1pct: bool


def compute_ic_ir(ics: FloatArray, min_obs: int = 30) -> ICIRResult:
    """Full IC IR analysis with significance test."""
    valid = ics[np.isfinite(ics)]
    n = len(valid)
    if n < min_obs:
        return ICIRResult(np.nan, np.nan, np.nan, np.nan, np.nan, n, False, False)
    mean_ic = float(valid.mean())
    std_ic = float(valid.std(ddof=1))
    ic_ir = mean_ic / (std_ic + 1e-12)
    se = std_ic / np.sqrt(n)
    t_stat = mean_ic / (se + 1e-12)
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    return ICIRResult(
        ic_ir=ic_ir,
        mean_ic=mean_ic,
        std_ic=std_ic,
        t_stat=t_stat,
        p_value=p_value,
        n_obs=n,
        is_significant_5pct=p_value < 0.05,
        is_significant_1pct=p_value < 0.01,
    )


def bootstrap_ic_ir(ics: FloatArray, n_bootstrap: int = 10000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap confidence interval for IC IR."""
    rng = np.random.default_rng(seed)
    valid = ics[np.isfinite(ics)]
    n = len(valid)
    if n < 10:
        return {"ic_ir_mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    boot_irs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(valid, size=n, replace=True)
        boot_irs[b] = sample.mean() / (sample.std(ddof=1) + 1e-12)
    return {
        "ic_ir_mean": float(boot_irs.mean()),
        "ci_lower": float(np.percentile(boot_irs, 2.5)),
        "ci_upper": float(np.percentile(boot_irs, 97.5)),
    }


# ===================================================================
# 3.  Quintile analysis
# ===================================================================

@dataclass
class QuintileResult:
    quintile_returns: FloatArray          # (n_periods, n_quintiles)
    mean_returns: FloatArray              # (n_quintiles,)
    long_short_spread: FloatArray         # (n_periods,)
    mean_spread: float
    spread_sharpe: float
    monotonicity_score: float


def quintile_analysis(
    signals: FloatArray,
    returns: FloatArray,
    n_quantiles: int = 5,
) -> QuintileResult:
    """Sort assets by signal into quantiles, compute return per quantile.

    Parameters
    ----------
    signals : (n_periods, n_assets)
    returns : (n_periods, n_assets) forward returns
    n_quantiles : number of quantiles (default 5 = quintiles)
    """
    n_periods, n_assets = signals.shape
    q_returns = np.full((n_periods, n_quantiles), np.nan)

    for t in range(n_periods):
        s = signals[t]
        r = returns[t]
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < n_quantiles * 2:
            continue
        s_valid = s[mask]
        r_valid = r[mask]
        # Rank and assign quantiles
        ranks = stats.rankdata(s_valid)
        n_valid = len(s_valid)
        quantile_labels = np.minimum(
            ((ranks - 1) / n_valid * n_quantiles).astype(int), n_quantiles - 1
        )
        for q in range(n_quantiles):
            q_mask = quantile_labels == q
            if q_mask.sum() > 0:
                q_returns[t, q] = r_valid[q_mask].mean()

    mean_returns = np.nanmean(q_returns, axis=0)
    long_short = q_returns[:, -1] - q_returns[:, 0]
    ls_valid = long_short[np.isfinite(long_short)]
    mean_spread = float(np.nanmean(long_short))
    spread_sharpe = float(np.nanmean(long_short) / (np.nanstd(long_short) + 1e-12) * np.sqrt(252))

    # Monotonicity: Spearman correlation of quintile index vs mean return
    if np.all(np.isfinite(mean_returns)):
        mono, _ = stats.spearmanr(np.arange(n_quantiles), mean_returns)
    else:
        mono = np.nan

    return QuintileResult(
        quintile_returns=q_returns,
        mean_returns=mean_returns,
        long_short_spread=long_short,
        mean_spread=mean_spread,
        spread_sharpe=spread_sharpe,
        monotonicity_score=float(mono),
    )


def decile_analysis(
    signals: FloatArray, returns: FloatArray
) -> QuintileResult:
    """Convenience: 10 quantiles."""
    return quintile_analysis(signals, returns, n_quantiles=10)


# ===================================================================
# 4.  Fama-MacBeth cross-sectional regression
# ===================================================================

@dataclass
class FamaMacBethResult:
    gamma_series: FloatArray              # (n_periods, n_factors + 1)
    mean_gamma: FloatArray                # (n_factors + 1,)
    std_gamma: FloatArray
    t_stats: FloatArray
    p_values: FloatArray
    r_squared_series: FloatArray          # (n_periods,)
    mean_r_squared: float


def fama_macbeth(
    returns: FloatArray,
    factors: FloatArray,
    intercept: bool = True,
) -> FamaMacBethResult:
    """Fama-MacBeth cross-sectional regression.

    Parameters
    ----------
    returns : (n_periods, n_assets)
    factors : (n_periods, n_assets, n_factors)
    intercept : include intercept in cross-sectional regression

    Returns
    -------
    FamaMacBethResult with time-series of gamma coefficients
    """
    n_periods, n_assets = returns.shape
    n_factors = factors.shape[2]
    k = n_factors + 1 if intercept else n_factors
    gamma_series = np.full((n_periods, k), np.nan)
    r2_series = np.full(n_periods, np.nan)

    for t in range(n_periods):
        y = returns[t]
        X = factors[t]
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < k + 2:
            continue
        y_v = y[mask]
        X_v = X[mask]
        if intercept:
            X_v = np.column_stack([np.ones(mask.sum()), X_v])
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)
            gamma_series[t] = beta
            ss_res = np.sum((y_v - X_v @ beta) ** 2)
            ss_tot = np.sum((y_v - y_v.mean()) ** 2)
            r2_series[t] = 1.0 - ss_res / (ss_tot + 1e-12)
        except np.linalg.LinAlgError:
            continue

    valid_mask = np.all(np.isfinite(gamma_series), axis=1)
    valid_gammas = gamma_series[valid_mask]
    n_valid = len(valid_gammas)

    if n_valid < 3:
        return FamaMacBethResult(
            gamma_series=gamma_series,
            mean_gamma=np.full(k, np.nan),
            std_gamma=np.full(k, np.nan),
            t_stats=np.full(k, np.nan),
            p_values=np.full(k, np.nan),
            r_squared_series=r2_series,
            mean_r_squared=float(np.nanmean(r2_series)),
        )

    mean_gamma = valid_gammas.mean(axis=0)
    std_gamma = valid_gammas.std(axis=0, ddof=1)
    se = std_gamma / np.sqrt(n_valid)
    t_stats = mean_gamma / (se + 1e-12)
    p_values = np.array(
        [float(2.0 * (1.0 - stats.t.cdf(abs(ts), df=n_valid - 1))) for ts in t_stats]
    )

    return FamaMacBethResult(
        gamma_series=gamma_series,
        mean_gamma=mean_gamma,
        std_gamma=std_gamma,
        t_stats=t_stats,
        p_values=p_values,
        r_squared_series=r2_series,
        mean_r_squared=float(np.nanmean(r2_series)),
    )


def newey_west_adjusted_fama_macbeth(
    gamma_series: FloatArray, n_lags: int = 5
) -> Tuple[FloatArray, FloatArray]:
    """Newey-West standard errors for Fama-MacBeth gammas."""
    valid = gamma_series[np.all(np.isfinite(gamma_series), axis=1)]
    n, k = valid.shape
    if n < n_lags + 2:
        return np.full(k, np.nan), np.full(k, np.nan)
    mean_g = valid.mean(axis=0)
    centered = valid - mean_g
    # Gamma_0
    gamma_0 = (centered.T @ centered) / n
    # Newey-West
    S = gamma_0.copy()
    for lag in range(1, n_lags + 1):
        weight = 1.0 - lag / (n_lags + 1.0)
        gamma_lag = (centered[lag:].T @ centered[:-lag]) / n
        S += weight * (gamma_lag + gamma_lag.T)
    se_nw = np.sqrt(np.diag(S) / n)
    t_stats_nw = mean_g / (se_nw + 1e-12)
    return t_stats_nw, se_nw


# ===================================================================
# 5.  Factor decay analysis
# ===================================================================

@dataclass
class FactorDecayResult:
    horizons: List[int]
    ic_by_horizon: Dict[int, float]
    ic_ir_by_horizon: Dict[int, float]
    half_life: float                     # estimated decay half-life in periods


def factor_decay_analysis(
    signals: FloatArray,
    price_matrix: FloatArray,
    horizons: List[int] | None = None,
    method: str = "spearman",
) -> FactorDecayResult:
    """Compute IC at multiple forward horizons and estimate half-life.

    Parameters
    ----------
    signals     : (n_periods, n_assets)
    price_matrix: (n_periods, n_assets) prices (not returns)
    horizons    : list of forward horizons in periods
    """
    if horizons is None:
        horizons = [1, 5, 10, 21, 42, 63]
    n_periods, n_assets = signals.shape
    ic_fn = spearman_ic if method == "spearman" else pearson_ic

    ic_by_horizon: Dict[int, float] = {}
    ic_ir_by_horizon: Dict[int, float] = {}

    for h in horizons:
        if h >= n_periods:
            ic_by_horizon[h] = np.nan
            ic_ir_by_horizon[h] = np.nan
            continue
        fwd_ret = price_matrix[h:] / price_matrix[:-h] - 1.0
        max_t = min(n_periods - h, signals.shape[0])
        sig_trunc = signals[:max_t]
        ret_trunc = fwd_ret[:max_t]
        ics = ic_series(sig_trunc, ret_trunc, method)
        summary = ic_summary(ics)
        ic_by_horizon[h] = summary["mean_ic"]
        ic_ir_by_horizon[h] = summary["ic_ir"]

    # Estimate half-life by fitting exponential decay
    valid_h = []
    valid_ic = []
    for h in horizons:
        if np.isfinite(ic_by_horizon.get(h, np.nan)):
            valid_h.append(h)
            valid_ic.append(abs(ic_by_horizon[h]))
    half_life = np.nan
    if len(valid_h) >= 3 and valid_ic[0] > 0:
        log_ic = np.log(np.array(valid_ic) + 1e-12)
        h_arr = np.array(valid_h, dtype=float)
        slope, intercept = np.polyfit(h_arr, log_ic, 1)
        if slope < 0:
            half_life = float(-np.log(2) / slope)

    return FactorDecayResult(
        horizons=horizons,
        ic_by_horizon=ic_by_horizon,
        ic_ir_by_horizon=ic_ir_by_horizon,
        half_life=half_life,
    )


# ===================================================================
# 6.  Signal turnover
# ===================================================================

def signal_rank_autocorrelation(
    signals: FloatArray, lag: int = 1
) -> FloatArray:
    """Autocorrelation of cross-sectional signal ranks over time.

    Parameters
    ----------
    signals : (n_periods, n_assets)

    Returns
    -------
    (n_periods - lag,) rank autocorrelation
    """
    n_periods, n_assets = signals.shape
    autocorrs = np.full(n_periods - lag, np.nan)
    for t in range(lag, n_periods):
        s_prev = signals[t - lag]
        s_curr = signals[t]
        mask = np.isfinite(s_prev) & np.isfinite(s_curr)
        if mask.sum() < 5:
            continue
        r_prev = stats.rankdata(s_prev[mask])
        r_curr = stats.rankdata(s_curr[mask])
        autocorrs[t - lag] = float(np.corrcoef(r_prev, r_curr)[0, 1])
    return autocorrs


def signal_turnover(signals: FloatArray) -> FloatArray:
    """Turnover = fraction of top/bottom quintile names that change each period."""
    n_periods, n_assets = signals.shape
    q = max(n_assets // 5, 1)
    turnover = np.full(n_periods - 1, np.nan)
    for t in range(1, n_periods):
        s_prev = signals[t - 1]
        s_curr = signals[t]
        mask = np.isfinite(s_prev) & np.isfinite(s_curr)
        if mask.sum() < q * 2:
            continue
        idx_prev = np.where(mask)[0]
        idx_curr = np.where(mask)[0]
        ranks_prev = stats.rankdata(s_prev[mask])
        ranks_curr = stats.rankdata(s_curr[mask])
        n_v = mask.sum()
        top_prev = set(idx_prev[ranks_prev > n_v - q])
        top_curr = set(idx_curr[ranks_curr > n_v - q])
        if len(top_prev) > 0:
            turnover[t - 1] = 1.0 - len(top_prev & top_curr) / len(top_prev)
    return turnover


def signal_turnover_summary(signals: FloatArray) -> Dict[str, float]:
    """Summary statistics for signal turnover."""
    to = signal_turnover(signals)
    valid = to[np.isfinite(to)]
    ac = signal_rank_autocorrelation(signals, lag=1)
    ac_valid = ac[np.isfinite(ac)]
    return {
        "mean_turnover": float(valid.mean()) if len(valid) > 0 else np.nan,
        "median_turnover": float(np.median(valid)) if len(valid) > 0 else np.nan,
        "mean_rank_autocorr": float(ac_valid.mean()) if len(ac_valid) > 0 else np.nan,
        "turnover_std": float(valid.std()) if len(valid) > 0 else np.nan,
    }


# ===================================================================
# 7.  Signal capacity
# ===================================================================

@dataclass
class CapacityResult:
    aum_levels: FloatArray
    ic_at_aum: FloatArray
    capacity_estimate: float             # AUM where IC drops by 50%
    breakeven_aum: float                 # AUM where alpha = costs


def estimate_signal_capacity(
    signals: FloatArray,
    returns: FloatArray,
    volumes: FloatArray,
    aum_levels: FloatArray | None = None,
    impact_coefficient: float = 0.1,
    base_cost_bps: float = 5.0,
) -> CapacityResult:
    """Estimate AUM capacity of a signal.

    As AUM grows, market impact degrades effective IC.
    """
    if aum_levels is None:
        aum_levels = np.logspace(6, 10, 20)  # 1M to 10B

    base_ics = ic_series(signals, returns, method="spearman")
    base_ic = np.nanmean(base_ics)
    base_ic = max(abs(base_ic), 0.001)

    avg_adv = np.nanmean(volumes) * 100.0  # rough dollar volume
    ic_at_aum = np.empty(len(aum_levels))

    for i, aum in enumerate(aum_levels):
        # Participation rate
        participation = aum / (avg_adv * 252 + 1e-12)
        # Impact cost in return space
        impact_cost = impact_coefficient * np.sqrt(participation)
        # Effective IC reduced by impact
        effective_ic = base_ic - impact_cost * 10  # rough scaling
        ic_at_aum[i] = max(effective_ic, 0.0)

    # Half-life capacity: where IC drops to 50% of base
    target_ic = base_ic * 0.5
    capacity_est = np.nan
    for i in range(len(aum_levels)):
        if ic_at_aum[i] < target_ic:
            if i > 0:
                # Linear interpolation
                frac = (target_ic - ic_at_aum[i]) / (ic_at_aum[i - 1] - ic_at_aum[i] + 1e-12)
                capacity_est = aum_levels[i] * (1 - frac) + aum_levels[i - 1] * frac
            else:
                capacity_est = aum_levels[0]
            break

    # Breakeven: where effective IC = 0
    breakeven = np.nan
    for i in range(len(aum_levels)):
        if ic_at_aum[i] <= 0:
            breakeven = aum_levels[i]
            break

    return CapacityResult(
        aum_levels=aum_levels,
        ic_at_aum=ic_at_aum,
        capacity_estimate=float(capacity_est) if np.isfinite(capacity_est) else float(aum_levels[-1]),
        breakeven_aum=float(breakeven) if np.isfinite(breakeven) else float(aum_levels[-1]),
    )


# ===================================================================
# 8.  Alpha overlap / marginal IC
# ===================================================================

@dataclass
class AlphaOverlapResult:
    signal_correlations: FloatArray       # (n_signals, n_signals)
    marginal_ics: FloatArray              # (n_signals,) after orthogonalizing
    redundancy_scores: FloatArray         # how much each signal overlaps with others


def alpha_overlap_analysis(
    signals_list: List[FloatArray],
    returns: FloatArray,
    method: str = "spearman",
) -> AlphaOverlapResult:
    """Analyze overlap between multiple signals.

    Parameters
    ----------
    signals_list : list of (n_periods, n_assets) arrays
    returns      : (n_periods, n_assets) forward returns
    """
    n_signals = len(signals_list)
    n_periods, n_assets = returns.shape

    # Cross-signal correlation (average cross-sectional correlation)
    corr_mat = np.eye(n_signals)
    for i in range(n_signals):
        for j in range(i + 1, n_signals):
            period_corrs = []
            for t in range(n_periods):
                si = signals_list[i][t]
                sj = signals_list[j][t]
                mask = np.isfinite(si) & np.isfinite(sj)
                if mask.sum() < 5:
                    continue
                c, _ = stats.spearmanr(si[mask], sj[mask])
                period_corrs.append(c)
            if period_corrs:
                corr_mat[i, j] = corr_mat[j, i] = float(np.mean(period_corrs))

    # Raw ICs
    raw_ics = np.array([
        np.nanmean(ic_series(sig, returns, method))
        for sig in signals_list
    ])

    # Marginal IC: orthogonalize each signal against all others
    marginal_ics = np.empty(n_signals)
    for k in range(n_signals):
        others_idx = [i for i in range(n_signals) if i != k]
        if not others_idx:
            marginal_ics[k] = raw_ics[k]
            continue
        # Orthogonalize signal k against others via cross-sectional regression
        residual_signal = np.copy(signals_list[k])
        for t in range(n_periods):
            y = signals_list[k][t]
            X_cols = [signals_list[oi][t] for oi in others_idx]
            X = np.column_stack(X_cols)
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if mask.sum() < len(others_idx) + 2:
                continue
            X_v = X[mask]
            y_v = y[mask]
            try:
                beta = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
                resid = y_v - X_v @ beta
                residual_signal[t, mask] = resid
                residual_signal[t, ~mask] = np.nan
            except np.linalg.LinAlgError:
                pass
        marginal_ics[k] = np.nanmean(ic_series(residual_signal, returns, method))

    # Redundancy: 1 - marginal_ic / raw_ic
    redundancy = 1.0 - np.abs(marginal_ics) / (np.abs(raw_ics) + 1e-12)
    redundancy = np.clip(redundancy, 0.0, 1.0)

    return AlphaOverlapResult(
        signal_correlations=corr_mat,
        marginal_ics=marginal_ics,
        redundancy_scores=redundancy,
    )


# ===================================================================
# 9.  Event-conditional IC
# ===================================================================

@dataclass
class EventConditionalICResult:
    event_name: str
    ic_pre: float
    ic_during: float
    ic_post: float
    ic_non_event: float
    ic_diff_t_stat: float


def event_conditional_ic(
    signals: FloatArray,
    returns: FloatArray,
    event_mask: FloatArray,
    pre_window: int = 5,
    post_window: int = 10,
    method: str = "spearman",
    event_name: str = "event",
) -> EventConditionalICResult:
    """IC conditioned on proximity to events.

    Parameters
    ----------
    event_mask : (n_periods,) boolean mask where True = event occurs
    """
    n_periods = signals.shape[0]
    event_times = np.where(event_mask)[0]

    pre_mask = np.zeros(n_periods, dtype=bool)
    during_mask = np.zeros(n_periods, dtype=bool)
    post_mask = np.zeros(n_periods, dtype=bool)

    for et in event_times:
        pre_start = max(0, et - pre_window)
        pre_mask[pre_start:et] = True
        during_mask[et] = True
        post_end = min(n_periods, et + post_window + 1)
        post_mask[et + 1 : post_end] = True

    non_event_mask = ~(pre_mask | during_mask | post_mask)

    def _mean_ic(mask: FloatArray) -> float:
        idx = np.where(mask)[0]
        if len(idx) < 3:
            return np.nan
        ics = []
        ic_fn = spearman_ic if method == "spearman" else pearson_ic
        for t in idx:
            ics.append(ic_fn(signals[t], returns[t]))
        valid = [x for x in ics if np.isfinite(x)]
        return float(np.mean(valid)) if valid else np.nan

    ic_pre = _mean_ic(pre_mask)
    ic_during = _mean_ic(during_mask)
    ic_post = _mean_ic(post_mask)
    ic_non = _mean_ic(non_event_mask)

    # t-test: event IC vs non-event IC
    ic_fn = spearman_ic if method == "spearman" else pearson_ic
    event_ics = [ic_fn(signals[t], returns[t]) for t in np.where(during_mask | pre_mask | post_mask)[0]]
    non_ics = [ic_fn(signals[t], returns[t]) for t in np.where(non_event_mask)[0]]
    event_ics = [x for x in event_ics if np.isfinite(x)]
    non_ics = [x for x in non_ics if np.isfinite(x)]
    if len(event_ics) > 2 and len(non_ics) > 2:
        t_stat, _ = stats.ttest_ind(event_ics, non_ics)
    else:
        t_stat = np.nan

    return EventConditionalICResult(
        event_name=event_name,
        ic_pre=ic_pre,
        ic_during=ic_during,
        ic_post=ic_post,
        ic_non_event=ic_non,
        ic_diff_t_stat=float(t_stat),
    )


# ===================================================================
# 10. Regime-conditional IC
# ===================================================================

@dataclass
class RegimeConditionalICResult:
    regime_labels: List[str]
    ic_per_regime: Dict[str, float]
    ic_ir_per_regime: Dict[str, float]
    n_periods_per_regime: Dict[str, int]


def regime_conditional_ic(
    signals: FloatArray,
    returns: FloatArray,
    regime_series: IntArray,
    regime_names: Dict[int, str] | None = None,
    method: str = "spearman",
) -> RegimeConditionalICResult:
    """IC broken down by regime."""
    if regime_names is None:
        unique_regimes = np.unique(regime_series)
        regime_names = {int(r): f"regime_{r}" for r in unique_regimes}

    ic_fn = spearman_ic if method == "spearman" else pearson_ic
    ic_per = {}
    ir_per = {}
    n_per = {}

    for regime_id, name in regime_names.items():
        mask = regime_series == regime_id
        idx = np.where(mask)[0]
        if len(idx) < 3:
            ic_per[name] = np.nan
            ir_per[name] = np.nan
            n_per[name] = len(idx)
            continue
        ics = []
        for t in idx:
            if t < signals.shape[0]:
                ics.append(ic_fn(signals[t], returns[t]))
        valid = [x for x in ics if np.isfinite(x)]
        if len(valid) > 1:
            ic_per[name] = float(np.mean(valid))
            ir_per[name] = float(np.mean(valid) / (np.std(valid, ddof=1) + 1e-12))
        else:
            ic_per[name] = np.nan
            ir_per[name] = np.nan
        n_per[name] = len(valid)

    return RegimeConditionalICResult(
        regime_labels=list(regime_names.values()),
        ic_per_regime=ic_per,
        ic_ir_per_regime=ir_per,
        n_periods_per_regime=n_per,
    )


def classify_regimes_from_returns(
    returns: FloatArray, n_regimes: int = 3
) -> IntArray:
    """Simple vol-based regime classification for conditioning IC analysis."""
    n_periods = returns.shape[0]
    window = 21
    rv = np.full(n_periods, np.nan)
    for i in range(window, n_periods):
        rv[i] = np.nanstd(returns[i - window : i]) * np.sqrt(252)
    rv_valid = rv[np.isfinite(rv)]
    if len(rv_valid) < n_regimes:
        return np.zeros(n_periods, dtype=np.int64)
    thresholds = np.percentile(rv_valid, np.linspace(0, 100, n_regimes + 1)[1:-1])
    regimes = np.zeros(n_periods, dtype=np.int64)
    for i in range(n_periods):
        if np.isfinite(rv[i]):
            regimes[i] = int(np.searchsorted(thresholds, rv[i]))
    return regimes


# ===================================================================
# 11. Signal combination
# ===================================================================

def combine_equal_weight(signals_list: List[FloatArray]) -> FloatArray:
    """Equal-weight z-score combination."""
    n = len(signals_list)
    stacked = np.zeros_like(signals_list[0])
    for sig in signals_list:
        # Cross-sectional z-score
        mu = np.nanmean(sig, axis=1, keepdims=True)
        std = np.nanstd(sig, axis=1, keepdims=True) + 1e-12
        z = (sig - mu) / std
        z = np.where(np.isfinite(z), z, 0.0)
        stacked += z
    return stacked / n


def combine_ic_weight(
    signals_list: List[FloatArray],
    returns: FloatArray,
    lookback: int = 63,
) -> FloatArray:
    """IC-weighted signal combination using rolling IC estimates."""
    n_signals = len(signals_list)
    n_periods, n_assets = returns.shape
    combined = np.zeros((n_periods, n_assets))

    for t in range(lookback, n_periods):
        weights = np.zeros(n_signals)
        for k, sig in enumerate(signals_list):
            start = max(0, t - lookback)
            ics = ic_series(sig[start:t], returns[start:t], method="spearman")
            valid = ics[np.isfinite(ics)]
            weights[k] = valid.mean() if len(valid) > 0 else 0.0
        # Normalize
        w_sum = np.abs(weights).sum()
        if w_sum > 0:
            weights = weights / w_sum
        for k, sig in enumerate(signals_list):
            mu = np.nanmean(sig[t])
            std = np.nanstd(sig[t]) + 1e-12
            z = (sig[t] - mu) / std
            z = np.where(np.isfinite(z), z, 0.0)
            combined[t] += weights[k] * z

    return combined


def combine_pca(
    signals_list: List[FloatArray],
    n_components: int = 1,
) -> FloatArray:
    """PCA-based signal combination: use first n_components principal components."""
    n_signals = len(signals_list)
    n_periods, n_assets = signals_list[0].shape

    combined = np.zeros((n_periods, n_assets))
    for t in range(n_periods):
        # Stack z-scored signals for this period
        Z = np.zeros((n_assets, n_signals))
        for k, sig in enumerate(signals_list):
            s = sig[t]
            mu = np.nanmean(s)
            std = np.nanstd(s) + 1e-12
            z = (s - mu) / std
            Z[:, k] = np.where(np.isfinite(z), z, 0.0)
        # SVD
        try:
            U, S, Vt = np.linalg.svd(Z, full_matrices=False)
            # Project onto first n_components
            combined[t] = U[:, :n_components] @ S[:n_components]
        except np.linalg.LinAlgError:
            combined[t] = Z.mean(axis=1)
    return combined


def combine_lasso(
    signals_list: List[FloatArray],
    returns: FloatArray,
    alpha: float = 0.01,
    lookback: int = 126,
) -> FloatArray:
    """LASSO combination: penalized regression of returns on signals."""
    n_signals = len(signals_list)
    n_periods, n_assets = returns.shape
    combined = np.zeros((n_periods, n_assets))

    for t in range(lookback, n_periods):
        # Build feature matrix from flattened cross-sections
        start = max(0, t - lookback)
        rows = []
        y_vals = []
        for tau in range(start, t):
            for j in range(n_assets):
                feat = np.array([signals_list[k][tau, j] for k in range(n_signals)])
                if np.all(np.isfinite(feat)) and np.isfinite(returns[tau, j]):
                    rows.append(feat)
                    y_vals.append(returns[tau, j])
        if len(rows) < n_signals + 5:
            continue
        X = np.array(rows)
        y = np.array(y_vals)
        # Standardize
        X_mu = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-12
        X_norm = (X - X_mu) / X_std
        y_mu = y.mean()
        y_c = y - y_mu
        # Coordinate descent LASSO
        beta = _coordinate_descent_lasso(X_norm, y_c, alpha, max_iter=200)
        # Apply weights
        for j in range(n_assets):
            feat = np.array([signals_list[k][t, j] for k in range(n_signals)])
            if np.all(np.isfinite(feat)):
                feat_norm = (feat - X_mu) / X_std
                combined[t, j] = feat_norm @ beta
    return combined


def _coordinate_descent_lasso(
    X: FloatArray, y: FloatArray, alpha: float, max_iter: int = 200
) -> FloatArray:
    """Simple coordinate descent for LASSO."""
    n, p = X.shape
    beta = np.zeros(p)
    XtX_diag = (X ** 2).sum(axis=0)
    Xty = X.T @ y
    for _ in range(max_iter):
        for j in range(p):
            r_j = Xty[j] - X[:, j] @ (X @ beta) + XtX_diag[j] * beta[j]
            beta[j] = _soft_threshold(r_j, alpha * n) / (XtX_diag[j] + 1e-12)
    return beta


def _soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    return 0.0


def combine_ridge(
    signals_list: List[FloatArray],
    returns: FloatArray,
    alpha: float = 1.0,
    lookback: int = 126,
) -> FloatArray:
    """Ridge regression combination."""
    n_signals = len(signals_list)
    n_periods, n_assets = returns.shape
    combined = np.zeros((n_periods, n_assets))

    for t in range(lookback, n_periods):
        start = max(0, t - lookback)
        rows = []
        y_vals = []
        for tau in range(start, t):
            for j in range(n_assets):
                feat = np.array([signals_list[k][tau, j] for k in range(n_signals)])
                if np.all(np.isfinite(feat)) and np.isfinite(returns[tau, j]):
                    rows.append(feat)
                    y_vals.append(returns[tau, j])
        if len(rows) < n_signals + 5:
            continue
        X = np.array(rows)
        y = np.array(y_vals)
        # Ridge closed form
        XtX = X.T @ X + alpha * np.eye(n_signals)
        Xty = X.T @ y
        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta = np.zeros(n_signals)
        for j in range(n_assets):
            feat = np.array([signals_list[k][t, j] for k in range(n_signals)])
            if np.all(np.isfinite(feat)):
                combined[t, j] = feat @ beta
    return combined


# ===================================================================
# 12. Walk-forward signal evaluation
# ===================================================================

@dataclass
class WalkForwardResult:
    train_ics: FloatArray
    test_ics: FloatArray
    train_ic_irs: FloatArray
    test_ic_irs: FloatArray
    periods: List[Tuple[int, int, int]]   # (train_end, test_start, test_end)
    overfit_ratio: float                  # test_ic_ir / train_ic_ir


def walk_forward_evaluation(
    signals: FloatArray,
    returns: FloatArray,
    train_window: int = 252,
    test_window: int = 63,
    step: int = 21,
    purge_gap: int = 5,
    method: str = "spearman",
) -> WalkForwardResult:
    """Walk-forward (expanding or rolling window) IC evaluation with purging.

    Parameters
    ----------
    purge_gap : number of periods between train end and test start to avoid
                look-ahead contamination
    """
    n_periods = signals.shape[0]
    train_ics = []
    test_ics = []
    train_irs = []
    test_irs = []
    periods = []

    t = train_window
    while t + purge_gap + test_window <= n_periods:
        train_end = t
        test_start = t + purge_gap
        test_end = min(test_start + test_window, n_periods)

        # Train IC
        tr_ics = ic_series(signals[:train_end], returns[:train_end], method)
        tr_summary = ic_summary(tr_ics)
        train_ics.append(tr_summary["mean_ic"])
        train_irs.append(tr_summary["ic_ir"])

        # Test IC
        te_ics = ic_series(signals[test_start:test_end], returns[test_start:test_end], method)
        te_summary = ic_summary(te_ics)
        test_ics.append(te_summary["mean_ic"])
        test_irs.append(te_summary["ic_ir"])

        periods.append((train_end, test_start, test_end))
        t += step

    train_ics_arr = np.array(train_ics)
    test_ics_arr = np.array(test_ics)
    train_irs_arr = np.array(train_irs)
    test_irs_arr = np.array(test_irs)

    mean_train_ir = np.nanmean(train_irs_arr)
    mean_test_ir = np.nanmean(test_irs_arr)
    overfit_ratio = mean_test_ir / (mean_train_ir + 1e-12)

    return WalkForwardResult(
        train_ics=train_ics_arr,
        test_ics=test_ics_arr,
        train_ic_irs=train_irs_arr,
        test_ic_irs=test_irs_arr,
        periods=periods,
        overfit_ratio=float(overfit_ratio),
    )


def expanding_window_evaluation(
    signals: FloatArray,
    returns: FloatArray,
    min_train: int = 126,
    step: int = 21,
    method: str = "spearman",
) -> FloatArray:
    """Expanding window IC: at each step, compute IC using all data up to that point."""
    n_periods = signals.shape[0]
    ics = np.full(n_periods, np.nan)
    for t in range(min_train, n_periods, step):
        ic_arr = ic_series(signals[:t], returns[:t], method)
        valid = ic_arr[np.isfinite(ic_arr)]
        if len(valid) > 0:
            ics[t] = valid.mean()
    return ics


# ===================================================================
# 13. Signal preprocessing utilities
# ===================================================================

def winsorize_signal(signal: FloatArray, limits: Tuple[float, float] = (0.01, 0.99)) -> FloatArray:
    """Cross-sectional winsorization."""
    result = signal.copy()
    for t in range(signal.shape[0]):
        row = signal[t]
        valid = row[np.isfinite(row)]
        if len(valid) < 5:
            continue
        lo, hi = np.percentile(valid, [limits[0] * 100, limits[1] * 100])
        result[t] = np.clip(row, lo, hi)
    return result


def zscore_signal(signal: FloatArray) -> FloatArray:
    """Cross-sectional z-score."""
    result = np.empty_like(signal)
    for t in range(signal.shape[0]):
        row = signal[t]
        mu = np.nanmean(row)
        std = np.nanstd(row)
        if std < 1e-12:
            result[t] = 0.0
        else:
            result[t] = (row - mu) / std
    return result


def rank_signal(signal: FloatArray) -> FloatArray:
    """Cross-sectional rank transform (0 to 1)."""
    result = np.empty_like(signal)
    for t in range(signal.shape[0]):
        row = signal[t]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            result[t] = 0.5
            continue
        ranks = np.full_like(row, np.nan)
        ranks[mask] = stats.rankdata(row[mask]) / mask.sum()
        result[t] = ranks
    return result


def neutralize_signal(
    signal: FloatArray,
    factors: FloatArray,
) -> FloatArray:
    """Cross-sectional neutralization: regress signal on factors, take residual.

    Parameters
    ----------
    signal  : (n_periods, n_assets)
    factors : (n_periods, n_assets, n_factors)
    """
    result = signal.copy()
    n_periods = signal.shape[0]
    for t in range(n_periods):
        y = signal[t]
        X = factors[t]
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < X.shape[1] + 2:
            continue
        X_v = np.column_stack([np.ones(mask.sum()), X[mask]])
        y_v = y[mask]
        try:
            beta = np.linalg.lstsq(X_v, y_v, rcond=None)[0]
            result[t, mask] = y_v - X_v @ beta
        except np.linalg.LinAlgError:
            pass
    return result


def decay_signal(signal: FloatArray, halflife: int = 5) -> FloatArray:
    """Exponentially weighted moving average of signal over time."""
    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    result = np.empty_like(signal)
    result[0] = signal[0]
    for t in range(1, signal.shape[0]):
        s_prev = result[t - 1]
        s_curr = signal[t]
        mask = np.isfinite(s_curr)
        result[t] = np.where(mask, alpha * s_curr + (1 - alpha) * s_prev, s_prev)
    return result


def orthogonalize_signals(signals_list: List[FloatArray]) -> List[FloatArray]:
    """Sequentially orthogonalize signals via Gram-Schmidt on IC space."""
    n_signals = len(signals_list)
    ortho = [signals_list[0].copy()]
    for k in range(1, n_signals):
        sig = signals_list[k].copy()
        for j in range(k):
            # Remove projection onto ortho[j]
            n_periods = sig.shape[0]
            for t in range(n_periods):
                y = sig[t]
                x = ortho[j][t]
                mask = np.isfinite(y) & np.isfinite(x)
                if mask.sum() < 3:
                    continue
                beta = np.dot(y[mask], x[mask]) / (np.dot(x[mask], x[mask]) + 1e-12)
                sig[t, mask] -= beta * x[mask]
        ortho.append(sig)
    return ortho


# ===================================================================
# 14. Comprehensive signal report
# ===================================================================

@dataclass
class SignalReport:
    name: str
    ic_summary: Dict[str, float]
    ic_ir_result: ICIRResult
    quintile_result: QuintileResult
    decay_result: FactorDecayResult | None
    turnover_summary: Dict[str, float]
    walk_forward: WalkForwardResult | None


def generate_signal_report(
    name: str,
    signals: FloatArray,
    returns: FloatArray,
    prices: FloatArray | None = None,
    train_window: int = 252,
    method: str = "spearman",
) -> SignalReport:
    """One-stop comprehensive signal report."""
    ics = ic_series(signals, returns, method)
    summary = ic_summary(ics)
    ir_result = compute_ic_ir(ics)
    quint = quintile_analysis(signals, returns)
    turnover = signal_turnover_summary(signals)

    decay = None
    if prices is not None:
        decay = factor_decay_analysis(signals, prices, method=method)

    wf = None
    if signals.shape[0] > train_window + 100:
        wf = walk_forward_evaluation(signals, returns, train_window=train_window, method=method)

    return SignalReport(
        name=name,
        ic_summary=summary,
        ic_ir_result=ir_result,
        quintile_result=quint,
        decay_result=decay,
        turnover_summary=turnover,
        walk_forward=wf,
    )


# ===================================================================
# 15. Signal generation helpers (for testing)
# ===================================================================

def generate_random_signal(
    n_periods: int, n_assets: int, ic_target: float = 0.05, seed: int = 42
) -> Tuple[FloatArray, FloatArray]:
    """Generate a synthetic signal with known IC level.

    Returns (signal, forward_returns).
    """
    rng = np.random.default_rng(seed)
    noise_signal = rng.standard_normal((n_periods, n_assets))
    true_alpha = rng.standard_normal((n_periods, n_assets))
    noise_return = rng.standard_normal((n_periods, n_assets))
    signal = ic_target * true_alpha + np.sqrt(1 - ic_target ** 2) * noise_signal
    forward_returns = true_alpha * 0.01 + noise_return * 0.02
    return signal, forward_returns


def generate_momentum_signal(prices: FloatArray, lookback: int = 21) -> FloatArray:
    """Simple momentum signal: trailing return."""
    n_periods, n_assets = prices.shape
    signal = np.full((n_periods, n_assets), np.nan)
    for t in range(lookback, n_periods):
        signal[t] = prices[t] / prices[t - lookback] - 1.0
    return signal


def generate_mean_reversion_signal(prices: FloatArray, lookback: int = 21) -> FloatArray:
    """Mean-reversion signal: negative z-score of price vs rolling mean."""
    n_periods, n_assets = prices.shape
    signal = np.full((n_periods, n_assets), np.nan)
    for t in range(lookback, n_periods):
        window = prices[t - lookback : t + 1]
        mu = window.mean(axis=0)
        std = window.std(axis=0) + 1e-12
        signal[t] = -(prices[t] - mu) / std
    return signal


def generate_volatility_signal(prices: FloatArray, lookback: int = 21) -> FloatArray:
    """Low-vol signal: negative realized vol (prefer lower vol)."""
    lr = np.diff(np.log(prices + 1e-12), axis=0)
    n_periods = lr.shape[0]
    n_assets = lr.shape[1]
    signal = np.full((n_periods + 1, n_assets), np.nan)
    for t in range(lookback, n_periods):
        signal[t + 1] = -lr[t - lookback : t].std(axis=0) * np.sqrt(252)
    return signal


# ===================================================================
# 16. Multi-signal backtest harness
# ===================================================================

@dataclass
class MultiSignalBacktestResult:
    individual_reports: List[SignalReport]
    combined_report: SignalReport
    overlap_result: AlphaOverlapResult
    combination_method: str


def multi_signal_backtest(
    signal_dict: Dict[str, FloatArray],
    returns: FloatArray,
    prices: FloatArray | None = None,
    combination_method: str = "ic_weight",
    lookback: int = 126,
) -> MultiSignalBacktestResult:
    """Run full analysis on multiple signals and their combination."""
    names = list(signal_dict.keys())
    signals_list = [signal_dict[n] for n in names]

    individual_reports = []
    for name, sig in signal_dict.items():
        report = generate_signal_report(name, sig, returns, prices)
        individual_reports.append(report)

    # Combine
    if combination_method == "equal_weight":
        combined = combine_equal_weight(signals_list)
    elif combination_method == "ic_weight":
        combined = combine_ic_weight(signals_list, returns, lookback=lookback)
    elif combination_method == "pca":
        combined = combine_pca(signals_list)
    elif combination_method == "lasso":
        combined = combine_lasso(signals_list, returns, lookback=lookback)
    elif combination_method == "ridge":
        combined = combine_ridge(signals_list, returns, lookback=lookback)
    else:
        combined = combine_equal_weight(signals_list)

    combined_report = generate_signal_report("combined", combined, returns, prices)
    overlap = alpha_overlap_analysis(signals_list, returns)

    return MultiSignalBacktestResult(
        individual_reports=individual_reports,
        combined_report=combined_report,
        overlap_result=overlap,
        combination_method=combination_method,
    )


# ===================================================================
# 17. Statistical tests for signal quality
# ===================================================================

def test_ic_stationarity(ics: FloatArray, window: int = 63) -> Dict[str, float]:
    """Test if IC is stationary using rolling mean comparison."""
    valid = ics[np.isfinite(ics)]
    n = len(valid)
    if n < window * 3:
        return {"is_stationary": np.nan, "first_half_ic": np.nan, "second_half_ic": np.nan}
    mid = n // 2
    first = valid[:mid]
    second = valid[mid:]
    t_stat, p_value = stats.ttest_ind(first, second)
    return {
        "is_stationary": float(p_value > 0.05),
        "first_half_ic": float(first.mean()),
        "second_half_ic": float(second.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def test_ic_autocorrelation(ics: FloatArray, max_lag: int = 10) -> Dict[int, float]:
    """Autocorrelation of IC series at various lags."""
    valid = ics[np.isfinite(ics)]
    n = len(valid)
    result = {}
    for lag in range(1, min(max_lag + 1, n // 3)):
        ac = np.corrcoef(valid[:-lag], valid[lag:])[0, 1]
        result[lag] = float(ac)
    return result


def multiple_testing_correction(p_values: FloatArray, method: str = "bh") -> FloatArray:
    """Benjamini-Hochberg or Bonferroni correction for multiple testing.

    Parameters
    ----------
    method : 'bh' for Benjamini-Hochberg, 'bonferroni' for Bonferroni
    """
    n = len(p_values)
    if method == "bonferroni":
        return np.minimum(p_values * n, 1.0)
    # Benjamini-Hochberg
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.empty(n)
    for i in range(n):
        adjusted[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    # Enforce monotonicity
    result = np.minimum.accumulate(adjusted[::-1])[::-1]
    return np.minimum(result, 1.0)


# ===================================================================
# 18. Cross-validation for signal parameters
# ===================================================================

def cross_validate_signal_param(
    param_values: List[float],
    signal_generator: Callable[[float], FloatArray],
    returns: FloatArray,
    n_folds: int = 5,
    method: str = "spearman",
) -> Dict[float, Dict[str, float]]:
    """Cross-validate a signal parameter (e.g., lookback) using time-series CV.

    Parameters
    ----------
    signal_generator : function that takes param value and returns (n_periods, n_assets) signal
    """
    n_periods = returns.shape[0]
    fold_size = n_periods // n_folds
    results = {}

    for param in param_values:
        signal = signal_generator(param)
        fold_ics = []
        for fold in range(n_folds):
            start = fold * fold_size
            end = min(start + fold_size, n_periods)
            ics = ic_series(signal[start:end], returns[start:end], method)
            valid = ics[np.isfinite(ics)]
            if len(valid) > 0:
                fold_ics.append(valid.mean())
        if fold_ics:
            results[param] = {
                "mean_ic": float(np.mean(fold_ics)),
                "std_ic": float(np.std(fold_ics)),
                "ic_ir": float(np.mean(fold_ics) / (np.std(fold_ics) + 1e-12)),
            }
        else:
            results[param] = {"mean_ic": np.nan, "std_ic": np.nan, "ic_ir": np.nan}
    return results


# ===================================================================
# 19. Signal clustering
# ===================================================================

def cluster_signals(
    signals_list: List[FloatArray],
    returns: FloatArray,
    n_clusters: int = 3,
    n_iter: int = 50,
) -> Tuple[IntArray, FloatArray]:
    """Cluster signals by their IC correlation structure.

    Uses k-means on IC time series.
    """
    n_signals = len(signals_list)
    ic_matrix = np.column_stack([
        ic_series(sig, returns, method="spearman") for sig in signals_list
    ])
    # Replace NaN with 0
    ic_matrix = np.where(np.isfinite(ic_matrix), ic_matrix, 0.0)
    # Correlation between IC series
    corr = np.corrcoef(ic_matrix.T)
    dist = 1.0 - np.abs(corr)
    # K-means on distance features
    rng = np.random.default_rng(42)
    centroids = dist[rng.choice(n_signals, n_clusters, replace=False)]
    labels = np.zeros(n_signals, dtype=np.int64)
    for _ in range(n_iter):
        for i in range(n_signals):
            dists = np.array([np.linalg.norm(dist[i] - c) for c in centroids])
            labels[i] = dists.argmin()
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = dist[mask].mean(axis=0)
    return labels, corr


# ===================================================================
# __all__
# ===================================================================

__all__ = [
    "pearson_ic",
    "spearman_ic",
    "ic_series",
    "ic_summary",
    "rolling_ic",
    "cumulative_ic",
    "compute_ic_ir",
    "ICIRResult",
    "bootstrap_ic_ir",
    "quintile_analysis",
    "decile_analysis",
    "QuintileResult",
    "fama_macbeth",
    "FamaMacBethResult",
    "newey_west_adjusted_fama_macbeth",
    "factor_decay_analysis",
    "FactorDecayResult",
    "signal_rank_autocorrelation",
    "signal_turnover",
    "signal_turnover_summary",
    "estimate_signal_capacity",
    "CapacityResult",
    "alpha_overlap_analysis",
    "AlphaOverlapResult",
    "event_conditional_ic",
    "EventConditionalICResult",
    "regime_conditional_ic",
    "RegimeConditionalICResult",
    "classify_regimes_from_returns",
    "combine_equal_weight",
    "combine_ic_weight",
    "combine_pca",
    "combine_lasso",
    "combine_ridge",
    "walk_forward_evaluation",
    "WalkForwardResult",
    "expanding_window_evaluation",
    "winsorize_signal",
    "zscore_signal",
    "rank_signal",
    "neutralize_signal",
    "decay_signal",
    "orthogonalize_signals",
    "generate_signal_report",
    "SignalReport",
    "multi_signal_backtest",
    "MultiSignalBacktestResult",
    "test_ic_stationarity",
    "test_ic_autocorrelation",
    "multiple_testing_correction",
    "cross_validate_signal_param",
    "cluster_signals",
    "generate_random_signal",
    "generate_momentum_signal",
    "generate_mean_reversion_signal",
    "generate_volatility_signal",
]
