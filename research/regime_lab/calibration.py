"""
research/regime_lab/calibration.py
=====================================
Calibrate regime-model parameters to historical price/return data.

Functions
---------
calibrate_regime_params(trades, price_history) -> dict
calibrate_transition_matrix(regime_series) -> np.ndarray
calibrate_garch(returns, regime_label) -> Tuple[float, float, float]
validate_calibration(generated_series, historical_series) -> CalibrationReport
rolling_calibration(price_history, window=252) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"
REGIMES  = (BULL, BEAR, SIDEWAYS, HIGH_VOL)


# ===========================================================================
# 1. CalibrationReport dataclass
# ===========================================================================

@dataclass
class CalibrationReport:
    """Results of a calibration validation exercise."""
    ks_statistic:   float       # Kolmogorov-Smirnov D statistic (returns)
    ks_pvalue:      float       # KS p-value (high = good match)
    acf_mae:        float       # mean absolute error of ACF (lag 1..20)
    tail_left_mae:  float       # MAE of left-tail quantiles (5th pct and lower)
    tail_right_mae: float       # MAE of right-tail quantiles (95th pct and higher)
    mean_diff:      float       # difference in means (generated - historical)
    std_ratio:      float       # ratio of std-devs (generated / historical)
    skew_diff:      float       # difference in skewness
    kurt_diff:      float       # difference in excess kurtosis
    regime_time_mae: float      # MAE of regime time-fraction vs empirical
    overall_score:  float       # composite score [0, 1] higher = better fit
    details:        Dict[str, Any] = field(default_factory=dict)

    def is_acceptable(self, threshold: float = 0.6) -> bool:
        return self.overall_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ks_statistic":    round(self.ks_statistic, 4),
            "ks_pvalue":       round(self.ks_pvalue, 4),
            "acf_mae":         round(self.acf_mae, 4),
            "tail_left_mae":   round(self.tail_left_mae, 4),
            "tail_right_mae":  round(self.tail_right_mae, 4),
            "mean_diff":       round(self.mean_diff, 6),
            "std_ratio":       round(self.std_ratio, 4),
            "skew_diff":       round(self.skew_diff, 4),
            "kurt_diff":       round(self.kurt_diff, 4),
            "regime_time_mae": round(self.regime_time_mae, 4),
            "overall_score":   round(self.overall_score, 4),
        }


# ===========================================================================
# 2. calibrate_regime_params
# ===========================================================================

def calibrate_regime_params(
        trades: Any,
        price_history: np.ndarray | pd.Series,
        regime_labels: Optional[np.ndarray] = None,
        min_obs: int = 10
) -> Dict[str, Dict[str, Any]]:
    """
    Estimate regime-specific strategy parameters from historical trade data.

    For each regime, estimates:
      - mu               : mean per-trade return
      - sigma            : std-dev of per-trade return
      - win_rate         : fraction of winning trades
      - trades_per_month : average trade frequency (from date stamps if available)
      - sharpe           : per-trade Sharpe ratio

    Parameters
    ----------
    trades        : list of trade dicts or pd.DataFrame
    price_history : 1-D price array (used to fit regime labels if not provided)
    regime_labels : optional pre-computed regime labels aligned to price_history
    min_obs       : minimum observations per regime for reliable estimates

    Returns
    -------
    Dict[regime, {mu, sigma, win_rate, trades_per_month, sharpe}]
    """
    from research.regime_lab.stress import _extract_trades, _trade_pnl, _trade_regime

    prices = np.asarray(price_history, dtype=float)

    # Ensure we have regime labels
    if regime_labels is None:
        from research.regime_lab.detector import RollingVolRegimeDetector
        det = RollingVolRegimeDetector()
        # prices[1:] to align with returns; use prices for direct call
        regime_labels = det.detect(prices)

    rl = np.asarray(regime_labels, dtype=object)

    trade_list = _extract_trades(trades)

    # Collect per-regime returns
    regime_returns: Dict[str, List[float]] = {r: [] for r in REGIMES}
    for trade in trade_list:
        r   = _trade_regime(trade)
        pnl = _trade_pnl(trade)
        # Normalise by entry value if available
        ev  = float(trade.get("dollar_pos") or trade.get("entry_value") or
                    trade.get("notional") or 0.0)
        ret = pnl / ev if ev > 0 else pnl
        if r in regime_returns:
            regime_returns[r].append(ret)

    # Trades per month estimation
    dates_col = None
    if trade_list:
        sample = trade_list[0]
        for key in ("entry_date", "date", "timestamp", "entry_time"):
            if key in sample and sample[key]:
                dates_col = key
                break

    trades_per_month_global = len(trade_list) / 12.0  # rough default

    result: Dict[str, Dict[str, Any]] = {}
    for r in REGIMES:
        rets = regime_returns.get(r, [])
        arr  = np.array(rets, dtype=float) if rets else np.zeros(0)

        # Time fraction in this regime
        mask  = rl == r
        t_frac = float(mask.sum() / max(len(rl), 1))

        if len(arr) >= min_obs:
            mu       = float(np.mean(arr))
            sigma    = float(np.std(arr, ddof=1))
            win_rate = float(np.mean(arr > 0))
            sharpe   = (mu / sigma * (252 ** 0.5)) if sigma > 0 else 0.0
        else:
            # Fall back to regime-class defaults scaled by t_frac
            from research.regime_lab.generator import DEFAULT_REGIME_PARAMS
            default_mu, default_sg = DEFAULT_REGIME_PARAMS.get(r, (0.0, 0.01))
            mu       = default_mu
            sigma    = default_sg
            win_rate = 0.55 if r == BULL else (0.45 if r == BEAR else 0.50)
            sharpe   = mu / sigma * (252 ** 0.5) if sigma > 0 else 0.0

        result[r] = {
            "mu":                round(mu, 6),
            "sigma":             round(sigma, 6),
            "win_rate":          round(win_rate, 4),
            "trades_per_month":  round(trades_per_month_global * t_frac * 4, 2),
            "sharpe":            round(sharpe, 4),
            "n_trades":          len(arr),
            "time_fraction":     round(t_frac, 4),
        }

    return result


# ===========================================================================
# 3. calibrate_transition_matrix
# ===========================================================================

def calibrate_transition_matrix(
        regime_series: np.ndarray | pd.Series,
        smoothing: float = 0.5,
        regimes: Optional[Tuple[str, ...]] = None
) -> np.ndarray:
    """
    Maximum Likelihood Estimate of the Markov transition matrix.

    Uses Laplace smoothing to avoid zero-probability transitions.

    Parameters
    ----------
    regime_series : 1-D sequence of observed regime labels
    smoothing     : Laplace pseudocount (default 0.5 = Jeffreys prior)
    regimes       : ordered tuple of all possible labels

    Returns
    -------
    np.ndarray of shape (K, K) — row-stochastic
    """
    from research.regime_lab.transition import compute_transition_matrix
    return compute_transition_matrix(regime_series, regimes=regimes, smoothing=smoothing)


# ===========================================================================
# 4. calibrate_garch
# ===========================================================================

def calibrate_garch(returns: np.ndarray | pd.Series,
                     regime_label: str = "MIXED",
                     method: str = "mle") -> Tuple[float, float, float]:
    """
    Estimate GARCH(1,1) parameters (omega, alpha, beta) from a return series.

    Primary: scipy.optimize MLE of log-likelihood.
    Fallback: method-of-moments via squared-return autocorrelation.

    Parameters
    ----------
    returns      : 1-D array of log-returns
    regime_label : label for logging purposes
    method       : 'mle' or 'mom'

    Returns
    -------
    (omega, alpha, beta) tuple
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]

    if len(r) < 50:
        logger.warning("GARCH calibration: too few observations (%d) for regime %s; "
                       "using defaults.", len(r), regime_label)
        var_r = float(np.var(r, ddof=1)) if len(r) > 1 else 1e-4
        return (var_r * 0.05, 0.08, 0.90)

    if method == "mle":
        try:
            return _garch_mle(r)
        except Exception as exc:
            logger.warning("GARCH MLE failed (%s); falling back to MoM.", exc)

    return _garch_mom(r)


def _garch_ll(params: np.ndarray, r: np.ndarray) -> float:
    """Negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.9999:
        return 1e10

    T     = len(r)
    h     = np.zeros(T)
    h[0]  = float(np.var(r, ddof=1))

    ll = 0.0
    for t in range(1, T):
        h[t] = omega + alpha * r[t-1]**2 + beta * h[t-1]
        h[t] = max(h[t], 1e-10)
        ll  -= 0.5 * (np.log(2 * np.pi) + np.log(h[t]) + r[t]**2 / h[t])

    return -ll  # minimise


def _garch_mle(r: np.ndarray) -> Tuple[float, float, float]:
    """Numerical MLE for GARCH(1,1)."""
    from scipy.optimize import minimize  # type: ignore

    var0  = float(np.var(r, ddof=1))
    x0    = np.array([var0 * 0.05, 0.08, 0.90])
    bounds = [(1e-8, None), (1e-4, 0.50), (1e-4, 0.9998)]

    res = minimize(_garch_ll, x0, args=(r,), method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-10, "gtol": 1e-8, "maxiter": 1000})

    omega, alpha, beta = res.x
    if alpha + beta >= 1.0:
        beta = min(beta, 0.9998 - alpha)
    return (float(omega), float(alpha), float(beta))


def _garch_mom(r: np.ndarray) -> Tuple[float, float, float]:
    """Method-of-moments GARCH(1,1) estimates."""
    r2   = r ** 2
    var0 = float(np.var(r, ddof=1))
    if var0 == 0:
        return (1e-6, 0.08, 0.90)

    # ACF of r^2 at lag 1 and lag 2
    acf1 = _acf_lag(r2, 1)
    acf2 = _acf_lag(r2, 2)

    alpha_beta = acf1
    beta_est   = acf2 / acf1 if acf1 > 0 else 0.70
    alpha_est  = max(0.03, min(0.20, alpha_beta - beta_est))
    beta_est   = max(0.60, min(0.96, beta_est))
    omega_est  = var0 * (1 - alpha_est - beta_est)
    omega_est  = max(omega_est, 1e-8)

    return (omega_est, alpha_est, beta_est)


def _acf_lag(x: np.ndarray, lag: int) -> float:
    """Sample autocorrelation at given lag."""
    n    = len(x)
    mean = float(np.mean(x))
    var  = float(np.var(x))
    if var == 0 or n <= lag:
        return 0.0
    cov = float(np.mean((x[lag:] - mean) * (x[:n-lag] - mean)))
    return cov / var


# ===========================================================================
# 5. validate_calibration
# ===========================================================================

def validate_calibration(generated_series: np.ndarray | pd.Series,
                          historical_series: np.ndarray | pd.Series,
                          regime_fracs_gen: Optional[Dict[str, float]] = None,
                          regime_fracs_hist: Optional[Dict[str, float]] = None,
                          n_acf_lags: int = 20) -> CalibrationReport:
    """
    Compare statistical properties of a generated series to historical data.

    Tests applied:
      1. Kolmogorov-Smirnov test on return distributions
      2. ACF comparison (lag 1..n_acf_lags)
      3. Left and right tail quantile comparison
      4. First four moments comparison
      5. Regime time-fraction comparison (if provided)

    Parameters
    ----------
    generated_series   : 1-D array of generated log-returns
    historical_series  : 1-D array of historical log-returns
    regime_fracs_gen   : dict regime → time fraction from generated series
    regime_fracs_hist  : dict regime → time fraction from historical series
    n_acf_lags         : number of ACF lags to compare (default 20)

    Returns
    -------
    CalibrationReport
    """
    gen  = np.asarray(generated_series, dtype=float)
    hist = np.asarray(historical_series, dtype=float)
    gen  = gen[~np.isnan(gen)]
    hist = hist[~np.isnan(hist)]

    details: Dict[str, Any] = {}

    # 1. KS test
    try:
        from scipy import stats as scipy_stats  # type: ignore
        ks_stat, ks_p = scipy_stats.ks_2samp(gen, hist)
    except ImportError:
        ks_stat, ks_p = _numpy_ks_2samp(gen, hist)

    # 2. ACF comparison
    gen_acf  = np.array([_acf_lag(gen,  l) for l in range(1, n_acf_lags + 1)])
    hist_acf = np.array([_acf_lag(hist, l) for l in range(1, n_acf_lags + 1)])
    acf_mae  = float(np.mean(np.abs(gen_acf - hist_acf)))

    # 3. Tail quantiles
    left_q   = np.array([1, 2, 3, 5]) / 100
    right_q  = 1 - left_q
    tail_left_mae  = float(np.mean(np.abs(
        np.quantile(gen, left_q) - np.quantile(hist, left_q))))
    tail_right_mae = float(np.mean(np.abs(
        np.quantile(gen, right_q) - np.quantile(hist, right_q))))

    # 4. Moments
    mean_diff = float(np.mean(gen) - np.mean(hist))
    std_ratio = float(np.std(gen, ddof=1) / max(np.std(hist, ddof=1), 1e-10))

    try:
        from scipy.stats import skew as _skew, kurtosis as _kurt  # type: ignore
        skew_diff = float(_skew(gen) - _skew(hist))
        kurt_diff = float(_kurt(gen) - _kurt(hist))   # excess
    except ImportError:
        skew_diff = float(_numpy_skew(gen) - _numpy_skew(hist))
        kurt_diff = float(_numpy_kurtosis(gen) - _numpy_kurtosis(hist))

    details["gen_mean"]  = float(np.mean(gen))
    details["hist_mean"] = float(np.mean(hist))
    details["gen_std"]   = float(np.std(gen, ddof=1))
    details["hist_std"]  = float(np.std(hist, ddof=1))

    # 5. Regime time-fraction comparison
    regime_time_mae = 0.0
    if regime_fracs_gen and regime_fracs_hist:
        diffs = []
        for r in REGIMES:
            fg = regime_fracs_gen.get(r, 0.0)
            fh = regime_fracs_hist.get(r, 0.0)
            diffs.append(abs(fg - fh))
        regime_time_mae = float(np.mean(diffs))

    # Composite score
    ks_score   = max(0.0, 1.0 - ks_stat * 2)       # higher p = better
    acf_score  = max(0.0, 1.0 - acf_mae * 5)
    tail_score = max(0.0, 1.0 - (tail_left_mae + tail_right_mae) * 50)
    mom_score  = max(0.0, 1.0 - abs(std_ratio - 1.0) * 2 - abs(skew_diff) * 0.5
                     - abs(kurt_diff) * 0.1)
    regime_score = max(0.0, 1.0 - regime_time_mae * 4)

    weights = {"ks": 0.25, "acf": 0.20, "tail": 0.25, "mom": 0.20, "regime": 0.10}
    if not (regime_fracs_gen and regime_fracs_hist):
        # Redistribute regime weight
        weights = {"ks": 0.30, "acf": 0.25, "tail": 0.30, "mom": 0.15, "regime": 0.00}

    overall = (weights["ks"]     * ks_score
               + weights["acf"]  * acf_score
               + weights["tail"] * tail_score
               + weights["mom"]  * mom_score
               + weights["regime"] * regime_score)

    return CalibrationReport(
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        acf_mae=acf_mae,
        tail_left_mae=tail_left_mae,
        tail_right_mae=tail_right_mae,
        mean_diff=mean_diff,
        std_ratio=std_ratio,
        skew_diff=skew_diff,
        kurt_diff=kurt_diff,
        regime_time_mae=regime_time_mae,
        overall_score=float(np.clip(overall, 0.0, 1.0)),
        details=details,
    )


def _numpy_ks_2samp(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Naive two-sample KS statistic (no p-value)."""
    combined = np.sort(np.concatenate([a, b]))
    cdf_a    = np.searchsorted(np.sort(a), combined, side="right") / len(a)
    cdf_b    = np.searchsorted(np.sort(b), combined, side="right") / len(b)
    ks_stat  = float(np.max(np.abs(cdf_a - cdf_b)))
    return ks_stat, 0.5   # p-value not computed


def _numpy_skew(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _numpy_kurtosis(x: np.ndarray) -> float:
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4)) - 3.0   # excess


# ===========================================================================
# 6. rolling_calibration
# ===========================================================================

def rolling_calibration(price_history: np.ndarray | pd.Series,
                          window: int = 252,
                          step: int = 21,
                          regimes: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """
    Estimate time-varying regime parameters using a rolling window.

    At each step, fits the RollingVolRegimeDetector on *price_history[t-window:t]*
    and estimates (mu, sigma) per regime.

    Parameters
    ----------
    price_history : 1-D price array
    window        : rolling window length in bars (default 252 = 1 year)
    step          : stride between windows (default 21 = 1 month)
    regimes       : regime tuple (default REGIMES)

    Returns
    -------
    pd.DataFrame with index = bar number and columns:
        {regime}_{mu|sigma|win_rate|n_obs}   for each regime
    """
    prices = np.asarray(price_history, dtype=float)
    T      = len(prices)
    if regimes is None:
        regimes = REGIMES

    from research.regime_lab.detector import RollingVolRegimeDetector
    det = RollingVolRegimeDetector()

    rows = []
    for t in range(window, T, step):
        p_window = prices[t - window : t]
        ret_window = np.diff(np.log(np.where(p_window > 0, p_window, 1e-10)))
        rl_window  = det.detect(p_window)
        rl_aligned = rl_window[1:]   # align to returns

        row: Dict[str, Any] = {"bar": t}
        for r in regimes:
            mask = rl_aligned == r
            rets = ret_window[mask]
            if len(rets) >= 5:
                row[f"{r}_mu"]      = round(float(np.mean(rets)), 7)
                row[f"{r}_sigma"]   = round(float(np.std(rets, ddof=1)), 7)
                row[f"{r}_win_rate"] = round(float(np.mean(rets > 0)), 4)
                row[f"{r}_n_obs"]   = int(mask.sum())
            else:
                row[f"{r}_mu"]      = None
                row[f"{r}_sigma"]   = None
                row[f"{r}_win_rate"] = None
                row[f"{r}_n_obs"]   = 0
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("bar")
    return df


# ===========================================================================
# 7. Parameter stability analysis
# ===========================================================================

def parameter_stability_report(rolling_df: pd.DataFrame,
                                 regimes: Optional[Tuple[str, ...]] = None
                                 ) -> pd.DataFrame:
    """
    Summarise the temporal stability of estimated regime parameters.

    Reports: mean, std, min, max, coefficient of variation for each
    regime's mu and sigma over the rolling windows.

    Parameters
    ----------
    rolling_df : output of rolling_calibration()
    regimes    : regime tuple

    Returns
    -------
    pd.DataFrame indexed by (regime, parameter)
    """
    if regimes is None:
        regimes = REGIMES

    rows = []
    for r in regimes:
        for param in ("mu", "sigma", "win_rate"):
            col = f"{r}_{param}"
            if col not in rolling_df.columns:
                continue
            vals = rolling_df[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            cv = float(np.std(vals, ddof=1) / abs(np.mean(vals))) if np.mean(vals) != 0 else np.nan
            rows.append({
                "regime":    r,
                "parameter": param,
                "mean":      round(float(np.mean(vals)), 7),
                "std":       round(float(np.std(vals, ddof=1)), 7),
                "min":       round(float(np.min(vals)), 7),
                "max":       round(float(np.max(vals)), 7),
                "cv":        round(cv, 4) if not np.isnan(cv) else None,
                "n_windows": len(vals),
            })

    return pd.DataFrame(rows)


# ===========================================================================
# 8. Calibrate to live LARSA regime data
# ===========================================================================

def calibrate_from_larsa_signals(
        signals: List[Dict[str, Any]],
        timeframes: Tuple[str, ...] = ("daily", "hourly", "15m"),
) -> Dict[str, Any]:
    """
    Build regime parameters from LARSA multi-timeframe signal history.

    Parameters
    ----------
    signals   : list of signal dicts, each containing
                {timeframe, regime, score, timestamp, pnl}
    timeframes : timeframes to include

    Returns
    -------
    dict with:
      - regime_params: calibrated per-regime params
      - transition_matrix: calibrated MLE transition matrix
      - timeframe_weights: estimated importance per TF
    """
    if not signals:
        return {}

    df = pd.DataFrame(signals)

    # Merge timeframe signals by timestamp
    tf_regime_series: Dict[str, List[str]] = {tf: [] for tf in timeframes}
    for tf in timeframes:
        tf_df = df[df.get("timeframe", pd.Series()) == tf] if "timeframe" in df.columns else df
        if "regime" in tf_df.columns:
            tf_regime_series[tf] = tf_df["regime"].astype(str).tolist()

    # Ensemble regime: majority vote across timeframes
    max_len = max((len(v) for v in tf_regime_series.values()), default=0)
    if max_len == 0:
        return {}

    ensemble_regimes = []
    for i in range(max_len):
        votes: Dict[str, int] = {}
        for tf in timeframes:
            series = tf_regime_series[tf]
            if i < len(series):
                r = series[i]
                votes[r] = votes.get(r, 0) + 1
        if votes:
            ensemble_regimes.append(max(votes, key=votes.__getitem__))
        else:
            ensemble_regimes.append(SIDEWAYS)

    # Transition matrix from ensemble regimes
    from research.regime_lab.transition import compute_transition_matrix
    trans = compute_transition_matrix(np.array(ensemble_regimes), smoothing=1.0)

    # Regime params from PnL if available
    regime_params: Dict[str, Dict[str, Any]] = {}
    if "pnl" in df.columns and "regime" in df.columns:
        for r in REGIMES:
            mask  = df["regime"].astype(str) == r
            pnls  = df.loc[mask, "pnl"].dropna().astype(float).tolist()
            arr   = np.array(pnls) if pnls else np.zeros(1)
            regime_params[r] = {
                "mu":        round(float(np.mean(arr)), 6),
                "sigma":     round(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.01, 6),
                "win_rate":  round(float(np.mean(arr > 0)), 4),
                "n_signals": len(pnls),
            }

    # Timeframe importance weights from signal counts / accuracy
    timeframe_weights: Dict[str, float] = {}
    for tf in timeframes:
        n = len(tf_regime_series[tf])
        timeframe_weights[tf] = n / max_len if max_len > 0 else 1.0 / len(timeframes)

    return {
        "regime_params":      regime_params,
        "transition_matrix":  trans,
        "timeframe_weights":  timeframe_weights,
        "n_ensemble_bars":    max_len,
    }
