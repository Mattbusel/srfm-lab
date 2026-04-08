"""
volatility_regime_miner.py — Volatility regime signal miner for idea-engine.

Mines signals from:
  - VIX term structure: contango vs backwardation
  - Realized vs implied vol spread
  - Vol-of-vol regime (VVIX-like)
  - Vol surface skew mining (put/call skew extremes)
  - Historical vol percentile (vs 1yr distribution)
  - GARCH-predicted vol vs realized vol
  - Vol clustering (persistence)
  - Vol breakout detection
  - Term structure roll-down
  - VVolatilityMinerResult with all components
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

SignalStrength = float  # [-1, +1]: -1 = strong short vol, +1 = strong long vol


@dataclass
class VolatilityMinerResult:
    """Aggregated output from the volatility regime miner."""
    # Individual component signals (each in [-1, +1])
    term_structure_signal: SignalStrength       # contango = short, backwardation = long
    rv_iv_spread_signal: SignalStrength         # high IV vs RV = short vol
    vvix_regime_signal: SignalStrength          # high vol-of-vol = long vol
    skew_signal: SignalStrength                 # extreme put skew = market fear
    vol_percentile_signal: SignalStrength       # vol at extremes = mean reversion
    garch_residual_signal: SignalStrength       # garch over-predict = short vol
    clustering_signal: SignalStrength           # persistent regime signal
    breakout_signal: SignalStrength             # vol expanding from base = directional
    roll_down_signal: SignalStrength            # contango + stable = sell vol

    # Composite
    composite_signal: SignalStrength

    # Metadata
    current_rv: float
    current_iv: float
    vol_percentile: float
    term_structure_slope: float
    vvix_level: float
    garch_forecast_vol: float
    skew_metric: float
    regime_label: str   # "low", "normal", "high", "transitioning"

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# 1. VIX Term Structure Analysis
# ---------------------------------------------------------------------------

@dataclass
class TermStructureResult:
    slope: float                    # (far_iv - near_iv) / days_diff
    contango_ratio: float           # far_iv / near_iv
    signal: SignalStrength          # +1 = backwardation (buy vol), -1 = contango (sell vol)
    regime: str


def analyze_term_structure(
    near_iv: float,         # front-month implied vol (e.g. VX1)
    far_iv: float,          # back-month implied vol (e.g. VX2)
    near_days: float = 30,
    far_days: float = 60,
) -> TermStructureResult:
    """
    Analyze VIX futures term structure.
    Contango (far > near): typical regime, sell vol / sell VIX.
    Backwardation (near > far): stress regime, buy vol.
    """
    ratio = far_iv / max(near_iv, 0.01)
    slope = (far_iv - near_iv) / max(far_days - near_days, 1)

    # Signal: contango → sell vol (signal = -1), backwardation → buy (signal = +1)
    # Normalize: ratio 0.90 → +1, ratio 1.10 → -1
    raw_signal = -(ratio - 1.0) / 0.10  # 10% contango = -1
    signal = float(np.clip(raw_signal, -1.0, 1.0))

    if ratio > 1.05:
        regime = "contango"
    elif ratio < 0.95:
        regime = "backwardation"
    else:
        regime = "flat"

    return TermStructureResult(
        slope=slope,
        contango_ratio=ratio,
        signal=signal,
        regime=regime,
    )


def analyze_term_structure_curve(
    vix_curve: np.ndarray,          # array of IVs at increasing maturities
    maturities_days: np.ndarray,    # corresponding maturities
) -> Dict:
    """Analyze full term structure curve shape."""
    if len(vix_curve) < 2:
        return {"slope": 0.0, "curvature": 0.0, "signal": 0.0}

    x = maturities_days.astype(float)
    y = vix_curve.astype(float)
    slope, intercept, r, p, se = stats.linregress(x, y)
    curvature = 0.0
    if len(x) >= 3:
        mid_idx = len(x) // 2
        expected_mid = intercept + slope * x[mid_idx]
        curvature = float(y[mid_idx] - expected_mid)  # positive = hump

    # Steeper contango = stronger sell signal
    raw_signal = -float(np.clip(slope / 0.01, -1, 1))

    return {
        "slope": float(slope),
        "curvature": curvature,
        "r_squared": float(r ** 2),
        "signal": float(np.clip(raw_signal, -1, 1)),
        "regime": "contango" if slope > 0 else "backwardation",
    }


# ---------------------------------------------------------------------------
# 2. Realized vs Implied Vol Spread
# ---------------------------------------------------------------------------

def rv_iv_spread_signal(
    realized_vol_history: np.ndarray,   # e.g. 21-day RV over past 252 days
    current_rv: float,
    current_iv: float,
    lookback: int = 63,
) -> Tuple[float, SignalStrength]:
    """
    Compute RV-IV spread and generate a trading signal.
    High IV relative to RV → sell vol (negative signal).
    IV below RV → buy vol (positive signal).
    Returns (spread_in_vol_points, signal).
    """
    spread = current_iv - current_rv

    # Historical context: percentile of (IV - RV) spread
    if len(realized_vol_history) >= lookback:
        hist_rv = realized_vol_history[-lookback:]
        # Assume IV ~ RV * (1 + premium), use historical RV as proxy for fair IV
        hist_spread = current_iv - hist_rv  # degenerate: compare current IV to hist RV
        pct = float(stats.percentileofscore(hist_rv, current_rv) / 100.0)
    else:
        pct = 0.5

    # Signal: high spread (IV >> RV) → short vol
    # Normalize: spread of 5 vol points → signal of -1
    signal = float(np.clip(-spread / 5.0, -1.0, 1.0))

    return spread, signal


# ---------------------------------------------------------------------------
# 3. Vol-of-Vol Regime (VVIX-like)
# ---------------------------------------------------------------------------

def compute_vvix(
    iv_history: np.ndarray,     # time series of implied vols
    window: int = 21,
) -> np.ndarray:
    """
    VVIX-like: volatility of the implied volatility.
    Returns rolling std of IV changes.
    """
    T = len(iv_history)
    iv_changes = np.diff(iv_history)
    vvix = np.full(T, np.nan)
    for t in range(window, T):
        vvix[t] = float(np.std(iv_changes[t - window:t], ddof=1)) * math.sqrt(252)
    return vvix


def vvix_regime_signal(
    iv_history: np.ndarray,
    window: int = 21,
    lookback: int = 252,
) -> Tuple[float, SignalStrength]:
    """
    High VVIX → high uncertainty → buy vol / sell short-term trades.
    Returns (current_vvix, signal).
    """
    vvix = compute_vvix(iv_history, window)
    valid = vvix[~np.isnan(vvix)]
    if len(valid) == 0:
        return 0.0, 0.0

    current_vvix = float(valid[-1])
    hist = valid[-lookback:] if len(valid) >= lookback else valid
    pct = float(stats.percentileofscore(hist, current_vvix) / 100.0)

    # High VVIX (> 80th percentile) → buy vol signal
    # Low VVIX (< 20th percentile) → sell vol signal
    signal = float(np.clip((pct - 0.5) * 2.0, -1.0, 1.0))

    return current_vvix, signal


# ---------------------------------------------------------------------------
# 4. Vol Surface Skew Mining
# ---------------------------------------------------------------------------

@dataclass
class SkewResult:
    put_skew: float             # OTM put IV - ATM IV
    call_skew: float            # OTM call IV - ATM IV
    skew_slope: float           # (put_iv - call_iv) normalized by ATM
    risk_reversal: float        # 25-delta RR: call_iv - put_iv
    butterfly: float            # 25-delta BF: (call_iv + put_iv)/2 - ATM
    signal: SignalStrength      # extreme put skew = fear = buy vol/buy puts


def mine_vol_skew(
    atm_iv: float,
    otm_put_iv: float,           # e.g. 25-delta put IV
    otm_call_iv: float,          # e.g. 25-delta call IV
    skew_history: Optional[np.ndarray] = None,
    lookback: int = 63,
) -> SkewResult:
    """
    Mine volatility surface skew for directional signals.
    Extreme left skew (puts >> calls) = fear, buy vol / bearish signal.
    """
    put_skew = otm_put_iv - atm_iv
    call_skew = otm_call_iv - atm_iv
    skew_slope = (otm_put_iv - otm_call_iv) / max(atm_iv, 0.01)
    risk_reversal = otm_call_iv - otm_put_iv   # positive = call premium
    butterfly = (otm_put_iv + otm_call_iv) / 2.0 - atm_iv

    # Percentile context
    if skew_history is not None and len(skew_history) >= lookback:
        pct = float(stats.percentileofscore(skew_history[-lookback:], skew_slope) / 100.0)
    else:
        pct = 0.5

    # Extreme put skew (left tail demand) = vol buy signal
    # Extreme call skew (right tail demand) = sell signal
    signal = float(np.clip(-(pct - 0.5) * 2.0, -1.0, 1.0))  # high put skew = +1

    return SkewResult(
        put_skew=put_skew,
        call_skew=call_skew,
        skew_slope=skew_slope,
        risk_reversal=risk_reversal,
        butterfly=butterfly,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# 5. Historical Vol Percentile
# ---------------------------------------------------------------------------

def vol_percentile_signal(
    current_rv: float,
    rv_history: np.ndarray,
    lookback: int = 252,
) -> Tuple[float, SignalStrength]:
    """
    Where is current realized vol vs 1-year distribution?
    Very high vol (>90th pct): mean-reversion → sell vol.
    Very low vol (<10th pct): potential breakout → buy vol protection.
    """
    hist = rv_history[-lookback:] if len(rv_history) >= lookback else rv_history
    hist = hist[~np.isnan(hist)]
    if len(hist) == 0:
        return 0.5, 0.0

    pct = float(stats.percentileofscore(hist, current_rv) / 100.0)

    # Non-linear signal: extreme low → buy, extreme high → sell
    if pct >= 0.90:
        signal = -1.0 + (pct - 0.90) / 0.10  # -1 at 90th, 0 at 100th (mean reversion)
        signal = -abs(signal)
    elif pct <= 0.10:
        signal = 1.0 - pct / 0.10  # +1 at 0th, 0 at 10th (breakout risk)
    else:
        signal = 0.0

    return pct, float(np.clip(signal, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 6. GARCH-Predicted vs Realized Vol
# ---------------------------------------------------------------------------

@dataclass
class GARCHFit:
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    conditional_vols: np.ndarray


def fit_garch_11(returns: np.ndarray, max_iter: int = 100) -> GARCHFit:
    """
    Fit GARCH(1,1) via quasi-MLE.
    sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
    Uses bounded Nelder-Mead for robustness.
    """
    r = returns[~np.isnan(returns)]
    T = len(r)
    if T < 20:
        var = float(np.var(r)) if len(r) > 1 else 1e-4
        cond_vols = np.full(T, math.sqrt(var))
        return GARCHFit(var * 0.1, 0.1, 0.85, 0.0, cond_vols)

    unconditional_var = float(np.var(r))

    def _garch_loglik(params: np.ndarray) -> float:
        omega, alpha, beta_ = params
        if omega <= 0 or alpha < 0 or beta_ < 0 or alpha + beta_ >= 0.9999:
            return 1e10
        sigma2 = np.zeros(T)
        sigma2[0] = unconditional_var
        for t in range(1, T):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta_ * sigma2[t - 1]
            if sigma2[t] <= 0:
                return 1e10
        ll = -0.5 * np.sum(np.log(2 * math.pi * sigma2) + r ** 2 / sigma2)
        return -float(ll)

    from scipy.optimize import minimize
    x0 = np.array([unconditional_var * 0.05, 0.10, 0.85])
    bounds = [(1e-8, None), (0.0, 0.5), (0.0, 0.99)]
    try:
        res = minimize(_garch_loglik, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter, "ftol": 1e-9})
        omega_hat, alpha_hat, beta_hat = res.x
    except Exception:
        omega_hat = unconditional_var * 0.05
        alpha_hat = 0.10
        beta_hat = 0.85

    # Compute conditional vols
    sigma2 = np.zeros(T)
    sigma2[0] = unconditional_var
    for t in range(1, T):
        sigma2[t] = omega_hat + alpha_hat * r[t - 1] ** 2 + beta_hat * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-10)
    cond_vols = np.sqrt(sigma2) * math.sqrt(252)

    ll_val = float(-_garch_loglik(np.array([omega_hat, alpha_hat, beta_hat])))

    return GARCHFit(
        omega=float(omega_hat),
        alpha=float(alpha_hat),
        beta=float(beta_hat),
        log_likelihood=ll_val,
        conditional_vols=cond_vols,
    )


def garch_forecast_vol(garch: GARCHFit, returns: np.ndarray, horizon: int = 5) -> float:
    """
    Multi-step GARCH forecast.
    E[sigma^2_{T+h}] = omega*(1 + sum_{k=1}^{h-1} (alpha+beta)^k) + (alpha+beta)^h * sigma^2_T
    """
    r = returns[~np.isnan(returns)]
    if len(r) == 0:
        return float(garch.conditional_vols[-1]) if len(garch.conditional_vols) > 0 else 0.0

    ab = garch.alpha + garch.beta
    omega_long = garch.omega / max(1 - ab, 1e-10)
    sigma2_T = float(garch.conditional_vols[-1] ** 2) / 252  # daily variance

    forecast_var = omega_long + (ab ** horizon) * (sigma2_T - omega_long)
    return math.sqrt(max(forecast_var, 0.0)) * math.sqrt(252)


def garch_residual_signal(
    garch: GARCHFit,
    current_rv: float,
    horizon: int = 5,
    returns: Optional[np.ndarray] = None,
) -> Tuple[float, SignalStrength]:
    """
    Compare GARCH forecast vs current realized vol.
    GARCH predicts higher vol than realized → long vol signal.
    GARCH predicts lower vol than realized → short vol signal.
    """
    if returns is None or len(returns) == 0:
        forecast = float(garch.conditional_vols[-1]) if len(garch.conditional_vols) > 0 else current_rv
    else:
        forecast = garch_forecast_vol(garch, returns, horizon)

    spread = forecast - current_rv
    # Normalize: 5 vol points = full signal
    signal = float(np.clip(spread / 5.0, -1.0, 1.0))
    return forecast, signal


# ---------------------------------------------------------------------------
# 7. Vol Clustering Persistence
# ---------------------------------------------------------------------------

def vol_clustering_signal(
    rv_history: np.ndarray,
    window: int = 21,
    threshold_high: float = 0.25,   # >25% annualized = high vol
    threshold_low: float = 0.10,
) -> Tuple[str, SignalStrength]:
    """
    Detect persistent vol regime (clustering effect).
    If vol has been elevated for >window days → likely to persist.
    Returns (regime, signal).
    """
    hist = rv_history[~np.isnan(rv_history)]
    if len(hist) < window:
        return "unknown", 0.0

    recent = hist[-window:]
    frac_high = float(np.mean(recent > threshold_high))
    frac_low = float(np.mean(recent < threshold_low))
    current = float(hist[-1])

    if frac_high > 0.70:
        regime = "high_vol_persistent"
        # High vol persists → buy vol protection
        signal = min(frac_high * 1.2, 1.0)
    elif frac_low > 0.70:
        regime = "low_vol_persistent"
        # Low vol persists → sell vol
        signal = -min(frac_low * 1.2, 1.0)
    elif frac_high > 0.40:
        regime = "transitioning_to_high"
        signal = 0.3 * (frac_high - 0.40) / 0.30
    else:
        regime = "normal"
        signal = 0.0

    return regime, float(np.clip(signal, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 8. Vol Breakout Detection
# ---------------------------------------------------------------------------

def vol_breakout_signal(
    rv_history: np.ndarray,
    base_window: int = 63,          # measure "calm" from this window
    breakout_threshold: float = 1.5,  # current > base * threshold = breakout
    early_warning_threshold: float = 1.2,
) -> Tuple[bool, float, SignalStrength]:
    """
    Detect when vol breaks out from a low base.
    Vol expands from calm regime = potential directional trend signal.
    Returns (is_breakout, expansion_ratio, signal).
    """
    hist = rv_history[~np.isnan(rv_history)]
    if len(hist) < base_window + 5:
        return False, 1.0, 0.0

    base_vol = float(np.median(hist[-(base_window + 5):-5]))  # exclude most recent 5
    current_vol = float(np.mean(hist[-5:]))

    ratio = current_vol / max(base_vol, 0.001)
    is_breakout = ratio >= breakout_threshold

    if is_breakout:
        # Strong vol expansion = go directional (long vol, short gamma)
        signal = float(np.clip((ratio - breakout_threshold) / (breakout_threshold * 0.5), 0, 1.0))
    elif ratio >= early_warning_threshold:
        signal = float(np.clip((ratio - early_warning_threshold) / (breakout_threshold - early_warning_threshold) * 0.5, 0, 0.5))
    else:
        # Compressing vol = favorable for selling vol
        compression_signal = float(np.clip(-(1.0 - ratio) / 0.3, -1.0, 0.0))
        signal = compression_signal

    return is_breakout, ratio, float(np.clip(signal, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 9. Term Structure Roll-Down Signal
# ---------------------------------------------------------------------------

def term_structure_rolldown_signal(
    near_iv: float,
    far_iv: float,
    near_days: float = 30,
    far_days: float = 60,
    current_rv: float = 0.15,
    stability_window_rv: Optional[np.ndarray] = None,
) -> Tuple[float, SignalStrength]:
    """
    In a stable contango environment, selling far-month vol and buying back
    near-month generates roll-down profit as maturities decrease.

    Roll-down profit = (far_iv - near_iv) * dt / (far_days - near_days)
    Signal is strongest when: (1) steep contango, (2) stable low vol.
    """
    contango = far_iv - near_iv
    roll_per_day = contango / max(far_days - near_days, 1)

    # Stability check: low vol-of-vol
    if stability_window_rv is not None and len(stability_window_rv) > 5:
        rv_std = float(np.std(stability_window_rv[-21:][~np.isnan(stability_window_rv[-21:])]))
        vol_stable = rv_std < 0.03  # low vol-of-RV
    else:
        vol_stable = current_rv < 0.20

    # Signal: positive roll-down + stable vol = sell vol (negative signal in vol space)
    if contango > 0 and vol_stable:
        signal_strength = min(contango / 0.05, 1.0)  # normalize 5pt contango = full signal
        signal = -signal_strength  # sell vol
    elif contango < 0:
        # Backwardation: don't sell vol
        signal = max(0.0, -contango / 0.05)
    else:
        signal = 0.0

    roll_annualized = roll_per_day * 252
    return roll_annualized, float(np.clip(signal, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Main Miner
# ---------------------------------------------------------------------------

@dataclass
class VolatilityRegimeMinerConfig:
    # Signal weights for composite
    weight_term_structure: float = 0.15
    weight_rv_iv: float = 0.20
    weight_vvix: float = 0.10
    weight_skew: float = 0.10
    weight_percentile: float = 0.15
    weight_garch: float = 0.15
    weight_clustering: float = 0.05
    weight_breakout: float = 0.05
    weight_rolldown: float = 0.05

    # Parameters
    rv_window: int = 21
    vol_percentile_lookback: int = 252
    vvix_window: int = 21
    garch_lookback: int = 252
    breakout_base_window: int = 63


class VolatilityRegimeMiner:
    """
    Mines volatility regime signals from multiple sources.

    Usage:
        miner = VolatilityRegimeMiner(cfg)
        result = miner.mine(closes, iv_history, ...)
    """
    def __init__(self, cfg: VolatilityRegimeMinerConfig):
        self.cfg = cfg
        self._garch_cache: Optional[GARCHFit] = None
        self._garch_cache_ts: int = 0

    def _compute_rv(self, closes: np.ndarray, window: int) -> np.ndarray:
        log_rets = np.full(len(closes), np.nan)
        log_rets[1:] = np.diff(np.log(np.maximum(closes, 1e-10)))
        rv = np.full(len(closes), np.nan)
        for t in range(window, len(closes)):
            rv[t] = float(np.std(log_rets[t - window + 1:t + 1], ddof=1)) * math.sqrt(252)
        return rv

    def mine(
        self,
        closes: np.ndarray,                         # price history
        iv_history: Optional[np.ndarray] = None,    # ATM IV history (same length as closes)
        near_iv: Optional[float] = None,            # current front-month IV
        far_iv: Optional[float] = None,             # current back-month IV
        near_days: float = 30,
        far_days: float = 60,
        otm_put_iv: Optional[float] = None,
        otm_call_iv: Optional[float] = None,
        skew_history: Optional[np.ndarray] = None,
        force_garch_refit: bool = False,
    ) -> VolatilityMinerResult:
        cfg = self.cfg
        T = len(closes)
        rv_arr = self._compute_rv(closes, cfg.rv_window)
        rv_history = rv_arr[~np.isnan(rv_arr)]
        current_rv = float(rv_history[-1]) if len(rv_history) > 0 else 0.15

        # Current IV
        if iv_history is not None and len(iv_history) > 0:
            current_iv = float(iv_history[~np.isnan(iv_history)][-1])
        else:
            current_iv = current_rv * 1.1  # default: slight IV premium

        # 1. Term structure
        if near_iv is not None and far_iv is not None:
            ts_result = analyze_term_structure(near_iv, far_iv, near_days, far_days)
            ts_signal = ts_result.signal
            ts_slope = ts_result.slope
        else:
            ts_signal = 0.0
            ts_slope = 0.0

        # 2. RV-IV spread
        rv_hist_for_spread = rv_history if len(rv_history) > 0 else np.array([current_rv])
        _, rv_iv_signal = rv_iv_spread_signal(rv_hist_for_spread, current_rv, current_iv)

        # 3. VVIX
        if iv_history is not None and len(iv_history) >= cfg.vvix_window + 2:
            vvix_level, vvix_signal = vvix_regime_signal(iv_history, cfg.vvix_window)
        else:
            vvix_level = 0.0
            vvix_signal = 0.0

        # 4. Skew
        if otm_put_iv is not None and otm_call_iv is not None:
            skew_res = mine_vol_skew(current_iv, otm_put_iv, otm_call_iv, skew_history)
            skew_signal = skew_res.signal
            skew_metric = skew_res.skew_slope
        else:
            skew_signal = 0.0
            skew_metric = 0.0

        # 5. Vol percentile
        vol_pct, pct_signal = vol_percentile_signal(current_rv, rv_history, cfg.vol_percentile_lookback)

        # 6. GARCH
        log_rets = np.diff(np.log(np.maximum(closes, 1e-10)))
        if len(log_rets) >= 50:
            n_fit = min(cfg.garch_lookback, len(log_rets))
            if force_garch_refit or self._garch_cache is None or T - self._garch_cache_ts > 21:
                self._garch_cache = fit_garch_11(log_rets[-n_fit:])
                self._garch_cache_ts = T
            garch = self._garch_cache
            garch_forecast, garch_signal = garch_residual_signal(garch, current_rv, returns=log_rets[-n_fit:])
        else:
            garch_forecast = current_rv
            garch_signal = 0.0

        # 7. Clustering
        regime_label_raw, clustering_signal = vol_clustering_signal(
            rv_history,
            window=cfg.rv_window,
        )

        # 8. Breakout
        is_breakout, expansion_ratio, breakout_signal = vol_breakout_signal(
            rv_history, cfg.breakout_base_window
        )

        # 9. Roll-down
        if near_iv is not None and far_iv is not None:
            roll_annualized, rolldown_signal = term_structure_rolldown_signal(
                near_iv, far_iv, near_days, far_days, current_rv, rv_arr[-63:]
            )
        else:
            rolldown_signal = 0.0

        # Composite signal (weighted sum)
        w = cfg
        composite = (
            w.weight_term_structure * ts_signal
            + w.weight_rv_iv * rv_iv_signal
            + w.weight_vvix * vvix_signal
            + w.weight_skew * skew_signal
            + w.weight_percentile * pct_signal
            + w.weight_garch * garch_signal
            + w.weight_clustering * clustering_signal
            + w.weight_breakout * breakout_signal
            + w.weight_rolldown * rolldown_signal
        )
        total_weight = (
            w.weight_term_structure + w.weight_rv_iv + w.weight_vvix
            + w.weight_skew + w.weight_percentile + w.weight_garch
            + w.weight_clustering + w.weight_breakout + w.weight_rolldown
        )
        composite = float(np.clip(composite / max(total_weight, 1e-10), -1.0, 1.0))

        # Regime label
        if current_rv > 0.35:
            regime = "high"
        elif current_rv < 0.12:
            regime = "low"
        elif is_breakout:
            regime = "transitioning"
        else:
            regime = "normal"

        return VolatilityMinerResult(
            term_structure_signal=ts_signal,
            rv_iv_spread_signal=rv_iv_signal,
            vvix_regime_signal=vvix_signal,
            skew_signal=skew_signal,
            vol_percentile_signal=pct_signal,
            garch_residual_signal=garch_signal,
            clustering_signal=clustering_signal,
            breakout_signal=breakout_signal,
            roll_down_signal=rolldown_signal,
            composite_signal=composite,
            current_rv=current_rv,
            current_iv=current_iv,
            vol_percentile=vol_pct,
            term_structure_slope=ts_slope,
            vvix_level=vvix_level,
            garch_forecast_vol=garch_forecast,
            skew_metric=skew_metric,
            regime_label=regime,
        )

    def mine_batch(
        self,
        closes: np.ndarray,
        iv_history: Optional[np.ndarray] = None,
        near_ivs: Optional[np.ndarray] = None,    # shape (T,) time series of front IV
        far_ivs: Optional[np.ndarray] = None,      # shape (T,) time series of back IV
        step: int = 1,
    ) -> List[VolatilityMinerResult]:
        """
        Run miner over a rolling window and return a list of results.
        """
        T = len(closes)
        results = []
        min_start = max(self.cfg.garch_lookback, self.cfg.vol_percentile_lookback, 126)

        for t in range(min_start, T, step):
            near_iv = float(near_ivs[t]) if near_ivs is not None and t < len(near_ivs) else None
            far_iv = float(far_ivs[t]) if far_ivs is not None and t < len(far_ivs) else None
            iv_h = iv_history[:t] if iv_history is not None else None
            result = self.mine(
                closes=closes[:t],
                iv_history=iv_h,
                near_iv=near_iv,
                far_iv=far_iv,
            )
            results.append(result)

        return results


# ---------------------------------------------------------------------------
# Convenience: batch signal extraction
# ---------------------------------------------------------------------------

def extract_signal_series(results: List[VolatilityMinerResult]) -> Dict[str, np.ndarray]:
    """Convert list of VVolatilityMinerResult to dict of signal arrays."""
    if not results:
        return {}
    keys = [
        "term_structure_signal", "rv_iv_spread_signal", "vvix_regime_signal",
        "skew_signal", "vol_percentile_signal", "garch_residual_signal",
        "clustering_signal", "breakout_signal", "roll_down_signal",
        "composite_signal", "current_rv", "current_iv", "vol_percentile",
        "term_structure_slope", "vvix_level", "garch_forecast_vol",
    ]
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.array([getattr(r, k) for r in results], dtype=float)
    return out
