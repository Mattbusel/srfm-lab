"""
signals/volatility_surface_signal.py

Volatility surface-based trading signals.

Computes:
  - Term structure slope (front/back month vol spread)
  - Skew signal (25d put IV - 25d call IV, normalized)
  - Vol surface curvature (butterfly spread in vol space)
  - Realized vs implied vol spread (RV - IV gap signal)
  - Volatility cone: current IV percentile vs historical cone
  - Vol regime detection (low / rising / high / falling) using HMM-like logic
  - Forward vol signal: implied forward vol from term structure
  - Vol surface PCA: first 3 components (level, slope, curvature)
  - Composite recommendation

Public entry point: compute_vol_surface_signal()

Dependencies: numpy, scipy
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VolRegime(str, Enum):
    LOW = "low"
    RISING = "rising"
    HIGH = "high"
    FALLING = "falling"
    TRANSITION = "transition"


class VolDirection(str, Enum):
    BUY_VOL = "buy_vol"       # long vol / long gamma
    SELL_VOL = "sell_vol"     # short vol / short gamma
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VolSurfaceSignal:
    """
    Complete volatility surface analysis output.

    Convention
    ----------
    - IVs are in annualized decimal form (0.20 = 20%)
    - Positive skew_signal → put premium is elevated → fear/bearish skew
    - Positive term_structure_slope → front > back → inverted / backwardation
    - Positive rv_iv_spread → RV > IV → vol is cheap (buy vol signal)
    """

    # --- term structure ---
    term_structure_slope: float           # front_iv - back_iv (normalized)
    term_structure_signal: float          # -1 to +1
    front_iv: float
    back_iv: float
    forward_vol: float                    # implied forward vol front→back

    # --- skew ---
    skew_raw: float                       # 25d put IV - 25d call IV
    skew_normalized: float                # vs historical distribution
    skew_signal: float                    # -1 to +1 (+ = fear/bearish)
    skew_percentile: float                # 0-100

    # --- curvature (butterfly) ---
    butterfly: float                      # (25d put IV + 25d call IV)/2 - ATM IV
    curvature_signal: float               # normalized -1 to +1

    # --- realized vs implied ---
    realized_vol: float                   # e.g. 10-day HV
    implied_vol_atm: float                # front month ATM IV
    rv_iv_spread: float                   # RV - IV
    rv_iv_signal: float                   # +1 = vol cheap, -1 = vol expensive
    rv_iv_percentile: float               # 0-100

    # --- vol cone ---
    iv_percentile: float                  # current IV vs historical cone (0-100)
    cone_signal: float                    # -1 (high percentile) to +1 (low percentile)
    historical_iv_mean: float
    historical_iv_std: float
    iv_zscore: float

    # --- vol regime ---
    vol_regime: VolRegime
    regime_confidence: float              # 0-1
    regime_duration_bars: int             # bars in current regime

    # --- PCA components ---
    pca_level: float                      # PC1: overall vol level
    pca_slope: float                      # PC2: term structure slope
    pca_curvature: float                  # PC3: smile curvature
    pca_explained_variance: list[float]   # variance explained by each PC

    # --- composite ---
    composite_score: float                # -1 (sell vol) to +1 (buy vol)
    recommendation: VolDirection
    confidence: float
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_realized_vol(prices: np.ndarray, window: int = 10) -> float:
    """
    Annualized realized volatility from log returns over `window` bars.

    Assumes daily bars: annualization factor = sqrt(252).
    """
    if len(prices) < window + 1:
        return float(np.std(np.diff(np.log(prices + 1e-12))) * math.sqrt(252))
    log_rets = np.diff(np.log(prices[-window - 1:] + 1e-12))
    return float(np.std(log_rets) * math.sqrt(252))


def _term_structure_slope(
    front_iv: float,
    back_iv: float,
    historical_spreads: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    Compute normalized term structure slope.

    slope = front - back  (positive = backwardation / inverted)
    Normalized to [-1, +1] using historical distribution if provided.
    """
    raw_slope = front_iv - back_iv
    if historical_spreads is not None and len(historical_spreads) > 10:
        mu = float(np.mean(historical_spreads))
        sig = float(np.std(historical_spreads)) + 1e-9
        norm = float(np.clip((raw_slope - mu) / (2.0 * sig), -1.0, 1.0))
    else:
        # Rule-of-thumb: >3% spread is highly inverted
        norm = float(np.clip(raw_slope / 0.03, -1.0, 1.0))
    return raw_slope, norm


def _compute_forward_vol(
    front_iv: float,
    back_iv: float,
    t_front: float = 1.0 / 12.0,  # 1 month in years
    t_back: float = 2.0 / 12.0,   # 2 months in years
) -> float:
    """
    Compute implied forward vol between front and back expiry.

    forward_var = (back_iv^2 * t_back - front_iv^2 * t_front) / (t_back - t_front)
    """
    var_back = back_iv ** 2 * t_back
    var_front = front_iv ** 2 * t_front
    dt = t_back - t_front
    if dt <= 0:
        return back_iv
    fwd_var = max(0.0, (var_back - var_front) / dt)
    return math.sqrt(fwd_var)


def _skew_signal(
    put_25d_iv: float,
    call_25d_iv: float,
    historical_skews: Optional[np.ndarray] = None,
) -> tuple[float, float, float, float]:
    """
    Compute skew raw, normalized, signal, and percentile.

    Positive skew_raw → puts more expensive → fear/protective demand.
    """
    skew_raw = put_25d_iv - call_25d_iv

    if historical_skews is not None and len(historical_skews) > 10:
        pct = float(sp_stats.percentileofscore(historical_skews, skew_raw))
        mu = float(np.mean(historical_skews))
        sig = float(np.std(historical_skews)) + 1e-9
        norm = float(np.clip((skew_raw - mu) / (2.0 * sig), -1.0, 1.0))
    else:
        # Typical skew range is ±5% for equity options
        norm = float(np.clip(skew_raw / 0.05, -1.0, 1.0))
        pct = float(np.clip((norm + 1.0) / 2.0 * 100.0, 0.0, 100.0))

    signal = norm  # positive = bearish fear → negative for asset
    return skew_raw, norm, signal, pct


def _butterfly_curvature(
    put_25d_iv: float,
    call_25d_iv: float,
    atm_iv: float,
    historical_butterflies: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    Butterfly = (25d_put + 25d_call)/2 - ATM_IV

    Positive butterfly = expensive wings / vol of vol is high.
    """
    bfly = (put_25d_iv + call_25d_iv) / 2.0 - atm_iv
    if historical_butterflies is not None and len(historical_butterflies) > 10:
        mu = float(np.mean(historical_butterflies))
        sig = float(np.std(historical_butterflies)) + 1e-9
        norm = float(np.clip((bfly - mu) / (2.0 * sig), -1.0, 1.0))
    else:
        norm = float(np.clip(bfly / 0.02, -1.0, 1.0))
    return bfly, norm


def _vol_cone(
    current_iv: float,
    historical_ivs: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Volatility cone: compute percentile of current IV vs historical distribution.

    Returns (percentile, cone_signal, mean, std, zscore).
    cone_signal: -1 = high percentile = sell vol, +1 = low percentile = buy vol.
    """
    if len(historical_ivs) < 10:
        return 50.0, 0.0, float(current_iv), 0.0, 0.0

    pct = float(sp_stats.percentileofscore(historical_ivs, current_iv))
    mu = float(np.mean(historical_ivs))
    sig = float(np.std(historical_ivs)) + 1e-9
    z = (current_iv - mu) / sig
    # Convert to signal: high percentile → sell vol
    cone_signal = float(np.clip(-(pct - 50.0) / 50.0, -1.0, 1.0))
    return pct, cone_signal, mu, sig, float(z)


def _rv_iv_signal(
    rv: float,
    iv: float,
    historical_spreads: Optional[np.ndarray] = None,
) -> tuple[float, float, float]:
    """
    RV - IV spread signal.

    Positive spread (RV > IV) → vol is cheap → buy vol.
    Negative spread (IV > RV) → vol is expensive → sell vol.
    Returns (spread, signal, percentile).
    """
    spread = rv - iv
    if historical_spreads is not None and len(historical_spreads) > 10:
        pct = float(sp_stats.percentileofscore(historical_spreads, spread))
        sig = float(np.clip(spread / (2.0 * float(np.std(historical_spreads)) + 1e-9), -1.0, 1.0))
    else:
        sig = float(np.clip(spread / 0.05, -1.0, 1.0))
        pct = float(np.clip((sig + 1.0) / 2.0 * 100.0, 0.0, 100.0))
    return spread, sig, pct


def _vol_regime_detection(
    iv_history: np.ndarray,
    lookback_short: int = 10,
    lookback_long: int = 60,
    high_percentile: float = 75.0,
    low_percentile: float = 25.0,
) -> tuple[VolRegime, float, int]:
    """
    HMM-like vol regime detection using level and momentum.

    States:
      LOW      : current IV < 25th pct of history
      HIGH     : current IV > 75th pct of history
      RISING   : short-term slope > 0, level not yet extreme
      FALLING  : short-term slope < 0, level not yet extreme
      TRANSITION: slope changing or unclear

    Returns (regime, confidence, duration_bars).
    """
    if len(iv_history) < lookback_short + 2:
        return VolRegime.TRANSITION, 0.5, 0

    recent = iv_history[-lookback_short:]
    current_iv = float(iv_history[-1])

    long_window = iv_history[-min(len(iv_history), lookback_long):]
    p_low = float(np.percentile(long_window, low_percentile))
    p_high = float(np.percentile(long_window, high_percentile))

    # Short-term momentum
    x = np.arange(len(recent), dtype=np.float64)
    slope, _, r, p_val, _ = sp_stats.linregress(x, recent)
    slope_sig = slope / (np.std(recent) + 1e-9)  # normalized slope

    # Determine regime
    if current_iv < p_low and slope_sig < 0.1:
        regime = VolRegime.LOW
        confidence = float(np.clip((p_low - current_iv) / (p_low + 1e-9), 0.0, 1.0))
    elif current_iv > p_high and slope_sig > -0.1:
        regime = VolRegime.HIGH
        confidence = float(np.clip((current_iv - p_high) / (current_iv + 1e-9), 0.0, 1.0))
    elif slope_sig > 0.2:
        regime = VolRegime.RISING
        confidence = float(np.clip(slope_sig / 1.0, 0.0, 1.0))
    elif slope_sig < -0.2:
        regime = VolRegime.FALLING
        confidence = float(np.clip(-slope_sig / 1.0, 0.0, 1.0))
    else:
        regime = VolRegime.TRANSITION
        confidence = 0.4

    # Duration: how many consecutive bars in this regime?
    duration = 0
    for i in range(len(iv_history) - 1, -1, -1):
        v = iv_history[i]
        in_regime = False
        if regime == VolRegime.LOW and v < p_low:
            in_regime = True
        elif regime == VolRegime.HIGH and v > p_high:
            in_regime = True
        elif regime in (VolRegime.RISING, VolRegime.FALLING, VolRegime.TRANSITION):
            in_regime = True  # slope-based, harder to measure back
            if duration > lookback_short:
                break
        if in_regime:
            duration += 1
        else:
            break

    return regime, confidence, duration


def _vol_surface_pca(
    surface_matrix: np.ndarray,
) -> tuple[float, float, float, list[float]]:
    """
    PCA of the volatility surface.

    surface_matrix: shape (T, K) where T = time observations, K = surface points
    (e.g., [25d_put, atm, 25d_call] × [front, mid, back] = 9 columns).

    Returns (pc1_score, pc2_score, pc3_score, explained_variance_ratios).
    """
    T, K = surface_matrix.shape
    if T < 4 or K < 3:
        return 0.0, 0.0, 0.0, [1.0, 0.0, 0.0]

    # Standardize
    mu = np.mean(surface_matrix, axis=0)
    std = np.std(surface_matrix, axis=0) + 1e-12
    X = (surface_matrix - mu) / std

    # SVD for PCA
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        total_var = float(np.sum(S ** 2))
        if total_var < 1e-12:
            return 0.0, 0.0, 0.0, [1.0, 0.0, 0.0]

        explained = [float(s ** 2 / total_var) for s in S[:3]]
        # Scores for the most recent observation
        x_latest = (surface_matrix[-1] - mu) / std
        scores = Vt[:3] @ x_latest
        return float(scores[0]), float(scores[1]), float(scores[2]), explained
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_vol_surface_signal(
    prices: np.ndarray,
    front_iv: float,
    back_iv: float,
    put_25d_iv: float,
    call_25d_iv: float,
    atm_iv: float,
    historical_ivs: Optional[np.ndarray] = None,
    historical_skews: Optional[np.ndarray] = None,
    historical_spreads_ts: Optional[np.ndarray] = None,
    historical_spreads_rv: Optional[np.ndarray] = None,
    historical_butterflies: Optional[np.ndarray] = None,
    surface_matrix: Optional[np.ndarray] = None,
    rv_window: int = 10,
    t_front: float = 1.0 / 12.0,
    t_back: float = 2.0 / 12.0,
) -> VolSurfaceSignal:
    """
    Compute full VolSurfaceSignal.

    Parameters
    ----------
    prices : np.ndarray
        Historical asset prices (daily, oldest first) for RV calculation.
    front_iv, back_iv : float
        ATM implied vol for front and back expiry (annualized decimal).
    put_25d_iv, call_25d_iv : float
        25-delta put and call implied vols (front expiry).
    atm_iv : float
        At-the-money implied vol (front expiry).
    historical_ivs : np.ndarray, optional
        Historical ATM IV time series for cone calculation.
    historical_skews : np.ndarray, optional
        Historical skew (25d put - 25d call) time series.
    historical_spreads_ts : np.ndarray, optional
        Historical term structure spread (front - back) time series.
    historical_spreads_rv : np.ndarray, optional
        Historical RV-IV spread time series.
    historical_butterflies : np.ndarray, optional
        Historical butterfly spread time series.
    surface_matrix : np.ndarray, optional
        (T, K) matrix of historical surface points for PCA.
    rv_window : int
        Window for realized vol calculation (days).
    t_front, t_back : float
        Time to expiry in years for front/back month.
    """
    warnings: list[str] = []

    rv = _compute_realized_vol(prices, window=rv_window)

    ts_raw, ts_norm = _term_structure_slope(
        front_iv, back_iv, historical_spreads_ts
    )
    fwd_vol = _compute_forward_vol(front_iv, back_iv, t_front, t_back)

    skew_raw, skew_norm, skew_sig, skew_pct = _skew_signal(
        put_25d_iv, call_25d_iv, historical_skews
    )

    bfly, bfly_norm = _butterfly_curvature(
        put_25d_iv, call_25d_iv, atm_iv, historical_butterflies
    )

    rv_iv_spread, rv_iv_sig, rv_iv_pct = _rv_iv_signal(
        rv, atm_iv, historical_spreads_rv
    )

    if historical_ivs is not None and len(historical_ivs) > 10:
        iv_pct, cone_sig, h_mu, h_std, iv_z = _vol_cone(atm_iv, historical_ivs)
        reg, reg_conf, reg_dur = _vol_regime_detection(historical_ivs)
    else:
        iv_pct = 50.0
        cone_sig = 0.0
        h_mu = atm_iv
        h_std = 0.0
        iv_z = 0.0
        reg = VolRegime.TRANSITION
        reg_conf = 0.3
        reg_dur = 0

    if surface_matrix is not None:
        pc1, pc2, pc3, exp_var = _vol_surface_pca(surface_matrix)
    else:
        # Build a minimal surface from current snapshot + previous snapshot
        point = np.array([put_25d_iv, atm_iv, call_25d_iv, front_iv, back_iv])
        if historical_ivs is not None and len(historical_ivs) >= 4:
            snap_count = min(len(historical_ivs) - 1, 20)
            mat = np.tile(point, (snap_count + 1, 1))
            pc1, pc2, pc3, exp_var = _vol_surface_pca(mat)
        else:
            pc1, pc2, pc3, exp_var = 0.0, 0.0, 0.0, [1.0, 0.0, 0.0]

    # --- composite scoring ---
    # Buy vol signals: RV > IV, IV low percentile, falling skew, term in contango
    # Sell vol signals: RV < IV, IV high percentile, steep skew, term inverted

    # Positive composite = buy vol
    ts_component = -ts_norm * 0.15        # inverted TS → sell vol
    skew_component = -skew_sig * 0.20     # high skew → fear → sell vol for seller
    bfly_component = -bfly_norm * 0.10    # high curvature → expensive wings → sell
    rv_iv_component = rv_iv_sig * 0.30    # RV > IV → cheap vol → buy
    cone_component = cone_sig * 0.25      # low percentile → buy vol

    composite = float(np.clip(
        ts_component + skew_component + bfly_component
        + rv_iv_component + cone_component,
        -1.0, 1.0,
    ))

    sub_signs = [
        np.sign(ts_component),
        np.sign(skew_component),
        np.sign(rv_iv_component),
        np.sign(cone_component),
    ]
    comp_sign = np.sign(composite)
    agreements = sum(1 for s in sub_signs if s == comp_sign and s != 0)
    confidence = float(agreements / max(len(sub_signs), 1))

    if composite > 0.15:
        recommendation = VolDirection.BUY_VOL
    elif composite < -0.15:
        recommendation = VolDirection.SELL_VOL
    else:
        recommendation = VolDirection.NEUTRAL

    if iv_pct > 85:
        warnings.append(f"IV at {iv_pct:.0f}th percentile — vol historically expensive")
    if iv_pct < 15:
        warnings.append(f"IV at {iv_pct:.0f}th percentile — vol historically cheap, tail risk elevated")
    if ts_raw > 0.03:
        warnings.append("Inverted vol term structure — elevated near-term fear")
    if skew_pct > 90:
        warnings.append(f"Extreme skew ({skew_raw:.1%}) — put premium at {skew_pct:.0f}th percentile")
    if reg in (VolRegime.RISING, VolRegime.HIGH) and reg_conf > 0.6:
        warnings.append(f"Vol regime: {reg.value} (confidence {reg_conf:.2f})")

    return VolSurfaceSignal(
        term_structure_slope=ts_raw,
        term_structure_signal=ts_norm,
        front_iv=front_iv,
        back_iv=back_iv,
        forward_vol=fwd_vol,
        skew_raw=skew_raw,
        skew_normalized=skew_norm,
        skew_signal=skew_sig,
        skew_percentile=skew_pct,
        butterfly=bfly,
        curvature_signal=bfly_norm,
        realized_vol=rv,
        implied_vol_atm=atm_iv,
        rv_iv_spread=rv_iv_spread,
        rv_iv_signal=rv_iv_sig,
        rv_iv_percentile=rv_iv_pct,
        iv_percentile=iv_pct,
        cone_signal=cone_sig,
        historical_iv_mean=h_mu,
        historical_iv_std=h_std,
        iv_zscore=iv_z,
        vol_regime=reg,
        regime_confidence=reg_conf,
        regime_duration_bars=reg_dur,
        pca_level=pc1,
        pca_slope=pc2,
        pca_curvature=pc3,
        pca_explained_variance=exp_var,
        composite_score=composite,
        recommendation=recommendation,
        confidence=confidence,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(99)
    prices = np.cumprod(1 + rng.normal(0, 0.01, 252)) * 100.0
    hist_ivs = 0.20 + rng.normal(0, 0.04, 252)
    hist_ivs = np.clip(hist_ivs, 0.05, 0.80)

    sig = compute_vol_surface_signal(
        prices=prices,
        front_iv=0.22,
        back_iv=0.19,
        put_25d_iv=0.25,
        call_25d_iv=0.20,
        atm_iv=0.22,
        historical_ivs=hist_ivs,
        historical_skews=rng.normal(0.03, 0.01, 252),
        rv_window=10,
    )
    print(f"Recommendation : {sig.recommendation.value}")
    print(f"Composite score: {sig.composite_score:+.4f}")
    print(f"Vol regime     : {sig.vol_regime.value} (conf={sig.regime_confidence:.2f})")
    print(f"IV percentile  : {sig.iv_percentile:.1f}th")
    print(f"Skew           : {sig.skew_raw:.2%} ({sig.skew_percentile:.0f}th pct)")
    print(f"RV / IV        : {sig.realized_vol:.2%} / {sig.implied_vol_atm:.2%}")
    print(f"Forward vol    : {sig.forward_vol:.2%}")
    print(f"PCA (L/S/C)    : {sig.pca_level:+.3f} / {sig.pca_slope:+.3f} / {sig.pca_curvature:+.3f}")
    if sig.warnings:
        print("Warnings:")
        for w in sig.warnings:
            print(f"  - {w}")
