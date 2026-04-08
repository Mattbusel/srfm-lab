"""
signals/macro_momentum_signal.py

Macro factor momentum signal generator for the idea engine.

Computes:
  - Nowcasting score: composite of PMI, NFP surprise, CPI surprise, GDP surprise
  - Economic surprise index: z-score of (actual - consensus) across recent releases
  - Yield curve slope signal: 2s10s spread as regime indicator
  - Credit spread signal: IG/HY OAS as risk appetite measure
  - Cross-asset momentum: equities/bonds/commodities relative momentum
  - Dollar index (DXY) momentum: global risk-off indicator
  - Volatility of volatility (VoV): VIX-of-VIX as tail risk gauge
  - Macro regime classification: expansion/slowdown/contraction/recovery
  - Composite macro signal: conviction-weighted blend of sub-signals
  - Signal output: DomainSignal with domain='macro'

All arrays are numpy-based. Public entry point: compute_macro_momentum_signal().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# DomainSignal (local definition)
# ---------------------------------------------------------------------------

@dataclass
class DomainSignal:
    """
    Standardised signal output container.

    Attributes
    ----------
    domain : str
    value : float — normalised in [-1, +1]
    conviction : float — confidence in [0, 1]
    regime : str
    components : dict
    metadata : dict
    warnings : list[str]
    """
    domain: str
    value: float
    conviction: float
    regime: str
    components: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MacroRegime(str, Enum):
    EXPANSION   = "expansion"
    SLOWDOWN    = "slowdown"
    CONTRACTION = "contraction"
    RECOVERY    = "recovery"
    UNCERTAIN   = "uncertain"


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EconomicRelease:
    """
    A single economic data release.

    Attributes
    ----------
    name : str
        Indicator name, e.g. 'PMI', 'NFP', 'CPI', 'GDP'.
    actual : float
        Realised value.
    consensus : float
        Bloomberg/Reuters median survey estimate prior to release.
    prior : float
        Previous release value.
    weight : float
        Relative importance weight (higher = more market-moving).
    release_age_days : int
        How many calendar days ago this release occurred.
        Used for exponential decay weighting.
    """
    name: str
    actual: float
    consensus: float
    prior: float
    weight: float = 1.0
    release_age_days: int = 0


@dataclass
class AssetMomentumSeries:
    """
    Price history for a single asset used in cross-asset momentum.

    Attributes
    ----------
    name : str
    prices : np.ndarray
        Closing prices, oldest first.
    asset_class : str
        'equity', 'bond', 'commodity', 'fx', 'vol'.
    """
    name: str
    prices: np.ndarray
    asset_class: str


@dataclass
class MacroMomentumInput:
    """
    Full macro context required by compute_macro_momentum_signal().

    Attributes
    ----------
    economic_releases : list[EconomicRelease]
        Recent economic data releases.
    yield_2y : float
        Current 2-year Treasury yield (decimal, e.g. 0.045 = 4.5 %).
    yield_10y : float
        Current 10-year Treasury yield.
    yield_30y : float
        Current 30-year Treasury yield.
    ig_oas : float
        Investment-grade option-adjusted spread (basis points).
    hy_oas : float
        High-yield option-adjusted spread (basis points).
    dxy_series : np.ndarray
        DXY (US Dollar Index) daily closing values, oldest first, >=60 bars.
    vix_series : np.ndarray
        VIX daily closing values, oldest first, >=30 bars.
    vvix_series : np.ndarray
        VVIX daily closing values, oldest first, >=30 bars.
    asset_series : list[AssetMomentumSeries]
        Cross-asset price series for relative momentum.
    pmi_series : np.ndarray, optional
        Monthly PMI readings (oldest first) for trend detection.
    lookback_momentum : int
        Lookback window (bars) for momentum calculation. Default 63 (≈3M).
    """
    economic_releases: list[EconomicRelease]
    yield_2y: float
    yield_10y: float
    yield_30y: float
    ig_oas: float
    hy_oas: float
    dxy_series: np.ndarray
    vix_series: np.ndarray
    vvix_series: np.ndarray
    asset_series: list[AssetMomentumSeries]
    pmi_series: Optional[np.ndarray] = None
    lookback_momentum: int = 63


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _zscore(x: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    Compute z-scores of array x.

    If window is given, use rolling statistics (last `window` elements for
    mean/std); otherwise use full-series statistics.
    """
    if window is not None and len(x) >= window:
        mu  = float(np.mean(x[-window:]))
        std = float(np.std(x[-window:]))
    else:
        mu  = float(np.mean(x))
        std = float(np.std(x))
    std = std if std > 1e-12 else 1e-12
    return (x - mu) / std


def _clip_signal(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _rolling_return(prices: np.ndarray, window: int) -> float:
    """
    Simple total return over last `window` bars.
    Returns 0 if not enough data.
    """
    if len(prices) < window + 1:
        return 0.0
    ret = (prices[-1] / prices[-(window + 1)]) - 1.0
    return float(ret)


def _exponential_decay_weights(ages: np.ndarray, halflife_days: float = 30.0) -> np.ndarray:
    """
    Compute exponential decay weights given ages in days.

    w_i = exp(-ln(2) * age_i / halflife_days)
    """
    return np.exp(-math.log(2) * ages / halflife_days)


def _normalise_to_signal(x: float, scale: float) -> float:
    """
    Map raw value x to [-1, +1] using tanh normalisation with given scale.
    """
    return float(np.tanh(x / (scale + 1e-12)))


# ---------------------------------------------------------------------------
# 1. Economic Surprise Index (ESI)
# ---------------------------------------------------------------------------

def _compute_economic_surprise_index(
    releases: list[EconomicRelease],
    halflife_days: float = 30.0,
) -> tuple[float, dict[str, float]]:
    """
    Compute the Economic Surprise Index (ESI).

    For each release: surprise_i = (actual - consensus) / |consensus| (normalised).
    Composite ESI = weighted mean of surprises, with weights = release_weight
    * exponential decay by age.

    Returns
    -------
    esi : float in [-1, +1]
    per_name : dict mapping release name -> surprise score
    """
    if not releases:
        return 0.0, {}

    names  = [r.name for r in releases]
    surps  = np.array([
        (r.actual - r.consensus) / (abs(r.consensus) + 1e-9)
        for r in releases
    ], dtype=float)
    ages   = np.array([float(r.release_age_days) for r in releases])
    wts    = np.array([r.weight for r in releases], dtype=float)
    decay  = _exponential_decay_weights(ages, halflife_days)
    combined_wts = wts * decay

    sum_wt = combined_wts.sum() + 1e-12
    esi_raw = float(np.dot(combined_wts, surps) / sum_wt)

    # Normalise: scale so ±0.10 (10 % surprise) maps near ±1
    esi = _normalise_to_signal(esi_raw, scale=0.08)

    per_name = {n: float(s) for n, s in zip(names, surps)}
    return esi, per_name


# ---------------------------------------------------------------------------
# 2. Nowcasting score
# ---------------------------------------------------------------------------

def _compute_nowcast_score(
    releases: list[EconomicRelease],
    pmi_series: Optional[np.ndarray],
) -> float:
    """
    Composite nowcasting score blending:
      - PMI level signal (50 threshold, trend)
      - NFP, CPI, GDP surprise contributions
      - Recent ESI momentum

    Returns
    -------
    nowcast : float in [-1, +1]
    """
    components: list[float] = []
    weights: list[float] = []

    # PMI component
    if pmi_series is not None and len(pmi_series) >= 3:
        pmi_last  = float(pmi_series[-1])
        pmi_prev  = float(pmi_series[-3])
        pmi_level = _clip_signal((pmi_last - 50.0) / 10.0)   # 60 => +1, 40 => -1
        pmi_mom   = _clip_signal((pmi_last - pmi_prev) / 5.0) # 5pt rise => +1
        components.append(0.6 * pmi_level + 0.4 * pmi_mom)
        weights.append(2.5)

    # Individual release contributions
    indicator_config = {
        "PMI":  {"weight": 2.0, "scale": 5.0},
        "NFP":  {"weight": 1.8, "scale": 200_000.0},  # NFP in jobs
        "CPI":  {"weight": 1.5, "scale": 0.3},         # CPI surprise in % pts
        "GDP":  {"weight": 2.0, "scale": 0.5},         # GDP in % pts
        "ISM":  {"weight": 1.5, "scale": 5.0},
        "RETAIL": {"weight": 1.2, "scale": 0.5},
    }

    for r in releases:
        cfg = indicator_config.get(r.name.upper())
        if cfg is None:
            cfg = {"weight": 1.0, "scale": 0.1}
        surprise = (r.actual - r.consensus) / (abs(r.consensus) + 1e-9)
        sig = _normalise_to_signal(surprise, cfg["scale"] / (abs(r.consensus) + 1e-9))
        decay = float(np.exp(-math.log(2) * r.release_age_days / 30.0))
        components.append(sig)
        weights.append(cfg["weight"] * decay)

    if not components:
        return 0.0

    wts = np.array(weights, dtype=float)
    comp = np.array(components, dtype=float)
    nowcast = float(np.dot(wts, comp) / (wts.sum() + 1e-12))
    return _clip_signal(nowcast)


# ---------------------------------------------------------------------------
# 3. Yield curve slope signal
# ---------------------------------------------------------------------------

def _compute_yield_curve_signal(
    yield_2y: float,
    yield_10y: float,
    yield_30y: float,
) -> tuple[float, str]:
    """
    Translate yield curve shape into a directional risk signal.

    2s10s spread:
      > +100bps  => steep / expansion  => bullish risk assets
      0 to +100  => neutral
      Inverted   => recessionary signal => bearish risk assets

    Also check 2s30s for additional context.

    Returns
    -------
    signal : float in [-1, +1]
    shape_label : str  ('steep', 'flat', 'inverted', 'bear_flat', etc.)
    """
    spread_2s10s = yield_10y - yield_2y   # in decimal; e.g. 0.01 = 1bp? no — yields in decimal
    # Assume yields are in decimal (0.045 = 4.5 %). Convert to basis points for readability.
    bps_2s10s = spread_2s10s * 100.0 * 100.0  # decimal -> bps (4.5% = 0.045 => bps = 450)
    # Actually standard: yield_10y - yield_2y in decimal * 10000 = bps
    bps_2s10s = (yield_10y - yield_2y) * 10_000.0  # bps
    bps_2s30s = (yield_30y - yield_2y) * 10_000.0

    if bps_2s10s > 100:
        signal = 0.6
        label = "steep"
    elif bps_2s10s > 25:
        signal = 0.2
        label = "normal"
    elif bps_2s10s > -25:
        signal = -0.1
        label = "flat"
    elif bps_2s10s > -100:
        signal = -0.5
        label = "inverted"
    else:
        signal = -0.9
        label = "deeply_inverted"

    # Soften if 30y is steep even if 2s10s is flat (partial normalisation)
    if bps_2s30s > 150 and signal < 0:
        signal = signal * 0.7

    return _clip_signal(signal), label


# ---------------------------------------------------------------------------
# 4. Credit spread signal
# ---------------------------------------------------------------------------

# Historical percentile reference ranges (approximate long-run medians)
_IG_OAS_MEDIAN  = 120.0   # bps
_HY_OAS_MEDIAN  = 400.0   # bps
_IG_OAS_STRESS  = 250.0   # bps — GFC / COVID peak range
_HY_OAS_STRESS  = 900.0   # bps


def _compute_credit_spread_signal(ig_oas: float, hy_oas: float) -> float:
    """
    Credit spread signal as risk appetite measure.

    Tight spreads => risk-on => bullish equities.
    Wide spreads  => risk-off => bearish equities.

    Scale:
      IG OAS < 80bps  => extreme tightening => +0.8
      IG OAS > 250bps => stress            => -0.9
    Returns composite in [-1, +1].
    """
    # IG signal
    ig_z = (ig_oas - _IG_OAS_MEDIAN) / ((_IG_OAS_STRESS - _IG_OAS_MEDIAN) / 2.0)
    ig_signal = _clip_signal(-ig_z)  # tight spreads = positive signal

    # HY signal
    hy_z = (hy_oas - _HY_OAS_MEDIAN) / ((_HY_OAS_STRESS - _HY_OAS_MEDIAN) / 2.0)
    hy_signal = _clip_signal(-hy_z)

    # HY is more sensitive to risk appetite; weight it more
    return _clip_signal(0.35 * ig_signal + 0.65 * hy_signal)


# ---------------------------------------------------------------------------
# 5. Cross-asset momentum
# ---------------------------------------------------------------------------

def _compute_cross_asset_momentum(
    asset_series: list[AssetMomentumSeries],
    lookback: int = 63,
    short_lookback: int = 21,
) -> tuple[float, dict[str, float]]:
    """
    Compute relative momentum across asset classes and aggregate.

    For each series:
      momentum = blended 1M + 3M return (equal weight)
      Normalise each asset's momentum by vol of rolling returns.

    Cross-asset signal = asset_class-weighted aggregate:
      equity momentum (positive) => bullish
      bond momentum (positive)   => risk-off / bearish equity (inverted)
      commodity momentum         => inflationary
      fx / DXY                   => strong dollar = bearish EM, commodities
      vol (VIX etc.)             => rising vol = bearish (inverted)

    Returns
    -------
    aggregate : float in [-1, +1]
    per_asset : dict
    """
    CLASS_SIGN = {
        "equity":    +1.0,
        "bond":      -0.3,   # bond rally often = risk-off
        "commodity": +0.5,
        "fx":        -0.5,   # strong dollar bearish for risk
        "vol":       -1.0,   # rising vol = bad for equities
    }

    signals: dict[str, float] = {}
    weights: dict[str, float] = {}

    for asset in asset_series:
        p = asset.prices
        if len(p) < short_lookback + 1:
            continue

        ret_short = _rolling_return(p, short_lookback)
        ret_long  = _rolling_return(p, lookback) if len(p) >= lookback + 1 else ret_short

        # Blended return
        blended = 0.4 * ret_short + 0.6 * ret_long

        # Volatility-normalise
        if len(p) >= lookback + 1:
            log_rets = np.diff(np.log(p[-(lookback + 1):]))
        else:
            log_rets = np.diff(np.log(p))
        vol = float(np.std(log_rets)) * math.sqrt(252) + 1e-9
        vol_adj = blended / vol

        # Tanh normalise
        normalised = float(np.tanh(vol_adj * 2.0))

        class_sign = CLASS_SIGN.get(asset.asset_class, 1.0)
        signals[asset.name] = normalised * class_sign
        weights[asset.name] = abs(class_sign)

    if not signals:
        return 0.0, {}

    total_wt = sum(weights.values()) + 1e-12
    aggregate = sum(signals[k] * weights[k] for k in signals) / total_wt
    return _clip_signal(aggregate), signals


# ---------------------------------------------------------------------------
# 6. DXY momentum
# ---------------------------------------------------------------------------

def _compute_dxy_momentum(
    dxy_series: np.ndarray,
    short_window: int = 20,
    long_window: int = 60,
) -> float:
    """
    DXY momentum signal: strong/rising dollar => global risk-off => bearish.

    Returns signal in [-1, +1]:
      Rising DXY => bearish (-); falling DXY => bullish (+).
    """
    if len(dxy_series) < long_window + 1:
        long_window = max(len(dxy_series) - 1, 1)

    ret_short = _rolling_return(dxy_series, short_window)
    ret_long  = _rolling_return(dxy_series, long_window)

    blended = 0.5 * ret_short + 0.5 * ret_long
    # Invert: dollar strength = bearish risk
    return _clip_signal(-blended / 0.05)  # 5 % DXY move = extreme signal


# ---------------------------------------------------------------------------
# 7. Volatility of volatility (VoV)
# ---------------------------------------------------------------------------

def _compute_vov_signal(
    vix_series: np.ndarray,
    vvix_series: np.ndarray,
    window: int = 20,
) -> tuple[float, float]:
    """
    Compute VIX level signal and VoV (VVIX) as tail risk gauge.

    VIX signal (contrarian):
      VIX > 35 => panic => bullish fade
      VIX < 13 => complacency => bearish

    VoV signal:
      VVIX > 120 => elevated tail-risk => reduce bullish conviction

    Returns
    -------
    vix_signal : float in [-1, +1]
    vov_scalar : float in [0, 1] — conviction multiplier (1 = normal, 0 = chaos)
    """
    if len(vix_series) == 0:
        return 0.0, 1.0

    vix_current = float(vix_series[-1])
    vix_ma = float(np.mean(vix_series[-window:])) if len(vix_series) >= window else float(np.mean(vix_series))

    # Level component (contrarian)
    if vix_current > 40:
        vix_lv = 0.9
    elif vix_current > 30:
        vix_lv = 0.5
    elif vix_current > 22:
        vix_lv = 0.1
    elif vix_current > 15:
        vix_lv = -0.2
    else:
        vix_lv = -0.6

    # Momentum component (VIX rising fast = near-term bearish, not contrarian yet)
    vix_ret = (vix_current / (vix_ma + 1e-9)) - 1.0
    vix_mom = _clip_signal(-vix_ret / 0.30)  # 30 % rise in VIX = -1

    vix_signal = 0.6 * vix_lv + 0.4 * vix_mom

    # VoV (VVIX) conviction dampener
    if len(vvix_series) > 0:
        vvix_current = float(vvix_series[-1])
        if vvix_current > 140:
            vov_scalar = 0.4
        elif vvix_current > 120:
            vov_scalar = 0.65
        elif vvix_current > 100:
            vov_scalar = 0.85
        else:
            vov_scalar = 1.0
    else:
        vov_scalar = 1.0

    return _clip_signal(vix_signal), float(vov_scalar)


# ---------------------------------------------------------------------------
# 8. Macro regime classification
# ---------------------------------------------------------------------------

def _classify_macro_regime(
    nowcast: float,
    esi: float,
    yield_curve_signal: float,
    credit_signal: float,
    pmi_series: Optional[np.ndarray],
) -> MacroRegime:
    """
    Classify macro regime into one of four quadrants.

    Heuristic decision tree using multiple signals:

    Expansion   : growth accelerating + credit supportive + curve steepening
    Slowdown    : growth decelerating but still positive + curve flattening
    Contraction : growth negative + credit stress + curve inverted
    Recovery    : growth turning + credit normalising after stress
    """
    # Combined growth score
    growth_score = 0.5 * nowcast + 0.3 * esi + 0.2 * yield_curve_signal

    # PMI trend
    pmi_expanding = True
    pmi_accelerating = True
    if pmi_series is not None and len(pmi_series) >= 3:
        pmi_expanding    = float(pmi_series[-1]) > 50.0
        pmi_accelerating = float(pmi_series[-1]) > float(pmi_series[-2])

    if growth_score > 0.3 and credit_signal > 0.0 and pmi_expanding and pmi_accelerating:
        return MacroRegime.EXPANSION
    elif growth_score > 0.0 and not pmi_accelerating:
        return MacroRegime.SLOWDOWN
    elif growth_score < -0.2 and credit_signal < -0.2:
        return MacroRegime.CONTRACTION
    elif growth_score < 0.0 and credit_signal > -0.1:
        return MacroRegime.RECOVERY
    else:
        return MacroRegime.UNCERTAIN


# ---------------------------------------------------------------------------
# 9. Regime-adjusted signal bias
# ---------------------------------------------------------------------------

_REGIME_BIAS: dict[MacroRegime, float] = {
    MacroRegime.EXPANSION:   +0.15,
    MacroRegime.SLOWDOWN:    -0.10,
    MacroRegime.CONTRACTION: -0.20,
    MacroRegime.RECOVERY:    +0.08,
    MacroRegime.UNCERTAIN:    0.00,
}


# ---------------------------------------------------------------------------
# Component weights
# ---------------------------------------------------------------------------

_COMPONENT_WEIGHTS = {
    "nowcast":          0.20,
    "esi":              0.15,
    "yield_curve":      0.15,
    "credit_spread":    0.18,
    "cross_asset_mom":  0.15,
    "dxy_momentum":     0.10,
    "vix_signal":       0.07,
}
assert abs(sum(_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_macro_momentum_signal(data: MacroMomentumInput) -> DomainSignal:
    """
    Compute the composite macro momentum signal.

    Parameters
    ----------
    data : MacroMomentumInput

    Returns
    -------
    DomainSignal
        domain='macro', value in [-1, +1].
    """
    warnings: list[str] = []

    # --- ESI ---
    esi, per_release_surprises = _compute_economic_surprise_index(
        data.economic_releases, halflife_days=30.0
    )

    # --- Nowcast ---
    nowcast = _compute_nowcast_score(data.economic_releases, data.pmi_series)

    # --- Yield curve ---
    yield_curve_signal, curve_shape = _compute_yield_curve_signal(
        data.yield_2y, data.yield_10y, data.yield_30y
    )

    # --- Credit spreads ---
    credit_signal = _compute_credit_spread_signal(data.ig_oas, data.hy_oas)

    # --- Cross-asset momentum ---
    cross_asset_signal, per_asset_signals = _compute_cross_asset_momentum(
        data.asset_series, lookback=data.lookback_momentum
    )

    # --- DXY momentum ---
    dxy_signal = _compute_dxy_momentum(data.dxy_series)

    # --- VIX / VoV ---
    vix_signal, vov_scalar = _compute_vov_signal(data.vix_series, data.vvix_series)

    # --- Regime classification ---
    regime = _classify_macro_regime(
        nowcast, esi, yield_curve_signal, credit_signal, data.pmi_series
    )

    # --- Composite ---
    components = {
        "nowcast":         nowcast,
        "esi":             esi,
        "yield_curve":     yield_curve_signal,
        "credit_spread":   credit_signal,
        "cross_asset_mom": cross_asset_signal,
        "dxy_momentum":    dxy_signal,
        "vix_signal":      vix_signal,
    }

    composite = float(sum(
        _COMPONENT_WEIGHTS[k] * v for k, v in components.items()
    ))

    # Apply regime bias
    regime_bias = _REGIME_BIAS.get(regime, 0.0)
    composite = _clip_signal(composite + regime_bias)

    # Apply VoV damping to conviction (not to value)
    raw_conviction = float(np.mean(np.abs(list(components.values()))))  # avg signal strength
    conviction = _clip_signal(raw_conviction * vov_scalar)

    # Agreement-based conviction adjustment
    values = np.array(list(components.values()))
    agree_frac = float(np.mean(np.sign(values) == np.sign(composite)))
    conviction = float(np.clip(0.5 * conviction + 0.5 * agree_frac * vov_scalar, 0.0, 1.0))

    # Warnings
    if not data.economic_releases:
        warnings.append("No economic releases provided; nowcast and ESI are zero.")
    if len(data.dxy_series) < 20:
        warnings.append("DXY series has fewer than 20 bars; DXY signal may be unreliable.")
    if len(data.vix_series) < 10:
        warnings.append("VIX series has fewer than 10 bars; VIX signal may be unreliable.")
    if regime == MacroRegime.CONTRACTION:
        warnings.append("Macro regime: CONTRACTION — elevated recession risk.")
    if vov_scalar < 0.7:
        warnings.append(f"VoV scalar {vov_scalar:.2f}: high volatility uncertainty, conviction dampened.")

    return DomainSignal(
        domain="macro",
        value=composite,
        conviction=conviction,
        regime=regime.value,
        components=components,
        metadata={
            "curve_shape":          curve_shape,
            "yield_2s10s_bps":      (data.yield_10y - data.yield_2y) * 10_000.0,
            "ig_oas":               data.ig_oas,
            "hy_oas":               data.hy_oas,
            "vov_scalar":           vov_scalar,
            "per_release_surprises": per_release_surprises,
            "per_asset_signals":    per_asset_signals,
            "regime_bias_applied":  regime_bias,
        },
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Convenience: compute from scalar inputs (no dataclasses needed)
# ---------------------------------------------------------------------------

def compute_macro_signal_simple(
    *,
    releases_names: list[str],
    releases_actual: list[float],
    releases_consensus: list[float],
    releases_prior: list[float],
    releases_weights: list[float],
    releases_ages: list[int],
    yield_2y: float,
    yield_10y: float,
    yield_30y: float,
    ig_oas: float,
    hy_oas: float,
    dxy_prices: np.ndarray,
    vix_prices: np.ndarray,
    vvix_prices: np.ndarray,
    equity_prices: Optional[np.ndarray] = None,
    bond_prices: Optional[np.ndarray] = None,
    commodity_prices: Optional[np.ndarray] = None,
    pmi_readings: Optional[np.ndarray] = None,
    lookback: int = 63,
) -> DomainSignal:
    """
    Simplified entry point: pass lists/arrays directly.

    Assembles MacroMomentumInput internally and calls compute_macro_momentum_signal().
    """
    releases = [
        EconomicRelease(
            name=n, actual=a, consensus=c, prior=p, weight=w, release_age_days=ag
        )
        for n, a, c, p, w, ag in zip(
            releases_names, releases_actual, releases_consensus,
            releases_prior, releases_weights, releases_ages
        )
    ]

    asset_series: list[AssetMomentumSeries] = []
    if equity_prices is not None and len(equity_prices) > 5:
        asset_series.append(AssetMomentumSeries("SPX", equity_prices, "equity"))
    if bond_prices is not None and len(bond_prices) > 5:
        asset_series.append(AssetMomentumSeries("TLT", bond_prices, "bond"))
    if commodity_prices is not None and len(commodity_prices) > 5:
        asset_series.append(AssetMomentumSeries("GSCI", commodity_prices, "commodity"))
    if len(dxy_prices) > 5:
        asset_series.append(AssetMomentumSeries("DXY", dxy_prices, "fx"))
    if len(vix_prices) > 5:
        asset_series.append(AssetMomentumSeries("VIX", vix_prices, "vol"))

    inp = MacroMomentumInput(
        economic_releases=releases,
        yield_2y=yield_2y,
        yield_10y=yield_10y,
        yield_30y=yield_30y,
        ig_oas=ig_oas,
        hy_oas=hy_oas,
        dxy_series=dxy_prices,
        vix_series=vix_prices,
        vvix_series=vvix_prices,
        asset_series=asset_series,
        pmi_series=pmi_readings,
        lookback_momentum=lookback,
    )
    return compute_macro_momentum_signal(inp)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # Synthetic price series
    def _fake_prices(start, n, drift=0.0003, vol=0.012):
        rets = rng.normal(drift, vol, n)
        return start * np.cumprod(1 + rets)

    releases = [
        EconomicRelease("PMI",  52.5, 51.0, 51.5, weight=2.0, release_age_days=3),
        EconomicRelease("NFP",  220_000, 185_000, 195_000, weight=2.5, release_age_days=7),
        EconomicRelease("CPI",  0.031, 0.032, 0.030, weight=2.0, release_age_days=14),
        EconomicRelease("GDP",  0.024, 0.020, 0.019, weight=2.5, release_age_days=30),
    ]

    dxy_p   = _fake_prices(103.0, 120, drift=-0.0001)
    vix_p   = _fake_prices(18.0,   90, drift=-0.001, vol=0.05)
    vvix_p  = _fake_prices(95.0,   90, drift=0.0,    vol=0.03)
    spx_p   = _fake_prices(5400,  120, drift=0.0004)
    tlt_p   = _fake_prices(93.0,  120, drift=-0.0002)
    gsci_p  = _fake_prices(550.0, 120, drift=0.0001)
    pmi_arr = np.array([49.5, 50.1, 51.0, 51.8, 52.5])

    asset_list = [
        AssetMomentumSeries("SPX",  spx_p,  "equity"),
        AssetMomentumSeries("TLT",  tlt_p,  "bond"),
        AssetMomentumSeries("GSCI", gsci_p, "commodity"),
        AssetMomentumSeries("DXY",  dxy_p,  "fx"),
        AssetMomentumSeries("VIX",  vix_p,  "vol"),
    ]

    inp = MacroMomentumInput(
        economic_releases=releases,
        yield_2y=0.0430,
        yield_10y=0.0455,
        yield_30y=0.0465,
        ig_oas=105.0,
        hy_oas=360.0,
        dxy_series=dxy_p,
        vix_series=vix_p,
        vvix_series=vvix_p,
        asset_series=asset_list,
        pmi_series=pmi_arr,
        lookback_momentum=63,
    )

    sig = compute_macro_momentum_signal(inp)

    print(f"domain    : {sig.domain}")
    print(f"value     : {sig.value:+.4f}")
    print(f"conviction: {sig.conviction:.4f}")
    print(f"regime    : {sig.regime}")
    print("components:")
    for k, v in sig.components.items():
        print(f"  {k:<20s}: {v:+.4f}")
    print("metadata (selected):")
    for k in ("curve_shape", "yield_2s10s_bps", "ig_oas", "hy_oas", "vov_scalar"):
        print(f"  {k:<24s}: {sig.metadata[k]}")
    if sig.warnings:
        print("warnings:", sig.warnings)
