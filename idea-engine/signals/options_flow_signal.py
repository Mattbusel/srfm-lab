"""
signals/options_flow_signal.py

Advanced options flow signal generator for the idea engine.

Computes:
  - GEX (Gamma Exposure) by strike
  - DEX (Delta Exposure): directional pressure from options market
  - Charm (delta decay): intraday directional pressure near expiry
  - 0DTE flow: same-day expiry options — immediate directional signal
  - Put/Call ratio (volume, OI, dollar-weighted)
  - Vol skew signal: 25-delta put vol minus 25-delta call vol
  - Options-implied move: sqrt(365/DTE) * ATM_iv * spot / sqrt(2*pi)
  - Dealer hedging flow: GEX sign flip => vol amplification regime
  - Large block sweep detection: aggressive fills above avg size threshold
  - Composite DomainSignal output with domain='options_flow'

All arrays are numpy-based. Public entry point: compute_options_flow_signal().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# DomainSignal (local definition — importable from idea_engine.signals if present)
# ---------------------------------------------------------------------------

@dataclass
class DomainSignal:
    """
    Standardised signal output container used across all idea-engine signal modules.

    Attributes
    ----------
    domain : str
        Signal domain identifier, e.g. 'options_flow'.
    value : float
        Normalised directional score in [-1, +1].
        +1 = maximum bullish conviction, -1 = maximum bearish conviction.
    conviction : float
        Confidence/weight in [0, 1].
    regime : str
        Human-readable regime label.
    components : dict
        Named sub-signal scores contributing to the composite.
    metadata : dict
        Arbitrary extra information (strikes, expiries, raw stats, etc.).
    warnings : list[str]
        Non-fatal advisory messages produced during computation.
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

class OptionsRegime(str, Enum):
    VOL_SUPPRESSION   = "vol_suppression"    # GEX strongly positive — dealers stabilise
    VOL_AMPLIFICATION = "vol_amplification"  # GEX negative — dealers destabilise
    NEUTRAL           = "neutral"


class PCRSentiment(str, Enum):
    EXTREME_FEAR  = "extreme_fear"
    FEAR          = "fear"
    NEUTRAL       = "neutral"
    GREED         = "greed"
    EXTREME_GREED = "extreme_greed"


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OptionContract:
    """
    A single option contract record.

    Parameters
    ----------
    strike : float
    expiry_dte : float
        Calendar days to expiration (0 = 0DTE).
    option_type : str
        'call' or 'put'.
    open_interest : float
        Number of contracts open.
    volume : float
        Contracts traded in the current session.
    implied_vol : float
        Annualised implied volatility (e.g. 0.25 = 25 %).
    delta : float
        Black-Scholes delta (positive for calls, negative for puts).
    gamma : float
        Black-Scholes gamma (always positive).
    last_price : float
        Last option premium traded.
    is_aggressive : bool
        True if the fill was at or through the ask (sweep).
    """
    strike: float
    expiry_dte: float
    option_type: str           # 'call' | 'put'
    open_interest: float
    volume: float
    implied_vol: float
    delta: float
    gamma: float
    last_price: float
    is_aggressive: bool = False


@dataclass
class OptionsFlowInput:
    """
    Full options chain snapshot required by compute_options_flow_signal().

    Parameters
    ----------
    spot : float
        Current underlying price.
    contracts : list[OptionContract]
        All contracts in the chain (any expiry, any strike).
    atm_iv : float
        At-the-money implied vol for the front-month (used for implied move).
    iv_25d_put : float
        25-delta put implied vol.
    iv_25d_call : float
        25-delta call implied vol.
    front_month_dte : float
        DTE of the front-month expiry used for implied-move calculation.
    vix : float, optional
        VIX level for regime context.
    vvix : float, optional
        VVIX (vol of vol) level.
    """
    spot: float
    contracts: list[OptionContract]
    atm_iv: float
    iv_25d_put: float
    iv_25d_call: float
    front_month_dte: float
    vix: float = 20.0
    vvix: float = 90.0


# ---------------------------------------------------------------------------
# Black-Scholes helpers (numpy, no scipy)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Approximation of the standard normal CDF using erf."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / _SQRT2))


def _norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal PDF."""
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT2PI


def _bs_delta(S: float, K: float, T: float, sigma: float,
              r: float = 0.05, option_type: str = "call") -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    if option_type == "call":
        return float(_norm_cdf(d1))
    return float(_norm_cdf(d1)) - 1.0


def _bs_gamma(S: float, K: float, T: float, sigma: float,
              r: float = 0.05) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return float(_norm_pdf(d1)) / (S * sigma * sqrtT)


def _bs_charm(S: float, K: float, T: float, sigma: float,
              r: float = 0.05, option_type: str = "call") -> float:
    """
    Charm (delta decay) = d(delta)/d(time).

    Approximation:
        charm_call = -pdf(d1) * [2*r*T - d2*sigma*sqrt(T)] / (2*T*sigma*sqrt(T))
    """
    if T <= 1e-6 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    pdf_d1 = float(_norm_pdf(d1))
    charm = -pdf_d1 * (2.0 * r * T - d2 * sigma * sqrtT) / (2.0 * T * sigma * sqrtT)
    if option_type == "put":
        charm = charm  # put charm is same formula; sign handled by delta convention
    return charm


# ---------------------------------------------------------------------------
# GEX — Gamma Exposure
# ---------------------------------------------------------------------------

def _compute_gex(
    contracts: list[OptionContract],
    spot: float,
) -> tuple[dict[float, float], float]:
    """
    Compute GEX by strike and aggregate GEX.

    GEX per contract = dealer_gamma * OI * 100 * spot^2 * 0.01

    Dealers are assumed to be short options (net short gamma):
      - For calls: dealers are short => dealer_gamma = +gamma (dealers buy
        underlying as price rises to stay delta-neutral, acting as stabiliser).
      - For puts: dealers are short => dealer_gamma = +gamma when customers are
        long puts.  However, if put/call sentiment suggests customers are long
        puts (fear), dealer GEX from puts is negative.

    Sign convention used here (standard GEX):
      call GEX contribution = +gamma * OI
      put GEX contribution  = -gamma * OI  (dealers short puts = long gamma
                                             from puts is unusual; standard
                                             GEX treats puts as negative)

    Returns
    -------
    gex_by_strike : dict mapping strike -> gex_dollars
    net_gex : float (positive = vol suppression, negative = vol amplification)
    """
    gex_by_strike: dict[float, float] = {}
    multiplier = 100.0 * (spot ** 2) * 0.01

    for c in contracts:
        sign = 1.0 if c.option_type == "call" else -1.0
        contract_gex = sign * c.gamma * c.open_interest * multiplier
        gex_by_strike[c.strike] = gex_by_strike.get(c.strike, 0.0) + contract_gex

    net_gex = sum(gex_by_strike.values())
    return gex_by_strike, net_gex


# ---------------------------------------------------------------------------
# DEX — Delta Exposure
# ---------------------------------------------------------------------------

def _compute_dex(
    contracts: list[OptionContract],
    spot: float,
) -> float:
    """
    Delta Exposure: net directional pressure from the options market.

    DEX = sum over all contracts of (delta * OI * 100)
    Positive DEX => net long delta (bullish pressure on dealers to sell hedge).
    Negative DEX => net short delta (bearish pressure).

    Returns normalised DEX in [-1, +1] relative to total OI.
    """
    total_oi = sum(c.open_interest for c in contracts) + 1e-9
    raw_dex = 0.0
    for c in contracts:
        # delta is signed: positive for calls, negative for puts
        raw_dex += c.delta * c.open_interest * 100.0

    # Normalise: max possible raw_dex ≈ total_oi * 100 (all calls ATM, delta=0.5)
    norm_factor = total_oi * 100.0 * 0.5 + 1e-9
    return float(np.clip(raw_dex / norm_factor, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Charm aggregation (intraday directional pressure from delta decay)
# ---------------------------------------------------------------------------

def _compute_aggregate_charm(
    contracts: list[OptionContract],
    spot: float,
    r: float = 0.05,
) -> float:
    """
    Aggregate Charm signal.

    Sum of charm * OI across all contracts gives dealer re-hedging pressure
    over the trading day.  Positive = dealers will buy underlying; negative =
    dealers will sell.

    Returns a normalised value in [-1, +1].
    """
    total_oi = sum(c.open_interest for c in contracts) + 1e-9
    charm_sum = 0.0

    for c in contracts:
        T = max(c.expiry_dte / 365.0, 1e-6)
        charm_val = _bs_charm(spot, c.strike, T, c.implied_vol, r, c.option_type)
        # Dealer charm: dealers are assumed short options, so charm impact is reversed
        sign = -1.0  # dealer short => negative charm impact on dealer book
        charm_sum += sign * charm_val * c.open_interest

    # Normalise by OI-weighted scale
    scale = total_oi * 0.01 + 1e-9
    return float(np.clip(charm_sum / scale, -1.0, 1.0))


# ---------------------------------------------------------------------------
# 0DTE flow
# ---------------------------------------------------------------------------

def _compute_0dte_signal(
    contracts: list[OptionContract],
    spot: float,
    dte_threshold: float = 1.0,
) -> tuple[float, float]:
    """
    Compute directional signal from same-day (0DTE) options.

    Logic:
    - Filter contracts with expiry_dte <= dte_threshold.
    - Compute volume-weighted delta skew: sum(delta * vol) / sum(vol).
    - Aggressive 0DTE call sweeps => bullish; put sweeps => bearish.

    Returns
    -------
    signal : float in [-1, +1]
    weight : float — fraction of total volume in 0DTE options (0-1)
    """
    zero_dte = [c for c in contracts if c.expiry_dte <= dte_threshold]
    all_vol = sum(c.volume for c in contracts) + 1e-9
    dte_vol = sum(c.volume for c in zero_dte) + 1e-9
    weight = min(dte_vol / all_vol, 1.0)

    if not zero_dte:
        return 0.0, 0.0

    # Volume-weighted delta
    vw_delta = sum(c.delta * c.volume for c in zero_dte) / dte_vol

    # Boost signal for aggressive fills
    aggr_vol = sum(c.volume for c in zero_dte if c.is_aggressive) + 1e-9
    aggr_ratio = min(aggr_vol / dte_vol, 1.0)

    signal = float(np.clip(vw_delta * (1.0 + 0.5 * aggr_ratio), -1.0, 1.0))
    return signal, float(weight)


# ---------------------------------------------------------------------------
# Put/Call ratios
# ---------------------------------------------------------------------------

@dataclass
class PCRatios:
    volume_pcr: float       # put volume / call volume
    oi_pcr: float           # put OI / call OI
    dollar_pcr: float       # put dollar volume / call dollar volume
    sentiment: PCRSentiment
    signal: float           # normalised -1 to +1 (negative pcr deviation = bullish)


def _compute_pcr(contracts: list[OptionContract]) -> PCRatios:
    """
    Compute put/call ratios and translate to directional signal.

    Interpretation (contrarian):
    - High PCR (>1.2) => fear => mean-revert signal = bullish (+)
    - Low PCR (<0.7)  => complacency => mean-revert signal = bearish (-)
    """
    calls = [c for c in contracts if c.option_type == "call"]
    puts  = [c for c in contracts if c.option_type == "put"]

    call_vol  = sum(c.volume for c in calls) + 1e-9
    put_vol   = sum(c.volume for c in puts)  + 1e-9
    call_oi   = sum(c.open_interest for c in calls) + 1e-9
    put_oi    = sum(c.open_interest for c in puts)  + 1e-9
    call_dlr  = sum(c.volume * c.last_price for c in calls) + 1e-9
    put_dlr   = sum(c.volume * c.last_price for c in puts)  + 1e-9

    vol_pcr    = put_vol / call_vol
    oi_pcr     = put_oi  / call_oi
    dollar_pcr = put_dlr / call_dlr

    # Blend the three ratios (equal weight)
    blend_pcr = (vol_pcr + oi_pcr + dollar_pcr) / 3.0

    # Classify sentiment
    if blend_pcr > 1.5:
        sentiment = PCRSentiment.EXTREME_FEAR
    elif blend_pcr > 1.1:
        sentiment = PCRSentiment.FEAR
    elif blend_pcr < 0.6:
        sentiment = PCRSentiment.EXTREME_GREED
    elif blend_pcr < 0.85:
        sentiment = PCRSentiment.GREED
    else:
        sentiment = PCRSentiment.NEUTRAL

    # Contrarian signal: high pcr => bullish, low pcr => bearish
    # Centre around 1.0, scale so that ±0.5 maps to ±1
    deviation = (blend_pcr - 1.0) / 0.5
    signal = float(np.clip(deviation, -2.0, 2.0)) / 2.0  # in [-1, +1]

    return PCRatios(
        volume_pcr=float(vol_pcr),
        oi_pcr=float(oi_pcr),
        dollar_pcr=float(dollar_pcr),
        sentiment=sentiment,
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Vol skew signal
# ---------------------------------------------------------------------------

def _compute_skew_signal(iv_25d_put: float, iv_25d_call: float) -> float:
    """
    Vol skew signal: 25-delta put IV minus 25-delta call IV.

    Positive skew (puts more expensive) => fear / downside hedging demand.
    Returns normalised signal in [-1, +1]:
      Large positive skew => fear => contrarian bullish or momentum bearish.
    Here we return the raw skew as a bearish signal (trend-following):
      positive skew => bearish pressure.
    Scale: typical skew ~2-8 vol points.  Clip at 10 points = ±1.
    """
    skew = iv_25d_put - iv_25d_call  # in decimal (0.03 = 3 vol pts)
    # Normalise to [-1, +1], treating 0.10 (10 vol pts) as extreme
    return float(np.clip(skew / 0.10, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Options-implied move
# ---------------------------------------------------------------------------

def _compute_implied_move(
    spot: float,
    atm_iv: float,
    dte: float,
) -> float:
    """
    Expected 1-standard-deviation move over DTE calendar days.

    implied_move = ATM_IV * spot * sqrt(DTE / 365) / sqrt(2*pi)

    Returns the move in price units (same units as spot).
    """
    if dte <= 0 or atm_iv <= 0:
        return 0.0
    return float(atm_iv * spot * math.sqrt(dte / 365.0) / _SQRT2PI)


# ---------------------------------------------------------------------------
# Dealer hedging regime
# ---------------------------------------------------------------------------

def _classify_dealer_regime(
    net_gex: float,
    spot: float,
    gex_by_strike: dict[float, float],
) -> tuple[OptionsRegime, float, float]:
    """
    Classify dealer hedging regime based on GEX.

    Parameters
    ----------
    net_gex : float
    spot : float
    gex_by_strike : dict

    Returns
    -------
    regime : OptionsRegime
    flip_distance_pct : float
        Distance to nearest GEX flip (zero-cross) as % of spot.
    dealer_pressure : float
        Normalised dealer pressure in [-1, +1].
        Positive = dealers buying (vol suppression), negative = selling.
    """
    # Find nearest GEX flip level (strike where cumulative GEX crosses zero)
    strikes = sorted(gex_by_strike.keys())
    if not strikes:
        return OptionsRegime.NEUTRAL, 0.0, 0.0

    # Classify regime
    if net_gex > 0:
        regime = OptionsRegime.VOL_SUPPRESSION
    elif net_gex < 0:
        regime = OptionsRegime.VOL_AMPLIFICATION
    else:
        regime = OptionsRegime.NEUTRAL

    # Find nearest GEX flip: where individual strike GEX changes sign
    flip_dist = float("inf")
    for k, g in gex_by_strike.items():
        if g * net_gex < 0:  # opposite sign to net
            dist = abs(k - spot) / (spot + 1e-9)
            flip_dist = min(flip_dist, dist)
    if flip_dist == float("inf"):
        flip_dist = 0.0

    # Dealer pressure: normalise net GEX
    total_abs_gex = sum(abs(v) for v in gex_by_strike.values()) + 1e-9
    dealer_pressure = float(np.clip(net_gex / total_abs_gex, -1.0, 1.0))

    return regime, float(flip_dist), dealer_pressure


# ---------------------------------------------------------------------------
# Block sweep detection
# ---------------------------------------------------------------------------

@dataclass
class SweepSummary:
    n_sweeps: int
    net_sweep_delta: float      # volume-weighted delta of sweeps
    sweep_call_volume: float
    sweep_put_volume: float
    sweep_signal: float         # -1 to +1


def _detect_sweeps(
    contracts: list[OptionContract],
    size_multiplier: float = 2.0,
) -> SweepSummary:
    """
    Identify large aggressive fills (block sweeps).

    A sweep is flagged when:
      1. is_aggressive == True
      2. volume > size_multiplier * mean_volume across all contracts

    Parameters
    ----------
    size_multiplier : float
        Threshold above mean volume to qualify as a block sweep.
    """
    if not contracts:
        return SweepSummary(0, 0.0, 0.0, 0.0, 0.0)

    volumes = np.array([c.volume for c in contracts], dtype=float)
    mean_vol = float(np.mean(volumes)) + 1e-9
    threshold = mean_vol * size_multiplier

    sweeps = [c for c in contracts if c.is_aggressive and c.volume >= threshold]
    n = len(sweeps)
    if n == 0:
        return SweepSummary(0, 0.0, 0.0, 0.0, 0.0)

    sweep_vol = sum(c.volume for c in sweeps) + 1e-9
    net_delta = sum(c.delta * c.volume for c in sweeps) / sweep_vol
    call_vol  = sum(c.volume for c in sweeps if c.option_type == "call")
    put_vol   = sum(c.volume for c in sweeps if c.option_type == "put")

    # Signal: volume-weighted delta of sweeps
    signal = float(np.clip(net_delta * 2.0, -1.0, 1.0))

    return SweepSummary(
        n_sweeps=n,
        net_sweep_delta=float(net_delta),
        sweep_call_volume=float(call_vol),
        sweep_put_volume=float(put_vol),
        sweep_signal=signal,
    )


# ---------------------------------------------------------------------------
# Composite signal assembly
# ---------------------------------------------------------------------------

# Component weights (must sum to 1.0)
_COMPONENT_WEIGHTS = {
    "gex_dealer_pressure": 0.18,
    "dex":                 0.15,
    "charm":               0.10,
    "zero_dte":            0.15,
    "pcr_contrarian":      0.12,
    "skew":                0.10,
    "sweep":               0.15,
    "vix_vvix":            0.05,
}
assert abs(sum(_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9


def _vix_vvix_signal(vix: float, vvix: float) -> float:
    """
    Translate VIX + VVIX levels into a directional signal.

    Logic (contrarian):
    - VIX > 30 and VVIX > 120 => panic => fade-down => bullish signal
    - VIX < 13               => complacency => bearish tilt
    """
    if vix > 40:
        vix_signal = 0.8
    elif vix > 30:
        vix_signal = 0.4
    elif vix > 20:
        vix_signal = 0.0
    elif vix > 15:
        vix_signal = -0.2
    else:
        vix_signal = -0.5

    if vvix > 130:
        vvix_boost = 0.3
    elif vvix > 110:
        vvix_boost = 0.1
    else:
        vvix_boost = 0.0

    return float(np.clip(vix_signal + vvix_boost, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_options_flow_signal(data: OptionsFlowInput) -> DomainSignal:
    """
    Compute the composite options flow signal.

    Parameters
    ----------
    data : OptionsFlowInput

    Returns
    -------
    DomainSignal
        domain='options_flow', value in [-1, +1].
    """
    warnings: list[str] = []

    if not data.contracts:
        warnings.append("No option contracts supplied; returning neutral signal.")
        return DomainSignal(
            domain="options_flow",
            value=0.0,
            conviction=0.0,
            regime=OptionsRegime.NEUTRAL.value,
            warnings=warnings,
        )

    # --- GEX ---
    gex_by_strike, net_gex = _compute_gex(data.contracts, data.spot)
    dealer_regime, flip_dist_pct, dealer_pressure = _classify_dealer_regime(
        net_gex, data.spot, gex_by_strike
    )

    # --- DEX ---
    dex = _compute_dex(data.contracts, data.spot)

    # --- Charm ---
    charm_signal = _compute_aggregate_charm(data.contracts, data.spot)

    # --- 0DTE ---
    zero_dte_signal, zero_dte_weight = _compute_0dte_signal(data.contracts, data.spot)

    # --- PCR (contrarian) ---
    pcr = _compute_pcr(data.contracts)
    # PCR contrarian: sign flip (high fear PCR => bullish fade)
    pcr_signal = -pcr.signal  # invert: high pcr deviation was positive (fearful) => bullish

    # --- Skew (trend-following: positive skew => bearish) ---
    skew_raw = _compute_skew_signal(data.iv_25d_put, data.iv_25d_call)
    skew_signal = -skew_raw  # invert: high fear skew = bullish contrarian OR used as bearish here

    # --- Sweeps ---
    sweeps = _detect_sweeps(data.contracts)
    sweep_signal = sweeps.sweep_signal

    # --- VIX/VVIX ---
    vix_signal = _vix_vvix_signal(data.vix, data.vvix)

    # --- Implied move (metadata only) ---
    implied_move = _compute_implied_move(data.spot, data.atm_iv, data.front_month_dte)
    implied_move_pct = implied_move / data.spot if data.spot > 0 else 0.0

    # --- Composite weighted average ---
    components = {
        "gex_dealer_pressure": dealer_pressure,
        "dex":                 dex,
        "charm":               charm_signal,
        "zero_dte":            zero_dte_signal,
        "pcr_contrarian":      pcr_signal,
        "skew":                skew_signal,
        "sweep":               sweep_signal,
        "vix_vvix":            vix_signal,
    }

    composite = float(sum(
        _COMPONENT_WEIGHTS[k] * v for k, v in components.items()
    ))
    composite = float(np.clip(composite, -1.0, 1.0))

    # --- Conviction: based on agreement across components ---
    values = np.array(list(components.values()))
    agree_fraction = float(np.mean(np.sign(values) == np.sign(composite)))
    conviction = float(np.clip(agree_fraction, 0.0, 1.0))

    # Reduce conviction if VVIX very high (unstable regime)
    if data.vvix > 140:
        conviction *= 0.7
        warnings.append("VVIX > 140: tail-risk regime, conviction reduced.")

    # Reduce conviction if fewer than 20 contracts
    if len(data.contracts) < 20:
        conviction *= 0.5
        warnings.append("Fewer than 20 contracts: limited data, conviction reduced.")

    # --- Regime string ---
    if dealer_regime == OptionsRegime.VOL_AMPLIFICATION:
        regime_label = "vol_amplification"
    elif dealer_regime == OptionsRegime.VOL_SUPPRESSION:
        regime_label = "vol_suppression"
    else:
        regime_label = "neutral"

    return DomainSignal(
        domain="options_flow",
        value=composite,
        conviction=conviction,
        regime=regime_label,
        components=components,
        metadata={
            "net_gex":             net_gex,
            "gex_flip_dist_pct":   flip_dist_pct,
            "implied_move":        implied_move,
            "implied_move_pct":    implied_move_pct,
            "vol_skew_raw":        skew_raw,
            "pcr_volume":          pcr.volume_pcr,
            "pcr_oi":              pcr.oi_pcr,
            "pcr_dollar":          pcr.dollar_pcr,
            "pcr_sentiment":       pcr.sentiment.value,
            "n_sweeps":            sweeps.n_sweeps,
            "sweep_call_volume":   sweeps.sweep_call_volume,
            "sweep_put_volume":    sweeps.sweep_put_volume,
            "zero_dte_weight":     zero_dte_weight,
            "vix":                 data.vix,
            "vvix":                data.vvix,
        },
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Convenience: build from raw numpy arrays (no OptionContract objects needed)
# ---------------------------------------------------------------------------

def compute_options_flow_from_arrays(
    spot: float,
    strikes: np.ndarray,
    expiry_dtes: np.ndarray,
    option_types: list[str],
    open_interests: np.ndarray,
    volumes: np.ndarray,
    implied_vols: np.ndarray,
    deltas: np.ndarray,
    gammas: np.ndarray,
    last_prices: np.ndarray,
    is_aggressive: np.ndarray,
    atm_iv: float,
    iv_25d_put: float,
    iv_25d_call: float,
    front_month_dte: float,
    vix: float = 20.0,
    vvix: float = 90.0,
) -> DomainSignal:
    """
    Build OptionsFlowInput from raw numpy arrays and compute the signal.

    All array arguments must have the same length (one entry per contract).

    Parameters
    ----------
    is_aggressive : np.ndarray
        Boolean or 0/1 array indicating aggressive fill.
    """
    n = len(strikes)
    contracts = []
    for i in range(n):
        contracts.append(OptionContract(
            strike=float(strikes[i]),
            expiry_dte=float(expiry_dtes[i]),
            option_type=str(option_types[i]).lower(),
            open_interest=float(open_interests[i]),
            volume=float(volumes[i]),
            implied_vol=float(implied_vols[i]),
            delta=float(deltas[i]),
            gamma=float(gammas[i]),
            last_price=float(last_prices[i]),
            is_aggressive=bool(is_aggressive[i]),
        ))

    inp = OptionsFlowInput(
        spot=spot,
        contracts=contracts,
        atm_iv=atm_iv,
        iv_25d_put=iv_25d_put,
        iv_25d_call=iv_25d_call,
        front_month_dte=front_month_dte,
        vix=vix,
        vvix=vvix,
    )
    return compute_options_flow_signal(inp)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    spot_ = 450.0
    n_c = 60

    strikes_ = np.linspace(400, 500, n_c)
    dtes_    = np.concatenate([np.zeros(10), np.full(20, 7), np.full(30, 30)])
    types_   = (["call"] * 30) + (["put"] * 30)
    oi_      = rng.uniform(100, 5000, n_c)
    vol_     = rng.uniform(10, 1000, n_c)
    ivs_     = rng.uniform(0.15, 0.45, n_c)
    deltas_  = np.array([
        rng.uniform(0.1, 0.9) if t == "call" else rng.uniform(-0.9, -0.1)
        for t in types_
    ])
    gammas_  = rng.uniform(0.001, 0.05, n_c)
    prices_  = rng.uniform(0.5, 20.0, n_c)
    aggr_    = rng.integers(0, 2, n_c)

    sig = compute_options_flow_from_arrays(
        spot=spot_,
        strikes=strikes_,
        expiry_dtes=dtes_,
        option_types=types_,
        open_interests=oi_,
        volumes=vol_,
        implied_vols=ivs_,
        deltas=deltas_,
        gammas=gammas_,
        last_prices=prices_,
        is_aggressive=aggr_,
        atm_iv=0.22,
        iv_25d_put=0.25,
        iv_25d_call=0.19,
        front_month_dte=30.0,
        vix=22.0,
        vvix=95.0,
    )

    print(f"domain   : {sig.domain}")
    print(f"value    : {sig.value:+.4f}")
    print(f"conviction: {sig.conviction:.4f}")
    print(f"regime   : {sig.regime}")
    print("components:")
    for k, v in sig.components.items():
        print(f"  {k:<28s}: {v:+.4f}")
    print("metadata (selected):")
    for k in ("net_gex", "implied_move_pct", "pcr_sentiment", "n_sweeps"):
        print(f"  {k:<28s}: {sig.metadata[k]}")
    if sig.warnings:
        print("warnings:", sig.warnings)
