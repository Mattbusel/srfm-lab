"""
idea-engine/signals/funding_rate_signal.py

Crypto perpetual futures funding rate signal generator.

Analyzes funding rate time series to produce actionable signals:
  - Cumulative funding over rolling window
  - Extreme funding detection: >2 std = crowd sentiment extreme
  - Funding divergence: price rising but funding turning negative = warning
  - Funding regime classification: positive / negative / neutral
  - Mean reversion thesis: extreme positive funding → price will revert down
  - Carry trade: systematic funding harvesting conditions
  - Term structure of funding: short-term vs historical comparison
  - Multi-exchange divergence analysis

Public entry point: compute_funding_signal(funding_rates, prices, ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ── Enums ─────────────────────────────────────────────────────────────────────

class FundingRegime(str, Enum):
    EXTREME_POSITIVE = "extreme_positive"   # >2 std: crowded longs → short signal
    POSITIVE = "positive"                   # longs pay shorts: bullish sentiment
    NEUTRAL = "neutral"                     # near zero: no directional bias
    NEGATIVE = "negative"                   # shorts pay longs: bearish sentiment
    EXTREME_NEGATIVE = "extreme_negative"   # <-2 std: crowded shorts → long signal


class FundingSignalDirection(str, Enum):
    STRONG_SELL = "strong_sell"   # mean reversion from extreme positive
    SELL = "sell"
    NEUTRAL = "neutral"
    BUY = "buy"
    STRONG_BUY = "strong_buy"     # mean reversion from extreme negative


class CarryCondition(str, Enum):
    FAVORABLE = "favorable"       # stable positive funding, low vol → harvest shorts
    MARGINAL = "marginal"
    UNFAVORABLE = "unfavorable"   # negative or highly volatile funding


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class FundingMetrics:
    """Complete funding rate analysis output."""
    # Raw statistics
    current_rate: float            # most recent 8h funding rate (bps or %)
    cumulative_funding_7d: float   # sum of funding over 7 days (21 periods)
    cumulative_funding_30d: float  # sum of funding over 30 days

    # Normalized measures
    funding_zscore: float          # z-score vs rolling window
    funding_percentile: float      # percentile rank vs history (0-100)

    # Regime and signal
    regime: FundingRegime
    direction: FundingSignalDirection
    signal_strength: float         # 0.0 to 1.0

    # Divergence
    price_funding_divergence: bool
    divergence_score: float        # positive = bullish divergence, negative = bearish
    divergence_bars: int           # how many bars the divergence has persisted

    # Term structure
    short_term_avg: float          # mean of last 5 periods
    long_term_avg: float           # mean of last 60 periods
    term_structure_slope: float    # short_term_avg - long_term_avg
    term_structure_regime: str     # "contango" / "backwardation" / "flat"

    # Carry trade
    carry_condition: CarryCondition
    carry_score: float             # annualized expected carry (%)
    carry_stability: float         # 1 - CV of recent funding rates (0=unstable, 1=stable)

    # Mean reversion
    mean_reversion_signal: float   # signed: >0 = expect down, <0 = expect up
    days_since_extreme: int        # days since last extreme reading

    # Metadata
    lookback_periods: int
    n_extreme_events_30d: int


@dataclass
class FundingExtremeEvent:
    """A detected funding extreme event."""
    period_index: int
    funding_rate: float
    zscore: float
    direction: str          # "positive" or "negative"
    price_at_event: float
    subsequent_return: Optional[float] = None   # filled in backtesting


@dataclass
class CarryTradeAnalysis:
    """Analysis of funding carry trade conditions."""
    eligible: bool
    expected_daily_carry_bps: float
    annualized_carry_pct: float
    sharpe_estimate: float
    max_drawdown_estimate: float
    recommendation: str


# ── Core Signal Functions ─────────────────────────────────────────────────────

def cumulative_funding(
    funding_rates: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Rolling cumulative sum of funding rates over `window` periods.
    Each period is typically 8 hours (3 per day).
    Returns array of same length as input (nan-padded at start).
    """
    n = len(funding_rates)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = funding_rates[i - window + 1:i + 1].sum()
    return result


def funding_zscore_series(
    funding_rates: np.ndarray,
    lookback: int = 336,  # 56 days of 8h periods
) -> np.ndarray:
    """
    Rolling z-score of funding rates.
    z_t = (f_t - mean(f_{t-lookback:t})) / std(f_{t-lookback:t})
    """
    n = len(funding_rates)
    z = np.full(n, np.nan)
    for i in range(lookback, n):
        window = funding_rates[i - lookback:i]
        mu = window.mean()
        sigma = window.std()
        if sigma > 1e-10:
            z[i] = (funding_rates[i] - mu) / sigma
        else:
            z[i] = 0.0
    return z


def classify_regime(zscore: float, threshold: float = 2.0) -> FundingRegime:
    """Classify current funding regime based on z-score."""
    if zscore > threshold:
        return FundingRegime.EXTREME_POSITIVE
    elif zscore > 0.5:
        return FundingRegime.POSITIVE
    elif zscore < -threshold:
        return FundingRegime.EXTREME_NEGATIVE
    elif zscore < -0.5:
        return FundingRegime.NEGATIVE
    else:
        return FundingRegime.NEUTRAL


def mean_reversion_signal(
    regime: FundingRegime,
    zscore: float,
    funding_momentum: float,  # recent change in funding rate
) -> float:
    """
    Mean reversion thesis:
    - Extreme positive funding + funding starting to decline → strong sell signal
    - Extreme negative funding + funding starting to recover → strong buy signal

    Returns signed float: positive = expect price decline, negative = expect rise.
    Range: [-1, 1]
    """
    if regime == FundingRegime.EXTREME_POSITIVE:
        # Stronger signal if momentum also turning negative
        base = min(zscore / 4.0, 1.0)
        momentum_adj = 0.2 if funding_momentum < 0 else 0.0
        return min(base + momentum_adj, 1.0)
    elif regime == FundingRegime.EXTREME_NEGATIVE:
        base = max(zscore / 4.0, -1.0)
        momentum_adj = -0.2 if funding_momentum > 0 else 0.0
        return max(base - momentum_adj, -1.0)
    elif regime == FundingRegime.POSITIVE:
        return min(zscore * 0.15, 0.3)
    elif regime == FundingRegime.NEGATIVE:
        return max(zscore * 0.15, -0.3)
    return 0.0


def detect_price_funding_divergence(
    prices: np.ndarray,
    funding_rates: np.ndarray,
    window: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect divergence between price trend and funding rate trend.

    Bullish divergence: price falling but funding going up (shorts piling → squeeze)
    Bearish divergence: price rising but funding turning negative (longs leaving)

    Returns:
        divergence_score: positive = bullish, negative = bearish
        divergence_bars: number of consecutive divergence periods
    """
    n = len(prices)
    div_score = np.zeros(n)
    div_bars = np.zeros(n, dtype=int)

    for i in range(window, n):
        price_window = prices[i - window:i + 1]
        fund_window = funding_rates[i - window:i + 1]

        # Compute slopes via linear regression
        x = np.arange(window + 1)
        price_slope = float(sp_stats.linregress(x, price_window).slope)
        fund_slope = float(sp_stats.linregress(x, fund_window).slope)

        # Normalize slopes
        price_std = price_window.std() + 1e-10
        fund_std = fund_window.std() + 1e-10
        pn = price_slope / price_std
        fn = fund_slope / fund_std

        # Divergence: opposite signs
        if pn > 0.3 and fn < -0.3:
            # Price up, funding down = bearish divergence
            div_score[i] = -abs(pn * fn)
        elif pn < -0.3 and fn > 0.3:
            # Price down, funding up = crowding of shorts = potential squeeze
            div_score[i] = abs(pn * fn)
        else:
            div_score[i] = 0.0

        # Count consecutive divergence
        if i > window and div_score[i] != 0 and np.sign(div_score[i]) == np.sign(div_score[i-1]):
            div_bars[i] = div_bars[i - 1] + 1
        else:
            div_bars[i] = 1 if div_score[i] != 0 else 0

    return div_score, div_bars


def funding_term_structure(
    funding_rates: np.ndarray,
    short_window: int = 15,   # 5 days of 8h periods
    long_window: int = 180,   # 60 days
) -> Tuple[float, float, float, str]:
    """
    Analyze funding rate term structure.
    Returns: (short_avg, long_avg, slope, regime_label)
    """
    short_avg = float(funding_rates[-short_window:].mean()) if len(funding_rates) >= short_window else float(funding_rates.mean())
    long_avg = float(funding_rates[-long_window:].mean()) if len(funding_rates) >= long_window else float(funding_rates.mean())
    slope = short_avg - long_avg

    if slope > 0.0001:
        regime_label = "contango"    # short-term funding elevated vs history
    elif slope < -0.0001:
        regime_label = "backwardation"  # short-term funding depressed
    else:
        regime_label = "flat"

    return short_avg, long_avg, slope, regime_label


def carry_trade_analysis(
    funding_rates: np.ndarray,
    prices: np.ndarray,
    window: int = 63,   # ~21 days of 8h periods
    min_carry_threshold_bps: float = 2.0,
    max_vol_threshold: float = 0.05,
) -> CarryTradeAnalysis:
    """
    Assess conditions for systematic funding carry trade (short perp, long spot).
    Favorable when: funding is consistently positive, price vol is manageable.

    Annualized carry = mean_funding * 3 * 365 (3 periods/day, 365 days)
    """
    recent = funding_rates[-window:]
    mean_f = float(recent.mean())
    std_f = float(recent.std())
    cv = abs(std_f / (abs(mean_f) + 1e-10))
    stability = float(np.clip(1.0 - cv / 5.0, 0.0, 1.0))

    # Price volatility (realized, annualized)
    if len(prices) >= 2:
        log_ret = np.diff(np.log(prices[-window:]))
        price_vol = float(log_ret.std() * math.sqrt(3 * 365))
    else:
        price_vol = 1.0

    expected_daily_carry = mean_f * 3.0  # 3 periods/day in bps
    annualized_carry = expected_daily_carry * 365.0 / 100.0  # convert to %

    eligible = (
        mean_f > min_carry_threshold_bps / 100.0 and
        price_vol < max_vol_threshold and
        stability > 0.4
    )

    # Rough Sharpe estimate
    if std_f > 1e-10:
        sharpe_est = (mean_f / std_f) * math.sqrt(3 * 252)
    else:
        sharpe_est = 0.0

    # Max drawdown estimate (rough: 3x std funding over 30 days)
    mdd_est = abs(std_f * 3.0 * 30 * 3)

    if eligible:
        recommendation = f"CARRY_FAVORABLE: {annualized_carry:.1f}% annual, stability={stability:.2f}"
    elif mean_f > 0:
        recommendation = f"CARRY_MARGINAL: positive but unstable (cv={cv:.1f})"
    else:
        recommendation = "CARRY_UNFAVORABLE: negative or near-zero funding"

    condition = (
        CarryCondition.FAVORABLE if eligible else
        CarryCondition.MARGINAL if mean_f > 0 else
        CarryCondition.UNFAVORABLE
    )

    return CarryTradeAnalysis(
        eligible=eligible,
        expected_daily_carry_bps=expected_daily_carry * 10000,
        annualized_carry_pct=annualized_carry,
        sharpe_estimate=float(sharpe_est),
        max_drawdown_estimate=float(mdd_est),
        recommendation=recommendation,
    )


def detect_extreme_events(
    funding_rates: np.ndarray,
    prices: np.ndarray,
    zscore_series: np.ndarray,
    threshold: float = 2.0,
) -> List[FundingExtremeEvent]:
    """Find all historical extreme funding events."""
    events = []
    for i in range(len(funding_rates)):
        if np.isnan(zscore_series[i]):
            continue
        if abs(zscore_series[i]) >= threshold:
            direction = "positive" if zscore_series[i] > 0 else "negative"
            events.append(FundingExtremeEvent(
                period_index=i,
                funding_rate=float(funding_rates[i]),
                zscore=float(zscore_series[i]),
                direction=direction,
                price_at_event=float(prices[i]) if i < len(prices) else np.nan,
            ))
    return events


def direction_from_mr_signal(mr_signal: float) -> FundingSignalDirection:
    """Map mean reversion signal to direction enum."""
    if mr_signal > 0.6:
        return FundingSignalDirection.STRONG_SELL
    elif mr_signal > 0.2:
        return FundingSignalDirection.SELL
    elif mr_signal < -0.6:
        return FundingSignalDirection.STRONG_BUY
    elif mr_signal < -0.2:
        return FundingSignalDirection.BUY
    return FundingSignalDirection.NEUTRAL


def days_since_last_extreme(
    zscore_series: np.ndarray,
    threshold: float = 2.0,
    periods_per_day: int = 3,
) -> int:
    """Count trading periods (converted to days) since last extreme funding event."""
    for i in range(len(zscore_series) - 1, -1, -1):
        if not np.isnan(zscore_series[i]) and abs(zscore_series[i]) >= threshold:
            return (len(zscore_series) - 1 - i) // periods_per_day
    return 999  # never seen an extreme


def multi_exchange_divergence(
    funding_by_exchange: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute pairwise divergence between funding rates across exchanges.
    Large divergence = potential cross-exchange arbitrage opportunity.
    Returns dict of pair_name → current spread in bps.
    """
    exchanges = list(funding_by_exchange.keys())
    spreads = {}
    for i in range(len(exchanges)):
        for j in range(i + 1, len(exchanges)):
            ex1, ex2 = exchanges[i], exchanges[j]
            f1 = funding_by_exchange[ex1]
            f2 = funding_by_exchange[ex2]
            min_len = min(len(f1), len(f2))
            if min_len > 0:
                spread = float(f1[-1] - f2[-1]) * 10000  # convert to bps
                spreads[f"{ex1}_{ex2}"] = spread
    return spreads


# ── Main Entry Point ──────────────────────────────────────────────────────────

def compute_funding_signal(
    funding_rates: np.ndarray,
    prices: np.ndarray,
    lookback: int = 336,
    zscore_threshold: float = 2.0,
    short_window: int = 15,
    long_window: int = 180,
) -> FundingMetrics:
    """
    Compute comprehensive funding rate signal metrics.

    Parameters
    ----------
    funding_rates : array of 8h funding rates (as decimals, e.g. 0.0001 = 1 bps)
    prices : corresponding price series (same length)
    lookback : periods for z-score rolling window (default 336 = 56 days)
    zscore_threshold : extreme detection threshold (default 2.0 std)
    short_window : short-term average window (periods)
    long_window : long-term average window (periods)

    Returns
    -------
    FundingMetrics with all computed signal components
    """
    n = len(funding_rates)
    if n < 21:
        raise ValueError(f"Need at least 21 funding rate observations, got {n}")

    # Cumulative funding
    cum_7d = float(cumulative_funding(funding_rates, min(21, n))[-1])
    cum_30d = float(cumulative_funding(funding_rates, min(90, n))[-1])

    # Z-scores
    zs = funding_zscore_series(funding_rates, lookback=min(lookback, n - 1))
    current_z = float(zs[-1]) if not np.isnan(zs[-1]) else 0.0

    # Percentile rank
    hist = funding_rates[max(0, n - lookback):]
    pctile = float(sp_stats.percentileofscore(hist, funding_rates[-1]))

    # Regime
    regime = classify_regime(current_z, zscore_threshold)

    # Funding momentum (recent change)
    if n >= 3:
        fund_momentum = float(funding_rates[-1] - funding_rates[-3])
    else:
        fund_momentum = 0.0

    # Mean reversion signal
    mr = mean_reversion_signal(regime, current_z, fund_momentum)
    direction = direction_from_mr_signal(mr)
    strength = abs(mr)

    # Divergence
    div_score, div_bars = detect_price_funding_divergence(
        prices, funding_rates, window=min(21, n // 2)
    )
    current_div = float(div_score[-1])
    current_div_bars = int(div_bars[-1])
    has_divergence = abs(current_div) > 0.5

    # Term structure
    st_avg, lt_avg, ts_slope, ts_regime = funding_term_structure(
        funding_rates, short_window=min(short_window, n), long_window=min(long_window, n)
    )

    # Carry trade
    carry = carry_trade_analysis(funding_rates, prices)
    carry_cond = carry.condition if hasattr(carry, 'condition') else (
        CarryCondition.FAVORABLE if carry.eligible else
        CarryCondition.MARGINAL if funding_rates[-1] > 0 else
        CarryCondition.UNFAVORABLE
    )
    carry_stab = float(np.clip(1.0 - funding_rates[-min(63, n):].std() /
                               (abs(funding_rates[-min(63, n):].mean()) + 1e-10) / 5.0, 0.0, 1.0))

    # Extreme events in last 30 days
    extreme_events_30d = [
        e for e in detect_extreme_events(
            funding_rates[-90:], prices[-90:] if len(prices) >= 90 else prices,
            zs[-90:]
        )
    ]
    n_extreme = len(extreme_events_30d)

    # Days since extreme
    dse = days_since_last_extreme(zs, zscore_threshold)

    return FundingMetrics(
        current_rate=float(funding_rates[-1]),
        cumulative_funding_7d=cum_7d,
        cumulative_funding_30d=cum_30d,
        funding_zscore=current_z,
        funding_percentile=pctile,
        regime=regime,
        direction=direction,
        signal_strength=strength,
        price_funding_divergence=has_divergence,
        divergence_score=current_div,
        divergence_bars=current_div_bars,
        short_term_avg=st_avg,
        long_term_avg=lt_avg,
        term_structure_slope=ts_slope,
        term_structure_regime=ts_regime,
        carry_condition=carry_cond,
        carry_score=carry.annualized_carry_pct,
        carry_stability=carry_stab,
        mean_reversion_signal=mr,
        days_since_extreme=dse,
        lookback_periods=lookback,
        n_extreme_events_30d=n_extreme,
    )


def summarize_signal(metrics: FundingMetrics) -> str:
    """Human-readable summary of funding signal."""
    lines = [
        f"Funding Signal Summary",
        f"  Current Rate:    {metrics.current_rate * 10000:.2f} bps",
        f"  Z-Score:         {metrics.funding_zscore:.2f}",
        f"  Percentile:      {metrics.funding_percentile:.0f}th",
        f"  Regime:          {metrics.regime.value}",
        f"  Direction:       {metrics.direction.value}",
        f"  Strength:        {metrics.signal_strength:.2f}",
        f"  Divergence:      {'YES' if metrics.price_funding_divergence else 'no'} (score={metrics.divergence_score:.3f}, bars={metrics.divergence_bars})",
        f"  Term Structure:  {metrics.term_structure_regime} (slope={metrics.term_structure_slope * 10000:.2f} bps)",
        f"  Carry:           {metrics.carry_condition.value} ({metrics.carry_score:.1f}% ann, stab={metrics.carry_stability:.2f})",
        f"  MR Signal:       {metrics.mean_reversion_signal:.3f}",
        f"  Days Since Ext.: {metrics.days_since_extreme}",
        f"  Extreme 30d:     {metrics.n_extreme_events_30d}",
    ]
    return "\n".join(lines)
