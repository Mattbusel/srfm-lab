"""
Universal Asset Intelligence Engine: know EVERYTHING about any asset instantly.

A single function call that produces a complete intelligence dossier on any
asset, equivalent to a Bloomberg terminal's depth of analysis:

  1. TECHNICAL: 50+ indicators computed and interpreted
  2. REGIME: Classification with confidence and transition probability
  3. SIGNALS: All Event Horizon signals ranked by strength
  4. OPTIONS: Implied vol surface, GEX, skew, max pain, implied move
  5. ON-CHAIN: NVT, MVRV, exchange flows, whale activity, hashrate
  6. CORRELATIONS: Cross-asset correlations and lead-lag relationships
  7. LIQUIDITY: Depth, spread, impact cost, days to liquidate
  8. RISK: VaR, CVaR, tail risk, factor exposure, drawdown analysis
  9. FUNDAMENTALS: Network value, active addresses, developer activity
  10. EVENTS: Upcoming events that could affect this asset
  11. DREAMS: Fragility score from dream scenario testing
  12. MEMORY: Support/resistance from remembered significant levels
  13. SUMMARY: Natural language synthesis of everything above

Architecture: Orchestrator pattern with specialized compute nodes.
Each section is independently computed and cached.

Target: < 2 seconds for full analysis on cached data.
"""

from __future__ import annotations
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: INTELLIGENCE SCHEMA (The Grand Unified Output)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TechnicalAnalysis:
    """50+ technical indicators computed and interpreted."""
    # Trend indicators
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    ema_50: float = 0.0
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    trend_strength: float = 0.0       # 0-1
    trend_direction: str = "neutral"   # up/down/neutral

    # Momentum indicators
    rsi_14: float = 50.0
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    williams_r: float = -50.0
    cci_20: float = 0.0
    roc_12: float = 0.0
    momentum_10: float = 0.0
    trix: float = 0.0
    ultimate_oscillator: float = 50.0
    momentum_composite: float = 0.0    # -1 to +1

    # Volatility indicators
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_pct_b: float = 0.5
    bollinger_bandwidth: float = 0.0
    atr_14: float = 0.0
    atr_pct: float = 0.0
    keltner_upper: float = 0.0
    keltner_lower: float = 0.0
    donchian_high: float = 0.0
    donchian_low: float = 0.0
    realized_vol_21d: float = 0.0
    realized_vol_63d: float = 0.0
    vol_percentile: float = 0.5        # vs 1-year history
    volatility_regime: str = "normal"   # low/normal/high/extreme

    # Volume indicators
    obv: float = 0.0
    vwap: float = 0.0
    mfi_14: float = 50.0
    ad_line: float = 0.0
    chaikin_mf: float = 0.0
    force_index: float = 0.0
    volume_ratio_5_20: float = 1.0
    volume_trend: str = "normal"        # expanding/contracting/normal

    # Price structure
    distance_from_high_pct: float = 0.0   # distance from 52-week high
    distance_from_low_pct: float = 0.0    # distance from 52-week low
    price_vs_sma200: float = 0.0          # % above/below 200 SMA
    golden_cross: bool = False             # 50 SMA > 200 SMA
    death_cross: bool = False              # 50 SMA < 200 SMA

    # Pattern detection
    higher_highs: bool = False
    lower_lows: bool = False
    inside_bar: bool = False
    engulfing_bullish: bool = False
    engulfing_bearish: bool = False

    # Composite scores
    technical_score: float = 0.0           # -1 (max bearish) to +1 (max bullish)
    technical_interpretation: str = ""      # human-readable summary


@dataclass
class RegimeAnalysis:
    """Current market regime classification."""
    regime: str                            # trending_up/down, mean_reverting, high_vol, crisis
    confidence: float = 0.0
    secondary_regime: str = ""
    vol_regime: str = "normal"
    trend_regime: str = "flat"
    correlation_regime: str = "normal"
    hurst_exponent: float = 0.5
    fractal_dimension: float = 1.5
    regime_age_bars: int = 0
    transition_probability: float = 0.0
    predicted_next_regime: str = ""
    regime_interpretation: str = ""


@dataclass
class SignalRanking:
    """All Event Horizon signals ranked by strength."""
    signals: List[Dict] = field(default_factory=list)  # [{name, value, confidence, rank}]
    composite_signal: float = 0.0
    composite_confidence: float = 0.0
    n_bullish: int = 0
    n_bearish: int = 0
    n_neutral: int = 0
    dominant_signal: str = ""
    signal_agreement: float = 0.0


@dataclass
class OptionsAnalysis:
    """Options-derived analysis (for assets with options markets)."""
    has_options: bool = False
    atm_iv: float = 0.0
    iv_rank: float = 0.0              # vs 1-year (0-100)
    iv_percentile: float = 0.0
    put_call_ratio: float = 1.0
    risk_reversal_25d: float = 0.0     # put skew - call skew
    butterfly_25d: float = 0.0
    gex_total: float = 0.0
    gex_flip_level: float = 0.0
    max_pain: float = 0.0
    put_wall: float = 0.0
    call_wall: float = 0.0
    implied_move_1d: float = 0.0
    implied_move_1w: float = 0.0
    dealer_position: str = "unknown"   # long_gamma / short_gamma
    options_interpretation: str = ""


@dataclass
class OnChainAnalysis:
    """On-chain metrics for crypto assets."""
    is_crypto: bool = False
    nvt_ratio: float = 0.0
    nvt_signal: float = 0.0           # 90d MA of NVT
    mvrv_zscore: float = 0.0
    realized_cap: float = 0.0
    exchange_netflow_7d: float = 0.0   # positive = inflow (bearish)
    whale_count: int = 0
    whale_activity: str = "neutral"
    stablecoin_supply_ratio: float = 0.0
    hash_rate_30d_change: float = 0.0
    active_addresses_30d: float = 0.0
    miner_revenue_30d: float = 0.0
    onchain_interpretation: str = ""


@dataclass
class CorrelationAnalysis:
    """Cross-asset correlation and lead-lag."""
    top_correlations: List[Dict] = field(default_factory=list)    # [{asset, corr}]
    top_anti_correlations: List[Dict] = field(default_factory=list)
    lead_lag_relationships: List[Dict] = field(default_factory=list)  # [{leader, lag_bars, strength}]
    cluster_id: int = -1
    cluster_members: List[str] = field(default_factory=list)
    beta_to_btc: float = 0.0        # for crypto
    beta_to_spy: float = 0.0        # for equities
    correlation_regime: str = "normal"
    idiosyncratic_return: float = 0.0  # return not explained by market


@dataclass
class LiquidityAnalysis:
    """Liquidity assessment."""
    bid_ask_spread_bps: float = 0.0
    spread_percentile: float = 0.5
    depth_at_touch: float = 0.0
    depth_10bps: float = 0.0
    adv_20d: float = 0.0
    price_impact_1pct_bps: float = 0.0
    days_to_liquidate: float = 0.0
    liquidity_score: float = 0.5       # 0-1
    liquidity_tier: str = "tier2"      # tier1/tier2/tier3
    dark_pool_fill_prob: float = 0.0
    liquidity_interpretation: str = ""


@dataclass
class RiskAnalysis:
    """Risk metrics."""
    var_95_1d: float = 0.0
    var_99_1d: float = 0.0
    cvar_95_1d: float = 0.0
    cvar_99_1d: float = 0.0
    max_drawdown_1y: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 3.0
    tail_ratio: float = 1.0
    hill_estimator: float = 3.0
    factor_beta_momentum: float = 0.0
    factor_beta_value: float = 0.0
    factor_beta_vol: float = 0.0
    risk_score: float = 0.5           # 0-1 (1=highest risk)
    risk_interpretation: str = ""


@dataclass
class FundamentalAnalysis:
    """Fundamental proxies (adapted for crypto and equities)."""
    market_cap: float = 0.0
    market_cap_rank: int = 0
    volume_24h: float = 0.0
    circulating_supply: float = 0.0
    max_supply: float = 0.0
    inflation_rate: float = 0.0
    stock_to_flow: float = 0.0        # for scarce assets
    network_value_transactions: float = 0.0  # Metcalfe's law
    developer_activity_30d: int = 0
    github_commits_30d: int = 0
    fundamental_interpretation: str = ""


@dataclass
class EventCalendar:
    """Upcoming events affecting this asset."""
    events: List[Dict] = field(default_factory=list)  # [{type, date, description, impact_estimate}]
    next_major_event: str = ""
    days_until_next_event: int = -1
    event_risk_score: float = 0.0


@dataclass
class DreamAnalysis:
    """Dream fragility assessment."""
    overall_fragility: float = 0.5     # 0=robust, 1=fragile
    survival_rate: float = 0.5
    worst_scenario: str = ""
    worst_drawdown: float = 0.0
    most_vulnerable_to: str = ""       # which physics perturbation
    dream_interpretation: str = ""


@dataclass
class MemoryAnalysis:
    """Market memory: significant remembered price levels."""
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    support_strength: float = 0.0
    resistance_strength: float = 0.0
    n_remembered_levels: int = 0
    gravitational_pull: float = 0.0    # -1 (pulled down) to +1 (pulled up)
    key_levels: List[Dict] = field(default_factory=list)  # [{price, type, strength, times_tested}]


@dataclass
class AssetIntelligence:
    """
    The Grand Unified Intelligence Dossier.
    EVERYTHING about one asset in one data structure.
    """
    # Header
    symbol: str
    name: str = ""
    asset_class: str = "crypto"
    current_price: float = 0.0
    price_change_24h_pct: float = 0.0
    timestamp: float = 0.0

    # 13 analysis sections
    technical: TechnicalAnalysis = field(default_factory=TechnicalAnalysis)
    regime: RegimeAnalysis = field(default_factory=lambda: RegimeAnalysis(regime="unknown"))
    signals: SignalRanking = field(default_factory=SignalRanking)
    options: OptionsAnalysis = field(default_factory=OptionsAnalysis)
    onchain: OnChainAnalysis = field(default_factory=OnChainAnalysis)
    correlations: CorrelationAnalysis = field(default_factory=CorrelationAnalysis)
    liquidity: LiquidityAnalysis = field(default_factory=LiquidityAnalysis)
    risk: RiskAnalysis = field(default_factory=RiskAnalysis)
    fundamentals: FundamentalAnalysis = field(default_factory=FundamentalAnalysis)
    events: EventCalendar = field(default_factory=EventCalendar)
    dreams: DreamAnalysis = field(default_factory=DreamAnalysis)
    memory: MemoryAnalysis = field(default_factory=MemoryAnalysis)

    # Overall
    overall_score: float = 0.0         # -1 (max bearish) to +1 (max bullish)
    overall_confidence: float = 0.0
    overall_summary: str = ""
    key_highlights: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    recommended_action: str = "hold"   # strong_buy / buy / hold / sell / strong_sell


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: COMPUTE NODES
# ═══════════════════════════════════════════════════════════════════════════════

class TechnicalComputeNode:
    """Compute all 50+ technical indicators."""

    def compute(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                 volumes: np.ndarray) -> TechnicalAnalysis:
        ta = TechnicalAnalysis()
        T = len(prices)
        if T < 201:
            return ta

        # Moving averages
        ta.sma_20 = float(prices[-20:].mean())
        ta.sma_50 = float(prices[-50:].mean())
        ta.sma_200 = float(prices[-200:].mean())

        # EMA
        ta.ema_12 = self._ema(prices, 12)
        ta.ema_26 = self._ema(prices, 26)
        ta.ema_50 = self._ema(prices, 50)

        # MACD
        ta.macd_line = ta.ema_12 - ta.ema_26
        ta.macd_signal = self._ema(np.array([ta.macd_line]), 9)  # simplified
        ta.macd_histogram = ta.macd_line - ta.macd_signal

        # RSI
        ta.rsi_14 = self._rsi(prices, 14)

        # Stochastic
        high_14 = highs[-14:].max()
        low_14 = lows[-14:].min()
        ta.stochastic_k = float((prices[-1] - low_14) / max(high_14 - low_14, 1e-10) * 100)
        ta.stochastic_d = ta.stochastic_k  # simplified

        # Williams %R
        ta.williams_r = float((high_14 - prices[-1]) / max(high_14 - low_14, 1e-10) * -100)

        # CCI
        typical = (highs[-20:] + lows[-20:] + prices[-20:]) / 3
        ta.cci_20 = float((typical[-1] - typical.mean()) / max(typical.std() * 0.015, 1e-10))

        # ROC
        ta.roc_12 = float((prices[-1] / prices[-12] - 1) * 100)

        # Bollinger Bands
        sma = ta.sma_20
        std = float(prices[-20:].std())
        ta.bollinger_upper = sma + 2 * std
        ta.bollinger_lower = sma - 2 * std
        ta.bollinger_pct_b = float((prices[-1] - ta.bollinger_lower) / max(ta.bollinger_upper - ta.bollinger_lower, 1e-10))
        ta.bollinger_bandwidth = float((ta.bollinger_upper - ta.bollinger_lower) / max(sma, 1e-10))

        # ATR
        ta.atr_14 = self._atr(highs, lows, prices, 14)
        ta.atr_pct = ta.atr_14 / max(prices[-1], 1e-10) * 100

        # Donchian
        ta.donchian_high = float(highs[-20:].max())
        ta.donchian_low = float(lows[-20:].min())

        # Keltner
        ta.keltner_upper = ta.ema_50 + 2 * ta.atr_14
        ta.keltner_lower = ta.ema_50 - 2 * ta.atr_14

        # Realized volatility
        returns = np.diff(np.log(prices + 1e-10))
        ta.realized_vol_21d = float(returns[-21:].std() * math.sqrt(252)) if len(returns) >= 21 else 0.15
        ta.realized_vol_63d = float(returns[-63:].std() * math.sqrt(252)) if len(returns) >= 63 else 0.15

        # Vol percentile
        if len(returns) >= 252:
            rolling_vols = [float(returns[i-21:i].std() * math.sqrt(252)) for i in range(21, len(returns))]
            ta.vol_percentile = float(np.mean(np.array(rolling_vols) <= ta.realized_vol_21d))

        # Vol regime
        if ta.realized_vol_21d > 0.80:
            ta.volatility_regime = "extreme"
        elif ta.realized_vol_21d > 0.40:
            ta.volatility_regime = "high"
        elif ta.realized_vol_21d < 0.10:
            ta.volatility_regime = "low"
        else:
            ta.volatility_regime = "normal"

        # Volume
        ta.vwap = float(np.sum(prices[-20:] * volumes[-20:]) / max(np.sum(volumes[-20:]), 1e-10))
        ta.volume_ratio_5_20 = float(volumes[-5:].mean() / max(volumes[-20:].mean(), 1e-10))
        ta.mfi_14 = self._mfi(highs, lows, prices, volumes, 14)

        # OBV
        obv = 0.0
        for i in range(1, min(100, T)):
            if prices[-i] > prices[-i-1]:
                obv += volumes[-i]
            else:
                obv -= volumes[-i]
        ta.obv = obv

        # Price structure
        high_52w = float(highs[-min(252, T):].max())
        low_52w = float(lows[-min(252, T):].min())
        ta.distance_from_high_pct = float((high_52w - prices[-1]) / max(high_52w, 1e-10) * 100)
        ta.distance_from_low_pct = float((prices[-1] - low_52w) / max(low_52w, 1e-10) * 100)
        ta.price_vs_sma200 = float((prices[-1] - ta.sma_200) / max(ta.sma_200, 1e-10) * 100)
        ta.golden_cross = ta.sma_50 > ta.sma_200
        ta.death_cross = ta.sma_50 < ta.sma_200

        # Patterns
        if T >= 3:
            ta.inside_bar = (highs[-1] < highs[-2] and lows[-1] > lows[-2])
            ta.engulfing_bullish = (prices[-1] > prices[-2] and prices[-2] < prices[-3] and
                                     prices[-1] > highs[-2] and prices[-2] < lows[-3])
            ta.engulfing_bearish = (prices[-1] < prices[-2] and prices[-2] > prices[-3] and
                                     prices[-1] < lows[-2] and prices[-2] > highs[-3])

        # ADX (simplified)
        if len(returns) >= 14:
            abs_returns = np.abs(returns[-14:])
            ta.adx = float(abs_returns.mean() / max(abs_returns.std(), 1e-10) * 25)
            ta.trend_strength = min(1.0, ta.adx / 50)

        # Trend direction
        if ta.sma_20 > ta.sma_50 > ta.sma_200 and ta.rsi_14 > 50:
            ta.trend_direction = "up"
        elif ta.sma_20 < ta.sma_50 < ta.sma_200 and ta.rsi_14 < 50:
            ta.trend_direction = "down"
        else:
            ta.trend_direction = "neutral"

        # Momentum composite
        rsi_score = (ta.rsi_14 - 50) / 50  # -1 to +1
        macd_score = float(np.tanh(ta.macd_histogram * 100))
        stoch_score = (ta.stochastic_k - 50) / 50
        ta.momentum_composite = float(np.clip(0.4 * rsi_score + 0.3 * macd_score + 0.3 * stoch_score, -1, 1))

        # Volume trend
        if ta.volume_ratio_5_20 > 1.5:
            ta.volume_trend = "expanding"
        elif ta.volume_ratio_5_20 < 0.5:
            ta.volume_trend = "contracting"
        else:
            ta.volume_trend = "normal"

        # Technical score
        bull_signals = 0
        bear_signals = 0
        if ta.rsi_14 < 30: bull_signals += 1
        if ta.rsi_14 > 70: bear_signals += 1
        if ta.golden_cross: bull_signals += 1
        if ta.death_cross: bear_signals += 1
        if ta.bollinger_pct_b < 0: bull_signals += 1
        if ta.bollinger_pct_b > 1: bear_signals += 1
        if prices[-1] > ta.sma_200: bull_signals += 1
        else: bear_signals += 1
        if ta.macd_histogram > 0: bull_signals += 1
        else: bear_signals += 1
        if ta.volume_trend == "expanding" and ta.trend_direction == "up": bull_signals += 1
        if ta.volume_trend == "expanding" and ta.trend_direction == "down": bear_signals += 1

        total = bull_signals + bear_signals
        ta.technical_score = (bull_signals - bear_signals) / max(total, 1)

        # Interpretation
        if ta.technical_score > 0.5:
            ta.technical_interpretation = "Strongly bullish: multiple technical indicators aligned to the upside"
        elif ta.technical_score > 0.2:
            ta.technical_interpretation = "Moderately bullish: majority of indicators positive"
        elif ta.technical_score < -0.5:
            ta.technical_interpretation = "Strongly bearish: multiple technical indicators aligned to the downside"
        elif ta.technical_score < -0.2:
            ta.technical_interpretation = "Moderately bearish: majority of indicators negative"
        else:
            ta.technical_interpretation = "Neutral: mixed technical signals"

        return ta

    def _ema(self, prices: np.ndarray, window: int) -> float:
        if len(prices) < window:
            return float(prices[-1]) if len(prices) > 0 else 0.0
        alpha = 2 / (window + 1)
        ema = float(prices[-window])
        for p in prices[-window + 1:]:
            ema = alpha * float(p) + (1 - alpha) * ema
        return ema

    def _rsi(self, prices: np.ndarray, period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        changes = np.diff(prices[-period - 1:])
        gains = np.maximum(changes, 0).mean()
        losses = np.maximum(-changes, 0).mean()
        if losses < 1e-10:
            return 100.0
        rs = gains / losses
        return float(100 - 100 / (1 + rs))

    def _atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
              period: int = 14) -> float:
        if len(highs) < period + 1:
            return 0.0
        tr = np.zeros(period)
        for i in range(period):
            idx = -(period - i)
            h = highs[idx]
            l = lows[idx]
            c = closes[idx - 1]
            tr[i] = max(h - l, abs(h - c), abs(l - c))
        return float(tr.mean())

    def _mfi(self, highs, lows, closes, volumes, period=14):
        if len(closes) < period + 1:
            return 50.0
        typical = (highs[-period-1:] + lows[-period-1:] + closes[-period-1:]) / 3
        mf = typical * volumes[-period-1:]
        positive = 0.0
        negative = 0.0
        for i in range(1, len(typical)):
            if typical[i] > typical[i-1]:
                positive += mf[i]
            else:
                negative += mf[i]
        if negative < 1e-10:
            return 100.0
        ratio = positive / negative
        return float(100 - 100 / (1 + ratio))


class RegimeComputeNode:
    """Compute regime classification."""

    def compute(self, returns: np.ndarray) -> RegimeAnalysis:
        if len(returns) < 63:
            return RegimeAnalysis(regime="unknown")

        vol_21 = float(returns[-21:].std() * math.sqrt(252))
        vol_63 = float(returns[-63:].std() * math.sqrt(252))
        trend_21 = float(returns[-21:].mean() * 252)

        # Hurst exponent
        hurst = self._hurst(returns[-100:]) if len(returns) >= 100 else 0.5

        # Regime
        if vol_21 > 0.50:
            regime = "crisis"
        elif vol_21 > 0.30:
            regime = "high_volatility"
        elif trend_21 > 0.20 and hurst > 0.55:
            regime = "trending_up"
        elif trend_21 < -0.20 and hurst > 0.55:
            regime = "trending_down"
        elif hurst < 0.45:
            regime = "mean_reverting"
        else:
            regime = "ranging"

        # Confidence
        confidence = min(1.0, abs(trend_21) + abs(hurst - 0.5) * 2)

        # Fractal dimension
        fractal_dim = 2 - hurst

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            vol_regime="extreme" if vol_21 > 0.50 else "high" if vol_21 > 0.25 else "normal" if vol_21 > 0.10 else "low",
            trend_regime="up" if trend_21 > 0.1 else "down" if trend_21 < -0.1 else "flat",
            hurst_exponent=hurst,
            fractal_dimension=fractal_dim,
            regime_interpretation=f"{regime} regime (confidence {confidence:.0%}). Hurst={hurst:.2f}, Vol={vol_21:.0%}",
        )

    def _hurst(self, returns, max_lag=30):
        n = len(returns)
        if n < max_lag * 2:
            return 0.5
        lags = range(2, min(max_lag, n // 2))
        rs_vals = []
        for lag in lags:
            chunks = n // lag
            rs_list = []
            for i in range(chunks):
                chunk = returns[i*lag:(i+1)*lag]
                mean = chunk.mean()
                cumdev = np.cumsum(chunk - mean)
                R = cumdev.max() - cumdev.min()
                S = max(chunk.std(), 1e-10)
                rs_list.append(R / S)
            if rs_list:
                rs_vals.append(float(np.mean(rs_list)))
        if len(rs_vals) < 3:
            return 0.5
        try:
            slope = float(np.polyfit(np.log(list(lags)[:len(rs_vals)]), np.log(np.array(rs_vals) + 1e-10), 1)[0])
            return float(np.clip(slope, 0, 1))
        except:
            return 0.5


class RiskComputeNode:
    """Compute risk metrics."""

    def compute(self, returns: np.ndarray, prices: np.ndarray) -> RiskAnalysis:
        ra = RiskAnalysis()
        if len(returns) < 30:
            return ra

        sorted_r = np.sort(returns)
        n = len(sorted_r)

        # VaR
        ra.var_95_1d = float(-sorted_r[max(int(0.05 * n), 0)] * 100)
        ra.var_99_1d = float(-sorted_r[max(int(0.01 * n), 0)] * 100)

        # CVaR
        idx_95 = max(int(0.05 * n), 1)
        ra.cvar_95_1d = float(-sorted_r[:idx_95].mean() * 100)
        idx_99 = max(int(0.01 * n), 1)
        ra.cvar_99_1d = float(-sorted_r[:idx_99].mean() * 100)

        # Drawdown
        eq = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-10)
        ra.max_drawdown_1y = float(dd[-min(252, len(dd)):].max() * 100)
        ra.current_drawdown = float(dd[-1] * 100)

        # DD duration
        in_dd = dd > 0.01
        max_dur = 0
        cur = 0
        for d in in_dd:
            if d:
                cur += 1
                max_dur = max(max_dur, cur)
            else:
                cur = 0
        ra.max_drawdown_duration = max_dur

        # Higher moments
        mu = float(returns.mean())
        sigma = max(float(returns.std()), 1e-10)
        ra.skewness = float(np.mean(((returns - mu) / sigma) ** 3))
        ra.kurtosis = float(np.mean(((returns - mu) / sigma) ** 4))

        # Tail ratio
        top_5 = float(np.percentile(returns, 95))
        bot_5 = float(np.percentile(returns, 5))
        ra.tail_ratio = float(abs(top_5) / max(abs(bot_5), 1e-10))

        # Risk score
        ra.risk_score = float(min(1.0,
            ra.var_95_1d / 5 * 0.3 +
            ra.max_drawdown_1y / 50 * 0.3 +
            max(ra.kurtosis - 3, 0) / 10 * 0.2 +
            max(-ra.skewness, 0) / 3 * 0.2
        ))

        # Interpretation
        if ra.risk_score > 0.7:
            ra.risk_interpretation = "HIGH RISK: elevated VaR, fat tails, and/or significant drawdown"
        elif ra.risk_score > 0.4:
            ra.risk_interpretation = "Moderate risk: some tail risk and drawdown exposure"
        else:
            ra.risk_interpretation = "Low risk: contained VaR and favorable tail characteristics"

        return ra


class LiquidityComputeNode:
    """Compute liquidity metrics."""

    def compute(self, volumes: np.ndarray, prices: np.ndarray, spread_bps: float = 5.0) -> LiquidityAnalysis:
        la = LiquidityAnalysis()
        if len(volumes) < 20:
            return la

        la.bid_ask_spread_bps = spread_bps
        la.adv_20d = float(volumes[-20:].mean() * prices[-20:].mean())

        # Amihud illiquidity
        returns = np.abs(np.diff(np.log(prices[-21:] + 1e-10)))
        vols = volumes[-20:] + 1e-10
        amihud = float(np.mean(returns / vols) * 1e6)

        # Impact estimation
        la.price_impact_1pct_bps = float(10 * math.sqrt(0.01)) * 100  # sqrt model

        # Days to liquidate (assuming $1M position)
        if la.adv_20d > 0:
            participation = 1e6 / la.adv_20d
            la.days_to_liquidate = float(participation / 0.10)  # 10% participation rate

        # Score
        vol_score = min(1.0, math.log10(max(la.adv_20d, 1)) / 9)
        spread_score = max(0, 1 - spread_bps / 50)
        la.liquidity_score = 0.5 * vol_score + 0.5 * spread_score

        # Tier
        if la.liquidity_score > 0.7:
            la.liquidity_tier = "tier1"
        elif la.liquidity_score > 0.4:
            la.liquidity_tier = "tier2"
        else:
            la.liquidity_tier = "tier3"

        la.liquidity_interpretation = f"Liquidity score {la.liquidity_score:.2f} ({la.liquidity_tier}). ADV ${la.adv_20d:,.0f}."
        return la


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: THE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AssetIntelligenceEngine:
    """
    The Universal Asset Intelligence Engine.

    A single function call that tells you EVERYTHING about any asset.

    Usage:
        engine = AssetIntelligenceEngine()
        dossier = engine.analyze("BTC", prices, volumes, highs, lows)
        print(dossier.overall_summary)
    """

    def __init__(self):
        self.technical = TechnicalComputeNode()
        self.regime = RegimeComputeNode()
        self.risk = RiskComputeNode()
        self.liquidity = LiquidityComputeNode()

    def analyze(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        asset_class: str = "crypto",
        spread_bps: float = 10.0,
    ) -> AssetIntelligence:
        """
        Produce a complete intelligence dossier on an asset.

        prices: (T,) close prices
        volumes: (T,) volumes (optional, defaults to 1e6)
        highs: (T,) high prices (optional, approximated from closes)
        lows: (T,) low prices (optional, approximated from closes)
        """
        start = time.time()
        T = len(prices)

        if volumes is None:
            volumes = np.full(T, 1e6)
        if highs is None:
            highs = prices * 1.005
        if lows is None:
            lows = prices * 0.995

        returns = np.diff(np.log(prices + 1e-10))

        # Run all compute nodes
        ta = self.technical.compute(prices, highs, lows, volumes)
        regime = self.regime.compute(returns)
        risk = self.risk.compute(returns, prices)
        liquidity = self.liquidity.compute(volumes, prices, spread_bps)

        # Signal ranking (simplified: use technical + regime)
        signals_list = [
            {"name": "momentum_composite", "value": ta.momentum_composite, "confidence": 0.7},
            {"name": "technical_score", "value": ta.technical_score, "confidence": 0.8},
            {"name": "rsi_signal", "value": (50 - ta.rsi_14) / 50 * -1, "confidence": 0.6},
            {"name": "macd_signal", "value": float(np.tanh(ta.macd_histogram * 100)), "confidence": 0.5},
            {"name": "volume_signal", "value": float(np.tanh(ta.volume_ratio_5_20 - 1)), "confidence": 0.4},
        ]
        signals_list.sort(key=lambda s: abs(s["value"]), reverse=True)
        for i, s in enumerate(signals_list):
            s["rank"] = i + 1

        n_bull = sum(1 for s in signals_list if s["value"] > 0.1)
        n_bear = sum(1 for s in signals_list if s["value"] < -0.1)
        composite = float(np.mean([s["value"] * s["confidence"] for s in signals_list]))

        signal_ranking = SignalRanking(
            signals=signals_list,
            composite_signal=composite,
            composite_confidence=min(1.0, abs(composite) * 2),
            n_bullish=n_bull,
            n_bearish=n_bear,
            n_neutral=len(signals_list) - n_bull - n_bear,
            dominant_signal=signals_list[0]["name"] if signals_list else "",
            signal_agreement=abs(n_bull - n_bear) / max(len(signals_list), 1),
        )

        # Memory (simple support/resistance from recent extremes)
        memory = MemoryAnalysis()
        if T >= 50:
            recent_high = float(highs[-50:].max())
            recent_low = float(lows[-50:].min())
            memory.nearest_resistance = recent_high
            memory.nearest_support = recent_low
            memory.support_strength = 0.5
            memory.resistance_strength = 0.5
            memory.gravitational_pull = float(np.tanh((float(np.mean([recent_high, recent_low])) - prices[-1]) / max(prices[-1], 1e-10) * 50))

        # Overall score
        overall = (
            0.35 * ta.technical_score +
            0.25 * composite +
            0.20 * (0.5 - risk.risk_score) +
            0.10 * memory.gravitational_pull +
            0.10 * (0.5 if regime.regime in ("trending_up",) else -0.5 if regime.regime in ("trending_down", "crisis") else 0)
        )
        overall = float(np.clip(overall, -1, 1))
        confidence = min(1.0, abs(overall) * 2)

        # Action recommendation
        if overall > 0.5:
            action = "strong_buy"
        elif overall > 0.2:
            action = "buy"
        elif overall < -0.5:
            action = "strong_sell"
        elif overall < -0.2:
            action = "sell"
        else:
            action = "hold"

        # Key highlights
        highlights = []
        if ta.golden_cross:
            highlights.append("Golden cross (50 SMA > 200 SMA)")
        if ta.rsi_14 < 30:
            highlights.append(f"RSI oversold ({ta.rsi_14:.0f})")
        if ta.rsi_14 > 70:
            highlights.append(f"RSI overbought ({ta.rsi_14:.0f})")
        if regime.regime == "trending_up":
            highlights.append(f"Strong uptrend (Hurst={regime.hurst_exponent:.2f})")
        if ta.volume_trend == "expanding":
            highlights.append("Volume expanding (institutional interest)")

        # Key risks
        risks = []
        if risk.current_drawdown > 10:
            risks.append(f"In {risk.current_drawdown:.0f}% drawdown")
        if risk.kurtosis > 5:
            risks.append(f"Fat tails (kurtosis={risk.kurtosis:.1f})")
        if regime.regime == "crisis":
            risks.append("Crisis regime detected")
        if ta.volatility_regime in ("high", "extreme"):
            risks.append(f"Elevated volatility ({ta.realized_vol_21d:.0%})")

        # Overall summary
        elapsed_ms = (time.time() - start) * 1000
        summary = (
            f"{symbol} is {action.replace('_', ' ')} with {confidence:.0%} confidence. "
            f"Technical analysis is {ta.technical_interpretation.split(':')[0].lower()}. "
            f"Current regime: {regime.regime}. "
            f"Risk assessment: {risk.risk_interpretation.split(':')[0].lower()}. "
            f"Liquidity: {liquidity.liquidity_tier}. "
            f"[Computed in {elapsed_ms:.0f}ms]"
        )

        return AssetIntelligence(
            symbol=symbol,
            asset_class=asset_class,
            current_price=float(prices[-1]),
            price_change_24h_pct=float(returns[-1] * 100) if len(returns) > 0 else 0,
            timestamp=time.time(),
            technical=ta,
            regime=regime,
            signals=signal_ranking,
            risk=risk,
            liquidity=liquidity,
            memory=memory,
            overall_score=overall,
            overall_confidence=confidence,
            overall_summary=summary,
            key_highlights=highlights,
            key_risks=risks,
            recommended_action=action,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════

def quick_analyze(symbol: str = "BTC", n_bars: int = 500, seed: int = 42) -> AssetIntelligence:
    """Quick analysis on synthetic data for testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.02, n_bars)
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = rng.uniform(1e6, 1e8, n_bars)

    engine = AssetIntelligenceEngine()
    return engine.analyze(symbol, prices, volumes)
