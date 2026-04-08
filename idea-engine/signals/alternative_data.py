"""
Alternative data processing for trading signals.

Handles:
  - Social sentiment scoring (news/social headline analysis)
  - Options flow interpretation (unusual activity, put/call skew)
  - Dark pool / block trade signals
  - Google Trends proxy signals
  - Fear & Greed index construction
  - Whale alert parsing
  - Derivatives market structure signals
  - Insider transaction proxies
  - Short interest signals
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Sentiment Scoring ─────────────────────────────────────────────────────────

@dataclass
class SentimentObservation:
    timestamp: float
    source: str          # twitter, reddit, news, telegram
    score: float         # -1 to +1
    volume: float        # number of mentions/articles
    uncertainty: float   # 0-1 how noisy


def sentiment_composite(
    observations: list[SentimentObservation],
    decay_halflife: float = 24.0,   # hours
    current_time: Optional[float] = None,
) -> dict:
    """
    Compute time-decayed composite sentiment score.
    Weights: volume * recency * source_reliability.
    """
    import time as _time
    if current_time is None:
        current_time = _time.time()

    SOURCE_RELIABILITY = {
        "news": 0.8, "twitter": 0.5, "reddit": 0.45,
        "telegram": 0.4, "bloomberg": 0.95, "unknown": 0.3,
    }

    if not observations:
        return {"score": 0.0, "volume": 0.0, "confidence": 0.0}

    total_weight = 0.0
    weighted_score = 0.0
    total_volume = 0.0

    decay_rate = math.log(2) / (decay_halflife * 3600)  # in seconds

    for obs in observations:
        age = max(current_time - obs.timestamp, 0)
        recency = math.exp(-decay_rate * age)
        reliability = SOURCE_RELIABILITY.get(obs.source, 0.4)
        certainty = 1 - obs.uncertainty
        weight = obs.volume * recency * reliability * certainty
        weighted_score += obs.score * weight
        total_weight += weight
        total_volume += obs.volume * recency

    composite = weighted_score / max(total_weight, 1e-10)
    confidence = min(math.log1p(total_weight) / 10.0, 1.0)

    return {
        "score": float(composite),
        "volume_decayed": float(total_volume),
        "confidence": float(confidence),
        "bullish_frac": float(sum(1 for o in observations if o.score > 0.2) / len(observations)),
        "bearish_frac": float(sum(1 for o in observations if o.score < -0.2) / len(observations)),
        "sentiment_regime": (
            "euphoria" if composite > 0.7
            else "greed" if composite > 0.3
            else "fear" if composite < -0.3
            else "panic" if composite < -0.7
            else "neutral"
        ),
    }


def sentiment_divergence_signal(
    sentiment_series: np.ndarray,
    price_series: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Detect sentiment-price divergence: sentiment rising while price falling (bullish)
    or sentiment falling while price rising (bearish reversal).
    Returns signal array: +1 (bullish divergence), -1 (bearish), 0 (no divergence).
    """
    n = min(len(sentiment_series), len(price_series))
    signal = np.zeros(n)

    for i in range(window, n):
        sent_change = float(np.polyfit(np.arange(window), sentiment_series[i-window:i], 1)[0])
        price_change = float(np.polyfit(np.arange(window), price_series[i-window:i], 1)[0])

        # Normalize
        sent_trend = sent_change / (abs(sentiment_series[i-window:i]).mean() + 1e-10)
        price_trend = price_change / (abs(price_series[i-window:i]).mean() + 1e-10)

        # Divergence when they point opposite directions
        if sent_trend > 0.1 and price_trend < -0.1:
            signal[i] = 1.0  # bullish divergence
        elif sent_trend < -0.1 and price_trend > 0.1:
            signal[i] = -1.0  # bearish divergence

    return signal


# ── Options Flow ──────────────────────────────────────────────────────────────

@dataclass
class OptionsFlow:
    strike: float
    expiry_days: int
    call_volume: float
    put_volume: float
    call_oi: float
    put_oi: float
    iv_call: float
    iv_put: float
    spot: float


def put_call_ratio_signal(flows: list[OptionsFlow]) -> dict:
    """
    Put/call ratio and skew signals from options flow.
    High PC ratio (>1.3) → contrarian bullish.
    Low PC ratio (<0.5) → contrarian bearish.
    """
    if not flows:
        return {"pc_ratio": 1.0, "signal": 0.0}

    total_call_vol = sum(f.call_volume for f in flows)
    total_put_vol = sum(f.put_volume for f in flows)
    pc_ratio = total_put_vol / max(total_call_vol, 1.0)

    # Volume-weighted IV skew
    weighted_iv_put = sum(f.iv_put * f.put_volume for f in flows) / max(total_put_vol, 1)
    weighted_iv_call = sum(f.iv_call * f.call_volume for f in flows) / max(total_call_vol, 1)
    iv_skew = weighted_iv_put - weighted_iv_call

    # Term structure: near vs far
    near_flows = [f for f in flows if f.expiry_days <= 30]
    far_flows = [f for f in flows if f.expiry_days > 30]

    near_pc = (sum(f.put_volume for f in near_flows) /
               max(sum(f.call_volume for f in near_flows), 1)) if near_flows else 1.0
    far_pc = (sum(f.put_volume for f in far_flows) /
              max(sum(f.call_volume for f in far_flows), 1)) if far_flows else 1.0

    # Signal: contrarian on extreme PC ratio
    if pc_ratio > 1.5:
        signal = 0.8  # very bullish contrarian
    elif pc_ratio > 1.2:
        signal = 0.4
    elif pc_ratio < 0.5:
        signal = -0.8  # very bearish contrarian
    elif pc_ratio < 0.7:
        signal = -0.4
    else:
        signal = 0.0

    # Adjust for IV skew
    if iv_skew > 0.05:
        signal -= 0.2  # put skew elevated, reduce bullish signal

    return {
        "pc_ratio": float(pc_ratio),
        "iv_skew": float(iv_skew),
        "near_pc": float(near_pc),
        "far_pc": float(far_pc),
        "signal": float(np.clip(signal, -1, 1)),
        "interpretation": (
            "extreme_fear_contrarian_bullish" if pc_ratio > 1.5
            else "elevated_fear_mild_bullish" if pc_ratio > 1.2
            else "extreme_greed_contrarian_bearish" if pc_ratio < 0.5
            else "mild_greed_bearish_tilt" if pc_ratio < 0.7
            else "neutral"
        ),
    }


def unusual_options_activity(
    flows: list[OptionsFlow],
    normal_volume_percentile: float = 80,
) -> list[dict]:
    """
    Detect unusual options activity: large volume relative to OI.
    High volume/OI ratio + out-of-money = speculative positioning.
    """
    alerts = []
    for f in flows:
        call_ratio = f.call_volume / max(f.call_oi, 1)
        put_ratio = f.put_volume / max(f.put_oi, 1)

        moneyness_call = f.spot / f.strike  # >1 = OTM call (above spot)
        moneyness_put = f.strike / f.spot   # >1 = OTM put (below spot)

        if call_ratio > 2.0 and moneyness_call > 1.05:
            alerts.append({
                "type": "unusual_call_buying",
                "strike": f.strike,
                "expiry_days": f.expiry_days,
                "vol_oi_ratio": float(call_ratio),
                "moneyness": float(moneyness_call),
                "direction": "bullish",
                "urgency": "high" if f.expiry_days < 14 else "medium",
            })

        if put_ratio > 2.0 and moneyness_put > 1.05:
            alerts.append({
                "type": "unusual_put_buying",
                "strike": f.strike,
                "expiry_days": f.expiry_days,
                "vol_oi_ratio": float(put_ratio),
                "moneyness": float(moneyness_put),
                "direction": "bearish",
                "urgency": "high" if f.expiry_days < 14 else "medium",
            })

    return sorted(alerts, key=lambda a: a["vol_oi_ratio"], reverse=True)


# ── Fear & Greed Index ────────────────────────────────────────────────────────

def fear_greed_index(
    price_momentum_30d: float,    # return over 30 days
    vol_vs_baseline: float,       # current vol / 90d avg vol
    safe_haven_demand: float,     # 0-1 (bond buying, gold demand)
    market_breadth: float,        # fraction of assets above 50d MA
    junk_bond_demand: float,      # credit spread tightening (0-1)
    put_call_ratio: float = 1.0,
    social_sentiment: float = 0.0,
) -> dict:
    """
    Composite Fear & Greed index (0=extreme fear, 100=extreme greed).
    Based on CNN-style multi-factor index.
    """
    def normalize(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0, 1))

    # Momentum: -0.3 to +0.3 → 0 to 1
    mom_score = normalize(price_momentum_30d, -0.30, 0.30)
    # Volatility: high vol = fear; vol_ratio 0.5 to 3.0
    vol_score = 1.0 - normalize(vol_vs_baseline, 0.5, 3.0)
    # Safe haven: high demand = fear
    safe_haven_score = 1.0 - safe_haven_demand
    # Market breadth: 0 to 1
    breadth_score = market_breadth
    # Junk bonds: tight spreads = greed
    junk_score = junk_bond_demand
    # PC ratio: high = fear (contrarian → fear when high)
    pc_score = 1.0 - normalize(put_call_ratio, 0.4, 2.0)
    # Sentiment: -1 to +1 → 0 to 1
    sent_score = normalize(social_sentiment, -1, 1)

    weights = [0.20, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10]
    components = [mom_score, vol_score, safe_haven_score, breadth_score, junk_score, pc_score, sent_score]
    fg = float(sum(w * c for w, c in zip(weights, components)) * 100)

    return {
        "fear_greed_score": fg,
        "label": (
            "Extreme Fear" if fg < 25
            else "Fear" if fg < 45
            else "Neutral" if fg < 55
            else "Greed" if fg < 75
            else "Extreme Greed"
        ),
        "components": {
            "momentum": float(mom_score * 100),
            "volatility": float(vol_score * 100),
            "safe_haven": float(safe_haven_score * 100),
            "breadth": float(breadth_score * 100),
            "junk_bonds": float(junk_score * 100),
            "put_call": float(pc_score * 100),
            "sentiment": float(sent_score * 100),
        },
        "contrarian_signal": float(
            -1.0 if fg < 20 else 1.0 if fg > 80 else (fg - 50) / 50 * -0.5
        ),
    }


# ── Dark Pool / Block Trade Signal ───────────────────────────────────────────

def dark_pool_signal(
    reported_volume: np.ndarray,
    total_volume: np.ndarray,
    price: np.ndarray,
    window: int = 10,
) -> dict:
    """
    Estimate dark pool activity from reported vs total volume anomalies.
    High dark pool fraction + price movement = informed dark pool trading.
    """
    n = min(len(reported_volume), len(total_volume), len(price))
    dark_frac = 1.0 - reported_volume[:n] / np.maximum(total_volume[:n], 1)
    dark_frac = np.clip(dark_frac, 0, 1)

    returns = np.diff(np.log(price[:n]), prepend=np.log(price[0]))

    # Dark pool signal: excess dark activity with price impact
    dp_signal = np.zeros(n)
    for i in range(window, n):
        df_recent = dark_frac[i - window: i].mean()
        df_baseline = dark_frac.mean()
        excess_dark = df_recent - df_baseline

        price_move = float(returns[i])
        # Dark accumulation + price rising = smart money buying
        dp_signal[i] = excess_dark * np.sign(price_move) * min(abs(price_move) * 50, 1.0)

    return {
        "dark_pool_fraction": dark_frac,
        "dark_pool_signal": dp_signal,
        "current_dark_frac": float(dark_frac[-1]),
        "elevated_dark_activity": bool(dark_frac[-1] > dark_frac.mean() + dark_frac.std()),
        "smart_money_direction": float(dp_signal[-window:].mean()),
    }


# ── Short Interest Signal ─────────────────────────────────────────────────────

def short_interest_signal(
    short_interest: np.ndarray,    # shares short
    float_shares: float,
    avg_daily_volume: np.ndarray,
    price: np.ndarray,
) -> dict:
    """
    Short interest based signals: short squeeze potential + contrarian.
    Days to cover (DTC) = short_interest / avg_daily_volume.
    High DTC + price rising = squeeze setup.
    """
    n = len(short_interest)
    dtc = short_interest / np.maximum(avg_daily_volume[:n], 1)
    short_pct_float = short_interest / max(float_shares, 1)

    returns = np.diff(np.log(price[:n]), prepend=0)

    # Squeeze pressure: high DTC + rising price
    squeeze_pressure = np.zeros(n)
    for i in range(1, n):
        if dtc[i] > 5 and returns[i] > 0.01:
            squeeze_pressure[i] = min(dtc[i] / 10 * returns[i] * 100, 1.0)

    # Short buildup signal (bearish): increasing short interest
    si_trend = np.zeros(n)
    for i in range(5, n):
        si_trend[i] = float(short_interest[i] - short_interest[i-5]) / max(short_interest[i-5], 1)

    return {
        "days_to_cover": dtc,
        "short_pct_float": short_pct_float,
        "squeeze_pressure": squeeze_pressure,
        "short_interest_trend": si_trend,
        "current_dtc": float(dtc[-1]),
        "current_si_pct": float(short_pct_float[-1]),
        "squeeze_risk": bool(dtc[-1] > 10 and returns[-1] > 0.02),
        "short_crowded": bool(short_pct_float[-1] > 0.20),
        "signal": float(squeeze_pressure[-1] - max(si_trend[-1] * 0.5, 0)),
    }


# ── Google Trends Proxy ───────────────────────────────────────────────────────

def search_interest_signal(
    search_index: np.ndarray,   # 0-100 weekly search interest
    price: np.ndarray,
    lag: int = 2,
    window: int = 12,
) -> dict:
    """
    Search interest as leading indicator.
    Rising search interest 2 weeks before price move = retail FOMO incoming.
    Falling interest during price rise = distribution without retail.
    """
    n = min(len(search_index), len(price))
    returns = np.diff(np.log(price[:n]), prepend=0)

    # Lagged correlation: does search interest predict future returns?
    if n > window + lag + 2:
        lagged_corr = float(np.corrcoef(search_index[:n-lag], returns[lag:n])[0, 1])
    else:
        lagged_corr = 0.0

    # Search momentum
    si_recent = search_index[-window:]
    si_trend = float(np.polyfit(np.arange(len(si_recent)), si_recent, 1)[0])

    # Normalization
    si_zscore = float((search_index[-1] - search_index[-window:].mean()) /
                      max(search_index[-window:].std(), 1e-10))

    # Signal: high search interest = retail FOMO → contrarian bearish near peak
    if si_zscore > 2.0 and si_trend > 0:
        signal = -0.6  # peak FOMO, contrarian short
    elif si_zscore > 1.0:
        signal = -0.2
    elif si_zscore < -1.5:
        signal = 0.3  # low interest, early cycle
    else:
        signal = lagged_corr * 0.3

    return {
        "search_interest_zscore": si_zscore,
        "search_trend": float(si_trend),
        "predictive_lag_corr": float(lagged_corr),
        "signal": float(signal),
        "fomo_alert": bool(si_zscore > 2.0),
        "low_interest_opportunity": bool(si_zscore < -1.5),
    }
