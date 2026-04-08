"""
signals/sentiment_composite_signal.py

Multi-source sentiment composite signal generator for the idea engine.

Computes:
  - News sentiment: exponential decay-weighted polarity scores
  - Social media sentiment: volume-weighted, bot-adjusted
  - Analyst revision momentum: EPS estimate revision breadth
  - Short interest changes: change in short % as contrarian signal
  - Insider transactions: buy/sell ratio from SEC Form 4 filings
  - Fund flow signal: ETF flow as retail sentiment proxy
  - Consumer confidence momentum
  - Google Trends spike detection: abnormal search volume as crowding
  - Composite sentiment z-score: normalised blend of all sources
  - Contrarian scoring: extreme sentiment => fade signal

All arrays are numpy-based. Public entry point: compute_sentiment_composite_signal().
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

class SentimentExtreme(str, Enum):
    EXTREME_BULLISH = "extreme_bullish"   # contrarian bearish
    BULLISH         = "bullish"
    NEUTRAL         = "neutral"
    BEARISH         = "bearish"
    EXTREME_BEARISH = "extreme_bearish"   # contrarian bullish


# ---------------------------------------------------------------------------
# Input dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """
    A single news article or headline with pre-computed sentiment.

    Attributes
    ----------
    polarity : float
        NLP sentiment polarity in [-1, +1]. +1 = maximally bullish.
    subjectivity : float
        Subjectivity score in [0, 1]. Higher = more opinionated.
    relevance : float
        Topic relevance score in [0, 1].
    age_hours : float
        How many hours ago the article was published.
    source_weight : float
        Publisher credibility weight (1.0 = baseline, 2.0 = major outlet).
    """
    polarity: float
    subjectivity: float = 0.5
    relevance: float = 1.0
    age_hours: float = 0.0
    source_weight: float = 1.0


@dataclass
class SocialPost:
    """
    A single social media post/tweet with pre-computed sentiment.

    Attributes
    ----------
    polarity : float
        Sentiment polarity in [-1, +1].
    engagement : float
        Normalised engagement (likes + shares + comments).
    bot_probability : float
        Estimated probability [0, 1] that the account is a bot.
    age_hours : float
        Hours since post.
    """
    polarity: float
    engagement: float = 1.0
    bot_probability: float = 0.0
    age_hours: float = 0.0


@dataclass
class AnalystRevision:
    """
    A single EPS estimate revision.

    Attributes
    ----------
    direction : int
        +1 = upward revision, -1 = downward revision.
    magnitude_pct : float
        Size of revision as % of prior estimate.
    days_ago : int
        How many calendar days ago the revision was made.
    """
    direction: int
    magnitude_pct: float
    days_ago: int = 0


@dataclass
class InsiderTransaction:
    """
    A single SEC Form 4 insider transaction.

    Attributes
    ----------
    transaction_type : str
        'buy' or 'sell'.
    value_usd : float
        Dollar value of the transaction.
    days_ago : int
        Calendar days since filing.
    is_plan_trade : bool
        True if part of a 10b5-1 pre-planned trading plan (less informative).
    """
    transaction_type: str   # 'buy' | 'sell'
    value_usd: float
    days_ago: int = 0
    is_plan_trade: bool = False


@dataclass
class SentimentInput:
    """
    Full multi-source sentiment data required by compute_sentiment_composite_signal().

    Attributes
    ----------
    news_articles : list[NewsArticle]
    social_posts : list[SocialPost]
    analyst_revisions : list[AnalystRevision]
    short_interest_series : np.ndarray
        Short interest as % of float, daily, oldest first. Min 10 bars.
    insider_transactions : list[InsiderTransaction]
    etf_flow_series : np.ndarray
        Daily ETF net flows in millions USD (positive = inflow). Oldest first.
    consumer_confidence_series : np.ndarray
        Monthly consumer confidence index readings. Oldest first.
    google_trends_series : np.ndarray
        Normalised Google Trends volume (0-100), daily or weekly. Oldest first.
    news_decay_halflife_hours : float
        Half-life for news sentiment decay. Default 24 hours.
    social_decay_halflife_hours : float
        Half-life for social post decay. Default 6 hours.
    contrarian_zscore_threshold : float
        Z-score magnitude above which composite sentiment triggers contrarian fade.
        Default 1.5.
    """
    news_articles: list[NewsArticle]
    social_posts: list[SocialPost]
    analyst_revisions: list[AnalystRevision]
    short_interest_series: np.ndarray
    insider_transactions: list[InsiderTransaction]
    etf_flow_series: np.ndarray
    consumer_confidence_series: np.ndarray
    google_trends_series: np.ndarray
    news_decay_halflife_hours: float = 24.0
    social_decay_halflife_hours: float = 6.0
    contrarian_zscore_threshold: float = 1.5


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _decay_weights(ages: np.ndarray, halflife: float) -> np.ndarray:
    """Exponential decay weights. w_i = exp(-ln(2) * age_i / halflife)."""
    return np.exp(-math.log(2.0) * ages / (halflife + 1e-12))


def _zscore_scalar(value: float, history: np.ndarray) -> float:
    """Z-score of a single value relative to a historical series."""
    if len(history) < 2:
        return 0.0
    mu  = float(np.mean(history))
    std = float(np.std(history)) + 1e-12
    return (value - mu) / std


def _rolling_zscore(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling z-score of a series."""
    n = len(series)
    out = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = series[start:i + 1]
        mu  = float(np.mean(seg))
        std = float(np.std(seg)) + 1e-12
        out[i] = (series[i] - mu) / std
    return out


def _tanh_normalise(x: float, scale: float = 1.0) -> float:
    return float(np.tanh(x / (scale + 1e-12)))


# ---------------------------------------------------------------------------
# 1. News Sentiment
# ---------------------------------------------------------------------------

def _compute_news_sentiment(
    articles: list[NewsArticle],
    halflife_hours: float = 24.0,
) -> tuple[float, float]:
    """
    Compute decay-weighted news sentiment score.

    Weighting: source_weight * relevance * (1 - subjectivity * 0.5) * decay(age)
    Subjectivity discount: highly opinionated pieces weighted down.

    Returns
    -------
    signal : float in [-1, +1]
    coverage_weight : float — relative data quality (0-1)
    """
    if not articles:
        return 0.0, 0.0

    polarities = np.array([a.polarity for a in articles], dtype=float)
    ages       = np.array([a.age_hours for a in articles], dtype=float)
    subj       = np.array([a.subjectivity for a in articles], dtype=float)
    relevance  = np.array([a.relevance for a in articles], dtype=float)
    src_wt     = np.array([a.source_weight for a in articles], dtype=float)

    decay = _decay_weights(ages, halflife_hours)
    quality = src_wt * relevance * (1.0 - 0.5 * subj)
    combined_wt = decay * quality
    total_wt = combined_wt.sum() + 1e-12

    raw_sentiment = float(np.dot(combined_wt, polarities) / total_wt)
    signal = _clip(raw_sentiment)

    # Coverage weight: how much data we have (normalised, saturates at 100 articles)
    coverage_weight = float(min(len(articles) / 100.0, 1.0))

    return signal, coverage_weight


# ---------------------------------------------------------------------------
# 2. Social Media Sentiment
# ---------------------------------------------------------------------------

def _compute_social_sentiment(
    posts: list[SocialPost],
    halflife_hours: float = 6.0,
    bot_penalty: float = 0.8,
) -> tuple[float, float]:
    """
    Compute volume-weighted, bot-adjusted social media sentiment.

    Bot adjustment: effective_weight = engagement * (1 - bot_prob * bot_penalty)
    Posts with bot_prob > 0.8 are discarded entirely.

    Returns
    -------
    signal : float in [-1, +1]
    coverage_weight : float
    """
    if not posts:
        return 0.0, 0.0

    # Filter obvious bots
    posts = [p for p in posts if p.bot_probability < 0.8]
    if not posts:
        return 0.0, 0.0

    polarities  = np.array([p.polarity for p in posts], dtype=float)
    ages        = np.array([p.age_hours for p in posts], dtype=float)
    engagement  = np.array([p.engagement for p in posts], dtype=float)
    bot_prob    = np.array([p.bot_probability for p in posts], dtype=float)

    decay   = _decay_weights(ages, halflife_hours)
    bot_adj = 1.0 - bot_prob * bot_penalty
    weights = decay * engagement * bot_adj
    total_wt = weights.sum() + 1e-12

    raw = float(np.dot(weights, polarities) / total_wt)
    signal = _clip(raw)
    coverage_weight = float(min(len(posts) / 500.0, 1.0))

    return signal, coverage_weight


# ---------------------------------------------------------------------------
# 3. Analyst Revision Momentum
# ---------------------------------------------------------------------------

def _compute_analyst_revision_signal(
    revisions: list[AnalystRevision],
    halflife_days: float = 30.0,
) -> float:
    """
    Analyst revision breadth and magnitude signal.

    Breadth = (n_up - n_down) / (n_up + n_down)
    Magnitude-weighted version decays by age.

    Returns signal in [-1, +1]:
      +1 = strong upward revision momentum.
    """
    if not revisions:
        return 0.0

    directions = np.array([float(r.direction) for r in revisions], dtype=float)
    magnitudes = np.array([r.magnitude_pct / 100.0 for r in revisions], dtype=float)
    ages       = np.array([float(r.days_ago) for r in revisions], dtype=float)

    decay = _decay_weights(ages, halflife_days)

    # Magnitude-weighted direction
    mag_wt = (np.abs(magnitudes) + 0.01) * decay  # +0.01 prevents zero-weight
    total_wt = mag_wt.sum() + 1e-12
    breadth = float(np.dot(mag_wt, directions) / total_wt)

    return _clip(breadth * 2.0)  # scale: breadth of ±0.5 => ±1


# ---------------------------------------------------------------------------
# 4. Short Interest (contrarian)
# ---------------------------------------------------------------------------

def _compute_short_interest_signal(
    si_series: np.ndarray,
    lookback: int = 20,
) -> float:
    """
    Short interest change signal (contrarian).

    Increasing short interest => crowded short => potential squeeze => bullish fade.
    Decreasing short interest => short covering already done => no edge.

    Z-score of the short interest level relative to history:
      High SI (z > 1.5) => contrarian bullish (+)
      Low  SI (z < -1.5) => contrarian bearish (-)

    Also factor in rate-of-change: rapid rise in SI is more meaningful.

    Returns signal in [-1, +1].
    """
    if len(si_series) < 3:
        return 0.0

    current_si = float(si_series[-1])
    window = min(lookback, len(si_series))
    hist = si_series[-window:]

    z_level = _zscore_scalar(current_si, hist)
    # Contrarian: high SI = potential squeeze = bullish
    level_signal = _tanh_normalise(z_level, scale=1.5)

    # Rate of change: recent change in SI
    if len(si_series) >= 5:
        si_change = (float(si_series[-1]) - float(si_series[-5])) / (abs(float(si_series[-5])) + 1e-9)
        mom_signal = _tanh_normalise(si_change, scale=0.10)  # 10 % change = near ±1
    else:
        mom_signal = 0.0

    # Contrarian interpretation: rising SI is bullish (squeeze potential)
    combined = 0.6 * level_signal + 0.4 * mom_signal
    return _clip(combined)


# ---------------------------------------------------------------------------
# 5. Insider Transactions (SEC Form 4)
# ---------------------------------------------------------------------------

def _compute_insider_signal(
    transactions: list[InsiderTransaction],
    halflife_days: float = 45.0,
    plan_trade_discount: float = 0.3,
) -> float:
    """
    Insider buy/sell signal from Form 4 filings.

    Buy/sell ratio by dollar value, decay-weighted by age.
    Pre-planned (10b5-1) trades are discounted.

    Returns signal in [-1, +1]:
      +1 = insiders heavily buying (bullish)
      -1 = insiders heavily selling (bearish)
    """
    if not transactions:
        return 0.0

    buy_value  = 0.0
    sell_value = 0.0

    for t in transactions:
        decay = math.exp(-math.log(2.0) * t.days_ago / (halflife_days + 1e-12))
        plan_adj = plan_trade_discount if t.is_plan_trade else 1.0
        effective = t.value_usd * decay * plan_adj
        if t.transaction_type.lower() == "buy":
            buy_value  += effective
        else:
            sell_value += effective

    total = buy_value + sell_value + 1e-9
    net_ratio = (buy_value - sell_value) / total  # in [-1, +1]
    return _clip(net_ratio * 2.0)  # scale to full range


# ---------------------------------------------------------------------------
# 6. ETF Fund Flow Signal
# ---------------------------------------------------------------------------

def _compute_fund_flow_signal(
    flow_series: np.ndarray,
    window: int = 20,
) -> float:
    """
    ETF fund flow signal as retail sentiment proxy.

    Large inflows => retail FOMO => crowding => mild contrarian negative.
    Large outflows => panic selling => contrarian positive.

    Z-score of recent cumulative flow relative to history.

    Returns signal in [-1, +1]:
      Trend-following sign: flows positive => retail bullish => +signal (mild).
      But with extreme contrarian fade at z > 2.
    """
    if len(flow_series) < 5:
        return 0.0

    window = min(window, len(flow_series))
    recent_cum = float(np.sum(flow_series[-5:]))    # 5-day cumulative
    hist_cum   = np.array([
        float(np.sum(flow_series[max(0, i - 5):i]))
        for i in range(5, len(flow_series) + 1)
    ])

    if len(hist_cum) < 3:
        return 0.0

    z = _zscore_scalar(recent_cum, hist_cum)

    # Mild trend-following for |z| < 1.5; contrarian fade beyond
    if abs(z) > 2.0:
        # Extreme flows => contrarian fade
        signal = -_tanh_normalise(z, scale=1.5)
    else:
        # Moderate: trend-following (inflows = bullish sentiment)
        signal = _tanh_normalise(z, scale=1.5) * 0.5

    return _clip(signal)


# ---------------------------------------------------------------------------
# 7. Consumer Confidence Momentum
# ---------------------------------------------------------------------------

def _compute_consumer_confidence_signal(
    cc_series: np.ndarray,
    level_threshold_high: float = 110.0,
    level_threshold_low: float = 85.0,
) -> float:
    """
    Consumer confidence momentum signal.

    Uses both the level and the month-over-month momentum:
      - Rising from low levels => bullish (recovery signal)
      - Falling from high levels => bearish (deterioration)
      - High level + rising => moderate bullish (expanding but near peak)

    Returns signal in [-1, +1].
    """
    if len(cc_series) < 2:
        return 0.0

    current = float(cc_series[-1])
    prev    = float(cc_series[-2])
    mom     = current - prev

    # Level signal
    if current > level_threshold_high:
        level_sig = 0.3   # high but watch for reversal
    elif current < level_threshold_low:
        level_sig = -0.5  # depressed consumer sentiment
    else:
        # Linear interpolation between thresholds
        level_sig = (current - level_threshold_low) / (level_threshold_high - level_threshold_low) * 0.6 - 0.3

    # Momentum signal
    if len(cc_series) >= 3:
        prev2 = float(cc_series[-3])
        accel = (current - prev) - (prev - prev2)  # second derivative
        mom_sig = _tanh_normalise(mom, scale=5.0)
        accel_sig = _tanh_normalise(accel, scale=3.0)
        momentum_component = 0.7 * mom_sig + 0.3 * accel_sig
    else:
        momentum_component = _tanh_normalise(mom, scale=5.0)

    return _clip(0.4 * level_sig + 0.6 * momentum_component)


# ---------------------------------------------------------------------------
# 8. Google Trends Spike Detection
# ---------------------------------------------------------------------------

def _compute_google_trends_signal(
    trends_series: np.ndarray,
    spike_window: int = 12,
    baseline_window: int = 52,
    spike_threshold_z: float = 2.0,
) -> tuple[float, bool]:
    """
    Detect abnormal Google Trends search volume as a crowding signal.

    Logic:
    - Compute z-score of recent volume relative to baseline.
    - A spike (z > spike_threshold_z) indicates crowding/FOMO => contrarian bearish.
    - Declining after a spike => bearish momentum continuation.
    - Low volume relative to history => under-the-radar = mildly bullish.

    Parameters
    ----------
    spike_window : int
        Recent window (bars) to define "current" volume.
    baseline_window : int
        Historical window (bars) for baseline statistics.

    Returns
    -------
    signal : float in [-1, +1]
    spike_detected : bool
    """
    if len(trends_series) < spike_window + 3:
        return 0.0, False

    baseline_n = min(baseline_window, len(trends_series) - spike_window)
    recent_avg = float(np.mean(trends_series[-spike_window:]))
    baseline   = trends_series[-(spike_window + baseline_n):-spike_window]

    if len(baseline) < 3:
        return 0.0, False

    z = _zscore_scalar(recent_avg, baseline)
    spike_detected = z > spike_threshold_z

    if spike_detected:
        # Strong spike => contrarian fade
        signal = _clip(-z / spike_threshold_z)
    elif z < -1.0:
        # Below-average search = under the radar = mildly bullish
        signal = _clip(-z * 0.3)
    else:
        signal = _tanh_normalise(-z, scale=1.0) * 0.4

    return _clip(signal), spike_detected


# ---------------------------------------------------------------------------
# 9. Composite z-score and contrarian scoring
# ---------------------------------------------------------------------------

def _compute_composite_zscore(
    component_signals: dict[str, float],
    component_weights: dict[str, float],
) -> tuple[float, float]:
    """
    Compute weighted composite sentiment score and its z-score equivalent.

    The composite is compared to a long-run "neutral" baseline of 0.0.
    The z-score is estimated by treating ±0.3 as one standard deviation
    of the composite distribution (empirical heuristic).

    Returns
    -------
    composite : float in [-1, +1]
    composite_z : float — standardised composite
    """
    total_wt = sum(component_weights.values()) + 1e-12
    composite = sum(
        component_signals.get(k, 0.0) * component_weights.get(k, 0.0)
        for k in component_weights
    ) / total_wt

    composite = _clip(composite)
    composite_z = composite / 0.30  # 0.30 = assumed 1-sigma of composite

    return composite, float(composite_z)


def _apply_contrarian_fade(
    raw_signal: float,
    composite_z: float,
    threshold: float = 1.5,
    fade_strength: float = 0.6,
) -> float:
    """
    Apply contrarian fade when composite z-score exceeds threshold.

    At z = threshold, the signal is unchanged.
    Beyond threshold, fade (reverse) it by fade_strength:
      adjusted = raw_signal * (1 - fade_strength * excess_z / threshold)

    Returns adjusted signal in [-1, +1].
    """
    excess = abs(composite_z) - threshold
    if excess <= 0:
        return raw_signal

    # Fade factor: reduces signal magnitude and can flip it
    fade = min(fade_strength * excess / threshold, 1.0)
    adjusted = raw_signal * (1.0 - fade) + (-raw_signal) * fade
    return _clip(adjusted)


# ---------------------------------------------------------------------------
# Classify extreme
# ---------------------------------------------------------------------------

def _classify_sentiment_extreme(composite_z: float) -> SentimentExtreme:
    az = abs(composite_z)
    sign = 1 if composite_z >= 0 else -1
    if az > 2.5:
        return SentimentExtreme.EXTREME_BULLISH if sign > 0 else SentimentExtreme.EXTREME_BEARISH
    elif az > 1.5:
        return SentimentExtreme.BULLISH if sign > 0 else SentimentExtreme.BEARISH
    else:
        return SentimentExtreme.NEUTRAL


# ---------------------------------------------------------------------------
# Component weights
# ---------------------------------------------------------------------------

_COMPONENT_WEIGHTS: dict[str, float] = {
    "news":              0.18,
    "social":            0.12,
    "analyst_revision":  0.16,
    "short_interest":    0.12,
    "insider":           0.14,
    "fund_flow":         0.10,
    "consumer_conf":     0.10,
    "google_trends":     0.08,
}
assert abs(sum(_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sentiment_composite_signal(data: SentimentInput) -> DomainSignal:
    """
    Compute the composite multi-source sentiment signal.

    Parameters
    ----------
    data : SentimentInput

    Returns
    -------
    DomainSignal
        domain='sentiment', value in [-1, +1].
    """
    warnings: list[str] = []

    # --- 1. News ---
    news_signal, news_cov = _compute_news_sentiment(
        data.news_articles, data.news_decay_halflife_hours
    )

    # --- 2. Social ---
    social_signal, social_cov = _compute_social_sentiment(
        data.social_posts, data.social_decay_halflife_hours
    )

    # --- 3. Analyst revisions ---
    revision_signal = _compute_analyst_revision_signal(data.analyst_revisions)

    # --- 4. Short interest (contrarian) ---
    si_signal = _compute_short_interest_signal(data.short_interest_series)

    # --- 5. Insider transactions ---
    insider_signal = _compute_insider_signal(data.insider_transactions)

    # --- 6. Fund flows ---
    flow_signal = _compute_fund_flow_signal(data.etf_flow_series)

    # --- 7. Consumer confidence ---
    cc_signal = _compute_consumer_confidence_signal(data.consumer_confidence_series)

    # --- 8. Google Trends ---
    trends_signal, spike_detected = _compute_google_trends_signal(data.google_trends_series)

    # --- Raw component map ---
    raw_components = {
        "news":             news_signal,
        "social":           social_signal,
        "analyst_revision": revision_signal,
        "short_interest":   si_signal,
        "insider":          insider_signal,
        "fund_flow":        flow_signal,
        "consumer_conf":    cc_signal,
        "google_trends":    trends_signal,
    }

    # --- Composite and z-score ---
    composite_raw, composite_z = _compute_composite_zscore(
        raw_components, _COMPONENT_WEIGHTS
    )

    # --- Contrarian fade on extremes ---
    composite_faded = _apply_contrarian_fade(
        composite_raw,
        composite_z,
        threshold=data.contrarian_zscore_threshold,
        fade_strength=0.6,
    )

    # --- Sentiment regime ---
    sentiment_extreme = _classify_sentiment_extreme(composite_z)

    # --- Conviction ---
    # Higher when components agree; lower when coverage is thin
    values = np.array(list(raw_components.values()))
    agree_frac = float(np.mean(np.sign(values) == np.sign(composite_faded)))

    # Coverage penalty: scale by data quality
    coverage_scores = {
        "news":   news_cov,
        "social": social_cov,
        "si":     float(min(len(data.short_interest_series) / 20.0, 1.0)),
        "cc":     float(min(len(data.consumer_confidence_series) / 12.0, 1.0)),
        "trends": float(min(len(data.google_trends_series) / 52.0, 1.0)),
    }
    avg_coverage = float(np.mean(list(coverage_scores.values())))

    conviction = float(np.clip(0.5 * agree_frac + 0.5 * avg_coverage, 0.0, 1.0))

    # Reduce conviction when sentiment is at an extreme (noise regime)
    if sentiment_extreme in (SentimentExtreme.EXTREME_BULLISH, SentimentExtreme.EXTREME_BEARISH):
        conviction *= 0.7
        warnings.append(
            f"Extreme sentiment detected ({sentiment_extreme.value}); "
            "contrarian fade applied, conviction reduced."
        )

    # --- Data quality warnings ---
    if not data.news_articles:
        warnings.append("No news articles; news sentiment set to zero.")
    if not data.social_posts:
        warnings.append("No social posts; social sentiment set to zero.")
    if not data.analyst_revisions:
        warnings.append("No analyst revisions; revision signal set to zero.")
    if len(data.short_interest_series) < 5:
        warnings.append("Short interest series too short; SI signal may be unreliable.")
    if not data.insider_transactions:
        warnings.append("No insider transaction data; insider signal set to zero.")
    if spike_detected:
        warnings.append("Google Trends spike detected: abnormal search volume, crowding risk.")

    return DomainSignal(
        domain="sentiment",
        value=composite_faded,
        conviction=conviction,
        regime=sentiment_extreme.value,
        components=raw_components,
        metadata={
            "composite_raw":       composite_raw,
            "composite_z":         composite_z,
            "contrarian_fade_applied": abs(composite_z) > data.contrarian_zscore_threshold,
            "spike_detected":      spike_detected,
            "news_coverage":       news_cov,
            "social_coverage":     social_cov,
            "avg_data_coverage":   avg_coverage,
            "coverage_by_source":  coverage_scores,
            "n_news":              len(data.news_articles),
            "n_social":            len(data.social_posts),
            "n_revisions":         len(data.analyst_revisions),
            "n_insiders":          len(data.insider_transactions),
        },
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Convenience builder: compute from raw arrays and scalars
# ---------------------------------------------------------------------------

def compute_sentiment_from_arrays(
    *,
    # News
    news_polarities: np.ndarray,
    news_ages_hours: np.ndarray,
    news_source_weights: Optional[np.ndarray] = None,
    news_relevances: Optional[np.ndarray] = None,
    news_subjectivities: Optional[np.ndarray] = None,
    # Social
    social_polarities: np.ndarray,
    social_ages_hours: np.ndarray,
    social_engagements: Optional[np.ndarray] = None,
    social_bot_probs: Optional[np.ndarray] = None,
    # Analyst revisions
    revision_directions: np.ndarray,        # +1 / -1 array
    revision_magnitudes_pct: np.ndarray,
    revision_ages_days: np.ndarray,
    # Short interest
    short_interest_series: np.ndarray,
    # Insider
    insider_types: list[str],               # 'buy'/'sell'
    insider_values: np.ndarray,
    insider_ages_days: np.ndarray,
    insider_plan_flags: Optional[np.ndarray] = None,
    # Fund flow
    etf_flow_series: np.ndarray,
    # Consumer confidence
    consumer_confidence_series: np.ndarray,
    # Google Trends
    google_trends_series: np.ndarray,
    # Params
    news_halflife_hours: float = 24.0,
    social_halflife_hours: float = 6.0,
    contrarian_threshold: float = 1.5,
) -> DomainSignal:
    """
    Build SentimentInput from raw arrays and compute the composite signal.
    """
    n_news = len(news_polarities)
    articles = []
    for i in range(n_news):
        articles.append(NewsArticle(
            polarity=float(news_polarities[i]),
            subjectivity=float(news_subjectivities[i]) if news_subjectivities is not None else 0.5,
            relevance=float(news_relevances[i]) if news_relevances is not None else 1.0,
            age_hours=float(news_ages_hours[i]),
            source_weight=float(news_source_weights[i]) if news_source_weights is not None else 1.0,
        ))

    n_soc = len(social_polarities)
    posts = []
    for i in range(n_soc):
        posts.append(SocialPost(
            polarity=float(social_polarities[i]),
            engagement=float(social_engagements[i]) if social_engagements is not None else 1.0,
            bot_probability=float(social_bot_probs[i]) if social_bot_probs is not None else 0.0,
            age_hours=float(social_ages_hours[i]),
        ))

    n_rev = len(revision_directions)
    revisions = []
    for i in range(n_rev):
        revisions.append(AnalystRevision(
            direction=int(revision_directions[i]),
            magnitude_pct=float(revision_magnitudes_pct[i]),
            days_ago=int(revision_ages_days[i]),
        ))

    n_ins = len(insider_types)
    insiders = []
    for i in range(n_ins):
        insiders.append(InsiderTransaction(
            transaction_type=str(insider_types[i]),
            value_usd=float(insider_values[i]),
            days_ago=int(insider_ages_days[i]),
            is_plan_trade=bool(insider_plan_flags[i]) if insider_plan_flags is not None else False,
        ))

    inp = SentimentInput(
        news_articles=articles,
        social_posts=posts,
        analyst_revisions=revisions,
        short_interest_series=np.asarray(short_interest_series, dtype=float),
        insider_transactions=insiders,
        etf_flow_series=np.asarray(etf_flow_series, dtype=float),
        consumer_confidence_series=np.asarray(consumer_confidence_series, dtype=float),
        google_trends_series=np.asarray(google_trends_series, dtype=float),
        news_decay_halflife_hours=news_halflife_hours,
        social_decay_halflife_hours=social_halflife_hours,
        contrarian_zscore_threshold=contrarian_threshold,
    )
    return compute_sentiment_composite_signal(inp)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(99)

    # Synthetic news (mildly bullish)
    n_news = 80
    articles = [
        NewsArticle(
            polarity=float(rng.uniform(-0.2, 0.8)),
            subjectivity=float(rng.uniform(0.2, 0.7)),
            relevance=float(rng.uniform(0.5, 1.0)),
            age_hours=float(rng.uniform(0, 48)),
            source_weight=float(rng.choice([0.8, 1.0, 1.5, 2.0])),
        )
        for _ in range(n_news)
    ]

    # Synthetic social (noisier, some bots)
    n_soc = 300
    posts = [
        SocialPost(
            polarity=float(rng.uniform(-0.5, 0.9)),
            engagement=float(rng.exponential(10.0)),
            bot_probability=float(rng.beta(1, 5)),
            age_hours=float(rng.uniform(0, 12)),
        )
        for _ in range(n_soc)
    ]

    # Analyst revisions: mostly up
    revisions = [
        AnalystRevision(
            direction=int(rng.choice([1, 1, 1, -1])),
            magnitude_pct=float(rng.uniform(0.5, 8.0)),
            days_ago=int(rng.integers(0, 60)),
        )
        for _ in range(25)
    ]

    # Short interest (low = not crowded short)
    si = rng.uniform(3.0, 5.0, 30)
    si[-5:] = 4.2  # declining

    # Insiders: net buying
    insiders = [
        InsiderTransaction("buy",  float(rng.uniform(50_000, 500_000)), days_ago=int(rng.integers(0, 90)))
        for _ in range(8)
    ] + [
        InsiderTransaction("sell", float(rng.uniform(20_000, 200_000)), days_ago=int(rng.integers(0, 90)),
                           is_plan_trade=True)
        for _ in range(5)
    ]

    # ETF flows: moderate inflows
    flows = rng.normal(150, 80, 60)

    # Consumer confidence: improving
    cc = np.linspace(88, 102, 18) + rng.normal(0, 2, 18)

    # Google Trends: slightly above average but no spike
    trends = rng.uniform(40, 65, 60)

    inp = SentimentInput(
        news_articles=articles,
        social_posts=posts,
        analyst_revisions=revisions,
        short_interest_series=si,
        insider_transactions=insiders,
        etf_flow_series=flows,
        consumer_confidence_series=cc,
        google_trends_series=trends,
        contrarian_zscore_threshold=1.5,
    )

    sig = compute_sentiment_composite_signal(inp)

    print(f"domain    : {sig.domain}")
    print(f"value     : {sig.value:+.4f}")
    print(f"conviction: {sig.conviction:.4f}")
    print(f"regime    : {sig.regime}")
    print("components:")
    for k, v in sig.components.items():
        print(f"  {k:<20s}: {v:+.4f}")
    print("metadata (selected):")
    for k in ("composite_raw", "composite_z", "contrarian_fade_applied",
              "spike_detected", "avg_data_coverage"):
        print(f"  {k:<28s}: {sig.metadata[k]}")
    if sig.warnings:
        print("warnings:", sig.warnings)
