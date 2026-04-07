"""
Sentiment-based alternative data signals.

Implements:
- VADER-style sentiment scoring and aggregation
- Sentiment momentum (trend of sentiment)
- Sentiment divergence (sentiment vs price)
- News volume and sentiment spike detection
- Cross-asset sentiment composite
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# VADER-style lexicon scorer (rule-based, no NLTK required)
# ---------------------------------------------------------------------------

# Compact positive / negative word lists covering common financial context
_POSITIVE_WORDS = frozenset([
    "good", "great", "excellent", "strong", "positive", "growth", "profit",
    "beat", "exceed", "outperform", "surge", "rally", "gain", "recovery",
    "expansion", "record", "high", "rise", "increase", "upside", "bullish",
    "upgrade", "buy", "recommend", "optimistic", "confident", "robust",
    "improve", "advance", "accelerate", "breakthrough", "innovative", "solid",
    "momentum", "strength", "opportunity", "attractive", "undervalued", "cheap",
])

_NEGATIVE_WORDS = frozenset([
    "bad", "poor", "weak", "negative", "decline", "loss", "miss", "disappoint",
    "underperform", "drop", "fall", "decrease", "contraction", "low", "bearish",
    "downgrade", "sell", "avoid", "pessimistic", "risk", "concern", "volatile",
    "uncertain", "challenge", "headwind", "pressure", "crisis", "bankruptcy",
    "lawsuit", "fraud", "investigation", "penalty", "fine", "warning", "cut",
    "layoff", "downside", "overvalued", "expensive",
])

_NEGATION_WORDS = frozenset([
    "not", "no", "never", "neither", "nor", "cannot", "can't", "won't", "didn't",
    "doesn't", "isn't", "aren't", "wasn't", "weren't", "don't", "haven't",
])

_INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "highly": 1.5, "significantly": 1.5,
    "slightly": 0.5, "somewhat": 0.7, "marginally": 0.5, "substantially": 1.5,
}


def score_text(text: str) -> float:
    """
    Rule-based sentiment score for a text string.

    Returns a value in [-1, 1]:
    - Positive words contribute +1
    - Negative words contribute -1
    - Negation flips the subsequent word's contribution
    - Intensifiers scale the next word's contribution

    Parameters
    ----------
    text : str

    Returns
    -------
    float
        Normalized sentiment score in [-1, 1].
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0

    words = text.lower().replace(",", " ").replace(".", " ").split()
    scores = []
    negate = False
    intensifier = 1.0

    for i, word in enumerate(words):
        word_clean = word.strip("!?;:\"'()")

        if word_clean in _NEGATION_WORDS:
            negate = True
            intensifier = 1.0
            continue

        if word_clean in _INTENSIFIERS:
            intensifier = _INTENSIFIERS[word_clean]
            continue

        if word_clean in _POSITIVE_WORDS:
            score = intensifier
            if negate:
                score = -score
            scores.append(score)
            negate = False
            intensifier = 1.0
        elif word_clean in _NEGATIVE_WORDS:
            score = -intensifier
            if negate:
                score = -score
            scores.append(score)
            negate = False
            intensifier = 1.0
        else:
            negate = False
            intensifier = 1.0

    if not scores:
        return 0.0

    raw = sum(scores)
    # Normalize: tanh-like compression to [-1, 1]
    n = len(scores)
    normalized = raw / (raw + np.sqrt(n + 1e-6)) if raw >= 0 else \
                 raw / (abs(raw) + np.sqrt(n + 1e-6))
    return float(np.clip(normalized, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Sentiment aggregator
# ---------------------------------------------------------------------------

class SentimentAggregator:
    """
    Aggregate text-level sentiment scores into asset-level daily signals.

    Parameters
    ----------
    lookback : int
        Rolling window for sentiment smoothing.
    decay : float
        Exponential decay factor for recency weighting (0 = equal weight,
        close to 1 = heavy recency weight).
    volume_weight : bool
        If True, weight each article's sentiment by its "volume" proxy
        (e.g., article count or engagement score).
    clip_std : float
        Clip raw z-scores to +/- clip_std before normalization.
    """

    def __init__(
        self,
        lookback: int = 10,
        decay: float = 0.7,
        volume_weight: bool = True,
        clip_std: float = 3.0,
    ) -> None:
        self.lookback = lookback
        self.decay = decay
        self.volume_weight = volume_weight
        self.clip_std = clip_std

    def score_corpus(
        self,
        texts: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Score a time-indexed series of text strings.

        Parameters
        ----------
        texts : pd.Series
            (dates,) or (dates, multi-article) series of text.
        volumes : pd.Series or None
            Optional article volumes for weighting.

        Returns
        -------
        pd.Series
            Daily sentiment scores.
        """
        scores = texts.apply(score_text)

        if volumes is not None and self.volume_weight:
            vol_aligned = volumes.reindex(scores.index).fillna(1.0)
            # Weight by sqrt(volume) to reduce outlier influence
            scores = scores * np.sqrt(vol_aligned)
            scores = scores / (np.sqrt(vol_aligned).replace(0, 1))

        return scores

    def aggregate_daily(
        self,
        raw_scores: pd.Series,
        freq: str = "D",
    ) -> pd.Series:
        """
        Aggregate intraday scores to daily frequency.

        Parameters
        ----------
        raw_scores : pd.Series
            High-frequency sentiment scores.
        freq : str
            Aggregation frequency ('D', 'W', etc.).

        Returns
        -------
        pd.Series
            Daily average sentiment.
        """
        if self.decay > 0:
            # Exponential weighting: more recent = higher weight
            def _ewa(group):
                vals = group.values
                if len(vals) == 0:
                    return float("nan")
                if len(vals) == 1:
                    return float(vals[0])
                weights = np.array([self.decay ** (len(vals) - 1 - i) for i in range(len(vals))])
                w_sum = float(weights.sum())
                if w_sum <= 0:
                    return float(np.nanmean(vals))
                return float(np.average(vals, weights=weights))
            return raw_scores.resample(freq).apply(_ewa)
        return raw_scores.resample(freq).mean()

    def smooth(self, daily_scores: pd.Series) -> pd.Series:
        """Apply rolling average to daily scores."""
        return daily_scores.rolling(self.lookback, min_periods=1).mean()

    def normalize(self, scores: pd.Series, window: Optional[int] = None) -> pd.Series:
        """Z-score normalize with rolling window."""
        w = window or self.lookback * 3
        mu = scores.rolling(w, min_periods=5).mean()
        sd = scores.rolling(w, min_periods=5).std().replace(0, 1.0)
        z = (scores - mu) / sd
        return z.clip(-self.clip_std, self.clip_std)

    def build_signal(
        self,
        texts: pd.Series,
        volumes: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        End-to-end pipeline: text → normalized daily sentiment signal.

        Returns
        -------
        pd.Series
            Signal in [-clip_std, clip_std].
        """
        raw = self.score_corpus(texts, volumes)
        daily = self.aggregate_daily(raw)
        smoothed = self.smooth(daily)
        normalized = self.normalize(smoothed)
        return normalized


# ---------------------------------------------------------------------------
# Sentiment Momentum
# ---------------------------------------------------------------------------

class SentimentMomentum:
    """
    Sentiment momentum: trend of sentiment signal.

    Signal = short_ma(sentiment) - long_ma(sentiment)

    Positive values → sentiment improving (bullish momentum).
    Negative values → sentiment deteriorating (bearish momentum).

    Parameters
    ----------
    fast : int
        Fast moving average window.
    slow : int
        Slow moving average window.
    signal_smooth : int
        Smoothing window for momentum signal.
    """

    def __init__(
        self,
        fast: int = 5,
        slow: int = 20,
        signal_smooth: int = 3,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_smooth = signal_smooth

    def compute(self, sentiment: pd.Series) -> pd.Series:
        """
        Compute sentiment momentum signal.

        Parameters
        ----------
        sentiment : pd.Series
            Daily sentiment score series.

        Returns
        -------
        pd.Series
            Sentiment momentum signal.
        """
        fast_ma = sentiment.rolling(self.fast, min_periods=1).mean()
        slow_ma = sentiment.rolling(self.slow, min_periods=1).mean()
        momentum = fast_ma - slow_ma
        signal = momentum.rolling(self.signal_smooth, min_periods=1).mean()
        return signal.rename("sentiment_momentum")

    def trend_strength(self, sentiment: pd.Series) -> pd.Series:
        """
        Measure consistency of sentiment direction over slow window.

        Returns
        -------
        pd.Series
            Fraction of days with positive sentiment in slow window.
        """
        return (sentiment > 0).rolling(self.slow, min_periods=1).mean().rename("sentiment_trend_strength")

    def sentiment_z_score(self, sentiment: pd.Series, window: int = 252) -> pd.Series:
        """Rolling z-score of sentiment."""
        mu = sentiment.rolling(window, min_periods=20).mean()
        sd = sentiment.rolling(window, min_periods=20).std().replace(0, 1.0)
        return ((sentiment - mu) / sd).clip(-3, 3).rename("sentiment_zscore")

    def ic_with_returns(
        self,
        sentiment: pd.Series,
        returns: pd.Series,
        forward_window: int = 5,
        ic_window: int = 63,
    ) -> pd.Series:
        """
        Compute rolling IC between sentiment signal and forward returns.

        Returns
        -------
        pd.Series
            Rolling Spearman IC.
        """
        fwd_ret = returns.rolling(forward_window).sum().shift(-forward_window)
        momentum_signal = self.compute(sentiment)
        combined = pd.concat([momentum_signal, fwd_ret], axis=1).dropna()
        combined.columns = ["signal", "fwd_ret"]

        ic_vals = []
        ic_idx = []
        for i in range(ic_window, len(combined)):
            window = combined.iloc[i - ic_window:i]
            rho, _ = stats.spearmanr(window["signal"], window["fwd_ret"])
            ic_vals.append(rho)
            ic_idx.append(combined.index[i])

        return pd.Series(ic_vals, index=ic_idx, name="sentiment_ic")


# ---------------------------------------------------------------------------
# Sentiment Divergence
# ---------------------------------------------------------------------------

class SentimentDivergence:
    """
    Detect divergence between sentiment and price action.

    Divergence occurs when:
    - Price rises but sentiment falls (bearish divergence)
    - Price falls but sentiment rises (bullish divergence)

    These divergences can signal mean reversion opportunities.

    Parameters
    ----------
    lookback : int
        Window to measure divergence.
    threshold : float
        Minimum |divergence| to generate a signal.
    price_change_window : int
        Window for computing price change in divergence calculation.
    """

    def __init__(
        self,
        lookback: int = 21,
        threshold: float = 1.5,
        price_change_window: int = 10,
    ) -> None:
        self.lookback = lookback
        self.threshold = threshold
        self.price_change_window = price_change_window

    def compute_divergence(
        self,
        sentiment: pd.Series,
        price: pd.Series,
    ) -> pd.Series:
        """
        Compute sentiment-price divergence index.

        Divergence = z(sentiment) - z(price_change)
        Positive → sentiment above price trend (supportive)
        Negative → price above sentiment (bearish risk)

        Returns
        -------
        pd.Series
        """
        # Z-score price change
        price_change = price.pct_change(self.price_change_window)
        pc_mu = price_change.rolling(self.lookback * 3, min_periods=10).mean()
        pc_sd = price_change.rolling(self.lookback * 3, min_periods=10).std().replace(0, 1.0)
        z_price = ((price_change - pc_mu) / pc_sd).clip(-3, 3)

        # Z-score sentiment
        s_mu = sentiment.rolling(self.lookback * 3, min_periods=10).mean()
        s_sd = sentiment.rolling(self.lookback * 3, min_periods=10).std().replace(0, 1.0)
        z_sent = ((sentiment - s_mu) / s_sd).clip(-3, 3)

        divergence = z_sent - z_price
        return divergence.rename("sentiment_divergence")

    def generate_signals(
        self,
        sentiment: pd.Series,
        price: pd.Series,
    ) -> pd.Series:
        """
        Generate trading signals from divergence.

        Bullish divergence (sent > price) → +1 signal
        Bearish divergence (sent < price) → -1 signal

        Returns
        -------
        pd.Series
            Signal in [-1, 0, 1].
        """
        div = self.compute_divergence(sentiment, price)
        signal = pd.Series(0.0, index=div.index)
        signal[div > self.threshold] = 1.0
        signal[div < -self.threshold] = -1.0
        return signal.rename("divergence_signal")

    def backtest(
        self,
        sentiment: pd.Series,
        price: pd.Series,
    ) -> Dict:
        """
        Simple backtest of divergence signal.

        Returns
        -------
        dict
            Performance statistics.
        """
        signal = self.generate_signals(sentiment, price)
        daily_ret = price.pct_change()
        port_rets = (signal.shift(1) * daily_ret).dropna()

        if len(port_rets) == 0:
            return {}

        freq = 252
        eq = (1 + port_rets).cumprod()
        total = float(eq.iloc[-1] - 1)
        n_years = len(port_rets) / freq
        cagr = (1 + total) ** (1 / max(n_years, 1e-6)) - 1
        sharpe = port_rets.mean() / (port_rets.std() + 1e-12) * np.sqrt(freq)

        return {
            "total_return": round(total, 4),
            "cagr": round(cagr, 4),
            "sharpe": round(float(sharpe), 4),
            "n_trades": int((signal.diff() != 0).sum()),
        }


# ---------------------------------------------------------------------------
# News Volume and Spike Detection
# ---------------------------------------------------------------------------

class NewsVolumeSignal:
    """
    Generate signals from news volume spikes.

    High news volume → heightened uncertainty → potential vol/reversal signal.

    Parameters
    ----------
    baseline_window : int
        Window to compute baseline news volume.
    spike_threshold : float
        Number of standard deviations above baseline to count as spike.
    sentiment_filter : bool
        If True, also condition on sentiment direction.
    holding_period : int
        Days to hold position after a spike.
    """

    def __init__(
        self,
        baseline_window: int = 30,
        spike_threshold: float = 2.0,
        sentiment_filter: bool = True,
        holding_period: int = 5,
    ) -> None:
        self.baseline_window = baseline_window
        self.spike_threshold = spike_threshold
        self.sentiment_filter = sentiment_filter
        self.holding_period = holding_period

    def detect_spikes(self, volume: pd.Series) -> pd.Series:
        """
        Detect news volume spikes.

        Returns
        -------
        pd.Series
            Boolean series: True on spike days.
        """
        vol_ma = volume.rolling(self.baseline_window, min_periods=5).mean()
        vol_std = volume.rolling(self.baseline_window, min_periods=5).std()
        z_vol = (volume - vol_ma) / (vol_std.replace(0, 1.0))
        return z_vol > self.spike_threshold

    def generate_signals(
        self,
        volume: pd.Series,
        sentiment: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Generate contrarian reversal signals on news volume spikes.

        On a spike day:
        - If sentiment is available: signal = -sign(sentiment) (fade extreme sentiment)
        - If no sentiment: signal = -1 (expect reversal of initial reaction)

        Returns
        -------
        pd.Series
        """
        spikes = self.detect_spikes(volume)
        signal = pd.Series(0.0, index=volume.index)

        for dt in volume.index[spikes]:
            idx = volume.index.get_loc(dt)
            # Hold for holding_period days after spike
            end_idx = min(idx + self.holding_period + 1, len(volume))

            if sentiment is not None and dt in sentiment.index and self.sentiment_filter:
                sent_val = sentiment.loc[dt]
                direction = -np.sign(sent_val) if abs(sent_val) > 0.1 else -1.0
            else:
                direction = -1.0  # default contrarian

            for j in range(idx + 1, end_idx):
                fade = 1.0 - (j - idx) / self.holding_period
                if abs(direction * fade) > abs(signal.iloc[j]):
                    signal.iloc[j] = direction * fade

        return signal.rename("news_volume_signal")

    def volume_regime(self, volume: pd.Series) -> pd.Series:
        """
        Classify each day into LOW/NORMAL/HIGH news volume regime.

        Returns
        -------
        pd.Series
            String labels.
        """
        vol_pct = volume.rank(pct=True)
        labels = pd.Series("NORMAL", index=volume.index)
        labels[vol_pct < 0.25] = "LOW"
        labels[vol_pct > 0.75] = "HIGH"
        return labels


# ---------------------------------------------------------------------------
# Cross-Asset Sentiment Composite
# ---------------------------------------------------------------------------

class CrossAssetSentiment:
    """
    Aggregate sentiment signals across multiple assets or topics
    into a composite market sentiment indicator.

    Parameters
    ----------
    lookback : int
        Window for composite smoothing.
    method : str
        'equal_weight', 'pca', or 'ic_weighted'.
    """

    def __init__(
        self,
        lookback: int = 21,
        method: str = "equal_weight",
    ) -> None:
        self.lookback = lookback
        self.method = method

    def build_composite(
        self,
        sentiment_dict: Dict[str, pd.Series],
        market_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Build composite sentiment from multiple sources.

        Parameters
        ----------
        sentiment_dict : dict
            {source_name: pd.Series} of daily sentiment scores.
        market_returns : pd.Series, optional
            Used for IC-weighted combination.

        Returns
        -------
        pd.Series
            Composite sentiment signal.
        """
        df = pd.concat(sentiment_dict.values(), axis=1)
        df.columns = list(sentiment_dict.keys())
        df = df.dropna(how="all")

        if self.method == "equal_weight":
            composite = df.mean(axis=1, skipna=True)

        elif self.method == "pca":
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            clean = df.dropna()
            if len(clean) < 20:
                composite = df.mean(axis=1, skipna=True)
            else:
                scaler = StandardScaler()
                X = scaler.fit_transform(clean.values)
                pca = PCA(n_components=1)
                pca.fit(X)
                loadings = pca.components_[0]
                # Orient: positive loadings sum should be positive
                if loadings.sum() < 0:
                    loadings = -loadings
                # Project
                proj_vals = []
                for dt in df.index:
                    row = df.loc[dt].fillna(0).values
                    x_scaled = (row - scaler.mean_) / (scaler.scale_ + 1e-10)
                    proj_vals.append(float(x_scaled @ loadings))
                composite = pd.Series(proj_vals, index=df.index)

        elif self.method == "ic_weighted" and market_returns is not None:
            weights = {}
            for col in df.columns:
                aligned = pd.concat([df[col], market_returns.shift(-1)], axis=1).dropna()
                if len(aligned) < 20:
                    weights[col] = 0.0
                    continue
                rho, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                weights[col] = max(rho, 0.0)  # floor at 0
            total_w = sum(weights.values())
            if total_w > 0:
                composite = sum(df[col] * w / total_w for col, w in weights.items())
            else:
                composite = df.mean(axis=1, skipna=True)
        else:
            composite = df.mean(axis=1, skipna=True)

        smoothed = composite.rolling(self.lookback, min_periods=1).mean()
        return smoothed.rename("composite_sentiment")

    def sentiment_regime(
        self,
        composite: pd.Series,
        n_regimes: int = 3,
    ) -> pd.Series:
        """
        Classify composite sentiment into regimes using quantile buckets.

        Returns
        -------
        pd.Series
            'BULLISH', 'NEUTRAL', or 'BEARISH'.
        """
        if n_regimes == 2:
            labels = pd.Series("NEUTRAL", index=composite.index)
            q50 = composite.quantile(0.5)
            labels[composite > q50] = "BULLISH"
            labels[composite <= q50] = "BEARISH"
        else:
            q33 = composite.quantile(0.33)
            q67 = composite.quantile(0.67)
            labels = pd.Series("NEUTRAL", index=composite.index)
            labels[composite > q67] = "BULLISH"
            labels[composite < q33] = "BEARISH"

        return labels

    def regime_statistics(
        self,
        composite: pd.Series,
        market_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute market return statistics per sentiment regime.

        Returns
        -------
        pd.DataFrame
        """
        regime = self.sentiment_regime(composite)
        combined = pd.concat(
            [market_returns.rename("ret"), regime.rename("regime")],
            axis=1,
        ).dropna()

        rows = []
        for r in combined["regime"].unique():
            subset = combined[combined["regime"] == r]["ret"]
            rows.append({
                "regime": r,
                "n_obs": len(subset),
                "mean_daily_ret": round(subset.mean(), 6),
                "annualized_ret": round(subset.mean() * 252, 4),
                "vol_annual": round(subset.std() * np.sqrt(252), 4),
                "sharpe": round(subset.mean() / (subset.std() + 1e-12) * np.sqrt(252), 4),
                "pct_positive": round((subset > 0).mean(), 4),
            })

        return pd.DataFrame(rows).set_index("regime")
