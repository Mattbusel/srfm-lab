"""
Funding rate miner for crypto perpetuals.

Mines funding rate data for actionable patterns:
  - Extreme funding rate events (long squeeze / short squeeze setups)
  - Funding rate divergence across exchanges
  - Funding spike followed by price decay
  - Funding rate trend correlation with price
  - Regime classification by funding level
"""

from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Any

from ..types import MinedPattern


@dataclass
class FundingRateMiner:
    extreme_positive_threshold: float = 0.001   # 0.1% per 8h
    extreme_negative_threshold: float = -0.0005
    lookback_periods: int = 90   # 90 * 8h ≈ 30 days
    min_events: int = 5
    forward_periods: int = 6    # look 6 * 8h = 2 days ahead

    def mine(self, data: dict[str, Any]) -> list[MinedPattern]:
        patterns = []
        funding = np.asarray(data.get("funding_rates", []))
        prices = np.asarray(data.get("prices", []))
        symbol = data.get("symbol", "UNKNOWN")

        if len(funding) < self.lookback_periods:
            return patterns

        patterns += self._mine_extreme_positive(funding, prices, symbol)
        patterns += self._mine_extreme_negative(funding, prices, symbol)
        patterns += self._mine_funding_spike_decay(funding, prices, symbol)
        patterns += self._mine_funding_trend_momentum(funding, prices, symbol)
        patterns += self._mine_cross_exchange_divergence(data)

        return patterns

    def _mine_extreme_positive(
        self,
        funding: np.ndarray,
        prices: np.ndarray,
        symbol: str,
    ) -> list[MinedPattern]:
        extreme_mask = funding > self.extreme_positive_threshold
        events = np.where(extreme_mask)[0]

        if len(events) < self.min_events:
            return []

        # Measure price change forward_periods after each extreme event
        fwd_returns = []
        for idx in events:
            if idx + self.forward_periods < len(prices):
                fwd_ret = float(prices[idx + self.forward_periods] / prices[idx] - 1)
                fwd_returns.append(fwd_ret)

        if not fwd_returns:
            return []

        arr = np.array(fwd_returns)
        reversion_freq = float((arr < 0).mean())  # how often price dropped

        return [MinedPattern(
            pattern_type="funding_extreme_positive",
            symbol=symbol,
            timeframe="8h",
            count=len(events),
            metadata={
                "reversion_frequency": reversion_freq,
                "avg_reversion_magnitude": float(arr[arr < 0].mean()) if (arr < 0).any() else 0.0,
                "avg_funding_at_extreme": float(funding[extreme_mask].mean()),
                "threshold": self.extreme_positive_threshold,
            },
            significance=float(abs(reversion_freq - 0.5) * 2 * math.sqrt(len(events))),
        )]

    def _mine_extreme_negative(
        self,
        funding: np.ndarray,
        prices: np.ndarray,
        symbol: str,
    ) -> list[MinedPattern]:
        extreme_mask = funding < self.extreme_negative_threshold
        events = np.where(extreme_mask)[0]

        if len(events) < self.min_events:
            return []

        fwd_returns = []
        for idx in events:
            if idx + self.forward_periods < len(prices):
                fwd_ret = float(prices[idx + self.forward_periods] / prices[idx] - 1)
                fwd_returns.append(fwd_ret)

        if not fwd_returns:
            return []

        arr = np.array(fwd_returns)
        bounce_freq = float((arr > 0).mean())

        return [MinedPattern(
            pattern_type="funding_extreme_negative",
            symbol=symbol,
            timeframe="8h",
            count=len(events),
            metadata={
                "bounce_frequency": bounce_freq,
                "avg_bounce_magnitude": float(arr[arr > 0].mean()) if (arr > 0).any() else 0.0,
                "threshold": self.extreme_negative_threshold,
            },
            significance=float(abs(bounce_freq - 0.5) * 2 * math.sqrt(len(events))),
        )]

    def _mine_funding_spike_decay(
        self,
        funding: np.ndarray,
        prices: np.ndarray,
        symbol: str,
    ) -> list[MinedPattern]:
        """After funding spike, does price decay (deleverage)?"""
        if len(funding) < 20:
            return []

        f_mean = funding.mean()
        f_std = funding.std()
        spike_mask = funding > f_mean + 2 * f_std
        spikes = np.where(spike_mask)[0]

        if len(spikes) < 3:
            return []

        decays = []
        for idx in spikes:
            if idx + 12 < len(prices):
                ret_12 = prices[idx + 12] / prices[idx] - 1
                decays.append(ret_12)

        if not decays:
            return []

        arr = np.array(decays)
        return [MinedPattern(
            pattern_type="funding_spike_decay",
            symbol=symbol,
            timeframe="8h",
            count=len(spikes),
            metadata={
                "decay_frequency": float((arr < 0).mean()),
                "avg_decay": float(arr.mean()),
                "spike_z_avg": float((funding[spike_mask] - f_mean).mean() / f_std),
            },
            significance=float(abs(arr.mean()) * 100),
        )]

    def _mine_funding_trend_momentum(
        self,
        funding: np.ndarray,
        prices: np.ndarray,
        symbol: str,
    ) -> list[MinedPattern]:
        """Correlate funding rate trend with price direction."""
        if len(funding) < 30 or len(prices) < 30:
            return []

        n = min(len(funding), len(prices)) - 6
        # 6-period funding trend vs next-period return
        funding_trend = np.array([
            funding[i: i + 6].mean() - funding[i - 6: i].mean()
            if i >= 6 else 0.0
            for i in range(6, n)
        ])
        price_ret = np.array([
            prices[i + 6] / prices[i] - 1
            for i in range(6, n)
            if i + 6 < len(prices)
        ])

        m = min(len(funding_trend), len(price_ret))
        if m < 20:
            return []

        corr = float(np.corrcoef(funding_trend[:m], price_ret[:m])[0, 1])

        return [MinedPattern(
            pattern_type="funding_trend",
            symbol=symbol,
            timeframe="8h",
            count=m,
            metadata={
                "funding_price_correlation": corr,
                "is_predictive": abs(corr) > 0.15,
            },
            significance=float(abs(corr) * math.sqrt(m)),
        )]

    def _mine_cross_exchange_divergence(self, data: dict) -> list[MinedPattern]:
        """Find cross-exchange funding divergence."""
        exchange_rates = data.get("exchange_funding_rates", {})
        symbol = data.get("symbol", "UNKNOWN")

        if len(exchange_rates) < 2:
            return []

        rates = {ex: np.asarray(r) for ex, r in exchange_rates.items()}
        n = min(len(v) for v in rates.values())
        if n < 10:
            return []

        # Compute pairwise divergences
        keys = list(rates.keys())
        divergences = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                diff = np.abs(rates[keys[i]][-n:] - rates[keys[j]][-n:])
                divergences.append(float(diff.mean()))

        if not divergences:
            return []

        avg_div = float(np.mean(divergences))
        if avg_div < 0.0001:
            return []

        return [MinedPattern(
            pattern_type="funding_rate_divergence",
            symbol=symbol,
            timeframe="8h",
            count=len(divergences),
            metadata={
                "avg_divergence": avg_div,
                "exchanges": keys,
                "max_divergence": float(max(divergences)),
            },
            significance=float(avg_div * 1000),
        )]
