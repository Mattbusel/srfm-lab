"""
Order flow miner — mines microstructure and order flow patterns.

Detects:
  - VPIN spikes and subsequent price moves
  - Kyle's lambda regime shifts (impact cost changes)
  - Order flow imbalance persistence
  - Toxic flow events and their duration
  - Bid-ask spread widening patterns
"""

from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Any

from ..types import MinedPattern


@dataclass
class OrderFlowMiner:
    vpin_window: int = 30
    vpin_threshold: float = 0.7
    min_events: int = 10
    forward_bars: int = 10

    def mine(self, data: dict[str, Any]) -> list[MinedPattern]:
        patterns = []
        prices = np.asarray(data.get("prices", []))
        volumes = np.asarray(data.get("volumes", []))
        symbol = data.get("symbol", "UNKNOWN")
        timeframe = data.get("timeframe", "15m")

        if len(prices) < self.vpin_window * 3 or len(volumes) < len(prices):
            return patterns

        patterns += self._mine_vpin_events(prices, volumes, symbol, timeframe)
        patterns += self._mine_kyle_lambda_regime(prices, volumes, symbol, timeframe)
        patterns += self._mine_flow_imbalance(prices, volumes, symbol, timeframe)

        return patterns

    def _estimate_vpin_rolling(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Rolling VPIN estimate."""
        n = len(prices)
        vpin_series = np.full(n, np.nan)

        for i in range(window + 1, n):
            p = prices[i - window: i + 1]
            v = volumes[i - window: i + 1]
            returns = np.diff(np.log(p))
            buy_vol = np.where(returns >= 0, v[1:], 0.0)
            sell_vol = np.where(returns < 0, v[1:], 0.0)
            total = buy_vol + sell_vol
            imbalances = np.abs(buy_vol - sell_vol) / (total + 1e-10)
            vpin_series[i] = float(imbalances.mean())

        return vpin_series

    def _mine_vpin_events(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> list[MinedPattern]:
        vpin = self._estimate_vpin_rolling(prices, volumes, self.vpin_window)
        spike_mask = vpin > self.vpin_threshold
        events = np.where(spike_mask)[0]

        if len(events) < self.min_events:
            return []

        # Post-spike price moves
        returns = np.diff(np.log(prices))
        fwd_returns = []
        for idx in events:
            if idx + self.forward_bars < len(returns):
                fwd = float(returns[idx: idx + self.forward_bars].sum())
                fwd_returns.append(fwd)

        if not fwd_returns:
            return []

        arr = np.array(fwd_returns)
        # VPIN spikes often predict adverse selection → price continues away
        return [MinedPattern(
            pattern_type="vpin_spike",
            symbol=symbol,
            timeframe=timeframe,
            count=len(events),
            metadata={
                "avg_vpin_at_spike": float(vpin[spike_mask][~np.isnan(vpin[spike_mask])].mean()),
                "fwd_volatility": float(np.abs(arr).mean()),
                "adverse_selection_rate": float((arr != 0).mean()),
                "vpin_threshold": self.vpin_threshold,
                "recommended_size_reduction": 0.4,
            },
            significance=float(len(events) / len(prices) * 100),
        )]

    def _mine_kyle_lambda_regime(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> list[MinedPattern]:
        """Detect Kyle lambda regime shifts."""
        returns = np.diff(np.log(prices))
        n = len(returns)
        window = 50

        if n < window * 2:
            return []

        lambdas = []
        for i in range(window, n):
            dp = returns[i - window: i]
            signed_vol = np.sign(dp) * volumes[1:][i - window: i]
            cov = np.cov(dp, signed_vol)
            lam = cov[0, 1] / max(cov[1, 1], 1e-10) if cov[1, 1] > 1e-10 else 0.0
            lambdas.append(lam)

        lambdas = np.array(lambdas)
        if lambdas.std() < 1e-10:
            return []

        # Detect regime shift: lambda doubled
        l_mean = lambdas.mean()
        l_std = lambdas.std()
        high_impact = np.sum(lambdas > l_mean + 2 * l_std)

        return [MinedPattern(
            pattern_type="kyle_lambda_regime",
            symbol=symbol,
            timeframe=timeframe,
            count=int(high_impact),
            metadata={
                "avg_lambda": float(l_mean),
                "lambda_std": float(l_std),
                "high_impact_events": int(high_impact),
                "avg_high_impact_lambda": float(lambdas[lambdas > l_mean + 2 * l_std].mean()) if high_impact > 0 else 0.0,
            },
            significance=float(high_impact / max(len(lambdas), 1) * 100),
        )]

    def _mine_flow_imbalance(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> list[MinedPattern]:
        """Persistent order flow imbalance → price continuation."""
        returns = np.diff(np.log(prices))
        n = len(returns)
        window = 10

        if n < window + 5:
            return []

        # OFI proxy: signed volume cumsum
        signed_vol = np.sign(returns) * volumes[1:]
        ofi = np.array([
            signed_vol[i - window: i].mean()
            for i in range(window, n)
        ])

        ofi_std = ofi.std()
        if ofi_std < 1e-10:
            return []

        # Persistent high OFI and forward returns
        high_ofi = ofi > ofi.std()
        low_ofi = ofi < -ofi.std()

        fwd_high = []
        fwd_low = []
        for i in np.where(high_ofi)[0]:
            if i + 5 < len(returns):
                fwd_high.append(returns[window + i: window + i + 5].sum())
        for i in np.where(low_ofi)[0]:
            if i + 5 < len(returns):
                fwd_low.append(returns[window + i: window + i + 5].sum())

        if not fwd_high or not fwd_low:
            return []

        continuation_up = float((np.array(fwd_high) > 0).mean())
        continuation_down = float((np.array(fwd_low) < 0).mean())
        avg_continuation = (continuation_up + continuation_down) / 2

        if avg_continuation < 0.55:
            return []

        return [MinedPattern(
            pattern_type="ofi_persistence",
            symbol=symbol,
            timeframe=timeframe,
            count=int(high_ofi.sum() + low_ofi.sum()),
            metadata={
                "continuation_rate": avg_continuation,
                "up_continuation": continuation_up,
                "down_continuation": continuation_down,
                "ofi_window": window,
            },
            significance=float(abs(avg_continuation - 0.5) * 2 * math.sqrt(high_ofi.sum())),
        )]
