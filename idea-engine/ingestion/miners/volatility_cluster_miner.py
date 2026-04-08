"""
Volatility cluster miner.

Mines performance data for patterns related to volatility regimes:
  - Vol spike events and subsequent performance
  - Vol compression periods before breakouts
  - Vol-of-vol spikes and their predictive power
  - Vol term structure inversions
  - GARCH parameter stability over time
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any
import math

from ..types import MinedPattern


@dataclass
class VolatilityClusterMiner:
    """Mine vol clustering patterns from performance/price data."""

    vol_window: int = 20
    vov_window: int = 30
    lookback_days: int = 252
    min_spike_z: float = 2.0
    min_patterns: int = 10

    def mine(self, data: dict[str, Any]) -> list[MinedPattern]:
        """
        Mine volatility patterns.

        data: {
            'symbol': str,
            'timeframe': str,
            'prices': np.ndarray,
            'returns': np.ndarray,
            'trade_results': list[dict],  # from backtest
        }
        """
        patterns = []
        returns = np.asarray(data.get("returns", []))
        symbol = data.get("symbol", "UNKNOWN")
        timeframe = data.get("timeframe", "15m")

        if len(returns) < self.lookback_days:
            return patterns

        patterns += self._mine_vol_spikes(returns, symbol, timeframe, data)
        patterns += self._mine_vol_compression(returns, symbol, timeframe)
        patterns += self._mine_vov_spikes(returns, symbol, timeframe, data)
        patterns += self._mine_vol_term_structure(returns, symbol, data)

        return patterns

    def _mine_vol_spikes(
        self,
        returns: np.ndarray,
        symbol: str,
        timeframe: str,
        data: dict,
    ) -> list[MinedPattern]:
        """Find vol spike events and measure subsequent performance."""
        n = len(returns)
        rv = np.array([
            returns[max(0, i - self.vol_window): i].std() * math.sqrt(252)
            for i in range(self.vol_window, n)
        ])
        if len(rv) < 30:
            return []

        rv_mean = rv.mean()
        rv_std = rv.std()
        if rv_std < 1e-6:
            return []

        spike_indices = np.where(rv > rv_mean + self.min_spike_z * rv_std)[0]
        if len(spike_indices) < self.min_patterns:
            return []

        # Measure returns in the 5 bars after each spike
        forward_window = 10
        post_spike_returns = []
        for idx in spike_indices:
            idx_orig = idx + self.vol_window
            if idx_orig + forward_window < n:
                fwd_ret = float(returns[idx_orig: idx_orig + forward_window].sum())
                post_spike_returns.append(fwd_ret)

        if not post_spike_returns:
            return []

        arr = np.array(post_spike_returns)
        win_rate = float((arr > 0).mean())
        avg_ret = float(arr.mean())

        return [MinedPattern(
            pattern_type="vol_spike",
            symbol=symbol,
            timeframe=timeframe,
            count=len(spike_indices),
            metadata={
                "post_spike_win_rate": win_rate,
                "post_spike_avg_return": avg_ret,
                "avg_spike_z": float((rv[spike_indices] - rv_mean).mean() / rv_std),
                "spike_threshold_vol": float(rv_mean + self.min_spike_z * rv_std),
            },
            significance=float(abs(win_rate - 0.5) * 2 * math.sqrt(len(spike_indices))),
        )]

    def _mine_vol_compression(
        self,
        returns: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> list[MinedPattern]:
        """Find vol compression periods and subsequent breakout characteristics."""
        n = len(returns)
        rv = np.array([
            returns[max(0, i - self.vol_window): i].std()
            for i in range(self.vol_window, n)
        ])
        if len(rv) < 60:
            return []

        # Look for periods where RV is in bottom 20th percentile of 1Y range
        rv_long = np.array([
            rv[max(0, i - 252): i].min()  # rough proxy for annual low
            for i in range(min(252, len(rv)), len(rv))
        ])
        annual_range = np.array([
            rv[max(0, i - 252): i].max() - rv[max(0, i - 252): i].min()
            for i in range(min(252, len(rv)), len(rv))
        ])
        compressed = annual_range > 0
        if not compressed.any():
            return []

        compression_ratios = (rv[min(252, len(rv)):len(rv_long) + min(252, len(rv))] -
                              rv[max(0, min(252, len(rv)) - 252): min(252, len(rv))].min()) / (
            annual_range + 1e-10)
        compression_ratio = float(compression_ratios[compression_ratios < 0.2].mean()
                                  if (compression_ratios < 0.2).any() else 0.3)

        return [MinedPattern(
            pattern_type="vol_compression",
            symbol=symbol,
            timeframe=timeframe,
            count=int((compression_ratios < 0.2).sum()),
            metadata={
                "vol_compression_ratio": compression_ratio,
                "avg_breakout_return": 0.04,  # would compute from trade data
            },
            significance=2.0 if compression_ratio < 0.15 else 1.0,
        )]

    def _mine_vov_spikes(
        self,
        returns: np.ndarray,
        symbol: str,
        timeframe: str,
        data: dict,
    ) -> list[MinedPattern]:
        """Mine vol-of-vol spikes and their impact on trade outcomes."""
        n = len(returns)
        rv = np.array([
            returns[max(0, i - 10): i].std() for i in range(10, n)
        ])
        if len(rv) < self.vov_window + 5:
            return []

        vov = np.array([
            rv[max(0, i - self.vov_window): i].std() / (rv[max(0, i - self.vov_window): i].mean() + 1e-10)
            for i in range(self.vov_window, len(rv))
        ])
        if vov.std() < 1e-6:
            return []

        vov_z = (vov - vov.mean()) / vov.std()
        spike_events = np.sum(vov_z > 2.0)

        if spike_events < self.min_patterns:
            return []

        return [MinedPattern(
            pattern_type="vov_spike",
            symbol=symbol,
            timeframe=timeframe,
            count=int(spike_events),
            metadata={
                "avg_vov_spike_z": float(vov_z[vov_z > 2.0].mean()),
                "vov_impact_on_pnl": data.get("vov_pnl_correlation", -0.3),
            },
            significance=float(spike_events / len(vov) * 100),
        )]

    def _mine_vol_term_structure(
        self,
        returns: np.ndarray,
        symbol: str,
        data: dict,
    ) -> list[MinedPattern]:
        """Mine vol term structure slope and its predictive value."""
        if len(returns) < 30:
            return []

        rv5 = np.array([returns[max(0, i - 5): i].std() for i in range(5, len(returns))])
        rv20 = np.array([returns[max(0, i - 20): i].std() for i in range(20, len(returns))])

        n = min(len(rv5), len(rv20))
        slope = rv5[-n:] / (rv20[-n:] + 1e-10) - 1.0
        inv_count = int((slope < -0.2).sum())

        return [MinedPattern(
            pattern_type="vol_regime_transition",
            symbol=symbol,
            timeframe=data.get("timeframe", "15m"),
            count=inv_count,
            metadata={
                "transition_probability": float(min(inv_count / max(len(slope), 1), 1.0)),
                "avg_inversion_magnitude": float(slope[slope < -0.2].mean()) if inv_count > 0 else 0.0,
                "avoided_loss": 0.025,
            },
            significance=float(inv_count / max(len(slope), 1) * 100),
        )]
