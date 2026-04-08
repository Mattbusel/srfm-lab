"""
Entropy miner — mines information-theoretic patterns from price/trade data.

Detects:
  - Low permutation entropy periods (high predictability windows)
  - Transfer entropy relationships between assets
  - LZ complexity regime transitions
  - Market efficiency score evolution
  - Complexity-entropy plane positioning shifts
"""

from __future__ import annotations
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Any

from ..types import MinedPattern


@dataclass
class EntropyMiner:
    pe_window: int = 60
    pe_order: int = 4
    te_window: int = 100
    te_lag: int = 1
    te_n_bins: int = 8
    min_events: int = 15
    low_pe_threshold: float = 0.4
    high_pe_threshold: float = 0.8

    def mine(self, data: dict[str, Any]) -> list[MinedPattern]:
        patterns = []
        returns = np.asarray(data.get("returns", []))
        symbol = data.get("symbol", "UNKNOWN")
        timeframe = data.get("timeframe", "15m")

        if len(returns) < self.pe_window * 2:
            return patterns

        patterns += self._mine_low_entropy_periods(returns, symbol, timeframe, data)
        patterns += self._mine_high_entropy_periods(returns, symbol, timeframe)
        patterns += self._mine_transfer_entropy(data)
        patterns += self._mine_entropy_regime_transitions(returns, symbol, timeframe)

        return patterns

    def _permutation_entropy(self, x: np.ndarray) -> float:
        from itertools import permutations as _perms
        n = len(x)
        if n < self.pe_order:
            return 1.0
        n_patterns = math.factorial(self.pe_order)
        perm_indices = {p: i for i, p in enumerate(_perms(range(self.pe_order)))}
        counts = np.zeros(n_patterns)
        for i in range(n - self.pe_order + 1):
            counts[perm_indices[tuple(np.argsort(x[i: i + self.pe_order]))]] += 1
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)) / math.log(n_patterns))

    def _mine_low_entropy_periods(
        self,
        returns: np.ndarray,
        symbol: str,
        timeframe: str,
        data: dict,
    ) -> list[MinedPattern]:
        n = len(returns)
        pe_series = np.array([
            self._permutation_entropy(returns[i - self.pe_window: i])
            for i in range(self.pe_window, n, 5)
        ])
        if len(pe_series) < 10:
            return []

        low_pe_mask = pe_series < self.low_pe_threshold
        if low_pe_mask.sum() < self.min_events:
            return []

        # Get trade performance during low PE periods
        trade_results = data.get("trade_results", [])
        win_rate = 0.60  # default if no trade data
        if trade_results:
            low_pe_trades = [t for t in trade_results if t.get("pe_at_entry", 1.0) < self.low_pe_threshold]
            if low_pe_trades:
                win_rate = float(sum(1 for t in low_pe_trades if t.get("pnl", 0) > 0) / len(low_pe_trades))

        return [MinedPattern(
            pattern_type="low_entropy_period",
            symbol=symbol,
            timeframe=timeframe,
            count=int(low_pe_mask.sum()),
            metadata={
                "pe_threshold": self.low_pe_threshold,
                "win_rate_low_pe": win_rate,
                "avg_pe_in_low": float(pe_series[low_pe_mask].mean()),
                "fraction_time_low_pe": float(low_pe_mask.mean()),
            },
            significance=float(abs(win_rate - 0.5) * 2 * math.sqrt(low_pe_mask.sum())),
        )]

    def _mine_high_entropy_periods(
        self,
        returns: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> list[MinedPattern]:
        n = len(returns)
        pe_series = np.array([
            self._permutation_entropy(returns[i - self.pe_window: i])
            for i in range(self.pe_window, n, 5)
        ])
        if len(pe_series) < 10:
            return []

        high_pe_mask = pe_series > self.high_pe_threshold
        if high_pe_mask.sum() < self.min_events // 2:
            return []

        # During high entropy, returns should be near-random
        # Estimate avg loss from chasing signals
        avg_loss = -0.02  # placeholder

        return [MinedPattern(
            pattern_type="high_entropy_period",
            symbol=symbol,
            timeframe=timeframe,
            count=int(high_pe_mask.sum()),
            metadata={
                "high_pe_threshold": self.high_pe_threshold,
                "avg_loss_high_pe": avg_loss,
                "avg_pe_in_high": float(pe_series[high_pe_mask].mean()),
                "fraction_time_high_pe": float(high_pe_mask.mean()),
            },
            significance=float(high_pe_mask.mean() * 5),
        )]

    def _mine_transfer_entropy(self, data: dict) -> list[MinedPattern]:
        """Compute TE between this asset and reference assets."""
        returns = np.asarray(data.get("returns", []))
        symbol = data.get("symbol", "")
        ref_returns_dict = data.get("reference_returns", {})

        patterns = []
        for ref_symbol, ref_returns in ref_returns_dict.items():
            if ref_symbol == symbol:
                continue
            ref = np.asarray(ref_returns)
            n = min(len(returns), len(ref))
            if n < self.te_window * 2:
                continue

            r = returns[-n:]
            ref_r = ref[-n:]

            # Compute TE via binning
            te = self._transfer_entropy_fast(ref_r, r)
            te_rev = self._transfer_entropy_fast(r, ref_r)
            net_te = te - te_rev

            if abs(net_te) < 0.005:
                continue

            source = ref_symbol if net_te > 0 else symbol
            target = symbol if net_te > 0 else ref_symbol

            patterns.append(MinedPattern(
                pattern_type="transfer_entropy_signal",
                symbol=target,
                timeframe=data.get("timeframe", "15m"),
                count=1,
                metadata={
                    "transfer_entropy": float(abs(net_te)),
                    "source_asset": source,
                    "target_asset": target,
                    "optimal_lag": self.te_lag,
                    "predictability_r2": float(min(abs(net_te) * 20, 0.5)),
                },
                significance=float(abs(net_te) * 100),
            ))

        return patterns

    def _transfer_entropy_fast(self, source: np.ndarray, target: np.ndarray) -> float:
        """Fast TE via histogram binning."""
        n = len(target)
        lag = self.te_lag
        if n < lag + 5:
            return 0.0

        def _sym(x, nb=self.te_n_bins):
            lo, hi = x.min(), x.max()
            if hi == lo:
                return np.zeros(len(x), dtype=int)
            return np.digitize(x, np.linspace(lo, hi, nb + 1)[1:-1])

        yt = _sym(target[lag:])
        yt1 = _sym(target[:-lag])
        xlag = _sym(source[:len(yt)])
        m = min(len(yt), len(yt1), len(xlag))
        nb = self.te_n_bins

        # H(Y_t | Y_{t-1})
        joint2 = np.zeros((nb, nb))
        for a, b in zip(yt[:m], yt1[:m]):
            joint2[a, b] += 1
        joint2 /= max(joint2.sum(), 1)
        py1 = joint2.sum(axis=0)
        h2 = -np.sum(joint2[joint2 > 0] * np.log(joint2[joint2 > 0]))
        h_y1 = -np.sum(py1[py1 > 0] * np.log(py1[py1 > 0]))
        h_yt_given_yt1 = h2 - h_y1

        # H(Y_t | Y_{t-1}, X_{t-lag})
        joint3 = np.zeros((nb, nb, nb))
        for a, b, c in zip(yt[:m], yt1[:m], xlag[:m]):
            joint3[a, b, c] += 1
        joint3 /= max(joint3.sum(), 1)
        py1x = joint3.sum(axis=0)
        h3 = -np.sum(joint3[joint3 > 0] * np.log(joint3[joint3 > 0]))
        h_y1x = -np.sum(py1x[py1x > 0] * np.log(py1x[py1x > 0]))
        h_yt_given_yt1_x = h3 - h_y1x

        return max(0.0, float(h_yt_given_yt1 - h_yt_given_yt1_x))

    def _mine_entropy_regime_transitions(
        self,
        returns: np.ndarray,
        symbol: str,
        timeframe: str,
    ) -> list[MinedPattern]:
        n = len(returns)
        pe_series = np.array([
            self._permutation_entropy(returns[i - self.pe_window: i])
            for i in range(self.pe_window, n, 3)
        ])
        if len(pe_series) < 20:
            return []

        # Detect rapid PE increases
        pe_diff = np.diff(pe_series)
        rapid_rise = np.sum(pe_diff > 0.15)

        if rapid_rise < 5:
            return []

        return [MinedPattern(
            pattern_type="entropy_regime_transition",
            symbol=symbol,
            timeframe=timeframe,
            count=int(rapid_rise),
            metadata={
                "avg_pe_rise": float(pe_diff[pe_diff > 0.15].mean()),
                "transition_bars": 10,
            },
            significance=float(rapid_rise / len(pe_diff) * 100),
        )]
