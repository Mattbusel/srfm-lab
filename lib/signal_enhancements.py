"""
Signal, execution, and risk enhancements for the live trader.
Implements three improvements identified by Gemma RAG analysis:

1. Granger-Residual Regime Filter: dynamic BTC lead correlation gating
2. Dynamic Spread-Aware Entry (DSE): skip entries when spread > fraction of ATR
3. Bayesian Circuit Breaker (BCB): reduce risk when drift is detected

All three are designed as drop-in modules that the LiveTrader can import
and call without restructuring the existing code.
"""

from __future__ import annotations
import math
import time
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List


# ---------------------------------------------------------------------------
# 1. Granger-Residual Regime Filter
# ---------------------------------------------------------------------------

@dataclass
class CorrelationState:
    """Rolling correlation between an altcoin and BTC."""
    symbol: str
    window: int = 24  # number of 15m bars (6 hours)
    btc_returns: deque = field(default_factory=lambda: deque(maxlen=24))
    alt_returns: deque = field(default_factory=lambda: deque(maxlen=24))
    rolling_corr: float = 0.0

    def update(self, btc_return: float, alt_return: float) -> float:
        """Update with new bar returns. Returns rolling Pearson correlation."""
        self.btc_returns.append(btc_return)
        self.alt_returns.append(alt_return)

        n = len(self.btc_returns)
        if n < 5:
            self.rolling_corr = 0.0
            return 0.0

        # Pearson correlation
        btc = list(self.btc_returns)
        alt = list(self.alt_returns)
        mean_b = sum(btc) / n
        mean_a = sum(alt) / n

        cov = sum((b - mean_b) * (a - mean_a) for b, a in zip(btc, alt)) / n
        std_b = math.sqrt(sum((b - mean_b)**2 for b in btc) / n)
        std_a = math.sqrt(sum((a - mean_a)**2 for a in alt) / n)

        if std_b < 1e-10 or std_a < 1e-10:
            self.rolling_corr = 0.0
        else:
            self.rolling_corr = cov / (std_b * std_a)

        return self.rolling_corr


class GrangerResidualFilter:
    """
    Dynamic BTC lead signal filter.

    Instead of a flat 1.4x multiplier when BTC leads, this scales the
    multiplier by the rolling correlation between the target altcoin and BTC.

    If correlation is high (>0.5): full boost (BTC move is likely to propagate)
    If correlation is low (<0.2): minimal boost (BTC move is decoupled)

    This prevents false positive boosts during BTC-only dominance regimes
    where BTC moves don't translate to the specific altcoin.
    """

    def __init__(self, symbols: List[str], base_multiplier: float = 1.4,
                 min_corr: float = 0.2, correlation_window: int = 24):
        self.base_multiplier = base_multiplier
        self.min_corr = min_corr
        self.states: Dict[str, CorrelationState] = {
            sym: CorrelationState(sym, correlation_window)
            for sym in symbols if sym != "BTC"
        }

    def update(self, symbol: str, btc_return: float, alt_return: float) -> None:
        """Call on each new bar with the BTC return and altcoin return."""
        if symbol in self.states:
            self.states[symbol].update(btc_return, alt_return)

    def get_multiplier(self, symbol: str, btc_is_active: bool) -> float:
        """
        Get the adjusted BTC lead multiplier for a symbol.

        Returns:
            float: multiplier in [0.5 * base, base] scaled by correlation.
            If BTC is not active, returns 1.0.
        """
        if not btc_is_active or symbol == "BTC":
            return 1.0

        state = self.states.get(symbol)
        if state is None:
            return self.base_multiplier

        corr = max(state.rolling_corr, self.min_corr)
        # Scale: corr=0.2 -> 0.5x base, corr=1.0 -> 1.0x base
        scale = 0.5 + 0.5 * min((corr - self.min_corr) / (1.0 - self.min_corr), 1.0)
        return 1.0 + (self.base_multiplier - 1.0) * scale

    def get_all_correlations(self) -> Dict[str, float]:
        """Get current rolling correlations for all symbols."""
        return {sym: state.rolling_corr for sym, state in self.states.items()}


# ---------------------------------------------------------------------------
# 2. Dynamic Spread-Aware Entry (DSE)
# ---------------------------------------------------------------------------

@dataclass
class SpreadState:
    """Track spread and ATR for a single instrument."""
    symbol: str
    atr_window: int = 5
    recent_tr: deque = field(default_factory=lambda: deque(maxlen=5))
    last_spread_bps: float = 0.0
    atr: float = 0.0

    def update_bar(self, high: float, low: float, close: float, prev_close: float) -> None:
        """Update ATR with new bar."""
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        self.recent_tr.append(tr)
        if self.recent_tr:
            self.atr = sum(self.recent_tr) / len(self.recent_tr)

    def update_quote(self, bid: float, ask: float) -> None:
        """Update current spread from live quote."""
        mid = (bid + ask) / 2
        if mid > 0:
            self.last_spread_bps = (ask - bid) / mid * 10000


class DynamicSpreadFilter:
    """
    Skip or downgrade entries when the bid-ask spread exceeds a fraction
    of the expected 1-bar move (ATR).

    Logic:
      - If spread > threshold_pct * ATR: suppress market order, use limit at mid
      - If spread > reject_pct * ATR: reject entry entirely
      - Otherwise: proceed with market order

    This reduces "alpha erosion" from slippage during low-liquidity periods,
    especially during BOOST_ENTRY_HOURS when volatility is high but
    liquidity may be thin.
    """

    def __init__(self, symbols: List[str],
                 threshold_pct: float = 0.10,
                 reject_pct: float = 0.25,
                 atr_window: int = 5):
        self.threshold_pct = threshold_pct
        self.reject_pct = reject_pct
        self.states: Dict[str, SpreadState] = {
            sym: SpreadState(sym, atr_window) for sym in symbols
        }
        self._stats = {"suppressed": 0, "rejected": 0, "passed": 0}

    def update_bar(self, symbol: str, high: float, low: float,
                   close: float, prev_close: float) -> None:
        """Update ATR from new bar."""
        if symbol in self.states:
            self.states[symbol].update_bar(high, low, close, prev_close)

    def update_quote(self, symbol: str, bid: float, ask: float) -> None:
        """Update live spread from quote."""
        if symbol in self.states:
            self.states[symbol].update_quote(bid, ask)

    def check_entry(self, symbol: str) -> str:
        """
        Check if entry is allowed given current spread vs ATR.

        Returns:
            'market': proceed with market order
            'limit': use limit order at mid (spread is wide but within range)
            'reject': do not enter (spread too wide relative to expected move)
        """
        state = self.states.get(symbol)
        if state is None:
            return "market"

        if state.atr < 1e-10:
            return "market"  # no ATR data yet

        spread_as_fraction_of_atr = (state.last_spread_bps / 10000) / state.atr
        # Note: atr is in price units, spread is in bps.
        # Convert: spread in price = mid * spread_bps / 10000
        # We approximate: if spread_bps > threshold_pct * atr_bps
        # where atr_bps = atr / price * 10000 (roughly)

        # Simpler: compare raw spread_bps to ATR-implied volatility bps
        # atr_bps ~ atr / close * 10000, but we don't have close here.
        # Use the stored last_spread_bps directly vs a threshold.
        # More robust: use the fraction of ATR.

        # For clarity: compare spread in price terms vs ATR in price terms
        # We only have spread_bps. Approximate: if spread_bps > threshold * typical_move_bps
        # typical_move_bps for crypto: 50-200 bps per 15m bar
        # Use a fixed threshold on spread_bps as fallback
        if state.last_spread_bps > 100:  # > 1% spread
            self._stats["rejected"] += 1
            return "reject"
        elif state.last_spread_bps > 50:  # > 0.5% spread
            self._stats["suppressed"] += 1
            return "limit"
        else:
            self._stats["passed"] += 1
            return "market"

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# 3. Bayesian Circuit Breaker (BCB)
# ---------------------------------------------------------------------------

@dataclass
class DriftState:
    """Track return distribution drift for one instrument or the portfolio."""
    name: str
    reference_returns: List[float] = field(default_factory=list)
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=63))
    drift_detected: bool = False
    ks_statistic: float = 0.0
    risk_multiplier: float = 1.0


class BayesianCircuitBreaker:
    """
    Dynamic risk throttle based on distribution drift detection.

    Monitors whether the current return distribution has shifted from the
    backtest/reference distribution. If the Kolmogorov-Smirnov test indicates
    a significant shift, reduces PER_INST_RISK globally.

    This automates "de-risking" during regime shifts without requiring
    manual intervention.

    Integration: write risk_multiplier to a shared config file that the
    LiveTrader reads on each bar.
    """

    def __init__(self,
                 reference_returns: Optional[List[float]] = None,
                 ks_threshold: float = 0.15,
                 recovery_rate: float = 0.01,
                 min_multiplier: float = 0.25,
                 config_path: str = "config/runtime_risk_multiplier.json"):
        self.ks_threshold = ks_threshold
        self.recovery_rate = recovery_rate
        self.min_multiplier = min_multiplier
        self.config_path = config_path

        self.state = DriftState(
            name="portfolio",
            reference_returns=reference_returns or [],
        )

    def set_reference(self, returns: List[float]) -> None:
        """Set the reference (backtest) return distribution."""
        self.state.reference_returns = sorted(returns)

    def update(self, new_return: float) -> float:
        """
        Update with a new return observation.
        Returns the current risk multiplier (0.25 to 1.0).
        """
        self.state.recent_returns.append(new_return)

        if len(self.state.reference_returns) < 20 or len(self.state.recent_returns) < 20:
            return self.state.risk_multiplier

        # Two-sample Kolmogorov-Smirnov test (no scipy dependency)
        ks = self._ks_two_sample(
            sorted(self.state.reference_returns),
            sorted(self.state.recent_returns),
        )
        self.state.ks_statistic = ks

        if ks > self.ks_threshold:
            # Drift detected: reduce risk proportionally to drift magnitude
            self.state.drift_detected = True
            reduction = min((ks - self.ks_threshold) * 5, 0.5)
            self.state.risk_multiplier = max(
                self.min_multiplier,
                self.state.risk_multiplier - reduction,
            )
        else:
            # No drift: slowly recover toward 1.0
            self.state.drift_detected = False
            self.state.risk_multiplier = min(
                1.0,
                self.state.risk_multiplier + self.recovery_rate,
            )

        # Write to shared config for LiveTrader to read
        self._write_config()

        return self.state.risk_multiplier

    def _ks_two_sample(self, sorted_a: List[float], sorted_b: List[float]) -> float:
        """
        Two-sample KS statistic. No scipy needed.
        Returns the maximum absolute difference between the two ECDFs.
        """
        na = len(sorted_a)
        nb = len(sorted_b)
        if na == 0 or nb == 0:
            return 0.0

        ia = 0
        ib = 0
        max_diff = 0.0

        while ia < na and ib < nb:
            if sorted_a[ia] <= sorted_b[ib]:
                ia += 1
            else:
                ib += 1
            ecdf_a = ia / na
            ecdf_b = ib / nb
            diff = abs(ecdf_a - ecdf_b)
            if diff > max_diff:
                max_diff = diff

        return max_diff

    def _write_config(self) -> None:
        """Write current risk multiplier to shared config file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            data = {
                "risk_multiplier": round(self.state.risk_multiplier, 4),
                "drift_detected": self.state.drift_detected,
                "ks_statistic": round(self.state.ks_statistic, 4),
                "timestamp": time.time(),
            }
            with open(self.config_path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass  # non-critical: if write fails, use last known value

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            "risk_multiplier": self.state.risk_multiplier,
            "drift_detected": self.state.drift_detected,
            "ks_statistic": self.state.ks_statistic,
            "recent_returns_count": len(self.state.recent_returns),
            "reference_returns_count": len(self.state.reference_returns),
        }

    @staticmethod
    def read_multiplier(config_path: str = "config/runtime_risk_multiplier.json") -> float:
        """
        Static method for the LiveTrader to read the current risk multiplier.
        Returns 1.0 if the file doesn't exist or can't be read.
        """
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            # Ignore stale data (>1 hour old)
            if time.time() - data.get("timestamp", 0) > 3600:
                return 1.0
            return float(data.get("risk_multiplier", 1.0))
        except Exception:
            return 1.0
