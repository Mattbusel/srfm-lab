"""
order_flow_imbalance.py — Order Flow Imbalance (OFI) signal.

Covers:
  - Bid/ask queue change-based OFI (Cont et al.)
  - Signed volume OFI
  - Kyle's lambda (price impact coefficient)
  - Amihud illiquidity ratio
  - OFI autocorrelation and predictive power
  - Multi-level LOB OFI
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from trade_classification import ClassifiedTrade, Trade, TradeSide, SyntheticTradeGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LOBLevel:
    """Single level of the limit order book."""
    price: float
    bid_qty: float
    ask_qty: float
    timestamp: datetime

    @property
    def imbalance(self) -> float:
        total = self.bid_qty + self.ask_qty
        return (self.bid_qty - self.ask_qty) / total if total > 0 else 0.0


@dataclass
class LOBSnapshot:
    """Multi-level limit order book snapshot."""
    timestamp: datetime
    bids: List[Tuple[float, float]]   # [(price, qty), ...]
    asks: List[Tuple[float, float]]
    n_levels: int = 10

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def total_bid_qty(self, n: int = None) -> float:
        levels = self.bids[:n] if n else self.bids
        return sum(q for _, q in levels)

    def total_ask_qty(self, n: int = None) -> float:
        levels = self.asks[:n] if n else self.asks
        return sum(q for _, q in levels)

    def level_imbalance(self, n: int = 5) -> float:
        bid_q = self.total_bid_qty(n)
        ask_q = self.total_ask_qty(n)
        total = bid_q + ask_q
        return (bid_q - ask_q) / total if total > 0 else 0.0


@dataclass
class OFIEvent:
    """Single OFI observation from queue changes."""
    timestamp: datetime
    bid_delta: float      # change in best bid qty (Cont et al. formula)
    ask_delta: float
    ofi: float            # bid_delta - ask_delta
    signed_volume: float  # net signed trade volume in same interval
    price: float
    return_next: float = 0.0   # filled in after the fact for backtesting


@dataclass
class KyleLambda:
    """Kyle's price impact coefficient."""
    lambda_: float         # price per unit volume ($/share/lot)
    r_squared: float       # regression R^2
    t_stat: float
    n_obs: int
    holding_period_s: int
    interpretation: str


@dataclass
class AmihudIlliquidity:
    """Amihud (2002) illiquidity ratio."""
    ratio: float          # avg |return| / volume (higher = less liquid)
    annualized: float
    percentile: float     # vs history
    regime: str           # "liquid", "normal", "illiquid"
    history: List[float] = field(default_factory=list)


@dataclass
class OFISignal:
    timestamp: datetime
    ofi_raw: float
    ofi_normalized: float     # z-score
    ofi_ma: float             # moving average
    signed_vol_imbalance: float
    price_impact_estimate: float
    direction: str            # "buy_pressure", "sell_pressure", "neutral"
    kyle_lambda: float
    amihud_ratio: float


# ---------------------------------------------------------------------------
# LOB-based OFI calculator (Cont et al. 2014)
# ---------------------------------------------------------------------------

class ContOFICalculator:
    """
    Cont, Kukanov, Stoikov (2014) OFI definition:
    OFI_t = ΔB_t - ΔA_t
    where:
      ΔB_t = (b_t - b_{t-1}) if b_t >= b_{t-1} else q^B_t - 0  (bid side)
      ΔA_t = similar for ask side

    Simplified 1-level version (best bid/ask queues only).
    """

    def compute_ofi(
        self,
        prev_snap: LOBSnapshot,
        curr_snap: LOBSnapshot,
    ) -> OFIEvent:
        """Compute OFI from two consecutive LOB snapshots."""
        pb, pq_b = prev_snap.bids[0] if prev_snap.bids else (0.0, 0.0)
        pa, pq_a = prev_snap.asks[0] if prev_snap.asks else (0.0, 0.0)
        cb, cq_b = curr_snap.bids[0] if curr_snap.bids else (0.0, 0.0)
        ca, cq_a = curr_snap.asks[0] if curr_snap.asks else (0.0, 0.0)

        # Bid side contribution
        if cb > pb:
            delta_b = cq_b          # new price level: count all qty
        elif cb == pb:
            delta_b = cq_b - pq_b  # same level: net change
        else:
            delta_b = -pq_b         # price moved down: lost all prev qty

        # Ask side contribution
        if ca < pa:
            delta_a = cq_a
        elif ca == pa:
            delta_a = cq_a - pq_a
        else:
            delta_a = -pq_a

        ofi = delta_b - delta_a

        return OFIEvent(
            timestamp=curr_snap.timestamp,
            bid_delta=delta_b,
            ask_delta=delta_a,
            ofi=ofi,
            signed_volume=0.0,
            price=curr_snap.midpoint,
        )

    def compute_series(self, snapshots: List[LOBSnapshot]) -> List[OFIEvent]:
        if len(snapshots) < 2:
            return []
        return [
            self.compute_ofi(snapshots[i-1], snapshots[i])
            for i in range(1, len(snapshots))
        ]

    def multi_level_ofi(
        self,
        prev_snap: LOBSnapshot,
        curr_snap: LOBSnapshot,
        n_levels: int = 5,
    ) -> float:
        """Multi-level OFI: weighted sum across multiple book levels."""
        total_ofi = 0.0
        for level in range(min(n_levels, len(prev_snap.bids), len(curr_snap.bids))):
            pb, pq_b = prev_snap.bids[level]
            cb, cq_b = curr_snap.bids[level]
            pa, pq_a = prev_snap.asks[level]
            ca, cq_a = curr_snap.asks[level]

            # Weight decays with level depth
            w = 1.0 / (level + 1)

            if cb > pb: db = cq_b
            elif cb == pb: db = cq_b - pq_b
            else: db = -pq_b

            if ca < pa: da = cq_a
            elif ca == pa: da = cq_a - pq_a
            else: da = -pq_a

            total_ofi += w * (db - da)

        return total_ofi


# ---------------------------------------------------------------------------
# Signed volume OFI
# ---------------------------------------------------------------------------

class SignedVolumeOFI:
    """
    Computes OFI from signed trade volume:
    OFI = sum(+volume for buys) - sum(volume for sells) over interval
    """

    def compute_interval(
        self,
        classified_trades: List[ClassifiedTrade],
        window: int = 100,   # trades per window
    ) -> List[OFIEvent]:
        events = []
        for i in range(0, len(classified_trades), window):
            batch = classified_trades[i : i + window]
            if not batch:
                continue

            buy_vol  = sum(ct.trade.size for ct in batch if ct.side == TradeSide.BUY)
            sell_vol = sum(ct.trade.size for ct in batch if ct.side == TradeSide.SELL)
            total_vol = buy_vol + sell_vol
            ofi = buy_vol - sell_vol
            signed_imbal = ofi / total_vol if total_vol > 0 else 0.0

            last_trade = batch[-1].trade
            events.append(OFIEvent(
                timestamp=last_trade.timestamp,
                bid_delta=buy_vol,
                ask_delta=sell_vol,
                ofi=signed_imbal,
                signed_volume=ofi,
                price=last_trade.price,
            ))

        return events

    def rolling_ofi(
        self,
        classified_trades: List[ClassifiedTrade],
        window: int = 200,
    ) -> List[float]:
        """Rolling signed volume OFI series."""
        result = []
        for i in range(len(classified_trades)):
            batch = classified_trades[max(0, i - window + 1): i + 1]
            buy_vol  = sum(ct.trade.size for ct in batch if ct.side == TradeSide.BUY)
            sell_vol = sum(ct.trade.size for ct in batch if ct.side == TradeSide.SELL)
            total = buy_vol + sell_vol
            result.append((buy_vol - sell_vol) / total if total > 0 else 0.0)
        return result


# ---------------------------------------------------------------------------
# Kyle's lambda estimator
# ---------------------------------------------------------------------------

class KyleLambdaEstimator:
    """
    Estimates Kyle's lambda from price impact regression:
    ΔP_t = λ * OFI_t + ε
    Higher lambda → more price impact per unit volume.
    """

    def estimate(
        self,
        ofi_events: List[OFIEvent],
        price_series: List[float] = None,
        holding_period_s: int = 60,
    ) -> KyleLambda:
        if len(ofi_events) < 20:
            return KyleLambda(0.0, 0.0, 0.0, 0, holding_period_s, "insufficient data")

        # Price returns between events
        prices = price_series or [e.price for e in ofi_events]
        if len(prices) < len(ofi_events) + 1:
            prices = [ofi_events[0].price] + prices

        returns = []
        for i in range(1, min(len(prices), len(ofi_events) + 1)):
            if prices[i-1] != 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])

        ofi_vals = [e.ofi for e in ofi_events[:len(returns)]]

        if len(returns) < 10 or len(ofi_vals) < 10:
            return KyleLambda(0.0, 0.0, 0.0, len(returns), holding_period_s, "insufficient data")

        X = np.array(ofi_vals)
        Y = np.array(returns[:len(X)])

        # OLS: Y = lambda * X + intercept
        X_dm = X - np.mean(X)
        Y_dm = Y - np.mean(Y)

        if np.dot(X_dm, X_dm) == 0:
            return KyleLambda(0.0, 0.0, 0.0, len(X), holding_period_s, "no variation in OFI")

        lambda_ = float(np.dot(X_dm, Y_dm) / np.dot(X_dm, X_dm))

        # R^2
        y_pred = lambda_ * X_dm + np.mean(Y)
        ss_res = np.sum((Y - y_pred)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # T-stat
        n = len(X)
        se = math.sqrt(ss_res / (n - 2) / np.dot(X_dm, X_dm)) if n > 2 and np.dot(X_dm, X_dm) > 0 else 1.0
        t_stat = lambda_ / se if se > 0 else 0.0

        if abs(lambda_) < 1e-6:
            interp = "effectively zero price impact"
        elif lambda_ > 0:
            interp = f"1 unit OFI moves price by {lambda_*100:.4f}% (buy pressure → price up)"
        else:
            interp = f"negative lambda: unusual, check data quality"

        return KyleLambda(
            lambda_=lambda_,
            r_squared=max(0.0, r2),
            t_stat=t_stat,
            n_obs=n,
            holding_period_s=holding_period_s,
            interpretation=interp,
        )


# ---------------------------------------------------------------------------
# Amihud illiquidity ratio
# ---------------------------------------------------------------------------

class AmihudIlliquidityCalculator:
    """
    Amihud (2002): ILLIQ_t = |r_t| / Vol_t
    averaged over recent periods.
    High ILLIQ = prices move more per unit volume = illiquid.
    """

    def compute(
        self,
        returns: List[float],
        volumes: List[float],
        window: int = 21,
        hist_window: int = 252,
    ) -> AmihudIlliquidity:
        if not returns or not volumes or len(returns) != len(volumes):
            return AmihudIlliquidity(0.0, 0.0, 50.0, "normal")

        daily = [
            abs(r) / v if v > 0 else 0.0
            for r, v in zip(returns, volumes)
        ]

        current = float(np.mean(daily[-window:])) if len(daily) >= window else float(np.mean(daily))
        hist = daily[-hist_window:]
        annualized = current * 252

        below = sum(1 for d in hist if d < current)
        percentile = 100.0 * below / len(hist) if hist else 50.0

        if percentile >= 80:
            regime = "illiquid"
        elif percentile <= 20:
            regime = "liquid"
        else:
            regime = "normal"

        return AmihudIlliquidity(
            ratio=current,
            annualized=annualized,
            percentile=percentile,
            regime=regime,
            history=daily[-60:],
        )

    def rolling_amihud(
        self,
        returns: List[float],
        volumes: List[float],
        window: int = 21,
    ) -> List[float]:
        results = []
        for i in range(len(returns)):
            lo = max(0, i - window + 1)
            segment_r = returns[lo:i+1]
            segment_v = volumes[lo:i+1]
            daily = [abs(r)/v if v>0 else 0 for r,v in zip(segment_r, segment_v)]
            results.append(float(np.mean(daily)) if daily else 0.0)
        return results


# ---------------------------------------------------------------------------
# OFI signal aggregator
# ---------------------------------------------------------------------------

class OFISignalAggregator:
    """Combines multiple OFI measures into a composite signal."""

    def __init__(self, window: int = 200):
        self.window = window
        self.sv_ofi = SignedVolumeOFI()
        self.kyle_est = KyleLambdaEstimator()
        self.amihud_calc = AmihudIlliquidityCalculator()
        self._ofi_history: List[float] = []

    def compute(self, classified_trades: List[ClassifiedTrade]) -> OFISignal:
        if not classified_trades:
            return OFISignal(
                timestamp=datetime.now(timezone.utc),
                ofi_raw=0, ofi_normalized=0, ofi_ma=0,
                signed_vol_imbalance=0, price_impact_estimate=0,
                direction="neutral", kyle_lambda=0, amihud_ratio=0,
            )

        # Signed volume OFI
        rolling = self.sv_ofi.rolling_ofi(classified_trades, self.window)
        ofi_raw = rolling[-1] if rolling else 0.0

        self._ofi_history.append(ofi_raw)
        if len(self._ofi_history) > 500:
            self._ofi_history.pop(0)

        arr = np.array(self._ofi_history)
        ofi_z = float((ofi_raw - np.mean(arr)) / np.std(arr)) if len(arr) > 5 and np.std(arr) > 0 else 0.0
        ofi_ma = float(np.mean(self._ofi_history[-20:])) if len(self._ofi_history) >= 20 else ofi_raw

        # OFI events for Kyle's lambda
        events = self.sv_ofi.compute_interval(classified_trades, window=50)
        prices = [ct.trade.price for ct in classified_trades[::50]] if classified_trades else []
        kyle = self.kyle_est.estimate(events, prices)

        # Amihud
        prices_all = [ct.trade.price for ct in classified_trades]
        returns_all = []
        for i in range(1, len(prices_all)):
            r = (prices_all[i] - prices_all[i-1]) / prices_all[i-1] if prices_all[i-1] != 0 else 0.0
            returns_all.append(r)
        volumes_all = [ct.trade.size for ct in classified_trades[1:]]
        amihud = self.amihud_calc.compute(returns_all, volumes_all)

        if ofi_z > 1.5:
            direction = "buy_pressure"
        elif ofi_z < -1.5:
            direction = "sell_pressure"
        else:
            direction = "neutral"

        price_impact_est = abs(ofi_raw) * kyle.lambda_ * classified_trades[-1].trade.price if classified_trades else 0.0

        return OFISignal(
            timestamp=datetime.now(timezone.utc),
            ofi_raw=ofi_raw,
            ofi_normalized=ofi_z,
            ofi_ma=ofi_ma,
            signed_vol_imbalance=ofi_raw,
            price_impact_estimate=price_impact_est,
            direction=direction,
            kyle_lambda=kyle.lambda_,
            amihud_ratio=amihud.ratio,
        )


# ---------------------------------------------------------------------------
# Main OFI analytics facade
# ---------------------------------------------------------------------------

class OrderFlowImbalanceAnalytics:
    """Unified API for OFI analysis."""

    def __init__(self):
        from trade_classification import TradeClassificationEngine, ClassificationMethod
        self.engine = TradeClassificationEngine()
        self.method = ClassificationMethod.LEE_READY
        self.cont_ofi = ContOFICalculator()
        self.sv_ofi = SignedVolumeOFI()
        self.kyle_est = KyleLambdaEstimator()
        self.amihud_calc = AmihudIlliquidityCalculator()
        self.aggregator = OFISignalAggregator()

    def analyze(self, trades: List[Trade]) -> Dict:
        classified = self.engine.classify(trades, self.method)
        signal = self.aggregator.compute(classified)

        # Kyle's lambda
        events = self.sv_ofi.compute_interval(classified, 50)
        prices = [ct.trade.price for ct in classified[::50]]
        kyle = self.kyle_est.estimate(events, prices)

        # Amihud
        prices_all = [t.price for t in trades]
        returns = [(prices_all[i]-prices_all[i-1])/prices_all[i-1] for i in range(1, len(prices_all)) if prices_all[i-1]!=0]
        vols = [t.size for t in trades[1:]]
        amihud = self.amihud_calc.compute(returns, vols)

        return {
            "ofi": {
                "raw": signal.ofi_raw,
                "z_score": signal.ofi_normalized,
                "ma_20": signal.ofi_ma,
                "direction": signal.direction,
            },
            "kyle_lambda": {
                "lambda": kyle.lambda_,
                "r_squared": kyle.r_squared,
                "t_stat": kyle.t_stat,
                "interpretation": kyle.interpretation,
            },
            "amihud": {
                "ratio": amihud.ratio,
                "percentile": amihud.percentile,
                "regime": amihud.regime,
            },
            "signed_vol": signal.signed_vol_imbalance,
            "price_impact_est": signal.price_impact_estimate,
        }

    def format_report(self, trades: List[Trade]) -> str:
        data = self.analyze(trades)
        ofi = data["ofi"]
        kyle = data["kyle_lambda"]
        amihud = data["amihud"]

        lines = [
            "=== Order Flow Imbalance Report ===",
            f"Trades analyzed: {len(trades)}",
            "",
            "OFI (Signed Volume):",
            f"  Raw OFI:     {ofi['raw']:+.4f}",
            f"  Z-score:     {ofi['z_score']:+.2f}",
            f"  MA (20):     {ofi['ma_20']:+.4f}",
            f"  Direction:   {ofi['direction'].replace('_',' ').upper()}",
            "",
            "Kyle's Lambda (Price Impact):",
            f"  Lambda:      {kyle['lambda']:.6f}",
            f"  R^2:         {kyle['r_squared']:.3f}",
            f"  T-stat:      {kyle['t_stat']:.2f}",
            f"  Interpretation: {kyle['interpretation']}",
            "",
            "Amihud Illiquidity:",
            f"  Ratio:       {amihud['ratio']:.6f}",
            f"  Percentile:  {amihud['percentile']:.0f}th",
            f"  Regime:      {amihud['regime'].upper()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic LOB generator
# ---------------------------------------------------------------------------

class SyntheticLOBGenerator:
    """Generates synthetic LOB snapshots for testing ContOFI."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate(self, n: int = 200, mid: float = 100.0, n_levels: int = 5) -> List[LOBSnapshot]:
        snaps = []
        price = mid
        for i in range(n):
            price += self.rng.normal(0, 0.01)
            spread = 0.02
            bids = [(price - spread/2 - j*0.01, self.rng.uniform(50, 500)) for j in range(n_levels)]
            asks = [(price + spread/2 + j*0.01, self.rng.uniform(50, 500)) for j in range(n_levels)]
            snaps.append(LOBSnapshot(
                timestamp=datetime.now(timezone.utc),
                bids=bids, asks=asks, n_levels=n_levels,
            ))
        return snaps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OFI analytics CLI")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--action", choices=["report", "kyle", "amihud"], default="report")
    args = parser.parse_args()

    gen = SyntheticTradeGenerator()
    trades = gen.generate(args.n)
    analytics = OrderFlowImbalanceAnalytics()

    if args.action == "report":
        print(analytics.format_report(trades))
    elif args.action == "kyle":
        from trade_classification import TradeClassificationEngine, ClassificationMethod
        engine = TradeClassificationEngine()
        classified = engine.classify(trades)
        sv = SignedVolumeOFI()
        events = sv.compute_interval(classified, 50)
        prices = [ct.trade.price for ct in classified[::50]]
        kyle = analytics.kyle_est.estimate(events, prices)
        print(f"Kyle's Lambda: {kyle.lambda_:.6f}")
        print(f"R-squared: {kyle.r_squared:.3f}")
        print(f"T-stat: {kyle.t_stat:.2f}")
        print(f"N observations: {kyle.n_obs}")
        print(kyle.interpretation)
    elif args.action == "amihud":
        prices = [t.price for t in trades]
        returns = [(prices[i]-prices[i-1])/prices[i-1] for i in range(1,len(prices)) if prices[i-1]!=0]
        vols = [t.size for t in trades[1:]]
        amihud = analytics.amihud_calc.compute(returns, vols)
        print(f"Amihud Ratio: {amihud.ratio:.6f}")
        print(f"Annualized:   {amihud.annualized:.4f}")
        print(f"Percentile:   {amihud.percentile:.0f}th")
        print(f"Regime:       {amihud.regime.upper()}")
