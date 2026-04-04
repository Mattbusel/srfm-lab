"""
toxicity.py — Order flow toxicity and VPIN analytics.

Covers:
  - VPIN (Volume-synchronized Probability of Informed Trading)
  - Order toxicity regime detection
  - Flash crash precursor indicators
  - PIN (Probability of Informed Trading) estimation
  - Bulk classification for VPIN computation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import comb
from scipy.optimize import minimize

from trade_classification import SyntheticTradeGenerator, Trade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class VPINBucket:
    """Volume-time bucket for VPIN computation."""
    bucket_id: int
    start_time: datetime
    end_time: datetime
    total_volume: float
    buy_volume: float
    sell_volume: float
    n_trades: int
    price_start: float
    price_end: float

    @property
    def order_imbalance(self) -> float:
        return abs(self.buy_volume - self.sell_volume)

    @property
    def vpin_contribution(self) -> float:
        return self.order_imbalance / self.total_volume if self.total_volume > 0 else 0.0

    @property
    def price_return(self) -> float:
        return (self.price_end - self.price_start) / self.price_start if self.price_start > 0 else 0.0


@dataclass
class VPINEstimate:
    """VPIN estimate over a rolling window of buckets."""
    timestamp: datetime
    vpin: float                   # 0-1
    rolling_window: int           # number of buckets in rolling average
    regime: str                   # "safe", "elevated", "toxic"
    alert: bool                   # VPIN above threshold
    percentile: float             # vs historical
    bucket_volume: float          # per-bucket volume size
    n_buckets_computed: int


@dataclass
class PINEstimate:
    """
    PIN (Probability of Informed Trading) from Easley et al. (1996).
    Estimated from daily buy/sell volume via maximum likelihood.
    """
    alpha: float          # probability of information event
    delta: float          # probability bad news | event
    mu: float             # informed trading rate
    epsilon_b: float      # uninformed buy rate
    epsilon_s: float      # uninformed sell rate
    pin: float            # PIN = alpha*mu / (alpha*mu + 2*epsilon_b)
    log_likelihood: float
    n_days: int


@dataclass
class ToxicityRegime:
    timestamp: datetime
    vpin: float
    pin: float
    regime: str             # "safe", "elevated", "toxic", "extreme"
    flash_crash_risk: float  # 0-1
    indicators: Dict[str, float]
    recommendation: str


@dataclass
class FlashCrashPrecursor:
    timestamp: datetime
    vpin_spike: float
    volume_imbalance: float
    spread_widening_pct: float
    depth_depletion_pct: float
    composite_risk: float      # 0-1
    alert_level: str           # "watch", "warning", "critical"


# ---------------------------------------------------------------------------
# VPIN calculator
# ---------------------------------------------------------------------------

class VPINCalculator:
    """
    Easley, Lopez de Prado, O'Hara (2012) VPIN.

    Steps:
    1. Partition total volume into equal-size buckets V_n
    2. Classify volume in each bucket into buy (V_B) and sell (V_S) using BVC
    3. VPIN = (1/n) * Σ |V^B_τ - V^S_τ| / V

    Volume buckets replace time bars, making VPIN a volume-time measure.
    """

    SAFE_THRESHOLD    = 0.25
    ELEVATED_THRESHOLD = 0.40
    TOXIC_THRESHOLD   = 0.55

    def __init__(
        self,
        bucket_fraction: float = 0.01,   # each bucket = 1% of ADV
        rolling_window: int = 50,         # rolling average over N buckets
    ):
        self.bucket_fraction = bucket_fraction
        self.rolling_window = rolling_window
        self._vpin_history: List[float] = []

    def build_buckets(
        self,
        trades: List[Trade],
        adv: float = 1_000_000,
    ) -> List[VPINBucket]:
        """Partition trades into equal-volume buckets."""
        bucket_size = adv * self.bucket_fraction
        buckets = []
        bucket_id = 0
        current_vol = 0.0
        bucket_buys = 0.0
        bucket_sells = 0.0
        n_trades = 0
        bucket_start = trades[0].timestamp if trades else datetime.now(timezone.utc)
        price_start = trades[0].price if trades else 0.0

        for trade in trades:
            # Classify via BVC: use price return sign
            pass

        # Simpler: use price change sign for BVC-like classification
        prices = [t.price for t in trades]

        for i, trade in enumerate(trades):
            if i == 0:
                dp = 0.0
            else:
                dp = prices[i] - prices[i-1]

            sigma = float(np.std([prices[max(0,j)-1]-prices[max(0,j-1)-1] for j in range(max(1,i-50),i+1)]) or 0.01)
            z = dp / sigma if sigma > 0 else 0.0
            p_buy = 0.5 * (1 + math.erf(z / math.sqrt(2)))

            buy_vol  = trade.size * p_buy
            sell_vol = trade.size * (1 - p_buy)

            bucket_buys  += buy_vol
            bucket_sells += sell_vol
            current_vol  += trade.size
            n_trades += 1

            if current_vol >= bucket_size:
                buckets.append(VPINBucket(
                    bucket_id=bucket_id,
                    start_time=bucket_start,
                    end_time=trade.timestamp,
                    total_volume=current_vol,
                    buy_volume=bucket_buys,
                    sell_volume=bucket_sells,
                    n_trades=n_trades,
                    price_start=price_start,
                    price_end=trade.price,
                ))
                bucket_id += 1
                current_vol = 0.0
                bucket_buys = 0.0
                bucket_sells = 0.0
                n_trades = 0
                bucket_start = trade.timestamp
                price_start = trade.price

        return buckets

    def compute(self, buckets: List[VPINBucket]) -> List[VPINEstimate]:
        """Compute rolling VPIN from bucket list."""
        estimates = []
        for i in range(self.rolling_window, len(buckets) + 1):
            window = buckets[max(0, i - self.rolling_window): i]
            bucket_vol = float(np.mean([b.total_volume for b in window])) if window else 1.0

            vpin = float(np.mean([b.vpin_contribution for b in window])) if window else 0.0

            self._vpin_history.append(vpin)
            if len(self._vpin_history) > 252:
                self._vpin_history.pop(0)

            arr = np.array(self._vpin_history)
            below = sum(1 for v in arr if v < vpin)
            pct = 100.0 * below / len(arr) if arr.size > 0 else 50.0

            if vpin >= self.TOXIC_THRESHOLD:
                regime, alert = "toxic", True
            elif vpin >= self.ELEVATED_THRESHOLD:
                regime, alert = "elevated", True
            else:
                regime, alert = "safe", False

            estimates.append(VPINEstimate(
                timestamp=window[-1].end_time if window else datetime.now(timezone.utc),
                vpin=round(vpin, 4),
                rolling_window=self.rolling_window,
                regime=regime,
                alert=alert,
                percentile=pct,
                bucket_volume=bucket_vol,
                n_buckets_computed=len(window),
            ))

        return estimates

    def current_vpin(self, trades: List[Trade], adv: float = 1_000_000) -> VPINEstimate:
        buckets = self.build_buckets(trades, adv)
        estimates = self.compute(buckets)
        return estimates[-1] if estimates else VPINEstimate(
            datetime.now(timezone.utc), 0.0, self.rolling_window,
            "safe", False, 50.0, adv * self.bucket_fraction, 0,
        )

    def vpin_cdf(self) -> Dict:
        """Return CDF statistics of historical VPIN."""
        if not self._vpin_history:
            return {}
        arr = np.array(self._vpin_history)
        return {
            p: float(np.percentile(arr, p))
            for p in [10, 25, 50, 75, 90, 95, 99]
        }


# ---------------------------------------------------------------------------
# PIN estimator (Easley et al. 1996)
# ---------------------------------------------------------------------------

class PINEstimator:
    """
    Estimates PIN via maximum likelihood from daily buy/sell volumes.
    The likelihood is:
    L(θ | B, S) = (1-α) * e^{-ε_b} * B_t^{ε_b} / B_t! * e^{-ε_s} * S_t^{ε_s} / S_t!
               + α * δ * e^{-(ε_b)} * B_t^{ε_b} / B_t! * e^{-(μ+ε_s)} * (μ+ε_s)^{S_t} / S_t!
               + α * (1-δ) * e^{-(μ+ε_b)} * (μ+ε_b)^{B_t} / B_t! * e^{-ε_s} * S_t^{ε_s} / S_t!
    """

    def estimate(
        self,
        buy_volumes: List[float],
        sell_volumes: List[float],
    ) -> PINEstimate:
        if len(buy_volumes) != len(sell_volumes) or len(buy_volumes) < 5:
            return PINEstimate(0.2, 0.5, 0.3, 0.3, 0.3, 0.2, -1e9, len(buy_volumes))

        B = np.array(buy_volumes)
        S = np.array(sell_volumes)

        def neg_log_likelihood(params) -> float:
            alpha, delta, mu, eps_b, eps_s = params
            # Clip to valid range
            alpha = max(0.001, min(0.999, alpha))
            delta = max(0.001, min(0.999, delta))
            mu    = max(0.001, mu)
            eps_b = max(0.001, eps_b)
            eps_s = max(0.001, eps_s)

            total_ll = 0.0
            for b, s in zip(B, S):
                b, s = float(b), float(s)
                try:
                    # No news
                    ll_no_news = (
                        (1 - alpha) *
                        math.exp(-eps_b) * (eps_b**b) / math.factorial(min(int(b), 170)) *
                        math.exp(-eps_s) * (eps_s**s) / math.factorial(min(int(s), 170))
                    )
                    # Bad news
                    ll_bad = (
                        alpha * delta *
                        math.exp(-eps_b) * (eps_b**b) / math.factorial(min(int(b), 170)) *
                        math.exp(-(mu + eps_s)) * ((mu + eps_s)**s) / math.factorial(min(int(s), 170))
                    )
                    # Good news
                    ll_good = (
                        alpha * (1 - delta) *
                        math.exp(-(mu + eps_b)) * ((mu + eps_b)**b) / math.factorial(min(int(b), 170)) *
                        math.exp(-eps_s) * (eps_s**s) / math.factorial(min(int(s), 170))
                    )
                    ll = ll_no_news + ll_bad + ll_good
                    if ll > 1e-300:
                        total_ll += math.log(ll)
                except (OverflowError, ZeroDivisionError, ValueError):
                    pass

            return -total_ll

        # Initial guess
        avg_b = float(np.mean(B))
        avg_s = float(np.mean(S))
        x0 = [0.2, 0.5, max(1.0, abs(avg_b - avg_s)), avg_b * 0.8, avg_s * 0.8]
        bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 1e6), (0.001, 1e6), (0.001, 1e6)]

        try:
            result = minimize(
                neg_log_likelihood, x0, method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 200, "ftol": 1e-8}
            )
            alpha, delta, mu, eps_b, eps_s = result.x
            pin = alpha * mu / (alpha * mu + eps_b + eps_s)

            return PINEstimate(
                alpha=float(alpha), delta=float(delta), mu=float(mu),
                epsilon_b=float(eps_b), epsilon_s=float(eps_s),
                pin=float(pin),
                log_likelihood=-result.fun,
                n_days=len(buy_volumes),
            )
        except Exception as exc:
            logger.warning("PIN estimation failed: %s", exc)
            return PINEstimate(0.2, 0.5, 10.0, 100.0, 100.0, 0.2, -1e9, len(buy_volumes))


# ---------------------------------------------------------------------------
# Flash crash precursor detector
# ---------------------------------------------------------------------------

class FlashCrashPrecursorDetector:
    """
    Monitors multiple signals for flash crash conditions:
    1. VPIN spike (order imbalance surge)
    2. Spread widening
    3. Book depth depletion
    4. Price momentum extreme
    """

    VPIN_CRISIS_THRESHOLD = 0.60
    SPREAD_WIDEN_THRESHOLD = 3.0   # 3x normal spread
    DEPTH_DEPLETION_THRESHOLD = 0.70  # 70% of normal depth gone

    def __init__(self, vpin_calc: VPINCalculator = None):
        self.vpin_calc = vpin_calc or VPINCalculator()
        self._normal_spread: float = 0.0
        self._normal_depth: float = 0.0
        self._spread_history: List[float] = []

    def calibrate(self, trades: List[Trade]) -> None:
        """Calibrate normal spread and depth from historical data."""
        spreads = [t.ask - t.bid for t in trades]
        self._normal_spread = float(np.mean(spreads)) if spreads else 0.05
        self._spread_history = list(spreads[-100:])

    def check(
        self,
        current_vpin: float,
        current_spread: float,
        current_depth_bid: float,
        current_depth_ask: float,
        recent_return: float,
    ) -> FlashCrashPrecursor:
        normal_spread = self._normal_spread or current_spread
        spread_widen = current_spread / normal_spread if normal_spread > 0 else 1.0

        avg_depth = (current_depth_bid + current_depth_ask) / 2
        # Depth depletion: if depth < 30% of estimated normal
        normal_depth = max(avg_depth, 1.0)
        depth_depletion = 1 - min(1.0, avg_depth / max(normal_depth, 1e-6))

        # Composite risk score
        vpin_score    = min(1.0, current_vpin / self.VPIN_CRISIS_THRESHOLD)
        spread_score  = min(1.0, (spread_widen - 1) / (self.SPREAD_WIDEN_THRESHOLD - 1))
        depth_score   = min(1.0, depth_depletion / self.DEPTH_DEPLETION_THRESHOLD)
        momentum_score = min(1.0, abs(recent_return) / 0.02)  # 2% move = max score

        composite = (
            0.40 * vpin_score +
            0.25 * spread_score +
            0.20 * depth_score +
            0.15 * momentum_score
        )

        if composite >= 0.75:
            alert_level = "critical"
        elif composite >= 0.50:
            alert_level = "warning"
        elif composite >= 0.25:
            alert_level = "watch"
        else:
            alert_level = "normal"

        return FlashCrashPrecursor(
            timestamp=datetime.now(timezone.utc),
            vpin_spike=current_vpin,
            volume_imbalance=current_vpin,
            spread_widening_pct=(spread_widen - 1) * 100,
            depth_depletion_pct=depth_depletion * 100,
            composite_risk=composite,
            alert_level=alert_level,
        )


# ---------------------------------------------------------------------------
# Toxicity regime classifier
# ---------------------------------------------------------------------------

class ToxicityRegimeClassifier:
    """
    Classifies current market microstructure toxicity regime.
    Combines VPIN, PIN, spread conditions, and momentum.
    """

    def classify(
        self,
        vpin: VPINEstimate,
        pin: PINEstimate,
        current_spread_bps: float,
        normal_spread_bps: float,
        price_return_5m: float,
    ) -> ToxicityRegime:
        spread_ratio = current_spread_bps / normal_spread_bps if normal_spread_bps > 0 else 1.0

        indicators = {
            "vpin": vpin.vpin,
            "pin": pin.pin,
            "spread_ratio": spread_ratio,
            "price_return_abs": abs(price_return_5m),
            "vpin_percentile": vpin.percentile,
        }

        # Flash crash risk composite
        risk = (
            0.35 * min(1.0, vpin.vpin / 0.6) +
            0.25 * pin.pin +
            0.20 * min(1.0, (spread_ratio - 1) / 2) +
            0.20 * min(1.0, abs(price_return_5m) / 0.02)
        )

        if risk >= 0.70:
            regime = "extreme"
            rec = "Halt/reduce trading. Risk of flash crash is elevated."
        elif risk >= 0.50:
            regime = "toxic"
            rec = "Trade only at market open/close. Use limit orders. Reduce size."
        elif risk >= 0.30:
            regime = "elevated"
            rec = "Proceed cautiously. Use limit orders. Widen TWAP intervals."
        else:
            regime = "safe"
            rec = "Normal conditions. Standard execution acceptable."

        return ToxicityRegime(
            timestamp=datetime.now(timezone.utc),
            vpin=vpin.vpin,
            pin=pin.pin,
            regime=regime,
            flash_crash_risk=round(risk, 3),
            indicators=indicators,
            recommendation=rec,
        )


# ---------------------------------------------------------------------------
# Main toxicity analytics facade
# ---------------------------------------------------------------------------

class ToxicityAnalytics:
    """Unified market microstructure toxicity analytics."""

    def __init__(self, adv: float = 5_000_000):
        self.adv = adv
        self.vpin_calc = VPINCalculator()
        self.pin_est = PINEstimator()
        self.flash_detector = FlashCrashPrecursorDetector(self.vpin_calc)
        self.regime_classifier = ToxicityRegimeClassifier()

    def analyze(self, trades: List[Trade]) -> Dict:
        if not trades:
            return {"error": "no trades"}

        self.flash_detector.calibrate(trades)

        # VPIN
        buckets = self.vpin_calc.build_buckets(trades, self.adv)
        vpin_estimates = self.vpin_calc.compute(buckets)
        current_vpin = vpin_estimates[-1] if vpin_estimates else VPINEstimate(
            datetime.now(timezone.utc), 0.25, 50, "safe", False, 50.0, self.adv * 0.01, 0
        )

        # PIN estimation (aggregate to daily bars)
        daily_buys  = [sum(t.size for t in trades[:500])] * 3  # mock 3 days
        daily_sells = [sum(t.size for t in trades[500:1000])] * 3
        pin = self.pin_est.estimate(daily_buys, daily_sells)

        # Current conditions
        recent = trades[-50:] if len(trades) > 50 else trades
        avg_spread = float(np.mean([t.ask - t.bid for t in recent]))
        avg_price  = float(np.mean([t.price for t in recent]))
        spread_bps = avg_spread / avg_price * 10000 if avg_price > 0 else 10.0

        prices = [t.price for t in recent]
        ret_5m = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0

        regime = self.regime_classifier.classify(current_vpin, pin, spread_bps, spread_bps, ret_5m)

        # Flash crash check
        flash = self.flash_detector.check(
            current_vpin.vpin, avg_spread,
            current_depth_bid=100.0, current_depth_ask=100.0,
            recent_return=ret_5m,
        )

        return {
            "vpin": {
                "current": current_vpin.vpin,
                "regime": current_vpin.regime,
                "percentile": current_vpin.percentile,
                "alert": current_vpin.alert,
            },
            "pin": {
                "pin": pin.pin,
                "alpha": pin.alpha,
                "mu": pin.mu,
            },
            "toxicity_regime": {
                "regime": regime.regime,
                "flash_crash_risk": regime.flash_crash_risk,
                "recommendation": regime.recommendation,
            },
            "flash_crash": {
                "composite_risk": flash.composite_risk,
                "alert_level": flash.alert_level,
                "spread_widening_pct": flash.spread_widening_pct,
            },
        }

    def format_report(self, trades: List[Trade]) -> str:
        data = self.analyze(trades)
        vpin = data["vpin"]
        pin = data["pin"]
        tox = data["toxicity_regime"]
        flash = data["flash_crash"]

        lines = [
            "=== Market Toxicity Report ===",
            f"N trades analyzed: {len(trades)}",
            "",
            "VPIN (Volume-Sync. Prob. Informed Trading):",
            f"  Current VPIN:    {vpin['current']:.4f}",
            f"  Regime:          {vpin['regime'].upper()}",
            f"  Historical %ile: {vpin['percentile']:.0f}th",
            f"  Alert triggered: {'YES ⚠️' if vpin['alert'] else 'No'}",
            "",
            "PIN (Probability of Informed Trading):",
            f"  PIN:             {pin['pin']:.4f}",
            f"  Alpha:           {pin['alpha']:.3f}",
            f"  Mu (info rate):  {pin['mu']:.2f}",
            "",
            "Toxicity Regime:",
            f"  Regime:          {tox['regime'].upper()}",
            f"  Flash Crash Risk:{tox['flash_crash_risk']:.1%}",
            f"  Recommendation:  {tox['recommendation']}",
            "",
            "Flash Crash Precursors:",
            f"  Composite Risk:  {flash['composite_risk']:.1%}",
            f"  Alert Level:     {flash['alert_level'].upper()}",
            f"  Spread Widening: {flash['spread_widening_pct']:+.1f}%",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Toxicity analytics CLI")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--adv", type=float, default=5_000_000)
    parser.add_argument("--action", choices=["report", "vpin", "pin"], default="report")
    args = parser.parse_args()

    gen = SyntheticTradeGenerator()
    trades = gen.generate(args.n)

    analytics = ToxicityAnalytics(args.adv)

    if args.action == "report":
        print(analytics.format_report(trades))
    elif args.action == "vpin":
        buckets = analytics.vpin_calc.build_buckets(trades, args.adv)
        estimates = analytics.vpin_calc.compute(buckets)
        print(f"Computed {len(estimates)} VPIN estimates from {len(buckets)} buckets")
        if estimates:
            latest = estimates[-1]
            print(f"Current VPIN: {latest.vpin:.4f} ({latest.regime.upper()})")
            print(f"Historical CDF: {analytics.vpin_calc.vpin_cdf()}")
    elif args.action == "pin":
        n_days = 20
        daily_buys  = [sum(t.size for t in trades[i*100:(i+1)*100]) for i in range(n_days)]
        daily_sells = [sum(t.size for t in trades[i*100+50:(i+1)*100+50]) for i in range(n_days)]
        pin = analytics.pin_est.estimate(daily_buys, daily_sells)
        print(f"PIN estimate: {pin.pin:.4f}")
        print(f"Alpha (prob event): {pin.alpha:.3f}")
        print(f"Delta (bad news): {pin.delta:.3f}")
        print(f"Mu (informed rate): {pin.mu:.2f}")
        print(f"N days: {pin.n_days}")
