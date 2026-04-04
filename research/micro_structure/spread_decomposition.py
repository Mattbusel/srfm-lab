"""
spread_decomposition.py — Bid-ask spread component decomposition.

Covers:
  - Adverse selection component (Glosten-Milgrom model)
  - Inventory component (Ho-Stoll model)
  - Order processing component
  - Stoll (1989) decomposition
  - Huang-Stoll (1997) VAR method
  - Spread dynamics over time and market conditions
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
class SpreadComponents:
    """Bid-ask spread decomposed into three economic components."""
    ticker: str
    total_spread_bps: float
    # Glosten-Milgrom: adverse selection
    adverse_selection_bps: float
    adverse_selection_pct: float      # fraction of spread
    # Ho-Stoll: inventory holding cost
    inventory_bps: float
    inventory_pct: float
    # Order processing (exchange fees, dealer cost)
    order_processing_bps: float
    order_processing_pct: float
    # Metadata
    n_trades: int
    method: str


@dataclass
class GlostenMilgromParameters:
    """Parameters of the Glosten-Milgrom (1985) sequential trade model."""
    alpha: float         # probability of informed trading (0-1)
    delta: float         # information content per trade
    spread_half: float   # half-spread in price units
    lambda_: float       # adverse selection component of half-spread


@dataclass
class HoStollParameters:
    """Parameters of the Ho-Stoll (1981) inventory model."""
    beta: float          # inventory cost coefficient
    phi: float           # order processing cost per unit
    gamma: float         # serial correlation of trade signs
    pi_star: float       # optimal inventory target


@dataclass
class HuangStollDecomposition:
    """Huang-Stoll (1997) VAR-based spread decomposition."""
    lambda_: float           # adverse selection fraction of half-spread
    theta: float             # order processing fraction
    inventory_pct: float     # 1 - lambda_ - theta
    half_spread: float
    r_squared: float
    n_obs: int


# ---------------------------------------------------------------------------
# Glosten-Milgrom adverse selection estimator
# ---------------------------------------------------------------------------

class GlostenMilgromEstimator:
    """
    Estimates the adverse selection component using a simplified GM model.
    The adverse selection component measures how much spread compensation
    is required for the risk of trading with an informed agent.

    Estimation via Glosten-Harris (1988):
    p_t - p_{t-1} = λ * Q_t + (c/2) * (Q_t - Q_{t-1}) + ε
    where:
      λ = permanent price impact (adverse selection)
      c = transitory component (inventory + processing)
      Q_t = trade sign (+1 buy, -1 sell)
    """

    def estimate(
        self,
        classified: List[ClassifiedTrade],
    ) -> GlostenMilgromParameters:
        n = len(classified)
        if n < 20:
            return GlostenMilgromParameters(0.2, 0.01, 0.0, 0.0)

        prices = [ct.trade.price for ct in classified]
        signs  = [1 if ct.side == TradeSide.BUY else -1 for ct in classified]

        dp = [prices[i] - prices[i-1] for i in range(1, n)]
        dq = [signs[i] - signs[i-1] for i in range(1, n)]
        q  = signs[1:]

        # OLS: dp = lambda*Q + c/2*dQ
        X = np.column_stack([q, dq])
        Y = np.array(dp)

        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            lambda_ = float(beta[0])  # adverse selection (permanent)
            c_half  = float(beta[1])  # transitory half

            avg_price = float(np.mean(prices))
            spread_half = abs(classified[0].trade.ask - classified[0].trade.bid) / 2 if classified else 0.01

            alpha = min(1.0, max(0.0, abs(lambda_) / (abs(lambda_) + abs(c_half)) if abs(lambda_) + abs(c_half) > 0 else 0.2))

            return GlostenMilgromParameters(
                alpha=alpha,
                delta=abs(lambda_),
                spread_half=spread_half,
                lambda_=abs(lambda_),
            )
        except Exception as exc:
            logger.warning("GM estimation failed: %s", exc)
            return GlostenMilgromParameters(0.2, 0.01, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Huang-Stoll (1997) decomposition
# ---------------------------------------------------------------------------

class HuangStollEstimator:
    """
    Huang and Stoll (1997) decompose the spread into three components using
    a VAR model of trade direction and quote changes.

    Model:
    Q_t = Q_{t-1} + u_t
    ΔM_t = (S/2) * (α + β) * Q_{t-1} - (S/2) * β * Q_{t-2} + e_t

    where S = spread, α = adverse selection fraction, β = inventory fraction
    """

    def estimate(
        self,
        classified: List[ClassifiedTrade],
    ) -> HuangStollDecomposition:
        n = len(classified)
        if n < 30:
            return HuangStollDecomposition(
                lambda_=0.20, theta=0.40, inventory_pct=0.40,
                half_spread=0.05, r_squared=0.0, n_obs=n,
            )

        prices = [ct.trade.price for ct in classified]
        signs  = np.array([1 if ct.side == TradeSide.BUY else -1 for ct in classified])
        spreads = [ct.trade.ask - ct.trade.bid for ct in classified]
        S = float(np.mean(spreads))
        half_S = S / 2

        # Mid-quote changes
        mids = [(ct.trade.bid + ct.trade.ask) / 2 for ct in classified]
        dM = np.array([mids[i] - mids[i-1] for i in range(1, n)])

        Q_prev = signs[1:-1]    # Q_{t-1}
        Q_prev2 = signs[:-2]    # Q_{t-2}
        dM_ = dM[1:]

        X = np.column_stack([Q_prev, Q_prev2])
        Y = dM_

        try:
            beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
            b1, b2 = float(beta_hat[0]), float(beta_hat[1])

            # b1 = (S/2)(alpha + beta), b2 = -(S/2)*beta
            beta = -b2 / half_S if half_S > 0 else 0.0
            alpha = (b1 / half_S - beta) if half_S > 0 else 0.0

            alpha = max(0.0, min(1.0, alpha))
            beta  = max(0.0, min(1.0 - alpha, beta))
            theta = 1.0 - alpha - beta

            y_pred = X @ beta_hat
            ss_res = float(np.sum((Y - y_pred)**2))
            ss_tot = float(np.sum((Y - np.mean(Y))**2))
            r2 = max(0.0, 1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

            return HuangStollDecomposition(
                lambda_=alpha,
                theta=theta,
                inventory_pct=beta,
                half_spread=half_S,
                r_squared=r2,
                n_obs=len(Y),
            )
        except Exception as exc:
            logger.warning("Huang-Stoll estimation failed: %s", exc)
            return HuangStollDecomposition(0.20, 0.40, 0.40, half_S, 0.0, n)


# ---------------------------------------------------------------------------
# Roll (1984) implied spread estimator
# ---------------------------------------------------------------------------

class RollImpliedSpreadEstimator:
    """
    Roll (1984): implied spread from negative serial covariance of price changes.
    S = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    """

    def estimate(self, prices: List[float]) -> Dict:
        if len(prices) < 20:
            return {"implied_spread": 0.0, "serial_covariance": 0.0, "r_squared": 0.0}

        dp = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        dp_arr = np.array(dp)
        n = len(dp_arr)

        # Lag-1 autocovariance
        cov = float(np.cov(dp_arr[:-1], dp_arr[1:])[0, 1])

        if cov >= 0:
            # Theory says cov should be negative; if positive, the Roll estimator breaks
            return {"implied_spread": 0.0, "serial_covariance": cov, "note": "positive serial cov — Roll inapplicable"}

        implied_spread = 2 * math.sqrt(-cov)

        return {
            "implied_spread": implied_spread,
            "serial_covariance": cov,
            "avg_price": float(np.mean(prices)),
            "implied_spread_bps": implied_spread / float(np.mean(prices)) * 10000,
        }


# ---------------------------------------------------------------------------
# Corwin-Schultz (2012) high-low spread estimator
# ---------------------------------------------------------------------------

class CorwinSchultzEstimator:
    """
    Corwin & Schultz (2012) estimate effective spread from daily high-low ranges.
    Uses the property that the high-low range reflects both volatility and spread.
    """

    def estimate(
        self,
        highs: List[float],
        lows: List[float],
    ) -> Dict:
        if len(highs) < 3 or len(lows) < 3 or len(highs) != len(lows):
            return {"spread_bps": 0.0, "alpha": 0.0, "beta": 0.0}

        spreads = []
        for i in range(1, len(highs)):
            h1, l1 = highs[i-1], lows[i-1]
            h2, l2 = highs[i], lows[i]

            if l1 <= 0 or l2 <= 0 or h1 <= 0 or h2 <= 0:
                continue

            beta = math.log(h1/l1)**2 + math.log(h2/l2)**2
            gamma = math.log(max(h1, h2) / min(l1, l2))**2

            alpha = (math.sqrt(2*beta) - math.sqrt(beta)) / (3 - 2*math.sqrt(2)) - math.sqrt(gamma / (3 - 2*math.sqrt(2)))

            if alpha > 0:
                s = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
                spreads.append(s)

        if not spreads:
            return {"spread_bps": 0.0}

        avg_spread = float(np.mean(spreads))
        avg_price = float(np.mean([(h+l)/2 for h, l in zip(highs, lows)]))

        return {
            "spread_estimate": avg_spread,
            "spread_bps": avg_spread / avg_price * 10000 if avg_price > 0 else 0.0,
            "n_periods": len(spreads),
        }


# ---------------------------------------------------------------------------
# Full spread decomposition
# ---------------------------------------------------------------------------

class SpreadDecompositor:
    """
    Combines multiple estimation methods to decompose the spread.
    """

    def __init__(self):
        self.gm_est = GlostenMilgromEstimator()
        self.hs_est = HuangStollEstimator()
        self.roll_est = RollImpliedSpreadEstimator()

    def decompose(
        self,
        classified: List[ClassifiedTrade],
        ticker: str = "UNKNOWN",
    ) -> SpreadComponents:
        if not classified:
            return SpreadComponents(ticker, 0, 0, 0, 0, 0, 0, 0, 0, "none")

        spreads = [ct.trade.ask - ct.trade.bid for ct in classified]
        prices  = [ct.trade.price for ct in classified]
        avg_spread = float(np.mean(spreads))
        avg_price  = float(np.mean(prices))
        total_bps  = avg_spread / avg_price * 10000 if avg_price > 0 else 0.0

        # Huang-Stoll is our primary method
        hs = self.hs_est.estimate(classified)

        adv_sel_pct   = max(0.0, min(1.0, hs.lambda_))
        inventory_pct = max(0.0, min(1.0 - adv_sel_pct, hs.inventory_pct))
        processing_pct = max(0.0, 1.0 - adv_sel_pct - inventory_pct)

        adv_sel_bps   = total_bps * adv_sel_pct
        inventory_bps = total_bps * inventory_pct
        processing_bps = total_bps * processing_pct

        return SpreadComponents(
            ticker=ticker,
            total_spread_bps=total_bps,
            adverse_selection_bps=adv_sel_bps,
            adverse_selection_pct=adv_sel_pct,
            inventory_bps=inventory_bps,
            inventory_pct=inventory_pct,
            order_processing_bps=processing_bps,
            order_processing_pct=processing_pct,
            n_trades=len(classified),
            method="huang_stoll",
        )

    def full_analysis(
        self,
        classified: List[ClassifiedTrade],
        ticker: str = "UNKNOWN",
    ) -> Dict:
        prices = [ct.trade.price for ct in classified]
        components = self.decompose(classified, ticker)
        hs = self.hs_est.estimate(classified)
        gm = self.gm_est.estimate(classified)
        roll = self.roll_est.estimate(prices)

        return {
            "ticker": ticker,
            "n_trades": len(classified),
            "avg_spread_bps": components.total_spread_bps,
            "components": {
                "adverse_selection": {
                    "bps": components.adverse_selection_bps,
                    "pct": components.adverse_selection_pct,
                },
                "inventory": {
                    "bps": components.inventory_bps,
                    "pct": components.inventory_pct,
                },
                "order_processing": {
                    "bps": components.order_processing_bps,
                    "pct": components.order_processing_pct,
                },
            },
            "huang_stoll": {
                "lambda": hs.lambda_,
                "theta": hs.theta,
                "inventory": hs.inventory_pct,
                "r_squared": hs.r_squared,
            },
            "glosten_milgrom": {
                "alpha": gm.alpha,
                "lambda": gm.lambda_,
            },
            "roll": roll,
        }

    def format_report(self, classified: List[ClassifiedTrade], ticker: str = "UNKNOWN") -> str:
        data = self.full_analysis(classified, ticker)
        c = data["components"]
        hs = data["huang_stoll"]
        roll = data["roll"]

        lines = [
            f"=== Spread Decomposition: {ticker} ===",
            f"N trades: {data['n_trades']}",
            f"Avg spread: {data['avg_spread_bps']:.2f} bps",
            "",
            "Component Breakdown (Huang-Stoll):",
            f"  Adverse selection: {c['adverse_selection']['bps']:.2f} bps ({c['adverse_selection']['pct']:.0%})",
            f"  Inventory:         {c['inventory']['bps']:.2f} bps ({c['inventory']['pct']:.0%})",
            f"  Order processing:  {c['order_processing']['bps']:.2f} bps ({c['order_processing']['pct']:.0%})",
            "",
            "Huang-Stoll VAR:",
            f"  Lambda (adv sel):  {hs['lambda']:.3f}",
            f"  Theta (processing):{hs['theta']:.3f}",
            f"  R-squared:         {hs['r_squared']:.3f}",
            "",
            "Roll Implied Spread:",
            f"  Implied spread:    {roll.get('implied_spread', 0):.4f}",
            f"  Implied spread bps:{roll.get('implied_spread_bps', 0):.2f}",
            f"  Serial covariance: {roll.get('serial_covariance', 0):.6f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spread decomposition CLI")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()

    gen = SyntheticTradeGenerator()
    trades = gen.generate(args.n)
    from trade_classification import TradeClassificationEngine
    engine = TradeClassificationEngine()
    classified = engine.classify(trades)

    decomp = SpreadDecompositor()
    print(decomp.format_report(classified, args.ticker))
