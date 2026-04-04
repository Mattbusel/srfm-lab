"""
price_impact.py — Price impact estimation.

Covers:
  - Temporary vs permanent price impact decomposition
  - Power-law impact model fitting (Almgren-Chriss / square-root law)
  - Cross-impact between correlated assets
  - Impact decay model
  - Execution cost estimation from impact
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from trade_classification import ClassifiedTrade, Trade, TradeSide, SyntheticTradeGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ImpactObservation:
    """Single price impact observation."""
    order_size: float        # shares / lots
    daily_volume: float      # ADV for normalization
    price_before: float
    price_after_temp: float  # price immediately after execution
    price_after_perm: float  # price after impact has decayed (e.g. 30min later)
    spread: float
    volatility: float        # daily vol
    side: str                # "buy" or "sell"

    @property
    def participation_rate(self) -> float:
        return self.order_size / self.daily_volume if self.daily_volume > 0 else 0.0

    @property
    def temporary_impact(self) -> float:
        """Temporary impact in bps."""
        if self.price_before == 0:
            return 0.0
        raw_impact = abs(self.price_after_temp - self.price_before) / self.price_before
        sign = 1 if self.side == "buy" else -1
        signed = (self.price_after_temp - self.price_before) / self.price_before * sign
        return signed * 10000   # in bps

    @property
    def permanent_impact(self) -> float:
        """Permanent impact in bps."""
        if self.price_before == 0:
            return 0.0
        sign = 1 if self.side == "buy" else -1
        signed = (self.price_after_perm - self.price_before) / self.price_before * sign
        return signed * 10000


@dataclass
class PowerLawModel:
    """
    Square-root / power-law impact model:
    I = η * σ * (Q / V)^α
    where:
      η = market impact coefficient
      σ = daily volatility
      Q = order size
      V = ADV (daily volume)
      α = power-law exponent (empirically ~0.5 for temp, ~0.3 for perm)
    """
    alpha: float             # exponent
    eta: float               # coefficient
    r_squared: float
    n_obs: int
    impact_type: str         # "temporary" or "permanent"

    def predict(self, order_size: float, adv: float, sigma: float) -> float:
        """Predict impact in bps."""
        prate = order_size / adv if adv > 0 else 0.0
        return self.eta * sigma * (prate ** self.alpha) * 10000

    def impact_for_participation(self, participation_rate: float, sigma: float = 0.02) -> float:
        return self.eta * sigma * (participation_rate ** self.alpha) * 10000


@dataclass
class ImpactDecomposition:
    observation: ImpactObservation
    temporary_bps: float
    permanent_bps: float
    total_bps: float
    temp_fraction: float     # temporary / total
    perm_fraction: float
    spread_cost_bps: float


@dataclass
class CrossImpactEstimate:
    """Impact of asset A's order flow on asset B's price."""
    asset_a: str
    asset_b: str
    cross_lambda: float      # cross-impact coefficient
    r_squared: float
    n_obs: int
    correlation: float       # return correlation
    significance: bool       # statistically significant


# ---------------------------------------------------------------------------
# Power-law impact model fitter
# ---------------------------------------------------------------------------

class PowerLawImpactFitter:
    """
    Fits I = η * σ * (Q/V)^α to empirical impact data.
    Uses log-linear regression: log(I) = log(η*σ) + α*log(Q/V)
    """

    def fit(
        self,
        observations: List[ImpactObservation],
        impact_type: str = "temporary",
    ) -> PowerLawModel:
        if len(observations) < 10:
            # Return typical market impact parameters
            return PowerLawModel(
                alpha=0.5, eta=0.314, r_squared=0.0, n_obs=0,
                impact_type=impact_type,
            )

        log_x = []
        log_y = []

        for obs in observations:
            prate = obs.participation_rate
            sigma = obs.volatility
            if prate <= 0 or sigma <= 0:
                continue

            if impact_type == "temporary":
                impact = obs.temporary_impact
            else:
                impact = obs.permanent_impact

            if impact <= 0:
                continue

            log_x.append(math.log(prate))
            log_y.append(math.log(impact / (sigma * 10000)) if sigma > 0 else 0.0)

        if len(log_x) < 5:
            return PowerLawModel(0.5, 0.314, 0.0, len(observations), impact_type)

        X = np.array(log_x)
        Y = np.array(log_y)

        # OLS in log-log space
        X_dm = X - np.mean(X)
        Y_dm = Y - np.mean(Y)
        alpha = float(np.dot(X_dm, Y_dm) / np.dot(X_dm, X_dm)) if np.dot(X_dm, X_dm) != 0 else 0.5
        log_eta = np.mean(Y) - alpha * np.mean(X)
        eta = math.exp(log_eta)

        y_pred = alpha * X_dm + np.mean(Y)
        ss_res = float(np.sum((Y - y_pred)**2))
        ss_tot = float(np.sum(Y_dm**2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return PowerLawModel(
            alpha=round(alpha, 4),
            eta=round(eta, 4),
            r_squared=max(0.0, r2),
            n_obs=len(log_x),
            impact_type=impact_type,
        )


# ---------------------------------------------------------------------------
# Temporary vs permanent impact decompostor
# ---------------------------------------------------------------------------

class ImpactDecompositor:
    """
    Separates price impact into temporary (market maker inventory) and
    permanent (information) components.

    Method: Hasbrouck (1991) VAR decomposition or simpler:
    - Temporary impact reverses: trades at t affect prices at t but mean-revert
    - Permanent impact persists: price adjustment for new information
    """

    def __init__(self, decay_window: int = 30):
        self.decay_window = decay_window

    def decompose_from_trades(
        self,
        classified: List[ClassifiedTrade],
        price_series: List[float],
        adv: float = 1_000_000,
        sigma: float = 0.02,
    ) -> List[ImpactDecomposition]:
        """
        Estimate impact decomposition from trade + price series.
        """
        if len(classified) < self.decay_window * 2:
            return []

        results = []
        for i in range(self.decay_window, len(classified) - self.decay_window):
            ct = classified[i]
            if ct.trade.size < adv * 0.001:   # skip tiny trades
                continue

            p_before = price_series[i] if i < len(price_series) else ct.trade.price
            p_after_temp = price_series[i + 1] if i + 1 < len(price_series) else p_before
            p_after_perm = price_series[i + self.decay_window] if i + self.decay_window < len(price_series) else p_before

            spread = ct.trade.ask - ct.trade.bid
            obs = ImpactObservation(
                order_size=ct.trade.size,
                daily_volume=adv,
                price_before=p_before,
                price_after_temp=p_after_temp,
                price_after_perm=p_after_perm,
                spread=spread,
                volatility=sigma,
                side="buy" if ct.side == TradeSide.BUY else "sell",
            )

            temp_bps = obs.temporary_impact
            perm_bps = obs.permanent_impact
            total = abs(temp_bps) + abs(perm_bps)

            results.append(ImpactDecomposition(
                observation=obs,
                temporary_bps=temp_bps,
                permanent_bps=perm_bps,
                total_bps=total,
                temp_fraction=abs(temp_bps) / total if total > 0 else 0.5,
                perm_fraction=abs(perm_bps) / total if total > 0 else 0.5,
                spread_cost_bps=spread / p_before * 5000 if p_before > 0 else 0.0,
            ))

        return results

    def aggregate_stats(self, decompositions: List[ImpactDecomposition]) -> Dict:
        if not decompositions:
            return {}
        temps = [d.temporary_bps for d in decompositions]
        perms = [d.permanent_bps for d in decompositions]
        totals = [d.total_bps for d in decompositions]
        return {
            "n_obs": len(decompositions),
            "avg_temp_bps": float(np.mean(temps)),
            "avg_perm_bps": float(np.mean(perms)),
            "avg_total_bps": float(np.mean(totals)),
            "avg_temp_fraction": float(np.mean([d.temp_fraction for d in decompositions])),
            "avg_spread_bps": float(np.mean([d.spread_cost_bps for d in decompositions])),
        }


# ---------------------------------------------------------------------------
# Impact decay model
# ---------------------------------------------------------------------------

class ImpactDecayModel:
    """
    Models how price impact decays over time after order execution.
    Typical shapes: exponential, power-law, or linear.
    """

    def exponential_decay(
        self,
        initial_impact: float,
        decay_rate: float,
        t: int,
    ) -> float:
        """I(t) = I_0 * exp(-decay_rate * t)"""
        return initial_impact * math.exp(-decay_rate * t)

    def power_law_decay(
        self,
        initial_impact: float,
        decay_exponent: float,
        t: int,
    ) -> float:
        """I(t) = I_0 / (1 + t)^exponent"""
        return initial_impact / (1 + t) ** decay_exponent

    def fit_decay(
        self,
        impact_over_time: List[float],
        model: str = "exponential",
    ) -> Dict:
        """Fit decay parameters from observed impact time series."""
        if len(impact_over_time) < 5:
            return {"model": model, "rate": 0.1, "r2": 0.0}

        t = np.arange(len(impact_over_time))
        I0 = abs(impact_over_time[0])
        if I0 == 0:
            return {"model": model, "rate": 0.0, "r2": 0.0}

        # Normalize
        y = np.array([max(0.0, i / I0) for i in impact_over_time])

        try:
            if model == "exponential":
                def f_exp(t_, r): return np.exp(-r * t_)
                popt, _ = curve_fit(f_exp, t, y, p0=[0.1], bounds=(0, 10), maxfev=500)
                y_pred = f_exp(t, *popt)
                params = {"rate": float(popt[0])}
            else:
                def f_pow(t_, gamma): return 1.0 / (1 + t_) ** gamma
                popt, _ = curve_fit(f_pow, t, y, p0=[0.5], bounds=(0, 5), maxfev=500)
                y_pred = f_pow(t, *popt)
                params = {"exponent": float(popt[0])}

            ss_res = float(np.sum((y - y_pred)**2))
            ss_tot = float(np.sum((y - np.mean(y))**2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            return {"model": model, "r2": max(0.0, r2), **params}
        except Exception:
            return {"model": model, "rate": 0.1, "r2": 0.0}

    def generate_decay_profile(
        self,
        initial_impact_bps: float,
        half_life_trades: int = 20,
        n_steps: int = 100,
    ) -> List[Tuple[int, float]]:
        """Generate expected impact decay profile."""
        rate = math.log(2) / half_life_trades
        return [
            (t, self.exponential_decay(initial_impact_bps, rate, t))
            for t in range(n_steps)
        ]


# ---------------------------------------------------------------------------
# Cross-impact estimator
# ---------------------------------------------------------------------------

class CrossImpactEstimator:
    """
    Estimates how order flow in asset A impacts the price of correlated asset B.
    Uses linear regression: Δp_B = λ_AB * OFI_A + ε
    """

    def estimate(
        self,
        ofi_a: List[float],      # order flow imbalance of asset A
        returns_b: List[float],  # price returns of asset B
        asset_a: str,
        asset_b: str,
    ) -> CrossImpactEstimate:
        n = min(len(ofi_a), len(returns_b))
        if n < 20:
            return CrossImpactEstimate(asset_a, asset_b, 0.0, 0.0, 0, 0.0, False)

        X = np.array(ofi_a[:n])
        Y = np.array(returns_b[:n])

        corr = float(np.corrcoef(X, Y)[0, 1]) if np.std(X) > 0 and np.std(Y) > 0 else 0.0

        X_dm = X - np.mean(X)
        Y_dm = Y - np.mean(Y)

        if np.dot(X_dm, X_dm) == 0:
            return CrossImpactEstimate(asset_a, asset_b, 0.0, 0.0, n, corr, False)

        lambda_ab = float(np.dot(X_dm, Y_dm) / np.dot(X_dm, X_dm))

        y_pred = lambda_ab * X_dm + np.mean(Y)
        ss_res = float(np.sum((Y - y_pred)**2))
        ss_tot = float(np.sum(Y_dm**2))
        r2 = max(0.0, 1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

        # T-stat for significance
        se = math.sqrt(ss_res / (n - 2) / np.dot(X_dm, X_dm)) if n > 2 else 1.0
        t_stat = lambda_ab / se if se > 0 else 0.0
        significant = abs(t_stat) > 2.0

        return CrossImpactEstimate(
            asset_a=asset_a,
            asset_b=asset_b,
            cross_lambda=lambda_ab,
            r_squared=r2,
            n_obs=n,
            correlation=corr,
            significance=significant,
        )

    def build_cross_impact_matrix(
        self,
        ofi_matrix: Dict[str, List[float]],
        return_matrix: Dict[str, List[float]],
    ) -> Dict[Tuple[str, str], CrossImpactEstimate]:
        """Build N×N cross-impact matrix for a set of assets."""
        assets = list(ofi_matrix.keys())
        result = {}
        for a in assets:
            for b in assets:
                if a == b:
                    continue
                est = self.estimate(ofi_matrix[a], return_matrix[b], a, b)
                result[(a, b)] = est
        return result


# ---------------------------------------------------------------------------
# Execution cost estimator
# ---------------------------------------------------------------------------

class ExecutionCostEstimator:
    """
    Estimates total execution cost for a given order using impact models.
    Cost = spread cost + temporary impact + market risk
    """

    def __init__(self, temp_model: PowerLawModel = None, perm_model: PowerLawModel = None):
        self.temp_model = temp_model or PowerLawModel(0.5, 0.314, 0.6, 100, "temporary")
        self.perm_model = perm_model or PowerLawModel(0.3, 0.157, 0.4, 100, "permanent")

    def estimate_cost(
        self,
        order_size: float,
        adv: float,
        price: float,
        sigma: float,
        spread: float,
        n_periods: int = 10,     # number of execution periods
    ) -> Dict:
        """
        Estimate total execution cost for splitting order over n_periods.
        """
        child_size = order_size / n_periods

        # Market risk: volatility of holding during execution
        sigma_period = sigma / math.sqrt(252 * 6.5 * 60 / n_periods)  # scaled to period
        market_risk_bps = sigma_period * 10000 * math.sqrt(n_periods)  # VaR proxy

        # Spread cost
        spread_bps = spread / price * 5000   # half-spread in bps

        # Temporary impact per child order
        temp_bps = self.temp_model.predict(child_size, adv, sigma)

        # Permanent impact (whole order)
        perm_bps = self.perm_model.predict(order_size, adv, sigma)

        # Total cost
        total_bps = spread_bps + temp_bps + perm_bps
        total_usd = total_bps / 10000 * price * order_size

        return {
            "order_size": order_size,
            "adv": adv,
            "participation_rate": order_size / adv,
            "spread_bps": spread_bps,
            "temp_impact_bps": temp_bps,
            "perm_impact_bps": perm_bps,
            "market_risk_bps": market_risk_bps,
            "total_cost_bps": total_bps,
            "total_cost_usd": total_usd,
            "cost_as_pct_of_notional": total_bps / 100,
            "recommended_n_periods": n_periods,
        }

    def optimal_execution_periods(
        self,
        order_size: float,
        adv: float,
        price: float,
        sigma: float,
        spread: float,
        max_periods: int = 100,
    ) -> int:
        """Find optimal VWAP/TWAP split to minimize expected cost."""
        best_n = 1
        best_cost = float("inf")
        for n in range(1, max_periods + 1):
            cost = self.estimate_cost(order_size, adv, price, sigma, spread, n)["total_cost_bps"]
            if cost < best_cost:
                best_cost = cost
                best_n = n
            elif cost > best_cost * 1.05:  # early stopping if cost rises
                break
        return best_n


# ---------------------------------------------------------------------------
# Main PriceImpactAnalytics facade
# ---------------------------------------------------------------------------

class PriceImpactAnalytics:
    """Unified price impact analytics."""

    def __init__(self):
        self.fitter = PowerLawImpactFitter()
        self.decompositor = ImpactDecompositor()
        self.decay_model = ImpactDecayModel()
        self.cross_impact = CrossImpactEstimator()
        self.cost_estimator = ExecutionCostEstimator()

    def analyze_trades(
        self,
        classified: List[ClassifiedTrade],
        adv: float = 5_000_000,
        sigma: float = 0.02,
    ) -> Dict:
        prices = [ct.trade.price for ct in classified]
        decomps = self.decompositor.decompose_from_trades(classified, prices, adv, sigma)
        stats = self.decompositor.aggregate_stats(decomps)

        # Build observations for model fitting
        obs_list = [d.observation for d in decomps[:200]]
        temp_model = self.fitter.fit(obs_list, "temporary")
        perm_model = self.fitter.fit(obs_list, "permanent")

        return {
            "decomposition": stats,
            "temp_model": {
                "alpha": temp_model.alpha,
                "eta": temp_model.eta,
                "r_squared": temp_model.r_squared,
                "n_obs": temp_model.n_obs,
            },
            "perm_model": {
                "alpha": perm_model.alpha,
                "eta": perm_model.eta,
                "r_squared": perm_model.r_squared,
                "n_obs": perm_model.n_obs,
            },
        }

    def execution_analysis(
        self,
        order_size: float,
        adv: float,
        price: float,
        sigma: float,
        spread: float,
    ) -> str:
        opt_n = self.cost_estimator.optimal_execution_periods(order_size, adv, price, sigma, spread)
        costs = self.cost_estimator.estimate_cost(order_size, adv, price, sigma, spread, opt_n)

        lines = [
            f"=== Execution Cost Analysis ===",
            f"Order: {order_size:,.0f} shares @ ${price:.2f}",
            f"ADV: {adv:,.0f} | Participation: {order_size/adv:.1%}",
            f"Optimal periods: {opt_n}",
            "",
            f"Cost breakdown:",
            f"  Spread:          {costs['spread_bps']:.2f} bps",
            f"  Temp impact:     {costs['temp_impact_bps']:.2f} bps",
            f"  Perm impact:     {costs['perm_impact_bps']:.2f} bps",
            f"  Market risk:     {costs['market_risk_bps']:.2f} bps",
            f"  TOTAL:           {costs['total_cost_bps']:.2f} bps  (${costs['total_cost_usd']:,.0f})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Price impact CLI")
    parser.add_argument("--action", choices=["analyze", "execution", "decay"], default="execution")
    parser.add_argument("--size", type=float, default=10000)
    parser.add_argument("--adv", type=float, default=5_000_000)
    parser.add_argument("--price", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--spread", type=float, default=0.05)
    args = parser.parse_args()

    analytics = PriceImpactAnalytics()

    if args.action == "execution":
        print(analytics.execution_analysis(args.size, args.adv, args.price, args.sigma, args.spread))
    elif args.action == "analyze":
        gen = SyntheticTradeGenerator()
        trades = gen.generate(1000, args.price)
        from trade_classification import TradeClassificationEngine
        engine = TradeClassificationEngine()
        classified = engine.classify(trades)
        import json as _json
        result = analytics.analyze_trades(classified, args.adv, args.sigma)
        print(_json.dumps(result, indent=2))
    elif args.action == "decay":
        profile = analytics.decay_model.generate_decay_profile(20.0, half_life_trades=10, n_steps=50)
        print(f"Impact decay profile (initial=20bps, half-life=10 trades):")
        for t, impact in profile[::5]:
            bar = "█" * int(impact / 20 * 20)
            print(f"  t={t:3d}: {impact:6.2f} bps  {bar}")
