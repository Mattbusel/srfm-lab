"""
Options risk management for the srfm-lab trading system.

Implements positions, portfolios, hedgers, and risk limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from lib.options.greeks import GreeksResult


# ---------------------------------------------------------------------------
# OptionsPosition
# ---------------------------------------------------------------------------

@dataclass
class OptionsPosition:
    """
    A single options position.

    Attributes
    ----------
    underlying : str
        Ticker or identifier of the underlying.
    option_type : str
        'call' or 'put'.
    strike : float
        Strike price.
    expiry : float
        Time to expiry in years.
    qty : float
        Signed quantity (positive = long, negative = short).
    entry_price : float
        Price paid/received per unit when position was opened.
    current_price : float
        Current market price of the option.
    greeks : GreeksResult
        Per-unit Greeks of this position.
    spot : float
        Current spot price of underlying.
    sigma : float
        Current implied vol of this option.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    multiplier : float
        Contract multiplier (e.g., 100 for equity options).
    """
    underlying: str
    option_type: str
    strike: float
    expiry: float
    qty: float
    entry_price: float
    current_price: float
    greeks: GreeksResult = field(default_factory=GreeksResult)
    spot: float = 100.0
    sigma: float = 0.20
    r: float = 0.05
    q: float = 0.0
    multiplier: float = 100.0

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.current_price * self.qty * self.multiplier

    @property
    def unrealised_pnl(self) -> float:
        """Unrealised P&L relative to entry price."""
        return (self.current_price - self.entry_price) * self.qty * self.multiplier

    @property
    def position_greeks(self) -> GreeksResult:
        """Greeks scaled by quantity and multiplier."""
        return self.greeks * (self.qty * self.multiplier)

    def refresh_greeks(self) -> None:
        """Recompute Greeks from current market state using BS."""
        from lib.options.pricing import BlackScholes
        bs = BlackScholes(self.spot, self.strike, self.expiry, self.r, self.sigma, self.q, self.option_type)
        d = bs.all_greeks()
        self.greeks = GreeksResult(**d)
        self.current_price = bs.price()

    def delta_dollars(self) -> float:
        """Dollar delta: delta * spot * qty * multiplier."""
        return self.greeks.delta * self.spot * self.qty * self.multiplier

    def gamma_dollars(self) -> float:
        """Dollar gamma: 0.5 * gamma * spot^2 / 100 (P&L per 1% spot move)."""
        return 0.5 * self.greeks.gamma * (self.spot * 0.01) ** 2 * self.qty * self.multiplier

    def vega_dollars(self) -> float:
        """Dollar vega: P&L for 1% increase in vol."""
        return self.greeks.vega * 0.01 * self.qty * self.multiplier


# ---------------------------------------------------------------------------
# OptionsPortfolio
# ---------------------------------------------------------------------------

class OptionsPortfolio:
    """
    Collection of options positions with portfolio-level risk analytics.

    Provides:
    - Aggregated Greeks
    - P&L scenario matrix
    - Delta-gamma VaR
    - Hedge recommendations
    """

    def __init__(self) -> None:
        self._positions: List[OptionsPosition] = []

    def add_position(self, position: OptionsPosition) -> None:
        """Add a position to the portfolio."""
        self._positions.append(position)

    def remove_position(self, idx: int) -> None:
        """Remove a position by index."""
        self._positions.pop(idx)

    @property
    def positions(self) -> List[OptionsPosition]:
        return list(self._positions)

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def portfolio_greeks(self) -> GreeksResult:
        """Net portfolio Greeks (sum of all position Greeks)."""
        result = GreeksResult.zero()
        for pos in self._positions:
            result = result + pos.position_greeks
        return result

    def greeks_by_underlying(self) -> Dict[str, GreeksResult]:
        """Portfolio Greeks grouped by underlying."""
        result: Dict[str, GreeksResult] = {}
        for pos in self._positions:
            und = pos.underlying
            result[und] = result.get(und, GreeksResult.zero()) + pos.position_greeks
        return result

    def net_delta_dollars(self) -> float:
        """Total dollar delta of the portfolio."""
        return sum(pos.delta_dollars() for pos in self._positions)

    def net_vega_dollars(self) -> float:
        """Total dollar vega of the portfolio."""
        return sum(pos.vega_dollars() for pos in self._positions)

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------

    def total_market_value(self) -> float:
        return sum(pos.market_value for pos in self._positions)

    def total_unrealised_pnl(self) -> float:
        return sum(pos.unrealised_pnl for pos in self._positions)

    def pnl_approximation(self, dS: float, d_sigma: float = 0.0, d_t_days: float = 0.0) -> float:
        """
        Approximate portfolio P&L using second-order Taylor expansion.

        P&L ~ delta*dS + 0.5*gamma*dS^2 + vega*d_sigma + theta*d_t
        """
        g = self.portfolio_greeks()
        return (
            g.delta * dS
            + 0.5 * g.gamma * dS ** 2
            + g.vega * d_sigma
            + g.theta * d_t_days
            + g.vanna * dS * d_sigma
            + 0.5 * g.volga * d_sigma ** 2
        )

    # ------------------------------------------------------------------
    # Scenario matrix
    # ------------------------------------------------------------------

    def scenario_matrix(
        self,
        spot_shocks: Optional[List[float]] = None,
        vol_shocks: Optional[List[float]] = None,
    ) -> dict:
        """
        Compute P&L scenarios for a grid of spot and vol shocks.

        Parameters
        ----------
        spot_shocks : list of float
            Fractional spot moves, e.g. [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20].
        vol_shocks : list of float
            Absolute vol moves, e.g. [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20].

        Returns
        -------
        dict with keys:
            'spot_shocks', 'vol_shocks', 'pnl_matrix'
        """
        if spot_shocks is None:
            spot_shocks = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        if vol_shocks is None:
            vol_shocks = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

        # Average spot for scaling
        spots = [pos.spot for pos in self._positions]
        ref_spot = float(np.mean(spots)) if spots else 100.0

        pnl_matrix = np.zeros((len(spot_shocks), len(vol_shocks)))
        for i, ss in enumerate(spot_shocks):
            dS = ref_spot * ss
            for j, vs in enumerate(vol_shocks):
                pnl_matrix[i, j] = self.pnl_approximation(dS, vs)

        return {
            "spot_shocks": spot_shocks,
            "vol_shocks": vol_shocks,
            "pnl_matrix": pnl_matrix,
        }

    # ------------------------------------------------------------------
    # VaR
    # ------------------------------------------------------------------

    def var_delta_gamma(
        self,
        spot_vol_annual: float,
        confidence: float = 0.99,
        horizon_days: float = 1.0,
    ) -> float:
        """
        Compute Value-at-Risk via delta-gamma normal approximation.

        Uses the Cornish-Fisher expansion to correct for gamma-induced skew.

        Parameters
        ----------
        spot_vol_annual : float
            Annualised spot return volatility.
        confidence : float
            Confidence level (e.g. 0.99 for 99% VaR).
        horizon_days : float
            VaR holding period in calendar days.

        Returns
        -------
        float
            VaR as a positive number (loss).
        """
        g = self.portfolio_greeks()
        spots = [pos.spot for pos in self._positions]
        ref_spot = float(np.mean(spots)) if spots else 100.0

        dt = horizon_days / 252.0
        sigma_S = spot_vol_annual * ref_spot * math.sqrt(dt)

        delta = g.delta
        gamma = g.gamma

        # First and second moments of P&L
        mu_pnl = 0.5 * gamma * sigma_S ** 2  # expected gain from convexity
        var_pnl = (delta * sigma_S) ** 2 + 0.5 * (gamma * sigma_S ** 2) ** 2
        std_pnl = math.sqrt(max(var_pnl, 0.0))

        # Skewness from gamma
        skew = gamma * sigma_S ** 2 * delta * sigma_S / max(std_pnl ** 3, 1e-20)
        kurtosis = 3.0 + 3.0 * (gamma * sigma_S ** 2) ** 2 / max(var_pnl ** 2, 1e-20)
        excess_kurt = kurtosis - 3.0

        z = norm.ppf(confidence)
        # Cornish-Fisher adjustment
        z_cf = (
            z
            + (z ** 2 - 1.0) * skew / 6.0
            + (z ** 3 - 3.0 * z) * excess_kurt / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * skew ** 2 / 36.0
        )

        var_loss = -(mu_pnl - z_cf * std_pnl)
        return max(var_loss, 0.0)

    def expected_shortfall(
        self,
        spot_vol_annual: float,
        confidence: float = 0.99,
        horizon_days: float = 1.0,
        n_simulations: int = 100000,
    ) -> float:
        """
        Expected shortfall (CVaR) via Monte Carlo delta-gamma approximation.
        """
        g = self.portfolio_greeks()
        spots = [pos.spot for pos in self._positions]
        ref_spot = float(np.mean(spots)) if spots else 100.0

        dt = horizon_days / 252.0
        sigma_S = spot_vol_annual * ref_spot * math.sqrt(dt)

        rng = np.random.default_rng(42)
        dS = rng.normal(0.0, sigma_S, n_simulations)
        pnl = g.delta * dS + 0.5 * g.gamma * dS ** 2

        cutoff = np.percentile(pnl, (1.0 - confidence) * 100.0)
        tail = pnl[pnl <= cutoff]
        es = -float(np.mean(tail)) if len(tail) > 0 else 0.0
        return max(es, 0.0)

    # ------------------------------------------------------------------
    # Hedge recommendations
    # ------------------------------------------------------------------

    def delta_hedge_recommendation(self) -> dict:
        """
        Recommend a spot hedge to flatten delta.

        Returns
        -------
        dict with 'underlying', 'quantity', 'direction', 'current_delta'
        per underlying.
        """
        recs = []
        for und, greeks in self.greeks_by_underlying().items():
            net_delta = greeks.delta
            # Find a representative spot for this underlying
            spots = [p.spot for p in self._positions if p.underlying == und]
            spot = float(np.mean(spots)) if spots else 100.0
            hedge_qty = -net_delta
            recs.append({
                "underlying": und,
                "current_net_delta": net_delta,
                "hedge_quantity": hedge_qty,
                "direction": "buy" if hedge_qty > 0 else "sell",
                "dollar_delta": net_delta * spot,
            })
        return {"delta_hedges": recs}

    def vega_hedge_recommendation(
        self, atm_strike: float, atm_vega_per_contract: float
    ) -> dict:
        """
        Recommend an ATM straddle trade to neutralise portfolio vega.

        Parameters
        ----------
        atm_strike : float
            ATM strike of the hedging instrument.
        atm_vega_per_contract : float
            Vega per contract of the hedging straddle.
        """
        net_vega = self.portfolio_greeks().vega
        if abs(atm_vega_per_contract) < 1e-10:
            return {"vega_hedge": {"error": "atm_vega_per_contract too small"}}
        hedge_contracts = -net_vega / atm_vega_per_contract
        return {
            "vega_hedge": {
                "current_net_vega": net_vega,
                "hedge_contracts": hedge_contracts,
                "atm_strike": atm_strike,
                "direction": "buy" if hedge_contracts > 0 else "sell",
            }
        }


# ---------------------------------------------------------------------------
# DeltaHedger
# ---------------------------------------------------------------------------

class DeltaHedger:
    """
    Delta hedge tracker for a single underlying.

    Tracks the cumulative hedge P&L, hedge position, and triggers
    rebalancing when the delta deviation exceeds a threshold.

    Parameters
    ----------
    underlying : str
        Underlying identifier.
    rebalance_threshold : float
        Delta deviation that triggers a rebalance (default 0.05).
    """

    def __init__(self, underlying: str, rebalance_threshold: float = 0.05) -> None:
        self.underlying = underlying
        self.rebalance_threshold = rebalance_threshold
        self._hedge_qty: float = 0.0
        self._hedge_pnl: float = 0.0
        self._last_hedge_price: Optional[float] = None
        self._history: List[dict] = []

    def compute_hedge(
        self,
        portfolio: OptionsPortfolio,
        current_spot: float,
    ) -> float:
        """
        Compute the required hedge quantity to flatten delta for this underlying.

        Returns the signed spot units needed (positive = buy, negative = sell).
        """
        by_und = portfolio.greeks_by_underlying()
        net_delta = by_und.get(self.underlying, GreeksResult.zero()).delta
        return -net_delta

    def should_rebalance(self, portfolio: OptionsPortfolio, current_spot: float) -> bool:
        """Return True if the current hedge deviates by more than the threshold."""
        required = self.compute_hedge(portfolio, current_spot)
        deviation = abs(required - self._hedge_qty)
        return deviation > self.rebalance_threshold

    def rebalance(self, portfolio: OptionsPortfolio, current_spot: float) -> dict:
        """
        Execute a rebalance trade.

        Updates internal hedge position and records P&L from the previous hedge.

        Returns a dict describing the trade.
        """
        if self._last_hedge_price is not None:
            price_change = current_spot - self._last_hedge_price
            self._hedge_pnl += self._hedge_qty * price_change

        required = self.compute_hedge(portfolio, current_spot)
        trade_qty = required - self._hedge_qty
        self._hedge_qty = required
        self._last_hedge_price = current_spot

        trade = {
            "underlying": self.underlying,
            "trade_qty": trade_qty,
            "direction": "buy" if trade_qty > 0 else "sell",
            "price": current_spot,
            "new_hedge_qty": self._hedge_qty,
            "cumulative_hedge_pnl": self._hedge_pnl,
        }
        self._history.append(trade)
        return trade

    def hedge_pnl(self) -> float:
        """Total P&L generated by the hedge position."""
        return self._hedge_pnl

    def hedge_history(self) -> List[dict]:
        """List of all hedge trades."""
        return list(self._history)


# ---------------------------------------------------------------------------
# VegaHedger
# ---------------------------------------------------------------------------

class VegaHedger:
    """
    Vega hedge manager.

    Constructs a vega-neutral hedge using calendar spreads or ratio spreads.

    Parameters
    ----------
    underlying : str
        Underlying identifier.
    """

    def __init__(self, underlying: str) -> None:
        self.underlying = underlying

    def calendar_spread_hedge(
        self,
        portfolio: OptionsPortfolio,
        near_vega: float,
        far_vega: float,
        near_expiry: float,
        far_expiry: float,
    ) -> dict:
        """
        Recommend a calendar spread to neutralise vega.

        The calendar spread is long far expiry / short near expiry.
        Net vega per spread = far_vega - near_vega.

        Parameters
        ----------
        near_vega, far_vega : float
            Vega per contract for the near/far legs.
        """
        by_und = portfolio.greeks_by_underlying()
        net_vega = by_und.get(self.underlying, GreeksResult.zero()).vega
        spread_vega = far_vega - near_vega
        if abs(spread_vega) < 1e-10:
            return {"error": "Near and far vega are equal; no hedge possible"}
        n_spreads = -net_vega / spread_vega
        return {
            "underlying": self.underlying,
            "current_net_vega": net_vega,
            "n_spreads": n_spreads,
            "direction": "buy" if n_spreads > 0 else "sell",
            "near_expiry": near_expiry,
            "far_expiry": far_expiry,
            "residual_vega_after_hedge": net_vega + n_spreads * spread_vega,
        }

    def ratio_spread_hedge(
        self,
        portfolio: OptionsPortfolio,
        long_vega: float,
        short_vega: float,
        ratio: float = 2.0,
    ) -> dict:
        """
        Recommend a ratio spread to neutralise vega.

        Ratio spread: long 1 option, short `ratio` options.
        Net vega per spread = long_vega - ratio * short_vega.
        """
        by_und = portfolio.greeks_by_underlying()
        net_vega = by_und.get(self.underlying, GreeksResult.zero()).vega
        spread_vega = long_vega - ratio * short_vega
        if abs(spread_vega) < 1e-10:
            return {"error": "Ratio spread vega is zero; adjust ratio"}
        n_spreads = -net_vega / spread_vega
        return {
            "underlying": self.underlying,
            "current_net_vega": net_vega,
            "n_spreads": n_spreads,
            "ratio": ratio,
            "residual_vega_after_hedge": net_vega + n_spreads * spread_vega,
        }


# ---------------------------------------------------------------------------
# RiskLimits
# ---------------------------------------------------------------------------

@dataclass
class RiskLimits:
    """
    Risk limits for options positions and portfolios.

    All limits are on absolute values unless noted.

    Attributes
    ----------
    max_delta_per_underlying : float
        Maximum net delta allowed per underlying (in shares/contracts).
    max_gamma_per_underlying : float
        Maximum net gamma allowed per underlying.
    max_vega_per_underlying : float
        Maximum net vega allowed per underlying.
    max_theta_per_underlying : float
        Maximum daily theta drain per underlying (positive = drain limit).
    max_portfolio_delta : float
        Maximum aggregate portfolio delta.
    max_portfolio_vega : float
        Maximum aggregate portfolio vega.
    max_portfolio_theta : float
        Maximum aggregate portfolio theta drain per day.
    max_position_size : float
        Maximum absolute qty per single position.
    max_concentration_pct : float
        Maximum fraction of portfolio notional in any single underlying (0-1).
    """
    max_delta_per_underlying: float = 1000.0
    max_gamma_per_underlying: float = 500.0
    max_vega_per_underlying: float = 50000.0
    max_theta_per_underlying: float = 10000.0
    max_portfolio_delta: float = 5000.0
    max_portfolio_vega: float = 200000.0
    max_portfolio_theta: float = 50000.0
    max_position_size: float = 1000.0
    max_concentration_pct: float = 0.25

    def check_portfolio(self, portfolio: OptionsPortfolio) -> List[str]:
        """
        Check all limits against a portfolio.

        Returns a list of violated limit descriptions (empty if all OK).
        """
        violations = []
        net = portfolio.portfolio_greeks()

        if abs(net.delta) > self.max_portfolio_delta:
            violations.append(
                f"Portfolio delta {net.delta:.1f} exceeds limit {self.max_portfolio_delta:.1f}"
            )
        if abs(net.vega) > self.max_portfolio_vega:
            violations.append(
                f"Portfolio vega {net.vega:.1f} exceeds limit {self.max_portfolio_vega:.1f}"
            )
        if abs(net.theta) > self.max_portfolio_theta:
            violations.append(
                f"Portfolio theta {net.theta:.1f} exceeds limit {self.max_portfolio_theta:.1f}"
            )

        for und, g in portfolio.greeks_by_underlying().items():
            if abs(g.delta) > self.max_delta_per_underlying:
                violations.append(
                    f"{und}: delta {g.delta:.1f} exceeds per-underlying limit {self.max_delta_per_underlying:.1f}"
                )
            if abs(g.gamma) > self.max_gamma_per_underlying:
                violations.append(
                    f"{und}: gamma {g.gamma:.4f} exceeds per-underlying limit {self.max_gamma_per_underlying:.4f}"
                )
            if abs(g.vega) > self.max_vega_per_underlying:
                violations.append(
                    f"{und}: vega {g.vega:.1f} exceeds per-underlying limit {self.max_vega_per_underlying:.1f}"
                )

        for pos in portfolio.positions:
            if abs(pos.qty) > self.max_position_size:
                violations.append(
                    f"Position {pos.underlying} {pos.option_type} K={pos.strike} qty={pos.qty} exceeds max_position_size={self.max_position_size}"
                )

        return violations

    def check_new_position(
        self,
        portfolio: OptionsPortfolio,
        new_position: OptionsPosition,
    ) -> List[str]:
        """
        Check if adding new_position to portfolio would breach any limit.

        Returns list of potential violations.
        """
        import copy
        trial = OptionsPortfolio()
        for p in portfolio.positions:
            trial.add_position(p)
        trial.add_position(new_position)
        return self.check_portfolio(trial)
