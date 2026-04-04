"""
Options strategy implementations.

Implements:
- Covered call
- Protective put
- Straddle (long/short)
- Strangle (long/short)
- Butterfly spread
- Iron condor
- Calendar spread
- Ratio spread
- Delta-hedged P&L simulation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .pricing import BlackScholes
from .greeks import BlackScholesGreeks


# ---------------------------------------------------------------------------
# Option leg representation
# ---------------------------------------------------------------------------

@dataclass
class OptionLeg:
    """A single option contract leg in a strategy."""
    option_type: str   # 'call' or 'put'
    K: float           # strike
    T: float           # expiration
    sigma: float       # implied vol
    position: float    # +1 = long, -1 = short
    quantity: float = 1.0


@dataclass
class StrategyResult:
    """Result of an options strategy analysis."""
    name: str
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    net_premium: float
    legs: List[OptionLeg]
    payoff_profile: pd.Series
    greeks: Dict[str, float]
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class OptionsStrategy:
    """Base class for multi-leg options strategies."""

    def __init__(self, S: float, r: float = 0.02, q: float = 0.0) -> None:
        self.S = S
        self.r = r
        self.q = q
        self.legs: List[OptionLeg] = []

    def _add_leg(
        self,
        option_type: str,
        K: float,
        T: float,
        sigma: float,
        position: float,
        quantity: float = 1.0,
    ) -> None:
        self.legs.append(OptionLeg(option_type, K, T, sigma, position, quantity))

    def _leg_price(self, leg: OptionLeg) -> float:
        bs = BlackScholes(self.S, leg.K, leg.T, self.r, leg.sigma, self.q)
        return bs.call() if leg.option_type == "call" else bs.put()

    def net_premium(self) -> float:
        """Net premium paid (negative = received)."""
        total = 0.0
        for leg in self.legs:
            price = self._leg_price(leg)
            total += leg.position * leg.quantity * price
        return total

    def _leg_payoff(self, leg: OptionLeg, S_T: np.ndarray) -> np.ndarray:
        """Payoff at expiration for a single leg."""
        if leg.option_type == "call":
            intrinsic = np.maximum(S_T - leg.K, 0.0)
        else:
            intrinsic = np.maximum(leg.K - S_T, 0.0)
        return leg.position * leg.quantity * intrinsic

    def payoff_profile(self, S_range: Optional[np.ndarray] = None) -> pd.Series:
        """
        Compute net payoff at expiration over a range of terminal prices.

        Returns
        -------
        pd.Series
            Index = S_T prices, values = net payoff (including premium).
        """
        if S_range is None:
            S_range = np.linspace(max(self.S * 0.5, 1.0), self.S * 1.5, 200)

        premium = self.net_premium()
        total_payoff = np.zeros(len(S_range))
        for leg in self.legs:
            total_payoff += self._leg_payoff(leg, S_range)

        # P&L = payoff - premium paid
        pnl = total_payoff - premium

        return pd.Series(pnl.round(4), index=np.round(S_range, 2))

    def aggregate_greeks(self) -> Dict[str, float]:
        """Sum of Greeks across all legs."""
        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
               "theta": 0.0, "rho": 0.0}
        for leg in self.legs:
            g = BlackScholesGreeks(self.S, leg.K, leg.T, self.r, leg.sigma, self.q)
            for greek in agg:
                val = getattr(g, greek)(leg.option_type) if greek in ("delta", "theta", "rho") \
                    else getattr(g, greek)()
                agg[greek] += leg.position * leg.quantity * val
        return {k: round(v, 6) for k, v in agg.items()}

    def _find_breakevens(self, pnl: pd.Series, tol: float = 0.1) -> List[float]:
        """Find breakeven points where P&L crosses zero."""
        breakevens = []
        S_vals = pnl.index.values
        pnl_vals = pnl.values
        for i in range(1, len(pnl_vals)):
            if pnl_vals[i - 1] * pnl_vals[i] < 0:
                # Linear interpolation
                be = S_vals[i - 1] - pnl_vals[i - 1] * (S_vals[i] - S_vals[i - 1]) / \
                     (pnl_vals[i] - pnl_vals[i - 1] + 1e-10)
                breakevens.append(round(be, 4))
        return breakevens

    def analyze(self) -> StrategyResult:
        """Full strategy analysis."""
        pnl = self.payoff_profile()
        breakevens = self._find_breakevens(pnl)
        greeks = self.aggregate_greeks()
        premium = self.net_premium()
        max_profit = float(pnl.max())
        max_loss = float(pnl.min())

        return StrategyResult(
            name=self.__class__.__name__,
            max_profit=round(max_profit, 4),
            max_loss=round(max_loss, 4),
            breakeven_points=breakevens,
            net_premium=round(premium, 4),
            legs=self.legs,
            payoff_profile=pnl,
            greeks=greeks,
        )


# ---------------------------------------------------------------------------
# Specific Strategies
# ---------------------------------------------------------------------------

class CoveredCall(OptionsStrategy):
    """
    Covered call: long stock + short OTM call.

    Parameters
    ----------
    K : float
        Strike of short call (typically OTM).
    T : float
        Expiration of call.
    sigma : float
        Implied vol of call.
    """

    def __init__(
        self, S: float, K: float, T: float, sigma: float,
        r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        self._add_leg("call", K, T, sigma, position=-1.0)
        self.K = K
        self.T = T

    def payoff_profile(self, S_range=None):
        if S_range is None:
            S_range = np.linspace(max(self.S * 0.5, 1.0), self.S * 1.5, 200)
        premium = self.net_premium()
        # Stock payoff: S_T - S (excluding initial cost)
        stock_pnl = S_range - self.S
        option_payoff = self._leg_payoff(self.legs[0], S_range)
        pnl = stock_pnl + option_payoff - premium
        return pd.Series(pnl.round(4), index=np.round(S_range, 2))


class ProtectivePut(OptionsStrategy):
    """Long stock + long ATM/OTM put."""

    def __init__(
        self, S: float, K: float, T: float, sigma: float,
        r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        self._add_leg("put", K, T, sigma, position=1.0)
        self.K = K

    def payoff_profile(self, S_range=None):
        if S_range is None:
            S_range = np.linspace(max(self.S * 0.5, 1.0), self.S * 1.5, 200)
        premium = self.net_premium()
        stock_pnl = S_range - self.S
        option_payoff = self._leg_payoff(self.legs[0], S_range)
        pnl = stock_pnl + option_payoff - premium
        return pd.Series(pnl.round(4), index=np.round(S_range, 2))


class Straddle(OptionsStrategy):
    """
    Long straddle: long ATM call + long ATM put.
    Short straddle: short both.

    Parameters
    ----------
    K : float
        ATM strike.
    T : float
        Expiration.
    sigma : float
        Implied vol.
    long : bool
        True = long straddle, False = short straddle.
    """

    def __init__(
        self, S: float, K: float, T: float, sigma: float,
        long: bool = True, r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        pos = 1.0 if long else -1.0
        self._add_leg("call", K, T, sigma, position=pos)
        self._add_leg("put", K, T, sigma, position=pos)
        self.long = long


class Strangle(OptionsStrategy):
    """
    Long/short strangle: OTM call + OTM put.

    Parameters
    ----------
    K_call : float
        Strike of OTM call (K_call > S).
    K_put : float
        Strike of OTM put (K_put < S).
    """

    def __init__(
        self, S: float, K_call: float, K_put: float, T: float,
        sigma_call: float, sigma_put: float,
        long: bool = True, r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        pos = 1.0 if long else -1.0
        self._add_leg("call", K_call, T, sigma_call, position=pos)
        self._add_leg("put", K_put, T, sigma_put, position=pos)
        self.long = long


class ButterflySpread(OptionsStrategy):
    """
    Long call butterfly: long K_lo + short 2*K_mid + long K_hi (all calls).

    Maximum profit at K_mid, maximum loss = net premium.

    Parameters
    ----------
    K_lo, K_mid, K_hi : float
        Low, middle, high strikes (K_lo < K_mid < K_hi).
    """

    def __init__(
        self, S: float, K_lo: float, K_mid: float, K_hi: float,
        T: float, sigma_lo: float, sigma_mid: float, sigma_hi: float,
        long: bool = True, option_type: str = "call",
        r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        sign = 1.0 if long else -1.0
        self._add_leg(option_type, K_lo, T, sigma_lo, position=sign)
        self._add_leg(option_type, K_mid, T, sigma_mid, position=-2.0 * sign)
        self._add_leg(option_type, K_hi, T, sigma_hi, position=sign)


class IronCondor(OptionsStrategy):
    """
    Iron condor: short OTM strangle + long wider OTM strangle.

    Legs (all same expiration):
    - Long K_put_lo (protective put)
    - Short K_put_hi (short put)
    - Short K_call_lo (short call)
    - Long K_call_hi (protective call)

    K_put_lo < K_put_hi < S < K_call_lo < K_call_hi
    """

    def __init__(
        self, S: float,
        K_put_lo: float, K_put_hi: float,
        K_call_lo: float, K_call_hi: float,
        T: float,
        sigma_put_lo: float, sigma_put_hi: float,
        sigma_call_lo: float, sigma_call_hi: float,
        r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        self._add_leg("put", K_put_lo, T, sigma_put_lo, position=1.0)   # long protective
        self._add_leg("put", K_put_hi, T, sigma_put_hi, position=-1.0)  # short
        self._add_leg("call", K_call_lo, T, sigma_call_lo, position=-1.0)  # short
        self._add_leg("call", K_call_hi, T, sigma_call_hi, position=1.0)   # long protective

    def max_profit_loss(self) -> Dict[str, float]:
        """Compute theoretical max profit and max loss."""
        premium = self.net_premium()
        put_spread = self.legs[1].K - self.legs[0].K
        call_spread = self.legs[3].K - self.legs[2].K
        return {
            "max_profit": round(-premium, 4),  # received premium
            "max_loss_put_side": round(put_spread + premium, 4),
            "max_loss_call_side": round(call_spread + premium, 4),
            "net_premium_received": round(-premium, 4),
        }


class CalendarSpread(OptionsStrategy):
    """
    Calendar (time) spread: short near-term option + long far-term option.

    Parameters
    ----------
    K : float
        Common strike.
    T_near : float
        Near-term expiration.
    T_far : float
        Far-term expiration.
    sigma_near, sigma_far : float
        Implied vols for near and far legs.
    """

    def __init__(
        self, S: float, K: float,
        T_near: float, T_far: float,
        sigma_near: float, sigma_far: float,
        option_type: str = "call",
        r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        self._add_leg(option_type, K, T_near, sigma_near, position=-1.0)
        self._add_leg(option_type, K, T_far, sigma_far, position=1.0)
        self.option_type = option_type
        self.K = K
        self.T_near = T_near

    def theta_advantage(self) -> float:
        """Positive theta from short near-term leg."""
        return self.aggregate_greeks()["theta"]

    def analyze_time_decay(
        self, n_days: int = 30, S_range: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Simulate P&L at different times between now and T_near.

        Returns
        -------
        pd.DataFrame
            P&L surface (days_elapsed x S_T).
        """
        if S_range is None:
            S_range = np.linspace(max(self.S * 0.8, 1.0), self.S * 1.2, 50)

        rows = {}
        for days in range(0, min(n_days, int(self.T_near * 252)), 5):
            t_elapsed = days / 252
            row_pnl = []
            for S_t in S_range:
                pnl = 0.0
                for leg in self.legs:
                    T_remaining = max(leg.T - t_elapsed, 1e-6)
                    bs = BlackScholes(S_t, leg.K, T_remaining, self.r, leg.sigma, self.q)
                    price = bs.call() if leg.option_type == "call" else bs.put()
                    # Original price
                    bs0 = BlackScholes(self.S, leg.K, leg.T, self.r, leg.sigma, self.q)
                    orig = bs0.call() if leg.option_type == "call" else bs0.put()
                    pnl += leg.position * (price - orig)
                row_pnl.append(round(pnl, 4))
            rows[f"day_{days}"] = row_pnl

        return pd.DataFrame(rows, index=S_range.round(2)).T


class RatioSpread(OptionsStrategy):
    """
    Ratio spread: long 1 option + short n options at different strike.

    Commonly: long 1 ATM call + short 2 OTM calls.

    Parameters
    ----------
    K_long : float
        Strike of long leg.
    K_short : float
        Strike of short legs.
    ratio : float
        Number of short options per 1 long option.
    """

    def __init__(
        self, S: float, K_long: float, K_short: float, T: float,
        sigma_long: float, sigma_short: float,
        ratio: float = 2.0, option_type: str = "call",
        r: float = 0.02, q: float = 0.0
    ) -> None:
        super().__init__(S, r, q)
        self._add_leg(option_type, K_long, T, sigma_long, position=1.0, quantity=1.0)
        self._add_leg(option_type, K_short, T, sigma_short, position=-1.0, quantity=ratio)


# ---------------------------------------------------------------------------
# Delta-Hedged P&L Simulation
# ---------------------------------------------------------------------------

class DeltaHedgeSimulator:
    """
    Simulate daily delta-hedging of a short option position.

    The hedger:
    1. Sells 1 option at inception.
    2. Delta-hedges daily using underlying shares.
    3. Records daily P&L = change in option value - delta * change in stock.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike.
    T : float
        Initial time to expiration (years).
    r : float
        Risk-free rate.
    sigma_market : float
        True (realized) volatility of the underlying.
    sigma_implied : float
        Implied vol at which the option was sold.
    q : float
        Dividend yield.
    n_paths : int
        Number of Monte Carlo paths for P&L distribution.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma_market: float,
        sigma_implied: float,
        q: float = 0.0,
        n_paths: int = 1000,
        seed: int = 42,
    ) -> None:
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma_market = sigma_market
        self.sigma_implied = sigma_implied
        self.q = q
        self.n_paths = n_paths
        self.seed = seed

    def simulate(self) -> pd.DataFrame:
        """
        Simulate delta-hedged P&L for each path.

        Returns
        -------
        pd.DataFrame
            (n_paths x n_steps) cumulative P&L matrix.
        """
        rng = np.random.default_rng(self.seed)
        n_steps = max(int(self.T * 252), 1)
        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma_market ** 2) * dt
        diffusion = self.sigma_market * np.sqrt(dt)

        # Simulate paths
        Z = rng.standard_normal((self.n_paths, n_steps))
        log_S = np.zeros((self.n_paths, n_steps + 1))
        log_S[:, 0] = np.log(self.S0)
        for t in range(n_steps):
            log_S[:, t + 1] = log_S[:, t] + drift + diffusion * Z[:, t]
        S_paths = np.exp(log_S)

        cum_pnl = np.zeros((self.n_paths, n_steps))

        for path in range(self.n_paths):
            S_arr = S_paths[path]
            daily_pnl = []

            # Short option at implied vol
            bs0 = BlackScholes(S_arr[0], self.K, self.T, self.r, self.sigma_implied, self.q)
            option_price_0 = bs0.call()  # received
            delta_0 = bs0.greeks()["delta"]
            hedge_cost = delta_0 * S_arr[0]  # cost of delta shares

            option_value_prev = option_price_0

            for t in range(1, n_steps + 1):
                T_remain = max(self.T - (t - 1) * dt, 1e-8)
                T_remain_new = max(self.T - t * dt, 1e-8)
                S_prev = S_arr[t - 1]
                S_curr = S_arr[t] if t < len(S_arr) else S_arr[-1]

                bs_prev = BlackScholes(S_prev, self.K, T_remain, self.r, self.sigma_implied, self.q)
                bs_curr = BlackScholes(S_curr, self.K, T_remain_new, self.r, self.sigma_implied, self.q)

                delta_prev = bs_prev.greeks()["delta"]
                option_value_curr = bs_curr.call()

                # Hedge P&L
                delta_pnl = delta_prev * (S_curr - S_prev)
                option_pnl = -(option_value_curr - option_value_prev)  # short option

                daily_pnl.append(delta_pnl + option_pnl)
                option_value_prev = option_value_curr

            cum_pnl[path] = np.cumsum(daily_pnl)

        return pd.DataFrame(
            cum_pnl,
            index=[f"path_{i}" for i in range(self.n_paths)],
            columns=[f"day_{t}" for t in range(n_steps)],
        )

    def pnl_summary(self) -> Dict[str, float]:
        """Summary statistics of final hedged P&L distribution."""
        pnl_df = self.simulate()
        final_pnl = pnl_df.iloc[:, -1]
        return {
            "mean": round(float(final_pnl.mean()), 6),
            "std": round(float(final_pnl.std()), 6),
            "sharpe": round(float(final_pnl.mean() / (final_pnl.std() + 1e-12)), 4),
            "pct_positive": round(float((final_pnl > 0).mean()), 4),
            "5th_percentile": round(float(final_pnl.quantile(0.05)), 6),
            "95th_percentile": round(float(final_pnl.quantile(0.95)), 6),
            "vp_edge_bps": round(
                (self.sigma_implied ** 2 - self.sigma_market ** 2) / (2 * self.sigma_implied) * 1e4, 4
            ),
        }
