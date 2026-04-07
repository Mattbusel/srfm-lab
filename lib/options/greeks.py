"""
Comprehensive Greeks engine for the srfm-lab trading system.

Provides analytical and numerical Greeks, portfolio aggregation,
stress scenarios, and market-implied Greeks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# GreeksResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class GreeksResult:
    """
    Container for all option Greeks.

    First-order Greeks
    ------------------
    delta : float
        dV/dS
    vega : float
        dV/d_sigma
    theta : float
        dV/dt (per calendar day, typically negative)
    rho : float
        dV/dr

    Second-order Greeks
    -------------------
    gamma : float
        d2V/dS2
    vanna : float
        d2V/(dS d_sigma)
    volga : float
        d2V/d_sigma2 (vomma)
    charm : float
        d2V/(dS dt)

    Third-order Greeks
    ------------------
    speed : float
        d3V/dS3
    color : float
        d3V/(dS2 dt)
    ultima : float
        d3V/d_sigma3
    zomma : float
        d3V/(dS2 d_sigma)
    """
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    vanna: float = 0.0
    volga: float = 0.0
    charm: float = 0.0
    speed: float = 0.0
    color: float = 0.0
    ultima: float = 0.0
    zomma: float = 0.0

    def __mul__(self, qty: float) -> "GreeksResult":
        return GreeksResult(
            delta=self.delta * qty,
            gamma=self.gamma * qty,
            vega=self.vega * qty,
            theta=self.theta * qty,
            rho=self.rho * qty,
            vanna=self.vanna * qty,
            volga=self.volga * qty,
            charm=self.charm * qty,
            speed=self.speed * qty,
            color=self.color * qty,
            ultima=self.ultima * qty,
            zomma=self.zomma * qty,
        )

    def __rmul__(self, qty: float) -> "GreeksResult":
        return self.__mul__(qty)

    def __add__(self, other: "GreeksResult") -> "GreeksResult":
        return GreeksResult(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
            vanna=self.vanna + other.vanna,
            volga=self.volga + other.volga,
            charm=self.charm + other.charm,
            speed=self.speed + other.speed,
            color=self.color + other.color,
            ultima=self.ultima + other.ultima,
            zomma=self.zomma + other.zomma,
        )

    def as_dict(self) -> dict:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "vanna": self.vanna,
            "volga": self.volga,
            "charm": self.charm,
            "speed": self.speed,
            "color": self.color,
            "ultima": self.ultima,
            "zomma": self.zomma,
        }

    @classmethod
    def zero(cls) -> "GreeksResult":
        return cls()

    @classmethod
    def sum(cls, greeks_list: List["GreeksResult"]) -> "GreeksResult":
        result = cls.zero()
        for g in greeks_list:
            result = result + g
        return result


# ---------------------------------------------------------------------------
# AnalyticalGreeks
# ---------------------------------------------------------------------------

class AnalyticalGreeks:
    """
    Closed-form Black-Scholes Greeks for European options.

    All Greeks are computed analytically using the BSM formula.

    Parameters
    ----------
    option_type : str
        'call' or 'put'.
    """

    def __init__(self, option_type: str = "call") -> None:
        self.option_type = option_type.lower()

    def compute(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> GreeksResult:
        """
        Compute full Greek set analytically.

        Parameters
        ----------
        S, K, T, r, sigma, q : float
            Standard option parameters.

        Returns
        -------
        GreeksResult
        """
        from lib.options.pricing import BlackScholes
        bs = BlackScholes(S, K, T, r, sigma, q, self.option_type)
        d = bs.all_greeks()
        return GreeksResult(
            delta=d["delta"],
            gamma=d["gamma"],
            vega=d["vega"],
            theta=d["theta"],
            rho=d["rho"],
            vanna=d["vanna"],
            volga=d["volga"],
            charm=d["charm"],
            speed=d["speed"],
            color=d["color"],
            ultima=d["ultima"],
            zomma=d["zomma"],
        )

    def delta(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        from lib.options.pricing import BlackScholes
        return BlackScholes(S, K, T, r, sigma, q, self.option_type).delta()

    def gamma(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        from lib.options.pricing import BlackScholes
        return BlackScholes(S, K, T, r, sigma, q, self.option_type).gamma()

    def vega(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        from lib.options.pricing import BlackScholes
        return BlackScholes(S, K, T, r, sigma, q, self.option_type).vega()

    def theta(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        from lib.options.pricing import BlackScholes
        return BlackScholes(S, K, T, r, sigma, q, self.option_type).theta()

    def rho(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        from lib.options.pricing import BlackScholes
        return BlackScholes(S, K, T, r, sigma, q, self.option_type).rho()


# ---------------------------------------------------------------------------
# NumericalGreeks
# ---------------------------------------------------------------------------

class NumericalGreeks:
    """
    Finite-difference Greeks for any pricer function.

    Supports first- and second-order Greeks via central differences.
    Uses Richardson extrapolation for improved accuracy when requested.

    Parameters
    ----------
    pricer : callable
        Function with signature (S, K, T, r, sigma, q=0.0) -> float.
    dS_frac : float
        Fractional spot bump for delta/gamma. Default 0.01 (1%).
    dSigma : float
        Absolute vol bump for vega/volga. Default 0.001 (0.1%).
    dT : float
        Time bump in years for theta. Default 1/365.
    dr : float
        Rate bump for rho. Default 0.0001 (1bp).
    """

    def __init__(
        self,
        pricer: Callable,
        dS_frac: float = 0.01,
        dSigma: float = 0.001,
        dT: float = 1.0 / 365.0,
        dr: float = 0.0001,
    ) -> None:
        self.pricer = pricer
        self.dS_frac = dS_frac
        self.dSigma = dSigma
        self.dT = dT
        self.dr = dr

    def _p(self, S, K, T, r, sigma, q) -> float:
        return self.pricer(S, K, T, r, sigma, q)

    def delta(self, S, K, T, r, sigma, q=0.0) -> float:
        h = S * self.dS_frac
        return (self._p(S + h, K, T, r, sigma, q) - self._p(S - h, K, T, r, sigma, q)) / (2.0 * h)

    def gamma(self, S, K, T, r, sigma, q=0.0) -> float:
        h = S * self.dS_frac
        p0 = self._p(S, K, T, r, sigma, q)
        pu = self._p(S + h, K, T, r, sigma, q)
        pd = self._p(S - h, K, T, r, sigma, q)
        return (pu - 2.0 * p0 + pd) / h ** 2

    def vega(self, S, K, T, r, sigma, q=0.0) -> float:
        h = self.dSigma
        return (self._p(S, K, T, r, sigma + h, q) - self._p(S, K, T, r, sigma - h, q)) / (2.0 * h)

    def theta(self, S, K, T, r, sigma, q=0.0) -> float:
        h = self.dT
        if T <= h:
            return 0.0
        return (self._p(S, K, T - h, r, sigma, q) - self._p(S, K, T, r, sigma, q))

    def rho(self, S, K, T, r, sigma, q=0.0) -> float:
        h = self.dr
        return (self._p(S, K, T, r + h, sigma, q) - self._p(S, K, T, r - h, sigma, q)) / (2.0 * h)

    def vanna(self, S, K, T, r, sigma, q=0.0) -> float:
        """d2V/(dS d_sigma) via mixed finite difference."""
        hS = S * self.dS_frac
        hv = self.dSigma
        ppp = self._p(S + hS, K, T, r, sigma + hv, q)
        pmm = self._p(S + hS, K, T, r, sigma - hv, q)
        mpp = self._p(S - hS, K, T, r, sigma + hv, q)
        mmm = self._p(S - hS, K, T, r, sigma - hv, q)
        return (ppp - pmm - mpp + mmm) / (4.0 * hS * hv)

    def volga(self, S, K, T, r, sigma, q=0.0) -> float:
        """d2V/d_sigma2."""
        h = self.dSigma
        p0 = self._p(S, K, T, r, sigma, q)
        pu = self._p(S, K, T, r, sigma + h, q)
        pd = self._p(S, K, T, r, sigma - h, q)
        return (pu - 2.0 * p0 + pd) / h ** 2

    def charm(self, S, K, T, r, sigma, q=0.0) -> float:
        """d2V/(dS dt) via mixed finite difference."""
        hS = S * self.dS_frac
        hT = self.dT
        if T <= hT:
            return 0.0
        delta_now = self.delta(S, K, T, r, sigma, q)
        delta_later = self.delta(S, K, T - hT, r, sigma, q)
        return delta_later - delta_now

    def speed(self, S, K, T, r, sigma, q=0.0) -> float:
        """d3V/dS3 via third-order finite difference."""
        h = S * self.dS_frac
        p0 = self._p(S, K, T, r, sigma, q)
        pp = self._p(S + h, K, T, r, sigma, q)
        pm = self._p(S - h, K, T, r, sigma, q)
        p2p = self._p(S + 2 * h, K, T, r, sigma, q)
        p2m = self._p(S - 2 * h, K, T, r, sigma, q)
        return (-p2p + 2.0 * pp - 2.0 * pm + p2m) / (2.0 * h ** 3)

    def compute(self, S, K, T, r, sigma, q=0.0) -> GreeksResult:
        """Compute all available Greeks via finite differences."""
        return GreeksResult(
            delta=self.delta(S, K, T, r, sigma, q),
            gamma=self.gamma(S, K, T, r, sigma, q),
            vega=self.vega(S, K, T, r, sigma, q),
            theta=self.theta(S, K, T, r, sigma, q),
            rho=self.rho(S, K, T, r, sigma, q),
            vanna=self.vanna(S, K, T, r, sigma, q),
            volga=self.volga(S, K, T, r, sigma, q),
            charm=self.charm(S, K, T, r, sigma, q),
            speed=self.speed(S, K, T, r, sigma, q),
        )


# ---------------------------------------------------------------------------
# GreeksAggregator
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """Result of a single stress scenario."""
    label: str
    spot_shock: float
    vol_shock: float
    time_decay_days: float
    pnl: float


class GreeksAggregator:
    """
    Portfolio-level Greeks aggregation and stress testing.

    Aggregates Greeks across positions, groups by underlying,
    and computes P&L under stress scenarios.

    Usage
    -----
    aggregator = GreeksAggregator()
    aggregator.add("AAPL", greeks, qty=10, spot=150.0, sigma=0.25)
    portfolio_greeks = aggregator.net_greeks()
    """

    def __init__(self) -> None:
        self._positions: List[Dict] = []

    def add(
        self,
        underlying: str,
        greeks: GreeksResult,
        qty: float,
        spot: float,
        sigma: float,
        T: float = 1.0,
    ) -> None:
        """
        Add a position to the aggregator.

        Parameters
        ----------
        underlying : str
            Underlying identifier.
        greeks : GreeksResult
            Per-unit Greeks for this position.
        qty : float
            Number of contracts (signed: positive = long).
        spot : float
            Current spot price.
        sigma : float
            Current implied vol.
        T : float
            Time to expiry in years.
        """
        self._positions.append({
            "underlying": underlying,
            "greeks": greeks * qty,
            "qty": qty,
            "spot": spot,
            "sigma": sigma,
            "T": T,
        })

    def net_greeks(self) -> GreeksResult:
        """Portfolio net Greeks (sum across all positions)."""
        return GreeksResult.sum([p["greeks"] for p in self._positions])

    def greeks_by_underlying(self) -> Dict[str, GreeksResult]:
        """Net Greeks grouped by underlying."""
        result: Dict[str, GreeksResult] = {}
        for p in self._positions:
            und = p["underlying"]
            result[und] = result.get(und, GreeksResult.zero()) + p["greeks"]
        return result

    def net_delta_by_underlying(self) -> Dict[str, float]:
        """Net delta per underlying."""
        return {k: v.delta for k, v in self.greeks_by_underlying().items()}

    def net_gamma_by_underlying(self) -> Dict[str, float]:
        """Net gamma per underlying."""
        return {k: v.gamma for k, v in self.greeks_by_underlying().items()}

    def net_vega_by_underlying(self) -> Dict[str, float]:
        """Net vega per underlying."""
        return {k: v.vega for k, v in self.greeks_by_underlying().items()}

    def stress_scenarios(self) -> List[StressScenario]:
        """
        Compute P&L under standard stress scenarios using delta-gamma-vega approximation.

        Scenarios: spot +/-10%, vol +/-1 vol point, 1 day time decay.

        P&L approx = delta*dS + 0.5*gamma*dS^2 + vega*d_sigma + theta*dt
        """
        net = self.net_greeks()
        scenarios = []
        spot_shocks = [0.10, -0.10]
        vol_shocks = [0.01, -0.01]

        # Reference spot (sum of weighted spots)
        if self._positions:
            ref_spot = sum(p["spot"] * abs(p["qty"]) for p in self._positions) / max(sum(abs(p["qty"]) for p in self._positions), 1e-10)
        else:
            ref_spot = 100.0

        for spot_shock in spot_shocks:
            dS = ref_spot * spot_shock
            pnl = net.delta * dS + 0.5 * net.gamma * dS ** 2
            scenarios.append(StressScenario(
                label=f"Spot {'+' if spot_shock > 0 else ''}{spot_shock*100:.0f}%",
                spot_shock=spot_shock,
                vol_shock=0.0,
                time_decay_days=0.0,
                pnl=pnl,
            ))

        for vol_shock in vol_shocks:
            pnl = net.vega * vol_shock
            scenarios.append(StressScenario(
                label=f"Vol {'+' if vol_shock > 0 else ''}{vol_shock*100:.0f}vp",
                spot_shock=0.0,
                vol_shock=vol_shock,
                time_decay_days=0.0,
                pnl=pnl,
            ))

        # 1-day theta
        pnl_theta = net.theta
        scenarios.append(StressScenario(
            label="1 day decay",
            spot_shock=0.0,
            vol_shock=0.0,
            time_decay_days=1.0,
            pnl=pnl_theta,
        ))

        return scenarios

    def extended_stress(self) -> List[StressScenario]:
        """
        Extended scenario matrix: spot +/-5%/10%/20%, vol +/-5/10/20vp.
        """
        net = self.net_greeks()
        scenarios = []
        if self._positions:
            ref_spot = sum(p["spot"] * abs(p["qty"]) for p in self._positions) / max(sum(abs(p["qty"]) for p in self._positions), 1e-10)
        else:
            ref_spot = 100.0

        spot_shocks = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
        vol_shocks = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]

        for ss in spot_shocks:
            dS = ref_spot * ss
            for vs in vol_shocks:
                pnl = (
                    net.delta * dS
                    + 0.5 * net.gamma * dS ** 2
                    + net.vega * vs
                    + net.vanna * dS * vs
                    + 0.5 * net.volga * vs ** 2
                )
                scenarios.append(StressScenario(
                    label=f"S{ss*100:+.0f}% V{vs*100:+.0f}vp",
                    spot_shock=ss,
                    vol_shock=vs,
                    time_decay_days=0.0,
                    pnl=pnl,
                ))

        return scenarios

    def clear(self) -> None:
        """Remove all positions."""
        self._positions.clear()


# ---------------------------------------------------------------------------
# ImpliedGreeks
# ---------------------------------------------------------------------------

class ImpliedGreeks:
    """
    Market-implied Greeks extracted from observed option prices.

    Estimates Greeks by fitting a local polynomial to the market
    price surface and differentiating analytically.

    Primary use case: deriving gamma and vanna from market prices when
    the vol surface model may differ from Black-Scholes.
    """

    def __init__(self) -> None:
        pass

    def from_market_prices(
        self,
        spot_prices: np.ndarray,
        option_prices: np.ndarray,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> GreeksResult:
        """
        Estimate Greeks from a sequence of observed (spot, option_price) pairs.

        Fits a quadratic polynomial in spot to estimate delta and gamma.

        Parameters
        ----------
        spot_prices : ndarray
            Array of spot prices (sorted ascending).
        option_prices : ndarray
            Corresponding observed option prices.
        K, T, r, sigma, q : float
            Option parameters for BS fallback and rho/theta.
        """
        from lib.options.pricing import BlackScholes

        spots = np.asarray(spot_prices, dtype=float)
        prices = np.asarray(option_prices, dtype=float)

        if len(spots) < 3:
            # Fall back to BS
            ag = AnalyticalGreeks(option_type)
            return ag.compute(float(spots[len(spots) // 2]), K, T, r, sigma, q)

        # Fit quadratic: price ~ a + b*S + c*S^2
        coeffs = np.polyfit(spots, prices, 2)
        c, b, a = coeffs  # polyfit returns highest degree first

        S0 = float(np.median(spots))
        implied_delta = b + 2.0 * c * S0
        implied_gamma = 2.0 * c

        # Use BS for vega, theta, rho as we can't identify them from spot series alone
        bs = BlackScholes(S0, K, T, r, sigma, q, option_type)

        return GreeksResult(
            delta=implied_delta,
            gamma=implied_gamma,
            vega=bs.vega(),
            theta=bs.theta(),
            rho=bs.rho(),
            vanna=bs.vanna(),
            volga=bs.volga(),
            charm=bs.charm(),
            speed=bs.speed(),
            color=bs.color(),
            ultima=bs.ultima(),
            zomma=bs.zomma(),
        )

    def implied_delta_from_price_sensitivity(
        self,
        S: float,
        dS: float,
        price_up: float,
        price_down: float,
    ) -> float:
        """
        Estimate delta from two observed option prices at S+dS and S-dS.
        """
        return (price_up - price_down) / (2.0 * dS)

    def implied_gamma_from_price_curvature(
        self,
        S: float,
        dS: float,
        price_up: float,
        price_mid: float,
        price_down: float,
    ) -> float:
        """
        Estimate gamma from three observed option prices.
        """
        return (price_up - 2.0 * price_mid + price_down) / dS ** 2

    def implied_vol_smile(
        self,
        strikes: np.ndarray,
        prices: np.ndarray,
        S: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> np.ndarray:
        """
        Convert market prices to implied vols for a smile.

        Returns array of implied vols (NaN where inversion fails).
        """
        from lib.options.pricing import BlackScholes
        ivs = np.zeros(len(strikes))
        for i, (K, p) in enumerate(zip(strikes, prices)):
            ivs[i] = BlackScholes.implied_vol(p, S, K, T, r, q, option_type)
        return ivs
