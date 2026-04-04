"""
Options Greeks — first and second order analytical derivatives.

Implements all Greeks for Black-Scholes model:
- First order: Delta, Vega, Theta, Rho, Psi (dividend rho)
- Second order: Gamma, Vanna, Volga (Vomma), Charm, Color, Speed, Ultima
- Greeks surfaces: delta surface, vega surface across strikes/maturities
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Black-Scholes Greeks
# ---------------------------------------------------------------------------

class BlackScholesGreeks:
    """
    Analytical Greeks for European options under Black-Scholes.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration (years).
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    q : float
        Dividend yield.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
    ) -> None:
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

        # Pre-compute
        self._sqrt_T = np.sqrt(max(T, 1e-10))
        self._d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / \
                   (sigma * self._sqrt_T + 1e-12)
        self._d2 = self._d1 - sigma * self._sqrt_T
        self._N_d1 = stats.norm.cdf(self._d1)
        self._N_d2 = stats.norm.cdf(self._d2)
        self._N_neg_d1 = stats.norm.cdf(-self._d1)
        self._N_neg_d2 = stats.norm.cdf(-self._d2)
        self._n_d1 = stats.norm.pdf(self._d1)  # standard normal density
        self._disc_r = np.exp(-r * T)
        self._disc_q = np.exp(-q * T)

    # ------------------------------------------------------------------
    # First-order Greeks
    # ------------------------------------------------------------------

    def delta(self, option_type: str = "call") -> float:
        """
        Delta = dV/dS.

        Call: e^{-qT} N(d1)
        Put:  -e^{-qT} N(-d1)
        """
        if self.T <= 0:
            if option_type == "call":
                return 1.0 if self.S > self.K else 0.0
            return -1.0 if self.S < self.K else 0.0

        if option_type == "call":
            return float(self._disc_q * self._N_d1)
        return float(-self._disc_q * self._N_neg_d1)

    def vega(self) -> float:
        """
        Vega = dV/d_sigma.  Same for call and put.

        S * e^{-qT} * n(d1) * sqrt(T)
        Per 1% vol change: divide by 100.
        """
        if self.T <= 0:
            return 0.0
        return float(self.S * self._disc_q * self._n_d1 * self._sqrt_T)

    def theta(self, option_type: str = "call", per_day: bool = True) -> float:
        """
        Theta = dV/dt (time decay, sign convention: positive theta = decay).

        Parameters
        ----------
        per_day : bool
            If True, divide by 365 to get daily theta.
        """
        if self.T <= 0:
            return 0.0

        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q
        common = -S * self._disc_q * self._n_d1 * sigma / (2 * self._sqrt_T)

        if option_type == "call":
            theta_val = (
                common
                - r * K * self._disc_r * self._N_d2
                + q * S * self._disc_q * self._N_d1
            )
        else:
            theta_val = (
                common
                + r * K * self._disc_r * self._N_neg_d2
                - q * S * self._disc_q * self._N_neg_d1
            )

        if per_day:
            theta_val /= 365.0
        return float(theta_val)

    def rho(self, option_type: str = "call") -> float:
        """
        Rho = dV/dr.  Sensitivity to risk-free rate.
        Per 1% rate change: divide by 100.
        """
        if self.T <= 0:
            return 0.0

        if option_type == "call":
            return float(self.K * self.T * self._disc_r * self._N_d2)
        return float(-self.K * self.T * self._disc_r * self._N_neg_d2)

    def psi(self, option_type: str = "call") -> float:
        """
        Psi (dividend rho) = dV/dq.  Sensitivity to dividend yield.
        """
        if self.T <= 0:
            return 0.0

        if option_type == "call":
            return float(-self.S * self.T * self._disc_q * self._N_d1)
        return float(self.S * self.T * self._disc_q * self._N_neg_d1)

    # ------------------------------------------------------------------
    # Second-order Greeks
    # ------------------------------------------------------------------

    def gamma(self) -> float:
        """
        Gamma = d²V/dS².  Same for call and put.

        e^{-qT} * n(d1) / (S * sigma * sqrt(T))
        """
        if self.T <= 0:
            return 0.0
        return float(self._disc_q * self._n_d1 / (self.S * self.sigma * self._sqrt_T + 1e-12))

    def vanna(self) -> float:
        """
        Vanna = d²V/(dS d_sigma) = dDelta/d_sigma = dVega/dS.

        -e^{-qT} * n(d1) * d2 / sigma
        """
        if self.T <= 0:
            return 0.0
        return float(-self._disc_q * self._n_d1 * self._d2 / (self.sigma + 1e-12))

    def volga(self) -> float:
        """
        Volga (Vomma) = d²V/d_sigma².  Sensitivity of Vega to vol.

        Vega * d1 * d2 / sigma
        """
        if self.T <= 0:
            return 0.0
        return float(self.vega() * self._d1 * self._d2 / (self.sigma + 1e-12))

    def charm(self, option_type: str = "call") -> float:
        """
        Charm (DdeltaDtime) = dDelta/dt.  Daily decay of delta.
        """
        if self.T <= 0:
            return 0.0
        S, K, T, r, sigma, q = self.S, self.K, self.T, self.r, self.sigma, self.q

        term1 = q * self._disc_q * self._n_d1
        term2 = self._disc_q * self._n_d1 * (
            2 * (r - q) * T - self._d2 * sigma * self._sqrt_T
        ) / (2 * T * sigma * self._sqrt_T + 1e-12)

        if option_type == "call":
            return float(term1 * self._N_d1 - term2)
        return float(-term1 * self._N_neg_d1 - term2)

    def color(self) -> float:
        """
        Color (DgammaDtime) = dGamma/dt.
        """
        if self.T <= 0:
            return 0.0
        S, T, r, sigma, q = self.S, self.T, self.r, self.sigma, self.q
        gamma = self.gamma()
        term = (
            2 * q * T
            + 1
            + self._d1 * (
                2 * (r - q) * T - self._d2 * sigma * self._sqrt_T
            ) / (sigma * self._sqrt_T + 1e-12)
        )
        return float(-gamma * term / (2 * T + 1e-12))

    def speed(self) -> float:
        """
        Speed = dGamma/dS = d³V/dS³.
        """
        if self.T <= 0:
            return 0.0
        return float(-self.gamma() / self.S * (self._d1 / (self.sigma * self._sqrt_T) + 1))

    def ultima(self) -> float:
        """
        Ultima = dVolga/d_sigma = d³V/d_sigma³.
        """
        if self.T <= 0:
            return 0.0
        volga = self.volga()
        d1, d2, sigma = self._d1, self._d2, self.sigma
        return float(-volga / sigma * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2) / (sigma ** 2 + 1e-12))

    def zomma(self) -> float:
        """
        Zomma = dGamma/d_sigma.
        """
        if self.T <= 0:
            return 0.0
        return float(self.gamma() * (self._d1 * self._d2 - 1) / (self.sigma + 1e-12))

    # ------------------------------------------------------------------
    # All Greeks summary
    # ------------------------------------------------------------------

    def all_greeks(self, option_type: str = "call") -> Dict[str, float]:
        """Return all available Greeks in a dictionary."""
        return {
            "delta": round(self.delta(option_type), 6),
            "gamma": round(self.gamma(), 6),
            "vega": round(self.vega(), 6),
            "theta": round(self.theta(option_type), 6),
            "rho": round(self.rho(option_type), 6),
            "psi": round(self.psi(option_type), 6),
            "vanna": round(self.vanna(), 6),
            "volga": round(self.volga(), 6),
            "charm": round(self.charm(option_type), 6),
            "color": round(self.color(), 6),
            "speed": round(self.speed(), 6),
            "ultima": round(self.ultima(), 6),
            "zomma": round(self.zomma(), 6),
        }

    # ------------------------------------------------------------------
    # Numerical Greeks (finite difference verification)
    # ------------------------------------------------------------------

    def numerical_delta(self, option_type: str = "call", h: float = 0.01) -> float:
        """Central-difference numerical delta for verification."""
        from .pricing import BlackScholes
        bs_up = BlackScholes(self.S + h, self.K, self.T, self.r, self.sigma, self.q)
        bs_dn = BlackScholes(self.S - h, self.K, self.T, self.r, self.sigma, self.q)
        if option_type == "call":
            return (bs_up.call() - bs_dn.call()) / (2 * h)
        return (bs_up.put() - bs_dn.put()) / (2 * h)

    def numerical_gamma(self, h: float = 0.01) -> float:
        """Central-difference numerical gamma."""
        from .pricing import BlackScholes
        bs = BlackScholes(self.S, self.K, self.T, self.r, self.sigma, self.q)
        bs_up = BlackScholes(self.S + h, self.K, self.T, self.r, self.sigma, self.q)
        bs_dn = BlackScholes(self.S - h, self.K, self.T, self.r, self.sigma, self.q)
        return (bs_up.call() - 2 * bs.call() + bs_dn.call()) / h ** 2

    def numerical_vega(self, option_type: str = "call", h: float = 0.001) -> float:
        """Central-difference numerical vega."""
        from .pricing import BlackScholes
        bs_up = BlackScholes(self.S, self.K, self.T, self.r, self.sigma + h, self.q)
        bs_dn = BlackScholes(self.S, self.K, self.T, self.r, self.sigma - h, self.q)
        if option_type == "call":
            return (bs_up.call() - bs_dn.call()) / (2 * h)
        return (bs_up.put() - bs_dn.put()) / (2 * h)

    def greeks_verification(self, option_type: str = "call") -> pd.DataFrame:
        """Compare analytical and numerical Greeks."""
        analytical = self.all_greeks(option_type)
        rows = []
        for greek_name in ["delta", "gamma", "vega"]:
            analytical_val = analytical[greek_name]
            if greek_name == "delta":
                numerical_val = self.numerical_delta(option_type)
            elif greek_name == "gamma":
                numerical_val = self.numerical_gamma()
            else:
                numerical_val = self.numerical_vega(option_type)
            rows.append({
                "greek": greek_name,
                "analytical": round(analytical_val, 8),
                "numerical": round(numerical_val, 8),
                "abs_error": round(abs(analytical_val - numerical_val), 10),
            })
        return pd.DataFrame(rows).set_index("greek")


# ---------------------------------------------------------------------------
# Greeks surface across strikes and maturities
# ---------------------------------------------------------------------------

class GreeksSurface:
    """
    Compute Greek surfaces across a grid of strikes and maturities.

    Parameters
    ----------
    S : float
        Current spot price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    sigma : float
        Implied vol (flat surface) or use vol_surface for skew.
    """

    def __init__(
        self,
        S: float,
        r: float = 0.02,
        q: float = 0.0,
        sigma: float = 0.20,
    ) -> None:
        self.S = S
        self.r = r
        self.q = q
        self.sigma = sigma

    def delta_surface(
        self,
        strikes: List[float],
        maturities: List[float],
        option_type: str = "call",
    ) -> pd.DataFrame:
        """
        Delta surface: (maturities x strikes).

        Returns
        -------
        pd.DataFrame
        """
        data = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                g = BlackScholesGreeks(self.S, K, T, self.r, self.sigma, self.q)
                data[i, j] = g.delta(option_type)
        return pd.DataFrame(
            data.round(4),
            index=[f"T={T:.2f}" for T in maturities],
            columns=[f"K={K}" for K in strikes],
        )

    def gamma_surface(
        self,
        strikes: List[float],
        maturities: List[float],
    ) -> pd.DataFrame:
        """Gamma surface: (maturities x strikes)."""
        data = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                g = BlackScholesGreeks(self.S, K, T, self.r, self.sigma, self.q)
                data[i, j] = g.gamma()
        return pd.DataFrame(
            data.round(6),
            index=[f"T={T:.2f}" for T in maturities],
            columns=[f"K={K}" for K in strikes],
        )

    def vega_surface(
        self,
        strikes: List[float],
        maturities: List[float],
    ) -> pd.DataFrame:
        """Vega surface: (maturities x strikes)."""
        data = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                g = BlackScholesGreeks(self.S, K, T, self.r, self.sigma, self.q)
                data[i, j] = g.vega()
        return pd.DataFrame(
            data.round(4),
            index=[f"T={T:.2f}" for T in maturities],
            columns=[f"K={K}" for K in strikes],
        )

    def theta_surface(
        self,
        strikes: List[float],
        maturities: List[float],
        option_type: str = "call",
    ) -> pd.DataFrame:
        """Theta surface: (maturities x strikes)."""
        data = np.zeros((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                g = BlackScholesGreeks(self.S, K, T, self.r, self.sigma, self.q)
                data[i, j] = g.theta(option_type)
        return pd.DataFrame(
            data.round(6),
            index=[f"T={T:.2f}" for T in maturities],
            columns=[f"K={K}" for K in strikes],
        )

    def all_surfaces(
        self,
        strikes: List[float],
        maturities: List[float],
        option_type: str = "call",
    ) -> Dict[str, pd.DataFrame]:
        """Compute all major Greek surfaces."""
        return {
            "delta": self.delta_surface(strikes, maturities, option_type),
            "gamma": self.gamma_surface(strikes, maturities),
            "vega": self.vega_surface(strikes, maturities),
            "theta": self.theta_surface(strikes, maturities, option_type),
        }


# ---------------------------------------------------------------------------
# Delta-Gamma-Vega hedging
# ---------------------------------------------------------------------------

class GreeksHedge:
    """
    Construct delta/delta-gamma/delta-gamma-vega neutral hedging portfolios.

    Parameters
    ----------
    S : float
        Spot price.
    r : float
        Risk-free rate.
    q : float
        Dividend yield.
    """

    def __init__(self, S: float, r: float = 0.02, q: float = 0.0) -> None:
        self.S = S
        self.r = r
        self.q = q

    def delta_hedge_ratio(
        self,
        K: float,
        T: float,
        sigma: float,
        option_type: str = "call",
        position: float = 1.0,
    ) -> Dict[str, float]:
        """
        Delta hedge: short 'position' options, hedge with shares.

        Returns
        -------
        dict with 'shares_per_option' and portfolio Greeks.
        """
        g = BlackScholesGreeks(self.S, K, T, self.r, sigma, self.q)
        delta = g.delta(option_type)
        gamma = g.gamma()
        vega = g.vega()

        # Short option, long delta shares
        port_delta = -position * delta + position * delta  # = 0 by construction
        port_gamma = -position * gamma
        port_vega = -position * vega

        return {
            "shares_per_option": round(delta, 6),
            "option_delta": round(delta, 6),
            "option_gamma": round(gamma, 6),
            "option_vega": round(vega, 6),
            "portfolio_delta": round(port_delta, 8),
            "portfolio_gamma": round(port_gamma, 6),
            "portfolio_vega": round(port_vega, 6),
        }

    def delta_gamma_hedge(
        self,
        target_K: float,
        target_T: float,
        target_sigma: float,
        hedge_K: float,
        hedge_T: float,
        hedge_sigma: float,
        option_type: str = "call",
    ) -> Dict[str, float]:
        """
        Delta-Gamma neutral hedge using a second option.

        Solve for:
          n_hedge * Gamma_h = Gamma_target  (gamma neutralize)
          n_share = Delta_target - n_hedge * Delta_h  (delta neutralize)

        Returns
        -------
        dict with hedge ratios.
        """
        g_t = BlackScholesGreeks(self.S, target_K, target_T, self.r, target_sigma, self.q)
        g_h = BlackScholesGreeks(self.S, hedge_K, hedge_T, self.r, hedge_sigma, self.q)

        # Position in target: long 1
        target_delta = g_t.delta(option_type)
        target_gamma = g_t.gamma()
        hedge_delta = g_h.delta(option_type)
        hedge_gamma = g_h.gamma()

        if abs(hedge_gamma) < 1e-10:
            n_hedge = 0.0
        else:
            n_hedge = -target_gamma / hedge_gamma

        n_shares = -(target_delta + n_hedge * hedge_delta)

        return {
            "n_hedge_options": round(n_hedge, 6),
            "n_shares": round(n_shares, 6),
            "residual_delta": round(target_delta + n_hedge * hedge_delta + n_shares, 8),
            "residual_gamma": round(target_gamma + n_hedge * hedge_gamma, 8),
            "target_vega": round(g_t.vega(), 6),
            "hedge_vega": round(g_h.vega(), 6),
            "portfolio_vega": round(g_t.vega() + n_hedge * g_h.vega(), 6),
        }

    def delta_gamma_vega_hedge(
        self,
        target_K: float,
        target_T: float,
        target_sigma: float,
        hedge1_K: float,
        hedge1_T: float,
        hedge1_sigma: float,
        hedge2_K: float,
        hedge2_T: float,
        hedge2_sigma: float,
        option_type: str = "call",
    ) -> Dict[str, float]:
        """
        Delta-Gamma-Vega neutral hedge using two options + shares.

        Solve 3x3 linear system:
          [Gamma_h1, Gamma_h2] [n_h1]   [Gamma_target]
          [Vega_h1,  Vega_h2 ] [n_h2] = [Vega_target ]
          then n_shares = -(Delta_target + n_h1*Delta_h1 + n_h2*Delta_h2)

        Returns
        -------
        dict
        """
        def _g(K, T, sigma):
            return BlackScholesGreeks(self.S, K, T, self.r, sigma, self.q)

        g_t = _g(target_K, target_T, target_sigma)
        g_h1 = _g(hedge1_K, hedge1_T, hedge1_sigma)
        g_h2 = _g(hedge2_K, hedge2_T, hedge2_sigma)

        A = np.array([
            [g_h1.gamma(), g_h2.gamma()],
            [g_h1.vega(), g_h2.vega()],
        ])
        b = np.array([-g_t.gamma(), -g_t.vega()])

        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(A, b, rcond=None)[0]

        n_h1, n_h2 = float(x[0]), float(x[1])
        n_shares = -(g_t.delta(option_type) + n_h1 * g_h1.delta(option_type) +
                     n_h2 * g_h2.delta(option_type))

        return {
            "n_hedge1": round(n_h1, 6),
            "n_hedge2": round(n_h2, 6),
            "n_shares": round(n_shares, 6),
            "residual_delta": round(
                g_t.delta(option_type) + n_h1 * g_h1.delta(option_type) +
                n_h2 * g_h2.delta(option_type) + n_shares, 10
            ),
            "residual_gamma": round(
                g_t.gamma() + n_h1 * g_h1.gamma() + n_h2 * g_h2.gamma(), 10
            ),
            "residual_vega": round(
                g_t.vega() + n_h1 * g_h1.vega() + n_h2 * g_h2.vega(), 10
            ),
        }

    def pnl_explained(
        self,
        delta: float,
        gamma: float,
        vega: float,
        dS: float,
        d_sigma: float,
        theta: float = 0.0,
        dt: float = 1 / 252,
    ) -> Dict[str, float]:
        """
        Approximate P&L decomposition via Taylor expansion.

        P&L ≈ delta*dS + 0.5*gamma*dS^2 + vega*d_sigma + theta*dt

        Returns
        -------
        dict with each component and total.
        """
        delta_pnl = delta * dS
        gamma_pnl = 0.5 * gamma * dS ** 2
        vega_pnl = vega * d_sigma
        theta_pnl = theta * dt

        return {
            "delta_pnl": round(delta_pnl, 6),
            "gamma_pnl": round(gamma_pnl, 6),
            "vega_pnl": round(vega_pnl, 6),
            "theta_pnl": round(theta_pnl, 6),
            "total_approx_pnl": round(delta_pnl + gamma_pnl + vega_pnl + theta_pnl, 6),
        }
