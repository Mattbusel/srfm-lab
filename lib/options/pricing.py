"""
Options pricing models for the srfm-lab trading system.

Implements Black-Scholes, Bjerksund-Stensland 2002, Heston (Carr-Madan FFT),
Binomial CRR, and Monte Carlo pricers with a standardized interface.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT2PI = math.sqrt(2.0 * math.pi)
_MIN_VOL = 1e-8
_MAX_VOL = 20.0
_MIN_T = 1e-10


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Compute d1 for Black-Scholes formula."""
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Compute d2 for Black-Scholes formula."""
    return _d1(S, K, T, r, sigma, q) - sigma * math.sqrt(T)


def _nd(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _cnd(x: float) -> float:
    """Standard normal CDF."""
    return norm.cdf(x)


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

class BlackScholes:
    """
    Black-Scholes-Merton model for European options.

    Provides analytical pricing, all first- and second-order Greeks,
    and implied volatility via Newton-Raphson with bisection fallback.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility (decimal, e.g. 0.20 for 20%).
    q : float
        Continuous dividend yield (default 0).
    option_type : str
        'call' or 'put'.
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> None:
        self.S = float(S)
        self.K = float(K)
        self.T = max(float(T), _MIN_T)
        self.r = float(r)
        self.sigma = max(float(sigma), _MIN_VOL)
        self.q = float(q)
        self.option_type = option_type.lower()
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        self._refresh()

    def _refresh(self) -> None:
        self._d1 = _d1(self.S, self.K, self.T, self.r, self.sigma, self.q)
        self._d2 = self._d1 - self.sigma * math.sqrt(self.T)
        self._sqrtT = math.sqrt(self.T)
        self._df = math.exp(-self.r * self.T)
        self._dq = math.exp(-self.q * self.T)
        self._nd1 = _nd(self._d1)
        self._nd2 = _nd(self._d2)
        self._Nd1 = _cnd(self._d1)
        self._Nd2 = _cnd(self._d2)
        self._NNd1 = _cnd(-self._d1)
        self._NNd2 = _cnd(-self._d2)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        S: Optional[float] = None,
        K: Optional[float] = None,
        T: Optional[float] = None,
        r: Optional[float] = None,
        sigma: Optional[float] = None,
        q: Optional[float] = None,
    ) -> float:
        """Return option price. Accepts override parameters for convenience."""
        if any(v is not None for v in (S, K, T, r, sigma, q)):
            return BlackScholes(
                S if S is not None else self.S,
                K if K is not None else self.K,
                T if T is not None else self.T,
                r if r is not None else self.r,
                sigma if sigma is not None else self.sigma,
                q if q is not None else self.q,
                self.option_type,
            ).price()
        if self.option_type == "call":
            return self.call_price()
        return self.put_price()

    def call_price(self) -> float:
        """Black-Scholes call price."""
        return (
            self.S * self._dq * self._Nd1
            - self.K * self._df * self._Nd2
        )

    def put_price(self) -> float:
        """Black-Scholes put price."""
        return (
            self.K * self._df * self._NNd2
            - self.S * self._dq * self._NNd1
        )

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(self) -> float:
        """Option delta: dV/dS."""
        if self.option_type == "call":
            return self._dq * self._Nd1
        return -self._dq * self._NNd1

    def gamma(self) -> float:
        """Option gamma: d2V/dS2."""
        return self._dq * self._nd1 / (self.S * self.sigma * self._sqrtT)

    def vega(self) -> float:
        """Option vega: dV/dsigma (per 1% move in vol reported as full derivative)."""
        return self.S * self._dq * self._nd1 * self._sqrtT

    def theta(self) -> float:
        """Option theta: dV/dt (per calendar day, negative means time decay)."""
        common = (
            -self.S * self._dq * self._nd1 * self.sigma / (2.0 * self._sqrtT)
        )
        if self.option_type == "call":
            return (
                common
                - self.r * self.K * self._df * self._Nd2
                + self.q * self.S * self._dq * self._Nd1
            ) / 365.0
        return (
            common
            + self.r * self.K * self._df * self._NNd2
            - self.q * self.S * self._dq * self._NNd1
        ) / 365.0

    def rho(self) -> float:
        """Option rho: dV/dr (per 1% move in rate, full derivative)."""
        if self.option_type == "call":
            return self.K * self.T * self._df * self._Nd2
        return -self.K * self.T * self._df * self._NNd2

    def vanna(self) -> float:
        """Vanna: d2V/(dS dsigma)."""
        return -self._dq * self._nd1 * self._d2 / self.sigma

    def volga(self) -> float:
        """Volga (vomma): d2V/dsigma2."""
        return self.vega() * self._d1 * self._d2 / self.sigma

    def charm(self) -> float:
        """Charm (delta decay): d2V/(dS dt), per calendar day."""
        if self.option_type == "call":
            return (
                self._dq
                * self._nd1
                * (
                    2.0 * (self.r - self.q) * self._sqrtT
                    - self._d2 * self.sigma
                )
                / (2.0 * self.T * self.sigma * self._sqrtT)
                - self.q * self._dq * self._Nd1
            ) / 365.0
        return (
            self._dq
            * self._nd1
            * (
                2.0 * (self.r - self.q) * self._sqrtT
                - self._d2 * self.sigma
            )
            / (2.0 * self.T * self.sigma * self._sqrtT)
            + self.q * self._dq * self._NNd1
        ) / 365.0

    def speed(self) -> float:
        """Speed: d3V/dS3."""
        return -self.gamma() / self.S * (self._d1 / (self.sigma * self._sqrtT) + 1.0)

    def color(self) -> float:
        """Color (gamma decay): d3V/(dS2 dt), per calendar day."""
        return (
            -self._dq
            * self._nd1
            / (2.0 * self.S * self.T * self.sigma * self._sqrtT)
            * (
                2.0 * self.q * self.T
                + 1.0
                + self._d1
                * (2.0 * (self.r - self.q) * self.T - self._d2 * self.sigma * self._sqrtT)
                / (self.sigma * self._sqrtT)
            )
        ) / 365.0

    def ultima(self) -> float:
        """Ultima: d3V/dsigma3."""
        vega = self.vega()
        d1, d2 = self._d1, self._d2
        sigma = self.sigma
        return (
            -vega
            / sigma ** 2
            * (d1 * d2 * (1.0 - d1 * d2) + d1 ** 2 + d2 ** 2)
        )

    def zomma(self) -> float:
        """Zomma: d3V/(dS2 dsigma)."""
        return self.gamma() * (self._d1 * self._d2 - 1.0) / self.sigma

    def all_greeks(self) -> dict:
        """Return all Greeks in a dictionary."""
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta": self.theta(),
            "rho": self.rho(),
            "vanna": self.vanna(),
            "volga": self.volga(),
            "charm": self.charm(),
            "speed": self.speed(),
            "color": self.color(),
            "ultima": self.ultima(),
            "zomma": self.zomma(),
        }

    # ------------------------------------------------------------------
    # Implied volatility
    # ------------------------------------------------------------------

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float = 0.0,
        option_type: str = "call",
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> float:
        """
        Compute implied volatility from a market price.

        Uses Newton-Raphson iteration. Falls back to Brent bisection if
        Newton-Raphson diverges or fails to converge.

        Returns NaN if no solution is found in (MIN_VOL, MAX_VOL).
        """
        T = max(T, _MIN_T)
        option_type = option_type.lower()

        def objective(sigma: float) -> float:
            model = BlackScholes(S, K, T, r, sigma, q, option_type)
            return model.price() - market_price

        # Intrinsic check
        df = math.exp(-r * T)
        dq = math.exp(-q * T)
        if option_type == "call":
            intrinsic = max(S * dq - K * df, 0.0)
        else:
            intrinsic = max(K * df - S * dq, 0.0)

        if market_price < intrinsic - 1e-6:
            return float("nan")

        # Newton-Raphson
        sigma = 0.3
        for _ in range(max_iter):
            model = BlackScholes(S, K, T, r, sigma, q, option_type)
            price = model.price()
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            vega = model.vega()
            if abs(vega) < 1e-12:
                break
            sigma_new = sigma - diff / vega
            if sigma_new <= _MIN_VOL or sigma_new >= _MAX_VOL:
                break
            sigma = sigma_new

        # Bisection fallback via Brent
        try:
            lo_val = objective(_MIN_VOL + 1e-6)
            hi_val = objective(_MAX_VOL - 1e-6)
            if lo_val * hi_val > 0:
                return float("nan")
            result = brentq(objective, _MIN_VOL + 1e-6, _MAX_VOL - 1e-6, xtol=tol, maxiter=max_iter)
            return float(result)
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# Bjerksund-Stensland 2002
# ---------------------------------------------------------------------------

class BjerksundStensland2002:
    """
    Bjerksund-Stensland (2002) model for American options.

    Provides a closed-form approximation for American calls and puts
    on equity (with continuous dividend yield) and on futures.

    Reference: Bjerksund, P. and Stensland, G. (2002). "Closed Form Valuation
    of American Options." Discussion Paper NHH.
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> float:
        """
        Price an American option.

        Parameters
        ----------
        S : float
            Spot price.
        K : float
            Strike.
        T : float
            Time to expiry (years).
        r : float
            Risk-free rate.
        sigma : float
            Volatility.
        q : float
            Continuous dividend yield (for equity) or cost-of-carry = r for futures.
        option_type : str
            'call' or 'put'.
        """
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_VOL)
        option_type = option_type.lower()

        if option_type == "put":
            # Use put-call symmetry: American put(S,K,r,b) = American call(K,S,r,r-b)
            # where b = r - q
            b = r - q
            return self._call_price(K, S, T, r, r - b, sigma)

        b = r - q
        return self._call_price(S, K, T, r, b, sigma)

    def _call_price(self, S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
        """Internal BS2002 call pricing with cost-of-carry b."""
        # If b >= r the option should never be exercised early -> European
        if b >= r:
            return BlackScholes(S, K, T, r, sigma, max(r - b, 0.0), "call").call_price()

        # Intrinsic
        intrinsic = max(S - K, 0.0)

        sigma2 = sigma * sigma
        sqrtT = math.sqrt(T)

        # Half-period helper
        t1 = 0.5 * (math.sqrt(5.0) - 1.0) * T

        # Trigger price at t1
        beta = (0.5 - b / sigma2) + math.sqrt(
            (b / sigma2 - 0.5) ** 2 + 2.0 * r / sigma2
        )
        B_inf = beta / (beta - 1.0) * K
        B0 = max(K, r / (r - b) * K)
        ht1 = -(b * t1 + 2.0 * sigma * math.sqrt(t1)) * K ** 2 / ((B_inf - B0) * B0)
        ht2 = -(b * T + 2.0 * sigma * sqrtT) * K ** 2 / ((B_inf - B0) * B0)
        I1 = B0 + (B_inf - B0) * (1.0 - math.exp(ht1))
        I2 = B0 + (B_inf - B0) * (1.0 - math.exp(ht2))

        if S >= I2:
            return intrinsic

        alpha1 = (I1 - K) * I1 ** (-beta)
        alpha2 = (I2 - K) * I2 ** (-beta)

        return (
            alpha2 * S ** beta
            - alpha2 * self._phi(S, t1, beta, I2, I2, r, b, sigma)
            + self._phi(S, t1, 1.0, I2, I2, r, b, sigma)
            - self._phi(S, t1, 1.0, I1, I2, r, b, sigma)
            - K * self._phi(S, t1, 0.0, I2, I2, r, b, sigma)
            + K * self._phi(S, t1, 0.0, I1, I2, r, b, sigma)
            + alpha1 * self._phi(S, t1, beta, I1, I2, r, b, sigma)
            - alpha1 * self._psi(S, T, beta, I1, I2, I1, t1, r, b, sigma)
            + self._psi(S, T, 1.0, I1, I2, I1, t1, r, b, sigma)
            - self._psi(S, T, 1.0, K, I2, I1, t1, r, b, sigma)
            - K * self._psi(S, T, 0.0, I1, I2, I1, t1, r, b, sigma)
            + K * self._psi(S, T, 0.0, K, I2, I1, t1, r, b, sigma)
        )

    def _phi(
        self,
        S: float,
        T: float,
        gamma: float,
        H: float,
        I: float,
        r: float,
        b: float,
        sigma: float,
    ) -> float:
        """Phi function used in BS2002."""
        sigma2 = sigma * sigma
        sqrtT = math.sqrt(T)
        lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma2
        d = -(math.log(S / H) + (b + (gamma - 0.5) * sigma2) * T) / (sigma * sqrtT)
        kappa = 2.0 * b / sigma2 + (2.0 * gamma - 1.0)
        return (
            math.exp(lam * T)
            * S ** gamma
            * (
                _cnd(d)
                - (I / S) ** kappa * _cnd(d - 2.0 * math.log(I / S) / (sigma * sqrtT))
            )
        )

    def _psi(
        self,
        S: float,
        T: float,
        gamma: float,
        H: float,
        I2: float,
        I1: float,
        t1: float,
        r: float,
        b: float,
        sigma: float,
    ) -> float:
        """Psi function used in BS2002."""
        sigma2 = sigma * sigma
        sqrtT = math.sqrt(T)
        sqrtt1 = math.sqrt(t1)
        lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma2
        kappa = 2.0 * b / sigma2 + (2.0 * gamma - 1.0)

        d1 = -(math.log(S / I1) + (b + (gamma - 0.5) * sigma2) * t1) / (sigma * sqrtt1)
        d2 = -(math.log(I2 ** 2 / (S * I1)) + (b + (gamma - 0.5) * sigma2) * t1) / (sigma * sqrtt1)
        d3 = -(math.log(S / I1) - (b + (gamma - 0.5) * sigma2) * t1) / (sigma * sqrtt1)
        d4 = -(math.log(I2 ** 2 / (S * I1)) - (b + (gamma - 0.5) * sigma2) * t1) / (sigma * sqrtt1)

        e1 = -(math.log(S / H) + (b + (gamma - 0.5) * sigma2) * T) / (sigma * sqrtT)
        e2 = -(math.log(I2 ** 2 / (S * H)) + (b + (gamma - 0.5) * sigma2) * T) / (sigma * sqrtT)
        e3 = -(math.log(I1 ** 2 / (S * H)) + (b + (gamma - 0.5) * sigma2) * T) / (sigma * sqrtT)
        e4 = -(math.log(S * I1 ** 2 / (H * I2 ** 2)) + (b + (gamma - 0.5) * sigma2) * T) / (sigma * sqrtT)

        rho = math.sqrt(t1 / T)
        from scipy.stats import multivariate_normal

        def bivariate_cnd(a: float, b_: float, rho_: float) -> float:
            cov = [[1.0, rho_], [rho_, 1.0]]
            return multivariate_normal.cdf([a, b_], mean=[0.0, 0.0], cov=cov)

        return (
            math.exp(lam * T)
            * S ** gamma
            * (
                bivariate_cnd(-d1, -e1, rho)
                - (I2 / S) ** kappa * bivariate_cnd(-d2, -e2, rho)
                - (I1 / S) ** kappa * bivariate_cnd(-d3, -e3, -rho)
                + (I1 / I2) ** kappa * bivariate_cnd(-d4, -e4, -rho)
            )
        )

    def early_exercise_boundary(
        self,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        n_points: int = 50,
    ) -> np.ndarray:
        """
        Compute the early exercise boundary S*(t) for t in [0, T].

        Returns an (n_points, 2) array of (t, S*) pairs.
        """
        b = r - q
        if option_type.lower() == "put":
            b = r - q

        sigma2 = sigma * sigma
        beta = (0.5 - b / sigma2) + math.sqrt((b / sigma2 - 0.5) ** 2 + 2.0 * r / sigma2)
        if abs(beta - 1.0) < 1e-10:
            beta = 1.0 + 1e-10
        B_inf = beta / (beta - 1.0) * K
        B0 = max(K, r / (r - b) * K) if abs(r - b) > 1e-10 else K * 1e6

        ts = np.linspace(1e-4, T, n_points)
        boundaries = []
        for t in ts:
            ht = -(b * t + 2.0 * sigma * math.sqrt(t)) * K ** 2 / ((B_inf - B0) * B0)
            I = B0 + (B_inf - B0) * (1.0 - math.exp(ht))
            boundaries.append(I)
        return np.column_stack([ts, boundaries])

    def futures_price(
        self,
        F: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """
        Price an American option on a futures contract (Black model).

        For futures, the cost of carry b = 0.
        """
        return self.price(F, K, T, r, sigma, q=r, option_type=option_type)


# ---------------------------------------------------------------------------
# Heston Model (Carr-Madan FFT)
# ---------------------------------------------------------------------------

class HestonModel:
    """
    Heston (1993) stochastic volatility model.

    Pricing via the Carr-Madan (1999) FFT method.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of variance.
    theta : float
        Long-run variance level.
    sigma_v : float
        Volatility of variance (vol of vol).
    rho : float
        Correlation between spot and variance Brownian motions.
    v0 : float
        Initial variance (v0 = sigma0^2).
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma_v: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0

    def char_func(self, u: complex, S: float, T: float, r: float, q: float = 0.0) -> complex:
        """
        Heston characteristic function for log(S_T).

        Uses the formulation from Albrecher et al. (2007) that avoids
        the discontinuity in the complex logarithm.
        """
        kappa = self.kappa
        theta = self.theta
        sigma_v = self.sigma_v
        rho = self.rho
        v0 = self.v0

        b = kappa - 1j * u * rho * sigma_v
        d = cmath_sqrt(b ** 2 + sigma_v ** 2 * (1j * u + u ** 2))
        g = (b - d) / (b + d)

        # Avoid divide-by-zero for very short T
        if abs(1.0 - g * cmath_exp(-d * T)) < 1e-30:
            return complex(1.0, 0.0)

        exp_dT = cmath_exp(-d * T)
        C = (r - q) * 1j * u * T + kappa * theta / sigma_v ** 2 * (
            (b - d) * T - 2.0 * cmath_log((1.0 - g * exp_dT) / (1.0 - g))
        )
        D = (b - d) / sigma_v ** 2 * (1.0 - exp_dT) / (1.0 - g * exp_dT)
        return cmath_exp(C + D * v0 + 1j * u * math.log(S))

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float = None,
        q: float = 0.0,
        option_type: str = "call",
        N: int = 4096,
        alpha: float = 1.5,
        eta: float = 0.25,
    ) -> float:
        """
        Price a European option via Carr-Madan FFT.

        Parameters
        ----------
        N : int
            Number of FFT points (power of 2).
        alpha : float
            Damping factor for the modified characteristic function.
        eta : float
            Grid spacing in frequency domain.
        """
        T = max(T, _MIN_T)
        lam = 2.0 * math.pi / (N * eta)
        b = math.pi / eta  # upper limit in log-strike

        ks = -b + lam * np.arange(N)
        vs = eta * np.arange(N)

        # Modified characteristic function
        def modified_char(v: float) -> complex:
            u = v - (alpha + 1.0) * 1j
            cf = self.char_func(u, S, T, r, q)
            num = math.exp(-r * T) * cf
            denom = alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v
            if abs(denom) < 1e-20:
                return complex(0.0, 0.0)
            return num / denom

        # Build the integrand
        psi = np.array([modified_char(v) for v in vs], dtype=complex)
        # Simpson weights
        delta = np.zeros(N)
        delta[0] = 1.0
        j = np.arange(N)
        weights = (3.0 + (-1.0) ** j - delta) / 3.0
        weights[0] = 1.0 / 3.0

        x = np.exp(1j * b * vs) * psi * weights * eta
        prices_fft = np.real(np.fft.fft(x)) * math.exp(-alpha * ks) / math.pi

        # Interpolate at log(K)
        log_K = math.log(K)
        call_price = float(np.interp(log_K, ks, prices_fft))
        call_price = max(call_price, 0.0)

        if option_type.lower() == "call":
            return call_price
        # Put via put-call parity
        parity_put = call_price - S * math.exp(-q * T) + K * math.exp(-r * T)
        return max(parity_put, 0.0)

    def calibrate(
        self,
        market_prices: list,
        strikes: list,
        expiries: list,
        S: float,
        r: float,
        q: float = 0.0,
        option_types: list = None,
    ) -> dict:
        """
        Calibrate Heston parameters to market option prices.

        Uses L-BFGS-B optimisation of sum-of-squared pricing errors.

        Returns a dict of calibrated parameters.
        """
        if option_types is None:
            option_types = ["call"] * len(market_prices)

        def objective(params):
            kappa, theta, sigma_v, rho, v0 = params
            if (
                kappa <= 0
                or theta <= 0
                or sigma_v <= 0
                or abs(rho) >= 1.0
                or v0 <= 0
                or 2.0 * kappa * theta <= sigma_v ** 2  # Feller condition
            ):
                return 1e10
            self.kappa = kappa
            self.theta = theta
            self.sigma_v = sigma_v
            self.rho = rho
            self.v0 = v0
            err = 0.0
            for mp, K, T, ot in zip(market_prices, strikes, expiries, option_types):
                model_p = self.price(S, K, T, r, q=q, option_type=ot)
                err += (model_p - mp) ** 2
            return err

        x0 = [self.kappa, self.theta, self.sigma_v, self.rho, self.v0]
        bounds = [(0.01, 20.0), (0.001, 2.0), (0.01, 5.0), (-0.999, 0.999), (0.0001, 2.0)]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        kappa, theta, sigma_v, rho, v0 = result.x
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho
        self.v0 = v0
        return {
            "kappa": kappa,
            "theta": theta,
            "sigma_v": sigma_v,
            "rho": rho,
            "v0": v0,
            "rmse": math.sqrt(result.fun / len(market_prices)),
            "success": result.success,
        }


def cmath_sqrt(z: complex) -> complex:
    """Complex square root that handles both real and complex inputs."""
    import cmath
    return cmath.sqrt(z)


def cmath_exp(z: complex) -> complex:
    """Complex exponential."""
    import cmath
    return cmath.exp(z)


def cmath_log(z: complex) -> complex:
    """Complex natural log."""
    import cmath
    return cmath.log(z)


# ---------------------------------------------------------------------------
# Binomial Tree (CRR)
# ---------------------------------------------------------------------------

class BinomialTree:
    """
    Cox-Ross-Rubinstein (CRR) binomial tree model.

    Supports European and American options with discrete or continuous
    dividend handling.

    Parameters
    ----------
    n_steps : int
        Number of time steps in the tree.
    """

    def __init__(self, n_steps: int = 200) -> None:
        self.n_steps = n_steps

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        american: bool = False,
        discrete_dividends: Optional[list] = None,
    ) -> float:
        """
        Price an option using the CRR binomial tree.

        Parameters
        ----------
        discrete_dividends : list of (ex_date_fraction, amount), optional
            Discrete cash dividends as list of (t/T, dividend_amount) tuples.
        """
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_VOL)
        n = self.n_steps
        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        pu = (math.exp((r - q) * dt) - d) / (u - d)
        pd = 1.0 - pu

        if pu < 0.0 or pu > 1.0:
            warnings.warn("Risk-neutral probabilities outside [0,1]; tree may be unstable")
            pu = max(min(pu, 1.0), 0.0)
            pd = 1.0 - pu

        is_call = option_type.lower() == "call"

        # Build asset price tree
        # Adjust S for discrete dividends by subtracting PV of dividends
        S_adj = S
        if discrete_dividends:
            for t_frac, div_amount in discrete_dividends:
                t_div = t_frac * T
                if t_div > 0:
                    S_adj -= div_amount * math.exp(-r * t_div)
        S_adj = max(S_adj, 1e-10)

        # Terminal asset prices (step n)
        j_arr = np.arange(n + 1, dtype=float)
        asset = S_adj * u ** (n - 2.0 * j_arr)

        # Terminal payoffs
        if is_call:
            values = np.maximum(asset - K, 0.0)
        else:
            values = np.maximum(K - asset, 0.0)

        # Backward induction
        for i in range(n - 1, -1, -1):
            # Roll back option values
            values = disc * (pu * values[:-1] + pd * values[1:])
            if american:
                # Asset prices at step i: S * u^(i - 2j) for j=0..i
                j_i = np.arange(i + 1, dtype=float)
                asset_i = S_adj * u ** (i - 2.0 * j_i)
                if is_call:
                    intrinsic = np.maximum(asset_i - K, 0.0)
                else:
                    intrinsic = np.maximum(K - asset_i, 0.0)
                values = np.maximum(values, intrinsic)

        return float(values[0])

    def price_american_put(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Convenience method for American put pricing."""
        return self.price(S, K, T, r, sigma, q, option_type="put", american=True)

    def price_american_call(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
        """Convenience method for American call pricing."""
        return self.price(S, K, T, r, sigma, q, option_type="call", american=True)


# ---------------------------------------------------------------------------
# Monte Carlo Pricer
# ---------------------------------------------------------------------------

class MonteCarloPricer:
    """
    Monte Carlo pricer for European options using GBM paths.

    Features:
    - Antithetic variates for variance reduction.
    - Black-Scholes control variates for further variance reduction.
    - Finite-difference Greeks estimation.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths.
    n_steps : int
        Number of time steps per path.
    seed : int, optional
        Random seed for reproducibility.
    use_antithetic : bool
        If True, use antithetic variates.
    use_control_variate : bool
        If True, apply BS control variate correction.
    """

    def __init__(
        self,
        n_paths: int = 50000,
        n_steps: int = 252,
        seed: Optional[int] = None,
        use_antithetic: bool = True,
        use_control_variate: bool = True,
    ) -> None:
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate
        self._rng = np.random.default_rng(seed)

    def _simulate_paths(self, S: float, T: float, r: float, sigma: float, q: float) -> np.ndarray:
        """Simulate GBM terminal prices."""
        n = self.n_paths // 2 if self.use_antithetic else self.n_paths
        dt = T / self.n_steps
        drift = (r - q - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt)
        Z = self._rng.standard_normal((n, self.n_steps))
        if self.use_antithetic:
            Z = np.vstack([Z, -Z])
        log_returns = drift + diffusion * Z
        log_S = math.log(S) + np.sum(log_returns, axis=1)
        return np.exp(log_S)

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> float:
        """Return Monte Carlo estimate of option price."""
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_VOL)
        is_call = option_type.lower() == "call"

        ST = self._simulate_paths(S, T, r, sigma, q)
        if is_call:
            payoffs = np.maximum(ST - K, 0.0)
        else:
            payoffs = np.maximum(K - ST, 0.0)

        disc_factor = math.exp(-r * T)

        if not self.use_control_variate:
            return float(disc_factor * np.mean(payoffs))

        # Control variate: use BS call/put as control
        bs = BlackScholes(S, K, T, r, sigma, q, option_type)
        bs_price = bs.price()
        bs_payoffs = np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0)
        cv_expected = bs_price / disc_factor  # E[payoff] under BS

        # Estimate optimal beta
        cov_mat = np.cov(payoffs, bs_payoffs)
        if cov_mat[1, 1] > 1e-20:
            beta_cv = cov_mat[0, 1] / cov_mat[1, 1]
        else:
            beta_cv = 0.0

        adjusted = payoffs - beta_cv * (bs_payoffs - cv_expected)
        return float(disc_factor * np.mean(adjusted))

    def price_with_ci(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Return (price, lower_CI, upper_CI) at the specified confidence level.
        """
        T = max(T, _MIN_T)
        sigma = max(sigma, _MIN_VOL)
        is_call = option_type.lower() == "call"

        ST = self._simulate_paths(S, T, r, sigma, q)
        if is_call:
            payoffs = np.maximum(ST - K, 0.0)
        else:
            payoffs = np.maximum(K - ST, 0.0)

        disc_factor = math.exp(-r * T)
        discounted = disc_factor * payoffs
        mean = float(np.mean(discounted))
        se = float(np.std(discounted) / math.sqrt(len(discounted)))
        z = norm.ppf(0.5 + confidence / 2.0)
        return mean, mean - z * se, mean + z * se

    # ------------------------------------------------------------------
    # Finite-difference Greeks
    # ------------------------------------------------------------------

    def delta(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> float:
        """Central finite-difference delta."""
        h = S * 0.01
        pu = self.price(S + h, K, T, r, sigma, q, option_type)
        pd = self.price(S - h, K, T, r, sigma, q, option_type)
        return (pu - pd) / (2.0 * h)

    def gamma(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> float:
        """Central finite-difference gamma."""
        h = S * 0.01
        pu = self.price(S + h, K, T, r, sigma, q, option_type)
        p0 = self.price(S, K, T, r, sigma, q, option_type)
        pd = self.price(S - h, K, T, r, sigma, q, option_type)
        return (pu - 2.0 * p0 + pd) / h ** 2

    def vega(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> float:
        """Central finite-difference vega."""
        h = 0.001
        pu = self.price(S, K, T, r, sigma + h, q, option_type)
        pd = self.price(S, K, T, r, sigma - h, q, option_type)
        return (pu - pd) / (2.0 * h)

    def theta(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> float:
        """Forward finite-difference theta (per day)."""
        h = 1.0 / 365.0
        if T <= h:
            return 0.0
        pu = self.price(S, K, T - h, r, sigma, q, option_type)
        p0 = self.price(S, K, T, r, sigma, q, option_type)
        return pu - p0

    def rho(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: str = "call") -> float:
        """Central finite-difference rho."""
        h = 0.0001
        pu = self.price(S, K, T, r + h, sigma, q, option_type)
        pd = self.price(S, K, T, r - h, sigma, q, option_type)
        return (pu - pd) / (2.0 * h)

    def all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call",
    ) -> dict:
        """Compute all first-order Greeks via finite differences."""
        return {
            "delta": self.delta(S, K, T, r, sigma, q, option_type),
            "gamma": self.gamma(S, K, T, r, sigma, q, option_type),
            "vega": self.vega(S, K, T, r, sigma, q, option_type),
            "theta": self.theta(S, K, T, r, sigma, q, option_type),
            "rho": self.rho(S, K, T, r, sigma, q, option_type),
        }
