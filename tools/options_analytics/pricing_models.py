"""
pricing_models.py — Options pricing model library.

Models
------
* BlackScholes     — closed-form European price + Greeks + IV solver
* BinomialTree     — Cox-Ross-Rubinstein American/European option pricing
* MonteCarlo       — GBM + Heston stochastic-vol path-dependent pricing
* SABR             — Hagan et al. IV approximation + calibration
* LocalVol         — Dupire local vol surface from market prices (discrete)
* ModelComparison  — side-by-side price + uncertainty estimate

All functions are fully vectorized where possible.

Usage
-----
    from pricing_models import BlackScholes, BinomialTree, MonteCarlo, SABR, LocalVol

    bs  = BlackScholes()
    p   = bs.price(S=450, K=455, T=0.25, r=0.05, sigma=0.20, option_type="call")
    g   = bs.greeks(S=450, K=455, T=0.25, r=0.05, sigma=0.20, option_type="call")
    iv  = bs.implied_vol(market_price=8.5, S=450, K=455, T=0.25, r=0.05, option_type="call")

    bt  = BinomialTree(steps=500)
    p_am = bt.price(S=450, K=455, T=0.25, r=0.05, sigma=0.20, option_type="put", american=True)

    mc  = MonteCarlo(n_paths=50_000, n_steps=252)
    p_mc = mc.price_gbm(S=450, K=455, T=0.25, r=0.05, sigma=0.20, option_type="call")
    p_heston = mc.price_heston(S=450, K=455, T=0.25, r=0.05,
                               v0=0.04, kappa=2.0, theta=0.04, xi=0.4, rho=-0.7,
                               option_type="call")

    sabr = SABR(beta=0.5)
    iv_sabr = sabr.vol(F=450, K=455, T=0.25, alpha=0.3, rho=-0.3, nu=0.4)

    comp = ModelComparison()
    report = comp.compare(S=450, K=455, T=0.25, r=0.05, sigma=0.20, option_type="call")
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize, differential_evolution
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline

RISK_FREE_RATE = 0.0525
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PriceResult:
    price: float
    model: str
    option_type: str
    S: float
    K: float
    T: float
    r: float
    sigma: Optional[float]
    extra: Dict = field(default_factory=dict)


@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float     # per calendar day
    vega: float      # per 1 vol pt
    rho: float
    vanna: float
    volga: float
    charm: float


@dataclass
class ModelComparisonResult:
    option_type: str
    S: float
    K: float
    T: float
    r: float
    sigma_input: float
    prices: Dict[str, float]
    price_mean: float
    price_std: float
    model_risk_spread: float   # max - min


@dataclass
class SABRParams:
    alpha: float   # vol of vol (initial vol)
    beta: float    # CEV exponent [0,1]
    rho: float     # correlation
    nu: float      # vol of vol


# ---------------------------------------------------------------------------
# 1. Black-Scholes
# ---------------------------------------------------------------------------

class BlackScholes:
    """
    Complete Black-Scholes-Merton implementation.

    All methods are static and operate on scalars. For vectorized
    operations, use the numpy_* variants.
    """

    # ---- Core price ----

    @staticmethod
    def price(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
    ) -> float:
        """European option price."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        disc = math.exp(-r * T)
        if option_type == "call":
            return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def numpy_price(
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: float,
        sigma: np.ndarray,
        option_type: str = "call",
    ) -> np.ndarray:
        """Vectorized price computation."""
        safe_T = np.maximum(T, 1e-8)
        safe_sig = np.maximum(sigma, 1e-8)
        d1 = (np.log(S / K) + (r + 0.5 * safe_sig**2) * safe_T) / (safe_sig * np.sqrt(safe_T))
        d2 = d1 - safe_sig * np.sqrt(safe_T)
        disc = np.exp(-r * safe_T)
        if option_type == "call":
            p = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        else:
            p = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        # Handle expired
        intrinsic = np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
        return np.where(T <= 0, intrinsic, p)

    # ---- Greeks ----

    @staticmethod
    def greeks(
        S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"
    ) -> GreeksResult:
        """Full first and second-order Greeks."""
        if T <= 0 or sigma <= 0:
            return GreeksResult(
                delta=(1.0 if (option_type == "call" and S > K) else 0.0),
                gamma=0, theta=0, vega=0, rho=0, vanna=0, volga=0, charm=0,
            )
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        nd1, nd2 = norm.cdf(d1), norm.cdf(d2)
        npd1 = norm.pdf(d1)
        disc = math.exp(-r * T)

        delta = nd1 if option_type == "call" else nd1 - 1.0
        gamma = npd1 / (S * sigma * sqrt_T)
        vega = S * npd1 * sqrt_T / 100.0   # per 1-vol-point
        if option_type == "call":
            theta = (-S * npd1 * sigma / (2 * sqrt_T) - r * K * disc * nd2) / 365.0
            rho = K * T * disc * nd2 / 100.0
        else:
            theta = (-S * npd1 * sigma / (2 * sqrt_T) + r * K * disc * norm.cdf(-d2)) / 365.0
            rho = -K * T * disc * norm.cdf(-d2) / 100.0

        vanna = -npd1 * d2 / sigma
        volga = vega * 100 * d1 * d2 / sigma   # raw, before /100 scaling
        charm = -npd1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T) / 365.0

        return GreeksResult(
            delta=delta, gamma=gamma, theta=theta, vega=vega,
            rho=rho, vanna=vanna, volga=volga, charm=charm,
        )

    # ---- IV solver ----

    @staticmethod
    def implied_vol(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        tol: float = 1e-8,
    ) -> Optional[float]:
        """Solve for implied volatility using Brent's method."""
        if market_price <= 0 or T <= 0:
            return None
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        if market_price <= intrinsic * (1 - 1e-6):
            return None

        def obj(sigma: float) -> float:
            return BlackScholes.price(S, K, T, r, sigma, option_type) - market_price

        try:
            lo, hi = 1e-4, 10.0
            if obj(lo) * obj(hi) > 0:
                return None
            return float(brentq(obj, lo, hi, xtol=tol, maxiter=300))
        except Exception:
            return None

    # ---- Put-call parity ----

    @staticmethod
    def put_call_parity_check(
        call_price: float, put_price: float, S: float, K: float, T: float, r: float
    ) -> float:
        """Returns deviation from put-call parity: C - P - S + K*e^{-rT}."""
        disc = math.exp(-r * T)
        return call_price - put_price - S + K * disc


# ---------------------------------------------------------------------------
# 2. Binomial Tree (CRR)
# ---------------------------------------------------------------------------

class BinomialTree:
    """
    Cox-Ross-Rubinstein binomial tree for American and European options.

    Supports early exercise for American puts/calls.
    """

    def __init__(self, steps: int = 500):
        self.steps = steps

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        american: bool = False,
    ) -> float:
        """Price option using CRR binomial tree."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

        N = self.steps
        dt = T / N
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp(r * dt) - d) / (u - d)
        q = 1.0 - p

        # Terminal asset prices
        j = np.arange(N + 1)
        S_T = S * (u ** (N - j)) * (d ** j)

        # Terminal payoffs
        if option_type == "call":
            V = np.maximum(S_T - K, 0.0)
        else:
            V = np.maximum(K - S_T, 0.0)

        # Backward induction
        for i in range(N - 1, -1, -1):
            S_node = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
            V = disc * (p * V[:-1] + q * V[1:])
            if american:
                if option_type == "call":
                    V = np.maximum(V, S_node - K)
                else:
                    V = np.maximum(V, K - S_node)

        return float(V[0])

    def price_result(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        american: bool = False,
    ) -> PriceResult:
        p = self.price(S, K, T, r, sigma, option_type, american)
        model_name = f"Binomial({'American' if american else 'European'}, N={self.steps})"
        return PriceResult(price=p, model=model_name, option_type=option_type,
                           S=S, K=K, T=T, r=r, sigma=sigma)

    def implied_vol(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        american: bool = True,
    ) -> Optional[float]:
        """Solve for IV using the binomial tree (slower, but handles Americans)."""
        if market_price <= 0 or T <= 0:
            return None
        try:
            def obj(sig: float) -> float:
                return self.price(S, K, T, r, sig, option_type, american) - market_price
            if obj(0.001) * obj(10.0) > 0:
                return None
            return float(brentq(obj, 0.001, 10.0, xtol=1e-6, maxiter=100))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 3. Monte Carlo (GBM + Heston)
# ---------------------------------------------------------------------------

class MonteCarlo:
    """
    Monte Carlo pricing engine.

    Supports:
    - Geometric Brownian Motion (risk-neutral GBM)
    - Heston stochastic volatility model
    - Antithetic variates for variance reduction
    """

    def __init__(self, n_paths: int = 50_000, n_steps: int = 252, seed: int = RNG_SEED):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

    # ---- GBM ----

    def price_gbm(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        antithetic: bool = True,
    ) -> float:
        """European option via GBM Monte Carlo with antithetic variates."""
        if T <= 0:
            return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

        dt = T / self.n_steps
        n = self.n_paths // 2 if antithetic else self.n_paths

        Z = self.rng.standard_normal((n, self.n_steps))
        if antithetic:
            Z = np.vstack([Z, -Z])

        log_S = np.log(S) + np.cumsum(
            (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z, axis=1
        )
        S_T = np.exp(log_S[:, -1])

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        return float(math.exp(-r * T) * payoffs.mean())

    def price_gbm_path_dependent(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        payoff_fn,          # callable(paths: ndarray) → ndarray of payoffs
    ) -> Tuple[float, float]:
        """
        Generic path-dependent option pricer.
        payoff_fn receives shape (n_paths, n_steps+1) array of S values.
        Returns (price, stderr).
        """
        dt = T / self.n_steps
        Z = self.rng.standard_normal((self.n_paths, self.n_steps))
        increments = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
        log_paths = np.zeros((self.n_paths, self.n_steps + 1))
        log_paths[:, 0] = math.log(S)
        log_paths[:, 1:] = math.log(S) + np.cumsum(increments, axis=1)
        paths = np.exp(log_paths)

        payoffs = payoff_fn(paths)
        disc = math.exp(-r * T)
        prices = disc * payoffs
        return float(prices.mean()), float(prices.std() / math.sqrt(self.n_paths))

    # ---- Heston SV ----

    def price_heston(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        v0: float,          # initial variance
        kappa: float,       # mean reversion speed
        theta: float,       # long-run variance
        xi: float,          # vol of vol
        rho: float,         # S-V correlation
        option_type: str = "call",
        antithetic: bool = True,
    ) -> float:
        """
        Heston stochastic volatility model via Euler-Maruyama Monte Carlo.

        dS = r*S*dt + sqrt(V)*S*dW1
        dV = kappa*(theta-V)*dt + xi*sqrt(V)*dW2
        corr(dW1, dW2) = rho
        """
        if T <= 0:
            return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

        dt = T / self.n_steps
        n = self.n_paths // 2 if antithetic else self.n_paths

        # Correlated Brownian motions
        Z1 = self.rng.standard_normal((n, self.n_steps))
        Z2 = rho * Z1 + math.sqrt(1 - rho**2) * self.rng.standard_normal((n, self.n_steps))
        if antithetic:
            Z1 = np.vstack([Z1, -Z1])
            Z2 = np.vstack([Z2, -Z2])

        log_S_paths = np.empty(self.n_paths)
        V = np.full(self.n_paths, v0)
        log_S_cur = np.full(self.n_paths, math.log(S))

        sqrt_dt = math.sqrt(dt)
        for i in range(self.n_steps):
            V_pos = np.maximum(V, 0.0)  # reflection at 0 (simple truncation)
            sqrt_V = np.sqrt(V_pos)
            log_S_cur += (r - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z1[:, i]
            V += kappa * (theta - V_pos) * dt + xi * sqrt_V * sqrt_dt * Z2[:, i]

        S_T = np.exp(log_S_cur)

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        return float(math.exp(-r * T) * payoffs.mean())

    def confidence_interval(
        self, payoffs: np.ndarray, r: float, T: float, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Return (lower, upper) CI for discounted mean payoff."""
        from scipy.stats import t as t_dist
        disc_payoffs = math.exp(-r * T) * payoffs
        n = len(disc_payoffs)
        mean = disc_payoffs.mean()
        se = disc_payoffs.std() / math.sqrt(n)
        t_val = t_dist.ppf((1 + confidence) / 2, df=n - 1)
        return float(mean - t_val * se), float(mean + t_val * se)


# ---------------------------------------------------------------------------
# 4. SABR model
# ---------------------------------------------------------------------------

class SABR:
    """
    SABR (Stochastic Alpha Beta Rho) model by Hagan et al. (2002).

    Provides:
    - Lognormal IV approximation under SABR dynamics
    - Parameter calibration to market smile via least squares
    """

    def __init__(self, beta: float = 0.5):
        """
        beta: CEV exponent [0, 1].
          0 = normal/absolute, 0.5 = square-root (CIR-like), 1 = lognormal
        """
        self.beta = beta

    def vol(
        self, F: float, K: float, T: float, alpha: float, rho: float, nu: float
    ) -> float:
        """
        Hagan et al. lognormal SABR implied vol formula.

        Parameters
        ----------
        F : Forward price
        K : Strike
        T : Time to expiry (years)
        alpha : Initial vol (SABR alpha)
        rho : Correlation F-sigma
        nu : Vol of vol
        """
        beta = self.beta
        if abs(F - K) < 1e-8:
            return self._atm_vol(F, T, alpha, rho, nu, beta)

        log_FK = math.log(F / K)
        FK_mid = math.sqrt(F * K)
        FK_mid_beta = FK_mid ** (1 - beta)

        # z factor
        z = (nu / alpha) * FK_mid_beta * log_FK

        # x(z) factor
        x_z = math.log(
            (math.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho)
        )

        # Numerator
        alpha_term = alpha / (FK_mid_beta * (
            1
            + ((1 - beta)**2 / 24) * log_FK**2
            + ((1 - beta)**4 / 1920) * log_FK**4
        ))

        correction = (
            1
            + (
                ((1 - beta)**2 / 24) * alpha**2 / FK_mid_beta**2
                + (rho * beta * nu * alpha) / (4 * FK_mid_beta)
                + (2 - 3 * rho**2) / 24 * nu**2
            ) * T
        )

        if abs(x_z) < 1e-8:
            sigma = alpha_term * correction
        else:
            sigma = alpha_term * (z / x_z) * correction

        return max(sigma, 1e-6)

    def _atm_vol(
        self, F: float, T: float, alpha: float, rho: float, nu: float, beta: float
    ) -> float:
        """ATM SABR IV (F == K special case)."""
        F_beta = F ** (1 - beta)
        correction = 1 + (
            ((1 - beta)**2 / 24) * alpha**2 / F_beta**2
            + (rho * beta * nu * alpha) / (4 * F_beta)
            + (2 - 3 * rho**2) / 24 * nu**2
        ) * T
        return alpha / F_beta * correction

    def smile(
        self,
        F: float,
        strikes: np.ndarray,
        T: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> np.ndarray:
        """Compute SABR smile across a strike grid."""
        return np.array([self.vol(F, K, T, alpha, rho, nu) for K in strikes])

    def calibrate(
        self,
        F: float,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        T: float,
        alpha0: float = 0.3,
        rho0: float = -0.3,
        nu0: float = 0.4,
    ) -> SABRParams:
        """
        Calibrate SABR alpha, rho, nu to market smile.
        Beta is fixed at construction.
        """
        def objective(params: np.ndarray) -> float:
            alpha, rho, nu = params
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
                return 1e9
            try:
                model_ivs = self.smile(F, strikes, T, alpha, rho, nu)
                return float(np.sum((model_ivs - market_ivs)**2))
            except Exception:
                return 1e9

        bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]
        res = minimize(
            objective, [alpha0, rho0, nu0],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-14},
        )

        alpha, rho, nu = res.x
        return SABRParams(alpha=alpha, beta=self.beta, rho=rho, nu=nu)

    def implied_vol_from_params(
        self,
        F: float,
        K: float,
        T: float,
        params: SABRParams,
    ) -> float:
        return self.vol(F, K, T, params.alpha, params.rho, params.nu)


# ---------------------------------------------------------------------------
# 5. Local volatility (Dupire)
# ---------------------------------------------------------------------------

class LocalVol:
    """
    Dupire local volatility surface computed from a discrete market
    IV surface.

    Dupire formula (in calendar-time parameterization):
        sigma_loc²(K, T) = (∂C/∂T) / (0.5 * K² * ∂²C/∂K²)

    where C(K, T) is the market call price surface.

    Implementation uses finite differences on a grid of BS-priced calls
    derived from interpolated IVs.
    """

    def __init__(
        self,
        S: float,
        r: float,
        strikes: np.ndarray,
        expiries: np.ndarray,     # in years
        iv_surface: np.ndarray,   # shape (n_expiry, n_strike)
    ):
        self.S = S
        self.r = r
        self.strikes = np.array(strikes, dtype=float)
        self.expiries = np.array(expiries, dtype=float)
        self.iv_surface = np.array(iv_surface, dtype=float)
        self._local_vol: Optional[np.ndarray] = None
        self._spline: Optional[RectBivariateSpline] = None
        self._build()

    def _bs_call(self, K_grid: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
        """Vectorized BS call prices over (K, T) grid."""
        S = self.S
        r = self.r
        # Interpolate IV from surface
        spline_iv = RectBivariateSpline(
            self.expiries, self.strikes, self.iv_surface, kx=1, ky=1
        )
        C = np.zeros_like(K_grid)
        for i in range(K_grid.shape[0]):
            for j in range(K_grid.shape[1]):
                K = K_grid[i, j]
                T = T_grid[i, j]
                iv_val = float(spline_iv(T, K)[0, 0])
                iv_val = max(iv_val, 1e-4)
                C[i, j] = BlackScholes.price(S, K, T, r, iv_val, "call")
        return C

    def _build(self) -> None:
        """Compute local vol surface using Dupire finite differences."""
        K = self.strikes
        T = self.expiries
        nT, nK = len(T), len(K)

        if nT < 3 or nK < 3:
            warnings.warn("Insufficient surface points for local vol computation.")
            return

        T_grid, K_grid = np.meshgrid(T, K, indexing="ij")  # (nT, nK)
        C = self._bs_call(K_grid, T_grid)

        # dC/dT via finite differences (interior only)
        dC_dT = np.gradient(C, T, axis=0)

        # dC/dK and d²C/dK²
        dC_dK = np.gradient(C, K, axis=1)
        d2C_dK2 = np.gradient(dC_dK, K, axis=1)

        # Dupire formula: sigma_loc² = dC/dT / (0.5 * K² * d²C/dK²)
        K_sq = K_grid ** 2
        denom = 0.5 * K_sq * d2C_dK2

        with np.errstate(divide="ignore", invalid="ignore"):
            lv2 = np.where(np.abs(denom) > 1e-12, dC_dT / denom, np.nan)

        # Clip to reasonable range
        lv2 = np.clip(lv2, 0.0001, 5.0)
        self._local_vol = np.sqrt(lv2)

        # Build bivariate spline for interpolation
        try:
            lv_clean = np.nan_to_num(self._local_vol, nan=float(np.nanmedian(self._local_vol)))
            self._spline = RectBivariateSpline(T, K, lv_clean, kx=1, ky=1)
        except Exception as e:
            warnings.warn(f"LocalVol spline failed: {e}")

    def sigma(self, K: float, T: float) -> Optional[float]:
        """Interpolated local vol at (K, T)."""
        if self._spline is None:
            return None
        try:
            val = float(self._spline(T, K)[0, 0])
            return max(val, 1e-4)
        except Exception:
            return None

    def price_european(
        self,
        K: float,
        T: float,
        option_type: str = "call",
        n_paths: int = 20_000,
        n_steps: int = 100,
        seed: int = RNG_SEED,
    ) -> float:
        """
        Price European option under local vol via Monte Carlo.
        Uses Euler-Maruyama with time-step-dependent local vol interpolation.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        S_paths = np.full(n_paths, self.S)
        t_cur = 0.0

        for step in range(n_steps):
            t_cur += dt
            lv = np.array([
                self.sigma(s, t_cur) or 0.25 for s in S_paths
            ])
            Z = rng.standard_normal(n_paths)
            S_paths = S_paths * np.exp(
                (self.r - 0.5 * lv**2) * dt + lv * sqrt_dt * Z
            )

        if option_type == "call":
            payoffs = np.maximum(S_paths - K, 0.0)
        else:
            payoffs = np.maximum(K - S_paths, 0.0)

        return float(math.exp(-self.r * T) * payoffs.mean())

    @property
    def surface(self) -> Optional[np.ndarray]:
        """Raw local vol surface array, shape (n_expiry, n_strike)."""
        return self._local_vol


# ---------------------------------------------------------------------------
# 6. Model comparison
# ---------------------------------------------------------------------------

class ModelComparison:
    """
    Price an option with all models and compute model-risk spread.
    """

    def __init__(
        self,
        binomial_steps: int = 300,
        mc_paths: int = 30_000,
        mc_steps: int = 100,
    ):
        self.bs = BlackScholes()
        self.bt = BinomialTree(steps=binomial_steps)
        self.mc = MonteCarlo(n_paths=mc_paths, n_steps=mc_steps)
        self.sabr = SABR(beta=0.5)

    def compare(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        # Heston parameters (default reasonable values)
        v0: Optional[float] = None,
        kappa: float = 2.0,
        theta: Optional[float] = None,
        xi: float = 0.4,
        rho_heston: float = -0.7,
        # SABR parameters
        sabr_alpha: Optional[float] = None,
        sabr_rho: float = -0.3,
        sabr_nu: float = 0.4,
    ) -> ModelComparisonResult:
        """
        Compute prices from all implemented models.
        Returns structured comparison with model risk estimate.
        """
        v0 = v0 or sigma**2
        theta_h = theta or sigma**2
        sabr_alpha = sabr_alpha or sigma  # alpha ≈ ATM vol for log-normal SABR

        prices: Dict[str, float] = {}

        # Black-Scholes
        prices["BlackScholes"] = BlackScholes.price(S, K, T, r, sigma, option_type)

        # Binomial European
        prices["Binomial_European"] = self.bt.price(S, K, T, r, sigma, option_type, american=False)

        # Binomial American
        prices["Binomial_American"] = self.bt.price(S, K, T, r, sigma, option_type, american=True)

        # Monte Carlo GBM
        prices["MonteCarlo_GBM"] = self.mc.price_gbm(S, K, T, r, sigma, option_type)

        # Monte Carlo Heston
        prices["MonteCarlo_Heston"] = self.mc.price_heston(
            S, K, T, r, v0=v0, kappa=kappa, theta=theta_h,
            xi=xi, rho=rho_heston, option_type=option_type,
        )

        # SABR (convert to price via BS with SABR IV)
        F = S * math.exp(r * T)
        sabr_iv = self.sabr.vol(F, K, T, sabr_alpha, sabr_rho, sabr_nu)
        prices["SABR"] = BlackScholes.price(S, K, T, r, sabr_iv, option_type)

        vals = list(prices.values())
        mean_p = float(np.mean(vals))
        std_p = float(np.std(vals))
        spread = float(max(vals) - min(vals))

        return ModelComparisonResult(
            option_type=option_type,
            S=S, K=K, T=T, r=r,
            sigma_input=sigma,
            prices=prices,
            price_mean=round(mean_p, 6),
            price_std=round(std_p, 6),
            model_risk_spread=round(spread, 6),
        )

    def print_report(self, result: ModelComparisonResult) -> None:
        """Pretty-print comparison."""
        print(f"\n{'='*60}")
        print(f"  MODEL COMPARISON")
        print(f"  {result.option_type.upper()} | S={result.S} K={result.K} "
              f"T={result.T:.4f}y r={result.r:.4f} σ={result.sigma_input:.4f}")
        print(f"{'='*60}")
        print(f"  {'Model':<28} {'Price':>10}")
        print(f"  {'-'*40}")
        for model, price in sorted(result.prices.items(), key=lambda x: x[1]):
            print(f"  {model:<28} {price:>10.4f}")
        print(f"  {'-'*40}")
        print(f"  {'Mean':<28} {result.price_mean:>10.4f}")
        print(f"  {'Std Dev (model risk)':<28} {result.price_std:>10.4f}")
        print(f"  {'Spread (max-min)':<28} {result.model_risk_spread:>10.4f}")
        print(f"  {'Model Risk %':<28} {100*result.price_std/max(result.price_mean,1e-8):>9.2f}%")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Convenience functions (module-level)
# ---------------------------------------------------------------------------

def bs_price(S: float, K: float, T: float, r: float, sigma: float, otype: str = "call") -> float:
    return BlackScholes.price(S, K, T, r, sigma, otype)


def bs_iv(mkt: float, S: float, K: float, T: float, r: float, otype: str = "call") -> Optional[float]:
    return BlackScholes.implied_vol(mkt, S, K, T, r, otype)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, otype: str = "call") -> GreeksResult:
    return BlackScholes.greeks(S, K, T, r, sigma, otype)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Options pricing model comparison")
    parser.add_argument("--S", type=float, default=450.0, help="Spot price")
    parser.add_argument("--K", type=float, default=455.0, help="Strike")
    parser.add_argument("--T", type=float, default=0.25, help="Time to expiry (years)")
    parser.add_argument("--r", type=float, default=RISK_FREE_RATE, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.20, help="Implied/input vol")
    parser.add_argument("--type", choices=["call", "put"], default="call", dest="option_type")
    parser.add_argument("--mc-paths", type=int, default=30_000)
    args = parser.parse_args()

    comp = ModelComparison(mc_paths=args.mc_paths)
    result = comp.compare(
        S=args.S, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
        option_type=args.option_type,
    )
    comp.print_report(result)

    # Also show full BS Greeks
    g = BlackScholes.greeks(args.S, args.K, args.T, args.r, args.sigma, args.option_type)
    print(f"  BLACK-SCHOLES GREEKS")
    print(f"    Delta:  {g.delta:+.6f}")
    print(f"    Gamma:  {g.gamma:+.6f}")
    print(f"    Theta:  {g.theta:+.6f}  $/day")
    print(f"    Vega:   {g.vega:+.6f}  $/vol-pt")
    print(f"    Rho:    {g.rho:+.6f}")
    print(f"    Vanna:  {g.vanna:+.6f}")
    print(f"    Volga:  {g.volga:+.6f}")
    print(f"    Charm:  {g.charm:+.8f}")


if __name__ == "__main__":
    _cli()
