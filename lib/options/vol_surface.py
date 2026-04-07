"""
Volatility surface construction and models for the srfm-lab trading system.

Implements SVI, SABR, VolSurface, VolSmile, and LocalVolSurface.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize, least_squares
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _black_iv(price: float, F: float, K: float, T: float, r: float, option_type: str = "call") -> float:
    """
    Compute Black-76 implied vol from a forward price and option premium.
    Thin wrapper around the BlackScholes implied vol.
    """
    from lib.options.pricing import BlackScholes
    return BlackScholes.implied_vol(price, F, K, T, r=0.0, q=0.0, option_type=option_type)


def _black76_price(F: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Black-76 option price (forward-based)."""
    from lib.options.pricing import BlackScholes
    return BlackScholes(F, K, T, r=0.0, sigma=sigma, q=0.0, option_type=option_type).price()


# ---------------------------------------------------------------------------
# SVI Model
# ---------------------------------------------------------------------------

@dataclass
class SVIParams:
    """Parameters for the SVI total variance parameterisation."""
    a: float = 0.04
    b: float = 0.1
    rho: float = -0.3
    m: float = 0.0
    sigma: float = 0.1

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SVIParams":
        return cls(a=arr[0], b=arr[1], rho=arr[2], m=arr[3], sigma=arr[4])


class SVIModel:
    """
    Jim Gatheral's SVI (Stochastic Volatility Inspired) parameterisation.

    The total implied variance is:
        w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

    where k = log(K/F) is the log-moneyness.

    No-arbitrage conditions checked:
    - Butterfly arbitrage: g(k) >= 0 where g(k) is derived from Breeden-Litzenberger.
    - Calendar spread arbitrage: w(k, T1) <= w(k, T2) for T1 < T2.
    """

    def __init__(self, params: Optional[SVIParams] = None) -> None:
        self.params = params or SVIParams()
        self._fitted = False

    # ------------------------------------------------------------------
    # Core formula
    # ------------------------------------------------------------------

    def total_variance(self, k: float, params: Optional[SVIParams] = None) -> float:
        """
        Compute total implied variance w(k).

        Parameters
        ----------
        k : float
            Log-moneyness log(K/F).
        """
        p = params or self.params
        return p.a + p.b * (p.rho * (k - p.m) + math.sqrt((k - p.m) ** 2 + p.sigma ** 2))

    def implied_vol(self, k: float, T: float, params: Optional[SVIParams] = None) -> float:
        """Implied volatility from total variance: sigma_impl = sqrt(w(k)/T)."""
        w = self.total_variance(k, params)
        if w < 0:
            return float("nan")
        return math.sqrt(max(w, 0.0) / T)

    def total_variance_array(self, ks: np.ndarray, params: Optional[SVIParams] = None) -> np.ndarray:
        """Vectorised total variance over array of log-moneyness values."""
        p = params or self.params
        return p.a + p.b * (p.rho * (ks - p.m) + np.sqrt((ks - p.m) ** 2 + p.sigma ** 2))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        log_moneyness: np.ndarray,
        market_total_variance: np.ndarray,
        weights: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
    ) -> SVIParams:
        """
        Calibrate SVI parameters to market data.

        Parameters
        ----------
        log_moneyness : ndarray
            Array of k = log(K/F) values.
        market_total_variance : ndarray
            Array of observed total implied variances (sigma_impl^2 * T).
        weights : ndarray, optional
            Per-observation weights. Defaults to uniform.
        method : str
            Scipy minimisation method.

        Returns
        -------
        SVIParams
            Fitted parameters.
        """
        if weights is None:
            weights = np.ones(len(log_moneyness))
        weights = weights / weights.sum()

        def objective(x: np.ndarray) -> float:
            a, b, rho, m, sigma = x
            if b < 0 or sigma <= 0 or abs(rho) >= 1.0:
                return 1e10
            if a + b * sigma * math.sqrt(1.0 - rho ** 2) < 0:
                return 1e10
            p = SVIParams(a, b, rho, m, sigma)
            w_model = self.total_variance_array(log_moneyness, p)
            residuals = w_model - market_total_variance
            return float(np.sum(weights * residuals ** 2))

        x0 = self.params.as_array()
        bounds = [(-1.0, 5.0), (1e-6, 5.0), (-0.9999, 0.9999), (-3.0, 3.0), (1e-6, 3.0)]
        result = minimize(objective, x0, method=method, bounds=bounds,
                          options={"maxiter": 5000, "ftol": 1e-12})
        self.params = SVIParams.from_array(result.x)
        self._fitted = True
        return self.params

    # ------------------------------------------------------------------
    # No-arbitrage checks
    # ------------------------------------------------------------------

    def check_butterfly_arbitrage(
        self, ks: np.ndarray, params: Optional[SVIParams] = None
    ) -> Tuple[bool, np.ndarray]:
        """
        Check butterfly (density) no-arbitrage condition.

        Returns (is_arbitrage_free, g_values) where g(k) >= 0 is required.

        The condition g(k) = (1 - k*w'/(2w))^2 - w''/2 + (w'/4)*(1/4 + 1/w) >= 0
        is derived from Gatheral (2004).
        """
        p = params or self.params
        ws = self.total_variance_array(ks, p)

        # Numerical derivatives
        dk = (ks[-1] - ks[0]) / (len(ks) * 100)
        ks_p = ks + dk
        ks_m = ks - dk
        ws_p = self.total_variance_array(ks_p, p)
        ws_m = self.total_variance_array(ks_m, p)
        w_prime = (ws_p - ws_m) / (2 * dk)
        w_double = (ws_p - 2 * ws + ws_m) / dk ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            g = (
                (1.0 - ks * w_prime / (2.0 * np.maximum(ws, 1e-12))) ** 2
                - w_prime ** 2 / 4.0 * (1.0 / np.maximum(ws, 1e-12) + 0.25)
                + w_double / 2.0
            )

        is_free = bool(np.all(g >= -1e-6))
        return is_free, g

    def check_calendar_arbitrage(
        self,
        params_t1: SVIParams,
        params_t2: SVIParams,
        ks: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """
        Check calendar spread no-arbitrage: w(k, T1) <= w(k, T2) for T1 < T2.

        Returns (is_arbitrage_free, diff) where diff = w(T2) - w(T1) >= 0 required.
        """
        w1 = self.total_variance_array(ks, params_t1)
        w2 = self.total_variance_array(ks, params_t2)
        diff = w2 - w1
        return bool(np.all(diff >= -1e-8)), diff

    def natural_parameterization(self) -> Tuple[float, float, float, float, float]:
        """Return (a, b, rho, m, sigma) tuple."""
        p = self.params
        return p.a, p.b, p.rho, p.m, p.sigma


# ---------------------------------------------------------------------------
# SABR Model
# ---------------------------------------------------------------------------

class SABRModel:
    """
    SABR stochastic volatility model with Hagan et al. (2002) approximation.

    The SABR model is:
        dF = alpha * F^beta * dW1
        d(alpha) = nu * alpha * dW2
        dW1 dW2 = rho dt

    Hagan approximation for implied Black vol is used.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.4,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self, F: float, K: float, T: float) -> float:
        """
        Compute Black implied volatility using the Hagan SABR approximation.

        Parameters
        ----------
        F : float
            Forward price.
        K : float
            Strike.
        T : float
            Expiry in years.
        """
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu

        T = max(T, 1e-10)
        eps = 1e-7

        if abs(F - K) < eps * F:
            # ATM approximation
            FK_mid = F
            factor1 = alpha / (FK_mid ** (1.0 - beta))
            factor2 = 1.0 + (
                ((1.0 - beta) ** 2 / 24.0 * alpha ** 2 / FK_mid ** (2.0 - 2.0 * beta))
                + (rho * beta * nu * alpha / (4.0 * FK_mid ** (1.0 - beta)))
                + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
            ) * T
            return factor1 * factor2

        FK_beta = (F * K) ** (0.5 * (1.0 - beta))
        log_FK = math.log(F / K)
        z = nu / alpha * FK_beta * log_FK

        # x(z) = log((sqrt(1 - 2*rho*z + z^2) + z - rho)/(1 - rho))
        sq = math.sqrt(max(1.0 - 2.0 * rho * z + z ** 2, 0.0))
        denom = 1.0 - rho
        if abs(denom) < 1e-10:
            x_z = z
        else:
            arg = (sq + z - rho) / denom
            if arg <= 0:
                arg = 1e-10
            x_z = math.log(arg)

        numer_factor = alpha
        denom_factor = FK_beta * (
            1.0
            + (1.0 - beta) ** 2 / 6.0 * log_FK ** 2
            + (1.0 - beta) ** 4 / 120.0 * log_FK ** 4
        )

        if abs(x_z) < 1e-10:
            zx_ratio = 1.0
        else:
            zx_ratio = z / x_z

        correction = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0 * alpha ** 2 / FK_beta ** 2)
            + (rho * beta * nu * alpha / (4.0 * FK_beta))
            + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        ) * T

        return numer_factor / denom_factor * zx_ratio * correction

    def fit(
        self,
        F: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        beta: float = 0.5,
        weights: Optional[np.ndarray] = None,
    ) -> "SABRModel":
        """
        Calibrate SABR parameters (alpha, rho, nu) given fixed beta.

        Parameters
        ----------
        F : float
            Forward price.
        strikes : ndarray
            Array of strikes.
        market_vols : ndarray
            Array of market implied Black vols.
        T : float
            Expiry in years.
        beta : float
            Fixed beta parameter (not calibrated).
        """
        self.beta = beta
        if weights is None:
            weights = np.ones(len(strikes))
        weights = weights / weights.sum()

        def objective(x: np.ndarray) -> float:
            alpha, rho, nu = x
            if alpha <= 0 or nu <= 0 or abs(rho) >= 1.0:
                return 1e10
            self.alpha = alpha
            self.rho = rho
            self.nu = nu
            model_vols = np.array([self.implied_vol(F, K, T) for K in strikes])
            residuals = model_vols - market_vols
            return float(np.sum(weights * residuals ** 2))

        x0 = np.array([self.alpha, self.rho, self.nu])
        bounds = [(1e-4, 5.0), (-0.9999, 0.9999), (1e-4, 5.0)]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
        self.alpha, self.rho, self.nu = result.x
        return self

    def atm_vol(self, F: float, T: float) -> float:
        """ATM implied volatility."""
        return self.implied_vol(F, F, T)

    def skew(self, F: float, T: float, dK: float = None) -> float:
        """
        Approximate vol skew dVol/dK at ATM.
        """
        if dK is None:
            dK = F * 0.001
        v_up = self.implied_vol(F, F + dK, T)
        v_dn = self.implied_vol(F, F - dK, T)
        return (v_up - v_dn) / (2.0 * dK)

    def curvature(self, F: float, T: float, dK: float = None) -> float:
        """
        Approximate vol curvature d2Vol/dK2 at ATM.
        """
        if dK is None:
            dK = F * 0.001
        v0 = self.implied_vol(F, F, T)
        v_up = self.implied_vol(F, F + dK, T)
        v_dn = self.implied_vol(F, F - dK, T)
        return (v_up - 2.0 * v0 + v_dn) / dK ** 2

    def smile_array(self, F: float, strikes: np.ndarray, T: float) -> np.ndarray:
        """Return array of implied vols for given strikes."""
        return np.array([self.implied_vol(F, K, T) for K in strikes])


# ---------------------------------------------------------------------------
# VolSmile
# ---------------------------------------------------------------------------

class VolSmile:
    """
    Single-expiry implied volatility smile.

    Stores (strike, implied_vol) pairs and provides interpolation,
    delta conversion, and risk-reversal/butterfly decomposition.

    Parameters
    ----------
    strikes : array-like
        Strike prices.
    vols : array-like
        Corresponding implied volatilities.
    expiry : float
        Expiry in years.
    F : float
        Forward price for this expiry.
    r : float
        Risk-free rate.
    """

    def __init__(
        self,
        strikes: np.ndarray,
        vols: np.ndarray,
        expiry: float,
        F: float,
        r: float = 0.0,
    ) -> None:
        strikes = np.asarray(strikes, dtype=float)
        vols = np.asarray(vols, dtype=float)
        idx = np.argsort(strikes)
        self.strikes = strikes[idx]
        self.vols = vols[idx]
        self.expiry = expiry
        self.F = F
        self.r = r
        self._interp = self._build_interpolator()

    def _build_interpolator(self) -> CubicSpline:
        """Build cubic spline interpolator in strike space."""
        if len(self.strikes) < 2:
            raise ValueError("At least 2 strike/vol pairs required for interpolation")
        return CubicSpline(self.strikes, self.vols, extrapolate=True)

    def vol(self, K: float) -> float:
        """Interpolated implied vol at strike K."""
        return float(np.clip(self._interp(K), 1e-6, 10.0))

    def vol_array(self, Ks: np.ndarray) -> np.ndarray:
        """Interpolated implied vols at array of strikes."""
        return np.clip(self._interp(Ks), 1e-6, 10.0)

    def atm_vol(self) -> float:
        """ATM implied vol (interpolated at F)."""
        return self.vol(self.F)

    def delta_to_strike(self, delta: float, option_type: str = "call") -> float:
        """
        Convert a Black-Scholes delta to a strike price.

        Uses Newton's method on the delta equation.
        """
        from scipy.optimize import brentq as _brentq
        from lib.options.pricing import BlackScholes

        def eq(K: float) -> float:
            iv = self.vol(K)
            bs = BlackScholes(self.F, K, self.expiry, self.r, iv, 0.0, option_type)
            return bs.delta() - delta

        # Bracket search
        K_lo, K_hi = self.strikes[0], self.strikes[-1]
        try:
            return float(_brentq(eq, K_lo, K_hi))
        except ValueError:
            return float("nan")

    def risk_reversal(self, delta: float = 0.25) -> float:
        """
        25-delta (or specified-delta) risk reversal: vol(+delta call) - vol(-delta put).
        """
        K_call = self.delta_to_strike(delta, "call")
        K_put = self.delta_to_strike(-delta, "put")
        if math.isnan(K_call) or math.isnan(K_put):
            return float("nan")
        return self.vol(K_call) - self.vol(K_put)

    def butterfly(self, delta: float = 0.25) -> float:
        """
        25-delta (or specified-delta) butterfly: 0.5*(vol(call) + vol(put)) - vol(ATM).
        """
        K_call = self.delta_to_strike(delta, "call")
        K_put = self.delta_to_strike(-delta, "put")
        if math.isnan(K_call) or math.isnan(K_put):
            return float("nan")
        return 0.5 * (self.vol(K_call) + self.vol(K_put)) - self.atm_vol()

    def smile_summary(self) -> dict:
        """Return key smile metrics."""
        return {
            "atm_vol": self.atm_vol(),
            "rr_25d": self.risk_reversal(0.25),
            "bf_25d": self.butterfly(0.25),
            "min_vol": float(np.min(self.vols)),
            "max_vol": float(np.max(self.vols)),
        }


# ---------------------------------------------------------------------------
# VolSurface
# ---------------------------------------------------------------------------

class VolSurface:
    """
    Implied volatility surface: matrix of vols keyed by (expiry, strike).

    Supports:
    - Cubic spline interpolation in strike at each expiry.
    - Linear interpolation across expiries.
    - Forward vol extraction.
    - Implied vol query at arbitrary (T, K).

    Parameters
    ----------
    expiries : array-like
        Expiry times in years, sorted ascending.
    strikes : list of array-like
        Strikes for each expiry slice. Can be ragged.
    vols : list of array-like
        Implied vols corresponding to each (expiry, strikes) slice.
    forwards : array-like
        Forward prices for each expiry.
    r : float
        Risk-free rate (flat, for simplicity).
    """

    def __init__(
        self,
        expiries: np.ndarray,
        strikes: List[np.ndarray],
        vols: List[np.ndarray],
        forwards: np.ndarray,
        r: float = 0.0,
    ) -> None:
        self.expiries = np.asarray(expiries, dtype=float)
        self.strikes = [np.asarray(k, dtype=float) for k in strikes]
        self.vols = [np.asarray(v, dtype=float) for v in vols]
        self.forwards = np.asarray(forwards, dtype=float)
        self.r = r
        self._smiles = self._build_smiles()

    def _build_smiles(self) -> List[VolSmile]:
        smiles = []
        for i, T in enumerate(self.expiries):
            smiles.append(
                VolSmile(self.strikes[i], self.vols[i], T, float(self.forwards[i]), self.r)
            )
        return smiles

    def smile(self, expiry_idx: int) -> VolSmile:
        """Return VolSmile for a given expiry index."""
        return self._smiles[expiry_idx]

    def implied_vol(self, T: float, K: float) -> float:
        """
        Interpolated implied vol at (T, K).

        Uses flat extrapolation outside the expiry range.
        Cubic spline in strike at each expiry, linear interpolation in time.
        """
        if T <= self.expiries[0]:
            return self._smiles[0].vol(K)
        if T >= self.expiries[-1]:
            return self._smiles[-1].vol(K)

        # Find bracketing expiries
        idx = np.searchsorted(self.expiries, T)
        T_lo = self.expiries[idx - 1]
        T_hi = self.expiries[idx]
        v_lo = self._smiles[idx - 1].vol(K)
        v_hi = self._smiles[idx].vol(K)

        # Linear interpolation in total variance for calendar consistency
        w_lo = v_lo ** 2 * T_lo
        w_hi = v_hi ** 2 * T_hi
        alpha = (T - T_lo) / (T_hi - T_lo)
        w_interp = w_lo + alpha * (w_hi - w_lo)
        return math.sqrt(max(w_interp / T, 0.0))

    def forward_vol(self, T1: float, T2: float, K: float) -> float:
        """
        Extract forward volatility between T1 and T2 at strike K.

        Uses: sigma_fwd^2 = (w(T2) - w(T1)) / (T2 - T1)
        """
        if T2 <= T1:
            raise ValueError("T2 must be greater than T1")
        v1 = self.implied_vol(T1, K)
        v2 = self.implied_vol(T2, K)
        w1 = v1 ** 2 * T1
        w2 = v2 ** 2 * T2
        dw = w2 - w1
        dt = T2 - T1
        if dw < 0:
            warnings.warn("Negative forward variance detected at K={:.4f}, T1={:.4f}, T2={:.4f}".format(K, T1, T2))
            return 0.0
        return math.sqrt(dw / dt)

    def term_structure_atm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (expiries, ATM implied vols) for term structure plot."""
        atm_vols = np.array([smile.atm_vol() for smile in self._smiles])
        return self.expiries.copy(), atm_vols

    def surface_matrix(self, T_grid: np.ndarray, K_grid: np.ndarray) -> np.ndarray:
        """
        Return (len(T_grid), len(K_grid)) matrix of implied vols.
        """
        result = np.zeros((len(T_grid), len(K_grid)))
        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                result[i, j] = self.implied_vol(T, K)
        return result

    def add_smile(self, expiry: float, strikes: np.ndarray, vols: np.ndarray, forward: float) -> None:
        """Insert or update a smile slice at a given expiry."""
        idx = np.searchsorted(self.expiries, expiry)
        if idx < len(self.expiries) and self.expiries[idx] == expiry:
            # Replace
            self.strikes[idx] = np.asarray(strikes)
            self.vols[idx] = np.asarray(vols)
            self.forwards[idx] = forward
            self._smiles[idx] = VolSmile(strikes, vols, expiry, forward, self.r)
        else:
            self.expiries = np.insert(self.expiries, idx, expiry)
            self.strikes.insert(idx, np.asarray(strikes))
            self.vols.insert(idx, np.asarray(vols))
            self.forwards = np.insert(self.forwards, idx, forward)
            new_smile = VolSmile(strikes, vols, expiry, forward, self.r)
            self._smiles.insert(idx, new_smile)


# ---------------------------------------------------------------------------
# Local Volatility Surface (Dupire)
# ---------------------------------------------------------------------------

class LocalVolSurface:
    """
    Dupire local volatility surface from implied vol surface.

    Dupire (1994) formula:
        sigma_local^2(K, T) = (dC/dT) / (0.5 * K^2 * d2C/dK2)

    In terms of implied vols (Gatheral form):

        sigma_local^2 = w_T / (1 - k*w_k/w + 0.25*(-0.25 - 1/w + k^2/w^2)*w_k^2 + 0.5*w_kk)

    where w = sigma_impl^2 * T, k = log(K/F), w_T = dw/dT, w_k = dw/dk, w_kk = d2w/dk2.
    """

    def __init__(self, vol_surface: VolSurface) -> None:
        self.vol_surface = vol_surface
        self._dT = 1.0 / 365.0
        self._dK_frac = 0.001

    def local_vol(self, S: float, t: float) -> float:
        """
        Compute Dupire local volatility at spot S and time t.

        Uses numerical derivatives of the implied vol surface.
        """
        T = max(t, 2.0 * self._dT)

        # Find forward for this expiry via linear interpolation
        expiries = self.vol_surface.expiries
        forwards = self.vol_surface.forwards
        F = float(np.interp(T, expiries, forwards))

        K = S
        dK = max(K * self._dK_frac, 0.01)
        dT = self._dT

        # Implied vol and total variance at grid points
        def w(T_: float, K_: float) -> float:
            iv = self.vol_surface.implied_vol(T_, K_)
            return iv ** 2 * T_

        w0 = w(T, K)
        k = math.log(K / F) if F > 0 else 0.0

        # Time derivative
        w_T = (w(T + dT, K) - w(T - dT, K)) / (2.0 * dT)

        # Strike derivatives in log-moneyness space
        w_kp = w(T, K * math.exp(dK / K))
        w_km = w(T, K * math.exp(-dK / K))
        w_k = (w_kp - w_km) / (2.0 * dK / K)
        w_kk = (w_kp - 2.0 * w0 + w_km) / (dK / K) ** 2

        denom = 1.0 - k * w_k / max(w0, 1e-10) + 0.25 * (-0.25 - 1.0 / max(w0, 1e-10) + k ** 2 / max(w0, 1e-10) ** 2) * w_k ** 2 + 0.5 * w_kk

        if abs(denom) < 1e-10 or w_T < 0:
            # Fall back to implied vol
            return self.vol_surface.implied_vol(T, K)

        local_var = w_T / denom
        if local_var <= 0:
            return self.vol_surface.implied_vol(T, K)

        return math.sqrt(local_var)

    def local_vol_surface(
        self, S_grid: np.ndarray, T_grid: np.ndarray
    ) -> np.ndarray:
        """
        Compute (len(T_grid), len(S_grid)) local vol matrix.
        """
        result = np.zeros((len(T_grid), len(S_grid)))
        for i, T in enumerate(T_grid):
            for j, S in enumerate(S_grid):
                result[i, j] = self.local_vol(S, T)
        return result

    def to_vol_surface(
        self,
        S: float,
        T_grid: np.ndarray,
        K_grid: np.ndarray,
    ) -> "VolSurface":
        """
        Build a VolSurface from local vol by Monte Carlo forward PDE.
        This is a stub that returns the implied vol surface as an approximation.
        Full Gyongy projection would require a PDE solver.
        """
        return self.vol_surface


# ---------------------------------------------------------------------------
# Convenience builder functions
# ---------------------------------------------------------------------------

def build_vol_surface_from_svi(
    expiries: np.ndarray,
    svi_params_list: List[SVIParams],
    forwards: np.ndarray,
    r: float,
    n_strikes: int = 101,
    moneyness_range: Tuple[float, float] = (-1.5, 1.5),
) -> VolSurface:
    """
    Build a VolSurface from a list of SVI parameter sets (one per expiry).

    Parameters
    ----------
    expiries : ndarray
        Expiry times in years.
    svi_params_list : list of SVIParams
        One SVI parameter set per expiry.
    forwards : ndarray
        Forward prices for each expiry.
    r : float
        Risk-free rate.
    n_strikes : int
        Number of strikes to generate per expiry slice.
    moneyness_range : tuple
        (min_k, max_k) log-moneyness range.
    """
    model = SVIModel()
    strikes_list = []
    vols_list = []
    for i, (T, params, F) in enumerate(zip(expiries, svi_params_list, forwards)):
        ks = np.linspace(moneyness_range[0], moneyness_range[1], n_strikes)
        Ks = F * np.exp(ks)
        iv_arr = np.array([model.implied_vol(k, T, params) for k in ks])
        strikes_list.append(Ks)
        vols_list.append(iv_arr)
    return VolSurface(expiries, strikes_list, vols_list, forwards, r)
