"""
Implied volatility surface construction.

Implements:
- Implied vol via Newton-Raphson and Brent's method
- SVI (Stochastic Volatility Inspired) parametrization (Gatheral 2004)
- Bicubic interpolation of the vol surface
- Dupire local volatility from implied vol surface
- Surface arbitrage checks (calendar spread, butterfly)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, stats


# ---------------------------------------------------------------------------
# Implied volatility inversion
# ---------------------------------------------------------------------------

def _bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "call", q: float = 0.0
) -> float:
    """Black-Scholes price (standalone function for speed)."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return (S * np.exp(-q * T) * stats.norm.cdf(d1)
                - K * np.exp(-r * T) * stats.norm.cdf(d2))
    return (K * np.exp(-r * T) * stats.norm.cdf(-d2)
            - S * np.exp(-q * T) * stats.norm.cdf(-d1))


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes vega."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-12)
    return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    method: str = "brentq",
    max_iter: int = 200,
    tol: float = 1e-8,
    sigma_lo: float = 1e-5,
    sigma_hi: float = 10.0,
) -> float:
    """
    Compute implied volatility by inverting the Black-Scholes formula.

    Parameters
    ----------
    market_price : float
        Observed market option price.
    S, K, T, r, q : floats
        Option parameters.
    option_type : 'call' or 'put'
    method : 'brentq' or 'newton'
    max_iter, tol : convergence parameters
    sigma_lo, sigma_hi : bounds for Brent search

    Returns
    -------
    float
        Implied volatility, or np.nan if not found.
    """
    # Intrinsic value check
    if option_type == "call":
        intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price < intrinsic - 1e-6:
        return np.nan
    if market_price < 1e-10:
        return np.nan

    def objective(sigma):
        return _bs_price(S, K, T, r, sigma, option_type, q) - market_price

    if method == "brentq":
        try:
            fa = objective(sigma_lo)
            fb = objective(sigma_hi)
            if fa * fb > 0:
                # Try to expand bounds
                for hi in [20.0, 50.0, 100.0]:
                    if objective(sigma_lo) * objective(hi) < 0:
                        sigma_hi = hi
                        break
                else:
                    return np.nan

            iv = optimize.brentq(
                objective, sigma_lo, sigma_hi, xtol=tol, maxiter=max_iter
            )
            return float(iv)
        except (ValueError, RuntimeError):
            return np.nan

    elif method == "newton":
        # Newton-Raphson with Vega as derivative
        sigma = 0.3  # initial guess
        for _ in range(max_iter):
            price = _bs_price(S, K, T, r, sigma, option_type, q)
            vega = _bs_vega(S, K, T, r, sigma, q)
            if abs(vega) < 1e-10:
                break
            sigma_new = sigma - (price - market_price) / vega
            sigma_new = np.clip(sigma_new, sigma_lo, sigma_hi)
            if abs(sigma_new - sigma) < tol:
                return float(sigma_new)
            sigma = sigma_new
        return float(sigma)

    raise ValueError(f"Unknown method: {method}")


def implied_vol_surface(
    price_matrix: pd.DataFrame,
    strikes: List[float],
    maturities: List[float],
    S: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    method: str = "brentq",
) -> pd.DataFrame:
    """
    Compute implied vol surface from a matrix of market option prices.

    Parameters
    ----------
    price_matrix : pd.DataFrame
        (len(maturities) x len(strikes)) market option prices.
    strikes : list of floats
    maturities : list of floats
    S, r, q, option_type, method : see implied_vol()

    Returns
    -------
    pd.DataFrame
        (len(maturities) x len(strikes)) implied vols.
    """
    iv_matrix = np.full((len(maturities), len(strikes)), np.nan)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            mp = float(price_matrix.iloc[i, j]) if isinstance(price_matrix, pd.DataFrame) \
                else float(price_matrix[i, j])
            iv_matrix[i, j] = implied_vol(mp, S, K, T, r, option_type, q, method)

    return pd.DataFrame(
        iv_matrix.round(6),
        index=[f"T={T:.3f}" for T in maturities],
        columns=[f"K={K}" for K in strikes],
    )


# ---------------------------------------------------------------------------
# SVI Parametrization
# ---------------------------------------------------------------------------

class SVIModel:
    """
    Stochastic Volatility Inspired (SVI) total implied variance parametrization.

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    where k = log(K/F), w = sigma_implied^2 * T (total variance).

    Parameters
    ----------
    a, b, rho, m, sigma : float
        SVI raw parameters.
        Constraints: b >= 0, |rho| < 1, sigma > 0,
                     a + b*sigma*sqrt(1-rho^2) >= 0 (no negative variance).
    """

    def __init__(
        self,
        a: float = 0.04,
        b: float = 0.1,
        rho: float = -0.5,
        m: float = 0.0,
        sigma: float = 0.2,
    ) -> None:
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def total_variance(self, k: "np.ndarray") -> "np.ndarray":
        """
        Compute SVI total variance w(k) for log-moneyness k = log(K/F).

        Returns
        -------
        np.ndarray
            Total implied variance.
        """
        k = np.asarray(k)
        inner = np.sqrt((k - self.m) ** 2 + self.sigma ** 2)
        return self.a + self.b * (self.rho * (k - self.m) + inner)

    def implied_vol(self, k: "np.ndarray", T: float) -> "np.ndarray":
        """Convert total variance to implied vol."""
        w = self.total_variance(k)
        return np.sqrt(np.maximum(w / T, 0.0))

    def is_arbitrage_free(self) -> bool:
        """
        Check butterfly arbitrage condition (Rogers-Tehranchi sufficient condition):
        w(k) > 0 and g(k) >= 0 where g(k) = (1 - k*w'/(2*w))^2 - w'^2/4*(1/w + 1/4) + w''/2
        """
        k_grid = np.linspace(-3, 3, 200)
        w = self.total_variance(k_grid)
        if np.any(w <= 0):
            return False
        # Numerical derivative
        dk = k_grid[1] - k_grid[0]
        dw = np.gradient(w, dk)
        d2w = np.gradient(dw, dk)
        g = (1 - k_grid * dw / (2 * w + 1e-10)) ** 2 - dw ** 2 / 4 * (1 / (w + 1e-10) + 0.25) + d2w / 2
        return bool(np.all(g >= -1e-6))

    def fit(
        self,
        k_data: "np.ndarray",
        iv_data: "np.ndarray",
        T: float,
        initial_params: Optional[List[float]] = None,
        weights: Optional["np.ndarray"] = None,
    ) -> Dict[str, float]:
        """
        Fit SVI parameters to market data.

        Parameters
        ----------
        k_data : array of log-moneyness values
        iv_data : array of implied vols
        T : float, expiration
        initial_params : [a, b, rho, m, sigma]

        Returns
        -------
        dict of fitted parameters and RMSE.
        """
        w_data = iv_data ** 2 * T

        if initial_params is None:
            initial_params = [w_data.mean(), 0.1, -0.3, k_data.mean(), 0.2]

        w_arr = weights if weights is not None else np.ones_like(k_data)

        def objective(params):
            a, b, rho, m, sigma = params
            if b < 0 or abs(rho) >= 1 or sigma <= 0:
                return 1e10
            self.a, self.b, self.rho, self.m, self.sigma = a, b, rho, m, sigma
            w_model = self.total_variance(k_data)
            if np.any(w_model <= 0):
                return 1e10
            return float(np.sum(w_arr * (w_model - w_data) ** 2))

        bounds = [(-1, 2), (0, 2), (-0.99, 0.99), (-2, 2), (0.001, 2)]
        result = optimize.minimize(
            objective, initial_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 2000},
        )
        a, b, rho, m, sigma = result.x
        self.a, self.b, self.rho, self.m, self.sigma = a, b, rho, m, sigma
        w_model = self.total_variance(k_data)
        rmse = float(np.sqrt(np.mean((np.sqrt(np.maximum(w_model / T, 0)) - iv_data) ** 2)))

        return {
            "a": round(a, 6), "b": round(b, 6), "rho": round(rho, 6),
            "m": round(m, 6), "sigma": round(sigma, 6),
            "rmse": round(rmse, 8),
            "arbitrage_free": self.is_arbitrage_free(),
            "success": result.success,
        }


# ---------------------------------------------------------------------------
# Bicubic interpolation
# ---------------------------------------------------------------------------

class VolSurfaceInterpolator:
    """
    Bicubic (or bilinear) interpolation of an implied vol surface.

    The surface is defined on a (strikes x maturities) grid and can be
    queried at arbitrary (K, T) points.

    Parameters
    ----------
    method : str
        'cubic' (bicubic) or 'linear' (bilinear).
    extrapolation : str
        'flat' — clip to boundary values.
        'linear' — linear extrapolation.
    """

    def __init__(
        self,
        method: str = "cubic",
        extrapolation: str = "flat",
    ) -> None:
        self.method = method
        self.extrapolation = extrapolation
        self._interp = None
        self._strikes = None
        self._maturities = None

    def fit(
        self,
        iv_surface: pd.DataFrame,
        strikes: List[float],
        maturities: List[float],
    ) -> None:
        """
        Fit the interpolator to a given implied vol surface.

        Parameters
        ----------
        iv_surface : (len(maturities) x len(strikes)) DataFrame of implied vols.
        strikes : list of strike prices.
        maturities : list of maturities in years.
        """
        self._strikes = np.array(strikes)
        self._maturities = np.array(maturities)
        z = iv_surface.values.astype(float)

        # Fill NaN with nearest-neighbour for stability
        from scipy.ndimage import generic_filter
        nan_mask = np.isnan(z)
        if nan_mask.any():
            z_filled = z.copy()
            z_filled[nan_mask] = np.nanmean(z)
            z = z_filled

        if self.method == "cubic":
            self._interp = interpolate.RectBivariateSpline(
                self._maturities, self._strikes, z, kx=3, ky=3
            )
        else:
            self._interp = interpolate.RegularGridInterpolator(
                (self._maturities, self._strikes), z,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

    def predict(self, K: float, T: float) -> float:
        """
        Interpolate implied vol at (K, T).

        Parameters
        ----------
        K : float
            Strike price.
        T : float
            Maturity in years.

        Returns
        -------
        float
            Interpolated implied vol.
        """
        if self._interp is None:
            raise ValueError("Call fit() first.")

        if self.extrapolation == "flat":
            K_clamped = np.clip(K, self._strikes[0], self._strikes[-1])
            T_clamped = np.clip(T, self._maturities[0], self._maturities[-1])
        else:
            K_clamped = K
            T_clamped = T

        if self.method == "cubic":
            val = float(self._interp(T_clamped, K_clamped))
        else:
            val = float(self._interp([[T_clamped, K_clamped]]))

        return max(val, 0.0)

    def smile(self, T: float, n_points: int = 50) -> pd.Series:
        """Return vol smile at maturity T."""
        Ks = np.linspace(self._strikes[0], self._strikes[-1], n_points)
        vols = [self.predict(K, T) for K in Ks]
        return pd.Series(vols, index=Ks, name=f"T={T:.2f}")

    def term_structure(self, K: float, n_points: int = 20) -> pd.Series:
        """Return vol term structure at strike K."""
        Ts = np.linspace(self._maturities[0], self._maturities[-1], n_points)
        vols = [self.predict(K, T) for T in Ts]
        return pd.Series(vols, index=Ts, name=f"K={K}")


# ---------------------------------------------------------------------------
# Dupire Local Volatility
# ---------------------------------------------------------------------------

class DupireLocalVol:
    """
    Dupire (1994) local volatility surface derived from implied vols.

    sigma_local^2(K, T) = (dw/dT) / ((1 - k*dw/dk/(2w))^2 - (dw/dk)^2/4*(1/w+1/4) + d^2w/dk^2/2)

    where w = sigma_implied^2 * T, k = log(K/F).

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

    def compute_local_vol(
        self,
        iv_surface: pd.DataFrame,
        strikes: List[float],
        maturities: List[float],
    ) -> pd.DataFrame:
        """
        Compute Dupire local vol on the given grid.

        Parameters
        ----------
        iv_surface : (maturities x strikes) implied vol DataFrame
        strikes : list of float
        maturities : list of float (in years)

        Returns
        -------
        pd.DataFrame
            (maturities x strikes) local vol surface.
        """
        K_arr = np.array(strikes)
        T_arr = np.array(maturities)
        iv_arr = iv_surface.values.astype(float)

        # Total implied variance w = sigma^2 * T
        w_arr = iv_arr ** 2 * T_arr[:, None]

        # Log-moneyness k = log(K/F) where F = S*e^{(r-q)*T}
        F_arr = self.S * np.exp((self.r - self.q) * T_arr[:, None]) * np.ones((1, len(K_arr)))
        k_arr = np.log(K_arr[None, :] / F_arr)

        # Numerical partial derivatives
        nT, nK = w_arr.shape
        local_var = np.full((nT, nK), np.nan)

        for i in range(1, nT - 1):
            for j in range(1, nK - 1):
                dT = T_arr[i + 1] - T_arr[i - 1]
                dk = k_arr[i, j + 1] - k_arr[i, j - 1]

                dw_dT = (w_arr[i + 1, j] - w_arr[i - 1, j]) / (dT + 1e-12)
                dw_dk = (w_arr[i, j + 1] - w_arr[i, j - 1]) / (dk + 1e-12)
                d2w_dk2 = (w_arr[i, j + 1] - 2 * w_arr[i, j] + w_arr[i, j - 1]) / (dk / 2 + 1e-12) ** 2

                w = w_arr[i, j]
                k = k_arr[i, j]

                if w <= 0:
                    continue

                # Denominator: Gatheral's formula
                term1 = (1 - k * dw_dk / (2 * w + 1e-10)) ** 2
                term2 = dw_dk ** 2 / 4 * (1 / (w + 1e-10) + 0.25)
                term3 = d2w_dk2 / 2

                denominator = term1 - term2 + term3
                if abs(denominator) < 1e-10 or dw_dT < 0:
                    continue

                local_var[i, j] = dw_dT / denominator

        local_vol = np.sqrt(np.maximum(local_var, 0.0))

        return pd.DataFrame(
            local_vol.round(6),
            index=[f"T={T:.3f}" for T in maturities],
            columns=[f"K={K}" for K in strikes],
        )


# ---------------------------------------------------------------------------
# Arbitrage checks
# ---------------------------------------------------------------------------

class ArbitrageChecker:
    """
    Check no-arbitrage conditions on an implied vol surface.

    Checks:
    1. Butterfly (convexity) arbitrage
    2. Calendar spread arbitrage
    3. Put-call parity
    """

    @staticmethod
    def butterfly_check(
        iv_surface: pd.DataFrame,
        strikes: List[float],
        maturities: List[float],
        S: float,
        r: float,
        T_index: int = 0,
        q: float = 0.0,
    ) -> pd.Series:
        """
        Check butterfly arbitrage at a given maturity slice.

        No-butterfly: call prices are convex in K.

        Returns
        -------
        pd.Series
            Boolean series: True = no arbitrage at that strike triplet.
        """
        T = maturities[T_index]
        iv_row = iv_surface.iloc[T_index].values
        call_prices = np.array([
            _bs_price(S, K, T, r, iv, "call", q)
            for K, iv in zip(strikes, iv_row)
        ])

        results = {}
        for j in range(1, len(strikes) - 1):
            # Butterfly value should be non-negative
            bfly = call_prices[j - 1] - 2 * call_prices[j] + call_prices[j + 1]
            results[f"K={strikes[j]}"] = bool(bfly >= -1e-6)

        return pd.Series(results)

    @staticmethod
    def calendar_check(
        iv_surface: pd.DataFrame,
        strikes: List[float],
        maturities: List[float],
        K_index: int = 0,
    ) -> pd.Series:
        """
        Check calendar spread arbitrage at a given strike slice.

        No-calendar arb: total variance w(T) = sigma^2*T is increasing in T.

        Returns
        -------
        pd.Series
            Boolean series: True = no arbitrage between consecutive maturities.
        """
        iv_col = iv_surface.iloc[:, K_index].values
        K = strikes[K_index]
        w = iv_col ** 2 * np.array(maturities)

        results = {}
        for i in range(1, len(maturities)):
            no_arb = bool(w[i] >= w[i - 1] - 1e-6)
            results[f"T={maturities[i - 1]:.2f}->{maturities[i]:.2f}"] = no_arb

        return pd.Series(results)

    @staticmethod
    def full_surface_check(
        iv_surface: pd.DataFrame,
        strikes: List[float],
        maturities: List[float],
        S: float,
        r: float,
        q: float = 0.0,
    ) -> Dict[str, bool]:
        """
        Run all arbitrage checks across the full surface.

        Returns
        -------
        dict
            'butterfly_free', 'calendar_free', 'all_arbitrage_free'.
        """
        butterfly_ok = True
        for T_idx in range(len(maturities)):
            bfly = ArbitrageChecker.butterfly_check(
                iv_surface, strikes, maturities, S, r, T_idx, q
            )
            if not bfly.all():
                butterfly_ok = False
                break

        calendar_ok = True
        for K_idx in range(len(strikes)):
            cal = ArbitrageChecker.calendar_check(iv_surface, strikes, maturities, K_idx)
            if not cal.all():
                calendar_ok = False
                break

        return {
            "butterfly_free": butterfly_ok,
            "calendar_free": calendar_ok,
            "all_arbitrage_free": butterfly_ok and calendar_ok,
        }
