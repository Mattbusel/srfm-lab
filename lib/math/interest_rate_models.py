"""
lib/math/interest_rate_models.py

Interest rate models and yield curve analytics for fixed income research.

Implements:
  - Vasicek model: analytical bond prices, simulation
  - CIR (Cox-Ingersoll-Ross): analytical + simulation
  - Hull-White extension: time-varying theta for exact curve fitting
  - Nelson-Siegel yield curve fitting (level, slope, curvature)
  - Svensson model (extended Nelson-Siegel with second curvature term)
  - Forward rate extraction from fitted curves
  - Par yield, zero coupon, forward bootstrapping
  - Duration, convexity, DV01 computation
  - Yield curve principal components (level/slope/curvature via PCA)
  - Spread analysis: OAS, Z-spread, I-spread
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats
from scipy.interpolate import CubicSpline


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class VasicekParams:
    kappa: float   # mean-reversion speed
    theta: float   # long-run mean
    sigma: float   # volatility
    r0: float      # initial short rate


@dataclass
class CIRParams:
    kappa: float
    theta: float
    sigma: float
    r0: float


@dataclass
class HullWhiteParams:
    kappa: float
    sigma: float
    theta_t: Callable[[float], float]  # time-varying theta function
    r0: float


@dataclass
class NelsonSiegelParams:
    beta0: float   # level
    beta1: float   # slope
    beta2: float   # curvature
    tau: float     # decay factor


@dataclass
class SvenssonParams:
    beta0: float
    beta1: float
    beta2: float
    beta3: float   # second curvature
    tau1: float
    tau2: float


@dataclass
class BondMetrics:
    price: float
    duration: float          # Macaulay duration
    modified_duration: float
    convexity: float
    dv01: float              # dollar value of 1bp


@dataclass
class SpreadMetrics:
    z_spread: float          # zero-volatility spread (bps)
    i_spread: float          # interpolated spread vs swap curve (bps)
    oas: float               # option-adjusted spread (bps, simplified)


@dataclass
class CurvePCA:
    explained_variance_ratio: np.ndarray
    components: np.ndarray   # shape (n_components, n_tenors)
    scores: np.ndarray       # shape (n_obs, n_components)
    level: np.ndarray        # PC1 loadings
    slope: np.ndarray        # PC2 loadings
    curvature: np.ndarray    # PC3 loadings


# ── Vasicek Model ─────────────────────────────────────────────────────────────

def vasicek_bond_price(params: VasicekParams, T: float) -> float:
    """
    Analytical zero-coupon bond price P(0,T) under Vasicek.
    P(0,T) = exp(A(T) - B(T)*r0)
    """
    k, th, s, r0 = params.kappa, params.theta, params.sigma, params.r0
    if T <= 0.0:
        return 1.0
    B = (1.0 - math.exp(-k * T)) / k
    A = (th - s**2 / (2.0 * k**2)) * (B - T) - (s**2 * B**2) / (4.0 * k)
    return math.exp(A - B * r0)


def vasicek_yield(params: VasicekParams, T: float) -> float:
    """Continuously compounded yield y(0,T) = -ln(P(0,T))/T."""
    if T <= 0.0:
        return params.r0
    p = vasicek_bond_price(params, T)
    return -math.log(p) / T


def vasicek_simulate(
    params: VasicekParams,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate Vasicek short rate paths using Euler-Maruyama.
    Returns array of shape (n_paths, n_steps+1).
    """
    if rng is None:
        rng = np.random.default_rng()
    k, th, s, r0 = params.kappa, params.theta, params.sigma, params.r0
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    dW = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    for i in range(n_steps):
        r = paths[:, i]
        paths[:, i + 1] = r + k * (th - r) * dt + s * dW[:, i]
    return paths


def vasicek_analytical_mean_var(params: VasicekParams, t: float) -> Tuple[float, float]:
    """E[r(t)] and Var[r(t)] under Vasicek."""
    k, th, s, r0 = params.kappa, params.theta, params.sigma, params.r0
    ekt = math.exp(-k * t)
    mean = th + (r0 - th) * ekt
    var = s**2 / (2.0 * k) * (1.0 - ekt**2)
    return mean, var


# ── CIR Model ─────────────────────────────────────────────────────────────────

def _cir_AB(params: CIRParams, T: float) -> Tuple[float, float]:
    """Helper: compute A(T) and B(T) for CIR bond pricing."""
    k, th, s = params.kappa, params.theta, params.sigma
    gamma = math.sqrt(k**2 + 2.0 * s**2)
    denom = (gamma + k) * (math.exp(gamma * T) - 1.0) + 2.0 * gamma
    B = 2.0 * (math.exp(gamma * T) - 1.0) / denom
    A_exp = (
        2.0 * gamma * math.exp((k + gamma) * T / 2.0) / denom
    ) ** (2.0 * k * th / s**2)
    return math.log(A_exp), B


def cir_bond_price(params: CIRParams, T: float) -> float:
    """Analytical zero-coupon bond price under CIR."""
    if T <= 0.0:
        return 1.0
    log_A, B = _cir_AB(params, T)
    return math.exp(log_A - B * params.r0)


def cir_yield(params: CIRParams, T: float) -> float:
    """Continuously compounded yield under CIR."""
    if T <= 0.0:
        return params.r0
    return -math.log(cir_bond_price(params, T)) / T


def cir_simulate(
    params: CIRParams,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate CIR paths using full truncation Euler scheme (ensures r >= 0).
    Returns (n_paths, n_steps+1).
    """
    if rng is None:
        rng = np.random.default_rng()
    k, th, s, r0 = params.kappa, params.theta, params.sigma, params.r0
    dt = T / n_steps
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    dW = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    for i in range(n_steps):
        r = paths[:, i]
        r_pos = np.maximum(r, 0.0)
        paths[:, i + 1] = r_pos + k * (th - r_pos) * dt + s * np.sqrt(r_pos) * dW[:, i]
    return np.maximum(paths, 0.0)


def cir_feller_condition(params: CIRParams) -> bool:
    """Check Feller condition 2*kappa*theta > sigma^2 (ensures r stays positive)."""
    return 2.0 * params.kappa * params.theta > params.sigma**2


# ── Hull-White Extended Vasicek ───────────────────────────────────────────────

def hull_white_simulate(
    params: HullWhiteParams,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate Hull-White extended Vasicek paths.
    dr = [theta(t) - kappa*r] dt + sigma dW
    Returns (n_paths, n_steps+1).
    """
    if rng is None:
        rng = np.random.default_rng()
    k, s, r0 = params.kappa, params.sigma, params.r0
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = r0
    dW = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
    for i in range(n_steps):
        t = times[i]
        r = paths[:, i]
        drift = params.theta_t(t) - k * r
        paths[:, i + 1] = r + drift * dt + s * dW[:, i]
    return paths


def hull_white_theta_from_market(
    market_discount_factors: np.ndarray,
    times: np.ndarray,
    kappa: float,
    sigma: float,
) -> Callable[[float], float]:
    """
    Calibrate Hull-White theta(t) from market discount factors so that
    the model exactly fits the initial yield curve.
    Uses: theta(t) = df_M/dt + kappa * f_M(t) + sigma^2/(2*kappa)*(1 - exp(-2*kappa*t))
    where f_M(t) is the market instantaneous forward rate.
    Returns an interpolating function for theta(t).
    """
    log_df = np.log(market_discount_factors)
    # Instantaneous forward rates: f(t) = -d ln P(0,t) / dt
    f_M = -np.gradient(log_df, times)
    df_M_dt = np.gradient(f_M, times)
    correction = sigma**2 / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * times))
    theta_vals = df_M_dt + kappa * f_M + correction
    interp = CubicSpline(times, theta_vals, extrapolate=True)
    return interp


# ── Nelson-Siegel Model ───────────────────────────────────────────────────────

def nelson_siegel_yield(params: NelsonSiegelParams, T: float) -> float:
    """Compute Nelson-Siegel yield for maturity T (years)."""
    if T <= 1e-8:
        return params.beta0 + params.beta1
    b0, b1, b2, tau = params.beta0, params.beta1, params.beta2, params.tau
    x = T / tau
    exp_x = math.exp(-x)
    loading1 = (1.0 - exp_x) / x
    loading2 = loading1 - exp_x
    return b0 + b1 * loading1 + b2 * loading2


def nelson_siegel_curve(params: NelsonSiegelParams, maturities: np.ndarray) -> np.ndarray:
    """Vectorized Nelson-Siegel yields."""
    tau = params.tau
    x = maturities / tau
    x = np.where(x < 1e-8, 1e-8, x)
    exp_x = np.exp(-x)
    l1 = (1.0 - exp_x) / x
    l2 = l1 - exp_x
    return params.beta0 + params.beta1 * l1 + params.beta2 * l2


def fit_nelson_siegel(
    maturities: np.ndarray,
    yields: np.ndarray,
    tau_grid: Optional[np.ndarray] = None,
) -> NelsonSiegelParams:
    """
    Fit Nelson-Siegel model to observed yields by OLS (given tau) or
    grid search over tau.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.5, 10.0, 50)

    best_sse = np.inf
    best_params = None

    for tau in tau_grid:
        x = maturities / tau
        x = np.where(x < 1e-8, 1e-8, x)
        exp_x = np.exp(-x)
        l1 = (1.0 - exp_x) / x
        l2 = l1 - exp_x
        X = np.column_stack([np.ones(len(maturities)), l1, l2])
        try:
            coeffs, res, _, _ = np.linalg.lstsq(X, yields, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = X @ coeffs
        sse = np.sum((yields - fitted) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_params = NelsonSiegelParams(
                beta0=coeffs[0], beta1=coeffs[1], beta2=coeffs[2], tau=tau
            )

    return best_params


# ── Svensson Model ────────────────────────────────────────────────────────────

def svensson_yield(params: SvenssonParams, T: float) -> float:
    """Compute Svensson yield for maturity T."""
    if T <= 1e-8:
        return params.beta0 + params.beta1
    b0, b1, b2, b3 = params.beta0, params.beta1, params.beta2, params.beta3
    t1, t2 = params.tau1, params.tau2
    x1 = T / t1
    x2 = T / t2
    exp1 = math.exp(-x1)
    exp2 = math.exp(-x2)
    l1 = (1.0 - exp1) / x1
    l2 = l1 - exp1
    l3 = (1.0 - exp2) / x2 - exp2
    return b0 + b1 * l1 + b2 * l2 + b3 * l3


def svensson_curve(params: SvenssonParams, maturities: np.ndarray) -> np.ndarray:
    """Vectorized Svensson yields."""
    b0, b1, b2, b3 = params.beta0, params.beta1, params.beta2, params.beta3
    t1, t2 = params.tau1, params.tau2
    x1 = np.where(maturities < 1e-8, 1e-8, maturities / t1)
    x2 = np.where(maturities < 1e-8, 1e-8, maturities / t2)
    exp1 = np.exp(-x1)
    exp2 = np.exp(-x2)
    l1 = (1.0 - exp1) / x1
    l2 = l1 - exp1
    l3 = (1.0 - exp2) / x2 - exp2
    return b0 + b1 * l1 + b2 * l2 + b3 * l3


def fit_svensson(
    maturities: np.ndarray,
    yields: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> SvenssonParams:
    """Fit Svensson model using nonlinear least squares."""
    if x0 is None:
        x0 = np.array([yields[-1], yields[0] - yields[-1], 1.0, 0.5, 1.5, 5.0])

    def residuals(p):
        b0, b1, b2, b3, t1, t2 = p
        t1 = max(t1, 0.05)
        t2 = max(t2, 0.05)
        sp = SvenssonParams(b0, b1, b2, b3, t1, t2)
        return svensson_curve(sp, maturities) - yields

    result = optimize.least_squares(residuals, x0, max_nfev=5000)
    p = result.x
    return SvenssonParams(p[0], p[1], p[2], p[3], max(p[4], 0.05), max(p[5], 0.05))


# ── Forward Rate Extraction ───────────────────────────────────────────────────

def forward_rate_from_ns(params: NelsonSiegelParams, T: float) -> float:
    """
    Instantaneous forward rate from Nelson-Siegel fitted curve.
    f(T) = y(T) + T * dy/dT
    Computed numerically.
    """
    dT = 1e-5
    y1 = nelson_siegel_yield(params, T + dT)
    y0 = nelson_siegel_yield(params, max(T - dT, 1e-6))
    dy_dT = (y1 - y0) / (2.0 * dT)
    return nelson_siegel_yield(params, T) + T * dy_dT


def forward_rate_from_svensson(params: SvenssonParams, T: float) -> float:
    """Instantaneous forward rate from Svensson fitted curve."""
    dT = 1e-5
    y1 = svensson_yield(params, T + dT)
    y0 = svensson_yield(params, max(T - dT, 1e-6))
    dy_dT = (y1 - y0) / (2.0 * dT)
    return svensson_yield(params, T) + T * dy_dT


def forward_rate_discrete(
    maturities: np.ndarray,
    yields: np.ndarray,
    T1: float,
    T2: float,
) -> float:
    """
    Discrete forward rate between T1 and T2 from zero rates.
    f(T1,T2) = [y(T2)*T2 - y(T1)*T1] / (T2 - T1)
    """
    y1 = np.interp(T1, maturities, yields)
    y2 = np.interp(T2, maturities, yields)
    return (y2 * T2 - y1 * T1) / (T2 - T1)


# ── Bootstrap: Par, Zero, Forward ────────────────────────────────────────────

def bootstrap_zero_rates(
    coupon_rates: np.ndarray,
    maturities: np.ndarray,
    freq: int = 2,
) -> np.ndarray:
    """
    Bootstrap zero coupon rates from par coupon rates using bootstrapping.
    Assumes annual coupon = coupon_rate, semi-annual payments by default.
    Returns continuously compounded zero rates.
    """
    n = len(maturities)
    zero_rates = np.zeros(n)
    dt = 1.0 / freq

    for i, T in enumerate(maturities):
        c = coupon_rates[i] / freq
        # coupon periods
        times = np.arange(dt, T + 1e-9, dt)
        n_coupons = len(times)
        # sum of discounted coupons using already bootstrapped zeros
        pv_coupons = 0.0
        for t in times[:-1]:
            z = np.interp(t, maturities[:i+1], zero_rates[:i+1])
            pv_coupons += c * math.exp(-z * t)
        # solve for last zero rate: (c + 1) * exp(-z_T * T) = 1 - pv_coupons
        residual = 1.0 - pv_coupons
        if residual <= 0.0:
            zero_rates[i] = zero_rates[max(i-1, 0)]
        else:
            zero_rates[i] = -math.log(residual / (1.0 + c)) / T

    return zero_rates


def par_yield_from_zeros(
    zero_rates: np.ndarray,
    maturities: np.ndarray,
    T: float,
    freq: int = 2,
) -> float:
    """
    Compute par yield at maturity T from zero curve.
    c = (1 - P(0,T)) / sum_i P(0,t_i) * freq
    """
    dt = 1.0 / freq
    times = np.arange(dt, T + 1e-9, dt)
    discount = np.array([
        math.exp(-np.interp(t, maturities, zero_rates) * t) for t in times
    ])
    P_T = discount[-1]
    annuity = np.sum(discount) / freq
    if annuity <= 1e-12:
        return 0.0
    return (1.0 - P_T) / annuity


# ── Bond Duration, Convexity, DV01 ───────────────────────────────────────────

def bond_price_from_yield(
    coupon_rate: float,
    face: float,
    ytm: float,
    maturity: float,
    freq: int = 2,
) -> float:
    """Compute bond price from YTM (continuously compounded)."""
    dt = 1.0 / freq
    times = np.arange(dt, maturity + 1e-9, dt)
    c = coupon_rate / freq * face
    pv = sum(c * math.exp(-ytm * t) for t in times)
    pv += face * math.exp(-ytm * maturity)
    return pv


def bond_metrics(
    coupon_rate: float,
    face: float,
    ytm: float,
    maturity: float,
    freq: int = 2,
) -> BondMetrics:
    """Compute full bond analytics: price, duration, convexity, DV01."""
    dt = 1.0 / freq
    times = np.arange(dt, maturity + 1e-9, dt)
    c = coupon_rate / freq * face
    cash_flows = np.array([c] * len(times))
    cash_flows[-1] += face

    discount = np.exp(-ytm * times)
    pvs = cash_flows * discount
    price = pvs.sum()

    mac_duration = (times * pvs).sum() / price
    mod_duration = mac_duration  # already continuously compounded
    convexity = (times**2 * pvs).sum() / price
    dv01 = price * mod_duration * 0.0001

    return BondMetrics(
        price=price,
        duration=mac_duration,
        modified_duration=mod_duration,
        convexity=convexity,
        dv01=dv01,
    )


# ── Spread Analysis ───────────────────────────────────────────────────────────

def z_spread(
    cash_flows: np.ndarray,
    cash_flow_times: np.ndarray,
    zero_rates: np.ndarray,
    zero_maturities: np.ndarray,
    market_price: float,
) -> float:
    """
    Compute Z-spread: constant spread added to zero rates so that
    PV(cash flows) = market price. Returns spread in basis points.
    """
    def pv_error(spread_bps):
        spread = spread_bps / 10000.0
        pv = 0.0
        for cf, t in zip(cash_flows, cash_flow_times):
            z = np.interp(t, zero_maturities, zero_rates)
            pv += cf * math.exp(-(z + spread) * t)
        return pv - market_price

    try:
        result = optimize.brentq(pv_error, -500.0, 5000.0, xtol=1e-6)
        return result
    except ValueError:
        return np.nan


def i_spread(
    bond_ytm: float,
    swap_rates: np.ndarray,
    swap_maturities: np.ndarray,
    bond_maturity: float,
) -> float:
    """
    I-spread: YTM minus interpolated swap rate at same maturity.
    Returns spread in basis points.
    """
    interp_swap = np.interp(bond_maturity, swap_maturities, swap_rates)
    return (bond_ytm - interp_swap) * 10000.0


def oas_simplified(
    z_spread_bps: float,
    option_cost_bps: float,
) -> float:
    """
    Simplified OAS computation.
    OAS ≈ Z-spread - option_cost
    For embedded call option: option_cost > 0 → OAS < Z-spread
    Returns OAS in basis points.
    """
    return z_spread_bps - option_cost_bps


def spread_metrics(
    cash_flows: np.ndarray,
    cash_flow_times: np.ndarray,
    zero_rates: np.ndarray,
    zero_maturities: np.ndarray,
    market_price: float,
    bond_ytm: float,
    swap_rates: np.ndarray,
    swap_maturities: np.ndarray,
    bond_maturity: float,
    option_cost_bps: float = 0.0,
) -> SpreadMetrics:
    """Compute all spread measures for a bond."""
    zs = z_spread(cash_flows, cash_flow_times, zero_rates, zero_maturities, market_price)
    is_ = i_spread(bond_ytm, swap_rates, swap_maturities, bond_maturity)
    oas = oas_simplified(zs if not np.isnan(zs) else 0.0, option_cost_bps)
    return SpreadMetrics(z_spread=zs, i_spread=is_, oas=oas)


# ── Yield Curve PCA ───────────────────────────────────────────────────────────

def yield_curve_pca(
    yield_matrix: np.ndarray,
    n_components: int = 3,
) -> CurvePCA:
    """
    Principal component analysis of yield curve dynamics.

    Parameters
    ----------
    yield_matrix : (n_obs, n_tenors) array of yield observations
    n_components : number of PCs to extract

    Returns
    -------
    CurvePCA with level, slope, curvature loadings and explained variance
    """
    # Demean
    Y = yield_matrix - yield_matrix.mean(axis=0)
    cov = np.cov(Y.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    evr = eigenvalues[:n_components] / total_var
    components = eigenvectors[:, :n_components].T  # (n_components, n_tenors)
    scores = Y @ eigenvectors[:, :n_components]     # (n_obs, n_components)

    return CurvePCA(
        explained_variance_ratio=evr,
        components=components,
        scores=scores,
        level=components[0] if n_components > 0 else np.array([]),
        slope=components[1] if n_components > 1 else np.array([]),
        curvature=components[2] if n_components > 2 else np.array([]),
    )


def reconstruct_curve_from_pca(
    pca: CurvePCA,
    mean_curve: np.ndarray,
    level_shift: float = 0.0,
    slope_shift: float = 0.0,
    curvature_shift: float = 0.0,
) -> np.ndarray:
    """
    Reconstruct yield curve by perturbing PCA factors.
    Useful for scenario analysis: +1bp level, steepening, etc.
    """
    shifts = np.array([level_shift, slope_shift, curvature_shift])
    n = len(pca.components)
    delta = np.zeros(mean_curve.shape)
    for i in range(min(n, 3)):
        delta += shifts[i] * pca.components[i]
    return mean_curve + delta


# ── Utility: Discount Factor Conversion ──────────────────────────────────────

def zero_rate_to_discount(rate: float, T: float) -> float:
    """Continuously compounded zero rate to discount factor."""
    return math.exp(-rate * T)


def discount_to_zero_rate(df: float, T: float) -> float:
    """Discount factor to continuously compounded zero rate."""
    if df <= 0.0 or T <= 0.0:
        return 0.0
    return -math.log(df) / T


def forward_discount(
    df_T1: float,
    df_T2: float,
) -> float:
    """Forward discount factor between T1 and T2."""
    if df_T1 <= 0.0:
        return 0.0
    return df_T2 / df_T1


def implied_forward_rate(
    zero_rate_T1: float,
    T1: float,
    zero_rate_T2: float,
    T2: float,
) -> float:
    """
    Forward rate implied between T1 and T2 from zero rates.
    Continuously compounded.
    """
    if T2 <= T1:
        raise ValueError("T2 must be greater than T1")
    return (zero_rate_T2 * T2 - zero_rate_T1 * T1) / (T2 - T1)


# ── Calibration Utilities ─────────────────────────────────────────────────────

def calibrate_vasicek(
    observed_yields: np.ndarray,
    maturities: np.ndarray,
    r0: float,
) -> VasicekParams:
    """
    Calibrate Vasicek model to observed yield curve via least squares.
    """
    def residuals(p):
        kappa, theta, sigma = p
        kappa = max(kappa, 1e-4)
        sigma = max(sigma, 1e-6)
        vp = VasicekParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)
        fitted = np.array([vasicek_yield(vp, T) for T in maturities])
        return fitted - observed_yields

    x0 = np.array([0.5, observed_yields[-1], 0.01])
    result = optimize.least_squares(residuals, x0, bounds=(
        [1e-4, -0.1, 1e-6], [10.0, 0.3, 1.0]
    ))
    k, th, s = result.x
    return VasicekParams(kappa=max(k, 1e-4), theta=th, sigma=max(s, 1e-6), r0=r0)


def calibrate_cir(
    observed_yields: np.ndarray,
    maturities: np.ndarray,
    r0: float,
) -> CIRParams:
    """Calibrate CIR model to observed yield curve."""
    def residuals(p):
        kappa, theta, sigma = p
        kappa = max(kappa, 1e-4)
        theta = max(theta, 1e-6)
        sigma = max(sigma, 1e-6)
        cp = CIRParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)
        fitted = np.array([cir_yield(cp, T) for T in maturities])
        return fitted - observed_yields

    x0 = np.array([0.5, max(r0, 0.01), 0.05])
    result = optimize.least_squares(
        residuals, x0,
        bounds=([1e-4, 1e-6, 1e-6], [10.0, 0.5, 1.0])
    )
    k, th, s = result.x
    return CIRParams(kappa=max(k, 1e-4), theta=max(th, 1e-6), sigma=max(s, 1e-6), r0=r0)
