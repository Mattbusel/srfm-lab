"""
lib/math/volatility_models.py

Advanced volatility models for quantitative research.

Implements:
  - EWMA (RiskMetrics): exponentially weighted moving average volatility
  - HAR-RV (Heterogeneous AutoRegressive Realized Volatility)
  - Realized variance from high-frequency returns
  - Bipower variation (jump-robust realized variance)
  - Rough volatility: fractional Brownian motion simulation and Hurst estimation
  - Volatility term structure: VIX-style implied volatility index computation
  - CBOE-style VIX calculation (model-free implied volatility)
  - Variance risk premium: IV² - RV²
  - Volatility surface interpolation (cubic spline in strike space)
  - GARCH(1,1) forecasting with multiple horizons
  - Realized GARCH model (incorporates realized measures)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from scipy import optimize, interpolate, stats


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class GARCHParams:
    omega: float   # constant
    alpha: float   # ARCH coefficient
    beta: float    # GARCH coefficient
    mu: float = 0.0  # mean return


@dataclass
class RealizedGARCHParams:
    omega: float
    alpha: float   # coefficient on lagged realized measure
    beta: float    # GARCH persistence
    gamma: float   # coefficient linking realized to conditional variance
    xi: float      # measurement equation intercept
    phi: float     # measurement equation slope
    delta: float   # leverage in measurement equation


@dataclass
class HARParams:
    mu: float       # intercept
    beta_d: float   # daily RV coefficient
    beta_w: float   # weekly RV coefficient (5-day average)
    beta_m: float   # monthly RV coefficient (22-day average)


@dataclass
class VolSurface:
    strikes: np.ndarray       # (n_strikes,)
    maturities: np.ndarray    # (n_maturities,)
    implied_vols: np.ndarray  # (n_maturities, n_strikes)


@dataclass
class VIXResult:
    vix: float                # annualized VIX-style index (%)
    variance_near: float
    variance_next: float
    t_near: float
    t_next: float


@dataclass
class VarianceRiskPremium:
    vrp: float                # IV^2 - RV^2 (annualized, %)
    implied_var: float
    realized_var: float
    zscore: float             # relative to historical VRP distribution


# ── EWMA Volatility ───────────────────────────────────────────────────────────

def ewma_volatility(
    returns: np.ndarray,
    lam: float = 0.94,
    annualize: bool = True,
    trading_days: int = 252,
) -> np.ndarray:
    """
    EWMA (RiskMetrics) volatility estimate.

    sigma²_t = lambda * sigma²_{t-1} + (1 - lambda) * r²_{t-1}

    Parameters
    ----------
    returns : 1D array of daily returns
    lam : decay factor (0.94 for daily, 0.97 for monthly)
    annualize : if True, multiply by sqrt(trading_days)
    """
    n = len(returns)
    var = np.empty(n)
    var[0] = returns[0] ** 2
    for i in range(1, n):
        var[i] = lam * var[i - 1] + (1.0 - lam) * returns[i - 1] ** 2
    vol = np.sqrt(var)
    if annualize:
        vol *= math.sqrt(trading_days)
    return vol


def ewma_covariance(
    returns_x: np.ndarray,
    returns_y: np.ndarray,
    lam: float = 0.94,
) -> np.ndarray:
    """EWMA covariance between two return series."""
    n = len(returns_x)
    cov = np.empty(n)
    cov[0] = returns_x[0] * returns_y[0]
    for i in range(1, n):
        cov[i] = lam * cov[i - 1] + (1.0 - lam) * returns_x[i - 1] * returns_y[i - 1]
    return cov


# ── Realized Variance ─────────────────────────────────────────────────────────

def realized_variance(
    intraday_returns: np.ndarray,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """
    Realized variance from high-frequency returns.
    RV = sum_i r_i^2
    """
    rv = np.sum(intraday_returns ** 2)
    if annualize:
        rv *= trading_days
    return rv


def realized_variance_series(
    returns: np.ndarray,
    freq: int = 78,
    annualize: bool = True,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Compute daily realized variance from a flat array of intraday returns.
    Assumes returns are ordered: [day1_bar1, ..., day1_barN, day2_bar1, ...]

    Parameters
    ----------
    returns : flat array of intraday returns
    freq : number of bars per day
    """
    n_days = len(returns) // freq
    rv = np.array([
        realized_variance(returns[i * freq:(i + 1) * freq], annualize=annualize,
                         trading_days=trading_days)
        for i in range(n_days)
    ])
    return rv


def bipower_variation(
    intraday_returns: np.ndarray,
    annualize: bool = True,
    trading_days: int = 252,
) -> float:
    """
    Bipower variation (jump-robust realized variance estimator).
    BV = (pi/2) * sum_i |r_i| * |r_{i-1}|

    Under continuous semimartingale, BV → IV (integrated variance).
    Jump component = max(RV - BV, 0).
    """
    mu1 = math.sqrt(2.0 / math.pi)  # E[|Z|] for Z ~ N(0,1)
    r_abs = np.abs(intraday_returns)
    bv = (1.0 / mu1**2) * np.sum(r_abs[1:] * r_abs[:-1])
    if annualize:
        bv *= trading_days
    return bv


def jump_variation(
    intraday_returns: np.ndarray,
    annualize: bool = True,
    trading_days: int = 252,
) -> Tuple[float, float]:
    """
    Decompose realized variance into continuous and jump components.
    Returns (continuous_variation, jump_variation).
    """
    rv = realized_variance(intraday_returns, annualize, trading_days)
    bv = bipower_variation(intraday_returns, annualize, trading_days)
    jv = max(rv - bv, 0.0)
    return bv, jv


# ── HAR-RV Model ──────────────────────────────────────────────────────────────

def har_features(
    rv_series: np.ndarray,
    window_w: int = 5,
    window_m: int = 22,
) -> np.ndarray:
    """
    Compute HAR feature matrix: [1, RV_d, RV_w, RV_m].
    Returns (n_valid, 4) array.
    """
    n = len(rv_series)
    start = window_m
    rows = []
    for i in range(start, n):
        rv_d = rv_series[i - 1]
        rv_w = rv_series[max(0, i - window_w):i].mean()
        rv_m = rv_series[max(0, i - window_m):i].mean()
        rows.append([1.0, rv_d, rv_w, rv_m])
    return np.array(rows)


def fit_har(
    rv_series: np.ndarray,
    window_w: int = 5,
    window_m: int = 22,
) -> HARParams:
    """
    Estimate HAR-RV model by OLS.
    RV_{t+1} = mu + beta_d * RV_d + beta_w * RV_w + beta_m * RV_m + eps
    """
    X = har_features(rv_series, window_w, window_m)
    y = rv_series[window_m:]
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return HARParams(mu=coeffs[0], beta_d=coeffs[1], beta_w=coeffs[2], beta_m=coeffs[3])


def har_forecast(
    params: HARParams,
    rv_series: np.ndarray,
    horizon: int = 1,
    window_w: int = 5,
    window_m: int = 22,
) -> float:
    """
    One-step-ahead HAR-RV forecast from current data.
    For multi-step horizons, uses recursive substitution.
    """
    rv_d = rv_series[-1]
    rv_w = rv_series[-window_w:].mean()
    rv_m = rv_series[-window_m:].mean()
    fc = params.mu + params.beta_d * rv_d + params.beta_w * rv_w + params.beta_m * rv_m
    if horizon > 1:
        # Simple iterative: treat forecast as next RV and re-forecast
        series = list(rv_series)
        for _ in range(horizon - 1):
            series.append(fc)
            arr = np.array(series)
            rv_d = arr[-1]
            rv_w = arr[-window_w:].mean()
            rv_m = arr[-window_m:].mean()
            fc = params.mu + params.beta_d * rv_d + params.beta_w * rv_w + params.beta_m * rv_m
    return max(fc, 0.0)


# ── Rough Volatility / Fractional Brownian Motion ────────────────────────────

def fbm_cholesky(n: int, H: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Simulate fractional Brownian motion with Hurst exponent H via Cholesky.
    Returns (n,) array of fBm increments (fGn).
    For H=0.5 reduces to standard Brownian motion.
    For H<0.5 (rough): anti-persistent, used in rough vol models.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Covariance of fGn
    k = np.arange(n)
    cov_row = 0.5 * (
        np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H)
    )
    cov_row[0] = 1.0  # var = 1 for normalized fGn
    # Build Toeplitz covariance matrix
    from scipy.linalg import toeplitz
    C = toeplitz(cov_row)
    # Regularize
    C += np.eye(n) * 1e-10
    try:
        L = np.linalg.cholesky(C)
        z = rng.standard_normal(n)
        return L @ z
    except np.linalg.LinAlgError:
        # Fallback: eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 0.0)
        z = rng.standard_normal(n)
        return eigvecs @ (np.sqrt(eigvals) * z)


def hurst_exponent_rs(series: np.ndarray) -> float:
    """
    Estimate Hurst exponent using Rescaled Range (R/S) analysis.
    Returns H ∈ (0, 1). H < 0.5 = rough/anti-persistent.
    """
    n = len(series)
    lags = np.floor(np.logspace(1, np.log10(n // 2), 20)).astype(int)
    lags = np.unique(lags)
    rs_vals = []
    for lag in lags:
        chunks = [series[i:i + lag] for i in range(0, n - lag + 1, lag)]
        rs_chunk = []
        for chunk in chunks:
            mean_c = chunk.mean()
            dev = np.cumsum(chunk - mean_c)
            R = dev.max() - dev.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))

    if len(rs_vals) < 2:
        return 0.5
    log_lags = np.log(lags[:len(rs_vals)])
    log_rs = np.log(rs_vals)
    slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
    return float(np.clip(slope, 0.01, 0.99))


def rough_vol_simulate(
    H: float,
    nu: float,
    rho: float,
    xi0: float,
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified rough Bergomi-style simulation.
    dS/S = sqrt(V) dW
    V_t = xi0 * exp(nu * W^H_t - 0.5 * nu^2 * t^(2H))
    where W^H is correlated fBm with Hurst H.

    Returns (price_paths, vol_paths) each (n_paths, n_steps+1).
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    price_paths = np.empty((n_paths, n_steps + 1))
    vol_paths = np.empty((n_paths, n_steps + 1))
    price_paths[:, 0] = S0
    vol_paths[:, 0] = xi0

    for p in range(n_paths):
        fgn = fbm_cholesky(n_steps, H, rng) * math.sqrt(dt)
        W_H = np.cumsum(fgn)
        dW_S = rng.standard_normal(n_steps) * math.sqrt(dt)
        # Correlated increments
        dW_S = rho * fgn + math.sqrt(1.0 - rho**2) * dW_S

        V = np.empty(n_steps + 1)
        V[0] = xi0
        for i in range(1, n_steps + 1):
            t = times[i]
            V[i] = xi0 * math.exp(nu * W_H[i - 1] - 0.5 * nu**2 * t**(2 * H))

        S = np.empty(n_steps + 1)
        S[0] = S0
        for i in range(n_steps):
            S[i + 1] = S[i] * math.exp(-0.5 * V[i] * dt + math.sqrt(max(V[i], 0.0)) * dW_S[i])

        price_paths[p] = S
        vol_paths[p] = V

    return price_paths, vol_paths


# ── VIX-Style Computation ─────────────────────────────────────────────────────

def vix_style_variance(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    F: float,
    T: float,
    r: float,
    K0: Optional[float] = None,
) -> float:
    """
    CBOE VIX-style model-free implied variance for a single maturity.
    sigma^2 = (2/T) * sum_i [DeltaK_i / K_i^2] * exp(rT) * Q(K_i) - (1/T)*[(F/K0) - 1]^2

    Parameters
    ----------
    strikes : sorted array of strikes
    call_prices : call option prices at each strike
    put_prices : put option prices at each strike
    F : forward price
    T : time to expiry (years)
    r : risk-free rate
    K0 : ATM reference strike (below F); if None, uses closest below F
    """
    n = len(strikes)
    if K0 is None:
        otm_idx = np.searchsorted(strikes, F) - 1
        K0 = strikes[max(otm_idx, 0)]

    # Midpoint prices: puts for K < K0, calls for K > K0, average at K0
    Q = np.where(strikes < K0, put_prices,
                 np.where(strikes > K0, call_prices,
                          0.5 * (call_prices + put_prices)))

    # Delta K
    dK = np.empty(n)
    dK[0] = strikes[1] - strikes[0]
    dK[-1] = strikes[-1] - strikes[-2]
    dK[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0

    disc = math.exp(r * T)
    sigma2 = (2.0 / T) * np.sum(dK / strikes**2 * disc * Q)
    adj = (F / K0 - 1.0) ** 2 / T
    return max(sigma2 - adj, 0.0)


def compute_vix(
    near_strikes: np.ndarray,
    near_calls: np.ndarray,
    near_puts: np.ndarray,
    next_strikes: np.ndarray,
    next_calls: np.ndarray,
    next_puts: np.ndarray,
    F_near: float,
    F_next: float,
    T_near: float,
    T_next: float,
    r: float,
    target_days: int = 30,
) -> VIXResult:
    """
    Full CBOE VIX computation interpolating between two maturities.
    Returns VIX index (annualized %).
    """
    N_target = target_days * 24 * 60   # in minutes
    N_near = T_near * 365 * 24 * 60
    N_next = T_next * 365 * 24 * 60
    N_30 = 30 * 24 * 60

    var_near = vix_style_variance(near_strikes, near_calls, near_puts, F_near, T_near, r)
    var_next = vix_style_variance(next_strikes, next_calls, next_puts, F_next, T_next, r)

    w_near = (N_next - N_30) / (N_next - N_near)
    w_next = (N_30 - N_near) / (N_next - N_near)

    sigma2 = (T_near * var_near * w_near + T_next * var_next * w_next) * (365.0 / target_days)
    vix = 100.0 * math.sqrt(max(sigma2, 0.0))

    return VIXResult(
        vix=vix,
        variance_near=var_near,
        variance_next=var_next,
        t_near=T_near,
        t_next=T_next,
    )


# ── Variance Risk Premium ─────────────────────────────────────────────────────

def variance_risk_premium(
    implied_vol_series: np.ndarray,
    realized_vol_series: np.ndarray,
    window: int = 252,
) -> np.ndarray:
    """
    Compute rolling variance risk premium: IV² - RV².
    Both series should be annualized (%).
    Returns array of VRP values.
    """
    iv2 = implied_vol_series**2
    rv2 = realized_vol_series**2
    vrp = iv2 - rv2
    return vrp


def vrp_zscore(
    implied_vol: float,
    realized_vol: float,
    vrp_history: np.ndarray,
) -> VarianceRiskPremium:
    """Compute current VRP and its z-score relative to history."""
    iv2 = implied_vol**2
    rv2 = realized_vol**2
    vrp_val = iv2 - rv2
    z = (vrp_val - vrp_history.mean()) / (vrp_history.std() + 1e-10)
    return VarianceRiskPremium(
        vrp=vrp_val,
        implied_var=iv2,
        realized_var=rv2,
        zscore=float(z),
    )


# ── Volatility Term Structure ─────────────────────────────────────────────────

def vol_term_structure(
    atm_vols: np.ndarray,
    maturities: np.ndarray,
    query_maturities: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute volatility term structure via cubic spline interpolation.
    Ensures total variance is monotonically increasing (no calendar arbitrage).

    Parameters
    ----------
    atm_vols : ATM implied vols for each maturity
    maturities : option maturities in years
    query_maturities : maturities at which to evaluate; if None, returns input
    """
    total_var = atm_vols**2 * maturities
    # Ensure monotone total variance
    for i in range(1, len(total_var)):
        if total_var[i] < total_var[i - 1]:
            total_var[i] = total_var[i - 1] * 1.0001

    if query_maturities is None:
        query_maturities = maturities

    cs = CubicSpline(maturities, total_var, extrapolate=True)
    tv_interp = np.maximum(cs(query_maturities), 1e-10)
    q_mats = np.maximum(query_maturities, 1e-10)
    return np.sqrt(tv_interp / q_mats)


# ── Volatility Surface Interpolation ─────────────────────────────────────────

def interpolate_vol_surface(
    surface: VolSurface,
    query_strike: float,
    query_maturity: float,
) -> float:
    """
    Bilinear interpolation on volatility surface.
    For each maturity slice, uses cubic spline in strike space.
    Then interpolates across maturities.
    """
    mats = surface.maturities
    stks = surface.strikes
    ivs = surface.implied_vols

    # Clamp queries
    K = float(np.clip(query_strike, stks[0], stks[-1]))
    T = float(np.clip(query_maturity, mats[0], mats[-1]))

    # Interpolate vol at query_strike for each maturity slice
    slice_vols = np.empty(len(mats))
    for j, _ in enumerate(mats):
        cs = CubicSpline(stks, ivs[j], extrapolate=True)
        slice_vols[j] = float(cs(K))

    # Interpolate across maturities
    cs_mat = CubicSpline(mats, slice_vols, extrapolate=True)
    return float(np.maximum(cs_mat(T), 1e-6))


def build_vol_surface_from_data(
    strikes: np.ndarray,
    maturities: np.ndarray,
    implied_vols: np.ndarray,
) -> VolSurface:
    """
    Build a VolSurface from raw data.
    implied_vols should be (n_maturities, n_strikes).
    """
    return VolSurface(
        strikes=np.sort(strikes),
        maturities=np.sort(maturities),
        implied_vols=implied_vols,
    )


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

def garch_variance_series(
    returns: np.ndarray,
    params: GARCHParams,
) -> np.ndarray:
    """
    Compute GARCH(1,1) conditional variance series.
    h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
    """
    n = len(returns)
    h = np.empty(n)
    h[0] = params.omega / (1.0 - params.alpha - params.beta)
    for i in range(1, n):
        eps_prev = returns[i - 1] - params.mu
        h[i] = params.omega + params.alpha * eps_prev**2 + params.beta * h[i - 1]
    return h


def garch_forecast(
    params: GARCHParams,
    h_last: float,
    eps_last: float,
    horizon: int = 10,
    annualize: bool = True,
    trading_days: int = 252,
) -> np.ndarray:
    """
    Multi-step GARCH(1,1) variance forecast.
    E[h_{t+k}] = omega/(1-alpha-beta) + (alpha+beta)^k * (h_t - omega/(1-alpha-beta))
    Returns array of volatility forecasts (annualized if requested).
    """
    omega, alpha, beta = params.omega, params.alpha, params.beta
    persistence = alpha + beta
    long_run_var = omega / max(1.0 - persistence, 1e-10)

    # One-step: use realized epsilon
    h1 = omega + alpha * (eps_last - params.mu)**2 + beta * h_last

    forecasts = np.empty(horizon)
    for k in range(horizon):
        if k == 0:
            forecasts[0] = h1
        else:
            forecasts[k] = long_run_var + persistence**k * (h1 - long_run_var)

    vol_forecasts = np.sqrt(np.maximum(forecasts, 0.0))
    if annualize:
        vol_forecasts *= math.sqrt(trading_days)
    return vol_forecasts


def fit_garch(
    returns: np.ndarray,
    mu: float = 0.0,
) -> GARCHParams:
    """
    Estimate GARCH(1,1) parameters via maximum likelihood (Gaussian).
    """
    def neg_log_likelihood(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.9999:
            return 1e10
        h = np.empty(len(returns))
        h[0] = np.var(returns)
        eps2 = (returns - mu)**2
        for i in range(1, len(returns)):
            h[i] = omega + alpha * eps2[i - 1] + beta * h[i - 1]
        h = np.maximum(h, 1e-10)
        nll = 0.5 * np.sum(np.log(h) + eps2 / h)
        return nll

    # Initial guess from moments
    var0 = np.var(returns)
    x0 = np.array([var0 * 0.05, 0.1, 0.85])
    bounds = [(1e-8, None), (1e-6, 0.999), (1e-6, 0.999)]

    result = optimize.minimize(
        neg_log_likelihood, x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}
    )
    omega, alpha, beta = result.x
    # Normalize if persistence >= 1
    if alpha + beta >= 1.0:
        total = alpha + beta
        alpha /= total * 1.001
        beta /= total * 1.001
    return GARCHParams(omega=omega, alpha=alpha, beta=beta, mu=mu)


# ── Realized GARCH ────────────────────────────────────────────────────────────

def realized_garch_variance(
    returns: np.ndarray,
    realized_vars: np.ndarray,
    params: RealizedGARCHParams,
) -> np.ndarray:
    """
    Realized GARCH conditional variance.
    log(h_t) = omega + beta * log(h_{t-1}) + gamma * log(RV_{t-1})
    Measurement: log(RV_t) = xi + phi * log(h_t) + delta * z_t + u_t
    Returns conditional variance series.
    """
    n = len(returns)
    h = np.empty(n)
    log_h = math.log(np.var(returns) + 1e-10)

    for i in range(n):
        h[i] = math.exp(log_h)
        if i < n - 1:
            rv_i = max(realized_vars[i], 1e-12)
            log_h = (params.omega
                     + params.beta * log_h
                     + params.gamma * math.log(rv_i))
    return h


def realized_garch_forecast(
    params: RealizedGARCHParams,
    h_last: float,
    rv_last: float,
    horizon: int = 5,
    annualize: bool = True,
    trading_days: int = 252,
) -> np.ndarray:
    """Multi-step Realized GARCH forecast."""
    log_h = math.log(max(h_last, 1e-12))
    log_rv = math.log(max(rv_last, 1e-12))
    forecasts = np.empty(horizon)
    for k in range(horizon):
        if k == 0:
            log_h_next = params.omega + params.beta * log_h + params.gamma * log_rv
        else:
            # Expected log(RV) from measurement equation
            log_rv_exp = params.xi + params.phi * log_h
            log_h_next = params.omega + params.beta * log_h + params.gamma * log_rv_exp
        forecasts[k] = math.exp(log_h_next)
        log_h = log_h_next
        log_rv = params.xi + params.phi * log_h_next

    vol = np.sqrt(forecasts)
    if annualize:
        vol *= math.sqrt(trading_days)
    return vol


# ── Convenience: Composite Vol Summary ───────────────────────────────────────

def vol_summary(
    returns: np.ndarray,
    realized_var_daily: Optional[np.ndarray] = None,
    ewma_lambda: float = 0.94,
    har_window: int = 22,
    trading_days: int = 252,
) -> dict:
    """
    Compute a comprehensive volatility summary for a return series.
    Returns dict with spot vol estimates and HAR forecast.
    """
    ewma_vol = ewma_volatility(returns, lam=ewma_lambda, trading_days=trading_days)
    spot_ewma = float(ewma_vol[-1])

    # Simple realized (using daily returns squared)
    rv_daily = returns**2 * trading_days
    har_params = fit_har(rv_daily)
    har_fc = har_forecast(har_params, rv_daily, horizon=1)

    result = {
        "spot_ewma_vol_ann": spot_ewma,
        "har_rv_1d_forecast": float(har_fc),
        "har_rv_vol_forecast": float(math.sqrt(max(har_fc, 0.0))),
        "realized_vol_30d": float(math.sqrt(rv_daily[-30:].mean())),
        "realized_vol_252d": float(math.sqrt(rv_daily.mean())),
    }

    if realized_var_daily is not None and len(realized_var_daily) >= 22:
        har_params_rv = fit_har(realized_var_daily)
        har_fc_rv = har_forecast(har_params_rv, realized_var_daily, horizon=5)
        result["har_rv_5d_forecast"] = float(har_fc_rv)
        result["hurst_estimate"] = float(hurst_exponent_rs(returns[-252:] if len(returns) >= 252 else returns))

    return result
