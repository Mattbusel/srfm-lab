"""
Mean reversion signals for pairs and statistical arbitrage.

Implements:
  - OU-process z-score signal
  - Bollinger Band mean reversion
  - Kalman spread z-score
  - Half-life adaptive position sizing
  - RSI mean reversion
  - Pairs cointegration signal
  - Multi-asset basket mean reversion
  - Fractional cointegration signal
  - VECM-based spread signal
  - Threshold AR (TAR) nonlinear mean reversion
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── OU Process Utilities ──────────────────────────────────────────────────────

@dataclass
class OUParams:
    kappa: float = 1.0    # mean reversion speed
    mu: float = 0.0       # long-run mean
    sigma: float = 0.01   # diffusion
    dt: float = 1.0       # time step


def fit_ou(spread: np.ndarray, dt: float = 1.0) -> OUParams:
    """
    Fit OU parameters from spread series via OLS:
    Δx_t = κ(μ - x_{t-1})dt + σ√dt * ε_t
    """
    x = spread[:-1]
    dx = np.diff(spread)

    # OLS: dx = a + b*x + eps
    n = len(x)
    X_reg = np.column_stack([np.ones(n), x])
    try:
        coeffs = np.linalg.lstsq(X_reg, dx, rcond=None)[0]
    except np.linalg.LinAlgError:
        return OUParams()

    a, b = coeffs
    kappa = max(-b / dt, 1e-6)
    mu = a / (kappa * dt)
    resid = dx - X_reg @ coeffs
    sigma = float(resid.std() / math.sqrt(dt))

    return OUParams(kappa=float(kappa), mu=float(mu), sigma=float(sigma), dt=dt)


def ou_half_life(params: OUParams) -> float:
    """Half-life of mean reversion in time steps."""
    return float(math.log(2) / max(params.kappa * params.dt, 1e-10))


def ou_zscore(spread: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    Z-score of spread using rolling OU equilibrium.
    If window is None, uses full-sample OU fit.
    """
    if window is None:
        params = fit_ou(spread)
        eq_std = params.sigma / math.sqrt(2 * params.kappa * params.dt + 1e-10)
        return (spread - params.mu) / max(eq_std, 1e-10)

    z = np.zeros(len(spread))
    for i in range(window, len(spread)):
        sub = spread[i - window: i]
        mu = sub.mean()
        sigma = sub.std()
        z[i] = (spread[i] - mu) / max(sigma, 1e-10)
    return z


# ── Bollinger Band Signal ──────────────────────────────────────────────────────

def bollinger_signal(
    prices: np.ndarray,
    window: int = 20,
    n_std: float = 2.0,
) -> dict:
    """
    Bollinger Band mean reversion signal.
    Returns signal: -1 (below lower), +1 (above upper), 0 (inside).
    """
    T = len(prices)
    upper = np.zeros(T)
    lower = np.zeros(T)
    mid = np.zeros(T)
    z = np.zeros(T)

    for i in range(window, T):
        sub = prices[i - window: i]
        m = sub.mean()
        s = sub.std()
        mid[i] = m
        upper[i] = m + n_std * s
        lower[i] = m - n_std * s
        z[i] = (prices[i] - m) / max(s, 1e-10)

    signal = np.where(z > n_std, 1.0, np.where(z < -n_std, -1.0, 0.0))

    return {
        "signal": signal,
        "z_score": z,
        "upper": upper,
        "lower": lower,
        "mid": mid,
        "bandwidth": (upper - lower) / (mid + 1e-10),
    }


# ── Kalman Spread Signal ──────────────────────────────────────────────────────

def kalman_spread_signal(
    y: np.ndarray,
    x: np.ndarray,
    delta: float = 1e-4,
    R_obs: float = 0.001,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> dict:
    """
    Time-varying Kalman filter spread signal for pairs trading.
    Returns trade signals: +1 (long spread), -1 (short spread), 0 (flat).
    """
    T = min(len(y), len(x))
    spreads = np.zeros(T)
    betas = np.zeros(T)
    alphas = np.zeros(T)

    # Kalman state
    state_x = np.zeros(2)       # [alpha, beta]
    state_P = np.eye(2) * 10.0
    Q_proc = np.eye(2) * delta

    for t in range(T):
        H = np.array([1.0, x[t]])
        P_pred = state_P + Q_proc
        y_pred = H @ state_x
        innov = y[t] - y_pred
        S = H @ P_pred @ H + R_obs
        K = P_pred @ H / S
        state_x = state_x + K * innov
        state_P = (np.eye(2) - np.outer(K, H)) @ P_pred

        alphas[t], betas[t] = state_x
        spreads[t] = y[t] - betas[t] * x[t] - alphas[t]

    # Z-score of spread
    mu_s = np.zeros(T)
    sigma_s = np.ones(T)
    window = 20
    for i in range(window, T):
        sub = spreads[i - window: i]
        mu_s[i] = sub.mean()
        sigma_s[i] = max(sub.std(), 1e-10)

    z = (spreads - mu_s) / sigma_s

    # Entry/exit logic
    signal = np.zeros(T)
    position = 0
    for t in range(window, T):
        if position == 0:
            if z[t] > entry_z:
                position = -1
            elif z[t] < -entry_z:
                position = 1
        else:
            if abs(z[t]) < exit_z:
                position = 0
        signal[t] = position

    return {
        "spreads": spreads,
        "betas": betas,
        "alphas": alphas,
        "z_score": z,
        "signal": signal,
        "half_life": float(ou_half_life(fit_ou(spreads))),
    }


# ── RSI Mean Reversion ────────────────────────────────────────────────────────

def rsi_mean_reversion_signal(
    prices: np.ndarray,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> dict:
    """
    RSI-based mean reversion signal.
    Buy when oversold, sell when overbought.
    """
    T = len(prices)
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.zeros(T)
    avg_gain = np.zeros(T)
    avg_loss = np.zeros(T)

    if T > period:
        avg_gain[period] = gains[1:period + 1].mean()
        avg_loss[period] = losses[1:period + 1].mean()
        for i in range(period + 1, T):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    rsi[:period] = 50

    signal = np.where(rsi < oversold, 1.0, np.where(rsi > overbought, -1.0, 0.0))

    return {
        "rsi": rsi,
        "signal": signal,
        "oversold_frac": float((rsi < oversold).mean()),
        "overbought_frac": float((rsi > overbought).mean()),
    }


# ── Half-life Adaptive Position Sizing ────────────────────────────────────────

def halflife_position_sizer(
    z_score: np.ndarray,
    spread: np.ndarray,
    max_position: float = 1.0,
    target_half_life_range: tuple = (5, 60),
) -> np.ndarray:
    """
    Scale position size by:
    1. |z-score| magnitude
    2. Quality of mean reversion (half-life in acceptable range)
    """
    min_hl, max_hl = target_half_life_range
    params = fit_ou(spread)
    hl = ou_half_life(params)

    # Position quality: 1 if HL in range, decays outside
    if hl < min_hl:
        quality = max(0.0, hl / min_hl)
    elif hl > max_hl:
        quality = max(0.0, 1 - (hl - max_hl) / max_hl)
    else:
        quality = 1.0

    # Position magnitude from z-score
    z_clipped = np.clip(np.abs(z_score), 0, 4)
    raw_pos = np.sign(-z_score) * z_clipped / 4.0  # -1 to +1

    return raw_pos * quality * max_position


# ── Basket Mean Reversion ──────────────────────────────────────────────────────

def basket_spread(
    prices: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute weighted basket price series.
    prices: (T, N), weights: (N,) — defaults to equal-weight long/short.
    """
    if weights is None:
        n = prices.shape[1]
        weights = np.ones(n) / n

    return prices @ weights


def johansen_cointegration(
    Y: np.ndarray,
    p: int = 1,
) -> dict:
    """
    Simplified Johansen cointegration test (trace test).
    Y: (T, N) matrix of price series.
    Returns cointegration vectors and trace statistics.
    """
    T, N = Y.shape
    dY = np.diff(Y, axis=0)
    Y_lag = Y[:-1]

    # Regress dY on lagged levels
    X = np.column_stack([Y_lag, np.ones(T - 1)])
    try:
        beta = np.linalg.lstsq(X, dY, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"n_cointegrated": 0, "vectors": np.eye(N)[:1]}

    pi = beta[:N]  # coefficient matrix on lagged levels

    # Eigenvalue decomposition of pi
    eigvals, eigvecs = np.linalg.eig(pi.T @ pi)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx].real
    eigvecs = eigvecs[:, idx].real

    # Trace statistics (approximate)
    trace_stats = [-T * np.log(max(1 - lam, 1e-10)) for lam in eigvals]

    # Critical values at 5% (N=2: 15.41, 3.76)
    cv_5pct = {1: 3.76, 2: 15.41, 3: 29.68, 4: 47.21}

    n_cointvec = sum(
        ts > cv_5pct.get(N, 50) * (i + 1) / N
        for i, ts in enumerate(trace_stats)
    )

    return {
        "n_cointegrated": int(min(n_cointvec, N - 1)),
        "eigenvalues": eigvals.tolist(),
        "trace_stats": [float(ts) for ts in trace_stats],
        "vectors": eigvecs[:, :max(n_cointvec, 1)],
    }


# ── Threshold AR Mean Reversion ───────────────────────────────────────────────

def tar_model(
    spread: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    """
    Threshold Autoregressive (TAR) model for nonlinear mean reversion.
    Mean reversion only when spread exceeds threshold.
    Two-regime AR: inner regime (|x| < c) and outer regime (|x| >= c).
    """
    n = len(spread)
    x = spread[:-1]
    dx = np.diff(spread)

    if threshold is None:
        threshold = float(np.abs(spread).std())

    inner_mask = np.abs(x) < threshold
    outer_mask = ~inner_mask

    def fit_regime(mask):
        if mask.sum() < 3:
            return {"kappa": 0.0, "mu": spread.mean(), "sigma": spread.std()}
        X_r = np.column_stack([np.ones(mask.sum()), x[mask]])
        coeffs = np.linalg.lstsq(X_r, dx[mask], rcond=None)[0]
        a, b = coeffs
        kappa = max(-b, 0)
        mu = a / (kappa + 1e-10)
        resid = dx[mask] - X_r @ coeffs
        return {"kappa": float(kappa), "mu": float(mu), "sigma": float(resid.std())}

    inner_params = fit_regime(inner_mask)
    outer_params = fit_regime(outer_mask)

    return {
        "threshold": float(threshold),
        "inner_regime": inner_params,
        "outer_regime": outer_params,
        "outer_kappa_ratio": float(
            outer_params["kappa"] / (inner_params["kappa"] + 1e-10)
        ),
        "stronger_reversion_outer": outer_params["kappa"] > inner_params["kappa"],
        "n_inner": int(inner_mask.sum()),
        "n_outer": int(outer_mask.sum()),
    }


# ── Fractional Cointegration ──────────────────────────────────────────────────

def fractional_spread_signal(
    spread: np.ndarray,
    d_range: tuple = (0.3, 0.7),
    window: int = 100,
) -> dict:
    """
    Signal based on fractional integration of the spread.
    If 0 < d < 0.5: spread is fractionally integrated (long memory, slow reversion).
    If d close to 0: strong mean reversion.
    Compute optimal d via Hurst exponent proxy.
    """
    from lib.math.fractal import hurst_dfa

    # Estimate Hurst exponent
    try:
        H = hurst_dfa(spread, min_window=10, max_window=len(spread) // 4)
    except Exception:
        H = 0.5

    # d = H - 0.5 (memory parameter)
    d = H - 0.5

    # Signal strength: stronger when d is more negative (fast reversion)
    d_min, d_max = d_range

    if d < d_min - 0.5:
        # Anti-persistent: very fast reversion, but noisy
        quality = 0.5
    elif d < 0:
        # Mean reverting: good for pairs trading
        quality = 1.0
    else:
        # Trending or random walk
        quality = max(0.0, 1.0 - d * 2)

    # Z-score signal
    if window < len(spread):
        mu = spread[-window:].mean()
        sigma = spread[-window:].std()
    else:
        mu = spread.mean()
        sigma = spread.std()

    z = float((spread[-1] - mu) / max(sigma, 1e-10))

    return {
        "hurst": float(H),
        "memory_parameter_d": float(d),
        "z_score": z,
        "signal_quality": float(quality),
        "mean_reverting": bool(d < 0),
        "signal": float(-np.sign(z) * quality if abs(z) > 1.5 else 0.0),
    }
