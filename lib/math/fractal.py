"""
Fractal and long-memory analysis for financial time series.

Implements:
  - Hurst exponent (R/S analysis, DFA, Higuchi, Whittle)
  - Detrended Fluctuation Analysis (DFA) with polynomial detrending
  - Multifractal DFA (MF-DFA): generalized Hurst spectrum
  - R/S analysis (Mandelbrot & Wallis)
  - Higuchi fractal dimension
  - Fractional Brownian motion simulation
  - Long-memory ARFIMA detection
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional


# ── R/S analysis ──────────────────────────────────────────────────────────────

def rs_analysis(x: np.ndarray, min_n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    R/S analysis: compute R/S statistic at multiple scales.
    Returns (log2(n), log2(R/S)) pairs.
    """
    n = len(x)
    scales = []
    rs_values = []

    # Use scales that are powers of 2
    scale = min_n
    while scale <= n // 2:
        sub_rs = []
        for start in range(0, n - scale + 1, scale):
            sub = x[start: start + scale]
            mean_sub = sub.mean()
            deviation = np.cumsum(sub - mean_sub)
            R = deviation.max() - deviation.min()
            S = sub.std(ddof=1)
            if S > 0:
                sub_rs.append(R / S)
        if sub_rs:
            scales.append(math.log2(scale))
            rs_values.append(math.log2(np.mean(sub_rs)))
        scale *= 2

    return np.array(scales), np.array(rs_values)


def hurst_rs(x: np.ndarray) -> float:
    """Hurst exponent via R/S analysis (OLS on log-log plot)."""
    log_n, log_rs = rs_analysis(x)
    if len(log_n) < 3:
        return 0.5
    H, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(H, 0.01, 0.99))


# ── Detrended Fluctuation Analysis ────────────────────────────────────────────

def dfa(
    x: np.ndarray,
    scales: Optional[np.ndarray] = None,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Detrended Fluctuation Analysis (DFA).

    Parameters:
      x      : input time series
      scales : array of window sizes (defaults to log-spaced 10..N/4)
      order  : polynomial detrending order (1=linear, 2=quadratic)

    Returns:
      (scales, F(n), H) where H is the DFA scaling exponent
    """
    n = len(x)
    if scales is None:
        scales = np.unique(np.round(
            np.exp(np.linspace(math.log(10), math.log(n // 4), 20))
        ).astype(int))
        scales = scales[scales >= 4]

    # Cumulative sum (profile)
    profile = np.cumsum(x - x.mean())

    F_values = []
    valid_scales = []
    for s in scales:
        s = int(s)
        n_segments = n // s
        if n_segments < 2:
            continue
        fluctuations = []
        for seg in range(n_segments):
            seg_data = profile[seg * s: (seg + 1) * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, seg_data, order)
            trend = np.polyval(coeffs, t)
            fluctuations.append(np.mean((seg_data - trend) ** 2))
        F_values.append(math.sqrt(np.mean(fluctuations)))
        valid_scales.append(s)

    scales_arr = np.array(valid_scales, dtype=float)
    F_arr = np.array(F_values)

    if len(scales_arr) < 3:
        return scales_arr, F_arr, 0.5

    H, _ = np.polyfit(np.log(scales_arr), np.log(F_arr), 1)
    return scales_arr, F_arr, float(np.clip(H, 0.01, 0.99))


def hurst_dfa(x: np.ndarray, order: int = 1) -> float:
    """Hurst exponent via DFA."""
    _, _, H = dfa(x, order=order)
    return H


# ── Multifractal DFA ──────────────────────────────────────────────────────────

def mfdfa(
    x: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    order: int = 1,
) -> dict:
    """
    Multifractal DFA (MF-DFA).

    Returns:
      h(q)  : generalized Hurst exponent at each q
      tau(q): mass exponent
      f(a)  : multifractal spectrum
      alpha : Hölder exponent
      width : spectrum width (f_max - f_min) — measure of multifractality
    """
    n = len(x)
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    if scales is None:
        scales = np.unique(np.round(
            np.exp(np.linspace(math.log(10), math.log(n // 4), 15))
        ).astype(int))
        scales = scales[scales >= 4]

    profile = np.cumsum(x - x.mean())
    Fq = np.zeros((len(q_values), len(scales)))

    for si, s in enumerate(scales):
        s = int(s)
        n_segments = n // s
        if n_segments < 2:
            Fq[:, si] = np.nan
            continue

        f2 = []
        for seg in range(n_segments):
            seg_data = profile[seg * s: (seg + 1) * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, seg_data, order)
            trend = np.polyval(coeffs, t)
            f2.append(np.mean((seg_data - trend) ** 2))
        f2 = np.array(f2)

        for qi, q in enumerate(q_values):
            if abs(q) < 1e-6:
                Fq[qi, si] = math.exp(0.5 * np.mean(np.log(f2 + 1e-12)))
            else:
                Fq[qi, si] = (np.mean(f2 ** (q / 2))) ** (1.0 / q)

    # Estimate h(q) from log-log slope
    valid = ~np.any(np.isnan(Fq), axis=0)
    log_s = np.log(scales[valid].astype(float))
    hq = np.zeros(len(q_values))
    for qi in range(len(q_values)):
        if valid.sum() >= 3:
            h, _ = np.polyfit(log_s, np.log(Fq[qi, valid] + 1e-12), 1)
            hq[qi] = h

    # Multifractal spectrum
    tau = q_values * hq - 1
    alpha = np.gradient(tau, q_values)
    f_alpha = q_values * alpha - tau

    return {
        "q": q_values,
        "h_q": hq,
        "tau_q": tau,
        "alpha": alpha,
        "f_alpha": f_alpha,
        "hurst": float(hq[np.argmin(np.abs(q_values - 2.0))]),  # H at q=2
        "width": float(alpha.max() - alpha.min()),  # multifractal width
        "multifractal": bool((alpha.max() - alpha.min()) > 0.1),
    }


# ── Higuchi fractal dimension ──────────────────────────────────────────────────

def higuchi_fd(x: np.ndarray, k_max: int = 10) -> float:
    """
    Higuchi's fractal dimension.
    FD = 1 + Hurst (for fBm), but directly estimated from length curves.
    """
    n = len(x)
    L_k = []
    k_vals = range(1, k_max + 1)

    for k in k_vals:
        Lm = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            sub = x[indices]
            num_intervals = (n - m) // k
            if num_intervals == 0:
                continue
            Lm.append(
                np.sum(np.abs(np.diff(sub))) * (n - 1) / (num_intervals * k)
            )
        if Lm:
            L_k.append(np.mean(Lm))
        else:
            L_k.append(np.nan)

    L_k = np.array(L_k)
    valid = ~np.isnan(L_k)
    if valid.sum() < 3:
        return 1.5

    slope, _ = np.polyfit(np.log(np.array(list(k_vals))[valid]), np.log(L_k[valid]), 1)
    return float(np.clip(-slope, 1.0, 2.0))


# ── Fractional Brownian motion ─────────────────────────────────────────────────

def fbm_cholesky(
    H: float,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate fractional Brownian motion via Cholesky.
    H = 0.5 → standard BM
    H > 0.5 → persistent (trending)
    H < 0.5 → anti-persistent (mean-reverting)
    """
    rng = rng or np.random.default_rng()
    # Covariance: E[B(s)B(t)] = 0.5*(|s|^2H + |t|^2H - |t-s|^2H)
    t = np.arange(1, n + 1, dtype=float)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = 0.5 * (
                (i + 1) ** (2 * H) + (j + 1) ** (2 * H) - abs(i - j) ** (2 * H)
            )
    try:
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(n))
    Z = rng.standard_normal(n)
    return L @ Z


def fbm_hosking(
    H: float,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate fBm increments (fGn) via Hosking's exact method.
    Returns n increments of fGn with Hurst index H.
    """
    rng = rng or np.random.default_rng()

    def cov_fgn(k: int) -> float:
        return 0.5 * (abs(k - 1) ** (2 * H) - 2 * abs(k) ** (2 * H) + abs(k + 1) ** (2 * H))

    x = np.zeros(n)
    x[0] = rng.standard_normal()
    d = np.zeros(n)
    d[0] = cov_fgn(0)

    for i in range(1, n):
        # Levinson recursion
        gamma = np.array([cov_fgn(k) for k in range(1, i + 1)])
        phi = np.zeros(i)
        if i == 1:
            phi[0] = gamma[0] / d[0]
        else:
            phi_prev = d[:i - 1].copy()
            phi[i - 1] = (gamma[i - 1] - np.dot(phi[:i - 1], gamma[i - 2::-1])) / d[i - 1]
            phi[:i - 1] = phi_prev[:i - 1] - phi[i - 1] * phi_prev[i - 2::-1]

        d[i] = d[i - 1] * (1 - phi[i - 1] ** 2)
        v_hat = np.dot(phi, x[i - 1::-1])
        x[i] = v_hat + math.sqrt(max(d[i], 1e-10)) * rng.standard_normal()

    return x


# ── Summary ────────────────────────────────────────────────────────────────────

def fractal_summary(x: np.ndarray) -> dict:
    """Full fractal analysis summary for a time series."""
    H_rs = hurst_rs(x)
    H_dfa = hurst_dfa(x)
    fd = higuchi_fd(x)
    mf = mfdfa(x)

    return {
        "hurst_rs": H_rs,
        "hurst_dfa": H_dfa,
        "hurst_average": (H_rs + H_dfa) / 2,
        "higuchi_fd": fd,
        "regime": (
            "trending" if H_dfa > 0.55
            else "mean_reverting" if H_dfa < 0.45
            else "random_walk"
        ),
        "multifractal_width": mf["width"],
        "is_multifractal": mf["multifractal"],
        "persistence": float(2 * H_dfa - 1),  # +1 = strong trend, -1 = strong MR
    }
