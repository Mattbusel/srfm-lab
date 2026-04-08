"""
Wavelet decomposition for multi-scale financial signal analysis.

Implements:
  - Discrete Wavelet Transform (DWT) decomposition
  - Continuous Wavelet Transform (CWT) with Morlet/Mexican hat
  - Wavelet power spectrum
  - Wavelet coherence between two series
  - Multi-resolution analysis: trend/cycle decomposition
  - Wavelet variance (scale-specific variance)
  - Wavelet correlation between assets
  - Denoising via wavelet thresholding
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional


# ── Haar wavelet ──────────────────────────────────────────────────────────────

def haar_dwt(x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Multi-level Haar DWT.
    Returns (approximation_coeffs, [detail_coeffs_level1, level2, ...]).
    """
    n = len(x)
    levels = int(math.log2(n))
    details = []
    approx = x.copy().astype(float)

    for _ in range(levels):
        n_half = len(approx) // 2
        if n_half == 0:
            break
        even = approx[0::2]
        odd = approx[1::2]
        m = min(len(even), len(odd))
        new_approx = (even[:m] + odd[:m]) / math.sqrt(2)
        detail = (even[:m] - odd[:m]) / math.sqrt(2)
        details.append(detail)
        approx = new_approx

    return approx, details[::-1]  # details from fine to coarse


def haar_idwt(approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    """Inverse Haar DWT reconstruction."""
    x = approx.copy()
    for detail in reversed(details):
        m = min(len(x), len(detail))
        even = (x[:m] + detail[:m]) / math.sqrt(2)
        odd = (x[:m] - detail[:m]) / math.sqrt(2)
        x_new = np.zeros(2 * m)
        x_new[0::2] = even
        x_new[1::2] = odd
        x = x_new
    return x


# ── Morlet CWT ────────────────────────────────────────────────────────────────

def morlet_wavelet(t: np.ndarray, omega0: float = 6.0) -> np.ndarray:
    """Morlet wavelet: psi(t) = pi^{-1/4} * exp(i*omega0*t) * exp(-t^2/2)."""
    return (math.pi ** (-0.25)) * np.exp(1j * omega0 * t) * np.exp(-0.5 * t ** 2)


def mexican_hat_wavelet(t: np.ndarray) -> np.ndarray:
    """Ricker (Mexican hat) wavelet: real-valued, second derivative of Gaussian."""
    return (2 / (math.sqrt(3) * math.pi ** 0.25)) * (1 - t ** 2) * np.exp(-0.5 * t ** 2)


def cwt(
    x: np.ndarray,
    scales: np.ndarray,
    wavelet: str = "morlet",
    omega0: float = 6.0,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Continuous Wavelet Transform using convolution in frequency domain.

    Parameters:
      x      : input signal
      scales : array of scales
      wavelet: 'morlet' or 'mexican_hat'
      dt     : sampling interval

    Returns:
      W : complex array shape (len(scales), len(x))
    """
    n = len(x)
    freqs = np.fft.fftfreq(n, d=dt) * 2 * math.pi  # angular frequencies
    X = np.fft.fft(x)

    W = np.zeros((len(scales), n), dtype=complex)
    for si, s in enumerate(scales):
        if wavelet == "morlet":
            # Morlet in frequency domain
            psi_hat = (math.pi ** (-0.25)) * np.sqrt(s / dt) * np.exp(
                -0.5 * (s * freqs - omega0) ** 2
            )
            psi_hat[freqs < 0] = 0.0
        elif wavelet == "mexican_hat":
            psi_hat = (math.sqrt(2 * math.pi) * s ** 2.5
                       * freqs ** 2 * np.exp(-0.5 * (s * freqs) ** 2))
        else:
            raise ValueError(f"Unknown wavelet: {wavelet}")

        W[si] = np.fft.ifft(X * np.conj(psi_hat))

    return W


def cwt_power(W: np.ndarray) -> np.ndarray:
    """Wavelet power spectrum |W|^2."""
    return np.abs(W) ** 2


def cwt_phase(W: np.ndarray) -> np.ndarray:
    """Wavelet phase angle."""
    return np.angle(W)


# ── Wavelet coherence ─────────────────────────────────────────────────────────

def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    scales: Optional[np.ndarray] = None,
    smooth_sigma: float = 2.0,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wavelet coherence between x and y.
    R^2(scale, time) = |<W_xy>|^2 / (<|W_x|^2> * <|W_y|^2>)

    Returns:
      (coherence, phase_diff, scales)
      coherence : shape (n_scales, n_time), values in [0,1]
      phase_diff: instantaneous phase difference
    """
    n = len(x)
    if scales is None:
        scales = np.arange(1, min(n // 4, 64) + 1, dtype=float)

    Wx = cwt(x, scales, dt=dt)
    Wy = cwt(y, scales, dt=dt)

    # Cross-spectrum
    Wxy = Wx * np.conj(Wy)

    # Smoothing (Gaussian over scale and time)
    from scipy.ndimage import gaussian_filter
    Sxy = gaussian_filter(np.abs(Wxy), sigma=smooth_sigma) * np.exp(1j * np.angle(Wxy))
    Sxx = gaussian_filter(np.abs(Wx) ** 2, sigma=smooth_sigma)
    Syy = gaussian_filter(np.abs(Wy) ** 2, sigma=smooth_sigma)

    coherence = np.abs(gaussian_filter(np.real(Wxy), smooth_sigma)) ** 2 / (
        Sxx * Syy + 1e-10
    )
    phase_diff = np.angle(Wxy)

    return np.clip(coherence, 0, 1), phase_diff, scales


# ── Multi-resolution decomposition ────────────────────────────────────────────

def multiresolution_decomposition(
    x: np.ndarray,
    n_levels: int = 5,
) -> dict:
    """
    Decompose price/return series into trend + oscillatory components
    at multiple timescales using Haar DWT.

    Returns dict with:
      'trend'     : long-term trend (coarsest approximation)
      'components': list of detail signals (fine to coarse)
      'scales'    : approximate timescale of each component
    """
    # Pad to power of 2
    n = len(x)
    n_pad = 2 ** math.ceil(math.log2(n))
    x_pad = np.pad(x, (0, n_pad - n), mode="edge")

    n_levels = min(n_levels, int(math.log2(n_pad)))
    approx, details = haar_dwt(x_pad)

    # Reconstruct each level individually
    components = []
    for level, detail in enumerate(details[:n_levels]):
        # Reconstruct signal from this detail only
        d_full = [np.zeros_like(d) for d in details]
        d_full[level] = detail
        reconstructed = haar_idwt(np.zeros_like(approx), d_full)[:n]
        components.append(reconstructed)

    # Trend = reconstruction from approximation only
    trend = haar_idwt(approx, [np.zeros_like(d) for d in details])[:n]

    scales = [2 ** (i + 1) for i in range(n_levels)]
    return {
        "trend": trend,
        "components": components,
        "scales": scales,
        "n_levels": n_levels,
    }


# ── Wavelet variance ──────────────────────────────────────────────────────────

def wavelet_variance(
    x: np.ndarray,
    scales: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> dict:
    """
    Scale-dependent variance via wavelet detail coefficients.
    Useful for identifying at which timescale volatility is concentrated.
    """
    if scales is None:
        n = len(x)
        scales = np.logspace(0, math.log10(n // 4), 20)

    W = cwt(x, scales, dt=dt)
    power = cwt_power(W)  # (n_scales, n_time)

    # Wavelet variance at each scale = time-averaged power
    wv = power.mean(axis=1)
    total = wv.sum()

    return {
        "scales": scales,
        "variance": wv,
        "fraction": wv / total if total > 0 else wv,
        "dominant_scale": float(scales[np.argmax(wv)]),
        "total_variance": float(total),
    }


# ── Wavelet denoising ────────────────────────────────────────────────────────

def wavelet_denoise(
    x: np.ndarray,
    threshold_method: str = "soft",
    threshold_factor: float = 2.0,
) -> np.ndarray:
    """
    Denoise a signal via Haar wavelet hard/soft thresholding.

    Methods:
      'hard' : zero out coefficients below threshold
      'soft' : shrink toward zero
    """
    n = len(x)
    n_pad = 2 ** math.ceil(math.log2(n))
    x_pad = np.pad(x, (0, n_pad - n), mode="edge")

    approx, details = haar_dwt(x_pad)

    denoised_details = []
    for detail in details:
        sigma = np.median(np.abs(detail)) / 0.6745  # MAD estimator
        thresh = threshold_factor * sigma * math.sqrt(2 * math.log(len(detail) + 1))

        if threshold_method == "hard":
            thresholded = np.where(np.abs(detail) < thresh, 0.0, detail)
        elif threshold_method == "soft":
            thresholded = np.sign(detail) * np.maximum(np.abs(detail) - thresh, 0.0)
        else:
            thresholded = detail

        denoised_details.append(thresholded)

    return haar_idwt(approx, denoised_details)[:n]


# ── Asset-pair wavelet correlation ────────────────────────────────────────────

def wavelet_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    scales: Optional[np.ndarray] = None,
) -> dict:
    """
    Scale-dependent cross-correlation between assets.
    Returns correlation coefficient at each timescale.
    """
    n = len(x)
    if scales is None:
        scales = np.logspace(0, math.log10(n // 4), 10)

    Wx = cwt(x, scales)
    Wy = cwt(y, scales)

    correlations = np.zeros(len(scales))
    for si in range(len(scales)):
        px = np.abs(Wx[si]) ** 2
        py = np.abs(Wy[si]) ** 2
        pxy = np.real(Wx[si] * np.conj(Wy[si]))
        corr = pxy.mean() / math.sqrt(max(px.mean() * py.mean(), 1e-10))
        correlations[si] = float(np.clip(corr, -1, 1))

    return {
        "scales": scales,
        "correlations": correlations,
        "low_freq_corr": float(correlations[-1]),   # long-term
        "high_freq_corr": float(correlations[0]),   # short-term
        "correlation_change": float(correlations[-1] - correlations[0]),
    }
