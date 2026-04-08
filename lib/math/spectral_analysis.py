"""
spectral_analysis.py — Spectral and frequency-domain analysis for financial time series.

Covers: FFT periodogram, Welch PSD, multi-taper DPSS, Hilbert transform,
Empirical Mode Decomposition, Hilbert-Huang Transform, wavelet coherence,
cross-spectral density, dominant cycle detection, spectral entropy, and
band-pass filtering.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import CubicSpline
from scipy.signal import hilbert, butter, filtfilt
from scipy.signal.windows import dpss


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PeriodogramResult:
    frequencies: np.ndarray      # Hz (cycles per sample if fs=1)
    power: np.ndarray            # Power spectral density
    dominant_freq: float
    dominant_period: float       # 1 / dominant_freq (samples)


@dataclass
class WelchResult:
    frequencies: np.ndarray
    psd: np.ndarray
    dominant_freq: float
    dominant_period: float


@dataclass
class MultiTaperResult:
    frequencies: np.ndarray
    psd: np.ndarray              # averaged across tapers
    psd_per_taper: np.ndarray    # shape (n_tapers, n_freq)
    eigenvalues: np.ndarray
    dominant_freq: float
    dominant_period: float


@dataclass
class HilbertResult:
    analytic_signal: np.ndarray
    amplitude_envelope: np.ndarray
    instantaneous_phase: np.ndarray
    instantaneous_frequency: np.ndarray  # cycles per sample


@dataclass
class IMF:
    """A single Intrinsic Mode Function from EMD."""
    index: int
    values: np.ndarray
    mean_period: float           # dominant period in samples
    mean_frequency: float
    amplitude_envelope: np.ndarray
    instantaneous_frequency: np.ndarray


@dataclass
class EMDResult:
    imfs: List[IMF]
    residual: np.ndarray
    n_imfs: int


@dataclass
class HHTResult:
    """Hilbert-Huang Transform: time-frequency energy distribution."""
    time: np.ndarray
    frequencies: np.ndarray      # frequency axis
    energy: np.ndarray           # shape (n_freq, n_time) Hilbert spectrum
    marginal_spectrum: np.ndarray  # integrate over time → marginal
    imfs: List[IMF]


@dataclass
class WaveletCoherenceResult:
    time: np.ndarray
    periods: np.ndarray
    coherence: np.ndarray        # shape (n_periods, n_time), in [0,1]
    phase_angle: np.ndarray      # shape (n_periods, n_time)
    mean_coherence: np.ndarray   # averaged over time for each period


@dataclass
class CrossSpectralResult:
    frequencies: np.ndarray
    csd: np.ndarray              # complex cross-spectral density
    coherence: np.ndarray        # magnitude-squared coherence
    phase: np.ndarray            # phase angle in radians


@dataclass
class BandPassResult:
    filtered: np.ndarray
    band_low: float              # Hz
    band_high: float             # Hz
    energy_fraction: float       # fraction of total power in this band


# ---------------------------------------------------------------------------
# 1. FFT Periodogram
# ---------------------------------------------------------------------------

def periodogram(x: np.ndarray, fs: float = 1.0, detrend: bool = True) -> PeriodogramResult:
    """
    Compute FFT-based periodogram of a 1-D signal.

    Parameters
    ----------
    x   : input time series
    fs  : sampling frequency (default 1 → cycles per sample)
    detrend : remove linear trend before computing
    """
    x = np.asarray(x, dtype=float)
    if detrend:
        x = sp_signal.detrend(x, type='linear')
    n = len(x)
    window = np.hanning(n)
    windowed = x * window
    # Normalise for power
    scale = 2.0 / (fs * np.sum(window ** 2))
    X = fft(windowed)
    freqs = fftfreq(n, d=1.0 / fs)
    # One-sided
    half = n // 2
    freqs = freqs[:half]
    power = scale * np.abs(X[:half]) ** 2
    power[0] /= 2  # DC
    # Dominant
    idx = int(np.argmax(power[1:])) + 1  # skip DC
    dom_freq = float(freqs[idx])
    dom_period = 1.0 / dom_freq if dom_freq > 0 else float('inf')
    return PeriodogramResult(frequencies=freqs, power=power,
                             dominant_freq=dom_freq, dominant_period=dom_period)


# ---------------------------------------------------------------------------
# 2. Welch's Method
# ---------------------------------------------------------------------------

def welch_psd(x: np.ndarray, fs: float = 1.0, nperseg: int = 64,
              noverlap: Optional[int] = None, window: str = 'hann') -> WelchResult:
    """
    Welch's overlapping-window power spectral density estimate.
    """
    x = np.asarray(x, dtype=float)
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, psd = sp_signal.welch(x, fs=fs, window=window,
                                  nperseg=min(nperseg, len(x)),
                                  noverlap=noverlap, detrend='linear',
                                  scaling='density')
    idx = int(np.argmax(psd[1:])) + 1
    dom_freq = float(freqs[idx])
    dom_period = 1.0 / dom_freq if dom_freq > 0 else float('inf')
    return WelchResult(frequencies=freqs, psd=psd,
                       dominant_freq=dom_freq, dominant_period=dom_period)


# ---------------------------------------------------------------------------
# 3. Multi-taper Spectral Estimation (DPSS)
# ---------------------------------------------------------------------------

def multitaper_psd(x: np.ndarray, fs: float = 1.0,
                   nw: float = 4.0, n_tapers: Optional[int] = None) -> MultiTaperResult:
    """
    Multi-taper spectral estimate using Discrete Prolate Spheroidal Sequences.

    Parameters
    ----------
    nw       : time-half-bandwidth product (controls spectral concentration)
    n_tapers : number of tapers to use; defaults to 2*nw - 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n_tapers is None:
        n_tapers = int(2 * nw) - 1
    # DPSS tapers and their concentration eigenvalues
    tapers, eigenvalues = dpss(n, nw, Kmax=n_tapers, return_ratios=True)
    # Each row of tapers is one taper, shape (n_tapers, n)
    psd_all = np.zeros((n_tapers, n // 2))
    freqs = fftfreq(n, d=1.0 / fs)[:n // 2]
    scale = 2.0 / fs
    for k in range(n_tapers):
        windowed = x * tapers[k]
        X = fft(windowed, n=n)
        psd_all[k] = scale * np.abs(X[:n // 2]) ** 2
    # Adaptive weighting by eigenvalue concentration
    weights = eigenvalues / eigenvalues.sum()
    psd_mean = np.average(psd_all, axis=0, weights=weights)
    idx = int(np.argmax(psd_mean[1:])) + 1
    dom_freq = float(freqs[idx])
    dom_period = 1.0 / dom_freq if dom_freq > 0 else float('inf')
    return MultiTaperResult(frequencies=freqs, psd=psd_mean,
                            psd_per_taper=psd_all, eigenvalues=eigenvalues,
                            dominant_freq=dom_freq, dominant_period=dom_period)


# ---------------------------------------------------------------------------
# 4. Hilbert Transform — Instantaneous Frequency
# ---------------------------------------------------------------------------

def hilbert_analysis(x: np.ndarray, fs: float = 1.0) -> HilbertResult:
    """
    Compute analytic signal via Hilbert transform.
    Returns amplitude envelope, instantaneous phase, instantaneous frequency.
    """
    x = np.asarray(x, dtype=float)
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    # Instantaneous frequency = derivative of phase / (2π) × fs
    inst_freq = np.diff(phase) / (2.0 * math.pi) * fs
    inst_freq = np.append(inst_freq, inst_freq[-1])  # pad to same length
    return HilbertResult(analytic_signal=analytic, amplitude_envelope=envelope,
                         instantaneous_phase=phase, instantaneous_frequency=inst_freq)


# ---------------------------------------------------------------------------
# 5. Empirical Mode Decomposition (EMD)
# ---------------------------------------------------------------------------

def _find_extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of local maxima and minima."""
    maxima = (np.diff(np.sign(np.diff(x))) < 0).nonzero()[0] + 1
    minima = (np.diff(np.sign(np.diff(x))) > 0).nonzero()[0] + 1
    return maxima, minima


def _envelope_spline(x: np.ndarray, indices: np.ndarray,
                     n: int) -> Optional[np.ndarray]:
    """Fit cubic spline through extrema; return evaluated envelope or None."""
    if len(indices) < 4:
        return None
    # Include endpoints to avoid boundary artefacts
    idx = np.concatenate([[0], indices, [n - 1]])
    vals = x[idx]
    # Deduplicate
    _, ui = np.unique(idx, return_index=True)
    idx = idx[ui]
    vals = vals[ui]
    cs = CubicSpline(idx, vals, bc_type='not-a-knot')
    return cs(np.arange(n))


def emd(x: np.ndarray, max_imfs: int = 10, max_sifting: int = 20,
        sd_threshold: float = 0.2, fs: float = 1.0) -> EMDResult:
    """
    Empirical Mode Decomposition via the sifting algorithm.

    Decomposes x into Intrinsic Mode Functions (IMFs) + residual.
    Each IMF satisfies: (1) equal zero-crossings and extrema ±1,
    (2) mean of upper+lower envelope ≈ 0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    residual = x.copy()
    imfs: List[IMF] = []

    for imf_idx in range(max_imfs):
        if np.std(residual) < 1e-10:
            break
        h = residual.copy()
        prev_h = h.copy()
        for _ in range(max_sifting):
            maxima, minima = _find_extrema(h)
            upper = _envelope_spline(h, maxima, n)
            lower = _envelope_spline(h, minima, n)
            if upper is None or lower is None:
                break
            mean_env = (upper + lower) / 2.0
            h_new = h - mean_env
            # Cauchy-type stopping criterion
            sd = np.sum((h_new - prev_h) ** 2) / (np.sum(prev_h ** 2) + 1e-12)
            prev_h = h.copy()
            h = h_new
            if sd < sd_threshold:
                break
        # Check that we got a valid IMF (more than 4 extrema)
        maxima, minima = _find_extrema(h)
        if len(maxima) < 2:
            break
        residual = residual - h
        # Hilbert-based instantaneous frequency for this IMF
        analytic = hilbert(h)
        envelope = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2.0 * math.pi) * fs
        inst_freq = np.append(inst_freq, inst_freq[-1])
        mean_freq = float(np.mean(np.abs(inst_freq)))
        mean_period = 1.0 / mean_freq if mean_freq > 0 else float('inf')
        imf_obj = IMF(index=imf_idx, values=h, mean_period=mean_period,
                      mean_frequency=mean_freq, amplitude_envelope=envelope,
                      instantaneous_frequency=inst_freq)
        imfs.append(imf_obj)

    return EMDResult(imfs=imfs, residual=residual, n_imfs=len(imfs))


# ---------------------------------------------------------------------------
# 6. Hilbert-Huang Transform
# ---------------------------------------------------------------------------

def hilbert_huang_transform(x: np.ndarray, fs: float = 1.0,
                             n_freq_bins: int = 100,
                             max_imfs: int = 10) -> HHTResult:
    """
    Compute the Hilbert-Huang Transform of a nonstationary signal.

    Returns a 2-D time-frequency energy representation (Hilbert spectrum).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    emd_result = emd(x, max_imfs=max_imfs, fs=fs)
    time = np.arange(n) / fs
    # Frequency axis from 0 to Nyquist
    freq_axis = np.linspace(0, fs / 2.0, n_freq_bins)
    energy = np.zeros((n_freq_bins, n))

    for imf in emd_result.imfs:
        inst_f = np.abs(imf.instantaneous_frequency)
        amp = imf.amplitude_envelope
        for t in range(n):
            f = inst_f[t]
            a = amp[t]
            # Find nearest frequency bin
            bin_idx = int(np.searchsorted(freq_axis, f))
            bin_idx = min(bin_idx, n_freq_bins - 1)
            energy[bin_idx, t] += a ** 2

    # Marginal spectrum: integrate energy over time
    marginal = np.sum(energy, axis=1) / n
    return HHTResult(time=time, frequencies=freq_axis, energy=energy,
                     marginal_spectrum=marginal, imfs=emd_result.imfs)


# ---------------------------------------------------------------------------
# 7. Wavelet Coherence (Morlet)
# ---------------------------------------------------------------------------

def _morlet_wavelet(n: int, scale: float, dt: float = 1.0,
                    omega0: float = 6.0) -> np.ndarray:
    """
    Return complex Morlet wavelet in frequency domain for convolution.
    """
    omega = 2 * math.pi * fftfreq(n, d=dt)
    # Heaviside: only positive frequencies
    heaviside = np.where(omega > 0, 1.0, 0.0)
    norm = math.pow(math.pi, -0.25) * math.sqrt(2 * math.pi * scale / dt)
    psi_hat = norm * heaviside * np.exp(-0.5 * (scale * omega - omega0) ** 2)
    return psi_hat


def wavelet_coherence(x: np.ndarray, y: np.ndarray,
                      dt: float = 1.0,
                      periods: Optional[np.ndarray] = None,
                      smooth_window: int = 5) -> WaveletCoherenceResult:
    """
    Compute wavelet coherence between two time series x and y using
    the Morlet wavelet.

    Parameters
    ----------
    x, y           : input time series (same length)
    dt             : sampling interval
    periods        : array of periods to analyse (in samples); auto if None
    smooth_window  : smoothing length for coherence calculation
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y), "x and y must have the same length"
    n = len(x)
    if periods is None:
        n_periods = 20
        periods = np.logspace(np.log10(2 * dt), np.log10(n * dt / 4), n_periods)

    Wx = np.zeros((len(periods), n), dtype=complex)
    Wy = np.zeros((len(periods), n), dtype=complex)

    Xf = fft(x)
    Yf = fft(y)

    for i, period in enumerate(periods):
        scale = period / (2 * math.pi / 6.0 * dt)  # Morlet scale from period
        psi_hat = _morlet_wavelet(n, scale=scale, dt=dt)
        Wx[i] = ifft(psi_hat * Xf)
        Wy[i] = ifft(psi_hat * Yf)

    # Smooth power and cross-spectrum
    def _smooth(arr: np.ndarray, w: int) -> np.ndarray:
        kernel = np.ones(w) / w
        out = np.zeros_like(arr, dtype=complex)
        for i in range(arr.shape[0]):
            re = np.convolve(arr[i].real, kernel, mode='same')
            im = np.convolve(arr[i].imag, kernel, mode='same')
            out[i] = re + 1j * im
        return out

    Sxy = _smooth(Wx * np.conj(Wy), smooth_window)
    Sxx = np.abs(_smooth(Wx * np.conj(Wx), smooth_window))
    Syy = np.abs(_smooth(Wy * np.conj(Wy), smooth_window))

    denom = np.sqrt(Sxx * Syy)
    coherence = np.abs(Sxy) / (denom + 1e-12)
    coherence = np.clip(coherence.real, 0.0, 1.0)
    phase_angle = np.angle(Sxy)
    mean_coh = np.mean(coherence, axis=1)
    time = np.arange(n) * dt
    return WaveletCoherenceResult(time=time, periods=periods, coherence=coherence,
                                   phase_angle=phase_angle, mean_coherence=mean_coh)


# ---------------------------------------------------------------------------
# 8. Cross-Spectral Density and Coherence
# ---------------------------------------------------------------------------

def cross_spectral_density(x: np.ndarray, y: np.ndarray,
                            fs: float = 1.0,
                            nperseg: int = 64) -> CrossSpectralResult:
    """
    Compute cross-spectral density and magnitude-squared coherence
    between two time series using Welch's method.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nperseg = min(nperseg, len(x))
    freqs, Pxy = sp_signal.csd(x, y, fs=fs, nperseg=nperseg,
                                detrend='linear', scaling='density')
    _, Pxx = sp_signal.welch(x, fs=fs, nperseg=nperseg,
                              detrend='linear', scaling='density')
    _, Pyy = sp_signal.welch(y, fs=fs, nperseg=nperseg,
                              detrend='linear', scaling='density')
    coherence = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-15)
    phase = np.angle(Pxy)
    return CrossSpectralResult(frequencies=freqs, csd=Pxy,
                               coherence=coherence, phase=phase)


# ---------------------------------------------------------------------------
# 9. Dominant Cycle Detection
# ---------------------------------------------------------------------------

def dominant_cycle(x: np.ndarray, fs: float = 1.0,
                   method: str = 'welch',
                   min_period: float = 4.0,
                   max_period: float = 200.0) -> Tuple[float, float]:
    """
    Detect the dominant market cycle period in a detrended price series.

    Parameters
    ----------
    method     : 'welch', 'periodogram', or 'multitaper'
    min_period : minimum cycle length in bars/samples
    max_period : maximum cycle length in bars/samples

    Returns
    -------
    (dominant_period, dominant_frequency)
    """
    x = np.asarray(x, dtype=float)
    # Detrend
    x = sp_signal.detrend(x, type='linear')

    if method == 'welch':
        res = welch_psd(x, fs=fs, nperseg=min(64, len(x) // 2))
        freqs, psd = res.frequencies, res.psd
    elif method == 'periodogram':
        res = periodogram(x, fs=fs)
        freqs, psd = res.frequencies, res.power
    elif method == 'multitaper':
        res = multitaper_psd(x, fs=fs)
        freqs, psd = res.frequencies, res.psd
    else:
        raise ValueError(f"Unknown method: {method}")

    # Filter to valid period range
    min_freq = fs / max_period
    max_freq = fs / min_period
    mask = (freqs >= min_freq) & (freqs <= max_freq) & (freqs > 0)
    if not np.any(mask):
        return float('inf'), 0.0
    freqs_m = freqs[mask]
    psd_m = psd[mask]
    idx = int(np.argmax(psd_m))
    dom_freq = float(freqs_m[idx])
    dom_period = 1.0 / dom_freq if dom_freq > 0 else float('inf')
    return dom_period, dom_freq


# ---------------------------------------------------------------------------
# 10. Spectral Entropy
# ---------------------------------------------------------------------------

def spectral_entropy(x: np.ndarray, fs: float = 1.0,
                     method: str = 'welch',
                     normalize: bool = True) -> float:
    """
    Compute spectral entropy of a time series.

    Spectral entropy = -Σ p(f) log p(f) where p(f) is the normalised PSD.
    High entropy → broadband / noise-like signal.
    Low entropy  → concentrated / periodic signal.

    Returns value in nats (use normalize=True to return in [0, 1]).
    """
    x = np.asarray(x, dtype=float)
    if method == 'welch':
        res = welch_psd(x, fs=fs, nperseg=min(64, len(x) // 2))
        psd = res.psd
    else:
        res = periodogram(x, fs=fs)
        psd = res.power

    psd = psd + 1e-15
    p = psd / psd.sum()
    entropy = float(-np.sum(p * np.log(p)))
    if normalize:
        max_entropy = math.log(len(p))
        entropy /= max_entropy if max_entropy > 0 else 1.0
    return entropy


# ---------------------------------------------------------------------------
# 11. Band-Pass Filtered Returns
# ---------------------------------------------------------------------------

def bandpass_filter(x: np.ndarray, low_period: float, high_period: float,
                    fs: float = 1.0, order: int = 4) -> BandPassResult:
    """
    Apply a Butterworth band-pass filter to extract returns in a specific
    frequency band defined by period range [low_period, high_period].

    Parameters
    ----------
    low_period  : shorter period bound (higher frequency), in samples
    high_period : longer period bound (lower frequency), in samples
    fs          : sampling frequency
    order       : Butterworth filter order

    Returns
    -------
    BandPassResult with filtered signal and energy fraction
    """
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    low_freq = fs / high_period   # long period → low frequency
    high_freq = fs / low_period   # short period → high frequency
    # Clamp frequencies to valid range
    low_freq = max(low_freq, 1e-4)
    high_freq = min(high_freq, nyq * 0.99)
    if low_freq >= high_freq:
        warnings.warn("Band-pass: low_freq >= high_freq after clamping; returning zeros.")
        return BandPassResult(filtered=np.zeros_like(x),
                              band_low=low_freq, band_high=high_freq,
                              energy_fraction=0.0)
    sos = butter(order, [low_freq / nyq, high_freq / nyq],
                 btype='bandpass', output='sos')
    filtered = filtfilt(*sp_signal.sos2tf(sos), x)
    # Energy fraction
    total_energy = float(np.sum(x ** 2)) + 1e-15
    band_energy = float(np.sum(filtered ** 2))
    energy_fraction = band_energy / total_energy
    return BandPassResult(filtered=filtered, band_low=low_freq,
                          band_high=high_freq, energy_fraction=energy_fraction)


# ---------------------------------------------------------------------------
# 12. Convenience: full spectral summary
# ---------------------------------------------------------------------------

@dataclass
class SpectralSummary:
    dominant_period: float
    dominant_frequency: float
    spectral_entropy: float           # normalised [0, 1]
    cycle_trend_ratio: float          # power in cycle band / total power
    welch: WelchResult
    multitaper: MultiTaperResult
    hilbert: HilbertResult
    mean_instantaneous_frequency: float
    std_instantaneous_frequency: float


def spectral_summary(x: np.ndarray, fs: float = 1.0,
                     cycle_min_period: float = 5.0,
                     cycle_max_period: float = 80.0) -> SpectralSummary:
    """
    Full spectral summary: dominant cycle, entropy, Hilbert envelope statistics,
    and multi-method PSD.
    """
    x = np.asarray(x, dtype=float)
    w = welch_psd(x, fs=fs, nperseg=min(64, len(x) // 2))
    mt = multitaper_psd(x, fs=fs)
    h = hilbert_analysis(x, fs=fs)
    dom_period, dom_freq = dominant_cycle(x, fs=fs, method='welch',
                                          min_period=cycle_min_period,
                                          max_period=cycle_max_period)
    ent = spectral_entropy(x, fs=fs)
    # Band power ratio
    nyq = fs / 2.0
    low_f = fs / cycle_max_period
    high_f = min(fs / cycle_min_period, nyq * 0.99)
    mask = (w.frequencies >= low_f) & (w.frequencies <= high_f)
    cycle_power = float(np.sum(w.psd[mask]))
    total_power = float(np.sum(w.psd)) + 1e-15
    cycle_ratio = cycle_power / total_power
    mean_if = float(np.mean(np.abs(h.instantaneous_frequency)))
    std_if = float(np.std(np.abs(h.instantaneous_frequency)))
    return SpectralSummary(dominant_period=dom_period,
                           dominant_frequency=dom_freq,
                           spectral_entropy=ent,
                           cycle_trend_ratio=cycle_ratio,
                           welch=w, multitaper=mt, hilbert=h,
                           mean_instantaneous_frequency=mean_if,
                           std_instantaneous_frequency=std_if)
