"""
Market entropy measures for regime detection and predictability scoring.

Implements:
  - Permutation entropy (Bandt-Pompe)
  - Approximate entropy (ApEn)
  - Sample entropy (SampEn)
  - Multiscale entropy (MSE)
  - Spectral entropy (frequency-domain)
  - Topological entropy
  - Market efficiency index (entropy-based)
  - Entropy rate estimation
  - Complexity-entropy causality plane
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional


# ── Spectral entropy ──────────────────────────────────────────────────────────

def spectral_entropy(x: np.ndarray, n_fft: Optional[int] = None) -> float:
    """
    Spectral entropy via power spectral density.
    High value → broadband (noise-like); low value → periodic/structured.
    """
    n = len(x)
    if n_fft is None:
        n_fft = n
    # Windowed FFT
    window = np.hanning(n)
    X = np.fft.rfft(x * window, n=n_fft)
    psd = np.abs(X) ** 2
    psd = psd / psd.sum()
    psd = psd[psd > 0]
    return float(-np.sum(psd * np.log2(psd)) / math.log2(len(X)))


# ── Approximate entropy ───────────────────────────────────────────────────────

def approximate_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Approximate Entropy ApEn(m, r).
    Lower = more regular. Higher = more complex/random.
    r = r_factor * std(x)
    """
    n = len(x)
    r = r_factor * x.std() + 1e-10

    def phi(template_len):
        count = np.zeros(n - template_len + 1)
        for i in range(n - template_len + 1):
            template = x[i: i + template_len]
            for j in range(n - template_len + 1):
                if np.max(np.abs(template - x[j: j + template_len])) < r:
                    count[i] += 1
        return np.mean(np.log(count / (n - template_len + 1)))

    return float(phi(m) - phi(m + 1))


# ── Multiscale entropy ────────────────────────────────────────────────────────

def multiscale_entropy(
    x: np.ndarray,
    max_scale: int = 10,
    m: int = 2,
    r_factor: float = 0.2,
) -> np.ndarray:
    """
    Multiscale Sample Entropy (Costa et al. 2002).
    Coarse-grains the series at multiple timescales and computes SampEn.
    Returns entropy at each scale 1..max_scale.
    """
    from .information_theory import sample_entropy

    mse = np.zeros(max_scale)
    for scale in range(1, max_scale + 1):
        # Coarse-grain: average over non-overlapping windows
        n_coarse = len(x) // scale
        if n_coarse < 10:
            mse[scale - 1] = np.nan
            continue
        coarse = x[:n_coarse * scale].reshape(n_coarse, scale).mean(axis=1)
        mse[scale - 1] = sample_entropy(coarse, m=m, r_factor=r_factor)

    return mse


def mse_complexity_index(x: np.ndarray, max_scale: int = 10) -> float:
    """
    Complexity index = area under MSE curve (normalized).
    High = complex (healthy market); low = simple (trending/crisis).
    """
    mse = multiscale_entropy(x, max_scale)
    valid = mse[~np.isnan(mse)]
    return float(valid.mean()) if len(valid) > 0 else 0.0


# ── Topological entropy (ordinal patterns) ───────────────────────────────────

def topological_entropy(x: np.ndarray, order: int = 4) -> float:
    """
    Topological entropy estimated via forbidden ordinal patterns.
    TE = log2(n_observed) / log2(n_possible)
    Low = many forbidden patterns = predictable.
    """
    from itertools import permutations as _perms
    import math

    n = len(x)
    n_possible = math.factorial(order)
    observed = set()

    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i: i + order]))
        observed.add(pattern)

    if n_possible == 0:
        return 0.0
    return float(len(observed) / n_possible)


# ── Entropy rate ──────────────────────────────────────────────────────────────

def entropy_rate_lempel_ziv(x: np.ndarray, n_bins: int = 4) -> float:
    """
    Entropy rate estimation via Lempel-Ziv complexity.
    Lower rate = more compressible = more predictable.
    """
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    symbols = np.digitize(x, bins[1:-1])
    s = ''.join(map(str, symbols))
    n = len(s)

    # LZ76 complexity
    c = 1
    i = 0
    l = 1
    while i + l <= n - 1:
        if s[i: i + l] not in s[:i + l - 1]:
            c += 1
            i += l
            l = 1
        else:
            l += 1

    return c * math.log2(n) / n if n > 1 else 0.0


# ── Complexity-entropy causality plane ───────────────────────────────────────

def complexity_entropy_plane(
    x: np.ndarray,
    order: int = 4,
) -> tuple[float, float]:
    """
    Compute (H, C) coordinates for causality plane.
    H = normalized permutation entropy
    C = Jensen-Shannon complexity (statistical complexity)

    Reference: Rosso et al. (2007) PRL 99, 154102
    """
    from itertools import permutations as _perms
    import math

    n = len(x)
    n_patterns = math.factorial(order)
    perm_indices = {p: i for i, p in enumerate(_perms(range(order)))}

    # Count ordinal patterns
    counts = np.zeros(n_patterns)
    for i in range(n - order + 1):
        pattern = tuple(np.argsort(x[i: i + order]))
        counts[perm_indices[pattern]] += 1

    p = counts / counts.sum()
    p_nonzero = p[p > 0]

    # Normalized permutation entropy
    H_max = math.log(n_patterns)
    H = float(-np.sum(p_nonzero * np.log(p_nonzero)) / H_max)

    # JS divergence with uniform
    p_uniform = np.ones(n_patterns) / n_patterns
    m = (p + p_uniform) / 2
    js_div = (
        -np.sum(m[m > 0] * np.log(m[m > 0]))
        + 0.5 * np.sum(p_nonzero * np.log(p_nonzero))
        + 0.5 * math.log(n_patterns)
    )

    # Normalization constant Q_0
    q0 = -0.5 * ((n_patterns + 1) / n_patterns * math.log(n_patterns + 1)
                  - 2 * math.log(2 * n_patterns)
                  + math.log(n_patterns))
    C = float(js_div / max(q0, 1e-10) * H) if q0 > 0 else 0.0

    return H, C


# ── Rolling entropy indicators ────────────────────────────────────────────────

def rolling_permutation_entropy(
    x: np.ndarray,
    window: int = 60,
    order: int = 3,
) -> np.ndarray:
    """Rolling permutation entropy. Returns array of length len(x)."""
    from .information_theory import permutation_entropy
    result = np.full(len(x), np.nan)
    for i in range(window, len(x) + 1):
        result[i - 1] = permutation_entropy(x[i - window: i], order=order)
    return result


def entropy_regime_signal(
    returns: np.ndarray,
    window: int = 60,
    order: int = 4,
    low_thresh: float = 0.4,
    high_thresh: float = 0.8,
) -> np.ndarray:
    """
    Generate regime signal from rolling permutation entropy.
    Returns: +1 = low entropy (predictable, trending)
              0 = neutral
             -1 = high entropy (chaotic, avoid)
    """
    pe = rolling_permutation_entropy(returns, window, order)
    signal = np.zeros(len(returns))
    signal[pe < low_thresh] = 1.0   # low entropy → trend-follow
    signal[pe > high_thresh] = -1.0  # high entropy → avoid
    return signal


# ── Market efficiency index ────────────────────────────────────────────────────

def market_efficiency_score(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """
    Composite market efficiency score [0, 1] combining:
    - Permutation entropy (high = efficient)
    - Spectral entropy (high = efficient)
    - LZ complexity (high = efficient)

    Returns rolling score. Near 1 = efficient (hard to predict).
    Near 0 = inefficient (potentially exploitable).
    """
    n = len(returns)
    scores = np.full(n, np.nan)

    for i in range(window, n):
        w = returns[i - window: i]
        pe = sum([].__class__.__mro__)  # placeholder
        try:
            from .information_theory import permutation_entropy
            pe = permutation_entropy(w, order=3, normalize=True)
            se = spectral_entropy(w)
            lz = entropy_rate_lempel_ziv(w)
            scores[i] = (pe + se + lz) / 3
        except Exception:
            scores[i] = np.nan

    return scores
