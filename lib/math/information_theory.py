"""
Information-theoretic measures for financial time series.

Implements:
  - Shannon entropy (discrete and continuous)
  - Kullback-Leibler divergence
  - Jensen-Shannon divergence
  - Mutual information (histogram and KSG estimator)
  - Transfer entropy (T: X→Y, asymmetric)
  - Conditional mutual information
  - Lempel-Ziv complexity
  - Permutation entropy
  - Market efficiency index (via entropy rate)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional


# ── Binning utilities ──────────────────────────────────────────────────────────

def _symbolize(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Map continuous series to integer symbols via uniform histogram."""
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros(len(x), dtype=int)
    bins = np.linspace(lo, hi, n_bins + 1)
    return np.digitize(x, bins[1:-1])  # symbols 0..n_bins-1


# ── Shannon entropy ────────────────────────────────────────────────────────────

def shannon_entropy(probs: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy H(X) = -sum p * log_b(p)."""
    p = probs[probs > 0]
    return float(-np.sum(p * np.log(p) / math.log(base)))


def empirical_entropy(x: np.ndarray, n_bins: int = 10, base: float = 2.0) -> float:
    """Empirical entropy of a continuous series via histogram."""
    symbols = _symbolize(x, n_bins)
    counts = np.bincount(symbols, minlength=n_bins).astype(float)
    probs = counts / counts.sum()
    return shannon_entropy(probs, base)


def differential_entropy_gaussian(sigma: float) -> float:
    """Differential entropy of Gaussian: 0.5 * log(2*pi*e*sigma^2)."""
    return 0.5 * math.log(2 * math.pi * math.e * sigma ** 2)


# ── KL and JS divergence ───────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P||Q) = sum p * log(p/q).  Not symmetric."""
    p = np.asarray(p, float)
    q = np.asarray(q, float) + eps
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (symmetric, bounded [0,1] in bits)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ── Mutual information ─────────────────────────────────────────────────────────

def mutual_information_histogram(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    base: float = 2.0,
) -> float:
    """
    I(X;Y) via joint histogram.
    MI = H(X) + H(Y) - H(X,Y)
    """
    sx = _symbolize(x, n_bins)
    sy = _symbolize(y, n_bins)

    joint = np.zeros((n_bins, n_bins))
    for a, b in zip(sx, sy):
        joint[a, b] += 1
    joint /= joint.sum()

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    hx = shannon_entropy(px, base)
    hy = shannon_entropy(py, base)
    hxy = shannon_entropy(joint.ravel(), base)
    return max(0.0, hx + hy - hxy)


def ksg_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> float:
    """
    Kraskov-Stögbauer-Grassberger (KSG) k-NN mutual information estimator.
    Equation 5 from Kraskov et al. (2004).
    """
    from scipy.special import digamma
    n = len(x)
    xy = np.column_stack([x, y])

    # Find k-th neighbor distance in joint space (Chebyshev)
    dists_joint = np.zeros(n)
    for i in range(n):
        d = np.maximum(np.abs(xy - xy[i]).max(axis=1))
        d[i] = np.inf
        dists_joint[i] = np.sort(d)[k - 1]

    # Count marginal neighbors within that distance
    nx = np.array([np.sum(np.abs(x - x[i]) < dists_joint[i]) - 1 for i in range(n)])
    ny = np.array([np.sum(np.abs(y - y[i]) < dists_joint[i]) - 1 for i in range(n)])

    mi = (digamma(k) + digamma(n)
          - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    return max(0.0, float(mi))


# ── Transfer entropy ───────────────────────────────────────────────────────────

def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    n_bins: int = 8,
    base: float = 2.0,
) -> float:
    """
    Transfer entropy T(source → target) at given lag.

    TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
       = I(Y_t; X_{t-lag} | Y_{t-1})

    Positive value means source Granger-causes target in information-theoretic sense.
    """
    n = len(target)
    # Align: Y_t, Y_{t-1}, X_{t-lag}
    yt  = _symbolize(target[max(lag, 1):], n_bins)
    yt1 = _symbolize(target[max(lag, 1) - 1: n - lag + max(lag, 1) - 1], n_bins)
    xlag = _symbolize(source[: len(yt)], n_bins)
    m = min(len(yt), len(yt1), len(xlag))
    yt, yt1, xlag = yt[:m], yt1[:m], xlag[:m]

    # H(Y_t | Y_{t-1}) via joint
    joint_yy = np.zeros((n_bins, n_bins))
    for a, b in zip(yt, yt1):
        joint_yy[a, b] += 1
    joint_yy /= joint_yy.sum()
    py1 = joint_yy.sum(axis=0)
    h_yt_given_yt1 = -np.sum(
        joint_yy[joint_yy > 0] * np.log(joint_yy[joint_yy > 0] / np.maximum(py1[None, :], 1e-12)[0, np.newaxis].T.ravel()[:joint_yy.size].reshape(n_bins, n_bins))[joint_yy > 0]
    ) / math.log(base)

    # H(Y_t | Y_{t-1}, X_{t-lag}) via 3-way joint
    joint3 = np.zeros((n_bins, n_bins, n_bins))
    for a, b, c in zip(yt, yt1, xlag):
        joint3[a, b, c] += 1
    joint3 /= joint3.sum()
    py1x = joint3.sum(axis=0)  # P(Y_{t-1}, X)

    h_yt_given_yt1_x = 0.0
    for a in range(n_bins):
        for b in range(n_bins):
            for c in range(n_bins):
                p3 = joint3[a, b, c]
                p2 = py1x[b, c]
                if p3 > 0 and p2 > 0:
                    h_yt_given_yt1_x -= p3 * math.log(p3 / p2) / math.log(base)

    return max(0.0, h_yt_given_yt1 - h_yt_given_yt1_x)


def transfer_entropy_matrix(
    returns: np.ndarray,
    n_bins: int = 8,
    lag: int = 1,
) -> np.ndarray:
    """
    Compute full N×N transfer entropy matrix.
    returns: shape (T, N) — T time steps, N assets.
    TE[i,j] = T(asset_i → asset_j)
    """
    T, N = returns.shape
    te_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                te_mat[i, j] = transfer_entropy(returns[:, i], returns[:, j], lag, n_bins)
    return te_mat


# ── Lempel-Ziv complexity ──────────────────────────────────────────────────────

def lempel_ziv_complexity(series: np.ndarray, threshold: Optional[float] = None) -> float:
    """
    Lempel-Ziv complexity (LZ76 normalized).
    Binarises around median or given threshold.
    Low LZ → regular/predictable; high LZ → random/complex.
    """
    if threshold is None:
        threshold = float(np.median(series))
    binary = (series > threshold).astype(int)
    s = ''.join(map(str, binary))
    n = len(s)

    i, c, l, q = 0, 1, 1, 1
    while i + l <= n - 1:
        if s[i:i + l] not in s[:i + l - 1]:
            c += 1
            i += l
            l = 1
        else:
            l += 1
    # Normalize
    return c * math.log2(n) / n if n > 1 else 0.0


# ── Permutation entropy ────────────────────────────────────────────────────────

def permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Permutation entropy (Bandt & Pompe 2002).
    Low value → regular/predictable price motion.
    High value (=1 when normalized) → maximally disordered.
    """
    from itertools import permutations as _perms
    import math

    n = len(x)
    factorial_order = math.factorial(order)
    run_length = n - (order - 1) * delay

    # Map each ordinal pattern to an integer index
    perm_indices = {p: i for i, p in enumerate(_perms(range(order)))}
    counts = np.zeros(factorial_order, dtype=int)

    for i in range(run_length):
        pattern = tuple(np.argsort(x[i: i + order * delay: delay]))
        counts[perm_indices[pattern]] += 1

    probs = counts / counts.sum()
    pe = shannon_entropy(probs, base=math.e)

    if normalize:
        pe /= math.log(factorial_order)
    return float(pe)


# ── Sample / approximate entropy ──────────────────────────────────────────────

def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample entropy SampEn(m, r, N).
    r = r_factor * std(x)
    Lower → more regular; higher → more complex.
    """
    x = np.asarray(x, float)
    r = r_factor * x.std()
    n = len(x)

    def _count_matches(length: int) -> int:
        count = 0
        for i in range(n - length):
            template = x[i: i + length]
            for j in range(i + 1, n - length):
                if np.max(np.abs(template - x[j: j + length])) < r:
                    count += 1
        return count

    A = _count_matches(m + 1)
    B = _count_matches(m)
    if B == 0:
        return 0.0
    return float(-math.log(A / B))


# ── Market efficiency index ────────────────────────────────────────────────────

def market_efficiency_index(
    returns: np.ndarray,
    n_bins: int = 10,
    window: int = 60,
) -> np.ndarray:
    """
    Rolling market efficiency index via entropy ratio.
    MEI_t = H_empirical(window) / H_max
    Values near 1.0 → efficient (random); near 0 → predictable.
    """
    h_max = math.log2(n_bins)
    mei = np.full(len(returns), np.nan)
    for i in range(window, len(returns)):
        w = returns[i - window: i]
        h = empirical_entropy(w, n_bins, base=2.0)
        mei[i] = h / h_max
    return mei
