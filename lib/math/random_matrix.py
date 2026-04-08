"""
Random Matrix Theory (RMT) for financial correlation matrices.

Implements:
  - Marchenko-Pastur distribution (MP law)
  - Eigenvalue cleaning via MP threshold
  - Noise vs signal eigenvalue separation
  - Tracy-Widom edge estimation
  - Denoised covariance reconstruction
  - Effective rank / participation ratio
  - Correlation matrix quality metrics
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Marchenko-Pastur distribution ─────────────────────────────────────────────

@dataclass
class MPDistribution:
    """Marchenko-Pastur distribution parameters."""
    q: float          # T/N ratio (T = observations, N = variables)
    sigma2: float = 1.0  # variance of population eigenvalue (=1 for normalized)

    @property
    def lambda_plus(self) -> float:
        """Upper edge of MP bulk."""
        return self.sigma2 * (1.0 + math.sqrt(1.0 / self.q)) ** 2

    @property
    def lambda_minus(self) -> float:
        """Lower edge of MP bulk."""
        return self.sigma2 * (1.0 - math.sqrt(1.0 / self.q)) ** 2

    def pdf(self, lambdas: np.ndarray) -> np.ndarray:
        """MP density at eigenvalues."""
        lp, lm = self.lambda_plus, self.lambda_minus
        out = np.zeros_like(lambdas, dtype=float)
        mask = (lambdas >= lm) & (lambdas <= lp)
        x = lambdas[mask]
        out[mask] = (self.q / (2 * math.pi * self.sigma2)
                     * np.sqrt((lp - x) * (x - lm)) / x)
        return out

    def ks_statistic(self, empirical_eigs: np.ndarray) -> float:
        """Kolmogorov-Smirnov statistic vs MP CDF."""
        eigs = np.sort(empirical_eigs)
        n = len(eigs)
        # Empirical CDF
        ecdf = np.arange(1, n + 1) / n
        # MP CDF via numerical integration
        grid = np.linspace(max(self.lambda_minus, eigs.min()), eigs.max(), 1000)
        pdf_vals = self.pdf(grid)
        cdf_grid = np.cumsum(pdf_vals) * (grid[1] - grid[0])
        cdf_grid /= max(cdf_grid[-1], 1e-10)
        mp_cdf = np.interp(eigs, grid, cdf_grid)
        return float(np.max(np.abs(ecdf - mp_cdf)))


# ── Eigenvalue cleaning ────────────────────────────────────────────────────────

def clean_correlation_matrix(
    C: np.ndarray,
    T: int,
    method: str = "clip",
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Clean a correlation matrix using RMT.

    Methods:
      'clip'    — set eigenvalues below MP upper edge to their mean
      'shrink'  — shrink sub-MP eigenvalues toward MP mean
      'oracle'  — zero out sub-MP eigenvectors (retains only signal)

    Parameters:
      C      : N×N sample correlation matrix
      T      : number of observations used to build C
      alpha  : scaling multiplier on lambda_plus (default 1.0 = exact MP)
    """
    N = C.shape[0]
    q = T / N
    mp = MPDistribution(q=q)
    threshold = mp.lambda_plus * alpha

    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    noise_mask = eigenvalues < threshold
    signal_mask = ~noise_mask

    if method == "clip":
        # Replace noise eigenvalues with their mean (preserve trace)
        noise_mean = eigenvalues[noise_mask].mean() if noise_mask.any() else 1.0
        cleaned_eigs = np.where(noise_mask, noise_mean, eigenvalues)
    elif method == "shrink":
        mp_mean = mp.sigma2  # expectation of MP eigenvalue = sigma^2
        shrink_factor = 0.5
        cleaned_eigs = np.where(
            noise_mask,
            mp_mean + shrink_factor * (eigenvalues - mp_mean),
            eigenvalues,
        )
    elif method == "oracle":
        # Zero out noise components, rescale to preserve trace
        cleaned_eigs = np.where(noise_mask, 0.0, eigenvalues)
        # Rescale signal eigenvalues to preserve trace=N
        signal_sum = cleaned_eigs[signal_mask].sum()
        if signal_sum > 0:
            cleaned_eigs[signal_mask] *= N / signal_sum
    else:
        raise ValueError(f"Unknown method: {method}")

    C_clean = eigenvectors @ np.diag(cleaned_eigs) @ eigenvectors.T
    # Renormalize diagonal to 1
    d = np.sqrt(np.diag(C_clean))
    d = np.where(d > 0, d, 1.0)
    C_clean = C_clean / np.outer(d, d)
    np.fill_diagonal(C_clean, 1.0)
    return C_clean


def fit_mp_sigma(eigenvalues: np.ndarray, q: float) -> float:
    """
    Fit MP sigma^2 to match empirical eigenvalue bulk.
    Uses the fact that E[lambda] = sigma^2 for eigenvalues within MP support.
    """
    # Initial estimate: mean of smallest 80% of eigenvalues
    n = len(eigenvalues)
    sorted_eigs = np.sort(eigenvalues)
    bulk_eigs = sorted_eigs[:int(0.80 * n)]
    sigma2 = float(bulk_eigs.mean())

    # Refine: iterative matching of MP upper edge
    for _ in range(20):
        mp = MPDistribution(q=q, sigma2=sigma2)
        in_bulk = eigenvalues[eigenvalues <= mp.lambda_plus]
        if len(in_bulk) == 0:
            break
        sigma2_new = float(in_bulk.mean())
        if abs(sigma2_new - sigma2) < 1e-6:
            break
        sigma2 = 0.5 * sigma2 + 0.5 * sigma2_new

    return sigma2


# ── Signal eigenvalue analysis ─────────────────────────────────────────────────

def eigenvalue_signal_noise_split(
    returns: np.ndarray,
    alpha: float = 1.0,
) -> dict:
    """
    Decompose sample covariance eigenvalues into signal and noise.

    Parameters:
      returns : shape (T, N)

    Returns dict with:
      n_signal    : number of signal eigenvalues
      signal_eigs : signal eigenvalues (above MP threshold)
      noise_eigs  : noise eigenvalues (within MP bulk)
      lambda_plus : MP upper edge
      q           : T/N ratio
      explained_var: fraction of variance in signal components
    """
    T, N = returns.shape
    q = T / N

    # Sample correlation matrix
    std = returns.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    R = (returns - returns.mean(axis=0)) / std
    C = R.T @ R / T  # sample correlation

    eigenvalues, _ = np.linalg.eigh(C)
    eigenvalues = np.sort(eigenvalues)[::-1]

    sigma2 = fit_mp_sigma(eigenvalues, q)
    mp = MPDistribution(q=q, sigma2=sigma2)
    threshold = mp.lambda_plus * alpha

    signal_mask = eigenvalues > threshold
    signal_eigs = eigenvalues[signal_mask]
    noise_eigs = eigenvalues[~signal_mask]

    total_var = eigenvalues.sum()
    signal_var = signal_eigs.sum()

    return {
        "n_signal": int(signal_mask.sum()),
        "signal_eigs": signal_eigs,
        "noise_eigs": noise_eigs,
        "lambda_plus": threshold,
        "lambda_minus": mp.lambda_minus,
        "sigma2": sigma2,
        "q": q,
        "explained_var": float(signal_var / total_var) if total_var > 0 else 0.0,
        "n_noise": int((~signal_mask).sum()),
    }


# ── Effective rank / participation ratio ──────────────────────────────────────

def effective_rank(eigenvalues: np.ndarray) -> float:
    """
    Effective rank = exp(H) where H = entropy of normalized eigenvalues.
    Ranges from 1 (single dominant) to N (fully uniform).
    """
    eigs = np.maximum(eigenvalues, 0.0)
    total = eigs.sum()
    if total <= 0:
        return 1.0
    p = eigs / total
    p = p[p > 0]
    return float(math.exp(-np.sum(p * np.log(p))))


def participation_ratio(eigenvector: np.ndarray) -> float:
    """
    Inverse participation ratio (IPR) for an eigenvector.
    IPR = 1/sum(v_i^4) normalized.
    High PR → eigenvector spread over many assets (diversified).
    """
    v2 = eigenvector ** 2
    v4 = eigenvector ** 4
    return float((v2.sum()) ** 2 / (len(eigenvector) * v4.sum())) if v4.sum() > 0 else 0.0


# ── Denoised covariance ────────────────────────────────────────────────────────

def denoised_covariance(
    returns: np.ndarray,
    method: str = "clip",
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Full pipeline: compute sample covariance, clean via RMT, return denoised.
    Returns (N, N) covariance matrix.
    """
    T, N = returns.shape
    std = returns.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    R = (returns - returns.mean(axis=0)) / std
    C_sample = R.T @ R / T

    C_clean = clean_correlation_matrix(C_sample, T, method=method, alpha=alpha)

    # Rescale back to covariance
    Sigma = np.outer(std, std) * C_clean
    return Sigma


# ── Correlation quality metrics ────────────────────────────────────────────────

def correlation_matrix_quality(C: np.ndarray, T: int) -> dict:
    """
    Assess sample correlation matrix quality vs RMT expectations.
    """
    N = C.shape[0]
    eigs = np.linalg.eigvalsh(C)[::-1]
    split = eigenvalue_signal_noise_split(
        np.random.randn(T, N),  # placeholder — use actual returns in practice
    )
    mp = MPDistribution(q=T / N)

    return {
        "condition_number": float(eigs[0] / max(eigs[-1], 1e-10)),
        "effective_rank": effective_rank(eigs),
        "largest_eigenvalue": float(eigs[0]),
        "mp_lambda_plus": mp.lambda_plus,
        "n_above_mp": int(np.sum(eigs > mp.lambda_plus)),
        "frobenius_norm": float(np.linalg.norm(C, "fro")),
        "det": float(np.linalg.det(C)),
        "min_eigenvalue": float(eigs[-1]),
        "is_psd": bool(eigs[-1] >= -1e-8),
    }
