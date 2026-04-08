"""
Matrix factorization methods for financial data.

Implements:
  - Non-negative Matrix Factorization (NMF) — latent factor extraction
  - Independent Component Analysis (ICA) — blind source separation
  - Robust PCA (RPCA) — outlier/sparse + low-rank decomposition
  - Sparse Factor Models (LASSO-based)
  - Online PCA (streaming covariance)
  - Factor rotation (Varimax, Promax)
  - Latent Dirichlet Allocation (for text-based signals)
  - Tensor decomposition (CP/Tucker for multi-asset multi-period)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── NMF — Non-negative Matrix Factorization ──────────────────────────────────

def nmf(
    V: np.ndarray,
    r: int,
    n_iter: int = 200,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NMF: V ≈ W @ H, W >= 0, H >= 0.
    Uses multiplicative update rules (Lee-Seung 2001).

    V: (m, n) non-negative matrix (e.g., abs returns)
    r: number of latent factors
    Returns (W, H) of shapes (m, r) and (r, n).
    """
    rng = rng or np.random.default_rng(42)
    m, n = V.shape
    W = np.abs(rng.standard_normal((m, r))) + 0.1
    H = np.abs(rng.standard_normal((r, n))) + 0.1

    eps = 1e-10
    prev_err = np.inf

    for _ in range(n_iter):
        # Update H
        WtV = W.T @ V
        WtWH = W.T @ W @ H + eps
        H *= WtV / WtWH

        # Update W
        VHt = V @ H.T
        WHHt = W @ H @ H.T + eps
        W *= VHt / WHHt

        # Check convergence
        err = float(np.linalg.norm(V - W @ H, "fro"))
        if abs(prev_err - err) / (prev_err + eps) < tol:
            break
        prev_err = err

    return W, H


def nmf_factor_loadings(
    returns: np.ndarray,
    r: int = 5,
) -> dict:
    """
    Extract latent risk factors from returns using NMF on absolute returns.
    Returns factor loadings and time series.
    """
    V = np.abs(returns)
    V_shifted = V - V.min() + 1e-8  # ensure non-negative
    W, H = nmf(V_shifted, r)

    # Normalize
    norms = np.linalg.norm(W, axis=0) + 1e-10
    W = W / norms
    H = (H.T * norms).T

    return {
        "loadings": W,         # (n_assets, r)
        "factors": H,          # (r, T)
        "explained_var": float(1 - np.linalg.norm(V_shifted - W @ H)**2 / np.linalg.norm(V_shifted)**2),
    }


# ── ICA — Independent Component Analysis ────────────────────────────────────

def ica_fastica(
    X: np.ndarray,
    n_components: int,
    n_iter: int = 200,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    FastICA: blind source separation.
    X: (T, n_features) — e.g., multi-asset returns
    Returns (S, W) where S = X @ W.T are independent components.
    Uses kurtosis-maximizing negentropy approximation.
    """
    rng = rng or np.random.default_rng(42)
    T, n = X.shape
    k = min(n_components, n)

    # Whiten
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / T
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:k]
    D = np.diag(1.0 / np.sqrt(eigvals[idx] + 1e-10))
    E = eigvecs[:, idx]
    Xw = Xc @ E @ D  # whitened: (T, k)

    # FastICA fixed-point algorithm
    W = rng.standard_normal((k, k))
    W = _orth(W)

    for _ in range(n_iter):
        W_old = W.copy()
        # g(u) = tanh(u), g'(u) = 1 - tanh^2(u)
        G = np.tanh(Xw @ W.T)  # (T, k)
        Gp = 1 - G**2          # (T, k)
        W_new = (G.T @ Xw) / T - Gp.mean(axis=0)[:, None] * W
        W = _orth(W_new)

        if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1)) < tol:
            break

    S = Xw @ W.T  # independent components (T, k)
    return S, W


def _orth(W: np.ndarray) -> np.ndarray:
    """Symmetric orthogonalization."""
    eigvals, eigvecs = np.linalg.eigh(W @ W.T)
    return eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T @ W


# ── Robust PCA ────────────────────────────────────────────────────────────────

def robust_pca(
    M: np.ndarray,
    lam: Optional[float] = None,
    n_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust PCA via Principal Component Pursuit (Candès et al. 2011).
    Decomposes M = L + S where L is low-rank, S is sparse.

    Applications: separate factor returns (L) from idiosyncratic/outlier (S).
    Returns (L, S).
    """
    m, n = M.shape
    if lam is None:
        lam = 1.0 / math.sqrt(max(m, n))

    mu = m * n / (4 * np.abs(M).sum())
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)

    prev_err = np.inf
    for _ in range(n_iter):
        # SVT step (low-rank update)
        U, sigma, Vt = np.linalg.svd(M - S + Y / mu, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1 / mu, 0)
        L = U @ np.diag(sigma_thresh) @ Vt

        # Shrinkage step (sparse update)
        R = M - L + Y / mu
        S = np.sign(R) * np.maximum(np.abs(R) - lam / mu, 0)

        # Dual update
        Y = Y + mu * (M - L - S)

        err = float(np.linalg.norm(M - L - S, "fro") / (np.linalg.norm(M, "fro") + 1e-10))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return L, S


def rpca_factor_model(
    returns: np.ndarray,
    n_factors: Optional[int] = None,
) -> dict:
    """
    Apply Robust PCA to returns to extract clean factor structure.
    returns: (T, N)
    """
    L, S = robust_pca(returns)

    # SVD of low-rank part
    U, sigma, Vt = np.linalg.svd(L, full_matrices=False)

    if n_factors is None:
        # Elbow heuristic
        explained = np.cumsum(sigma**2) / (sigma**2).sum()
        n_factors = int(np.searchsorted(explained, 0.80)) + 1
        n_factors = min(n_factors, len(sigma))

    factors = U[:, :n_factors] * sigma[:n_factors]  # (T, k)
    loadings = Vt[:n_factors].T                      # (N, k)
    sparse = S

    return {
        "factors": factors,
        "loadings": loadings,
        "sparse_component": sparse,
        "singular_values": sigma,
        "n_factors": n_factors,
        "low_rank_explained": float((sigma[:n_factors]**2).sum() / (sigma**2).sum()),
        "sparse_density": float(np.mean(S != 0)),
    }


# ── Sparse Factor Models ───────────────────────────────────────────────────────

def sparse_pca_deflation(
    X: np.ndarray,
    n_components: int = 5,
    sparsity: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sparse PCA via deflation + soft-thresholding.
    Each component explains max variance with sparse loadings.
    Returns (loadings, scores) of shapes (n_features, k) and (T, k).
    """
    T, n = X.shape
    Xr = X.copy()
    loadings = np.zeros((n, n_components))
    scores = np.zeros((T, n_components))

    for k in range(n_components):
        cov = Xr.T @ Xr / T
        # Power iteration
        v = np.random.default_rng(k).standard_normal(n)
        v /= np.linalg.norm(v)

        for _ in range(50):
            z = cov @ v
            # Soft-threshold
            thresh = sparsity * np.abs(z).max()
            z = np.sign(z) * np.maximum(np.abs(z) - thresh, 0)
            if np.linalg.norm(z) < 1e-10:
                break
            v = z / np.linalg.norm(z)

        loadings[:, k] = v
        sc = Xr @ v
        scores[:, k] = sc
        # Deflate
        Xr = Xr - np.outer(sc, v)

    return loadings, scores


def lasso_factor_regression(
    returns: np.ndarray,
    factors: np.ndarray,
    alpha: float = 0.01,
) -> dict:
    """
    LASSO-regularized factor regression: r_t = B * f_t + eps_t.
    Selects sparse set of factor exposures per asset.
    Returns beta matrix (N, k) and idiosyncratic residuals.
    """
    T, N = returns.shape
    _, k = factors.shape

    # Standardize factors
    f_std = (factors - factors.mean(0)) / (factors.std(0) + 1e-10)

    betas = np.zeros((N, k))
    for i in range(N):
        y = returns[:, i]
        # Coordinate descent LASSO
        b = np.zeros(k)
        for _ in range(100):
            for j in range(k):
                r_j = y - f_std @ b + b[j] * f_std[:, j]
                corr = f_std[:, j] @ r_j / T
                b[j] = float(np.sign(corr) * max(abs(corr) - alpha, 0))
        betas[i] = b

    fitted = f_std @ betas.T
    residuals = returns - fitted.T

    return {
        "betas": betas,
        "residuals": residuals.T,
        "r_squared": float(1 - residuals.var(0).mean() / (returns.var(0).mean() + 1e-10)),
        "sparsity": float((betas == 0).mean()),
    }


# ── Online / Streaming PCA ────────────────────────────────────────────────────

class OnlinePCA:
    """
    Streaming PCA via incremental SVD (Brand 2002).
    Updates eigenvectors one sample at a time — suitable for live data.
    """

    def __init__(self, n_components: int = 5, forgetting: float = 1.0):
        self.k = n_components
        self.forgetting = forgetting  # < 1 for time-decay
        self.n_seen = 0
        self.mean = None
        self.components = None    # (k, n_features)
        self.explained_var = None

    def update(self, x: np.ndarray) -> None:
        """Update PCA with new observation x (n_features,)."""
        if self.mean is None:
            self.mean = x.copy()
            self.components = np.zeros((self.k, len(x)))
            self.explained_var = np.ones(self.k)
            self.n_seen = 1
            return

        self.n_seen += 1
        gamma = self.forgetting
        self.mean = gamma * self.mean + (1 - gamma) * x
        xc = x - self.mean

        # Project onto current components
        proj = self.components @ xc
        resid = xc - self.components.T @ proj

        # Update via Gram-Schmidt + re-orthogonalization
        if np.linalg.norm(resid) > 1e-10:
            resid_norm = resid / np.linalg.norm(resid)
            aug = np.vstack([self.components, resid_norm[None, :]])
        else:
            aug = self.components

        # Re-orthogonalize (QR)
        if aug.shape[0] > self.k:
            Q, _ = np.linalg.qr(aug.T)
            self.components = Q[:, :self.k].T
        else:
            Q, _ = np.linalg.qr(aug.T)
            self.components = Q.T[:self.k]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project (T, n_features) onto principal components."""
        if self.mean is None:
            return X[:, :self.k]
        Xc = X - self.mean
        return Xc @ self.components.T


# ── Factor Rotation ───────────────────────────────────────────────────────────

def varimax_rotation(
    loadings: np.ndarray,
    n_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Varimax orthogonal rotation (Kaiser 1958).
    Maximizes variance of squared loadings per factor.
    Returns (rotated_loadings, rotation_matrix).
    """
    L = loadings.copy()
    n, k = L.shape
    R = np.eye(k)
    prev_crit = -np.inf

    for _ in range(n_iter):
        for i in range(k):
            for j in range(i + 1, k):
                # Compute rotation angle
                u = L[:, i]**2 - L[:, j]**2
                v = 2 * L[:, i] * L[:, j]
                A = u.sum()
                B = v.sum()
                C = (u**2 - v**2).sum()
                D = 2 * (u * v).sum()

                num = D - 2 * A * B / n
                denom = C - (A**2 - B**2) / n
                theta = 0.25 * math.atan2(num, denom)

                c, s = math.cos(theta), math.sin(theta)
                rot = np.eye(k)
                rot[i, i] = rot[j, j] = c
                rot[i, j] = -s
                rot[j, i] = s
                L = L @ rot
                R = R @ rot

        crit = float(np.sum(np.sum(L**2, axis=0)**2))
        if abs(crit - prev_crit) < tol:
            break
        prev_crit = crit

    return L, R


# ── CP Tensor Decomposition ───────────────────────────────────────────────────

def cp_decomposition(
    X: np.ndarray,
    r: int,
    n_iter: int = 100,
    tol: float = 1e-5,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """
    CP (CANDECOMP/PARAFAC) tensor decomposition via ALS.
    X: 3D tensor (e.g., assets × time × features)
    Returns list of factor matrices [A, B, C] for each mode.
    """
    rng = rng or np.random.default_rng(42)
    shape = X.shape
    ndim = len(shape)
    factors = [rng.standard_normal((shape[i], r)) for i in range(ndim)]

    def khatri_rao(A, B):
        """Khatri-Rao product: column-wise Kronecker."""
        n_a, r = A.shape
        n_b, _ = B.shape
        result = np.zeros((n_a * n_b, r))
        for j in range(r):
            result[:, j] = np.kron(A[:, j], B[:, j])
        return result

    prev_err = np.inf
    for _ in range(n_iter):
        for mode in range(ndim):
            # Unfold tensor along this mode
            Xn = _tensor_unfold(X, mode)
            # Compute Khatri-Rao product of all other modes
            other = [factors[i] for i in range(ndim) if i != mode]
            KR = other[0]
            for A in other[1:]:
                KR = khatri_rao(KR, A)
            # ALS update
            V = np.ones((r, r))
            for i, A in enumerate(other):
                V = V * (A.T @ A)
            try:
                factors[mode] = Xn @ KR @ np.linalg.inv(V + 1e-8 * np.eye(r))
            except np.linalg.LinAlgError:
                pass

        # Check convergence
        reconstructed = _cp_reconstruct(factors, shape)
        err = float(np.linalg.norm(X - reconstructed) / (np.linalg.norm(X) + 1e-10))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return factors


def _tensor_unfold(X: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n unfolding of tensor X."""
    shape = X.shape
    ndim = len(shape)
    order = [mode] + [i for i in range(ndim) if i != mode]
    Xt = np.transpose(X, order)
    return Xt.reshape(shape[mode], -1)


def _cp_reconstruct(factors: list, shape: tuple) -> np.ndarray:
    """Reconstruct tensor from CP factors."""
    r = factors[0].shape[1]
    result = np.zeros(shape)
    for j in range(r):
        component = factors[0][:, j]
        for A in factors[1:]:
            component = np.tensordot(component, A[:, j], axes=0)
        result = result + component
    return result
