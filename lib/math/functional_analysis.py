"""
functional_analysis.py — Functional data analysis for financial time series.

Implements:
  FPCAResult           – functional PCA of observed curves
  bspline_basis        – B-spline basis via de Boor recursion
  functional_mean      – pointwise mean curve
  functional_covariance– empirical covariance operator (kernel)
  kl_expansion         – Karhunen-Loève expansion (eigenfunctions + scores)
  FunctionalLinearModel– scalar-on-function regression
  functional_kmeans    – k-means clustering on L² distance
  fraiman_muniz_depth  – Fraiman-Muniz functional depth
  band_depth           – López-Pintado & Romo band depth
  functional_outliers  – depth-based outlier detection
  hilbert_inner        – Hilbert space ⟨f, g⟩_{L²}
  hilbert_norm         – Hilbert space ‖f‖_{L²}
  frechet_mean         – Fréchet mean (Wasserstein barycenter approximation)
  persistence_summary  – Topological data analysis: Betti numbers & persistence stats
  path_signature       – Rough path signature up to depth 3
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import interpolate, linalg, optimize, stats


# ---------------------------------------------------------------------------
# B-spline basis (de Boor algorithm)
# ---------------------------------------------------------------------------

def _de_boor(k: int, t: np.ndarray, c: np.ndarray, x: float) -> float:
    """
    Evaluate a B-spline at a single point x using de Boor's algorithm.
    k : degree, t : knot vector, c : control points.
    """
    n = len(t) - k - 1
    # find knot span
    i = np.searchsorted(t, x, side="right") - 1
    i = int(np.clip(i, k, n - 1))
    d = c[i - k: i + 1].copy().astype(float)
    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            denom = t[i - k + j + r] - t[i - k + j]
            if abs(denom) < 1e-14:
                alpha = 0.0
            else:
                alpha = (x - t[i - k + j]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return float(d[k])


def bspline_basis(
    x: np.ndarray,
    n_basis: int = 10,
    degree: int = 3,
    domain: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Evaluate B-spline basis functions on x.

    Returns B of shape (len(x), n_basis).
    Each column is one basis function evaluated at every point in x.

    Parameters
    ----------
    x        : evaluation points
    n_basis  : number of basis functions
    degree   : spline degree (3 = cubic)
    domain   : (lo, hi) — defaults to (x.min(), x.max())
    """
    if domain is None:
        domain = (float(x.min()), float(x.max()))
    lo, hi = domain

    # Uniform interior knots
    n_interior = n_basis - degree - 1
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])

    n_funcs = len(knots) - degree - 1
    B = np.zeros((len(x), n_funcs))
    for j in range(n_funcs):
        c = np.zeros(n_funcs)
        c[j] = 1.0
        for i, xi in enumerate(x):
            xi_clip = float(np.clip(xi, lo, hi - 1e-12))
            B[i, j] = _de_boor(degree, knots, c, xi_clip)
    return B[:, :n_basis]


# ---------------------------------------------------------------------------
# Functional mean and covariance operator
# ---------------------------------------------------------------------------

def functional_mean(curves: np.ndarray) -> np.ndarray:
    """
    Pointwise mean of a set of curves.

    Parameters
    ----------
    curves : (n_curves, n_grid) array

    Returns mean curve of shape (n_grid,).
    """
    return curves.mean(axis=0)


def functional_covariance(curves: np.ndarray) -> np.ndarray:
    """
    Empirical covariance operator K(s, t) = E[(f(s)-μ(s))(f(t)-μ(t))].

    Returns (n_grid, n_grid) covariance matrix.
    """
    n = curves.shape[0]
    centered = curves - functional_mean(curves)
    return centered.T @ centered / (n - 1)


# ---------------------------------------------------------------------------
# Karhunen-Loève expansion
# ---------------------------------------------------------------------------

@dataclass
class KLExpansion:
    """Result of a Karhunen-Loève expansion."""
    eigenvalues: np.ndarray        # descending order
    eigenfunctions: np.ndarray     # (n_components, n_grid)
    scores: np.ndarray             # (n_curves, n_components)
    explained_variance_ratio: np.ndarray


def kl_expansion(
    curves: np.ndarray,
    n_components: int = 5,
    grid: Optional[np.ndarray] = None,
) -> KLExpansion:
    """
    Karhunen-Loève expansion via eigendecomposition of the covariance operator.
    Approximated by PCA on the discretised curves weighted by grid spacing.

    Parameters
    ----------
    curves       : (n_curves, n_grid)
    n_components : number of eigenfunctions to retain
    grid         : 1-D array of grid points (default = equally spaced 0..1)
    """
    n_curves, n_grid = curves.shape
    if grid is None:
        grid = np.linspace(0, 1, n_grid)
    h = float(np.mean(np.diff(grid)))          # uniform spacing assumption

    cov = functional_covariance(curves) * h    # integral approximation
    eigenvalues, eigenvectors = linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = np.maximum(eigenvalues, 0.0)
    n_components = min(n_components, n_grid, n_curves - 1)
    eigenvalues = eigenvalues[:n_components]
    eigenfunctions = eigenvectors[:, :n_components].T   # (n_comp, n_grid)

    # Scores: ξ_{ik} = ⟨f_i - μ, φ_k⟩
    centered = curves - functional_mean(curves)
    scores = centered @ eigenfunctions.T * h            # (n_curves, n_comp)

    total_var = np.maximum(eigenvalues.sum(), 1e-12)
    evr = eigenvalues / total_var

    return KLExpansion(eigenvalues, eigenfunctions, scores, evr)


# ---------------------------------------------------------------------------
# Functional PCA result wrapper
# ---------------------------------------------------------------------------

@dataclass
class FPCAResult:
    """Functional PCA result (wraps KLExpansion with reconstruction helper)."""
    kl: KLExpansion
    mean_curve: np.ndarray
    grid: np.ndarray

    def reconstruct(self, scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Reconstruct curves from FPC scores."""
        s = scores if scores is not None else self.kl.scores
        return self.mean_curve + s @ self.kl.eigenfunctions

    @classmethod
    def fit(cls, curves: np.ndarray, n_components: int = 5,
            grid: Optional[np.ndarray] = None) -> "FPCAResult":
        n_grid = curves.shape[1]
        if grid is None:
            grid = np.linspace(0, 1, n_grid)
        mean_c = functional_mean(curves)
        kl = kl_expansion(curves, n_components=n_components, grid=grid)
        return cls(kl=kl, mean_curve=mean_c, grid=grid)


# ---------------------------------------------------------------------------
# Functional linear regression (scalar-on-function)
# ---------------------------------------------------------------------------

@dataclass
class FunctionalLinearModel:
    """
    Scalar-on-function regression: Y = α + ∫ β(t) X(t) dt + ε.
    β(t) is represented in B-spline basis; fitted via OLS.
    """
    n_basis: int = 10
    degree: int = 3
    coef_: Optional[np.ndarray] = field(default=None, repr=False)
    intercept_: float = 0.0
    grid_: Optional[np.ndarray] = field(default=None, repr=False)

    def fit(self, X_curves: np.ndarray, y: np.ndarray,
            grid: Optional[np.ndarray] = None) -> "FunctionalLinearModel":
        """
        Parameters
        ----------
        X_curves : (n, n_grid) functional predictors
        y        : (n,) scalar responses
        """
        n, n_grid = X_curves.shape
        if grid is None:
            grid = np.linspace(0, 1, n_grid)
        self.grid_ = grid
        h = float(np.mean(np.diff(grid)))

        B = bspline_basis(grid, self.n_basis, self.degree)     # (n_grid, n_basis)
        Z = X_curves @ B * h                                   # (n, n_basis)
        Z_aug = np.column_stack([np.ones(n), Z])
        coef, *_ = np.linalg.lstsq(Z_aug, y, rcond=None)
        self.intercept_ = float(coef[0])
        self.coef_ = coef[1:]
        return self

    def predict(self, X_curves: np.ndarray) -> np.ndarray:
        assert self.coef_ is not None and self.grid_ is not None
        n_grid = X_curves.shape[1]
        h = float(np.mean(np.diff(self.grid_)))
        B = bspline_basis(self.grid_, self.n_basis, self.degree)
        Z = X_curves @ B * h
        return self.intercept_ + Z @ self.coef_

    def beta_function(self) -> np.ndarray:
        """Return β(t) evaluated at the training grid."""
        assert self.coef_ is not None and self.grid_ is not None
        B = bspline_basis(self.grid_, self.n_basis, self.degree)
        return B @ self.coef_


# ---------------------------------------------------------------------------
# Hilbert space inner product and norm
# ---------------------------------------------------------------------------

def hilbert_inner(
    f: np.ndarray,
    g: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> float:
    """
    L² inner product ⟨f, g⟩ = ∫ f(t) g(t) dt  (trapezoidal rule).
    """
    if grid is None:
        grid = np.linspace(0, 1, len(f))
    return float(np.trapz(f * g, grid))


def hilbert_norm(f: np.ndarray, grid: Optional[np.ndarray] = None) -> float:
    """L² norm ‖f‖ = sqrt(⟨f, f⟩)."""
    return math.sqrt(max(hilbert_inner(f, f, grid), 0.0))


# ---------------------------------------------------------------------------
# Functional k-means clustering
# ---------------------------------------------------------------------------

def functional_kmeans(
    curves: np.ndarray,
    k: int = 3,
    max_iter: int = 100,
    n_init: int = 5,
    grid: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    k-means clustering of curves using L² distance.

    Returns (labels, centroids, inertia).
    labels    : (n_curves,) integer cluster assignments
    centroids : (k, n_grid) mean curves per cluster
    inertia   : total within-cluster L² squared distance
    """
    rng = rng or np.random.default_rng()
    n, n_grid = curves.shape
    if grid is None:
        grid = np.linspace(0, 1, n_grid)
    h = float(np.mean(np.diff(grid)))

    def _l2_sq(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum((a - b) ** 2) * h)

    best_labels, best_centroids, best_inertia = None, None, np.inf

    for _ in range(n_init):
        # Initialise centroids by random selection
        idx = rng.choice(n, size=k, replace=False)
        centroids = curves[idx].copy()

        labels = np.zeros(n, dtype=int)
        for iteration in range(max_iter):
            # Assign
            dists = np.array([[_l2_sq(curves[i], centroids[j]) for j in range(k)]
                               for i in range(n)])
            new_labels = dists.argmin(axis=1)
            if np.all(new_labels == labels) and iteration > 0:
                break
            labels = new_labels
            # Update
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centroids[j] = curves[mask].mean(axis=0)

        inertia = float(sum(_l2_sq(curves[i], centroids[labels[i]]) for i in range(n)))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids, best_inertia  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Functional depth measures
# ---------------------------------------------------------------------------

def fraiman_muniz_depth(
    curves: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fraiman-Muniz (2001) functional depth.
    D(x) = ∫ D_t(x(t)) dt  where D_t is the univariate depth at each t.
    Univariate depth: D_t(x) = 1 - |F_t(x) - 0.5| / 0.5  (rank-based).

    Returns depth scores of shape (n_curves,).
    """
    n, n_grid = curves.shape
    if grid is None:
        grid = np.linspace(0, 1, n_grid)

    depths = np.zeros(n)
    for t in range(n_grid):
        col = curves[:, t]
        ranks = stats.rankdata(col) / (n + 1)          # ≈ F_t(x)
        d_t = 1.0 - np.abs(ranks - 0.5) / 0.5
        depths += d_t
    depths /= n_grid
    return depths


def band_depth(
    curves: np.ndarray,
    J: int = 2,
) -> np.ndarray:
    """
    López-Pintado & Romo (2009) band depth BD_J.
    For J=2: fraction of pairs (i,j) such that x lies within the band [min,max].

    Returns depth scores of shape (n_curves,).
    """
    n, n_grid = curves.shape
    depths = np.zeros(n)

    for i in range(n):
        count = 0
        total = 0
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                lo = np.minimum(curves[j1], curves[j2])
                hi = np.maximum(curves[j1], curves[j2])
                if np.all(curves[i] >= lo) and np.all(curves[i] <= hi):
                    count += 1
                total += 1
        depths[i] = count / max(total, 1)
    return depths


# ---------------------------------------------------------------------------
# Functional outlier detection
# ---------------------------------------------------------------------------

def functional_outliers(
    curves: np.ndarray,
    depth_fn: str = "fraiman_muniz",
    threshold_quantile: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Depth-based functional outlier detection.

    Parameters
    ----------
    curves             : (n, n_grid)
    depth_fn           : 'fraiman_muniz' or 'band'
    threshold_quantile : curves below this depth quantile are flagged

    Returns
    -------
    is_outlier : boolean array (n,)
    depths     : depth scores (n,)
    """
    if depth_fn == "fraiman_muniz":
        depths = fraiman_muniz_depth(curves)
    else:
        depths = band_depth(curves)

    cutoff = float(np.quantile(depths, threshold_quantile))
    is_outlier = depths < cutoff
    return is_outlier, depths


# ---------------------------------------------------------------------------
# Fréchet mean / Wasserstein barycenter (1-D distributions)
# ---------------------------------------------------------------------------

def frechet_mean(
    quantile_functions: np.ndarray,
    grid: Optional[np.ndarray] = None,
    n_iter: int = 50,
) -> np.ndarray:
    """
    Fréchet mean under the Wasserstein-2 metric for 1-D distributions.
    Each row of quantile_functions is a quantile function Q_i evaluated on [0,1].

    The Wasserstein barycenter of 1-D distributions is the pointwise mean
    of their quantile functions.

    Parameters
    ----------
    quantile_functions : (n_distributions, n_grid) array of quantile curves
    grid               : probability grid (default: uniform on (0,1))

    Returns the barycenter quantile function (n_grid,).
    """
    # For 1-D Wasserstein, the barycenter is simply the mean quantile function.
    return quantile_functions.mean(axis=0)


# ---------------------------------------------------------------------------
# Persistence homology features (TDA lite)
# ---------------------------------------------------------------------------

def _sublevel_persistence(
    x: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Sub-level set persistence for a 1-D signal.
    Returns list of (birth, death) pairs for H0 (connected components).
    Uses the standard union-find approach on sorted values.
    """
    n = len(x)
    # Each local minimum is a birth event; merging events are deaths.
    pairs: List[Tuple[float, float]] = []
    components: dict = {}   # value → component id

    # Process in increasing order
    idx_sorted = np.argsort(x)
    alive: dict = {}   # component representative → birth value

    for idx in idx_sorted:
        val = float(x[idx])
        # Check neighbours (1-D: left and right)
        left = idx - 1 if idx > 0 else None
        right = idx + 1 if idx < n - 1 else None

        left_alive = (left is not None and left in components)
        right_alive = (right is not None and right in components)

        if not left_alive and not right_alive:
            # New component born
            components[idx] = idx
            alive[idx] = val
        elif left_alive and not right_alive:
            # Extend left component
            root = components[left]
            components[idx] = root
        elif not left_alive and right_alive:
            root = components[right]
            components[idx] = root
        else:
            # Merge: kill younger (higher birth)
            root_l = components[left]
            root_r = components[right]
            if root_l != root_r:
                birth_l = alive.get(root_l, val)
                birth_r = alive.get(root_r, val)
                if birth_l >= birth_r:
                    pairs.append((birth_l, val))
                    alive.pop(root_l, None)
                    components[idx] = root_r
                    # Re-root left component
                    for k in list(components):
                        if components[k] == root_l:
                            components[k] = root_r
                else:
                    pairs.append((birth_r, val))
                    alive.pop(root_r, None)
                    components[idx] = root_l
                    for k in list(components):
                        if components[k] == root_r:
                            components[k] = root_l
            else:
                components[idx] = root_l

    # Remaining alive components are paired with +inf (essential classes)
    for root, birth in alive.items():
        pairs.append((birth, float("inf")))

    return pairs


def persistence_summary(
    x: np.ndarray,
    max_death: Optional[float] = None,
) -> dict:
    """
    Topological summary statistics from sub-level persistence of a 1-D signal.

    Returns a dict with:
      betti_0          : number of H0 generators (connected components) at midpoint
      n_pairs          : total persistence pairs (excluding essential)
      persistence_mean : mean persistence (death - birth)
      persistence_std  : std of persistence
      persistence_entropy : entropy of normalised persistence lengths
      max_persistence  : maximum finite persistence
    """
    pairs = _sublevel_persistence(x)
    if max_death is None:
        max_death = float(x.max())

    finite_pairs = [(b, d) for b, d in pairs if math.isfinite(d)]
    n_infinite = sum(1 for b, d in pairs if not math.isfinite(d))

    if not finite_pairs:
        return {
            "betti_0": n_infinite,
            "n_pairs": 0,
            "persistence_mean": 0.0,
            "persistence_std": 0.0,
            "persistence_entropy": 0.0,
            "max_persistence": 0.0,
        }

    pers = np.array([d - b for b, d in finite_pairs])
    pers_norm = pers / (pers.sum() + 1e-12)
    pers_entropy = float(-np.sum(pers_norm * np.log(np.maximum(pers_norm, 1e-300))))

    return {
        "betti_0": n_infinite,
        "n_pairs": len(finite_pairs),
        "persistence_mean": float(pers.mean()),
        "persistence_std": float(pers.std()),
        "persistence_entropy": pers_entropy,
        "max_persistence": float(pers.max()),
    }


# ---------------------------------------------------------------------------
# Path signature (rough path theory, iterated integrals up to depth 3)
# ---------------------------------------------------------------------------

def path_signature(
    path: np.ndarray,
    depth: int = 3,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute the signature of a path up to a given depth.
    Signature captures iterated integrals: S(X)^{i₁,...,iₖ} = ∫∫···dX^{i₁}···dX^{iₖ}.

    Parameters
    ----------
    path      : (T, d) array — d-dimensional path (each row is a time point)
    depth     : maximum depth of iterated integrals (1, 2, or 3)
    normalize : if True, divide each level by factorial(level)

    Returns a 1-D feature vector of length sum_{k=1}^{depth} d^k.
    """
    if path.ndim == 1:
        path = path.reshape(-1, 1)
    T, d = path.shape
    increments = np.diff(path, axis=0)   # (T-1, d)
    features = []

    # Level 1: S^i = sum_t dX^i_t
    level1 = increments.sum(axis=0)
    if normalize:
        level1 = level1 / math.factorial(1)
    features.append(level1)

    if depth >= 2:
        # Level 2: S^{ij} = sum_{s<t} dX^i_s * dX^j_t  (iterated integral)
        # = (1/2)[(sum dX^i)(sum dX^j) - sum dX^i*dX^j]  by Chen's identity
        level2 = np.zeros((d, d))
        cum = np.zeros(d)
        for t in range(T - 1):
            dx = increments[t]
            level2 += np.outer(cum, dx)
            cum += dx
        if normalize:
            level2 = level2 / math.factorial(2)
        features.append(level2.ravel())

    if depth >= 3:
        # Level 3: S^{ijk} via nested cumulative sums
        level3 = np.zeros((d, d, d))
        cum1 = np.zeros(d)
        cum2 = np.zeros((d, d))
        for t in range(T - 1):
            dx = increments[t]
            # S^{ijk}: integrate level2 against dx^k
            level3 += np.einsum("ij,k->ijk", cum2, dx)
            cum2 += np.outer(cum1, dx)
            cum1 += dx
        if normalize:
            level3 = level3 / math.factorial(3)
        features.append(level3.ravel())

    return np.concatenate(features)
