"""
Tensor decomposition methods for multi-dimensional financial data.

Implements CP, Tucker, non-negative tensor factorization, tensor completion,
multi-linear PCA, tensor regression, and online/streaming decomposition.
All operations use numpy/scipy only.
"""

import numpy as np
from scipy.linalg import svd, qr, solve
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Core tensor utilities
# ---------------------------------------------------------------------------

def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """
    Mode-n unfolding (matricization) of a tensor.

    Parameters
    ----------
    tensor : ndarray of arbitrary order
    mode : which mode to unfold along (0-indexed)

    Returns
    -------
    2-D array of shape (tensor.shape[mode], prod of other dims)
    """
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix: np.ndarray, mode: int, shape: tuple) -> np.ndarray:
    """Inverse of unfold: reshape matrix back into tensor."""
    full_shape = [shape[mode]] + [shape[i] for i in range(len(shape)) if i != mode]
    return np.moveaxis(matrix.reshape(full_shape), 0, mode)


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """
    Mode-n product of tensor with matrix.
    Result shape: same as tensor except dimension mode becomes matrix.shape[0].
    """
    unf = unfold(tensor, mode)
    result_unf = matrix @ unf
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(result_unf, mode, tuple(new_shape))


def khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Column-wise Khatri-Rao product of A (I x R) and B (J x R).
    Result is (I*J x R).
    """
    I, R = A.shape
    J, _ = B.shape
    result = np.zeros((I * J, R))
    for r in range(R):
        result[:, r] = np.outer(A[:, r], B[:, r]).ravel()
    return result


def outer_product_factors(factors: list) -> np.ndarray:
    """
    Reconstruct full tensor from a list of factor matrices (CP format).
    factors[n] has shape (I_n, R).
    """
    R = factors[0].shape[1]
    shape = tuple(f.shape[0] for f in factors)
    tensor = np.zeros(shape)
    for r in range(R):
        component = factors[0][:, r]
        for n in range(1, len(factors)):
            component = np.multiply.outer(component, factors[n][:, r])
        tensor += component
    return tensor


def tensor_norm(tensor: np.ndarray) -> float:
    """Frobenius norm of a tensor."""
    return np.sqrt(np.sum(tensor**2))


# ---------------------------------------------------------------------------
# CP Decomposition via ALS
# ---------------------------------------------------------------------------

def cp_als(
    tensor: np.ndarray,
    rank: int,
    max_iter: int = 200,
    tol: float = 1e-8,
    seed: int = 42,
) -> dict:
    """
    CP decomposition via Alternating Least Squares (ALS).

    Decomposes tensor X ≈ sum_{r=1}^R a_r ⊗ b_r ⊗ c_r (for 3-way).

    Parameters
    ----------
    tensor : 3-D ndarray of shape (I, J, K)
    rank : number of components R
    max_iter : maximum ALS iterations
    tol : convergence tolerance on relative reconstruction error change
    seed : random seed

    Returns
    -------
    dict with factors (list of 3 matrices), lambdas (weights), reconstruction_errors
    """
    rng = np.random.default_rng(seed)
    ndims = tensor.ndim
    shape = tensor.shape

    # Initialize factors randomly
    factors = [rng.standard_normal((shape[n], rank)) for n in range(ndims)]

    # Normalise columns
    lambdas = np.ones(rank)
    for n in range(ndims):
        norms = np.linalg.norm(factors[n], axis=0)
        norms[norms == 0] = 1.0
        factors[n] /= norms
        lambdas *= norms

    errors = []
    prev_error = np.inf

    for iteration in range(max_iter):
        for n in range(ndims):
            # V = Hadamard product of all (F_m^T F_m) for m != n
            V = np.ones((rank, rank))
            for m in range(ndims):
                if m != n:
                    V *= (factors[m].T @ factors[m])

            # Khatri-Rao product of all factors except n (in reverse order)
            kr_indices = [m for m in range(ndims) if m != n]
            kr = factors[kr_indices[-1]].copy()
            for m in reversed(kr_indices[:-1]):
                kr = khatri_rao(factors[m], kr)

            # Update factor n
            Xn = unfold(tensor, n)
            factors[n] = Xn @ kr @ np.linalg.pinv(V)

            # Normalise
            norms = np.linalg.norm(factors[n], axis=0)
            norms[norms == 0] = 1.0
            lambdas_new = norms
            factors[n] /= norms

        lambdas = lambdas_new

        # Reconstruction error
        recon = outer_product_factors([factors[n] * lambdas**(1.0 / ndims) for n in range(ndims)])
        error = tensor_norm(tensor - recon) / max(tensor_norm(tensor), 1e-16)
        errors.append(error)

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    # Scale lambdas into first factor for cleaner output
    scaled_factors = [f.copy() for f in factors]
    scaled_factors[0] = factors[0] * lambdas[np.newaxis, :]

    return {
        "factors": scaled_factors,
        "lambdas": lambdas,
        "reconstruction_errors": errors,
        "n_iterations": len(errors),
    }


# ---------------------------------------------------------------------------
# Tucker Decomposition with HOSVD initialization
# ---------------------------------------------------------------------------

def hosvd(tensor: np.ndarray, ranks: tuple) -> dict:
    """
    Higher-Order SVD (truncated) for Tucker decomposition initialization.

    Parameters
    ----------
    tensor : N-D array
    ranks : tuple of target ranks for each mode

    Returns
    -------
    dict with core (core tensor), factors (list of orthogonal matrices)
    """
    ndims = tensor.ndim
    factors = []
    for n in range(ndims):
        Xn = unfold(tensor, n)
        U, _, _ = svd(Xn, full_matrices=False)
        factors.append(U[:, :ranks[n]])

    # Core tensor
    core = tensor.copy()
    for n in range(ndims):
        core = mode_n_product(core, factors[n].T, n)

    return {"core": core, "factors": factors}


def tucker_als(
    tensor: np.ndarray,
    ranks: tuple,
    max_iter: int = 100,
    tol: float = 1e-8,
    seed: int = 42,
) -> dict:
    """
    Tucker decomposition via ALS with HOSVD initialization.

    X ≈ G ×_1 U_1 ×_2 U_2 ×_3 U_3
    """
    ndims = tensor.ndim

    # Initialize with HOSVD
    init = hosvd(tensor, ranks)
    factors = init["factors"]

    errors = []
    prev_error = np.inf

    for iteration in range(max_iter):
        for n in range(ndims):
            # Y = X ×_{m!=n} U_m^T
            Y = tensor.copy()
            for m in range(ndims):
                if m != n:
                    Y = mode_n_product(Y, factors[m].T, m)

            Yn = unfold(Y, n)
            U, _, _ = svd(Yn, full_matrices=False)
            factors[n] = U[:, :ranks[n]]

        # Core
        core = tensor.copy()
        for n in range(ndims):
            core = mode_n_product(core, factors[n].T, n)

        # Reconstruction
        recon = core.copy()
        for n in range(ndims):
            recon = mode_n_product(recon, factors[n], n)

        error = tensor_norm(tensor - recon) / max(tensor_norm(tensor), 1e-16)
        errors.append(error)

        if abs(prev_error - error) < tol:
            break
        prev_error = error

    return {
        "core": core,
        "factors": factors,
        "reconstruction_errors": errors,
        "n_iterations": len(errors),
    }


# ---------------------------------------------------------------------------
# Non-negative Tensor Factorization (NTF) via multiplicative updates
# ---------------------------------------------------------------------------

def nonneg_cp(
    tensor: np.ndarray,
    rank: int,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """
    Non-negative CP decomposition via Lee-Seung multiplicative updates.

    Parameters
    ----------
    tensor : non-negative 3-D array
    rank : number of components
    max_iter, tol : convergence parameters

    Returns
    -------
    dict with factors (list), reconstruction_errors
    """
    rng = np.random.default_rng(seed)
    ndims = tensor.ndim
    shape = tensor.shape
    eps = 1e-12

    factors = [np.abs(rng.standard_normal((shape[n], rank))) + eps for n in range(ndims)]
    errors = []

    for iteration in range(max_iter):
        for n in range(ndims):
            # Numerator: X_(n) @ KR(all except n)
            kr_indices = [m for m in range(ndims) if m != n]
            kr = factors[kr_indices[-1]].copy()
            for m in reversed(kr_indices[:-1]):
                kr = khatri_rao(factors[m], kr)

            Xn = unfold(tensor, n)
            numerator = Xn @ kr

            # Denominator: factors[n] @ (Hadamard of F^T F)
            V = np.ones((rank, rank))
            for m in range(ndims):
                if m != n:
                    V *= (factors[m].T @ factors[m])
            denominator = factors[n] @ V

            factors[n] *= (numerator / (denominator + eps))
            factors[n] = np.maximum(factors[n], eps)

        recon = outer_product_factors(factors)
        error = tensor_norm(tensor - recon) / max(tensor_norm(tensor), 1e-16)
        errors.append(error)

        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tol:
            break

    return {"factors": factors, "reconstruction_errors": errors, "n_iterations": len(errors)}


# ---------------------------------------------------------------------------
# Tensor Completion (nuclear norm approach via ALS on low-rank)
# ---------------------------------------------------------------------------

def tensor_completion_als(
    tensor: np.ndarray,
    mask: np.ndarray,
    rank: int,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """
    Tensor completion for missing data via weighted CP-ALS.

    Parameters
    ----------
    tensor : 3-D array with missing entries (can be any value where mask=0)
    mask : binary 3-D array, 1 = observed, 0 = missing
    rank : CP rank for completion
    max_iter, tol, seed : convergence parameters

    Returns
    -------
    dict with completed_tensor, factors, reconstruction_errors
    """
    rng = np.random.default_rng(seed)
    ndims = tensor.ndim
    shape = tensor.shape

    factors = [rng.standard_normal((shape[n], rank)) * 0.1 for n in range(ndims)]

    # Fill missing values with mean of observed
    observed_mean = np.mean(tensor[mask > 0]) if np.any(mask > 0) else 0.0
    filled = tensor * mask + observed_mean * (1 - mask)

    errors = []

    for iteration in range(max_iter):
        # Update filled tensor with current CP approximation at missing entries
        approx = outer_product_factors(factors)
        filled = tensor * mask + approx * (1 - mask)

        # Standard CP-ALS on filled tensor
        for n in range(ndims):
            V = np.ones((rank, rank))
            for m in range(ndims):
                if m != n:
                    V *= (factors[m].T @ factors[m])

            kr_indices = [m for m in range(ndims) if m != n]
            kr = factors[kr_indices[-1]].copy()
            for m in reversed(kr_indices[:-1]):
                kr = khatri_rao(factors[m], kr)

            Xn = unfold(filled, n)
            factors[n] = Xn @ kr @ np.linalg.pinv(V)

        approx = outer_product_factors(factors)
        # Error only on observed entries
        diff = (tensor - approx) * mask
        error = np.sqrt(np.sum(diff**2)) / max(np.sqrt(np.sum((tensor * mask)**2)), 1e-16)
        errors.append(error)

        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tol:
            break

    completed = tensor * mask + outer_product_factors(factors) * (1 - mask)
    return {"completed_tensor": completed, "factors": factors, "reconstruction_errors": errors}


# ---------------------------------------------------------------------------
# Multi-linear PCA
# ---------------------------------------------------------------------------

def multilinear_pca(
    tensor: np.ndarray,
    ranks: tuple,
) -> dict:
    """
    Multi-linear PCA via mode-wise SVD (equivalent to truncated HOSVD).

    For a 3-way tensor (assets x features x time), extracts principal
    subspaces along each mode.

    Returns dict with projections, explained_variance_ratios per mode.
    """
    ndims = tensor.ndim
    factors = []
    explained_ratios = []

    for n in range(ndims):
        Xn = unfold(tensor, n)
        U, S, _ = svd(Xn, full_matrices=False)
        total_var = np.sum(S**2)
        kept_var = np.sum(S[:ranks[n]]**2)
        factors.append(U[:, :ranks[n]])
        explained_ratios.append(kept_var / max(total_var, 1e-16))

    # Projected (compressed) core
    core = tensor.copy()
    for n in range(ndims):
        core = mode_n_product(core, factors[n].T, n)

    # Reconstruction
    recon = core.copy()
    for n in range(ndims):
        recon = mode_n_product(recon, factors[n], n)

    rel_error = tensor_norm(tensor - recon) / max(tensor_norm(tensor), 1e-16)

    return {
        "core": core,
        "factors": factors,
        "explained_variance_ratios": explained_ratios,
        "reconstruction_error": rel_error,
    }


# ---------------------------------------------------------------------------
# Tensor Regression: scalar-on-tensor via CP penalty
# ---------------------------------------------------------------------------

def tensor_regression_cp(
    X_tensors: np.ndarray,
    y: np.ndarray,
    rank: int,
    reg_lambda: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> dict:
    """
    Scalar-on-tensor regression: y_i = <B, X_i> + epsilon_i
    where B is a low-rank (CP) coefficient tensor.

    Parameters
    ----------
    X_tensors : (n_samples, I, J, K) array of predictor tensors
    y : (n_samples,) response vector
    rank : CP rank of coefficient tensor B
    reg_lambda : L2 regularization
    max_iter, tol, seed : convergence parameters

    Returns
    -------
    dict with coefficient_tensor, factors, mse, r_squared
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    tensor_shape = X_tensors.shape[1:]
    ndims = len(tensor_shape)

    factors = [rng.standard_normal((tensor_shape[n], rank)) * 0.01 for n in range(ndims)]

    prev_mse = np.inf

    for iteration in range(max_iter):
        for n in range(ndims):
            # For each sample, compute the "reduced" inner product excluding mode n
            # This gives a matrix Z of shape (n_samples, I_n * R)
            Z = np.zeros((n_samples, tensor_shape[n], rank))

            for i in range(n_samples):
                Xi = X_tensors[i]
                # Contract Xi with all factors except n
                contracted = Xi.copy()
                # We need: for each r, contract Xi with product of factor columns except mode n
                for r in range(rank):
                    temp = Xi.copy()
                    for m in range(ndims):
                        if m != n:
                            # Contract mode m with factors[m][:, r]
                            temp = np.tensordot(temp, factors[m][:, r], axes=([m if m < n else m], [0]))
                            # After contraction, dimensions shift
                            # This is tricky; use simpler approach
                    pass

            # Simpler approach: compute B, then use gradient update
            B = outer_product_factors(factors)
            y_pred = np.array([np.sum(X_tensors[i] * B) for i in range(n_samples)])
            residuals = y - y_pred

            # Gradient w.r.t. factors[n]
            grad = np.zeros_like(factors[n])
            for i in range(n_samples):
                Xi = X_tensors[i]
                # d<B, Xi>/d factors[n][:, r] involves contracting Xi with all other factor columns
                for r in range(rank):
                    # Contract Xi along all modes except n with factor columns r
                    temp = Xi
                    contract_order = []
                    for m in range(ndims - 1, -1, -1):
                        if m != n:
                            temp = np.tensordot(temp, factors[m][:, r], axes=([m], [0]))

                    grad[:, r] += residuals[i] * temp.ravel()

            factors[n] += 0.01 * (grad / n_samples - reg_lambda * factors[n])

        B = outer_product_factors(factors)
        y_pred = np.array([np.sum(X_tensors[i] * B) for i in range(n_samples)])
        mse = np.mean((y - y_pred)**2)

        if abs(prev_mse - mse) < tol:
            break
        prev_mse = mse

    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-16)

    return {
        "coefficient_tensor": B,
        "factors": factors,
        "mse": mse,
        "r_squared": r2,
    }


# ---------------------------------------------------------------------------
# Dynamic tensor decomposition: online CP update
# ---------------------------------------------------------------------------

def online_cp_update(
    factors: list,
    new_slice: np.ndarray,
    mode: int,
    learning_rate: float = 0.01,
    n_inner: int = 5,
) -> list:
    """
    Online CP update for streaming data along one mode.

    Given existing CP factors and a new slice of data, update factors
    using stochastic gradient descent.

    Parameters
    ----------
    factors : list of factor matrices from previous decomposition
    new_slice : new data slice (ndarray with one fewer dimension than the full tensor)
    mode : the streaming mode (dimension that grows)
    learning_rate : SGD step size
    n_inner : number of inner gradient steps

    Returns
    -------
    Updated list of factor matrices (mode-th factor gets a new row appended)
    """
    rank = factors[0].shape[1]
    ndims = len(factors)

    # Append new row to the streaming mode factor
    new_row = np.random.randn(1, rank) * 0.01
    updated_factors = [f.copy() for f in factors]
    updated_factors[mode] = np.vstack([factors[mode], new_row])

    # The new slice should be approximated by sum_r prod_{m!=mode} factors[m][:, r] * new_factor_row[r]
    # Optimize new_factor_row and slightly adjust other factors

    new_idx = updated_factors[mode].shape[0] - 1

    for _ in range(n_inner):
        # Reconstruct the slice from factors
        approx = np.zeros(new_slice.shape)
        for r in range(rank):
            component = np.ones(1)
            for m in range(ndims):
                if m != mode:
                    component = np.multiply.outer(component, updated_factors[m][:, r])
            component = component.reshape(new_slice.shape) * updated_factors[mode][new_idx, r]
            approx += component

        residual = new_slice - approx

        # Gradient for the new row
        for r in range(rank):
            component = np.ones(1)
            for m in range(ndims):
                if m != mode:
                    component = np.multiply.outer(component, updated_factors[m][:, r])
            component = component.reshape(new_slice.shape)
            grad = np.sum(residual * component)
            updated_factors[mode][new_idx, r] += learning_rate * grad

    return updated_factors


# ---------------------------------------------------------------------------
# Regime factor extraction via tensor decomposition
# ---------------------------------------------------------------------------

def regime_factor_extraction(
    returns: np.ndarray,
    features: np.ndarray,
    n_regimes: int = 3,
    rank: int = 5,
    window: int = 60,
    seed: int = 42,
) -> dict:
    """
    Multi-asset multi-feature regime factor extraction.

    Constructs a 3-way tensor (assets x features x time_windows) and
    decomposes to find latent regime factors.

    Parameters
    ----------
    returns : (n_assets, T) array of returns
    features : (n_features, T) array of market features
    n_regimes : number of regimes to identify
    rank : CP rank
    window : rolling window size
    seed : random seed

    Returns
    -------
    dict with regime_assignments, asset_loadings, feature_loadings, time_loadings
    """
    n_assets, T = returns.shape
    n_features = features.shape[0]
    n_windows = T - window + 1

    # Build tensor: covariance of assets x features in each window
    tensor = np.zeros((n_assets, n_features, n_windows))
    for w in range(n_windows):
        ret_w = returns[:, w:w + window]  # (n_assets, window)
        feat_w = features[:, w:w + window]  # (n_features, window)
        # Cross-covariance
        tensor[:, :, w] = (ret_w @ feat_w.T) / window

    # CP decomposition
    result = cp_als(tensor, rank=rank, seed=seed)
    factors = result["factors"]

    # Cluster time loadings into regimes via simple k-means
    time_loadings = factors[2]  # (n_windows, rank)
    # Simple k-means
    rng = np.random.default_rng(seed)
    centroids = time_loadings[rng.choice(n_windows, n_regimes, replace=False)]

    for _ in range(50):
        dists = np.linalg.norm(time_loadings[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        for k in range(n_regimes):
            mask = assignments == k
            if np.any(mask):
                new_centroids[k] = time_loadings[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return {
        "regime_assignments": assignments,
        "asset_loadings": factors[0],
        "feature_loadings": factors[1],
        "time_loadings": factors[2],
        "centroids": centroids,
        "cp_errors": result["reconstruction_errors"],
    }


# ---------------------------------------------------------------------------
# Tensor coherence and uniqueness
# ---------------------------------------------------------------------------

def tensor_coherence(factors: list) -> dict:
    """
    Compute coherence measures for a CP decomposition to assess uniqueness.

    The coherence of factor matrix U is max |<u_r, u_s>| / (||u_r|| ||u_s||)
    for r != s. Low coherence suggests the decomposition is more unique.

    Returns dict with coherence per mode and overall.
    """
    coherences = []
    for n, F in enumerate(factors):
        R = F.shape[1]
        if R <= 1:
            coherences.append(0.0)
            continue

        # Normalise columns
        norms = np.linalg.norm(F, axis=0)
        norms[norms == 0] = 1.0
        F_norm = F / norms

        G = np.abs(F_norm.T @ F_norm)
        np.fill_diagonal(G, 0.0)
        coherences.append(float(np.max(G)))

    return {
        "mode_coherences": coherences,
        "max_coherence": max(coherences) if coherences else 0.0,
    }


def kruskal_uniqueness_bound(factors: list) -> dict:
    """
    Check Kruskal's uniqueness condition for CP decomposition.

    A sufficient condition for uniqueness (up to permutation and scaling):
        k_1 + k_2 + ... + k_N >= 2R + (N-1)
    where k_n = k-rank of factor n and R = CP rank.

    k-rank of a matrix is the maximum k such that every set of k columns is linearly independent.
    """
    R = factors[0].shape[1]
    N = len(factors)

    k_ranks = []
    for F in factors:
        I, r = F.shape
        # k-rank: check subsets (heuristic: use rank of submatrices)
        # Exact computation is NP-hard; use lower bound = rank(F)
        k_rank = min(np.linalg.matrix_rank(F), r)
        k_ranks.append(k_rank)

    k_sum = sum(k_ranks)
    threshold = 2 * R + (N - 1)
    is_unique = k_sum >= threshold

    return {
        "k_ranks": k_ranks,
        "k_sum": k_sum,
        "threshold": threshold,
        "is_unique": is_unique,
    }


# ---------------------------------------------------------------------------
# Tensor power method for symmetric tensors
# ---------------------------------------------------------------------------

def symmetric_tensor_power_method(
    tensor: np.ndarray,
    rank: int = 1,
    max_iter: int = 100,
    tol: float = 1e-8,
    seed: int = 42,
) -> dict:
    """
    Tensor power method for symmetric 3rd-order tensor decomposition.

    For a symmetric tensor T of shape (d, d, d), finds eigenvectors v
    such that T(I, v, v) = lambda * v.

    Uses deflation to find multiple components.
    """
    rng = np.random.default_rng(seed)
    d = tensor.shape[0]
    assert tensor.ndim == 3 and tensor.shape[1] == d and tensor.shape[2] == d

    eigenvalues = []
    eigenvectors = []
    residual = tensor.copy()

    for r in range(rank):
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)

        for _ in range(max_iter):
            # T(I, v, v)
            Tvv = np.einsum('ijk,j,k->i', residual, v, v)
            lam = np.linalg.norm(Tvv)
            if lam < 1e-14:
                break
            v_new = Tvv / lam
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new

        lam = np.einsum('ijk,i,j,k', residual, v, v, v)
        eigenvalues.append(float(lam))
        eigenvectors.append(v.copy())

        # Deflate
        residual = residual - lam * np.einsum('i,j,k->ijk', v, v, v)

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": np.array(eigenvectors),
        "residual_norm": float(tensor_norm(residual)),
    }
