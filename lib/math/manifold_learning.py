"""
Manifold learning methods for financial state space analysis.

Implements Isomap, LLE, Laplacian Eigenmaps, diffusion maps, t-SNE,
UMAP-like embedding, geodesic distances on correlation manifold,
persistent homology, and manifold-based anomaly detection.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import eigh, svd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# K-nearest neighbors graph utilities
# ---------------------------------------------------------------------------

def knn_graph(X: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    """
    Build symmetric k-nearest-neighbor adjacency matrix.

    Parameters
    ----------
    X : (n, d) data matrix
    k : number of neighbors
    metric : distance metric

    Returns
    -------
    (n, n) symmetric distance matrix with non-neighbors set to 0
    """
    n = X.shape[0]
    D = squareform(pdist(X, metric=metric))
    adj = np.zeros((n, n))

    for i in range(n):
        idx = np.argsort(D[i])
        neighbors = idx[1:k + 1]  # exclude self
        for j in neighbors:
            adj[i, j] = D[i, j]
            adj[j, i] = D[j, i]

    return adj


def epsilon_graph(X: np.ndarray, epsilon: float, metric: str = "euclidean") -> np.ndarray:
    """Build epsilon-neighborhood graph."""
    D = squareform(pdist(X, metric=metric))
    adj = np.where(D <= epsilon, D, 0.0)
    np.fill_diagonal(adj, 0.0)
    return adj


# ---------------------------------------------------------------------------
# Isomap
# ---------------------------------------------------------------------------

def isomap(
    X: np.ndarray,
    n_components: int = 2,
    k: int = 10,
) -> dict:
    """
    Isomap: geodesic distance estimation + classical MDS.

    Parameters
    ----------
    X : (n, d) data matrix
    n_components : embedding dimension
    k : number of nearest neighbors

    Returns
    -------
    dict with embedding, geodesic_distances, stress
    """
    n = X.shape[0]

    # Build kNN graph
    adj = knn_graph(X, k)

    # Replace zeros with inf for shortest path (except diagonal)
    graph = np.where(adj > 0, adj, np.inf)
    np.fill_diagonal(graph, 0.0)

    # Geodesic distances via Dijkstra
    geo_dist = shortest_path(graph, method="D")

    # Check connectivity
    if np.any(np.isinf(geo_dist)):
        # Fill disconnected with max finite distance
        max_finite = np.max(geo_dist[np.isfinite(geo_dist)])
        geo_dist[np.isinf(geo_dist)] = max_finite * 2

    # Classical MDS on geodesic distances
    embedding = _classical_mds(geo_dist, n_components)

    # Stress measure
    embed_dist = squareform(pdist(embedding))
    stress = np.sqrt(np.sum((geo_dist - embed_dist)**2) / max(np.sum(geo_dist**2), 1e-16))

    return {
        "embedding": embedding,
        "geodesic_distances": geo_dist,
        "stress": float(stress),
    }


def _classical_mds(D: np.ndarray, n_components: int) -> np.ndarray:
    """Classical (metric) MDS from a distance matrix."""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # centering matrix
    B = -0.5 * H @ (D**2) @ H  # double centering

    eigenvalues, eigenvectors = eigh(B)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top components
    lam = np.maximum(eigenvalues[:n_components], 0)
    embedding = eigenvectors[:, :n_components] * np.sqrt(lam)[np.newaxis, :]

    return embedding


# ---------------------------------------------------------------------------
# Locally Linear Embedding (LLE)
# ---------------------------------------------------------------------------

def lle(
    X: np.ndarray,
    n_components: int = 2,
    k: int = 10,
    reg: float = 1e-3,
) -> dict:
    """
    Locally Linear Embedding.

    Step 1: Find reconstruction weights W such that X_i ≈ sum_j W_ij X_j (neighbors only).
    Step 2: Find Y minimizing sum_i |Y_i - sum_j W_ij Y_j|^2.

    Parameters
    ----------
    X : (n, d) data
    n_components : embedding dimension
    k : number of neighbors
    reg : regularization for weight computation

    Returns
    -------
    dict with embedding, reconstruction_error
    """
    n, d = X.shape
    D = squareform(pdist(X))

    # Step 1: compute weights
    W = np.zeros((n, n))
    recon_error = 0.0

    for i in range(n):
        neighbors = np.argsort(D[i])[1:k + 1]
        Z = X[neighbors] - X[i]  # (k, d)
        C = Z @ Z.T  # local covariance
        C += reg * np.eye(k) * np.trace(C)  # regularize

        try:
            w = np.linalg.solve(C, np.ones(k))
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(C, np.ones(k), rcond=None)[0]

        w /= np.sum(w)
        W[i, neighbors] = w
        recon_error += np.sum((X[i] - w @ X[neighbors])**2)

    # Step 2: eigenvalue problem
    M = (np.eye(n) - W).T @ (np.eye(n) - W)

    eigenvalues, eigenvectors = eigh(M)

    # Skip the smallest eigenvalue (should be ~0), take next n_components
    embedding = eigenvectors[:, 1:n_components + 1]

    return {
        "embedding": embedding,
        "reconstruction_error": float(recon_error),
        "eigenvalues": eigenvalues[:n_components + 2],
    }


# ---------------------------------------------------------------------------
# Laplacian Eigenmaps
# ---------------------------------------------------------------------------

def laplacian_eigenmaps(
    X: np.ndarray,
    n_components: int = 2,
    k: int = 10,
    sigma: float = None,
) -> dict:
    """
    Laplacian Eigenmaps: spectral embedding via graph Laplacian.

    Parameters
    ----------
    X : (n, d) data
    n_components : embedding dimension
    k : number of neighbors for graph construction
    sigma : heat kernel bandwidth (default: median neighbor distance)

    Returns
    -------
    dict with embedding, eigenvalues
    """
    n = X.shape[0]
    adj = knn_graph(X, k)

    # Determine sigma
    if sigma is None:
        nonzero = adj[adj > 0]
        sigma = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

    # Heat kernel weights
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                W[i, j] = np.exp(-adj[i, j]**2 / (2 * sigma**2))

    # Symmetrize
    W = (W + W.T) / 2

    # Graph Laplacian
    D_diag = np.sum(W, axis=1)
    D = np.diag(D_diag)
    L = D - W

    # Generalized eigenvalue problem: L y = lambda D y
    eigenvalues, eigenvectors = eigh(L, D + 1e-10 * np.eye(n))

    # Skip first (trivial) eigenvector
    embedding = eigenvectors[:, 1:n_components + 1]

    return {
        "embedding": embedding,
        "eigenvalues": eigenvalues[:n_components + 2],
        "sigma": sigma,
    }


# ---------------------------------------------------------------------------
# Diffusion Maps
# ---------------------------------------------------------------------------

def diffusion_maps(
    X: np.ndarray,
    n_components: int = 2,
    sigma: float = None,
    alpha: float = 0.5,
    t: int = 1,
) -> dict:
    """
    Diffusion maps via Markov chain on data.

    Parameters
    ----------
    X : (n, d) data
    n_components : embedding dimension
    sigma : kernel bandwidth (default: median pairwise distance)
    alpha : normalization parameter (0 = graph Laplacian, 0.5 = Fokker-Planck, 1 = Laplace-Beltrami)
    t : diffusion time

    Returns
    -------
    dict with embedding, eigenvalues, diffusion_distances
    """
    n = X.shape[0]
    D_pw = squareform(pdist(X))

    if sigma is None:
        sigma = float(np.median(D_pw[D_pw > 0]))

    # Kernel matrix
    K = np.exp(-D_pw**2 / (2 * sigma**2))

    # Alpha normalization
    q = np.sum(K, axis=1)
    K_alpha = K / np.outer(q**alpha, q**alpha)

    # Row-normalize to get Markov transition matrix
    row_sums = np.sum(K_alpha, axis=1)
    P = K_alpha / row_sums[:, np.newaxis]

    # Diffusion at time t
    P_t = np.linalg.matrix_power(P, t)

    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(P_t + P_t.T)  # symmetrize for numerical stability

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Diffusion map embedding (skip first trivial eigenvector)
    lam = eigenvalues[1:n_components + 1]
    psi = eigenvectors[:, 1:n_components + 1]
    embedding = psi * (lam**t)[np.newaxis, :]

    # Diffusion distances
    diff_dist = squareform(pdist(embedding))

    return {
        "embedding": embedding,
        "eigenvalues": eigenvalues[:n_components + 2],
        "diffusion_distances": diff_dist,
        "sigma": sigma,
    }


# ---------------------------------------------------------------------------
# t-SNE (simplified gradient descent)
# ---------------------------------------------------------------------------

def tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    momentum: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    t-SNE via direct gradient descent (simplified, no Barnes-Hut).

    Parameters
    ----------
    X : (n, d) high-dimensional data
    n_components : embedding dimension
    perplexity : controls effective number of neighbors
    learning_rate : gradient descent step size
    n_iter : number of iterations
    momentum : momentum factor
    seed : random seed

    Returns
    -------
    dict with embedding, kl_divergence_history
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # Compute pairwise affinities P
    D = squareform(pdist(X, "sqeuclidean"))
    P = _compute_joint_probabilities(D, perplexity)

    # Initialize embedding
    Y = rng.standard_normal((n, n_components)) * 1e-4
    dY = np.zeros_like(Y)
    gains = np.ones_like(Y)

    kl_history = []

    for it in range(n_iter):
        # Compute Q (Student-t)
        D_low = squareform(pdist(Y, "sqeuclidean"))
        num = 1.0 / (1.0 + D_low)
        np.fill_diagonal(num, 0.0)
        Q = num / max(np.sum(num), 1e-16)
        Q = np.maximum(Q, 1e-12)

        # KL divergence
        P_safe = np.maximum(P, 1e-12)
        kl = np.sum(P_safe * np.log(P_safe / Q))
        kl_history.append(float(kl))

        # Gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4.0 * np.sum((PQ_diff[i] * num[i])[:, np.newaxis] * diff, axis=0)

        # Adaptive gains
        gains = np.where(
            np.sign(grad) != np.sign(dY),
            gains + 0.2,
            gains * 0.8
        )
        gains = np.maximum(gains, 0.01)

        dY = momentum * dY - learning_rate * gains * grad
        Y += dY

        # Center
        Y -= np.mean(Y, axis=0)

    return {
        "embedding": Y,
        "kl_divergence_history": kl_history,
    }


def _compute_joint_probabilities(D_sq: np.ndarray, perplexity: float) -> np.ndarray:
    """Compute symmetric joint probabilities from squared distances."""
    n = D_sq.shape[0]
    P = np.zeros((n, n))
    target_entropy = np.log(perplexity)

    for i in range(n):
        # Binary search for sigma_i
        lo, hi = 1e-10, 1e4
        for _ in range(50):
            sigma = (lo + hi) / 2
            Pi = np.exp(-D_sq[i] / (2 * sigma**2))
            Pi[i] = 0.0
            sum_Pi = max(np.sum(Pi), 1e-16)
            Pi /= sum_Pi

            entropy = -np.sum(Pi[Pi > 0] * np.log(Pi[Pi > 0]))
            if entropy > target_entropy:
                hi = sigma
            else:
                lo = sigma

        P[i] = Pi

    # Symmetrize
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    return P


# ---------------------------------------------------------------------------
# UMAP-like embedding (simplified fuzzy simplicial set + CE optimization)
# ---------------------------------------------------------------------------

def umap_embed(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    learning_rate: float = 1.0,
    n_epochs: int = 200,
    seed: int = 42,
) -> dict:
    """
    Simplified UMAP-like embedding.

    1. Build fuzzy simplicial set (weighted kNN graph with exponential decay).
    2. Initialize with spectral embedding.
    3. Optimize cross-entropy between high-D and low-D fuzzy sets.

    Parameters
    ----------
    X : (n, d) data
    n_components : embedding dimension
    n_neighbors : effective neighborhood size
    min_dist : minimum distance in low-D
    learning_rate : SGD step size
    n_epochs : optimization epochs
    seed : random seed

    Returns
    -------
    dict with embedding, graph_weights
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # Step 1: Fuzzy simplicial set
    D = squareform(pdist(X))
    W = np.zeros((n, n))

    for i in range(n):
        sorted_idx = np.argsort(D[i])
        knn = sorted_idx[1:n_neighbors + 1]
        rho = D[i, knn[0]]  # distance to nearest neighbor
        # Find sigma via binary search (local connectivity)
        target = np.log2(n_neighbors)
        lo, hi = 1e-6, 100.0
        for _ in range(50):
            sigma = (lo + hi) / 2
            vals = np.exp(-(np.maximum(D[i, knn] - rho, 0)) / sigma)
            if np.sum(vals) > target:
                hi = sigma
            else:
                lo = sigma

        for j in knn:
            W[i, j] = np.exp(-max(D[i, j] - rho, 0) / max(sigma, 1e-10))

    # Symmetrize: W = W + W^T - W * W^T
    W = W + W.T - W * W.T
    W = np.clip(W, 0, 1)

    # Step 2: Initialize with Laplacian eigenmaps
    D_diag = np.sum(W, axis=1)
    L = np.diag(D_diag) - W
    eigenvalues, eigenvectors = eigh(L)
    Y = eigenvectors[:, 1:n_components + 1] * 10  # scale up

    # Step 3: Optimize cross-entropy
    # Parameters for the smooth approximation in low-D
    a, b = _find_ab_params(min_dist)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0.01:
                edges.append((i, j, W[i, j]))

    for epoch in range(n_epochs):
        lr = learning_rate * (1 - epoch / n_epochs)

        for i, j, w_ij in edges:
            diff = Y[i] - Y[j]
            dist_sq = np.sum(diff**2)
            # Attractive force
            grad_coeff = -2 * a * b * dist_sq**(b - 1) / (1 + a * dist_sq**b + 1e-10)
            grad = w_ij * grad_coeff * diff

            Y[i] += lr * grad
            Y[j] -= lr * grad

            # Repulsive: sample a random negative
            k = rng.integers(n)
            while k == i:
                k = rng.integers(n)
            diff_neg = Y[i] - Y[k]
            dist_sq_neg = np.sum(diff_neg**2) + 0.001
            grad_neg = 2 * b / ((0.001 + dist_sq_neg) * (1 + a * dist_sq_neg**b + 1e-10))
            Y[i] += lr * (1 - w_ij) * grad_neg * diff_neg

    return {
        "embedding": Y,
        "graph_weights": W,
    }


def _find_ab_params(min_dist: float, spread: float = 1.0) -> tuple:
    """Find a, b parameters for UMAP's smooth distance approximation."""
    from scipy.optimize import curve_fit

    x = np.linspace(0, spread * 3, 300)

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x**(2 * b))

    y = np.where(x <= min_dist, 1.0, np.exp(-(x - min_dist) / spread))

    try:
        (a, b), _ = curve_fit(curve, x, y, p0=(1.0, 1.0), maxfev=5000)
    except Exception:
        a, b = 1.0, 1.0

    return a, b


# ---------------------------------------------------------------------------
# Geodesic distance on correlation manifold
# ---------------------------------------------------------------------------

def correlation_geodesic(C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Geodesic distance between two correlation matrices on the SPD manifold.

    d(C1, C2) = || log(C1^{-1/2} C2 C1^{-1/2}) ||_F

    (Affine-invariant Riemannian metric)
    """
    # C1^{-1/2}
    eigenvalues, eigenvectors = eigh(C1)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    C1_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    M = C1_inv_sqrt @ C2 @ C1_inv_sqrt

    # Ensure symmetry
    M = (M + M.T) / 2

    eigenvalues_M = np.maximum(eigh(M, eigvals_only=True), 1e-10)
    log_eigenvalues = np.log(eigenvalues_M)

    return float(np.sqrt(np.sum(log_eigenvalues**2)))


def correlation_manifold_distances(corr_matrices: list) -> np.ndarray:
    """
    Compute pairwise geodesic distances between a list of correlation matrices.
    """
    n = len(corr_matrices)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = correlation_geodesic(corr_matrices[i], corr_matrices[j])
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# Market regime visualization
# ---------------------------------------------------------------------------

def regime_visualization(
    returns: np.ndarray,
    window: int = 60,
    step: int = 5,
    n_components: int = 2,
    method: str = "diffusion",
) -> dict:
    """
    Embed rolling correlation matrices into low-D for regime visualization.

    Parameters
    ----------
    returns : (T, n_assets) return matrix
    window : rolling window size
    step : step between windows
    n_components : embedding dimension
    method : 'diffusion', 'isomap', or 'laplacian'

    Returns
    -------
    dict with embedding, window_centers
    """
    T, n_assets = returns.shape
    windows = list(range(0, T - window + 1, step))
    n_windows = len(windows)

    # Compute features from each window (vectorized upper-triangle of correlation)
    idx_upper = np.triu_indices(n_assets, k=1)
    features = np.zeros((n_windows, len(idx_upper[0])))

    for w_idx, start in enumerate(windows):
        R = returns[start:start + window]
        corr = np.corrcoef(R.T)
        features[w_idx] = corr[idx_upper]

    # Embed
    if method == "diffusion":
        result = diffusion_maps(features, n_components=n_components)
    elif method == "isomap":
        result = isomap(features, n_components=n_components, k=min(10, n_windows - 1))
    elif method == "laplacian":
        result = laplacian_eigenmaps(features, n_components=n_components, k=min(10, n_windows - 1))
    else:
        result = diffusion_maps(features, n_components=n_components)

    window_centers = [start + window // 2 for start in windows]

    return {
        "embedding": result["embedding"],
        "window_centers": window_centers,
        "n_windows": n_windows,
    }


# ---------------------------------------------------------------------------
# Persistent homology (Vietoris-Rips, simplified)
# ---------------------------------------------------------------------------

def vietoris_rips_persistence(
    X: np.ndarray,
    max_dim: int = 1,
    max_radius: float = None,
    n_steps: int = 100,
) -> dict:
    """
    Simplified persistent homology via Vietoris-Rips filtration.

    Computes Betti numbers (b0, b1) as a function of filtration radius.

    Parameters
    ----------
    X : (n, d) point cloud
    max_dim : maximum homology dimension to track (0 or 1)
    max_radius : maximum filtration radius
    n_steps : number of filtration steps

    Returns
    -------
    dict with betti_0 (connected components), betti_1 (loops), radii,
    persistence_pairs_0
    """
    n = X.shape[0]
    D = squareform(pdist(X))

    if max_radius is None:
        max_radius = np.max(D) * 0.5

    radii = np.linspace(0, max_radius, n_steps)
    betti_0 = np.zeros(n_steps, dtype=int)
    betti_1 = np.zeros(n_steps, dtype=int)

    # Track connected components via union-find
    persistence_0 = []  # (birth, death) pairs for H0

    for step, r in enumerate(radii):
        # Adjacency at radius r
        adj = D <= r
        np.fill_diagonal(adj, False)

        # Connected components via BFS
        visited = np.zeros(n, dtype=bool)
        components = []
        for i in range(n):
            if not visited[i]:
                component = []
                stack = [i]
                while stack:
                    node = stack.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    component.append(node)
                    for j in range(n):
                        if adj[node, j] and not visited[j]:
                            stack.append(j)
                components.append(component)

        betti_0[step] = len(components)

        # Betti-1 estimate: edges - vertices + components (Euler characteristic)
        if max_dim >= 1:
            n_edges = np.sum(adj) // 2
            # Count triangles
            n_triangles = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        for k in range(j + 1, n):
                            if adj[i, k] and adj[j, k]:
                                n_triangles += 1

            # Euler characteristic: V - E + F = chi
            # For simplicial complex: b0 - b1 + b2 = chi
            # Approximate b1 = E - V + b0 - triangles (rough)
            chi = n - n_edges + n_triangles
            betti_1[step] = max(0, n_edges - n + len(components) - n_triangles)

    # Persistence pairs for H0
    # Track when components merge
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    # Sort edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((D[i, j], i, j))
    edges.sort()

    birth_times = {i: 0.0 for i in range(n)}
    persistence_pairs = []

    for dist, i, j in edges:
        pi, pj = find(i), find(j)
        if pi != pj:
            # Younger component dies
            younger = max(pi, pj, key=lambda x: birth_times.get(x, 0))
            older = pi if younger == pj else pj
            persistence_pairs.append((birth_times[younger], dist))
            union(pi, pj)

    return {
        "radii": radii,
        "betti_0": betti_0,
        "betti_1": betti_1,
        "persistence_pairs_0": persistence_pairs,
        "n_points": n,
    }


def persistence_diagram_distance(
    pairs1: list,
    pairs2: list,
) -> float:
    """
    Bottleneck-like distance between two persistence diagrams.
    Simplified: uses Wasserstein-1 with diagonal matching.
    """
    if not pairs1 and not pairs2:
        return 0.0

    # Convert to arrays
    p1 = np.array(pairs1) if pairs1 else np.zeros((0, 2))
    p2 = np.array(pairs2) if pairs2 else np.zeros((0, 2))

    # Add diagonal projections
    diag1 = np.column_stack([
        (p1[:, 0] + p1[:, 1]) / 2,
        (p1[:, 0] + p1[:, 1]) / 2
    ]) if len(p1) > 0 else np.zeros((0, 2))

    diag2 = np.column_stack([
        (p2[:, 0] + p2[:, 1]) / 2,
        (p2[:, 0] + p2[:, 1]) / 2
    ]) if len(p2) > 0 else np.zeros((0, 2))

    # Augment: match points in p1 to either p2 or diagonal, and vice versa
    all1 = np.vstack([p1, diag2]) if len(diag2) > 0 else p1
    all2 = np.vstack([p2, diag1]) if len(diag1) > 0 else p2

    if len(all1) == 0 or len(all2) == 0:
        return 0.0

    # Greedy matching (approximation)
    cost_matrix = cdist(all1, all2, metric="chebyshev")
    n1, n2 = cost_matrix.shape
    total_cost = 0.0
    used = set()
    for i in range(min(n1, n2)):
        min_val = np.inf
        min_j = 0
        for j in range(n2):
            if j not in used and cost_matrix[i, j] < min_val:
                min_val = cost_matrix[i, j]
                min_j = j
        total_cost += min_val
        used.add(min_j)

    return float(total_cost)


# ---------------------------------------------------------------------------
# Manifold-based anomaly detection
# ---------------------------------------------------------------------------

def manifold_anomaly_detection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 2,
    k: int = 10,
    method: str = "isomap",
) -> dict:
    """
    Anomaly detection based on distance to learned manifold.

    1. Learn manifold embedding from X_train.
    2. For each test point, compute reconstruction error / distance to manifold.
    3. Points far from the manifold are anomalous.

    Parameters
    ----------
    X_train : (n_train, d) training data
    X_test : (n_test, d) test data
    n_components : manifold dimension
    k : nearest neighbors
    method : 'isomap', 'lle', or 'knn_distance'

    Returns
    -------
    dict with anomaly_scores, threshold_95, is_anomaly
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if method == "knn_distance":
        # Simple: anomaly score = mean distance to k nearest training neighbors
        D = cdist(X_test, X_train)
        knn_dists = np.sort(D, axis=1)[:, :k]
        scores = np.mean(knn_dists, axis=1)
    else:
        # Embed training data
        if method == "isomap":
            result = isomap(X_train, n_components=n_components, k=k)
        elif method == "lle":
            result = lle(X_train, n_components=n_components, k=k)
        else:
            result = isomap(X_train, n_components=n_components, k=k)

        embedding = result["embedding"]

        # For test points: find kNN in training set, reconstruct in embedding space,
        # measure discrepancy
        D = cdist(X_test, X_train)
        scores = np.zeros(n_test)

        for i in range(n_test):
            knn_idx = np.argsort(D[i])[:k]
            knn_orig = X_train[knn_idx]
            knn_embed = embedding[knn_idx]

            # Reconstruction weights (LLE-style)
            Z = knn_orig - X_test[i]
            C = Z @ Z.T + 1e-3 * np.eye(k) * np.trace(Z @ Z.T)
            try:
                w = np.linalg.solve(C, np.ones(k))
            except np.linalg.LinAlgError:
                w = np.ones(k)
            w /= np.sum(w)

            # Reconstruct test point in embedding space
            test_embed = w @ knn_embed

            # Score: distance from reconstructed point to nearest training point in embed space
            embed_dists = np.linalg.norm(embedding - test_embed, axis=1)
            scores[i] = np.min(embed_dists)

    # Threshold at 95th percentile of training scores
    if method != "knn_distance":
        # Compute scores for training data too
        train_D = squareform(pdist(X_train))
        train_scores = np.zeros(n_train)
        for i in range(n_train):
            knn_idx = np.argsort(train_D[i])[1:k + 1]
            train_scores[i] = np.mean(train_D[i, knn_idx])
        threshold = np.percentile(train_scores, 95)
    else:
        train_D = squareform(pdist(X_train))
        knn_dists_train = np.sort(train_D, axis=1)[:, 1:k + 1]
        train_scores = np.mean(knn_dists_train, axis=1)
        threshold = np.percentile(train_scores, 95)

    return {
        "anomaly_scores": scores,
        "threshold_95": float(threshold),
        "is_anomaly": scores > threshold,
        "anomaly_fraction": float(np.mean(scores > threshold)),
    }
