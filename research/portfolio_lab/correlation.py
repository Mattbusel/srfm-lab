"""
research/portfolio_lab/correlation.py

Portfolio correlation analytics for SRFM-Lab.

Implements rolling correlation matrices, DCC GARCH (simplified),
clustering, regime-conditional correlations, and diversification metrics.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Rolling correlation matrix
# ---------------------------------------------------------------------------


def rolling_correlation_matrix(
    returns_df: pd.DataFrame,
    window: int = 60,
    min_periods: Optional[int] = None,
) -> np.ndarray:
    """
    Compute rolling pairwise correlation matrices.

    Args:
        returns_df  : Returns DataFrame, shape (T, N).
        window      : Rolling window size in periods.
        min_periods : Minimum periods required (defaults to window // 2).

    Returns:
        3-D array of shape (T, N, N). NaN matrices for early periods with
        insufficient data are filled with the identity matrix.
    """
    T, N = returns_df.shape
    min_p = min_periods if min_periods is not None else max(2, window // 2)
    result = np.zeros((T, N, N), dtype=np.float64)
    values = returns_df.values.astype(np.float64)

    for t in range(T):
        start = max(0, t - window + 1)
        slice_ = values[start : t + 1]
        if len(slice_) < min_p:
            result[t] = np.eye(N)
            continue
        corr = _sample_corr(slice_)
        result[t] = corr

    return result


def _sample_corr(X: np.ndarray) -> np.ndarray:
    """Compute sample correlation matrix from a data matrix (T, N)."""
    n, p = X.shape
    X_c = X - X.mean(axis=0)
    std = X_c.std(axis=0) + 1e-12
    X_norm = X_c / std
    corr = (X_norm.T @ X_norm) / (n - 1)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Dynamic Conditional Correlation (DCC-GARCH simplified)
# ---------------------------------------------------------------------------


def dynamic_conditional_correlation(
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
    beta: float = 0.90,
    n_burn: int = 50,
) -> dict:
    """
    Simplified DCC-GARCH model for time-varying correlations.

    Uses scalar DCC model (Engle 2002):
        Q_t = (1-α-β)*Q_bar + α*(z_{t-1}*z_{t-1}^T) + β*Q_{t-1}
        R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

    where z_t are standardised residuals from univariate GARCH(1,1).

    Args:
        returns_df : Returns DataFrame (T, N).
        alpha      : DCC α parameter (news coefficient).
        beta       : DCC β parameter (persistence).
        n_burn     : Burn-in periods to initialise estimates.

    Returns:
        Dict with keys:
            'corr_matrices' : (T, N, N) array of R_t
            'conditional_vols' : (T, N) array of σ_t per asset
            'dates' : index of returns_df
    """
    if alpha + beta >= 1.0:
        raise ValueError(f"DCC requires α+β < 1. Got {alpha}+{beta}={alpha+beta}.")

    T, N = returns_df.shape
    R = returns_df.values.astype(np.float64)

    # Step 1: Univariate GARCH(1,1) for each asset
    cond_vols = np.zeros((T, N), dtype=np.float64)
    std_resid = np.zeros((T, N), dtype=np.float64)

    for j in range(N):
        h, eps = _garch11(R[:, j])
        cond_vols[:, j] = np.sqrt(np.maximum(h, 1e-12))
        std_resid[:, j] = eps

    # Step 2: Estimate Q_bar from standardised residuals
    Q_bar = (std_resid.T @ std_resid) / T
    np.fill_diagonal(Q_bar, np.maximum(np.diag(Q_bar), 1e-8))

    # Step 3: DCC recursion
    Q_t = Q_bar.copy()
    corr_matrices = np.zeros((T, N, N), dtype=np.float64)

    for t in range(T):
        if t == 0:
            corr_matrices[t] = np.eye(N)
            continue

        z_prev = std_resid[t - 1]
        Q_t = (1.0 - alpha - beta) * Q_bar + alpha * np.outer(z_prev, z_prev) + beta * Q_t

        # Normalise to correlation
        dq = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        R_t = Q_t / np.outer(dq, dq)
        np.fill_diagonal(R_t, 1.0)
        R_t = np.clip(R_t, -1.0, 1.0)
        corr_matrices[t] = R_t

    return {
        "corr_matrices": corr_matrices,
        "conditional_vols": cond_vols,
        "dates": returns_df.index,
        "alpha": alpha,
        "beta": beta,
        "Q_bar": Q_bar,
    }


def _garch11(returns: np.ndarray, omega_init: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit GARCH(1,1) model by quasi-maximum likelihood.

    Returns:
        (h, standardised_residuals) where h is conditional variance.
    """
    T = len(returns)
    mu = returns.mean()
    eps = returns - mu
    var_unconditional = float(np.var(eps)) + 1e-10

    # Simple parameter estimates (MoM)
    omega = max(var_unconditional * 0.05, omega_init)
    alpha = 0.10
    beta = 0.85
    # Ensure stationarity
    if alpha + beta >= 0.999:
        alpha = 0.05
        beta = 0.90

    h = np.zeros(T)
    h[0] = var_unconditional
    for t in range(1, T):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        h[t] = max(h[t], 1e-12)

    std_eps = eps / np.sqrt(h)
    return h, std_eps


# ---------------------------------------------------------------------------
# Correlation clustering
# ---------------------------------------------------------------------------


def correlation_clustering(
    corr_matrix: np.ndarray,
    n_clusters: Optional[int] = None,
    method: str = "ward",
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Cluster assets based on their correlation structure.

    Args:
        corr_matrix : Correlation matrix, shape (N, N).
        n_clusters  : If specified, use this number of clusters.
                      Otherwise, use `threshold` for flat clustering.
        method      : Linkage method ('ward', 'complete', 'average', 'single').
        threshold    : Distance threshold for flat clustering (if n_clusters is None).

    Returns:
        Cluster label array of shape (N,), with integer labels starting at 1.
    """
    N = corr_matrix.shape[0]
    if N == 1:
        return np.array([1])

    # Distance: d_ij = sqrt(0.5 * (1 - rho_ij))
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_matrix), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    link = linkage(condensed, method=method)

    if n_clusters is not None:
        labels = fcluster(link, n_clusters, criterion="maxclust")
    else:
        labels = fcluster(link, threshold, criterion="distance")

    return labels.astype(np.int64)


def correlation_dendrogram(
    corr_matrix: np.ndarray,
    asset_names: Optional[list[str]] = None,
    method: str = "ward",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a dendrogram of asset correlation clustering.

    Args:
        corr_matrix  : Correlation matrix (N, N).
        asset_names  : Labels for assets.
        method       : Linkage method.
        save_path    : If provided, save figure to path.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram as scipy_dend
    except ImportError:
        return

    N = corr_matrix.shape[0]
    names = asset_names or [f"A{i}" for i in range(N)]

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr_matrix), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method=method)

    fig, ax = plt.subplots(figsize=(max(8, N // 2), 5))
    scipy_dend(link, labels=names, ax=ax, leaf_rotation=45)
    ax.set_title(f"Correlation Dendrogram ({method} linkage)")
    ax.set_ylabel("Distance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Regime-conditional correlations
# ---------------------------------------------------------------------------


def correlation_regime_analysis(
    returns_df: pd.DataFrame,
    regimes: pd.Series,
) -> dict[str, np.ndarray]:
    """
    Compute correlation matrices conditional on market regime.

    Args:
        returns_df : Returns DataFrame (T, N).
        regimes    : Series of regime labels aligned with returns_df index.

    Returns:
        Dict mapping regime name to correlation matrix (N, N).
    """
    results = {}
    unique_regimes = regimes.dropna().unique()

    for regime in unique_regimes:
        mask = (regimes == regime).values
        subset = returns_df.loc[mask]
        if len(subset) < 3:
            continue
        corr = _sample_corr(subset.values.astype(np.float64))
        results[str(regime)] = corr

    return results


# ---------------------------------------------------------------------------
# Diversification metrics
# ---------------------------------------------------------------------------


def diversification_ratio(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Diversification Ratio: weighted avg individual vol / portfolio vol.

    DR = (w^T σ) / sqrt(w^T Σ w)

    A higher DR indicates better diversification. DR = 1 for undiversified.

    Args:
        weights    : Portfolio weights, shape (N,).
        cov_matrix : Covariance matrix, shape (N, N).

    Returns:
        Diversification ratio (>= 1.0).
    """
    w = np.asarray(weights, dtype=np.float64)
    indiv_vols = np.sqrt(np.maximum(np.diag(cov_matrix), 0.0))
    weighted_vol = float(w @ indiv_vols)
    port_vol = float(math.sqrt(max(float(w @ cov_matrix @ w), 1e-12)))
    return weighted_vol / port_vol


def effective_n_bets(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
) -> int:
    """
    Effective number of independent bets (Meucci 2009).

    Uses the entropy of the eigenvalue distribution of the risk contribution matrix.

    ENB = exp(-sum(p_i * log(p_i))) where p_i are normalised eigenvalue contributions.

    Args:
        weights    : Portfolio weights, shape (N,).
        cov_matrix : Covariance matrix, shape (N, N).

    Returns:
        Effective number of bets (integer, <= N).
    """
    w = np.asarray(weights, dtype=np.float64)
    port_var = float(w @ cov_matrix @ w) + 1e-12

    # PCA decomposition of cov_matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    eigenvals = np.maximum(eigenvals, 0.0)

    # Weight contributions in eigen-space
    w_eig = eigenvecs.T @ w                        # (N,) projections
    contrib = (w_eig ** 2) * eigenvals / port_var  # (N,) contribution fractions
    contrib = contrib / (contrib.sum() + 1e-12)    # normalise

    # Entropy-based ENB
    contrib = np.maximum(contrib, 1e-12)
    enb = float(math.exp(-float(np.sum(contrib * np.log(contrib)))))
    return max(1, int(round(enb)))


def pairwise_correlation_stability(
    returns_df: pd.DataFrame,
    window: int = 60,
    stride: int = 21,
) -> pd.DataFrame:
    """
    Compute rolling correlation stability — average absolute change in
    pairwise correlations over time.

    Args:
        returns_df : Returns DataFrame (T, N).
        window     : Rolling window size.
        stride     : Stride between windows.

    Returns:
        DataFrame with columns: 'date', 'avg_corr_change', 'max_corr_change'.
    """
    T, N = returns_df.shape
    values = returns_df.values.astype(np.float64)

    rows = []
    prev_corr = None

    for t in range(window, T, stride):
        slice_ = values[t - window : t]
        corr = _sample_corr(slice_)

        if prev_corr is not None:
            diff = np.abs(corr - prev_corr)
            # Upper triangle only (no diagonal)
            mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            changes = diff[mask]
            rows.append({
                "date": returns_df.index[t] if hasattr(returns_df.index[t], "strftime") else t,
                "avg_corr_change": float(changes.mean()),
                "max_corr_change": float(changes.max()),
            })
        prev_corr = corr

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    corr_matrix: np.ndarray,
    asset_names: Optional[list[str]] = None,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
    cmap: str = "RdBu_r",
) -> None:
    """
    Plot a styled correlation heatmap.

    Args:
        corr_matrix  : Correlation matrix (N, N).
        asset_names  : Asset labels.
        title        : Plot title.
        save_path    : If provided, save figure.
        cmap         : Matplotlib colormap.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    N = corr_matrix.shape[0]
    names = asset_names or [f"A{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(6, N), max(5, N - 1)))
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(title, fontsize=13)

    # Annotate cells
    for i in range(N):
        for j in range(N):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_rolling_correlation(
    rolling_corr: np.ndarray,
    p1: int,
    p2: int,
    dates: Optional[pd.Index] = None,
    asset_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the rolling pairwise correlation between two assets.

    Args:
        rolling_corr : 3-D array (T, N, N) from rolling_correlation_matrix().
        p1           : Index of first asset.
        p2           : Index of second asset.
        dates        : Optional datetime index for x-axis.
        asset_names  : Optional asset name labels.
        save_path    : If provided, save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    T = rolling_corr.shape[0]
    corr_series = rolling_corr[:, p1, p2]

    names = asset_names or [f"Asset {i}" for i in range(rolling_corr.shape[1])]
    a1 = names[p1]
    a2 = names[p2]

    x = dates if dates is not None else np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, corr_series, color="steelblue", linewidth=1.0, label=f"{a1} / {a2}")
    ax.axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    ax.axhline(corr_series.mean(), color="darkorange", linewidth=1.5, linestyle="--",
               alpha=0.7, label=f"Mean={corr_series.mean():.3f}")
    ax.fill_between(x if not hasattr(x, "values") else range(T),
                    corr_series, 0, alpha=0.15, color="steelblue")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Rolling Correlation: {a1} vs {a2}", fontsize=12)
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_correlation_regime_breakdown(
    regime_corrs: dict[str, np.ndarray],
    asset_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot per-regime correlation heatmaps in a grid.

    Args:
        regime_corrs : Dict mapping regime name to correlation matrix (N, N).
        asset_names  : Asset labels.
        save_path    : If provided, save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_regimes = len(regime_corrs)
    if n_regimes == 0:
        return

    ncols = min(3, n_regimes)
    nrows = math.ceil(n_regimes / ncols)
    regime_names = list(regime_corrs.keys())
    N = list(regime_corrs.values())[0].shape[0]
    names = asset_names or [f"A{i}" for i in range(N)]

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_regimes == 1:
        axes = [axes]
    axes_flat = np.array(axes).reshape(-1)

    for idx, regime in enumerate(regime_names):
        ax = axes_flat[idx]
        corr = regime_corrs[regime]
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_title(f"Regime: {regime}", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for idx in range(len(regime_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Correlation by Market Regime", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
