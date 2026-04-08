"""
Sparse methods for high-dimensional finance.

LASSO, Elastic Net, Graphical LASSO, Sparse PCA, Compressed Sensing,
LARS, Group LASSO, Debiased LASSO, Sparse Portfolio, Sparse Factor Model.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Soft-thresholding operator S(x, lam) = sign(x) * max(|x| - lam, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _hard_threshold(x: np.ndarray, k: int) -> np.ndarray:
    """Hard-thresholding: keep top-k entries by absolute value, zero rest."""
    out = np.zeros_like(x)
    if k >= len(x):
        return x.copy()
    idx = np.argsort(np.abs(x))[-k:]
    out[idx] = x[idx]
    return out


def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale columns to unit variance."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma


# ---------------------------------------------------------------------------
# 1. LASSO via Coordinate Descent (full regularization path)
# ---------------------------------------------------------------------------

class LassoPath:
    """LASSO regression via coordinate descent with full lambda path."""

    def __init__(self, n_lambdas: int = 100, eps: float = 1e-3,
                 max_iter: int = 1000, tol: float = 1e-6):
        self.n_lambdas = n_lambdas
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.coef_path_: Optional[np.ndarray] = None
        self.lambdas_: Optional[np.ndarray] = None
        self.intercepts_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoPath":
        n, p = X.shape
        Xs, mu_x, sigma_x = _standardize(X)
        y_c = y - y.mean()
        lambda_max = np.max(np.abs(Xs.T @ y_c)) / n
        lambdas = np.logspace(np.log10(lambda_max),
                              np.log10(lambda_max * self.eps),
                              self.n_lambdas)
        coef_path = np.zeros((self.n_lambdas, p))
        beta = np.zeros(p)
        for li, lam in enumerate(lambdas):
            beta = self._coordinate_descent(Xs, y_c, beta.copy(), lam, n)
            coef_path[li] = beta
        self.coef_path_ = coef_path / sigma_x[np.newaxis, :]
        self.intercepts_ = y.mean() - (coef_path / sigma_x) @ mu_x
        self.lambdas_ = lambdas
        return self

    def _coordinate_descent(self, X: np.ndarray, y: np.ndarray,
                            beta: np.ndarray, lam: float, n: int) -> np.ndarray:
        p = X.shape[1]
        r = y - X @ beta
        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r += X[:, j] * beta[j]
                rho = X[:, j] @ r / n
                beta[j] = _soft_threshold(np.array([rho]), lam)[0]
                r -= X[:, j] * beta[j]
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
        return beta

    def predict(self, X: np.ndarray, lam_idx: int = -1) -> np.ndarray:
        return X @ self.coef_path_[lam_idx] + self.intercepts_[lam_idx]

    def select_by_bic(self, X: np.ndarray, y: np.ndarray) -> int:
        n = X.shape[0]
        best_bic = np.inf
        best_idx = 0
        for i in range(self.n_lambdas):
            resid = y - self.predict(X, i)
            rss = np.sum(resid ** 2)
            k = np.sum(np.abs(self.coef_path_[i]) > 1e-10)
            bic = n * np.log(rss / n + 1e-30) + k * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_idx = i
        return best_idx


# ---------------------------------------------------------------------------
# 2. Elastic Net via Coordinate Descent
# ---------------------------------------------------------------------------

class ElasticNet:
    """Elastic Net: alpha * L1 + (1 - alpha) * L2 penalty."""

    def __init__(self, lam: float = 0.1, alpha: float = 0.5,
                 max_iter: int = 1000, tol: float = 1e-6):
        self.lam = lam
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        n, p = X.shape
        Xs, mu_x, sigma_x = _standardize(X)
        y_c = y - y.mean()
        beta = np.zeros(p)
        r = y_c - Xs @ beta
        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r += Xs[:, j] * beta[j]
                rho = Xs[:, j] @ r / n
                beta[j] = _soft_threshold(np.array([rho]), self.lam * self.alpha)[0]
                beta[j] /= (1.0 + self.lam * (1.0 - self.alpha))
                r -= Xs[:, j] * beta[j]
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
        self.coef_ = beta / sigma_x
        self.intercept_ = y.mean() - self.coef_ @ mu_x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


# ---------------------------------------------------------------------------
# 3. Graphical LASSO (ADMM for sparse precision matrix)
# ---------------------------------------------------------------------------

class GraphicalLasso:
    """Estimate sparse precision matrix via ADMM."""

    def __init__(self, lam: float = 0.1, rho: float = 1.0,
                 max_iter: int = 200, tol: float = 1e-6):
        self.lam = lam
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.precision_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "GraphicalLasso":
        n, p = X.shape
        S = np.cov(X, rowvar=False, bias=True)
        Theta = np.eye(p)
        Z = np.eye(p)
        U = np.zeros((p, p))
        for _ in range(self.max_iter):
            Theta_old = Theta.copy()
            # Theta update: minimize -log det Theta + trace(S Theta) + (rho/2)||Theta - Z + U||^2
            M = self.rho * (Z - U) - S
            eigvals, eigvecs = np.linalg.eigh(M)
            d = (eigvals + np.sqrt(eigvals ** 2 + 4 * self.rho)) / (2 * self.rho)
            Theta = eigvecs @ np.diag(d) @ eigvecs.T
            # Z update: soft threshold
            Z_arg = Theta + U
            Z = _soft_threshold(Z_arg, self.lam / self.rho)
            np.fill_diagonal(Z, np.diag(Z_arg))
            # U update
            U = U + Theta - Z
            if np.linalg.norm(Theta - Theta_old) < self.tol:
                break
        self.precision_ = Theta
        try:
            self.covariance_ = np.linalg.inv(Theta)
        except np.linalg.LinAlgError:
            self.covariance_ = np.linalg.pinv(Theta)
        return self

    def partial_correlations(self) -> np.ndarray:
        P = self.precision_.copy()
        d = np.sqrt(np.diag(P))
        d[d == 0] = 1.0
        pcorr = -P / np.outer(d, d)
        np.fill_diagonal(pcorr, 1.0)
        return pcorr


# ---------------------------------------------------------------------------
# 4. Sparse PCA via Iterative Thresholding
# ---------------------------------------------------------------------------

class SparsePCA:
    """Sparse PCA with L1 penalty on loadings via iterative thresholding."""

    def __init__(self, n_components: int = 3, lam: float = 0.1,
                 max_iter: int = 500, tol: float = 1e-6):
        self.n_components = n_components
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "SparsePCA":
        n, p = X.shape
        Xc = X - X.mean(axis=0)
        S = Xc.T @ Xc / n
        components = []
        variances = []
        residual = S.copy()
        for _ in range(self.n_components):
            v = self._extract_one(residual, p)
            components.append(v)
            var = v @ S @ v
            variances.append(var)
            residual = residual - var * np.outer(v, v)
        self.components_ = np.array(components)
        self.explained_variance_ = np.array(variances)
        return self

    def _extract_one(self, S: np.ndarray, p: int) -> np.ndarray:
        _, eigvecs = np.linalg.eigh(S)
        v = eigvecs[:, -1].copy()
        for _ in range(self.max_iter):
            v_old = v.copy()
            z = S @ v
            v = _soft_threshold(z, self.lam)
            norm = np.linalg.norm(v)
            if norm < 1e-12:
                v = v_old
                break
            v /= norm
            if np.linalg.norm(v - v_old) < self.tol:
                break
        return v

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(axis=0)
        return Xc @ self.components_.T


# ---------------------------------------------------------------------------
# 5. Compressed Sensing: Iterative Hard Thresholding
# ---------------------------------------------------------------------------

class CompressedSensing:
    """Recover sparse signal from compressed measurements via IHT."""

    def __init__(self, sparsity: int = 10, max_iter: int = 500,
                 step_size: Optional[float] = None, tol: float = 1e-8):
        self.sparsity = sparsity
        self.max_iter = max_iter
        self.step_size = step_size
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.residual_history_: List[float] = []

    def fit(self, A: np.ndarray, y: np.ndarray) -> "CompressedSensing":
        """A: measurement matrix (m x n), y: measurements (m,)."""
        m, n = A.shape
        mu = self.step_size or 1.0 / np.linalg.norm(A, ord=2) ** 2
        x = np.zeros(n)
        self.residual_history_ = []
        for _ in range(self.max_iter):
            r = y - A @ x
            self.residual_history_.append(np.linalg.norm(r))
            grad_step = x + mu * A.T @ r
            x_new = _hard_threshold(grad_step, self.sparsity)
            if np.linalg.norm(x_new - x) < self.tol:
                x = x_new
                break
            x = x_new
        self.coef_ = x
        return self

    @staticmethod
    def random_measurement_matrix(m: int, n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((m, n)) / np.sqrt(m)


# ---------------------------------------------------------------------------
# 6. LARS (Least Angle Regression)
# ---------------------------------------------------------------------------

class LARS:
    """Least Angle Regression: full solution path."""

    def __init__(self, max_features: Optional[int] = None):
        self.max_features = max_features
        self.coef_path_: Optional[np.ndarray] = None
        self.active_sets_: List[List[int]] = []
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LARS":
        n, p = X.shape
        Xs, mu_x, sigma_x = _standardize(X)
        y_c = y - y.mean()
        max_feat = self.max_features or p
        beta = np.zeros(p)
        residual = y_c.copy()
        active: List[int] = []
        inactive = list(range(p))
        path = [beta.copy()]
        active_sets = [[]]
        for step in range(min(max_feat, p)):
            correlations = Xs.T @ residual
            abs_corr = np.abs(correlations)
            if not inactive:
                break
            j_star = max(inactive, key=lambda j: abs_corr[j])
            active.append(j_star)
            inactive.remove(j_star)
            Xa = Xs[:, active]
            try:
                Ga_inv = np.linalg.inv(Xa.T @ Xa)
            except np.linalg.LinAlgError:
                Ga_inv = np.linalg.pinv(Xa.T @ Xa)
            ones_a = np.ones(len(active))
            Aa = 1.0 / np.sqrt(ones_a @ Ga_inv @ ones_a)
            wa = Aa * Ga_inv @ ones_a
            sa = np.sign(correlations[active])
            wa = wa * sa
            ua = Xa @ wa
            C_max = abs_corr[j_star]
            if not inactive:
                gamma = C_max / (Aa + 1e-30)
            else:
                a_vec = Xs.T @ ua
                gammas = []
                for j in inactive:
                    cj = correlations[j]
                    aj = a_vec[j]
                    g1 = (C_max - cj) / (Aa - aj + 1e-30)
                    g2 = (C_max + cj) / (Aa + aj + 1e-30)
                    for g in [g1, g2]:
                        if g > 1e-10:
                            gammas.append(g)
                gamma = min(gammas) if gammas else C_max / (Aa + 1e-30)
            beta_active = beta[active] + gamma * wa
            beta = np.zeros(p)
            for i, idx in enumerate(active):
                beta[idx] = beta_active[i]
            residual = y_c - Xs @ beta
            path.append(beta.copy())
            active_sets.append(active.copy())
        self.coef_path_ = np.array(path)
        self.active_sets_ = active_sets
        # Convert back to original scale
        for i in range(len(self.coef_path_)):
            self.coef_path_[i] /= sigma_x
        self.coef_ = self.coef_path_[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_


# ---------------------------------------------------------------------------
# 7. Group LASSO
# ---------------------------------------------------------------------------

class GroupLasso:
    """Group LASSO: enforce group-level sparsity for factor selection."""

    def __init__(self, groups: List[List[int]], lam: float = 0.1,
                 max_iter: int = 1000, tol: float = 1e-6):
        self.groups = groups
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupLasso":
        n, p = X.shape
        Xs, mu_x, sigma_x = _standardize(X)
        y_c = y - y.mean()
        beta = np.zeros(p)
        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for g in self.groups:
                g = list(g)
                pg = len(g)
                r_g = y_c - Xs @ beta + Xs[:, g] @ beta[g]
                z_g = Xs[:, g].T @ r_g / n
                norm_z = np.linalg.norm(z_g)
                threshold = self.lam * np.sqrt(pg)
                if norm_z > threshold:
                    beta[g] = z_g * (1.0 - threshold / norm_z)
                else:
                    beta[g] = 0.0
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
        self.coef_ = beta / sigma_x
        self.intercept_ = y.mean() - self.coef_ @ mu_x
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def selected_groups(self) -> List[int]:
        selected = []
        for i, g in enumerate(self.groups):
            if np.linalg.norm(self.coef_[g]) > 1e-10:
                selected.append(i)
        return selected


# ---------------------------------------------------------------------------
# 8. Debiased LASSO: inference on high-dimensional regression
# ---------------------------------------------------------------------------

class DebiasedLasso:
    """Debiased (desparsified) LASSO for valid confidence intervals."""

    def __init__(self, lam: float = 0.1, max_iter: int = 1000, tol: float = 1e-6):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: Optional[np.ndarray] = None
        self.coef_debiased_: Optional[np.ndarray] = None
        self.std_errors_: Optional[np.ndarray] = None
        self.confidence_intervals_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> "DebiasedLasso":
        n, p = X.shape
        # Step 1: initial LASSO
        lasso = ElasticNet(lam=self.lam, alpha=1.0, max_iter=self.max_iter, tol=self.tol)
        lasso.fit(X, y)
        beta_lasso = lasso.coef_
        self.coef_ = beta_lasso
        # Step 2: nodewise LASSO to estimate precision-like projection
        Sigma_hat = X.T @ X / n
        M = np.zeros((p, p))
        for j in range(p):
            X_j = X[:, j]
            X_nj = np.delete(X, j, axis=1)
            nw_lasso = ElasticNet(lam=self.lam, alpha=1.0,
                                  max_iter=self.max_iter, tol=self.tol)
            nw_lasso.fit(X_nj, X_j)
            gamma_j = nw_lasso.coef_
            residual_j = X_j - X_nj @ gamma_j
            tau_j_sq = np.mean(residual_j ** 2)
            if tau_j_sq < 1e-12:
                tau_j_sq = 1e-12
            m_j = np.zeros(p)
            m_j[j] = 1.0 / tau_j_sq
            idx_nj = [k for k in range(p) if k != j]
            for ki, k in enumerate(idx_nj):
                m_j[k] = -gamma_j[ki] / tau_j_sq
            M[j] = m_j
        # Step 3: debiased estimator
        resid = y - X @ beta_lasso
        self.coef_debiased_ = beta_lasso + M @ (X.T @ resid) / n
        # Step 4: variance estimation
        sigma_sq = np.mean(resid ** 2)
        var_db = np.diag(M @ Sigma_hat @ M.T) * sigma_sq / n
        self.std_errors_ = np.sqrt(np.maximum(var_db, 0))
        # Confidence intervals
        from scipy.stats import norm as normal_dist
        z = normal_dist.ppf(1 - alpha / 2)
        self.confidence_intervals_ = np.column_stack([
            self.coef_debiased_ - z * self.std_errors_,
            self.coef_debiased_ + z * self.std_errors_
        ])
        return self

    def p_values(self) -> np.ndarray:
        from scipy.stats import norm as normal_dist
        z_stats = self.coef_debiased_ / (self.std_errors_ + 1e-30)
        return 2 * (1 - normal_dist.cdf(np.abs(z_stats)))

    def significant_features(self, alpha: float = 0.05) -> np.ndarray:
        return np.where(self.p_values() < alpha)[0]


# ---------------------------------------------------------------------------
# 9. Sparse Portfolio: L1 Constrained Markowitz
# ---------------------------------------------------------------------------

class SparsePortfolio:
    """L1-constrained mean-variance portfolio (sparse Markowitz)."""

    def __init__(self, lam_sparse: float = 0.1, risk_aversion: float = 1.0,
                 max_iter: int = 2000, tol: float = 1e-8):
        self.lam_sparse = lam_sparse
        self.risk_aversion = risk_aversion
        self.max_iter = max_iter
        self.tol = tol
        self.weights_: Optional[np.ndarray] = None

    def fit(self, mu: np.ndarray, Sigma: np.ndarray,
            long_only: bool = False) -> "SparsePortfolio":
        """
        min  gamma/2 * w' Sigma w - mu' w + lam * ||w||_1
        s.t. sum(w) = 1, (w >= 0 if long_only)
        """
        p = len(mu)
        w = np.ones(p) / p
        gamma = self.risk_aversion
        # Proximal gradient descent
        L = gamma * np.linalg.norm(Sigma, ord=2) + 1e-3
        step = 1.0 / L
        for _ in range(self.max_iter):
            w_old = w.copy()
            grad = gamma * Sigma @ w - mu
            w_half = w - step * grad
            w = _soft_threshold(w_half, step * self.lam_sparse)
            if long_only:
                w = np.maximum(w, 0.0)
            # Project onto sum=1
            w = w + (1.0 - np.sum(w)) / p
            if np.linalg.norm(w - w_old) < self.tol:
                break
        self.weights_ = w
        return self

    def portfolio_stats(self, mu: np.ndarray, Sigma: np.ndarray) -> Dict[str, float]:
        w = self.weights_
        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)
        n_active = int(np.sum(np.abs(w) > 1e-6))
        return {"return": ret, "volatility": vol,
                "sharpe": ret / (vol + 1e-30), "n_active": n_active,
                "L1_norm": float(np.sum(np.abs(w)))}


# ---------------------------------------------------------------------------
# 10. Application: Sparse Factor Model
# ---------------------------------------------------------------------------

class SparseFactorModel:
    """
    Sparse factor model for asset returns:
    R = B @ F + epsilon, with sparse B estimated via LASSO.
    Covariance structure: Sigma = B @ Cov(F) @ B' + diag(sigma_eps^2).
    """

    def __init__(self, n_factors: int = 5, lam: float = 0.05,
                 max_iter: int = 500, tol: float = 1e-6):
        self.n_factors = n_factors
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.loadings_: Optional[np.ndarray] = None
        self.factors_: Optional[np.ndarray] = None
        self.idio_var_: Optional[np.ndarray] = None
        self.factor_cov_: Optional[np.ndarray] = None

    def fit(self, returns: np.ndarray) -> "SparseFactorModel":
        """
        returns: (T, N) asset return matrix.
        Two-step: PCA for factor extraction, then sparse regression for loadings.
        """
        T, N = returns.shape
        r_demean = returns - returns.mean(axis=0)
        # Step 1: extract factors via PCA
        cov = r_demean.T @ r_demean / T
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:self.n_factors]
        V = eigvecs[:, idx]
        F = r_demean @ V  # (T, K)
        self.factors_ = F
        self.factor_cov_ = F.T @ F / T
        # Step 2: sparse loadings per asset via LASSO
        B = np.zeros((N, self.n_factors))
        idio_var = np.zeros(N)
        for i in range(N):
            en = ElasticNet(lam=self.lam, alpha=1.0,
                            max_iter=self.max_iter, tol=self.tol)
            en.fit(F, r_demean[:, i])
            B[i] = en.coef_
            resid = r_demean[:, i] - F @ en.coef_
            idio_var[i] = np.var(resid)
        self.loadings_ = B
        self.idio_var_ = idio_var
        return self

    def covariance(self) -> np.ndarray:
        B = self.loadings_
        return B @ self.factor_cov_ @ B.T + np.diag(self.idio_var_)

    def predict(self, factors: np.ndarray) -> np.ndarray:
        return factors @ self.loadings_.T

    def factor_exposure_summary(self) -> Dict[str, Any]:
        B = self.loadings_
        n_assets, n_factors = B.shape
        summary: Dict[str, Any] = {}
        for k in range(n_factors):
            active = np.where(np.abs(B[:, k]) > 1e-6)[0]
            summary[f"factor_{k}"] = {
                "n_exposed": len(active),
                "assets": active.tolist(),
                "mean_loading": float(np.mean(B[active, k])) if len(active) > 0 else 0.0,
                "max_loading": float(np.max(np.abs(B[:, k]))),
            }
        return summary

    def r_squared(self, returns: np.ndarray) -> np.ndarray:
        """Per-asset R^2 from factor model."""
        r_demean = returns - returns.mean(axis=0)
        fitted = self.factors_ @ self.loadings_.T
        ss_tot = np.sum(r_demean ** 2, axis=0)
        ss_res = np.sum((r_demean - fitted) ** 2, axis=0)
        return 1.0 - ss_res / (ss_tot + 1e-30)


# ---------------------------------------------------------------------------
# Convenience: run a full sparse analysis pipeline
# ---------------------------------------------------------------------------

def sparse_analysis_pipeline(returns: np.ndarray, factors: np.ndarray,
                             lam_lasso: float = 0.05,
                             lam_glasso: float = 0.1,
                             n_sparse_pca: int = 3,
                             sparsity_portfolio: float = 0.1) -> Dict[str, Any]:
    """
    End-to-end sparse analysis:
    1. Sparse factor model
    2. Graphical LASSO on residuals
    3. Sparse PCA on returns
    4. Sparse portfolio
    """
    T, N = returns.shape
    results: Dict[str, Any] = {}

    # 1. Sparse factor model
    sfm = SparseFactorModel(n_factors=min(factors.shape[1], 10), lam=lam_lasso)
    sfm.fit(returns)
    results["factor_model"] = {
        "r_squared": sfm.r_squared(returns).tolist(),
        "exposure_summary": sfm.factor_exposure_summary(),
    }

    # 2. Graphical LASSO on residuals
    residuals = returns - sfm.factors_ @ sfm.loadings_.T
    glasso = GraphicalLasso(lam=lam_glasso)
    glasso.fit(residuals)
    pcorr = glasso.partial_correlations()
    n_edges = np.sum(np.abs(pcorr[np.triu_indices(N, k=1)]) > 0.05)
    results["residual_network"] = {
        "n_edges": int(n_edges),
        "density": float(n_edges / (N * (N - 1) / 2)),
    }

    # 3. Sparse PCA
    spca = SparsePCA(n_components=n_sparse_pca, lam=lam_lasso)
    spca.fit(returns)
    results["sparse_pca"] = {
        "explained_variance": spca.explained_variance_.tolist(),
        "sparsity_per_component": [int(np.sum(np.abs(c) > 1e-6))
                                   for c in spca.components_],
    }

    # 4. Sparse portfolio
    mu_hat = returns.mean(axis=0)
    Sigma_hat = sfm.covariance()
    sp = SparsePortfolio(lam_sparse=sparsity_portfolio)
    sp.fit(mu_hat, Sigma_hat, long_only=True)
    results["sparse_portfolio"] = sp.portfolio_stats(mu_hat, Sigma_hat)

    return results
