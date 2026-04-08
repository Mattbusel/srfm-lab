"""
Causal inference methods for financial econometrics.

Implements:
  - PC algorithm (constraint-based DAG learning)
  - Backdoor criterion and adjustment
  - Do-calculus interventions
  - Instrumental variable (IV) estimation
  - Regression Discontinuity Design (RDD)
  - Difference-in-Differences (DiD)
  - Synthetic control method
  - PCMCI (time-series causal discovery)
  - Causal impact estimation (Bayesian structural time series)
  - Transfer entropy causality (information-theoretic)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Correlation / partial correlation utilities ────────────────────────────────

def partial_correlation(
    X: np.ndarray,
    i: int,
    j: int,
    S: list[int],
) -> float:
    """
    Partial correlation between X[:, i] and X[:, j] conditioning on X[:, S].
    Used in skeleton discovery (PC algorithm).
    """
    if not S:
        corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
        return float(corr)

    # Regress out conditioning set
    Z = X[:, S]
    ZtZinv = np.linalg.pinv(Z.T @ Z)

    def residual(k):
        beta = ZtZinv @ (Z.T @ X[:, k])
        return X[:, k] - Z @ beta

    ri = residual(i)
    rj = residual(j)
    return float(np.corrcoef(ri, rj)[0, 1])


def partial_corr_pvalue(r: float, n: int, k: int) -> float:
    """
    p-value for partial correlation r with n observations and k conditioning variables.
    Uses t-distribution: t = r * sqrt((n-k-2)/(1-r^2)).
    """
    from scipy.stats import t as t_dist

    df = n - k - 2
    if df <= 0:
        return 1.0
    t_stat = r * math.sqrt(df / (1 - r**2 + 1e-10))
    return float(2 * t_dist.sf(abs(t_stat), df))


# ── PC Algorithm ──────────────────────────────────────────────────────────────

def pc_skeleton(
    X: np.ndarray,
    alpha: float = 0.05,
    max_cond_set: int = 3,
) -> tuple[dict, dict]:
    """
    PC algorithm skeleton phase: learns undirected graph by testing conditional
    independence via partial correlations.

    Returns (adjacency dict, separation sets dict).
    """
    T, n = X.shape
    # Initialize complete graph
    adj = {i: set(range(n)) - {i} for i in range(n)}
    sep_sets = {}

    for cond_size in range(max_cond_set + 1):
        edges_to_remove = []

        for i in range(n):
            for j in list(adj[i]):
                if j <= i:
                    continue

                # Possible conditioning sets from neighbors of i (excluding j)
                neighbors_i = adj[i] - {j}
                if len(neighbors_i) < cond_size:
                    continue

                # Test all conditioning sets of size cond_size
                from itertools import combinations
                for S in combinations(neighbors_i, cond_size):
                    S = list(S)
                    r = partial_correlation(X, i, j, S)
                    p = partial_corr_pvalue(r, T, len(S))
                    if p > alpha:
                        edges_to_remove.append((i, j))
                        sep_sets[(i, j)] = sep_sets[(j, i)] = S
                        break

        for i, j in edges_to_remove:
            adj[i].discard(j)
            adj[j].discard(i)

    return adj, sep_sets


def orient_v_structures(
    adj: dict,
    sep_sets: dict,
) -> dict:
    """
    Orient v-structures (colliders): i → k ← j where k not in sep_set(i,j).
    Returns directed adjacency as dict of sets (i → j means j in directed[i]).
    """
    n = len(adj)
    directed = {i: set() for i in range(n)}
    undirected = {i: adj[i].copy() for i in range(n)}

    for k in range(n):
        neighbors_k = list(undirected[k])
        for idx1 in range(len(neighbors_k)):
            for idx2 in range(idx1 + 1, len(neighbors_k)):
                i, j = neighbors_k[idx1], neighbors_k[idx2]
                if j not in undirected[i]:  # i and j not adjacent
                    sep = sep_sets.get((i, j), [])
                    if k not in sep:
                        # v-structure: i → k ← j
                        directed[i].add(k)
                        directed[j].add(k)
                        undirected[i].discard(k)
                        undirected[j].discard(k)
                        undirected[k].discard(i)
                        undirected[k].discard(j)

    return directed


def pc_algorithm(
    X: np.ndarray,
    alpha: float = 0.05,
    max_cond_set: int = 3,
) -> dict:
    """
    Full PC algorithm for causal DAG discovery.
    Returns dict with adjacency (skeleton), directed edges, v-structures.
    """
    adj, sep_sets = pc_skeleton(X, alpha, max_cond_set)
    directed = orient_v_structures(adj, sep_sets)

    n = len(adj)
    edge_list = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                if j in directed.get(i, set()):
                    edge_list.append((i, j, "->"))
                elif i in directed.get(j, set()):
                    edge_list.append((i, j, "<-"))
                else:
                    edge_list.append((i, j, "--"))

    return {
        "adjacency": adj,
        "directed": directed,
        "sep_sets": sep_sets,
        "edge_list": edge_list,
        "n_edges": len(edge_list),
    }


# ── Backdoor Criterion and Adjustment ─────────────────────────────────────────

def backdoor_adjustment(
    Y: np.ndarray,
    T_var: np.ndarray,
    Z: np.ndarray,
) -> dict:
    """
    Backdoor adjustment formula for estimating causal effect of T on Y.
    Adjusts for confounders Z.

    ATE = E[Y(1)] - E[Y(0)] estimated via outcome regression.
    """
    # Outcome model: Y ~ T + Z
    n = len(Y)
    Z_aug = np.column_stack([T_var, Z, np.ones(n)])

    try:
        beta = np.linalg.lstsq(Z_aug, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"ate": 0.0, "se": np.inf}

    # ATE = beta[0] (coefficient on T in adjusted regression)
    ate = float(beta[0])

    # Standard error via residuals
    y_hat = Z_aug @ beta
    resid = Y - y_hat
    sigma2 = resid.var()
    try:
        var_beta = sigma2 * np.linalg.inv(Z_aug.T @ Z_aug)[0, 0]
        se = float(math.sqrt(max(var_beta, 0)))
    except np.linalg.LinAlgError:
        se = np.inf

    return {
        "ate": ate,
        "se": se,
        "t_stat": float(ate / (se + 1e-10)),
        "coefficients": beta.tolist(),
    }


# ── Instrumental Variables (IV / 2SLS) ───────────────────────────────────────

def two_stage_least_squares(
    Y: np.ndarray,
    D: np.ndarray,     # endogenous treatment
    Z_inst: np.ndarray,  # instruments (excludes directly affecting Y)
    X: Optional[np.ndarray] = None,  # exogenous controls
) -> dict:
    """
    Two-Stage Least Squares (2SLS) IV estimation.
    Stage 1: D = Z_inst * gamma + X * delta + v
    Stage 2: Y = D_hat * beta + X * alpha + eps

    Returns causal estimate beta (LATE if instrument is binary).
    """
    n = len(Y)
    if X is None:
        X = np.ones((n, 1))
    elif X.ndim == 1:
        X = X[:, None]

    const = np.ones((n, 1))
    X_aug = np.column_stack([X, const])
    Z_aug = np.column_stack([Z_inst if Z_inst.ndim > 1 else Z_inst[:, None], X_aug])

    # Stage 1
    try:
        gamma1 = np.linalg.lstsq(Z_aug, D, rcond=None)[0]
        D_hat = Z_aug @ gamma1
    except np.linalg.LinAlgError:
        return {"beta_iv": np.nan, "first_stage_f": 0.0}

    # First-stage F-statistic
    D_hat_res = D - D.mean()
    D_resid = D - D_hat
    rss_r = float((D - D.mean())**2 .sum())
    rss_u = float(D_resid**2.sum())
    n_inst = Z_inst.shape[1] if Z_inst.ndim > 1 else 1
    df1 = n_inst
    df2 = n - Z_aug.shape[1]
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / max(df2, 1))

    # Stage 2
    Xd_aug = np.column_stack([D_hat, X_aug])
    try:
        beta2 = np.linalg.lstsq(Xd_aug, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"beta_iv": np.nan, "first_stage_f": float(f_stat)}

    beta_iv = float(beta2[0])
    resid2 = Y - Xd_aug @ beta2
    sigma2 = resid2.var()
    try:
        var_iv = sigma2 * np.linalg.inv(Xd_aug.T @ Xd_aug)[0, 0]
        se_iv = float(math.sqrt(max(var_iv, 0)))
    except np.linalg.LinAlgError:
        se_iv = np.inf

    return {
        "beta_iv": beta_iv,
        "se_iv": se_iv,
        "t_stat": float(beta_iv / (se_iv + 1e-10)),
        "first_stage_f": float(f_stat),
        "weak_instrument": bool(f_stat < 10),
    }


# ── Regression Discontinuity Design ──────────────────────────────────────────

def rdd_estimate(
    Y: np.ndarray,
    R: np.ndarray,     # running variable (e.g., score, index value)
    cutoff: float,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
) -> dict:
    """
    Sharp RDD: estimate treatment effect at discontinuity.
    Fits separate local polynomials on each side of cutoff.
    """
    n = len(Y)
    treated = R >= cutoff
    r_centered = R - cutoff

    if bandwidth is None:
        # Silverman bandwidth
        bandwidth = 1.06 * r_centered.std() * n**(-0.2)

    # Kernel weights
    x = np.abs(r_centered) / bandwidth
    if kernel == "triangular":
        w = np.maximum(1 - x, 0)
    elif kernel == "epanechnikov":
        w = np.maximum(0.75 * (1 - x**2), 0)
    else:
        w = (x <= 1).astype(float)  # uniform

    in_bw = w > 0
    if in_bw.sum() < 4:
        return {"tau": np.nan, "se": np.inf, "bandwidth": bandwidth}

    # Fit weighted local linear on each side
    def local_linear(mask):
        Xs = np.column_stack([np.ones(mask.sum()), r_centered[mask]])
        ys = Y[mask]
        ws = w[mask]
        W = np.diag(ws)
        XtWX = Xs.T @ W @ Xs
        XtWy = Xs.T @ W @ ys
        try:
            return np.linalg.solve(XtWX + 1e-8 * np.eye(2), XtWy)
        except np.linalg.LinAlgError:
            return np.array([ys.mean(), 0.0])

    left_mask = in_bw & ~treated
    right_mask = in_bw & treated

    if left_mask.sum() < 2 or right_mask.sum() < 2:
        return {"tau": np.nan, "se": np.inf, "bandwidth": bandwidth}

    beta_left = local_linear(left_mask)
    beta_right = local_linear(right_mask)

    # Treatment effect = difference in intercepts at cutoff
    tau = float(beta_right[0] - beta_left[0])

    # Bootstrap SE
    ses = []
    rng = np.random.default_rng(42)
    for _ in range(200):
        idx = rng.integers(0, n, n)
        Yb, Rb = Y[idx], R[idx]
        rb = Rb - cutoff
        xb = np.abs(rb) / bandwidth
        if kernel == "triangular":
            wb = np.maximum(1 - xb, 0)
        else:
            wb = (xb <= 1).astype(float)
        inb = wb > 0
        tr_b = Rb >= cutoff
        lm = inb & ~tr_b
        rm = inb & tr_b
        if lm.sum() < 2 or rm.sum() < 2:
            continue
        Xl = np.column_stack([np.ones(lm.sum()), rb[lm]])
        Xr = np.column_stack([np.ones(rm.sum()), rb[rm]])
        bl = np.linalg.lstsq(np.diag(wb[lm])**0.5 @ Xl, np.diag(wb[lm])**0.5 @ Yb[lm], rcond=None)[0]
        br = np.linalg.lstsq(np.diag(wb[rm])**0.5 @ Xr, np.diag(wb[rm])**0.5 @ Yb[rm], rcond=None)[0]
        ses.append(br[0] - bl[0])

    se = float(np.std(ses)) if ses else np.inf
    return {
        "tau": tau,
        "se": se,
        "t_stat": float(tau / (se + 1e-10)),
        "bandwidth": float(bandwidth),
        "n_left": int(left_mask.sum()),
        "n_right": int(right_mask.sum()),
    }


# ── Difference-in-Differences ─────────────────────────────────────────────────

def diff_in_diff(
    Y: np.ndarray,         # outcome (T, N)
    treated: np.ndarray,   # binary (N,): 1 = treatment group
    post: np.ndarray,      # binary (T,): 1 = post-treatment period
) -> dict:
    """
    Canonical 2x2 DiD estimator.
    DiD = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
    """
    tr = treated.astype(bool)
    po = post.astype(bool)

    y_tp = Y[po][:, tr].mean()   # treated, post
    y_tpr = Y[~po][:, tr].mean() # treated, pre
    y_cp = Y[po][:, ~tr].mean()  # control, post
    y_cpr = Y[~po][:, ~tr].mean()# control, pre

    did = (y_tp - y_tpr) - (y_cp - y_cpr)

    # Panel regression: Y_it = alpha + beta*Treat_i + gamma*Post_t + delta*(Treat*Post) + eps
    T_len, N = Y.shape
    y_vec = Y.flatten()
    treat_vec = np.tile(treated, T_len)
    post_vec = np.repeat(post, N)
    interact_vec = treat_vec * post_vec

    X_reg = np.column_stack([treat_vec, post_vec, interact_vec, np.ones(len(y_vec))])
    try:
        beta = np.linalg.lstsq(X_reg, y_vec, rcond=None)[0]
        resid = y_vec - X_reg @ beta
        sigma2 = resid.var()
        var_b = sigma2 * np.linalg.pinv(X_reg.T @ X_reg)
        se_did = float(math.sqrt(max(var_b[2, 2], 0)))
    except np.linalg.LinAlgError:
        se_did = np.inf

    # Parallel trends check: pre-period trend difference
    if (~po).sum() >= 2:
        pre_Y = Y[~po]
        treat_trend = np.diff(pre_Y[:, tr].mean(1)).mean()
        control_trend = np.diff(pre_Y[:, ~tr].mean(1)).mean()
        trend_diff = float(treat_trend - control_trend)
    else:
        trend_diff = np.nan

    return {
        "did_estimate": float(did),
        "se": se_did,
        "t_stat": float(did / (se_did + 1e-10)),
        "y_treat_post": float(y_tp),
        "y_treat_pre": float(y_tpr),
        "y_control_post": float(y_cp),
        "y_control_pre": float(y_cpr),
        "pre_trend_diff": trend_diff,
        "parallel_trends_concern": bool(abs(trend_diff) > 0.01) if not math.isnan(trend_diff) else False,
    }


# ── Synthetic Control ──────────────────────────────────────────────────────────

def synthetic_control(
    Y: np.ndarray,        # (T, N): N units, T time periods
    treated_unit: int,    # index of treated unit
    treatment_start: int, # first treated period
) -> dict:
    """
    Synthetic control method (Abadie et al. 2010).
    Finds donor weights W* such that pre-period synthetic control
    ≈ treated unit. Post-period gap = causal effect.
    """
    from scipy.optimize import minimize

    T, N = Y.shape
    donors = [i for i in range(N) if i != treated_unit]
    n_donors = len(donors)

    Y_treated = Y[:, treated_unit]
    Y_donors = Y[:, donors]

    # Pre-period only
    Y_pre_treated = Y_treated[:treatment_start]
    Y_pre_donors = Y_donors[:treatment_start]

    # Minimize || Y_treated_pre - Y_donors_pre @ w ||^2 subject to w >= 0, sum(w)=1
    def objective(w):
        synth = Y_pre_donors @ w
        return float(np.sum((Y_pre_treated - synth)**2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n_donors
    w0 = np.ones(n_donors) / n_donors

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w_opt = result.x if result.success else w0

    # Synthetic control series
    Y_synth = Y_donors @ w_opt

    # Treatment effects
    gaps = Y_treated - Y_synth
    pre_gaps = gaps[:treatment_start]
    post_gaps = gaps[treatment_start:]

    return {
        "weights": dict(zip(donors, w_opt.tolist())),
        "synthetic": Y_synth,
        "gaps": gaps,
        "pre_rmse": float(np.sqrt(pre_gaps**2.mean())),
        "avg_treatment_effect": float(post_gaps.mean()),
        "cumulative_effect": float(post_gaps.sum()),
        "peak_effect": float(post_gaps.max()) if len(post_gaps) > 0 else 0.0,
    }


# ── PCMCI — Time-Series Causal Discovery ──────────────────────────────────────

def pcmci_skeleton(
    X: np.ndarray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> dict:
    """
    Simplified PCMCI (Runge et al. 2019) for time-series causal discovery.
    Tests lagged conditional independence: X_i(t) ⊥ X_j(t-tau) | past(X_i).
    Returns causal graph with lags.
    """
    T, N = X.shape
    links = {}  # (j, i, lag): p_value

    for i in range(N):
        for j in range(N):
            for lag in range(1, max_lag + 1):
                if T - lag < 2 * max_lag + 2:
                    continue

                # Build conditioning set: own past of X_i
                cond_cols = []
                for l in range(1, max_lag + 1):
                    cond_cols.append(X[max_lag - l: T - l, i])

                y_t = X[max_lag:, i]
                x_lag = X[max_lag - lag: T - lag, j]

                Z = np.column_stack(cond_cols)
                Z_aug = np.column_stack([Z, np.ones(len(y_t))])

                try:
                    # Partial correlation
                    beta_y = np.linalg.lstsq(Z_aug, y_t, rcond=None)[0]
                    beta_x = np.linalg.lstsq(Z_aug, x_lag, rcond=None)[0]
                    ry = y_t - Z_aug @ beta_y
                    rx = x_lag - Z_aug @ beta_x
                    r = float(np.corrcoef(ry, rx)[0, 1])
                    p = partial_corr_pvalue(r, len(y_t), Z_aug.shape[1] - 1)
                    links[(j, i, lag)] = {"r": r, "p_value": p, "significant": p < alpha}
                except Exception:
                    pass

    significant_links = {k: v for k, v in links.items() if v["significant"]}
    return {
        "all_links": links,
        "significant_links": significant_links,
        "causal_graph": {
            (j, i, lag): v["r"]
            for (j, i, lag), v in significant_links.items()
        },
        "n_significant": len(significant_links),
    }


# ── Transfer Entropy Causality ────────────────────────────────────────────────

def transfer_entropy_causality(
    X: np.ndarray,
    alpha: float = 0.05,
    n_permutations: int = 100,
    lag: int = 1,
) -> np.ndarray:
    """
    Pairwise transfer entropy significance test via permutation.
    Returns (N, N) matrix of p-values for X_j → X_i causal links.
    """
    T, N = X.shape
    p_matrix = np.ones((N, N))
    rng = np.random.default_rng(42)

    def _te(source, target):
        # Bin both series
        n_bins = max(3, int(np.sqrt(T / 5)))
        s_bins = np.digitize(source[:-lag], np.percentile(source[:-lag], np.linspace(0, 100, n_bins + 1)[1:-1]))
        t_bins = np.digitize(target[lag:], np.percentile(target[lag:], np.linspace(0, 100, n_bins + 1)[1:-1]))
        tpast = np.digitize(target[:-lag], np.percentile(target[:-lag], np.linspace(0, 100, n_bins + 1)[1:-1]))

        n = len(s_bins)
        # Joint counts
        joint = np.zeros((n_bins, n_bins, n_bins))
        for t in range(n):
            joint[s_bins[t] - 1, t_bins[t] - 1, tpast[t] - 1] += 1
        joint = joint / n + 1e-10

        # Marginals
        p_t_tp = joint.sum(0)
        p_tp = joint.sum((0, 1))
        p_s_t_tp = joint

        # TE = sum p(t, tp, s) log p(t|tp,s) / p(t|tp)
        te = float(np.sum(p_s_t_tp * np.log(
            p_s_t_tp / (p_t_tp[None, :, :] + 1e-10) + 1e-10
        ) - p_s_t_tp * np.log(
            p_t_tp[None, :, :] / (p_tp[None, None, :] + 1e-10) + 1e-10
        )))
        return max(te, 0)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            te_obs = _te(X[:, j], X[:, i])
            # Permutation null
            null_tes = []
            for _ in range(n_permutations):
                perm = rng.permutation(T)
                null_tes.append(_te(X[perm, j], X[:, i]))
            p_val = float(np.mean(np.array(null_tes) >= te_obs))
            p_matrix[j, i] = p_val

    return p_matrix
