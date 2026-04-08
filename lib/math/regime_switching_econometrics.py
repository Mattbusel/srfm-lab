"""
lib/math/regime_switching_econometrics.py

Econometric regime-switching models for quantitative research.

Implements:
  - Hamilton Markov-switching AR model (2-state):
      filter_hamilton(y, params) → state probabilities
      estimate_hamilton(y, n_states) → EM estimation
  - Threshold AR (TAR): different AR dynamics in different regimes
  - Smooth Transition AR (STAR): logistic and exponential transitions
  - Markov-switching GARCH: volatility regimes
  - Regime-switching factor model: factor loadings change by regime
  - Change point detection: BOCPD (Bayesian online changepoint detection)
  - Structural break tests: Bai-Perron style multiple breaks
  - Regime duration distribution (geometric/Poisson approximation)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import optimize, stats


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class HamiltonParams:
    """Parameters for Hamilton Markov-switching AR model."""
    mu: np.ndarray       # means for each state (n_states,)
    sigma: np.ndarray    # std devs for each state (n_states,)
    P: np.ndarray        # transition matrix (n_states, n_states)
    n_states: int = 2
    ar_order: int = 0
    ar_coefs: Optional[np.ndarray] = None  # (n_states, ar_order) AR coefficients


@dataclass
class HamiltonFilterResult:
    filtered_probs: np.ndarray    # (T, n_states) P(S_t=k | y_1,...,y_t)
    smoothed_probs: np.ndarray    # (T, n_states) P(S_t=k | y_1,...,y_T)
    predicted_probs: np.ndarray   # (T, n_states) P(S_t=k | y_1,...,y_{t-1})
    log_likelihood: float
    regime_sequence: np.ndarray   # most likely regime at each t (argmax)


@dataclass
class TARParams:
    """Threshold AR model parameters."""
    threshold: float
    ar_coefs_low: np.ndarray    # AR coefficients when y_{t-d} <= threshold
    ar_coefs_high: np.ndarray   # AR coefficients when y_{t-d} > threshold
    mu_low: float
    mu_high: float
    sigma_low: float
    sigma_high: float
    delay: int = 1              # delay parameter d


@dataclass
class STARParams:
    """Smooth Transition AR model parameters."""
    ar_coefs_1: np.ndarray      # AR coefficients in first regime
    ar_coefs_2: np.ndarray      # AR coefficients in second regime
    mu_1: float
    mu_2: float
    gamma: float                # transition speed
    c: float                    # threshold
    delay: int = 1
    transition_type: str = "logistic"   # "logistic" or "exponential"


@dataclass
class MSGARCHParams:
    """Markov-switching GARCH parameters."""
    n_states: int
    omega: np.ndarray       # (n_states,) GARCH constants
    alpha: np.ndarray       # (n_states,) ARCH coefficients
    beta: np.ndarray        # (n_states,) GARCH coefficients
    P: np.ndarray           # (n_states, n_states) transition matrix


@dataclass
class RegimeSwitchingFactorModel:
    """Factor model with regime-dependent loadings."""
    n_states: int
    loadings: np.ndarray    # (n_states, n_assets, n_factors)
    factor_means: np.ndarray  # (n_states, n_factors)
    idio_vols: np.ndarray    # (n_states, n_assets)
    P: np.ndarray            # (n_states, n_states) transition matrix


@dataclass
class ChangePoint:
    """A detected change point."""
    index: int
    probability: float
    regime_before: int
    regime_after: int


@dataclass
class StructuralBreakResult:
    break_dates: List[int]
    f_statistic: float
    critical_value: float
    n_breaks: int
    segment_means: np.ndarray
    bic: float


@dataclass
class RegimeDuration:
    expected_duration: np.ndarray   # (n_states,) expected time in each regime
    std_duration: np.ndarray
    regime_probs: np.ndarray        # stationary distribution


# ── Hamilton Markov-Switching AR ──────────────────────────────────────────────

def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal PDF."""
    if sigma <= 0:
        return 1e-300
    z = (x - mu) / sigma
    return math.exp(-0.5 * z**2) / (sigma * math.sqrt(2.0 * math.pi))


def filter_hamilton(
    y: np.ndarray,
    params: HamiltonParams,
) -> HamiltonFilterResult:
    """
    Hamilton filter for Markov-switching model.

    Computes filtered state probabilities P(S_t = k | y_1,...,y_t)
    and log-likelihood via prediction-error decomposition.

    For AR order p: y_t = mu_k + sum_j phi_{k,j} * y_{t-j} + sigma_k * eps_t
    """
    T = len(y)
    K = params.n_states
    P_trans = params.P
    p = params.ar_order

    # Start index (skip first p observations for AR)
    start = max(p, 1)

    filtered = np.zeros((T, K))
    predicted = np.zeros((T, K))
    log_lik = 0.0

    # Initialize with stationary distribution
    try:
        eigvals, eigvecs = np.linalg.eig(P_trans.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()
    except Exception:
        pi = np.ones(K) / K

    prev_filtered = pi.copy()

    for t in range(T):
        # Prediction step
        pred = P_trans.T @ prev_filtered
        pred = np.maximum(pred, 1e-300)
        predicted[t] = pred

        # Emission probabilities
        eta = np.zeros(K)
        for k in range(K):
            if p > 0 and params.ar_coefs is not None and t >= p:
                residual = y[t]
                for j in range(p):
                    residual -= params.ar_coefs[k, j] * y[t - 1 - j]
                residual -= params.mu[k]
                eta[k] = _normal_pdf(residual, 0.0, params.sigma[k])
            else:
                eta[k] = _normal_pdf(y[t], params.mu[k], params.sigma[k])

        # Update step
        joint = pred * eta
        total = joint.sum()
        if total <= 0:
            total = 1e-300
        filtered[t] = joint / total
        log_lik += math.log(total)
        prev_filtered = filtered[t].copy()

    # Kim smoother for smoothed probabilities
    smoothed = np.zeros((T, K))
    smoothed[-1] = filtered[-1].copy()
    for t in range(T - 2, -1, -1):
        for k in range(K):
            numerator = 0.0
            for j in range(K):
                denom = predicted[t + 1, j]
                if denom > 1e-300:
                    numerator += smoothed[t + 1, j] * P_trans[k, j] * filtered[t, k] / denom
            smoothed[t, k] = numerator
        s_sum = smoothed[t].sum()
        if s_sum > 0:
            smoothed[t] /= s_sum

    regime_seq = np.argmax(filtered, axis=1)

    return HamiltonFilterResult(
        filtered_probs=filtered,
        smoothed_probs=smoothed,
        predicted_probs=predicted,
        log_likelihood=log_lik,
        regime_sequence=regime_seq,
    )


def estimate_hamilton(
    y: np.ndarray,
    n_states: int = 2,
    ar_order: int = 0,
    max_iter: int = 200,
    tol: float = 1e-6,
    n_init: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[HamiltonParams, HamiltonFilterResult]:
    """
    Estimate Hamilton Markov-switching model via EM algorithm.

    Multiple random initializations to avoid local maxima.
    Returns (best_params, best_filter_result).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    K = n_states
    p = ar_order
    T = len(y)
    y_std = float(y.std())
    y_mean = float(y.mean())

    best_ll = -np.inf
    best_result = None
    best_params = None

    for init in range(n_init):
        # Random initialization
        mu_init = y_mean + rng.standard_normal(K) * y_std
        mu_init = np.sort(mu_init)  # sort for identifiability
        sigma_init = np.abs(rng.standard_normal(K)) * y_std * 0.5 + y_std * 0.2
        P_init = rng.dirichlet(np.ones(K) * 5, size=K)  # prefer staying in regime

        if p > 0:
            ar_init = rng.standard_normal((K, p)) * 0.1
        else:
            ar_init = None

        params = HamiltonParams(
            mu=mu_init,
            sigma=sigma_init,
            P=P_init,
            n_states=K,
            ar_order=p,
            ar_coefs=ar_init,
        )

        prev_ll = -np.inf
        for iteration in range(max_iter):
            # E-step
            fr = filter_hamilton(y, params)

            if fr.log_likelihood - prev_ll < tol and iteration > 5:
                break
            prev_ll = fr.log_likelihood

            # M-step: update parameters using smoothed probs
            S = fr.smoothed_probs  # (T, K)

            # Update means and sigmas
            new_mu = np.zeros(K)
            new_sigma = np.zeros(K)
            for k in range(K):
                w = S[:, k]
                w_sum = w.sum() + 1e-10
                if p == 0:
                    new_mu[k] = (w * y).sum() / w_sum
                    res = y - new_mu[k]
                    new_sigma[k] = math.sqrt((w * res**2).sum() / w_sum)
                else:
                    # AR: y_t - phi @ y_{t-p:t} = mu + eps
                    new_mu[k] = (w * y).sum() / w_sum
                    res = y - new_mu[k]
                    if ar_init is not None:
                        for j in range(p):
                            if j + 1 < T:
                                res[j+1:] -= params.ar_coefs[k, j] * y[:-j-1] if j+1 <= T-1 else 0
                    new_sigma[k] = max(math.sqrt((w * res**2).sum() / w_sum), 1e-6)

            # Update transition matrix
            # Xi_{t,i,j} ∝ filtered[t-1,i] * P[i,j] * eta[t,j] * smoothed[t,j] / predicted[t,j]
            new_P = np.zeros((K, K))
            for k in range(K):
                for j in range(K):
                    num = 0.0
                    for t in range(1, T):
                        pred_j = fr.predicted_probs[t, j]
                        if pred_j > 1e-300:
                            num += (S[t, j] * params.P[k, j] * S[t-1, k]) / (pred_j + 1e-300)
                    new_P[k, j] = max(num, 1e-6)
                row_sum = new_P[k].sum()
                new_P[k] /= row_sum

            params = HamiltonParams(
                mu=new_mu,
                sigma=np.maximum(new_sigma, 1e-6),
                P=new_P,
                n_states=K,
                ar_order=p,
                ar_coefs=params.ar_coefs,
            )

        fr_final = filter_hamilton(y, params)
        if fr_final.log_likelihood > best_ll:
            best_ll = fr_final.log_likelihood
            best_params = params
            best_result = fr_final

    return best_params, best_result


# ── Threshold AR (TAR) ────────────────────────────────────────────────────────

def _tar_residuals(
    y: np.ndarray,
    threshold: float,
    delay: int,
    ar_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute TAR regime indicator and OLS residuals for given threshold."""
    p = ar_order
    d = delay
    T = len(y)
    start = max(p, d)

    regime = (y[start - d: T - d] <= threshold).astype(float)  # 1 = low regime
    X_low = []
    X_high = []
    y_dep = []

    for t in range(start, T):
        lags = y[t - p:t][::-1]  # y_{t-1}, ..., y_{t-p}
        if regime[t - start]:
            X_low.append(np.concatenate([[1.0], lags]))
            X_high.append(np.zeros(p + 1))
        else:
            X_high.append(np.concatenate([[1.0], lags]))
            X_low.append(np.zeros(p + 1))
        y_dep.append(y[t])

    X_low = np.array(X_low)
    X_high = np.array(X_high)
    y_dep = np.array(y_dep)
    return X_low, X_high, y_dep, regime


def estimate_tar(
    y: np.ndarray,
    ar_order: int = 1,
    delay: int = 1,
    n_threshold_grid: int = 100,
) -> TARParams:
    """
    Estimate TAR model by grid search over threshold values.
    Threshold is chosen to minimize total SSR.
    """
    p = ar_order
    d = delay
    start = max(p, d)
    T = len(y)

    # Grid of threshold candidates (middle 70% of delay-lagged values)
    thresh_series = y[d:T - start + d]
    lo, hi = np.percentile(thresh_series, 15), np.percentile(thresh_series, 85)
    grid = np.linspace(lo, hi, n_threshold_grid)

    best_ssr = np.inf
    best_thresh = grid[len(grid) // 2]
    best_coefs_low = None
    best_coefs_high = None

    for thresh in grid:
        X_low, X_high, y_dep, _ = _tar_residuals(y, thresh, d, p)
        X = np.column_stack([X_low, X_high])
        try:
            coefs, residuals, _, _ = np.linalg.lstsq(X, y_dep, rcond=None)
            ssr = np.sum((y_dep - X @ coefs)**2) if len(residuals) == 0 else residuals[0]
            if not np.isscalar(ssr):
                ssr = np.sum((y_dep - X @ coefs)**2)
            if ssr < best_ssr:
                best_ssr = ssr
                best_thresh = thresh
                best_coefs_low = coefs[:p + 1]
                best_coefs_high = coefs[p + 1:]
        except np.linalg.LinAlgError:
            continue

    # Estimate regime-specific sigmas
    if best_coefs_low is not None:
        X_low, X_high, y_dep, regime = _tar_residuals(y, best_thresh, d, p)
        X = np.column_stack([X_low, X_high])
        fitted = X @ np.concatenate([best_coefs_low, best_coefs_high])
        res = y_dep - fitted
        mask_low = regime.astype(bool)
        mask_high = ~mask_low
        sigma_low = res[mask_low].std() if mask_low.sum() > 1 else 0.01
        sigma_high = res[mask_high].std() if mask_high.sum() > 1 else 0.01
    else:
        best_coefs_low = np.zeros(p + 1)
        best_coefs_high = np.zeros(p + 1)
        sigma_low = y.std()
        sigma_high = y.std()

    return TARParams(
        threshold=best_thresh,
        ar_coefs_low=best_coefs_low[1:],
        ar_coefs_high=best_coefs_high[1:],
        mu_low=float(best_coefs_low[0]),
        mu_high=float(best_coefs_high[0]),
        sigma_low=float(sigma_low),
        sigma_high=float(sigma_high),
        delay=d,
    )


# ── Smooth Transition AR (STAR) ───────────────────────────────────────────────

def logistic_transition(y_lagged: float, gamma: float, c: float) -> float:
    """Logistic transition function F(y; gamma, c) = 1 / (1 + exp(-gamma*(y-c)))."""
    return 1.0 / (1.0 + math.exp(-gamma * (y_lagged - c)))


def exponential_transition(y_lagged: float, gamma: float, c: float) -> float:
    """Exponential transition: F = 1 - exp(-gamma * (y-c)^2)."""
    return 1.0 - math.exp(-gamma * (y_lagged - c)**2)


def star_predict(
    params: STARParams,
    y_history: np.ndarray,
) -> float:
    """One-step-ahead STAR prediction."""
    p = len(params.ar_coefs_1)
    y_lag = y_history[-params.delay]
    lags = y_history[-p:][::-1]

    if params.transition_type == "logistic":
        G = logistic_transition(y_lag, params.gamma, params.c)
    else:
        G = exponential_transition(y_lag, params.gamma, params.c)

    y_hat = (params.mu_1 + params.ar_coefs_1 @ lags) * (1.0 - G) + \
            (params.mu_2 + params.ar_coefs_2 @ lags) * G
    return y_hat


def estimate_star(
    y: np.ndarray,
    ar_order: int = 1,
    delay: int = 1,
    transition_type: str = "logistic",
) -> STARParams:
    """
    Estimate STAR model via nonlinear least squares.
    Linearization approach with grid search for starting values of gamma, c.
    """
    p = ar_order
    d = delay
    T = len(y)
    start = max(p, d)

    y_dep = y[start:]
    n = len(y_dep)

    c_grid = np.percentile(y, np.linspace(10, 90, 10))
    gamma_grid = np.array([1.0, 5.0, 10.0, 50.0])

    best_ssr = np.inf
    best_p = None

    for c0 in c_grid:
        for gamma0 in gamma_grid:
            def residuals(params_flat):
                mu1, mu2 = params_flat[0], params_flat[1]
                ar1 = params_flat[2:2 + p]
                ar2 = params_flat[2 + p:2 + 2 * p]
                gamma = abs(params_flat[-2])
                c = params_flat[-1]
                res = np.empty(n)
                for i, t in enumerate(range(start, T)):
                    lags = y[t - p:t][::-1]
                    y_lag = y[t - d]
                    if transition_type == "logistic":
                        try:
                            G = logistic_transition(y_lag, gamma, c)
                        except OverflowError:
                            G = 0.0 if gamma * (y_lag - c) < 0 else 1.0
                    else:
                        G = exponential_transition(y_lag, gamma, c)
                    yhat = (mu1 + ar1 @ lags) * (1.0 - G) + (mu2 + ar2 @ lags) * G
                    res[i] = y[t] - yhat
                return res

            x0 = np.zeros(4 + 2 * p)
            x0[0] = y[:T//2].mean()
            x0[1] = y[T//2:].mean()
            x0[-2] = gamma0
            x0[-1] = c0

            try:
                result = optimize.least_squares(residuals, x0, max_nfev=1000)
                ssr = (result.fun**2).sum()
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_p = result.x
            except Exception:
                continue

    if best_p is None:
        best_p = np.zeros(4 + 2 * p)

    mu1, mu2 = best_p[0], best_p[1]
    ar1 = best_p[2:2 + p]
    ar2 = best_p[2 + p:2 + 2 * p]
    gamma = abs(best_p[-2])
    c = best_p[-1]

    return STARParams(
        ar_coefs_1=ar1,
        ar_coefs_2=ar2,
        mu_1=float(mu1),
        mu_2=float(mu2),
        gamma=float(gamma),
        c=float(c),
        delay=d,
        transition_type=transition_type,
    )


# ── Markov-Switching GARCH ────────────────────────────────────────────────────

def ms_garch_filter(
    returns: np.ndarray,
    params: MSGARCHParams,
) -> Tuple[np.ndarray, float]:
    """
    Filter Markov-switching GARCH model.
    Returns (filtered_state_probs (T, n_states), log_likelihood).
    Uses Gray (1996) collapsing approximation for tractability.
    """
    T = len(returns)
    K = params.n_states
    P_trans = params.P

    # Initialize stationary distribution
    try:
        eigvals, eigvecs = np.linalg.eig(P_trans.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()
    except Exception:
        pi = np.ones(K) / K

    filtered = np.zeros((T, K))
    log_lik = 0.0

    # Initial variance: unconditional GARCH variance for each state
    h = np.array([
        params.omega[k] / max(1.0 - params.alpha[k] - params.beta[k], 1e-4)
        for k in range(K)
    ])

    prev_filt = pi.copy()

    for t in range(T):
        r = returns[t]
        # Predicted probs
        pred = P_trans.T @ prev_filt
        pred = np.maximum(pred, 1e-300)

        # Collapsed variance (weighted sum across states)
        h_collapse = float(np.dot(prev_filt, h))

        # Update state-specific variances
        h_new = np.array([
            params.omega[k] + params.alpha[k] * r**2 + params.beta[k] * h_collapse
            for k in range(K)
        ])
        h_new = np.maximum(h_new, 1e-10)

        # Emission probabilities
        eta = np.array([
            math.exp(-0.5 * r**2 / h_new[k]) / math.sqrt(2.0 * math.pi * h_new[k])
            for k in range(K)
        ])

        joint = pred * eta
        total = joint.sum()
        if total <= 0:
            total = 1e-300
        filtered[t] = joint / total
        log_lik += math.log(total)
        h = h_new
        prev_filt = filtered[t].copy()

    return filtered, log_lik


# ── BOCPD: Bayesian Online Changepoint Detection ──────────────────────────────

def bocpd(
    data: np.ndarray,
    hazard_rate: float = 0.01,
    mu0: float = 0.0,
    kappa0: float = 1.0,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Tuple[np.ndarray, List[ChangePoint]]:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay 2007).
    Uses Normal-Inverse-Gamma conjugate prior for Gaussian observations.

    Parameters
    ----------
    data : 1D time series
    hazard_rate : prior probability of changepoint at each step
    mu0, kappa0, alpha0, beta0 : NIG hyperparameters

    Returns
    -------
    run_length_probs : (T, T) posterior over run lengths
    changepoints : list of detected ChangePoint events
    """
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Sufficient statistics for each possible run length
    mu_arr = np.full(T + 1, mu0)
    kappa_arr = np.full(T + 1, kappa0)
    alpha_arr = np.full(T + 1, alpha0)
    beta_arr = np.full(T + 1, beta0)

    changepoints = []
    max_run_probs = np.zeros(T)

    for t in range(T):
        x = data[t]

        # Predictive probability under Student-t distribution
        # p(x_t | run length r) = t_{2*alpha}(x | mu, beta*(kappa+1)/(alpha*kappa))
        pred_alpha = alpha_arr[:t + 1]
        pred_beta = beta_arr[:t + 1]
        pred_kappa = kappa_arr[:t + 1]
        pred_mu = mu_arr[:t + 1]

        df = 2.0 * pred_alpha
        scale = np.sqrt(pred_beta * (pred_kappa + 1.0) / (pred_alpha * pred_kappa))
        # t-distribution PDF
        t_pdf = stats.t.pdf(x, df=df, loc=pred_mu, scale=scale)
        t_pdf = np.maximum(t_pdf, 1e-300)

        # Growth probabilities: R[t+1, r+1] = R[t, r] * P(x|r) * (1 - hazard)
        R[t + 1, 1:t + 2] = R[t, :t + 1] * t_pdf * (1.0 - hazard_rate)

        # Changepoint probability: R[t+1, 0] = sum_r R[t, r] * P(x|r) * hazard
        R[t + 1, 0] = np.sum(R[t, :t + 1] * t_pdf) * hazard_rate

        # Normalize
        total = R[t + 1, :t + 2].sum()
        if total > 0:
            R[t + 1, :t + 2] /= total

        # Update sufficient statistics for each run length
        # New run (r=0): reset to prior
        mu_arr[0] = mu0
        kappa_arr[0] = kappa0
        alpha_arr[0] = alpha0
        beta_arr[0] = beta0

        # Extend existing runs (r → r+1)
        kappa_new = pred_kappa + 1.0
        mu_new = (pred_kappa * pred_mu + x) / kappa_new
        alpha_new = pred_alpha + 0.5
        beta_new = pred_beta + pred_kappa * (x - pred_mu)**2 / (2.0 * kappa_new)

        mu_arr[1:t + 2] = mu_new
        kappa_arr[1:t + 2] = kappa_new
        alpha_arr[1:t + 2] = alpha_new
        beta_arr[1:t + 2] = beta_new

        # Detect changepoint: high probability of run length = 0
        cp_prob = R[t + 1, 0]
        max_run_probs[t] = cp_prob
        if cp_prob > 0.5 and t > 0:
            changepoints.append(ChangePoint(
                index=t,
                probability=float(cp_prob),
                regime_before=0,  # simplified: single regime tracking not implemented
                regime_after=1,
            ))

    return R, changepoints


# ── Bai-Perron Structural Breaks ─────────────────────────────────────────────

def bai_perron_breaks(
    y: np.ndarray,
    max_breaks: int = 5,
    min_segment: int = 15,
) -> StructuralBreakResult:
    """
    Bai-Perron multiple structural break test.
    Minimizes global SSR via dynamic programming.
    Returns break dates and BIC-selected number of breaks.

    Parameters
    ----------
    y : time series (demeaned or raw)
    max_breaks : maximum number of breaks to consider
    min_segment : minimum segment length between breaks
    """
    T = len(y)

    # Compute SSR matrix: SSR[i,j] = SSR of segment y[i:j]
    # Use vectorized computation
    SSR = np.full((T, T), np.inf)
    for i in range(T):
        for j in range(i + min_segment, T + 1):
            seg = y[i:j]
            SSR[i, j - 1] = np.sum((seg - seg.mean())**2)

    # Dynamic programming to find optimal break configuration
    # V[k, t] = minimum SSR using k+1 segments up to time t
    V = np.full((max_breaks + 1, T), np.inf)
    breaks_dp = np.zeros((max_breaks + 1, T), dtype=int)

    # Initialize: 0 breaks = 1 segment
    for t in range(min_segment - 1, T):
        V[0, t] = SSR[0, t]

    # Fill DP table
    for k in range(1, max_breaks + 1):
        for t in range((k + 1) * min_segment - 1, T):
            for s in range(k * min_segment - 1, t - min_segment + 1):
                val = V[k - 1, s] + SSR[s + 1, t]
                if val < V[k, t]:
                    V[k, t] = val
                    breaks_dp[k, t] = s + 1  # break at index s+1

    # Select number of breaks by BIC
    best_bic = np.inf
    best_m = 0
    sigma2 = np.var(y)

    for m in range(max_breaks + 1):
        if V[m, T - 1] == np.inf:
            continue
        n_params = 2 * (m + 1)  # m+1 means, m+1 sigmas (simplified)
        bic = T * math.log(V[m, T - 1] / T + 1e-10) + n_params * math.log(T)
        if bic < best_bic:
            best_bic = bic
            best_m = m

    # Backtrack to find break dates
    def backtrack(m, T_end):
        if m == 0:
            return []
        bp = breaks_dp[m, T_end - 1]
        return backtrack(m - 1, bp) + [bp]

    break_dates = backtrack(best_m, T)

    # Compute segment means
    boundaries = [0] + break_dates + [T]
    segment_means = np.array([y[boundaries[i]:boundaries[i + 1]].mean()
                               for i in range(len(boundaries) - 1)])

    # F-statistic (simplified Chow-style)
    ssr_full = V[best_m, T - 1]
    ssr_null = SSR[0, T - 1]
    if ssr_full > 0 and best_m > 0:
        f_stat = ((ssr_null - ssr_full) / (best_m * 2)) / (ssr_full / (T - 2 * (best_m + 1)))
    else:
        f_stat = 0.0

    return StructuralBreakResult(
        break_dates=break_dates,
        f_statistic=float(f_stat),
        critical_value=8.85,  # approximate 5% critical value for 1 break
        n_breaks=best_m,
        segment_means=segment_means,
        bic=float(best_bic),
    )


# ── Regime Duration Distribution ─────────────────────────────────────────────

def regime_duration(params: HamiltonParams) -> RegimeDuration:
    """
    Compute expected regime durations from transition matrix.
    Under Markov chain, duration in state k is geometric with p = P[k,k].
    E[duration_k] = 1 / (1 - P[k,k])
    Std[duration_k] = sqrt(P[k,k]) / (1 - P[k,k])
    """
    K = params.n_states
    P = params.P
    expected = np.array([1.0 / max(1.0 - P[k, k], 1e-6) for k in range(K)])
    std = np.array([math.sqrt(P[k, k]) / max(1.0 - P[k, k], 1e-6) for k in range(K)])

    # Stationary distribution
    try:
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = np.real(eigvecs[:, idx])
        pi = np.abs(pi) / np.abs(pi).sum()
    except Exception:
        pi = np.ones(K) / K

    return RegimeDuration(
        expected_duration=expected,
        std_duration=std,
        regime_probs=pi,
    )


def regime_conditional_moments(
    y: np.ndarray,
    filtered_probs: np.ndarray,
    n_states: int = 2,
) -> dict:
    """
    Compute regime-conditional moments of y using filtered probabilities.
    Returns dict with mean, variance, skewness per regime.
    """
    results = {}
    for k in range(n_states):
        w = filtered_probs[:, k]
        w_sum = w.sum() + 1e-10
        mean_k = (w * y).sum() / w_sum
        var_k = (w * (y - mean_k)**2).sum() / w_sum
        skew_k = (w * (y - mean_k)**3).sum() / (w_sum * max(var_k**1.5, 1e-10))
        results[f"regime_{k}"] = {
            "mean": float(mean_k),
            "variance": float(var_k),
            "std": float(math.sqrt(max(var_k, 0.0))),
            "skewness": float(skew_k),
            "weight": float(w_sum / len(y)),
        }
    return results
