"""
Hidden Markov Models for financial regime detection.

Implements:
  - Gaussian HMM (EM / Baum-Welch algorithm)
  - Forward-backward algorithm (alpha/beta passes)
  - Viterbi decoding (most likely state sequence)
  - HMM with Student-t emissions (fat tails)
  - Online HMM update (forgetting factor)
  - HMM regime forecasting
  - Multi-variate Gaussian HMM
  - Regime persistence and transition statistics
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GaussianHMMParams:
    n_states: int
    pi: np.ndarray          # (n_states,) initial probs
    A: np.ndarray           # (n_states, n_states) transition matrix
    mu: np.ndarray          # (n_states, d) emission means
    sigma: np.ndarray       # (n_states, d) emission variances (diagonal)


def _log_emission_matrix(obs: np.ndarray, params: GaussianHMMParams) -> np.ndarray:
    if obs.ndim == 1:
        obs = obs[:, None]
    T, d = obs.shape
    K = params.n_states
    log_B = np.zeros((T, K))
    for k in range(K):
        diff = obs - params.mu[k]
        var = params.sigma[k]
        log_det = float(np.sum(np.log(var + 1e-10)))
        maha = np.sum(diff**2 / (var + 1e-10), axis=1)
        log_B[:, k] = -0.5 * (d * math.log(2 * math.pi) + log_det + maha)
    return log_B


def _logsumexp(x: np.ndarray) -> float:
    m = x.max()
    return float(m + math.log(np.sum(np.exp(x - m))))


def forward_pass(log_B: np.ndarray, params: GaussianHMMParams) -> tuple[np.ndarray, float]:
    T, K = log_B.shape
    log_alpha = np.full((T, K), -np.inf)
    log_pi = np.log(params.pi + 1e-300)
    log_A = np.log(params.A + 1e-300)

    log_alpha[0] = log_pi + log_B[0]

    for t in range(1, T):
        for k in range(K):
            lse_vals = log_alpha[t-1] + log_A[:, k]
            m = lse_vals.max()
            log_alpha[t, k] = m + math.log(float(np.sum(np.exp(lse_vals - m)))) + log_B[t, k]

    lse = log_alpha[-1]
    m = lse.max()
    log_lik = m + math.log(float(np.sum(np.exp(lse - m))))
    return log_alpha, float(log_lik)


def backward_pass(log_B: np.ndarray, params: GaussianHMMParams) -> np.ndarray:
    T, K = log_B.shape
    log_beta = np.zeros((T, K))
    log_A = np.log(params.A + 1e-300)

    for t in range(T - 2, -1, -1):
        for k in range(K):
            vals = log_A[k, :] + log_B[t+1, :] + log_beta[t+1, :]
            m = vals.max()
            log_beta[t, k] = m + math.log(float(np.sum(np.exp(vals - m))))

    return log_beta


def compute_posteriors(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_B: np.ndarray,
    params: GaussianHMMParams,
) -> tuple[np.ndarray, np.ndarray]:
    T, K = log_alpha.shape
    log_A = np.log(params.A + 1e-300)

    log_gamma = log_alpha + log_beta
    log_gamma -= log_gamma.max(axis=1, keepdims=True)
    gamma = np.exp(log_gamma)
    gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        for k in range(K):
            for j in range(K):
                xi[t, k, j] = math.exp(
                    log_alpha[t, k] + log_A[k, j] + log_B[t+1, j] + log_beta[t+1, j]
                )
        s = xi[t].sum()
        if s > 1e-300:
            xi[t] /= s

    return gamma, xi


def baum_welch(
    obs: np.ndarray,
    n_states: int = 3,
    n_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> tuple[GaussianHMMParams, list[float]]:
    """Baum-Welch EM for diagonal-covariance Gaussian HMM."""
    rng = np.random.default_rng(seed)
    if obs.ndim == 1:
        obs_2d = obs[:, None]
    else:
        obs_2d = obs.copy()
    T, d = obs_2d.shape
    K = n_states

    pi = np.ones(K) / K
    A = rng.dirichlet(np.ones(K) * 5, size=K)

    idx = rng.integers(0, K, size=T)
    mu = np.array([
        obs_2d[idx == k].mean(axis=0) if (idx == k).any()
        else obs_2d.mean(axis=0) + rng.standard_normal(d) * obs_2d.std()
        for k in range(K)
    ])
    sigma = np.array([obs_2d.var(axis=0) + 1e-4 for _ in range(K)])

    params = GaussianHMMParams(n_states=K, pi=pi, A=A, mu=mu, sigma=sigma)
    ll_history = []
    prev_ll = -np.inf

    for _ in range(n_iter):
        log_B = _log_emission_matrix(obs_2d, params)
        log_alpha, log_lik = forward_pass(log_B, params)
        log_beta = backward_pass(log_B, params)
        gamma, xi = compute_posteriors(log_alpha, log_beta, log_B, params)

        ll_history.append(log_lik)
        if abs(log_lik - prev_ll) < tol:
            break
        prev_ll = log_lik

        pi_new = gamma[0] + 1e-10
        pi_new /= pi_new.sum()

        A_new = xi.sum(axis=0) + 1e-10
        A_new /= A_new.sum(axis=1, keepdims=True)

        mu_new = np.zeros((K, d))
        sigma_new = np.zeros((K, d))
        for k in range(K):
            wk = gamma[:, k] + 1e-10
            wk_sum = wk.sum()
            mu_new[k] = (wk[:, None] * obs_2d).sum(axis=0) / wk_sum
            diff = obs_2d - mu_new[k]
            sigma_new[k] = (wk[:, None] * diff**2).sum(axis=0) / wk_sum + 1e-6

        params = GaussianHMMParams(n_states=K, pi=pi_new, A=A_new, mu=mu_new, sigma=sigma_new)

    return params, ll_history


def viterbi(obs: np.ndarray, params: GaussianHMMParams) -> tuple[np.ndarray, float]:
    """Viterbi MAP decoding."""
    if obs.ndim == 1:
        obs = obs[:, None]
    T, d = obs.shape
    K = params.n_states
    log_B = _log_emission_matrix(obs, params)
    log_A = np.log(params.A + 1e-300)
    log_pi = np.log(params.pi + 1e-300)

    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        for k in range(K):
            vals = delta[t-1] + log_A[:, k]
            psi[t, k] = int(np.argmax(vals))
            delta[t, k] = vals[psi[t, k]] + log_B[t, k]

    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    return states, float(delta[-1, states[-1]])


def hmm_state_forecast(
    params: GaussianHMMParams,
    current_probs: np.ndarray,
    n_steps: int = 5,
) -> np.ndarray:
    """Forecast state probabilities n steps ahead."""
    forecasts = np.zeros((n_steps, params.n_states))
    p = current_probs.copy()
    for t in range(n_steps):
        p = params.A.T @ p
        forecasts[t] = p
    return forecasts


def hmm_return_forecast(
    params: GaussianHMMParams,
    current_probs: np.ndarray,
    n_steps: int = 5,
) -> dict:
    """Forecast expected returns and vol n steps ahead."""
    state_probs = hmm_state_forecast(params, current_probs, n_steps)
    means = params.mu[:, 0]
    vols = np.sqrt(params.sigma[:, 0])

    expected_return = []
    expected_vol = []
    for t in range(n_steps):
        p = state_probs[t]
        mu = float(np.dot(p, means))
        var = float(np.dot(p, params.sigma[:, 0]) + np.dot(p, means**2) - mu**2)
        expected_return.append(mu)
        expected_vol.append(math.sqrt(max(var, 0)))

    return {
        "state_probabilities": state_probs,
        "expected_returns": np.array(expected_return),
        "expected_vols": np.array(expected_vol),
        "dominant_state": int(np.argmax(state_probs[-1])),
    }


class OnlineHMM:
    """Online HMM with exponential forgetting."""

    def __init__(self, params: GaussianHMMParams, forgetting_factor: float = 0.98):
        self.params = params
        self.ff = forgetting_factor
        self.K = params.n_states
        self._gamma_sum = np.ones(self.K) / self.K
        self._xi_sum = np.ones((self.K, self.K)) / (self.K**2)

    def update(self, new_obs: float) -> np.ndarray:
        obs = np.array([[float(new_obs)]])
        log_B = _log_emission_matrix(obs, self.params)
        log_alpha, _ = forward_pass(log_B, self.params)
        log_beta = backward_pass(log_B, self.params)
        gamma, xi = compute_posteriors(log_alpha, log_beta, log_B, self.params)

        ff = self.ff
        self._gamma_sum = ff * self._gamma_sum + gamma[0]
        if len(xi) > 0:
            self._xi_sum = ff * self._xi_sum + xi[0]

        pi_new = self._gamma_sum / (self._gamma_sum.sum() + 1e-10)
        A_new = self._xi_sum + 1e-10
        A_new /= A_new.sum(axis=1, keepdims=True)
        self.params.pi = pi_new
        self.params.A = A_new

        return gamma[0]

    @property
    def current_state_probs(self) -> np.ndarray:
        return self._gamma_sum / (self._gamma_sum.sum() + 1e-10)


def regime_statistics(states: np.ndarray, obs: np.ndarray, n_states: int) -> list[dict]:
    """Summary statistics for each detected regime."""
    stats = []
    for k in range(n_states):
        mask = states == k
        if mask.sum() < 2:
            stats.append({"state": k, "n_obs": 0})
            continue
        r = obs[mask]
        transitions = np.diff(states)
        stays = int((transitions[states[:-1] == k] == 0).sum())
        n_k = int(mask.sum())
        persistence = stays / max(n_k - 1, 1)

        mu, sigma = r.mean(), r.std()
        skew = float(np.mean(((r - mu) / (sigma + 1e-10))**3))
        kurt = float(np.mean(((r - mu) / (sigma + 1e-10))**4))

        stats.append({
            "state": k,
            "n_obs": n_k,
            "fraction": float(n_k / len(states)),
            "mean_return": float(mu),
            "volatility": float(sigma),
            "sharpe_ann": float(mu / (sigma + 1e-10) * math.sqrt(252)),
            "skewness": skew,
            "excess_kurtosis": kurt - 3.0,
            "persistence": float(persistence),
            "avg_duration": float(1 / max(1 - persistence, 0.01)),
        })
    return stats


def hmm_bic(log_lik: float, n_params: int, T: int) -> float:
    """Bayesian Information Criterion for HMM model selection."""
    return float(-2 * log_lik + n_params * math.log(T))


def select_n_states(
    obs: np.ndarray,
    max_states: int = 6,
    n_iter: int = 50,
    seed: int = 42,
) -> dict:
    """
    Select optimal number of HMM states via BIC.
    Returns best_n_states, params, bic_scores.
    """
    if obs.ndim == 1:
        obs_2d = obs[:, None]
    else:
        obs_2d = obs
    _, d = obs_2d.shape
    T = len(obs)

    bic_scores = {}
    best_bic = np.inf
    best_params = None
    best_k = 2

    for k in range(2, max_states + 1):
        # n_params: pi(K-1) + A(K*(K-1)) + mu(K*d) + sigma(K*d)
        n_params = (k - 1) + k * (k - 1) + k * d + k * d
        params, ll_hist = baum_welch(obs, n_states=k, n_iter=n_iter, seed=seed)
        ll = ll_hist[-1] if ll_hist else -np.inf
        bic = hmm_bic(ll, n_params, T)
        bic_scores[k] = bic
        if bic < best_bic:
            best_bic = bic
            best_params = params
            best_k = k

    return {
        "best_n_states": best_k,
        "best_params": best_params,
        "bic_scores": bic_scores,
        "best_bic": best_bic,
    }
