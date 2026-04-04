"""
Hidden Markov Model for financial regime detection.

Implements:
- Gaussian HMM with Baum-Welch EM algorithm
- Viterbi algorithm for most-likely state sequence
- Regime-conditional statistics (mean, vol, Sharpe per regime)
- Forward filtering for real-time regime probabilities
- Multi-asset regime detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HMMResult:
    """Output of Gaussian HMM fitting."""
    n_states: int
    means: np.ndarray           # (n_states, n_features)
    covariances: np.ndarray     # (n_states, n_features, n_features)
    transition_matrix: np.ndarray  # (n_states, n_states) row-stochastic
    initial_probs: np.ndarray   # (n_states,)
    log_likelihood: float
    bic: float
    aic: float
    state_sequence: np.ndarray  # (T,) Viterbi most-likely states
    smoothed_probs: np.ndarray  # (T, n_states) forward-backward smoothed
    filtered_probs: np.ndarray  # (T, n_states) filtered (causal)
    n_iter: int                 # EM iterations


# ---------------------------------------------------------------------------
# Helper: Gaussian log-likelihood
# ---------------------------------------------------------------------------

def _log_gaussian(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Log-likelihood of x under N(mu, Sigma).

    Parameters
    ----------
    x : (d,) observed vector
    mu : (d,) mean
    Sigma : (d, d) covariance matrix

    Returns
    -------
    float
    """
    d = len(x)
    diff = x - mu
    try:
        sign, logdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return -1e10
        Sigma_inv = np.linalg.inv(Sigma)
        mahal = float(diff @ Sigma_inv @ diff)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)
    except np.linalg.LinAlgError:
        return -1e10


def _emission_log_probs(
    observations: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    """
    Compute emission log-probability matrix.

    Parameters
    ----------
    observations : (T, d)
    means : (K, d)
    covariances : (K, d, d)

    Returns
    -------
    np.ndarray (T, K)
    """
    T, d = observations.shape
    K = means.shape[0]
    log_B = np.full((T, K), -1e10)
    for k in range(K):
        for t in range(T):
            log_B[t, k] = _log_gaussian(observations[t], means[k], covariances[k])
    return log_B


# ---------------------------------------------------------------------------
# Forward-Backward algorithm
# ---------------------------------------------------------------------------

def _forward(
    log_B: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward algorithm in log-space.

    Returns
    -------
    log_alpha : (T, K)
    log_scale : (T,) log-scaling factors
    """
    T, K = log_B.shape
    log_alpha = np.full((T, K), -np.inf)

    # t=0
    log_alpha[0] = log_pi + log_B[0]
    log_scale = np.zeros(T)
    # Numerical stabilization: subtract max
    c = np.max(log_alpha[0])
    log_scale[0] = c
    log_alpha[0] -= c

    for t in range(1, T):
        for k in range(K):
            vals = log_alpha[t - 1] + log_A[:, k]
            log_alpha[t, k] = np.logaddexp.reduce(vals) + log_B[t, k]
        c = np.max(log_alpha[t])
        log_scale[t] = c
        log_alpha[t] -= c

    return log_alpha, log_scale


def _backward(
    log_B: np.ndarray,
    log_A: np.ndarray,
    log_scale: np.ndarray,
) -> np.ndarray:
    """
    Backward algorithm in log-space.

    Returns
    -------
    log_beta : (T, K)
    """
    T, K = log_B.shape
    log_beta = np.full((T, K), -np.inf)
    log_beta[T - 1] = 0.0

    for t in range(T - 2, -1, -1):
        for k in range(K):
            vals = log_A[k, :] + log_B[t + 1] + log_beta[t + 1]
            log_beta[t, k] = np.logaddexp.reduce(vals)
        log_beta[t] -= log_scale[t + 1]

    return log_beta


def _compute_gamma(
    log_alpha: np.ndarray, log_beta: np.ndarray
) -> np.ndarray:
    """(T, K) smoothed state probabilities."""
    log_gamma = log_alpha + log_beta
    # Normalize
    log_norm = np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    log_gamma -= log_norm
    return np.exp(log_gamma)


def _compute_xi(
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
    log_B: np.ndarray,
    log_A: np.ndarray,
) -> np.ndarray:
    """(T-1, K, K) pairwise state probabilities."""
    T, K = log_alpha.shape
    log_xi = np.full((T - 1, K, K), -np.inf)
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                log_xi[t, i, j] = (
                    log_alpha[t, i] + log_A[i, j] + log_B[t + 1, j] + log_beta[t + 1, j]
                )
        # Normalize over (i, j)
        log_norm = np.logaddexp.reduce(log_xi[t].ravel())
        log_xi[t] -= log_norm

    return np.exp(log_xi)


# ---------------------------------------------------------------------------
# Gaussian HMM
# ---------------------------------------------------------------------------

class GaussianHMM:
    """
    Gaussian Hidden Markov Model with Baum-Welch EM learning.

    Parameters
    ----------
    n_states : int
        Number of hidden states.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance (change in log-likelihood).
    covariance_type : str
        'full', 'diag', or 'tied'.
    reg_covar : float
        Regularization added to diagonal of covariance matrices.
    random_state : int
        Random seed for initialization.
    """

    def __init__(
        self,
        n_states: int = 2,
        n_iter: int = 200,
        tol: float = 1e-4,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _initialize(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Initialize HMM parameters via k-means-like assignment."""
        rng = np.random.default_rng(self.random_state)
        T, d = X.shape
        K = self.n_states

        # Initial state probs
        pi = np.ones(K) / K

        # Transition matrix: slight diagonal preference
        A = np.ones((K, K)) / K
        np.fill_diagonal(A, 0.7)
        A /= A.sum(axis=1, keepdims=True)

        # Initialize means: quantile-based assignment
        idx = np.argsort(X[:, 0])
        segment_size = T // K
        means = np.array([
            X[idx[i * segment_size:(i + 1) * segment_size]].mean(axis=0)
            for i in range(K)
        ])
        if len(means) < K:
            means = rng.normal(X.mean(axis=0), X.std(axis=0), size=(K, d))

        # Initialize covariances
        covariances = np.array([np.eye(d) * np.var(X, axis=0).mean() + self.reg_covar * np.eye(d)
                                 for _ in range(K)])

        return pi, A, means, covariances

    def _m_step_covariance(
        self,
        X: np.ndarray,
        gamma: np.ndarray,
        means: np.ndarray,
    ) -> np.ndarray:
        """M-step for covariance matrices."""
        T, d = X.shape
        K = self.n_states
        covariances = np.zeros((K, d, d))

        for k in range(K):
            Nk = gamma[:, k].sum() + 1e-10
            diff = X - means[k]  # (T, d)
            weighted_diff = gamma[:, k:k+1] * diff  # (T, d)
            cov = weighted_diff.T @ diff / Nk

            if self.covariance_type == "diag":
                cov = np.diag(np.diag(cov))
            elif self.covariance_type == "tied":
                # Will be averaged across states after this loop
                pass

            cov += self.reg_covar * np.eye(d)
            covariances[k] = cov

        if self.covariance_type == "tied":
            avg_cov = covariances.mean(axis=0)
            avg_cov += self.reg_covar * np.eye(d)
            covariances = np.stack([avg_cov] * K)

        return covariances

    def fit(self, observations: np.ndarray) -> HMMResult:
        """
        Fit Gaussian HMM via Baum-Welch EM.

        Parameters
        ----------
        observations : (T, d) or (T,) array
            Observed sequence.

        Returns
        -------
        HMMResult
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        X = observations.copy()
        T, d = X.shape
        K = self.n_states

        pi, A, means, covariances = self._initialize(X)
        log_A = np.log(A + 1e-10)
        log_pi = np.log(pi + 1e-10)

        log_likelihood = -np.inf
        n_iter = 0

        for iteration in range(self.n_iter):
            # E-step
            log_B = _emission_log_probs(X, means, covariances)
            log_alpha, log_scale = _forward(log_B, log_A, log_pi)
            log_beta = _backward(log_B, log_A, log_scale)

            gamma = _compute_gamma(log_alpha, log_beta)  # (T, K)
            xi = _compute_xi(log_alpha, log_beta, log_B, log_A)  # (T-1, K, K)

            # Log-likelihood
            new_ll = float(np.sum(log_scale))
            delta_ll = new_ll - log_likelihood
            log_likelihood = new_ll
            n_iter = iteration + 1

            if abs(delta_ll) < self.tol and iteration > 5:
                break

            # M-step
            pi = gamma[0] + 1e-10
            pi /= pi.sum()

            # Transition matrix
            A_new = xi.sum(axis=0) + 1e-10
            A_new /= A_new.sum(axis=1, keepdims=True)
            A = A_new

            log_A = np.log(A + 1e-10)
            log_pi = np.log(pi + 1e-10)

            # Means
            for k in range(K):
                Nk = gamma[:, k].sum() + 1e-10
                means[k] = (gamma[:, k:k+1] * X).sum(axis=0) / Nk

            covariances = self._m_step_covariance(X, gamma, means)

        # Viterbi decoding
        state_seq = self._viterbi(X, means, covariances, A, pi)

        # BIC/AIC
        n_params = K * d + K * d * (d + 1) // 2 + K * (K - 1) + (K - 1)
        bic = -2 * log_likelihood + n_params * np.log(T)
        aic = -2 * log_likelihood + 2 * n_params

        # Forward filter only (no backward: causal)
        filtered_probs = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
        filtered_probs /= filtered_probs.sum(axis=1, keepdims=True)

        # Sort states by mean return (ascending)
        if d == 1:
            order = np.argsort(means[:, 0])
            means = means[order]
            covariances = covariances[order]
            gamma = gamma[:, order]
            filtered_probs = filtered_probs[:, order]
            state_seq = np.array([int(np.where(order == s)[0][0]) for s in state_seq])
            # Reorder transition matrix
            A = A[order][:, order]
            pi = pi[order]

        return HMMResult(
            n_states=K,
            means=means,
            covariances=covariances,
            transition_matrix=A,
            initial_probs=pi,
            log_likelihood=log_likelihood,
            bic=bic,
            aic=aic,
            state_sequence=state_seq,
            smoothed_probs=gamma,
            filtered_probs=filtered_probs,
            n_iter=n_iter,
        )

    def _viterbi(
        self,
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        A: np.ndarray,
        pi: np.ndarray,
    ) -> np.ndarray:
        """Viterbi algorithm for most-likely state sequence."""
        T, d = X.shape
        K = self.n_states
        log_A = np.log(A + 1e-10)
        log_pi = np.log(pi + 1e-10)
        log_B = _emission_log_probs(X, means, covariances)

        log_delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)

        log_delta[0] = log_pi + log_B[0]

        for t in range(1, T):
            for k in range(K):
                vals = log_delta[t - 1] + log_A[:, k]
                psi[t, k] = int(np.argmax(vals))
                log_delta[t, k] = vals[psi[t, k]] + log_B[t, k]

        # Backtrack
        seq = np.zeros(T, dtype=int)
        seq[T - 1] = int(np.argmax(log_delta[T - 1]))
        for t in range(T - 2, -1, -1):
            seq[t] = psi[t + 1, seq[t + 1]]

        return seq

    def predict(
        self,
        new_obs: np.ndarray,
        result: HMMResult,
        return_probs: bool = False,
    ) -> np.ndarray:
        """
        Run forward filter on new observations using fitted parameters.

        Parameters
        ----------
        new_obs : (T, d) or (T,) array
        result : HMMResult
        return_probs : bool
            If True, return (T, K) probability matrix; else return (T,) states.

        Returns
        -------
        np.ndarray
        """
        if new_obs.ndim == 1:
            new_obs = new_obs.reshape(-1, 1)
        log_A = np.log(result.transition_matrix + 1e-10)
        log_pi = np.log(result.initial_probs + 1e-10)
        log_B = _emission_log_probs(new_obs, result.means, result.covariances)
        log_alpha, _ = _forward(log_B, log_A, log_pi)

        probs = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        if return_probs:
            return probs
        return np.argmax(probs, axis=1)


# ---------------------------------------------------------------------------
# Regime analysis wrapper
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    High-level wrapper for HMM-based regime detection.

    Parameters
    ----------
    n_states : int
        Number of market regimes.
    features : list of str
        Which features to use: 'returns', 'volatility', 'both'.
    vol_window : int
        Rolling window for realized volatility feature.
    regime_labels : list of str or None
        Custom labels for states (sorted by mean return).
    hmm_kwargs : dict
        Additional kwargs for GaussianHMM.
    """

    def __init__(
        self,
        n_states: int = 3,
        features: str = "both",
        vol_window: int = 21,
        regime_labels: Optional[List[str]] = None,
        **hmm_kwargs,
    ) -> None:
        self.n_states = n_states
        self.features = features
        self.vol_window = vol_window
        self.regime_labels = regime_labels
        self.hmm = GaussianHMM(n_states=n_states, **hmm_kwargs)
        self._result: Optional[HMMResult] = None

    def _build_features(self, price: pd.Series) -> np.ndarray:
        """Build observation matrix from price series."""
        ret = price.pct_change().dropna()
        vol = ret.rolling(self.vol_window, min_periods=5).std()

        if self.features == "returns":
            X = ret.values.reshape(-1, 1)
        elif self.features == "volatility":
            X = vol.dropna().values.reshape(-1, 1)
        else:  # both
            combined = pd.concat([ret, vol], axis=1).dropna()
            X = combined.values

        return X, ret.index if self.features != "volatility" else vol.dropna().index

    def fit(self, price: pd.Series) -> HMMResult:
        """
        Fit HMM to price series.

        Parameters
        ----------
        price : pd.Series
            Daily price series.

        Returns
        -------
        HMMResult
        """
        X, index = self._build_features(price)
        self._index = index
        self._result = self.hmm.fit(X)
        self._result._index = index
        return self._result

    def get_regime_series(
        self, price: pd.Series, result: Optional[HMMResult] = None
    ) -> pd.Series:
        """
        Return regime label series aligned to price index.

        Returns
        -------
        pd.Series
            Integer regime labels.
        """
        if result is None:
            result = self._result
        if result is None:
            raise ValueError("Call fit() first.")

        X, index = self._build_features(price)
        seq = self.hmm._viterbi(X, result.means, result.covariances,
                                 result.transition_matrix, result.initial_probs)

        regime_series = pd.Series(seq, index=index, name="regime")

        if self.regime_labels is not None and len(self.regime_labels) == self.n_states:
            regime_series = regime_series.map(
                {i: self.regime_labels[i] for i in range(self.n_states)}
            )

        return regime_series.reindex(price.index, method="ffill")

    def regime_statistics(
        self,
        price: pd.Series,
        result: Optional[HMMResult] = None,
    ) -> pd.DataFrame:
        """
        Compute return/vol/Sharpe statistics for each regime.

        Returns
        -------
        pd.DataFrame
        """
        regime = self.get_regime_series(price, result)
        ret = price.pct_change().dropna()
        combined = pd.concat([ret, regime], axis=1).dropna()
        combined.columns = ["return", "regime"]

        rows = []
        for state in sorted(combined["regime"].unique()):
            subset = combined[combined["regime"] == state]["return"]
            freq = len(subset) / len(combined)
            label = (self.regime_labels[int(state)]
                     if self.regime_labels and not isinstance(state, str)
                     else str(state))
            rows.append({
                "regime": label,
                "freq": round(freq, 4),
                "mean_annual": round(subset.mean() * 252, 4),
                "vol_annual": round(subset.std() * np.sqrt(252), 4),
                "sharpe": round(subset.mean() / (subset.std() + 1e-12) * np.sqrt(252), 4),
                "skew": round(float(subset.skew()), 4),
                "n_obs": len(subset),
            })

        return pd.DataFrame(rows).set_index("regime")

    def regime_transition_stats(
        self, price: pd.Series, result: Optional[HMMResult] = None
    ) -> Dict:
        """
        Expected duration and transition probabilities.

        Returns
        -------
        dict with 'expected_duration' and 'transition_matrix'.
        """
        if result is None:
            result = self._result
        A = result.transition_matrix
        expected_durations = 1 / (1 - np.diag(A) + 1e-10)
        return {
            "expected_duration_days": {f"state_{i}": round(float(d), 2)
                                        for i, d in enumerate(expected_durations)},
            "transition_matrix": pd.DataFrame(
                A.round(4),
                index=[f"from_{i}" for i in range(self.n_states)],
                columns=[f"to_{i}" for i in range(self.n_states)],
            ),
        }

    def select_n_states(
        self,
        price: pd.Series,
        max_states: int = 6,
        criterion: str = "bic",
    ) -> pd.DataFrame:
        """
        Fit HMMs with 2..max_states and compare by BIC or AIC.

        Returns
        -------
        pd.DataFrame
            n_states, log_likelihood, BIC, AIC per model.
        """
        X, _ = self._build_features(price)
        rows = []
        for k in range(2, max_states + 1):
            hmm_k = GaussianHMM(n_states=k, n_iter=self.hmm.n_iter, tol=self.hmm.tol)
            result_k = hmm_k.fit(X)
            rows.append({
                "n_states": k,
                "log_likelihood": round(result_k.log_likelihood, 2),
                "bic": round(result_k.bic, 2),
                "aic": round(result_k.aic, 2),
                "n_iter": result_k.n_iter,
            })
        df = pd.DataFrame(rows).set_index("n_states")
        optimal = df[criterion].idxmin()
        df["optimal"] = df.index == optimal
        return df

    def regime_conditional_backtest(
        self,
        price: pd.Series,
        factor_returns: pd.DataFrame,
        result: Optional[HMMResult] = None,
    ) -> pd.DataFrame:
        """
        Compute factor returns conditioned on each regime.

        Parameters
        ----------
        price : pd.Series
            Asset price series (used to determine regimes).
        factor_returns : pd.DataFrame
            (dates x factors) return series.
        result : HMMResult or None

        Returns
        -------
        pd.DataFrame
            (regimes x factors) mean annualized return.
        """
        regime = self.get_regime_series(price, result)
        combined = pd.concat([regime.rename("regime"), factor_returns], axis=1).dropna()

        rows = []
        for state in sorted(combined["regime"].unique()):
            subset = combined[combined["regime"] == state].drop(columns="regime")
            label = (self.regime_labels[int(state)]
                     if self.regime_labels and not isinstance(state, str)
                     else str(state))
            row = {"regime": label}
            for col in subset.columns:
                row[f"{col}_mean"] = round(subset[col].mean() * 252, 4)
                row[f"{col}_sharpe"] = round(
                    subset[col].mean() / (subset[col].std() + 1e-12) * np.sqrt(252), 4
                )
            rows.append(row)

        return pd.DataFrame(rows).set_index("regime")
