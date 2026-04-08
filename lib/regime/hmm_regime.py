"""
HMM Regime Detector (T2-8)
3 states: 0=bull (low vol, positive drift), 1=bear (high vol, negative drift), 2=sideways (medium vol, zero drift)

Uses Baum-Welch EM for parameter estimation, Viterbi for state decoding.
Modulates signal stack weights: bull→momentum, bear→reduce size, sideways→mean reversion.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HMMRegimeConfig:
    n_states: int = 3
    min_history: int = 500  # bars before HMM is reliable
    refit_every: int = 1000  # refit params every N bars
    emission_update_alpha: float = 0.05  # EMA for online emission updates

class HMMRegimeDetector:
    """
    Gaussian emission HMM with 3 states.

    State signal weights returned:
      bull:     bh_weight=1.3, ou_weight=0.7, size_scale=1.0
      bear:     bh_weight=0.8, ou_weight=0.8, size_scale=0.6
      sideways: bh_weight=0.7, ou_weight=1.4, size_scale=0.8
    """
    STATE_WEIGHTS = {
        0: {"bh": 1.3, "ou": 0.7, "size": 1.0, "label": "bull"},
        1: {"bh": 0.8, "ou": 0.8, "size": 0.6, "label": "bear"},
        2: {"bh": 0.7, "ou": 1.4, "size": 0.8, "label": "sideways"},
    }

    def __init__(self, cfg: HMMRegimeConfig = None):
        self.cfg = cfg or HMMRegimeConfig()
        self._returns: list[float] = []
        self._bars_since_fit: int = 0
        self._state: int = 0
        self._state_probs: list[float] = [1/3, 1/3, 1/3]
        self._fitted = False

        # HMM parameters (initialized to reasonable defaults)
        n = self.cfg.n_states
        self._trans = np.full((n, n), 1/n)  # transition matrix
        self._means = np.array([-0.001, -0.0005, 0.0008])  # bear, sideways, bull drift
        self._stds = np.array([0.025, 0.015, 0.012])

    def update(self, garch_filtered_return: float) -> dict:
        """
        Update with one new GARCH-filtered return.
        Returns dict with: state (int), label (str), weights (dict), probs (list)
        """
        self._returns.append(garch_filtered_return)
        self._bars_since_fit += 1

        if len(self._returns) < self.cfg.min_history:
            return self._current_output()

        # Refit periodically
        if self._bars_since_fit >= self.cfg.refit_every or not self._fitted:
            self._fit()
            self._bars_since_fit = 0

        # Online Viterbi step (single-step forward filter)
        self._forward_step(garch_filtered_return)
        return self._current_output()

    def _fit(self):
        """Fit HMM parameters using simplified Baum-Welch on recent history."""
        data = np.array(self._returns[-min(len(self._returns), 5000):])
        # Sort states by mean return (bear=lowest, sideways=mid, bull=highest)
        # K-means-style initialization
        sorted_idx = np.argsort(data)
        n = len(data)
        k = self.cfg.n_states

        # Simple 3-quantile initialization for means/stds
        for i in range(k):
            segment = data[sorted_idx[i*n//k:(i+1)*n//k]]
            self._means[i] = float(np.mean(segment))
            self._stds[i] = float(max(np.std(segment), 1e-6))

        # Run 5 iterations of Baum-Welch
        for _ in range(5):
            # E-step: compute responsibilities
            log_lik = np.zeros((len(data), k))
            for j in range(k):
                from math import log, sqrt, pi, exp
                log_lik[:, j] = -0.5 * ((data - self._means[j]) / self._stds[j])**2 - np.log(self._stds[j]) - 0.5*np.log(2*np.pi)

            # Forward-backward (simplified: independent emissions)
            log_lik -= log_lik.max(axis=1, keepdims=True)
            responsibilities = np.exp(log_lik)
            responsibilities /= responsibilities.sum(axis=1, keepdims=True) + 1e-12

            # M-step
            Nk = responsibilities.sum(axis=0) + 1e-12
            self._means = (responsibilities * data[:, None]).sum(axis=0) / Nk
            self._stds = np.sqrt((responsibilities * (data[:, None] - self._means)**2).sum(axis=0) / Nk)
            self._stds = np.maximum(self._stds, 1e-6)

        # Sort states by mean: index 0 = most negative (bear), 2 = most positive (bull)
        order = np.argsort(self._means)
        self._means = self._means[order]
        self._stds = self._stds[order]
        self._fitted = True

    def _forward_step(self, obs: float):
        """Single-step Bayesian update of state probabilities."""
        k = self.cfg.n_states
        # Emission probabilities
        log_emit = np.zeros(k)
        for j in range(k):
            z = (obs - self._means[j]) / (self._stds[j] + 1e-12)
            log_emit[j] = -0.5 * z * z - np.log(self._stds[j] + 1e-12)
        log_emit -= log_emit.max()
        emit = np.exp(log_emit)

        # Predict + update
        prior = np.array(self._state_probs) @ self._trans
        posterior = prior * emit
        posterior /= posterior.sum() + 1e-12

        self._state_probs = posterior.tolist()
        self._state = int(np.argmax(posterior))

    def _current_output(self) -> dict:
        weights = self.STATE_WEIGHTS[self._state]
        return {
            "state": self._state,
            "label": weights["label"],
            "bh_weight": weights["bh"],
            "ou_weight": weights["ou"],
            "size_scale": weights["size"],
            "probs": self._state_probs,
        }

    @property
    def current_state(self) -> int:
        return self._state

    @property
    def current_label(self) -> str:
        return self.STATE_WEIGHTS[self._state]["label"]
