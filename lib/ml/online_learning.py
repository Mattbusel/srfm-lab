"""
Online learning algorithms for non-stationary financial markets.

Algorithms:
  - FTRL  (Follow-the-Regularized-Leader)
  - ONS   (Online Newton Step)
  - EG    (Exponentiated Gradient)
  - Hedge (Exponential Weights / Expert Combination)
  - AdaGrad, Adam
  - Passive-Aggressive (PA) regression
  - Cover's Universal Portfolio
  - ADWIN concept drift detector
  - Online Sharpe gradient ascent
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _simplex_project(v: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = int(np.nonzero(u > (cssv - 1.0) / np.arange(1, n + 1))[0][-1])
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _log_barrier_grad(w: np.ndarray) -> np.ndarray:
    return -1.0 / np.maximum(w, 1e-15)


# ---------------------------------------------------------------------------
# FTRL — Follow-the-Regularized-Leader
# ---------------------------------------------------------------------------

class FTRL:
    """
    FTRL with L2 regularization for portfolio weight updates.
    w_{t+1} = argmin_{w in simplex} [eta * sum_{s<=t} <g_s, w> + ||w||^2 / 2]
    """

    def __init__(self, n: int, eta: float = 0.01, lambda_: float = 1.0):
        self.n = n
        self.eta = eta
        self.lambda_ = lambda_
        self.cumgrad = np.zeros(n)
        self.weights = np.ones(n) / n
        self.t = 0

    def update(self, gradient: np.ndarray) -> np.ndarray:
        """Receive gradient loss, update cumulative gradient, return new weights."""
        self.cumgrad += gradient
        self.t += 1
        # Closed-form for L2: w* = simplex_proj( -eta * cumgrad / lambda_ )
        w_unconstrained = -self.eta * self.cumgrad / self.lambda_
        self.weights = _simplex_project(w_unconstrained)
        return self.weights.copy()

    def predict(self) -> np.ndarray:
        return self.weights.copy()

    def regret_bound(self) -> float:
        """O(sqrt(T)) regret bound for L2 regularization."""
        if self.t == 0:
            return 0.0
        return self.lambda_ / (2 * self.eta) + self.eta * self.t / 2.0

    def reset(self):
        self.cumgrad[:] = 0.0
        self.weights[:] = 1.0 / self.n
        self.t = 0


# ---------------------------------------------------------------------------
# ONS — Online Newton Step
# ---------------------------------------------------------------------------

class ONS:
    """
    Online Newton Step for log-loss minimization (Hazan et al. 2007).
    Achieves O(log T) regret on exp-concave losses.
    """

    def __init__(self, n: int, eta: float = 0.0, eps: float = 1e-8,
                 beta: float = 1.0):
        self.n = n
        self.beta = beta  # learning rate for ONS
        self.eps = eps
        self.A = np.eye(n) * eps  # accumulated outer products + eps*I
        self.b = np.zeros(n)
        self.weights = np.ones(n) / n
        self.t = 0

    def update(self, gradient: np.ndarray) -> np.ndarray:
        """
        gradient: loss gradient at current weights.
        Uses Newton step: A^{-1} g, projected to simplex.
        """
        self.t += 1
        self.A += np.outer(gradient, gradient)
        self.b += gradient
        try:
            A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.n) / self.eps
        w_newton = _simplex_project(self.weights - (1.0 / self.beta) * A_inv @ gradient)
        self.weights = w_newton
        return self.weights.copy()

    def predict(self) -> np.ndarray:
        return self.weights.copy()

    def reset(self):
        self.A = np.eye(self.n) * self.eps
        self.b = np.zeros(self.n)
        self.weights = np.ones(self.n) / self.n
        self.t = 0


# ---------------------------------------------------------------------------
# EG — Exponentiated Gradient
# ---------------------------------------------------------------------------

class ExponentiatedGradient:
    """
    Exponentiated Gradient (Kivinen & Warmuth 1997).
    Multiplicative weight update preserving non-negativity.
    Suitable for portfolio weights over assets.
    """

    def __init__(self, n: int, eta: float = 0.01):
        self.n = n
        self.eta = eta
        self.weights = np.ones(n) / n
        self.t = 0

    def update(self, gradient: np.ndarray) -> np.ndarray:
        """gradient: loss (negative return) gradient at current weights."""
        self.t += 1
        # Multiplicative update
        log_w = np.log(np.maximum(self.weights, 1e-300)) - self.eta * gradient
        log_w -= np.max(log_w)  # numerical stability
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()
        return self.weights.copy()

    def predict(self) -> np.ndarray:
        return self.weights.copy()

    def regret_bound(self) -> float:
        """O(sqrt(T log n)) regret."""
        return math.sqrt(self.t * math.log(self.n) / (2 * self.eta)) if self.t > 0 else 0.0

    def reset(self):
        self.weights = np.ones(self.n) / self.n
        self.t = 0


# ---------------------------------------------------------------------------
# Hedge Algorithm (Exponential Weights for Expert Combination)
# ---------------------------------------------------------------------------

class Hedge:
    """
    Hedge algorithm (Freund & Schapire 1997).
    Maintains probability distribution over N experts.
    Guaranteed: regret <= sqrt(T log N / 2).
    """

    def __init__(self, n_experts: int, eta: Optional[float] = None,
                 T_horizon: int = 1000):
        self.n = n_experts
        self.T = T_horizon
        self.eta = eta if eta is not None else math.sqrt(2 * math.log(n_experts) / T_horizon)
        self.weights = np.ones(n_experts) / n_experts
        self.cumulative_loss = np.zeros(n_experts)
        self.t = 0

    def get_distribution(self) -> np.ndarray:
        return self.weights.copy()

    def update(self, expert_losses: np.ndarray) -> np.ndarray:
        """
        expert_losses: array of length n_experts, loss incurred by each expert.
        """
        self.t += 1
        self.cumulative_loss += expert_losses
        log_w = -self.eta * self.cumulative_loss
        log_w -= np.max(log_w)
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()
        return self.weights.copy()

    def expected_loss(self, expert_losses: np.ndarray) -> float:
        return float(self.weights @ expert_losses)

    def regret_bound(self) -> float:
        return math.sqrt(self.T * math.log(self.n) / 2.0)

    def reset(self):
        self.weights = np.ones(self.n) / self.n
        self.cumulative_loss = np.zeros(self.n)
        self.t = 0


# ---------------------------------------------------------------------------
# AdaGrad
# ---------------------------------------------------------------------------

class AdaGrad:
    """
    AdaGrad for online optimization of unconstrained parameters.
    Diagonal pre-conditioning by accumulated squared gradients.
    """

    def __init__(self, n: int, eta: float = 0.01, eps: float = 1e-8):
        self.n = n
        self.eta = eta
        self.eps = eps
        self.G = np.zeros(n)  # accumulated squared gradients
        self.theta = np.zeros(n)
        self.t = 0

    def step(self, gradient: np.ndarray) -> np.ndarray:
        self.t += 1
        self.G += gradient ** 2
        lr = self.eta / (np.sqrt(self.G) + self.eps)
        self.theta -= lr * gradient
        return self.theta.copy()

    def reset(self):
        self.G[:] = 0.0
        self.theta[:] = 0.0
        self.t = 0


# ---------------------------------------------------------------------------
# Adam (Online)
# ---------------------------------------------------------------------------

class Adam:
    """
    Adam optimizer for online parameter updates.
    """

    def __init__(self, n: int, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.0):
        self.n = n
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.theta = np.zeros(n)
        self.t = 0

    def step(self, gradient: np.ndarray) -> np.ndarray:
        self.t += 1
        g = gradient + self.weight_decay * self.theta
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return self.theta.copy()

    def reset(self):
        self.m[:] = 0.0
        self.v[:] = 0.0
        self.theta[:] = 0.0
        self.t = 0


# ---------------------------------------------------------------------------
# Passive-Aggressive (PA) Regression
# ---------------------------------------------------------------------------

class PassiveAggressive:
    """
    Passive-Aggressive algorithm for online regression (Crammer et al. 2006).
    PA-II variant with soft margin.
    Minimizes hinge loss: max(0, |y_hat - y| - epsilon).
    """

    def __init__(self, n: int, eps: float = 0.1, C: float = 1.0):
        self.n = n
        self.eps = eps
        self.C = C
        self.w = np.zeros(n)
        self.t = 0

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ x)

    def update(self, x: np.ndarray, y: float) -> float:
        """Update weights, return instantaneous loss."""
        self.t += 1
        y_hat = self.predict(x)
        loss = max(0.0, abs(y_hat - y) - self.eps)
        if loss == 0.0:
            return 0.0
        norm_sq = float(x @ x)
        if norm_sq < 1e-12:
            return loss
        # PA-II: tau = loss / (norm_sq + 1/(2C))
        tau = loss / (norm_sq + 1.0 / (2.0 * self.C))
        sign = 1.0 if y_hat < y else -1.0
        self.w += tau * sign * x
        return loss

    def reset(self):
        self.w[:] = 0.0
        self.t = 0


# ---------------------------------------------------------------------------
# Cover's Universal Portfolio
# ---------------------------------------------------------------------------

class UniversalPortfolio:
    """
    Cover's Universal Portfolio (1991).
    Achieves asymptotically optimal growth rate without market knowledge.
    Uses Monte Carlo sampling from Dirichlet to approximate integral.
    """

    def __init__(self, n_assets: int, n_samples: int = 1000,
                 random_seed: int = 42):
        self.n = n_assets
        self.n_samples = n_samples
        rng = np.random.default_rng(random_seed)
        # Sample portfolios uniformly from simplex via Dirichlet(1,...,1)
        self.portfolios = rng.dirichlet(np.ones(n_assets), size=n_samples)  # (S, n)
        self.wealth = np.ones(n_samples)  # wealth of each sampled portfolio
        self.t = 0

    def predict(self) -> np.ndarray:
        """Wealth-weighted average of sampled portfolios."""
        total = self.wealth.sum()
        if total < 1e-300:
            return np.ones(self.n) / self.n
        return (self.wealth[:, None] * self.portfolios).sum(axis=0) / total

    def update(self, returns: np.ndarray) -> np.ndarray:
        """
        returns: array of shape (n_assets,) of gross returns (1 + r).
        Updates internal wealth of each sampled portfolio.
        """
        self.t += 1
        port_returns = self.portfolios @ returns  # (S,)
        self.wealth *= port_returns
        # Prevent underflow: re-normalize wealth
        max_w = self.wealth.max()
        if max_w > 0:
            self.wealth /= max_w
        return self.predict()

    def log_wealth(self) -> float:
        """Log wealth of the universal portfolio."""
        total = self.wealth.sum()
        return float(np.log(total + 1e-300))

    def reset(self):
        self.wealth[:] = 1.0
        self.t = 0


# ---------------------------------------------------------------------------
# Tracking Regret
# ---------------------------------------------------------------------------

class TrackingRegret:
    """
    Tracks regret vs best switching strategy in hindsight.
    Uses the Fixed Share algorithm (Herbster & Warmuth 1998).
    """

    def __init__(self, n_experts: int, eta: float = 0.1,
                 alpha: float = 0.05):
        self.n = n_experts
        self.eta = eta
        self.alpha = alpha  # mixing parameter
        self.weights = np.ones(n_experts) / n_experts
        self.cumulative_algo_loss = 0.0
        self.t = 0

    def update(self, expert_losses: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns (new weights, algorithm loss at this step).
        """
        self.t += 1
        algo_loss = float(self.weights @ expert_losses)
        self.cumulative_algo_loss += algo_loss

        # Exponential weight update
        log_w = np.log(np.maximum(self.weights, 1e-300)) - self.eta * expert_losses
        log_w -= np.max(log_w)
        w_tilde = np.exp(log_w)
        w_tilde /= w_tilde.sum()

        # Mixing step (Fixed Share)
        self.weights = (1 - self.alpha) * w_tilde + self.alpha / self.n
        return self.weights.copy(), algo_loss

    def regret_vs_switching(self, best_losses: float,
                            n_switches: int) -> float:
        """
        Regret relative to best strategy with n_switches.
        Approximate bound: regret <= eta*T/8 + (n_switches+1)*log(n)/eta +
                                      n_switches * log(1/alpha)/eta
        """
        return (self.eta * self.t / 8.0
                + (n_switches + 1) * math.log(self.n) / self.eta
                + n_switches * math.log(1.0 / max(self.alpha, 1e-9)) / self.eta)

    def reset(self):
        self.weights = np.ones(self.n) / self.n
        self.cumulative_algo_loss = 0.0
        self.t = 0


# ---------------------------------------------------------------------------
# ADWIN — Adaptive Windowing for Concept Drift Detection
# ---------------------------------------------------------------------------

class ADWIN:
    """
    ADWIN algorithm (Bifet & Gavalda 2007) for concept drift detection.
    Maintains adaptive window of data; fires on mean shift.
    """

    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self._window: List[float] = []
        self._bucket_totals: List[float] = []
        self._bucket_counts: List[int] = []
        self.total = 0.0
        self.n = 0
        self.drift_detected = False
        self.n_detections = 0

    def add_element(self, value: float) -> bool:
        """
        Add observation. Returns True if drift was detected.
        """
        self._window.append(value)
        self.total += value
        self.n += 1
        self.drift_detected = self._detect()
        if self.drift_detected:
            self.n_detections += 1
        return self.drift_detected

    def _detect(self) -> bool:
        n = self.n
        if n < 2:
            return False
        total = self.total
        mean_total = total / n
        # Check all splits
        sum_left = 0.0
        for i in range(1, n):
            sum_left += self._window[i - 1]
            n0, n1 = i, n - i
            mu0 = sum_left / n0
            mu1 = (total - sum_left) / n1
            delta_mu = abs(mu0 - mu1)
            # Hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps_cut = math.sqrt(math.log(4 * n / self.delta) / (2 * m))
            if delta_mu >= eps_cut:
                # Shrink window: remove older portion
                self._window = self._window[i:]
                self.n = n - i
                self.total = total - sum_left
                return True
        return False

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n > 0 else 0.0

    @property
    def window_size(self) -> int:
        return self.n

    def reset(self):
        self._window.clear()
        self.total = 0.0
        self.n = 0
        self.drift_detected = False
        self.n_detections = 0


# ---------------------------------------------------------------------------
# Online Sharpe Maximization
# ---------------------------------------------------------------------------

class OnlineSharpeOptimizer:
    """
    Online gradient ascent to maximize a rolling Sharpe ratio.
    Maintains running estimates of mean and variance of portfolio returns.
    Uses projected gradient ascent onto the simplex.
    """

    def __init__(self, n_assets: int, eta: float = 0.01,
                 window: int = 60, eps: float = 1e-6):
        self.n = n_assets
        self.eta = eta
        self.window = window
        self.eps = eps
        self.weights = np.ones(n_assets) / n_assets
        self._return_history: List[float] = []
        self.t = 0
        # For Adam acceleration
        self._opt = Adam(n=n_assets, lr=eta)
        self._opt.theta = np.log(self.weights)  # parameterize in log-space

    def _sharpe_gradient(self, returns: np.ndarray) -> np.ndarray:
        """
        Gradient of Sharpe w.r.t. portfolio weights.
        r_p = w^T r,  SR = E[r_p] / std[r_p]
        dSR/dw = (1/sigma) * r - (mu/sigma^3) * (r - mu) * w
        Uses recent history for mu and sigma.
        """
        r_p = float(self.weights @ returns)
        self._return_history.append(r_p)
        if len(self._return_history) > self.window:
            self._return_history.pop(0)
        hist = np.array(self._return_history)
        mu = float(hist.mean())
        sigma = float(hist.std()) + self.eps
        # Gradient of Sharpe wrt weights (treating mu, sigma as fixed)
        grad = returns / sigma - mu * (r_p - mu) / sigma ** 3
        return grad  # ascending = maximize Sharpe

    def update(self, returns: np.ndarray) -> np.ndarray:
        """
        returns: array of shape (n_assets,) of realized returns (not gross).
        Returns updated portfolio weights.
        """
        self.t += 1
        grad = self._sharpe_gradient(returns)
        # Gradient ascent: subtract negative gradient
        neg_grad = -grad
        log_w = self._opt.step(neg_grad)
        # Softmax to get valid weights
        log_w -= np.max(log_w)
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()
        return self.weights.copy()

    def predict(self) -> np.ndarray:
        return self.weights.copy()

    def rolling_sharpe(self) -> float:
        if len(self._return_history) < 2:
            return 0.0
        hist = np.array(self._return_history)
        mu = hist.mean()
        sigma = hist.std() + self.eps
        return float(mu / sigma * math.sqrt(252))  # annualized

    def reset(self):
        self.weights = np.ones(self.n) / self.n
        self._return_history.clear()
        self._opt.reset()
        self.t = 0


# ---------------------------------------------------------------------------
# Composite Online Learner — combines all above via Hedge meta-weighting
# ---------------------------------------------------------------------------

class CompositeOnlineLearner:
    """
    Meta-learner that combines FTRL, EG, ONS, and Universal Portfolio
    via the Hedge algorithm. Dynamically allocates weight to best-performing
    base learner, adapting to concept drift.
    """

    def __init__(self, n_assets: int, eta_hedge: float = 0.05,
                 T_horizon: int = 2000):
        self.n = n_assets
        self.learners = {
            "FTRL": FTRL(n=n_assets, eta=0.01),
            "EG": ExponentiatedGradient(n=n_assets, eta=0.01),
            "ONS": ONS(n=n_assets, eps=1e-6, beta=1.0),
            "UP": UniversalPortfolio(n_assets=n_assets),
        }
        self.names = list(self.learners.keys())
        self.hedge = Hedge(n_experts=len(self.names), eta=eta_hedge,
                          T_horizon=T_horizon)
        self.adwin = ADWIN(delta=0.002)
        self.t = 0

    def predict(self) -> np.ndarray:
        """Return Hedge-weighted combination of base learner predictions."""
        d = self.hedge.get_distribution()
        combined = np.zeros(self.n)
        predictions = [self.learners[name].predict() for name in self.names]
        for w, pred in zip(d, predictions):
            combined += w * pred
        combined /= combined.sum() + 1e-15
        return combined

    def update(self, returns: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        returns: gross returns (1+r) for assets this period.
        gradient: loss gradient for gradient-based learners.
        """
        self.t += 1
        # Compute loss for each learner BEFORE updating
        expert_losses = np.zeros(len(self.names))
        for i, name in enumerate(self.names):
            pred = self.learners[name].predict()
            port_return = float(pred @ (returns - 1.0))  # excess return
            expert_losses[i] = -port_return  # loss = negative return

        # Update Hedge
        self.hedge.update(expert_losses)

        # Check for drift
        combined_loss = float(self.predict() @ (returns - 1.0))
        drift = self.adwin.add_element(-combined_loss)
        if drift:
            # Reset all learners on drift
            for learner in self.learners.values():
                learner.reset()
            self.hedge.reset()

        # Update base learners
        self.learners["FTRL"].update(gradient)
        self.learners["EG"].update(gradient)
        self.learners["ONS"].update(gradient)
        self.learners["UP"].update(returns)

        return self.predict()

    def status(self) -> dict:
        d = self.hedge.get_distribution()
        return {
            "t": self.t,
            "hedge_weights": dict(zip(self.names, d.tolist())),
            "drift_count": self.adwin.n_detections,
            "window_size": self.adwin.window_size,
        }
