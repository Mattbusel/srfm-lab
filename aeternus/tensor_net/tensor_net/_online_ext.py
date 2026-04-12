"""Extension for online_learning.py — appended programmatically."""


# ---------------------------------------------------------------------------
# Section: Advanced online tensor learning
# ---------------------------------------------------------------------------

import numpy as np
import warnings
from collections import deque as _dq


class OnlineKalmanTT:
    """
    Kalman filter for a low-rank state-space model.

    Models the state evolution as::

        x_t = A x_{t-1} + w_t,   w_t ~ N(0, Q)
        y_t = C x_t + v_t,       v_t ~ N(0, R)

    Parameters
    ----------
    state_dim : int
        Latent state dimension.
    obs_dim : int
        Observation dimension.
    process_noise : float
        Diagonal process noise variance.
    obs_noise : float
        Diagonal observation noise variance.
    """

    def __init__(
        self,
        state_dim: int = 8,
        obs_dim: int = 64,
        process_noise: float = 1e-3,
        obs_noise: float = 1e-2,
    ) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self._A = np.eye(state_dim) * 0.95
        self._C = np.random.randn(obs_dim, state_dim) * 0.1
        self._Q = np.eye(state_dim) * process_noise
        self._R = np.eye(obs_dim) * obs_noise
        self._x = np.zeros(state_dim)
        self._P = np.eye(state_dim)
        self._n_updates = 0

    def predict(self) -> None:
        """Kalman predict step."""
        self._x = self._A @ self._x
        self._P = self._A @ self._P @ self._A.T + self._Q

    def update(self, y: np.ndarray) -> np.ndarray:
        """
        Kalman update step.

        Parameters
        ----------
        y : np.ndarray, shape (obs_dim,)

        Returns
        -------
        x_updated : np.ndarray, shape (state_dim,)
        """
        y_pred = self._C @ self._x
        innov = y - y_pred
        S = self._C @ self._P @ self._C.T + self._R
        K = self._P @ self._C.T @ np.linalg.inv(S)
        self._x = self._x + K @ innov
        self._P = (np.eye(self.state_dim) - K @ self._C) @ self._P
        self._n_updates += 1
        return self._x.copy()

    def filter(self, observations: np.ndarray) -> np.ndarray:
        """
        Run Kalman filter over a sequence.

        Parameters
        ----------
        observations : np.ndarray, shape (T, obs_dim)

        Returns
        -------
        states : np.ndarray, shape (T, state_dim)
        """
        T = observations.shape[0]
        states = np.zeros((T, self.state_dim), dtype=np.float32)
        for t in range(T):
            self.predict()
            x = self.update(observations[t])
            states[t] = x
        return states

    def reconstruct(self, states: np.ndarray) -> np.ndarray:
        """Reconstruct observations from latent states."""
        return (states @ self._C.T).astype(np.float32)

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def n_updates(self) -> int:
        return self._n_updates


class BayesianOnlineTT:
    """
    Bayesian online learning for TT core parameters.

    Maintains a mean-field Gaussian posterior over TT core elements and
    updates it incrementally using variational inference.

    Parameters
    ----------
    n_features : int
    n_components : int
    prior_std : float
    lr : float
    """

    def __init__(
        self,
        n_features: int = 64,
        n_components: int = 8,
        prior_std: float = 1.0,
        lr: float = 0.01,
    ) -> None:
        self.n_features = n_features
        self.n_components = n_components
        self.prior_std = prior_std
        self.lr = lr
        self._mu = np.zeros((n_features, n_components), dtype=np.float64)
        self._log_sigma2 = np.full((n_features, n_components), np.log(prior_std ** 2))
        self._n_updates = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Posterior mean prediction, shape (n_components,)."""
        return (x @ self._mu).astype(np.float32)

    def predict_with_uncertainty(self, x: np.ndarray) -> tuple:
        """Returns (mean, std) each shape (n_components,)."""
        mean = x @ self._mu
        sigma2 = np.exp(self._log_sigma2)
        pred_var = x ** 2 @ sigma2
        return mean.astype(np.float32), np.sqrt(pred_var).astype(np.float32)

    def update(self, x: np.ndarray, y: np.ndarray, noise_var: float = 0.1) -> None:
        """Update posterior with a new observation (x, y)."""
        sigma2 = np.exp(self._log_sigma2)
        pred = x @ self._mu
        error = y - pred
        grad_mu = np.outer(x, error) / noise_var - self._mu / (self.prior_std ** 2)
        self._mu += self.lr * grad_mu
        precision_likelihood = np.outer(x ** 2, np.ones(self.n_components)) / noise_var
        precision_prior = 1.0 / (self.prior_std ** 2)
        grad_lv = 0.5 * (1.0 - sigma2 * (precision_likelihood + precision_prior))
        self._log_sigma2 += self.lr * grad_lv
        self._n_updates += 1

    def kl_divergence(self) -> float:
        """KL divergence from posterior to prior."""
        sigma2 = np.exp(self._log_sigma2)
        prior_var = self.prior_std ** 2
        kl = 0.5 * np.sum(
            sigma2 / prior_var + self._mu ** 2 / prior_var - 1
            - self._log_sigma2 + np.log(prior_var)
        )
        return float(kl)


class GradientTrackerTT:
    """
    Tracks gradient statistics for TT core updates.

    Parameters
    ----------
    window : int
        Rolling window for statistics.
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self._grad_norms: _dq = _dq(maxlen=window)
        self._grad_max: _dq = _dq(maxlen=window)
        self._n_recorded = 0

    def record(self, grad: np.ndarray) -> None:
        """Record a gradient array."""
        g = np.array(grad, dtype=np.float64)
        self._grad_norms.append(float(np.linalg.norm(g)))
        self._grad_max.append(float(np.abs(g).max()))
        self._n_recorded += 1

    def is_exploding(self, threshold: float = 100.0) -> bool:
        if not self._grad_norms:
            return False
        return self._grad_norms[-1] > threshold

    def is_vanishing(self, threshold: float = 1e-7) -> bool:
        if not self._grad_norms:
            return False
        return self._grad_norms[-1] < threshold

    def summary(self) -> dict:
        if not self._grad_norms:
            return {}
        norms = np.array(self._grad_norms)
        return {
            "n_recorded": self._n_recorded,
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "max_norm": float(norms.max()),
            "min_norm": float(norms.min()),
            "latest_norm": float(self._grad_norms[-1]),
            "is_exploding": self.is_exploding(),
            "is_vanishing": self.is_vanishing(),
        }


class AdaptiveLearningRateTT:
    """
    AdaGrad-style adaptive per-core learning rate for TT networks.

    Parameters
    ----------
    base_lr : float
    eps : float
    max_lr : float
    """

    def __init__(self, base_lr: float = 0.01, eps: float = 1e-8, max_lr: float = 0.1) -> None:
        self.base_lr = base_lr
        self.eps = eps
        self.max_lr = max_lr
        self._acc: dict = {}

    def step(self, core_name: str, grad: np.ndarray) -> float:
        """Compute adaptive learning rate for a core."""
        g_sq = float(np.sum(grad ** 2))
        if core_name not in self._acc:
            self._acc[core_name] = 0.0
        self._acc[core_name] += g_sq
        lr = self.base_lr / (np.sqrt(self._acc[core_name]) + self.eps)
        return float(min(lr, self.max_lr))

    def reset(self, core_name=None) -> None:
        if core_name is not None:
            self._acc.pop(core_name, None)
        else:
            self._acc.clear()

    def learning_rates(self) -> dict:
        return {
            k: float(self.base_lr / (np.sqrt(v) + self.eps))
            for k, v in self._acc.items()
        }


class OnlineModelSelector:
    """
    Online selection between multiple streaming TT models using
    exponentially-weighted prediction performance.

    Parameters
    ----------
    model_names : list of str
    window : int
    decay : float
    """

    def __init__(self, model_names: list, window: int = 50, decay: float = 0.99) -> None:
        self.model_names = model_names
        self.window = window
        self.decay = decay
        self._errors: dict = {name: _dq(maxlen=window) for name in model_names}
        self._weights = {name: 1.0 / max(1, len(model_names)) for name in model_names}

    def update(self, predictions: dict, actual: np.ndarray) -> None:
        """Update model performance weights given actual observation."""
        for name in self.model_names:
            if name in predictions:
                pred = np.array(predictions[name])
                err = float(np.mean((pred - actual) ** 2))
                self._errors[name].append(err)

        avg_errors = {}
        for name in self.model_names:
            if self._errors[name]:
                avg_errors[name] = float(np.mean(list(self._errors[name])))
            else:
                avg_errors[name] = float("inf")

        min_err = min(avg_errors.values())
        raw_w = {n: np.exp(-(avg_errors[n] - min_err)) for n in self.model_names}
        total = sum(raw_w.values()) + 1e-12
        for name in self.model_names:
            target_w = raw_w[name] / total
            self._weights[name] = (
                self.decay * self._weights[name] + (1 - self.decay) * target_w
            )

    def best_model(self) -> str:
        """Return name of currently best-performing model."""
        return max(self._weights, key=self._weights.get)

    def ensemble_predict(self, predictions: dict) -> np.ndarray:
        """Weighted ensemble prediction."""
        total_w = sum(self._weights.get(n, 0.0) for n in predictions)
        if total_w < 1e-12:
            return np.array(list(predictions.values())[0], dtype=np.float32)
        result = None
        for name, pred in predictions.items():
            w = self._weights.get(name, 0.0) / total_w
            arr = np.array(pred, dtype=np.float64) * w
            result = arr if result is None else result + arr
        return result.astype(np.float32)

    def weights(self) -> dict:
        return dict(self._weights)


class StreamingCorrelationEstimator:
    """
    Streaming exponentially-weighted correlation estimator.

    Maintains (N x N) correlation matrix estimate using a running
    scatter matrix accumulation with optional shrinkage.

    Parameters
    ----------
    n_assets : int
    half_life : float
        Exponential decay half-life in observations.
    shrinkage : float
        Ledoit-Wolf shrinkage intensity.
    """

    def __init__(self, n_assets: int = 64, half_life: float = 63.0, shrinkage: float = 0.1) -> None:
        self.n_assets = n_assets
        self.half_life = half_life
        self.shrinkage = shrinkage
        import math
        self._decay = math.pow(0.5, 1.0 / half_life)
        self._S = np.zeros((n_assets, n_assets), dtype=np.float64)
        self._mu = np.zeros(n_assets, dtype=np.float64)
        self._n = 0.0

    def update(self, x: np.ndarray) -> None:
        """Update with a new observation or batch."""
        x = np.atleast_2d(x).astype(np.float64)
        lam = self._decay
        for xi in x:
            self._S = lam * self._S + (1 - lam) * np.outer(xi, xi)
            self._mu = lam * self._mu + (1 - lam) * xi
            self._n = lam * self._n + 1.0

    def correlation(self) -> np.ndarray:
        """Return current correlation matrix estimate."""
        cov = self._S - np.outer(self._mu, self._mu)
        std = np.sqrt(np.diag(cov) + 1e-12)
        corr = cov / np.outer(std, std)
        # Apply shrinkage toward identity
        alpha = self.shrinkage
        shrunk = (1 - alpha) * corr + alpha * np.eye(self.n_assets)
        np.clip(shrunk, -1, 1, out=shrunk)
        np.fill_diagonal(shrunk, 1.0)
        return shrunk.astype(np.float32)

    def covariance(self) -> np.ndarray:
        """Return current covariance matrix estimate."""
        cov = self._S - np.outer(self._mu, self._mu)
        alpha = self.shrinkage
        mu_target = np.trace(cov) / self.n_assets
        shrunk = (1 - alpha) * cov + alpha * mu_target * np.eye(self.n_assets)
        return (shrunk + 1e-6 * np.eye(self.n_assets)).astype(np.float32)

    def reset(self) -> None:
        self._S = np.zeros((self.n_assets, self.n_assets), dtype=np.float64)
        self._mu = np.zeros(self.n_assets, dtype=np.float64)
        self._n = 0.0


class StreamingRegressionTT:
    """
    Online least-squares regression with TT-structured weight matrix.

    Uses the recursive least squares (RLS) algorithm for exact online updates.

    Parameters
    ----------
    n_features : int
        Input dimensionality.
    n_targets : int
        Output dimensionality.
    forgetting_factor : float
        Lambda in (0, 1]. 1.0 = no forgetting.
    ridge : float
        L2 regularisation for initial covariance.
    """

    def __init__(
        self,
        n_features: int = 64,
        n_targets: int = 1,
        forgetting_factor: float = 0.99,
        ridge: float = 1e-3,
    ) -> None:
        self.n_features = n_features
        self.n_targets = n_targets
        self.lam = forgetting_factor
        self._W = np.zeros((n_features, n_targets), dtype=np.float64)
        self._P = np.eye(n_features) / ridge
        self._n_updates = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        RLS update with new observation.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
        y : np.ndarray, shape (n_targets,)
        """
        x = x.reshape(-1).astype(np.float64)
        y = y.reshape(-1).astype(np.float64)
        lam = self.lam
        Px = self._P @ x
        denom = lam + x @ Px
        K = Px / denom
        error = y - x @ self._W
        self._W += np.outer(K, error)
        self._P = (self._P - np.outer(Px, Px) / denom) / lam
        self._n_updates += 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict output for input x."""
        return (x @ self._W).astype(np.float32)

    @property
    def weights(self) -> np.ndarray:
        return self._W.astype(np.float32)

    @property
    def n_updates(self) -> int:
        return self._n_updates
