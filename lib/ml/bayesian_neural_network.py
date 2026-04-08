"""
Bayesian Neural Network for Uncertainty-Aware Predictions.

Variational inference with mean-field Gaussian approximation,
MC Dropout, calibration metrics, and active learning.  NumPy only.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Activations / utilities
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -20, 20)))


def _softplus_inv(x: np.ndarray) -> np.ndarray:
    """Inverse of softplus: rho such that softplus(rho) = x."""
    return np.log(np.exp(np.clip(x, 1e-8, 20)) - 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _log_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Log probability under N(mu, sigma^2)."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-12) \
           - 0.5 * ((x - mu) / (sigma + 1e-12)) ** 2


def _kl_gaussian(mu_q: np.ndarray, sigma_q: np.ndarray,
                 mu_p: float, sigma_p: float) -> float:
    """KL(q || p) for diagonal Gaussians."""
    var_q = sigma_q ** 2
    var_p = sigma_p ** 2
    kl = 0.5 * (np.log(var_p / (var_q + 1e-12)) + var_q / var_p
                + (mu_q - mu_p) ** 2 / var_p - 1.0)
    return float(kl.sum())


# ---------------------------------------------------------------------------
# Bayesian Dense Layer (mean-field variational)
# ---------------------------------------------------------------------------

class BayesianDenseLayer:
    """Fully-connected layer with weight uncertainty.

    Each weight w_ij ~ N(mu_ij, sigma_ij^2)
    sigma = softplus(rho) = log(1 + exp(rho))
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu",
                 prior_mu: float = 0.0, prior_sigma: float = 1.0,
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.rng = rng

        # Variational parameters
        scale = np.sqrt(2.0 / in_dim)
        self.W_mu = rng.normal(0, scale, (in_dim, out_dim))
        self.W_rho = np.full((in_dim, out_dim), _softplus_inv(0.1))
        self.b_mu = np.zeros(out_dim)
        self.b_rho = np.full(out_dim, _softplus_inv(0.1))

        # Adam state for variational params
        self._init_adam()

        # Cache
        self._input: Optional[np.ndarray] = None
        self._pre_act: Optional[np.ndarray] = None
        self._W_sampled: Optional[np.ndarray] = None
        self._b_sampled: Optional[np.ndarray] = None
        self._eps_W: Optional[np.ndarray] = None
        self._eps_b: Optional[np.ndarray] = None

    def _init_adam(self):
        for name in ["W_mu", "W_rho", "b_mu", "b_rho"]:
            setattr(self, f"m_{name}", np.zeros_like(getattr(self, name)))
            setattr(self, f"v_{name}", np.zeros_like(getattr(self, name)))

    @property
    def W_sigma(self) -> np.ndarray:
        return _softplus(self.W_rho)

    @property
    def b_sigma(self) -> np.ndarray:
        return _softplus(self.b_rho)

    def sample_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Sample W and b from variational posterior."""
        self._eps_W = self.rng.standard_normal(self.W_mu.shape)
        self._eps_b = self.rng.standard_normal(self.b_mu.shape)
        W = self.W_mu + self.W_sigma * self._eps_W
        b = self.b_mu + self.b_sigma * self._eps_b
        self._W_sampled = W
        self._b_sampled = b
        return W, b

    def forward(self, x: np.ndarray, sample: bool = True) -> np.ndarray:
        self._input = x
        if sample:
            W, b = self.sample_weights()
        else:
            W, b = self.W_mu, self.b_mu

        z = x @ W + b
        self._pre_act = z

        if self.activation == "relu":
            return _relu(z)
        elif self.activation == "sigmoid":
            return _sigmoid(z)
        return z

    def kl_divergence(self) -> float:
        """KL(q(W) || p(W)) + KL(q(b) || p(b))."""
        kl_W = _kl_gaussian(self.W_mu, self.W_sigma,
                            self.prior_mu, self.prior_sigma)
        kl_b = _kl_gaussian(self.b_mu, self.b_sigma,
                            self.prior_mu, self.prior_sigma)
        return kl_W + kl_b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        z = self._pre_act
        if self.activation == "relu":
            grad_act = (z > 0).astype(np.float64)
        elif self.activation == "sigmoid":
            s = _sigmoid(z)
            grad_act = s * (1 - s)
        else:
            grad_act = np.ones_like(z)

        dz = grad_out * grad_act
        B = dz.shape[0]

        W = self._W_sampled if self._W_sampled is not None else self.W_mu

        # Gradient w.r.t. W_mu
        self.dW_mu = self._input.T @ dz / B
        # Gradient w.r.t. W_rho (through reparameterization)
        # d_loss/d_rho = d_loss/d_W * eps * d_sigma/d_rho
        sig_deriv = _sigmoid(self.W_rho)  # d softplus / d rho
        if self._eps_W is not None:
            self.dW_rho = (self._input.T @ dz / B) * self._eps_W * sig_deriv

        self.db_mu = dz.mean(axis=0)
        if self._eps_b is not None:
            sig_deriv_b = _sigmoid(self.b_rho)
            self.db_rho = dz.mean(axis=0) * self._eps_b * sig_deriv_b

        return dz @ W.T

    def update(self, lr: float, t: int = 1, kl_weight: float = 1e-3,
               b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        """Adam update including KL gradient."""
        # KL gradients
        W_sig = self.W_sigma
        b_sig = self.b_sigma

        kl_dW_mu = (self.W_mu - self.prior_mu) / (self.prior_sigma ** 2)
        kl_dW_rho = ((W_sig ** 2 - self.prior_sigma ** 2)
                     / (self.prior_sigma ** 2 * W_sig + 1e-12)
                     * _sigmoid(self.W_rho))
        kl_db_mu = (self.b_mu - self.prior_mu) / (self.prior_sigma ** 2)
        kl_db_rho = ((b_sig ** 2 - self.prior_sigma ** 2)
                     / (self.prior_sigma ** 2 * b_sig + 1e-12)
                     * _sigmoid(self.b_rho))

        grads = {
            "W_mu": self.dW_mu + kl_weight * kl_dW_mu,
            "W_rho": getattr(self, "dW_rho", np.zeros_like(self.W_rho)) + kl_weight * kl_dW_rho,
            "b_mu": self.db_mu + kl_weight * kl_db_mu,
            "b_rho": getattr(self, "db_rho", np.zeros_like(self.b_rho)) + kl_weight * kl_db_rho,
        }

        for name, grad in grads.items():
            param = getattr(self, name)
            m = getattr(self, f"m_{name}")
            v = getattr(self, f"v_{name}")
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            setattr(self, f"m_{name}", m)
            setattr(self, f"v_{name}", v)
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            setattr(self, name, param - lr * mhat / (np.sqrt(vhat) + eps))


# ---------------------------------------------------------------------------
# MC Dropout Layer
# ---------------------------------------------------------------------------

class MCDropoutLayer:
    """Dropout layer active at both train and inference time."""

    def __init__(self, rate: float = 0.1,
                 rng: Optional[np.random.Generator] = None):
        self.rate = rate
        self.rng = rng or np.random.default_rng()
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if self.rate <= 0 or not training:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) > self.rate).astype(np.float64)
        return x * self._mask / (1 - self.rate)

    def backward(self, g: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return g
        return g * self._mask / (1 - self.rate)


# ---------------------------------------------------------------------------
# Bayesian Neural Network
# ---------------------------------------------------------------------------

class BayesianNeuralNetwork:
    """BNN with variational inference and MC Dropout.

    Parameters
    ----------
    input_dim    : number of input features
    hidden_dims  : hidden layer widths
    output_dim   : output dimension (1 for regression)
    prior_sigma  : prior std dev on weights
    dropout_rate : MC dropout rate
    kl_weight    : weight for KL term in ELBO
    seed         : random seed
    """

    def __init__(self, input_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 output_dim: int = 1,
                 prior_sigma: float = 1.0,
                 dropout_rate: float = 0.1,
                 kl_weight: float = 1e-3,
                 seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.kl_weight = kl_weight
        self.output_dim = output_dim

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.bayes_layers: List[BayesianDenseLayer] = []
        self.dropout_layers: List[MCDropoutLayer] = []

        d = input_dim
        for h in hidden_dims:
            self.bayes_layers.append(
                BayesianDenseLayer(d, h, "relu", prior_sigma=prior_sigma,
                                   rng=self.rng)
            )
            self.dropout_layers.append(MCDropoutLayer(dropout_rate, self.rng))
            d = h

        # Output layer (no activation, no dropout)
        self.bayes_layers.append(
            BayesianDenseLayer(d, output_dim, "none",
                               prior_sigma=prior_sigma, rng=self.rng)
        )

        # Optional learned log-noise for heteroscedastic regression
        self.log_noise = np.array([-2.0])  # log(sigma_noise)
        self.m_log_noise = np.zeros(1)
        self.v_log_noise = np.zeros(1)

        self.step_count = 0

    # ---- forward ---------------------------------------------------------

    def forward(self, x: np.ndarray, sample: bool = True,
                mc_dropout: bool = True) -> np.ndarray:
        """Forward pass with optional weight sampling and dropout."""
        h = x
        for i, blayer in enumerate(self.bayes_layers[:-1]):
            h = blayer.forward(h, sample=sample)
            h = self.dropout_layers[i].forward(h, training=mc_dropout)
        h = self.bayes_layers[-1].forward(h, sample=sample)
        return h

    # ---- loss (ELBO) -----------------------------------------------------

    def elbo_loss(self, x: np.ndarray, y: np.ndarray,
                  n_samples: int = 1) -> Tuple[float, np.ndarray]:
        """Compute negative ELBO = reconstruction + KL.

        Returns (loss, grad_output).
        """
        if y.ndim == 1:
            y = y[:, None]

        total_recon = 0.0
        total_grad = np.zeros_like(y, dtype=np.float64)

        for _ in range(n_samples):
            pred = self.forward(x, sample=True, mc_dropout=True)
            noise_var = np.exp(self.log_noise) ** 2
            diff = pred - y
            recon = 0.5 * np.mean(diff ** 2 / noise_var) + 0.5 * self.log_noise[0]
            total_recon += recon
            total_grad += diff / (noise_var * y.shape[0] * n_samples)

        total_recon /= n_samples

        # KL
        kl = sum(bl.kl_divergence() for bl in self.bayes_layers)
        loss = total_recon + self.kl_weight * kl

        return float(loss), total_grad

    # ---- backward --------------------------------------------------------

    def backward(self, grad: np.ndarray) -> None:
        g = grad
        g = self.bayes_layers[-1].backward(g)
        for i in range(len(self.bayes_layers) - 2, -1, -1):
            g = self.dropout_layers[i].backward(g)
            g = self.bayes_layers[i].backward(g)

    def update(self, lr: float) -> None:
        self.step_count += 1
        t = self.step_count
        for bl in self.bayes_layers:
            bl.update(lr, t, self.kl_weight)

        # Update log_noise
        # gradient of recon w.r.t. log_noise is approximate
        # (we skip for simplicity; could be added)

    # ---- training step ---------------------------------------------------

    def train_step(self, x: np.ndarray, y: np.ndarray,
                   lr: float = 1e-3) -> float:
        loss, grad = self.elbo_loss(x, y)
        self.backward(grad)
        self.update(lr)
        return loss

    # ---- full training ---------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 200, batch_size: int = 64,
            lr: float = 1e-3,
            val_X: Optional[np.ndarray] = None,
            val_y: Optional[np.ndarray] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        N = X.shape[0]
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for ep in range(epochs):
            perm = self.rng.permutation(N)
            running = 0.0
            nb = 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size]
                loss = self.train_step(X[idx], y[idx], lr)
                running += loss
                nb += 1
            avg = running / max(nb, 1)
            history["train_loss"].append(avg)

            if val_X is not None:
                vl, _ = self.elbo_loss(val_X, val_y)
                history["val_loss"].append(vl)

            if verbose and (ep + 1) % max(1, epochs // 10) == 0:
                msg = f"  epoch {ep+1:4d}/{epochs}  elbo={avg:.4f}"
                if val_X is not None:
                    msg += f"  val={history['val_loss'][-1]:.4f}"
                print(msg)

        return history

    # ---- predictive distribution via MC forward passes -------------------

    def predict(self, x: np.ndarray, n_forward: int = 50,
                mc_dropout: bool = True) -> Dict[str, np.ndarray]:
        """Predictive distribution from multiple stochastic forward passes.

        Returns dict with 'mean', 'std', 'samples', 'epistemic', 'aleatoric'.
        """
        preds = []
        for _ in range(n_forward):
            p = self.forward(x, sample=True, mc_dropout=mc_dropout)
            preds.append(p.squeeze(-1) if p.ndim > 1 else p)

        preds = np.array(preds)  # (n_forward, B)
        mean = preds.mean(axis=0)
        epistemic = preds.var(axis=0)
        aleatoric = np.exp(self.log_noise[0]) ** 2
        total_var = epistemic + aleatoric

        return {
            "mean": mean,
            "std": np.sqrt(total_var),
            "epistemic_std": np.sqrt(epistemic),
            "aleatoric_std": np.sqrt(aleatoric) * np.ones_like(mean),
            "samples": preds,
        }

    def predict_interval(self, x: np.ndarray, alpha: float = 0.05,
                         n_forward: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prediction interval.

        Returns (mean, lower, upper).
        """
        result = self.predict(x, n_forward)
        samples = result["samples"]  # (n_forward, B)
        mean = result["mean"]
        lower = np.quantile(samples, alpha / 2, axis=0)
        upper = np.quantile(samples, 1 - alpha / 2, axis=0)
        return mean, lower, upper


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

class CalibrationMetrics:
    """Evaluate calibration of predictive uncertainty."""

    @staticmethod
    def reliability_diagram(y_true: np.ndarray, pred_mean: np.ndarray,
                            pred_std: np.ndarray,
                            n_bins: int = 10) -> Dict[str, np.ndarray]:
        """Compute reliability diagram data.

        For each confidence level, check fraction of observations
        falling within the predicted interval.
        """
        confidences = np.linspace(0.1, 0.99, n_bins)
        observed_freqs = np.zeros(n_bins)

        for i, conf in enumerate(confidences):
            alpha = 1 - conf
            z = _norm_ppf(1 - alpha / 2)
            lower = pred_mean - z * pred_std
            upper = pred_mean + z * pred_std
            in_interval = (y_true >= lower) & (y_true <= upper)
            observed_freqs[i] = in_interval.mean()

        return {
            "expected_confidence": confidences,
            "observed_frequency": observed_freqs,
        }

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray,
                                   pred_mean: np.ndarray,
                                   pred_std: np.ndarray,
                                   n_bins: int = 10) -> float:
        """ECE: mean absolute difference between expected and observed
        coverage across confidence levels."""
        diag = CalibrationMetrics.reliability_diagram(
            y_true, pred_mean, pred_std, n_bins)
        return float(np.mean(np.abs(
            diag["expected_confidence"] - diag["observed_frequency"]
        )))

    @staticmethod
    def sharpness(pred_std: np.ndarray) -> float:
        """Average predicted std (lower is sharper)."""
        return float(pred_std.mean())

    @staticmethod
    def nll(y_true: np.ndarray, pred_mean: np.ndarray,
            pred_std: np.ndarray) -> float:
        """Negative log-likelihood under Gaussian predictive."""
        return float(-np.mean(
            _log_gaussian_vec(y_true, pred_mean, pred_std)
        ))

    @staticmethod
    def crps(y_true: np.ndarray, pred_mean: np.ndarray,
             pred_std: np.ndarray) -> float:
        """Continuous Ranked Probability Score (Gaussian)."""
        z = (y_true - pred_mean) / (pred_std + 1e-12)
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        cdf = 0.5 * (1 + _erf_approx(z / np.sqrt(2)))
        crps_vals = pred_std * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / np.sqrt(np.pi))
        return float(crps_vals.mean())


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (rational approximation)."""
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        t = np.sqrt(-2 * np.log(p))
    else:
        t = np.sqrt(-2 * np.log(1 - p))

    # Abramowitz & Stegun 26.2.23
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t ** 2
    den = 1 + d1 * t + d2 * t ** 2 + d3 * t ** 3
    val = t - num / den

    return val if p > 0.5 else -val


def _erf_approx(x: np.ndarray) -> np.ndarray:
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                           -1.453152027, 1.061405429)
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def _log_gaussian_vec(x: np.ndarray, mu: np.ndarray,
                      sigma: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-12) \
           - 0.5 * ((x - mu) / (sigma + 1e-12)) ** 2


# ---------------------------------------------------------------------------
# Uncertainty decomposition
# ---------------------------------------------------------------------------

class UncertaintyDecomposer:
    """Decompose predictive uncertainty into epistemic and aleatoric."""

    def __init__(self, model: BayesianNeuralNetwork):
        self.model = model

    def decompose(self, x: np.ndarray,
                  n_forward: int = 100) -> Dict[str, np.ndarray]:
        result = self.model.predict(x, n_forward)
        return {
            "total_variance": result["std"] ** 2,
            "epistemic_variance": result["epistemic_std"] ** 2,
            "aleatoric_variance": result["aleatoric_std"] ** 2,
            "epistemic_fraction": (
                result["epistemic_std"] ** 2 / (result["std"] ** 2 + 1e-12)
            ),
        }

    def high_uncertainty_mask(self, x: np.ndarray,
                              threshold_quantile: float = 0.9,
                              n_forward: int = 50) -> np.ndarray:
        """Boolean mask for high-uncertainty points."""
        result = self.model.predict(x, n_forward)
        total = result["std"]
        thresh = np.quantile(total, threshold_quantile)
        return total >= thresh


# ---------------------------------------------------------------------------
# Active Learning
# ---------------------------------------------------------------------------

class ActiveLearner:
    """Active learning loop selecting most uncertain points for labeling.

    Strategies:
    - 'max_entropy' : highest predictive variance
    - 'max_epistemic' : highest epistemic uncertainty
    - 'bald' : Bayesian Active Learning by Disagreement
    """

    def __init__(self, model: BayesianNeuralNetwork,
                 strategy: str = "max_entropy"):
        self.model = model
        self.strategy = strategy

    def acquisition_scores(self, x_pool: np.ndarray,
                           n_forward: int = 50) -> np.ndarray:
        """Compute acquisition scores for each candidate."""
        result = self.model.predict(x_pool, n_forward)

        if self.strategy == "max_entropy":
            return result["std"]
        elif self.strategy == "max_epistemic":
            return result["epistemic_std"]
        elif self.strategy == "bald":
            # BALD = H[y|x] - E_w[H[y|x,w]]
            # For Gaussian: H = 0.5 * log(2*pi*e*var)
            total_var = result["std"] ** 2
            aleatoric_var = result["aleatoric_std"] ** 2
            bald = 0.5 * (np.log(total_var + 1e-12) - np.log(aleatoric_var + 1e-12))
            return bald
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def select_batch(self, x_pool: np.ndarray, batch_size: int = 10,
                     n_forward: int = 50) -> np.ndarray:
        """Select indices of most informative points."""
        scores = self.acquisition_scores(x_pool, n_forward)
        return np.argsort(-scores)[:batch_size]

    def active_learning_loop(self, x_pool: np.ndarray,
                             y_pool: np.ndarray,
                             x_init: np.ndarray,
                             y_init: np.ndarray,
                             n_rounds: int = 10,
                             batch_size: int = 10,
                             train_epochs: int = 50,
                             lr: float = 1e-3,
                             val_X: Optional[np.ndarray] = None,
                             val_y: Optional[np.ndarray] = None
                             ) -> Dict[str, List]:
        """Run active learning loop.

        Returns history of metrics per round.
        """
        x_train = x_init.copy()
        y_train = y_init.copy()
        pool_mask = np.ones(len(x_pool), dtype=bool)

        history: Dict[str, List] = {
            "n_labeled": [],
            "train_loss": [],
            "val_rmse": [],
            "val_nll": [],
            "selected_indices": [],
        }

        for r in range(n_rounds):
            # Train model
            self.model.fit(x_train, y_train, epochs=train_epochs,
                           lr=lr, verbose=False)

            # Record metrics
            history["n_labeled"].append(len(x_train))
            last_loss = self.model.train_step(x_train, y_train, lr)
            history["train_loss"].append(last_loss)

            if val_X is not None:
                pred = self.model.predict(val_X, 30)
                rmse = float(np.sqrt(np.mean((pred["mean"] - val_y) ** 2)))
                nll = CalibrationMetrics.nll(val_y, pred["mean"], pred["std"])
                history["val_rmse"].append(rmse)
                history["val_nll"].append(nll)

            # Select new points from pool
            x_avail = x_pool[pool_mask]
            if len(x_avail) == 0:
                break
            avail_idx = np.where(pool_mask)[0]
            scores = self.acquisition_scores(x_avail)
            local_sel = np.argsort(-scores)[:batch_size]
            global_sel = avail_idx[local_sel]

            history["selected_indices"].append(global_sel.tolist())

            # Add to training set
            x_train = np.vstack([x_train, x_pool[global_sel]])
            y_train = np.concatenate([y_train, y_pool[global_sel]])
            pool_mask[global_sel] = False

        return history


# ---------------------------------------------------------------------------
# BNN Ensemble (deep ensembles + Bayesian)
# ---------------------------------------------------------------------------

class BNNEnsemble:
    """Ensemble of BNNs for improved uncertainty estimation."""

    def __init__(self, input_dim: int, n_members: int = 5,
                 hidden_dims: Optional[List[int]] = None,
                 seed: int = 42, **kwargs):
        self.members: List[BayesianNeuralNetwork] = []
        for i in range(n_members):
            self.members.append(
                BayesianNeuralNetwork(input_dim, hidden_dims,
                                      seed=seed + i, **kwargs)
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        N = X.shape[0]
        for i, m in enumerate(self.members):
            rng = np.random.default_rng(42 + i)
            idx = rng.choice(N, N, replace=True)
            m.fit(X[idx], y[idx], **kwargs)

    def predict(self, x: np.ndarray,
                n_forward_per_member: int = 20) -> Dict[str, np.ndarray]:
        all_means = []
        all_vars = []
        for m in self.members:
            r = m.predict(x, n_forward_per_member)
            all_means.append(r["mean"])
            all_vars.append(r["std"] ** 2)

        means = np.array(all_means)
        vars_ = np.array(all_vars)

        ens_mean = means.mean(axis=0)
        aleatoric = vars_.mean(axis=0)
        epistemic = means.var(axis=0)
        total = aleatoric + epistemic

        return {
            "mean": ens_mean,
            "std": np.sqrt(total),
            "epistemic_std": np.sqrt(epistemic),
            "aleatoric_std": np.sqrt(aleatoric),
        }


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def fit_bnn_for_returns(features: np.ndarray, returns: np.ndarray,
                        hidden_dims: Optional[List[int]] = None,
                        epochs: int = 200, lr: float = 1e-3,
                        val_frac: float = 0.2,
                        seed: int = 42) -> Dict:
    """Train BNN and evaluate uncertainty quality.

    Returns dict with model, predictions, calibration metrics.
    """
    N = features.shape[0]
    split = int(N * (1 - val_frac))
    X_tr, X_val = features[:split], features[split:]
    y_tr, y_val = returns[:split], returns[split:]

    model = BayesianNeuralNetwork(features.shape[1], hidden_dims, seed=seed)
    history = model.fit(X_tr, y_tr, epochs=epochs, lr=lr,
                        val_X=X_val, val_y=y_val)

    pred = model.predict(X_val, n_forward=50)
    rmse = float(np.sqrt(np.mean((pred["mean"] - y_val) ** 2)))
    ece = CalibrationMetrics.expected_calibration_error(
        y_val, pred["mean"], pred["std"])
    sharpness = CalibrationMetrics.sharpness(pred["std"])
    nll = CalibrationMetrics.nll(y_val, pred["mean"], pred["std"])
    crps = CalibrationMetrics.crps(y_val, pred["mean"], pred["std"])
    reliability = CalibrationMetrics.reliability_diagram(
        y_val, pred["mean"], pred["std"])

    return {
        "model": model,
        "history": history,
        "predictions": pred,
        "rmse": rmse,
        "ece": ece,
        "sharpness": sharpness,
        "nll": nll,
        "crps": crps,
        "reliability": reliability,
    }
