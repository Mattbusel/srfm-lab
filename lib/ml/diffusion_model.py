"""
Denoising Diffusion Probabilistic Model for Financial Time Series Generation.

Implements DDPM / DDIM for generating synthetic return paths, with
classifier-free guidance and quality metrics.  NumPy only.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sinusoidal_encoding(timesteps: np.ndarray, dim: int) -> np.ndarray:
    """Sinusoidal positional / timestep encoding (Transformer-style).

    Parameters
    ----------
    timesteps : (B,) int or float array
    dim       : embedding dimension (must be even)

    Returns
    -------
    enc : (B, dim)
    """
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)  # (half,)
    args = timesteps[:, None] * freqs[None, :]                  # (B, half)
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -20, 20)))


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(T: int, beta_start: float = 1e-4,
                         beta_end: float = 0.02) -> np.ndarray:
    """Linear variance schedule."""
    return np.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Cosine variance schedule (improved DDPM)."""
    steps = np.arange(T + 1, dtype=np.float64)
    alpha_bar = np.cos((steps / T + s) / (1 + s) * (np.pi / 2)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 1e-6, 0.999).astype(np.float64)


def sigmoid_beta_schedule(T: int, beta_start: float = 1e-4,
                          beta_end: float = 0.02) -> np.ndarray:
    """Sigmoid variance schedule."""
    t = np.linspace(-6, 6, T)
    betas = _sigmoid(t) * (beta_end - beta_start) + beta_start
    return betas


# ---------------------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------------------

class DenseLayer:
    """Single fully-connected layer with optional activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu",
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation

        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        # Cache for backprop
        self._input: Optional[np.ndarray] = None
        self._pre_act: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        z = x @ self.W + self.b
        self._pre_act = z
        if self.activation == "relu":
            return _relu(z)
        elif self.activation == "sigmoid":
            return _sigmoid(z)
        elif self.activation == "none":
            return z
        elif self.activation == "swish":
            return z * _sigmoid(z)
        return z

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        z = self._pre_act
        if self.activation == "relu":
            grad_act = (z > 0).astype(np.float64)
        elif self.activation == "sigmoid":
            s = _sigmoid(z)
            grad_act = s * (1 - s)
        elif self.activation == "swish":
            s = _sigmoid(z)
            grad_act = s + z * s * (1 - s)
        else:
            grad_act = np.ones_like(z)
        dz = grad_out * grad_act
        self.dW = self._input.T @ dz / dz.shape[0]
        self.db = dz.mean(axis=0)
        return dz @ self.W.T

    def update(self, lr: float, beta1: float = 0.9, beta2: float = 0.999,
               eps: float = 1e-8, t: int = 1):
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW ** 2
        mhat = self.mW / (1 - beta1 ** t)
        vhat = self.vW / (1 - beta2 ** t)
        self.W -= lr * mhat / (np.sqrt(vhat) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * self.db ** 2
        mbhat = self.mb / (1 - beta1 ** t)
        vbhat = self.vb / (1 - beta2 ** t)
        self.b -= lr * mbhat / (np.sqrt(vbhat) + eps)


# ---------------------------------------------------------------------------
# Denoiser network with timestep + optional class conditioning
# ---------------------------------------------------------------------------

class DenoisingNetwork:
    """Dense denoiser: x_noisy, t, [class_label] -> predicted noise."""

    def __init__(self, data_dim: int, hidden_dims: List[int],
                 time_emb_dim: int = 64, num_classes: int = 0,
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.rng = rng

        # Time MLP: sinusoidal -> dense -> dense
        self.time_fc1 = DenseLayer(time_emb_dim, time_emb_dim * 2, "swish", rng)
        self.time_fc2 = DenseLayer(time_emb_dim * 2, time_emb_dim, "swish", rng)

        # Optional class embedding
        self.class_emb_dim = 0
        if num_classes > 0:
            self.class_emb_dim = time_emb_dim
            self.class_embed = rng.normal(0, 0.02, (num_classes, self.class_emb_dim))

        # Main backbone
        in_dim = data_dim + time_emb_dim + self.class_emb_dim
        self.layers: List[DenseLayer] = []
        for h in hidden_dims:
            self.layers.append(DenseLayer(in_dim, h, "swish", rng))
            in_dim = h
        self.layers.append(DenseLayer(in_dim, data_dim, "none", rng))

    def forward(self, x_noisy: np.ndarray, t: np.ndarray,
                class_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parameters
        ----------
        x_noisy     : (B, D)
        t           : (B,) integer timesteps
        class_labels: (B,) integer class indices or None

        Returns
        -------
        eps_pred : (B, D)  predicted noise
        """
        t_emb = _sinusoidal_encoding(t.astype(np.float64), self.time_emb_dim)
        t_emb = self.time_fc1.forward(t_emb)
        t_emb = self.time_fc2.forward(t_emb)

        parts = [x_noisy, t_emb]
        if class_labels is not None and self.num_classes > 0:
            c_emb = self.class_embed[class_labels]  # (B, class_emb_dim)
            parts.append(c_emb)

        h = np.concatenate(parts, axis=-1)
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, grad: np.ndarray) -> None:
        g = grad
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def update(self, lr: float, step: int = 1):
        for layer in self.layers:
            layer.update(lr, t=step)
        self.time_fc1.update(lr, t=step)
        self.time_fc2.update(lr, t=step)


# ---------------------------------------------------------------------------
# Denoising Diffusion Probabilistic Model
# ---------------------------------------------------------------------------

class DiffusionModel:
    """DDPM for financial time-series generation.

    Parameters
    ----------
    data_dim    : dimensionality of each sample (e.g. window length)
    T           : number of diffusion steps
    hidden_dims : list of hidden-layer widths for denoiser
    schedule    : 'linear', 'cosine', or 'sigmoid'
    num_classes : number of regime classes (0 = unconditional)
    cfg_prob    : probability of dropping class label during training
                  (for classifier-free guidance)
    seed        : random seed
    """

    def __init__(self, data_dim: int = 64, T: int = 200,
                 hidden_dims: Optional[List[int]] = None,
                 schedule: str = "cosine",
                 num_classes: int = 0,
                 cfg_prob: float = 0.1,
                 seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.data_dim = data_dim
        self.T = T
        self.num_classes = num_classes
        self.cfg_prob = cfg_prob

        # Noise schedule
        if schedule == "linear":
            self.betas = linear_beta_schedule(T)
        elif schedule == "cosine":
            self.betas = cosine_beta_schedule(T)
        elif schedule == "sigmoid":
            self.betas = sigmoid_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)

        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        self.net = DenoisingNetwork(
            data_dim, hidden_dims, time_emb_dim=64,
            num_classes=num_classes, rng=self.rng,
        )
        self.step_count = 0

    # ---- forward diffusion (add noise) -----------------------------------

    def q_sample(self, x0: np.ndarray, t: np.ndarray,
                 noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample x_t from q(x_t | x_0).

        Returns (x_t, noise).
        """
        if noise is None:
            noise = self.rng.standard_normal(x0.shape)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t][:, None]
        x_t = sqrt_ab * x0 + sqrt_omab * noise
        return x_t, noise

    # ---- training step ---------------------------------------------------

    def train_step(self, x0: np.ndarray, lr: float = 1e-4,
                   class_labels: Optional[np.ndarray] = None) -> float:
        """One gradient step (simple L2 noise-prediction loss).

        Parameters
        ----------
        x0          : (B, D) clean data
        lr          : learning rate
        class_labels: (B,) integer class indices or None

        Returns
        -------
        loss : scalar MSE
        """
        B = x0.shape[0]
        t = self.rng.integers(0, self.T, size=B)
        x_t, noise = self.q_sample(x0, t)

        # Classifier-free guidance: randomly drop labels
        labels_input = class_labels
        if class_labels is not None and self.cfg_prob > 0:
            mask = self.rng.random(B) < self.cfg_prob
            labels_input = class_labels.copy()
            labels_input[mask] = 0  # use class-0 as "unconditional"

        eps_pred = self.net.forward(x_t, t, labels_input)

        # MSE loss and gradient
        diff = eps_pred - noise
        loss = np.mean(diff ** 2)
        grad = 2.0 * diff / diff.size

        self.net.backward(grad)
        self.step_count += 1
        self.net.update(lr, self.step_count)
        return float(loss)

    # ---- DDPM sampling ---------------------------------------------------

    def _p_sample(self, x_t: np.ndarray, t_idx: int,
                  class_labels: Optional[np.ndarray] = None,
                  guidance_scale: float = 1.0) -> np.ndarray:
        """Single reverse step p(x_{t-1} | x_t)."""
        B = x_t.shape[0]
        t_arr = np.full(B, t_idx, dtype=np.int64)

        # Predict noise (with optional CFG)
        if class_labels is not None and guidance_scale != 1.0:
            eps_cond = self.net.forward(x_t, t_arr, class_labels)
            eps_uncond = self.net.forward(x_t, t_arr, np.zeros(B, dtype=np.int64))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = self.net.forward(x_t, t_arr, class_labels)

        beta_t = self.betas[t_idx]
        alpha_t = self.alphas[t_idx]
        alpha_bar_t = self.alpha_bar[t_idx]

        coef = beta_t / np.sqrt(1.0 - alpha_bar_t)
        mean = (x_t - coef * eps) / np.sqrt(alpha_t)

        if t_idx > 0:
            noise = self.rng.standard_normal(x_t.shape)
            sigma = np.sqrt(beta_t)
            return mean + sigma * noise
        return mean

    def sample_ddpm(self, n_samples: int,
                    class_labels: Optional[np.ndarray] = None,
                    guidance_scale: float = 1.0) -> np.ndarray:
        """Full DDPM reverse-process sampling.

        Returns
        -------
        x0 : (n_samples, data_dim)
        """
        x = self.rng.standard_normal((n_samples, self.data_dim))
        for t in reversed(range(self.T)):
            x = self._p_sample(x, t, class_labels, guidance_scale)
        return x

    # ---- DDIM sampling ---------------------------------------------------

    def sample_ddim(self, n_samples: int, num_steps: int = 50,
                    eta: float = 0.0,
                    class_labels: Optional[np.ndarray] = None,
                    guidance_scale: float = 1.0) -> np.ndarray:
        """DDIM deterministic/stochastic sampler for faster inference.

        Parameters
        ----------
        num_steps      : number of DDIM steps (< T)
        eta            : 0 = deterministic, 1 = same as DDPM
        guidance_scale : classifier-free guidance weight
        """
        # Sub-sequence of timesteps
        step_size = max(1, self.T // num_steps)
        timesteps = np.arange(0, self.T, step_size)[::-1]

        x = self.rng.standard_normal((n_samples, self.data_dim))
        B = n_samples

        for i, t_cur in enumerate(timesteps):
            t_arr = np.full(B, t_cur, dtype=np.int64)

            if class_labels is not None and guidance_scale != 1.0:
                eps_cond = self.net.forward(x, t_arr, class_labels)
                eps_uncond = self.net.forward(x, t_arr, np.zeros(B, dtype=np.int64))
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.net.forward(x, t_arr, class_labels)

            ab_t = self.alpha_bar[t_cur]
            x0_pred = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)

            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                ab_prev = self.alpha_bar[t_prev]
            else:
                ab_prev = 1.0

            sigma = eta * np.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
            dir_xt = np.sqrt(1 - ab_prev - sigma ** 2) * eps
            noise = self.rng.standard_normal(x.shape) if sigma > 0 else 0.0
            x = np.sqrt(ab_prev) * x0_pred + dir_xt + sigma * noise

        return x

    # ---- conditional generation ------------------------------------------

    def generate_conditional(self, n_samples: int, regime: int,
                             guidance_scale: float = 3.0,
                             use_ddim: bool = True,
                             ddim_steps: int = 50) -> np.ndarray:
        """Generate samples conditioned on a regime label.

        Parameters
        ----------
        regime         : integer regime class
        guidance_scale : CFG scale (higher = more class-faithful)
        """
        labels = np.full(n_samples, regime, dtype=np.int64)
        if use_ddim:
            return self.sample_ddim(n_samples, ddim_steps,
                                    class_labels=labels,
                                    guidance_scale=guidance_scale)
        return self.sample_ddpm(n_samples, class_labels=labels,
                                guidance_scale=guidance_scale)

    # ---- quality metrics -------------------------------------------------

    @staticmethod
    def compute_statistics(real: np.ndarray,
                           generated: np.ndarray) -> Dict[str, float]:
        """Statistical comparison between real and generated data.

        Returns dict with various distance / test metrics.
        """
        metrics: Dict[str, float] = {}

        # Marginal moments
        metrics["mean_diff"] = float(np.abs(real.mean() - generated.mean()))
        metrics["std_diff"] = float(np.abs(real.std() - generated.std()))
        metrics["skew_diff"] = float(np.abs(
            _skewness(real) - _skewness(generated)))
        metrics["kurt_diff"] = float(np.abs(
            _kurtosis(real) - _kurtosis(generated)))

        # Auto-correlation comparison (lag-1)
        ac_real = _autocorr(real.ravel(), 1)
        ac_gen = _autocorr(generated.ravel(), 1)
        metrics["autocorr_lag1_diff"] = float(np.abs(ac_real - ac_gen))

        # Maximum Mean Discrepancy (Gaussian kernel)
        metrics["mmd"] = float(_mmd_rbf(real, generated))

        # Kolmogorov-Smirnov-like statistic on marginals
        metrics["ks_stat"] = float(_ks_statistic(real.ravel(), generated.ravel()))

        # Wasserstein-1 approximation (sorted quantile diff)
        metrics["wasserstein1"] = float(_wasserstein1(real.ravel(), generated.ravel()))

        return metrics

    # ---- full training loop ----------------------------------------------

    def fit(self, data: np.ndarray, epochs: int = 100,
            batch_size: int = 64, lr: float = 1e-4,
            class_labels: Optional[np.ndarray] = None,
            verbose: bool = True) -> List[float]:
        """Train the diffusion model.

        Parameters
        ----------
        data         : (N, D) training data
        epochs       : number of full passes
        batch_size   : mini-batch size
        lr           : learning rate
        class_labels : (N,) integer regime labels or None
        verbose      : print progress

        Returns
        -------
        losses : list of per-epoch average losses
        """
        N = data.shape[0]
        epoch_losses: List[float] = []

        for epoch in range(epochs):
            perm = self.rng.permutation(N)
            running = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                xb = data[idx]
                lb = class_labels[idx] if class_labels is not None else None
                loss = self.train_step(xb, lr, lb)
                running += loss
                n_batches += 1

            avg = running / max(n_batches, 1)
            epoch_losses.append(avg)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  epoch {epoch+1:4d}/{epochs}  loss={avg:.6f}")

        return epoch_losses

    # ---- stress-test scenario generation ---------------------------------

    def generate_stress_scenarios(self, n_scenarios: int = 1000,
                                  regime: Optional[int] = None,
                                  tail_quantile: float = 0.05,
                                  guidance_scale: float = 3.0) -> Dict[str, np.ndarray]:
        """Generate synthetic return paths for stress testing.

        Returns
        -------
        dict with keys: 'all_paths', 'tail_paths', 'tail_threshold'
        """
        labels = None
        if regime is not None:
            labels = np.full(n_scenarios, regime, dtype=np.int64)

        paths = self.sample_ddim(n_scenarios, 50, class_labels=labels,
                                 guidance_scale=guidance_scale)

        # Identify tail scenarios by cumulative return
        cum_returns = paths.sum(axis=1)
        threshold = np.quantile(cum_returns, tail_quantile)
        tail_mask = cum_returns <= threshold
        return {
            "all_paths": paths,
            "tail_paths": paths[tail_mask],
            "tail_threshold": threshold,
            "cum_returns": cum_returns,
        }


# ---------------------------------------------------------------------------
# Statistical helper functions
# ---------------------------------------------------------------------------

def _skewness(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _autocorr(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if n <= lag:
        return 0.0
    m = x.mean()
    c0 = np.mean((x - m) ** 2)
    if c0 < 1e-12:
        return 0.0
    c_lag = np.mean((x[:n - lag] - m) * (x[lag:] - m))
    return float(c_lag / c0)


def _mmd_rbf(X: np.ndarray, Y: np.ndarray,
             gamma: Optional[float] = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel."""
    if gamma is None:
        combined = np.vstack([X, Y])
        dists = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / max(median_dist, 1e-8)

    def rbf(A: np.ndarray, B: np.ndarray) -> float:
        d = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return float(np.mean(np.exp(-gamma * d)))

    return rbf(X, X) + rbf(Y, Y) - 2 * rbf(X, Y)


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    all_vals = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(np.sort(x), all_vals, side="right") / len(x)
    cdf_y = np.searchsorted(np.sort(y), all_vals, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _wasserstein1(x: np.ndarray, y: np.ndarray) -> float:
    """1D Wasserstein-1 distance via sorted quantiles."""
    xs = np.sort(x)
    ys = np.sort(y)
    n = min(len(xs), len(ys))
    q = np.linspace(0, 1, n + 2)[1:-1]
    qx = np.quantile(xs, q)
    qy = np.quantile(ys, q)
    return float(np.mean(np.abs(qx - qy)))


# ---------------------------------------------------------------------------
# Convenience: EMA of model weights
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of denoiser weights for better sampling."""

    def __init__(self, model: DiffusionModel, decay: float = 0.999):
        self.decay = decay
        self.shadow: List[Tuple[np.ndarray, np.ndarray]] = []
        for layer in model.net.layers:
            self.shadow.append((layer.W.copy(), layer.b.copy()))

    def update(self, model: DiffusionModel) -> None:
        d = self.decay
        for i, layer in enumerate(model.net.layers):
            self.shadow[i] = (
                d * self.shadow[i][0] + (1 - d) * layer.W,
                d * self.shadow[i][1] + (1 - d) * layer.b,
            )

    def apply(self, model: DiffusionModel) -> None:
        for i, layer in enumerate(model.net.layers):
            layer.W = self.shadow[i][0].copy()
            layer.b = self.shadow[i][1].copy()

    def restore(self, model: DiffusionModel,
                backup: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        for i, layer in enumerate(model.net.layers):
            layer.W = backup[i][0]
            layer.b = backup[i][1]

    def backup_and_apply(self, model: DiffusionModel
                         ) -> List[Tuple[np.ndarray, np.ndarray]]:
        bk = [(l.W.copy(), l.b.copy()) for l in model.net.layers]
        self.apply(model)
        return bk


# ---------------------------------------------------------------------------
# Data preprocessing helpers
# ---------------------------------------------------------------------------

class ReturnScaler:
    """Standardise returns and inverse-transform generated samples."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "ReturnScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def train_and_generate(returns: np.ndarray,
                       window: int = 64,
                       T: int = 200,
                       epochs: int = 50,
                       n_gen: int = 500,
                       regime_labels: Optional[np.ndarray] = None,
                       seed: int = 42) -> Dict:
    """High-level helper: train a diffusion model and generate paths.

    Parameters
    ----------
    returns       : (N,) daily returns
    window        : rolling window length => each sample is (window,)
    T             : diffusion steps
    epochs        : training epochs
    n_gen         : how many synthetic paths to generate
    regime_labels : (N,) optional per-day regime labels

    Returns
    -------
    dict with model, scaler, generated paths, metrics
    """
    # Build windowed dataset
    N = len(returns)
    samples = np.array([returns[i:i + window]
                        for i in range(N - window + 1)])

    scaler = ReturnScaler().fit(samples)
    data = scaler.transform(samples)

    labels = None
    num_classes = 0
    if regime_labels is not None:
        labels = np.array([int(regime_labels[i + window - 1])
                           for i in range(N - window + 1)])
        num_classes = int(labels.max()) + 1

    model = DiffusionModel(data_dim=window, T=T,
                           num_classes=num_classes, seed=seed)
    losses = model.fit(data, epochs=epochs, lr=1e-4, class_labels=labels)

    gen = model.sample_ddim(n_gen, num_steps=50)
    gen_orig = scaler.inverse_transform(gen)

    metrics = DiffusionModel.compute_statistics(samples, gen_orig)

    return {
        "model": model,
        "scaler": scaler,
        "generated": gen_orig,
        "losses": losses,
        "metrics": metrics,
    }
