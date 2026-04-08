"""
Variational Autoencoder for financial regime encoding.

Numpy-only implementation with manual backpropagation and Adam optimizer.
Supports Beta-VAE, Conditional VAE, anomaly detection, and latent
space utilities for encoding market states into low-dimensional
representations.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_grad(s: np.ndarray) -> np.ndarray:
    return s * (1.0 - s)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(t: np.ndarray) -> np.ndarray:
    return 1.0 - t ** 2


_ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "tanh": (_tanh, _tanh_grad),
}

# ---------------------------------------------------------------------------
# Xavier / He initialization
# ---------------------------------------------------------------------------

def _he_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return rng.randn(fan_in, fan_out) * std


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.RandomState) -> np.ndarray:
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.randn(fan_in, fan_out) * std

# ---------------------------------------------------------------------------
# Adam optimizer state
# ---------------------------------------------------------------------------

class AdamState:
    """Maintains per-parameter Adam optimizer state."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, key: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
        self.t += 1
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ---------------------------------------------------------------------------
# Dense layer
# ---------------------------------------------------------------------------

class DenseLayer:
    """Fully-connected layer with configurable activation."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu",
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        if activation == "relu":
            self.W = _he_init(in_dim, out_dim, rng)
        else:
            self.W = _xavier_init(in_dim, out_dim, rng)
        self.b = np.zeros((1, out_dim))
        self.activation = activation
        self._act_fn, self._act_grad = _ACTIVATIONS.get(
            activation, (_relu, _relu_grad))
        # cache
        self._input = None
        self._pre_act = None
        self._output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        self._pre_act = x @ self.W + self.b
        self._output = self._act_fn(self._pre_act)
        return self._output

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.activation == "relu":
            d_pre = d_out * _relu_grad(self._pre_act)
        elif self.activation == "sigmoid":
            d_pre = d_out * _sigmoid_grad(self._output)
        elif self.activation == "tanh":
            d_pre = d_out * _tanh_grad(self._output)
        else:
            d_pre = d_out * _relu_grad(self._pre_act)
        dW = self._input.T @ d_pre / d_pre.shape[0]
        db = np.mean(d_pre, axis=0, keepdims=True)
        d_input = d_pre @ self.W.T
        return d_input, dW, db

# ---------------------------------------------------------------------------
# Linear (no activation) layer
# ---------------------------------------------------------------------------

class LinearLayer:
    """Linear projection without activation."""

    def __init__(self, in_dim: int, out_dim: int,
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.W = _xavier_init(in_dim, out_dim, rng)
        self.b = np.zeros((1, out_dim))
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.W + self.b

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dW = self._input.T @ d_out / d_out.shape[0]
        db = np.mean(d_out, axis=0, keepdims=True)
        d_input = d_out @ self.W.T
        return d_input, dW, db

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder:
    """Dense encoder: input -> hidden layers -> (mu, log_var)."""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int,
                 activation: str = "relu",
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.layers: List[DenseLayer] = []
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(DenseLayer(prev, h, activation, rng))
            prev = h
        self.mu_layer = LinearLayer(prev, latent_dim, rng)
        self.logvar_layer = LinearLayer(prev, latent_dim, rng)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        mu = self.mu_layer.forward(h)
        log_var = self.logvar_layer.forward(h)
        return mu, log_var

    def backward(self, d_mu: np.ndarray, d_logvar: np.ndarray):
        grads = {}
        d_h_mu, dW, db = self.mu_layer.backward(d_mu)
        grads["mu_W"] = dW
        grads["mu_b"] = db
        d_h_lv, dW, db = self.logvar_layer.backward(d_logvar)
        grads["logvar_W"] = dW
        grads["logvar_b"] = db
        d_h = d_h_mu + d_h_lv
        for i in reversed(range(len(self.layers))):
            d_h, dW, db = self.layers[i].backward(d_h)
            grads[f"enc_{i}_W"] = dW
            grads[f"enc_{i}_b"] = db
        return grads

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder:
    """Dense decoder: z -> hidden layers -> reconstructed input."""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int,
                 activation: str = "relu",
                 output_activation: str = "sigmoid",
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.layers: List[DenseLayer] = []
        prev = latent_dim
        for h in hidden_dims:
            self.layers.append(DenseLayer(prev, h, activation, rng))
            prev = h
        self.output_layer = DenseLayer(prev, output_dim, output_activation, rng)

    def forward(self, z: np.ndarray) -> np.ndarray:
        h = z
        for layer in self.layers:
            h = layer.forward(h)
        return self.output_layer.forward(h)

    def backward(self, d_out: np.ndarray):
        grads = {}
        d_h, dW, db = self.output_layer.backward(d_out)
        grads["dec_out_W"] = dW
        grads["dec_out_b"] = db
        for i in reversed(range(len(self.layers))):
            d_h, dW, db = self.layers[i].backward(d_h)
            grads[f"dec_{i}_W"] = dW
            grads[f"dec_{i}_b"] = db
        return d_h, grads

# ---------------------------------------------------------------------------
# Reparameterization
# ---------------------------------------------------------------------------

def reparameterize(mu: np.ndarray, log_var: np.ndarray,
                   rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """z = mu + eps * exp(0.5 * log_var).  Returns (z, eps)."""
    eps = rng.randn(*mu.shape)
    std = np.exp(0.5 * log_var)
    z = mu + eps * std
    return z, eps

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_loss_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean((x - x_hat) ** 2))


def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """KL(q(z|x) || p(z)) for diagonal Gaussian q and standard normal p."""
    return float(-0.5 * np.mean(np.sum(1 + log_var - mu ** 2 - np.exp(log_var), axis=1)))


def elbo_loss(x: np.ndarray, x_hat: np.ndarray,
              mu: np.ndarray, log_var: np.ndarray,
              beta: float = 1.0) -> Tuple[float, float, float]:
    recon = reconstruction_loss_mse(x, x_hat)
    kl = kl_divergence(mu, log_var)
    total = recon + beta * kl
    return total, recon, kl

# ---------------------------------------------------------------------------
# Gradients of loss w.r.t. decoder output, mu, log_var
# ---------------------------------------------------------------------------

def _grad_recon_mse(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    return 2.0 * (x_hat - x) / x.size


def _grad_kl_mu(mu: np.ndarray) -> np.ndarray:
    return mu / mu.shape[0]


def _grad_kl_logvar(log_var: np.ndarray) -> np.ndarray:
    return 0.5 * (np.exp(log_var) - 1.0) / log_var.shape[0]

# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------

class VariationalAutoencoder:
    """
    Full VAE with manual backprop and Adam optimizer.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features (e.g., market state vector).
    hidden_dims_enc : list of int
        Encoder hidden layer sizes.
    latent_dim : int
        Size of latent space.
    hidden_dims_dec : list of int
        Decoder hidden layer sizes.
    beta : float
        Weight on KL divergence term (beta-VAE).
    lr : float
        Learning rate for Adam.
    activation : str
        Activation function for hidden layers.
    output_activation : str
        Activation for decoder output layer.
    seed : int
        Random seed.
    """

    def __init__(self, input_dim: int, hidden_dims_enc: List[int],
                 latent_dim: int, hidden_dims_dec: Optional[List[int]] = None,
                 beta: float = 1.0, lr: float = 1e-3,
                 activation: str = "relu", output_activation: str = "sigmoid",
                 seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.beta = beta
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        if hidden_dims_dec is None:
            hidden_dims_dec = list(reversed(hidden_dims_enc))
        self.encoder = Encoder(input_dim, hidden_dims_enc, latent_dim,
                               activation, self.rng)
        self.decoder = Decoder(latent_dim, hidden_dims_dec, input_dim,
                               activation, output_activation, self.rng)
        self.adam = AdamState(lr=lr)
        self.history: List[Dict[str, float]] = []

    # ---- forward pass ----
    def _forward(self, x: np.ndarray):
        mu, log_var = self.encoder.forward(x)
        z, eps = reparameterize(mu, log_var, self.rng)
        x_hat = self.decoder.forward(z)
        return x_hat, mu, log_var, z, eps

    # ---- single training step ----
    def train_step(self, x: np.ndarray) -> Dict[str, float]:
        x_hat, mu, log_var, z, eps = self._forward(x)
        total, recon, kl = elbo_loss(x, x_hat, mu, log_var, self.beta)

        # Backprop through decoder
        d_xhat = _grad_recon_mse(x, x_hat)
        d_z, dec_grads = self.decoder.backward(d_xhat)

        # Backprop through reparameterization
        std = np.exp(0.5 * log_var)
        d_mu_reparam = d_z
        d_logvar_reparam = d_z * eps * 0.5 * std

        # Add KL gradients
        d_mu = d_mu_reparam + self.beta * _grad_kl_mu(mu)
        d_logvar = d_logvar_reparam + self.beta * _grad_kl_logvar(log_var)

        enc_grads = self.encoder.backward(d_mu, d_logvar)

        # Update encoder
        self.encoder.mu_layer.W = self.adam.step("mu_W", self.encoder.mu_layer.W, enc_grads["mu_W"])
        self.encoder.mu_layer.b = self.adam.step("mu_b", self.encoder.mu_layer.b, enc_grads["mu_b"])
        self.encoder.logvar_layer.W = self.adam.step("logvar_W", self.encoder.logvar_layer.W, enc_grads["logvar_W"])
        self.encoder.logvar_layer.b = self.adam.step("logvar_b", self.encoder.logvar_layer.b, enc_grads["logvar_b"])
        for i, layer in enumerate(self.encoder.layers):
            layer.W = self.adam.step(f"enc_{i}_W", layer.W, enc_grads[f"enc_{i}_W"])
            layer.b = self.adam.step(f"enc_{i}_b", layer.b, enc_grads[f"enc_{i}_b"])

        # Update decoder
        self.decoder.output_layer.W = self.adam.step("dec_out_W", self.decoder.output_layer.W, dec_grads["dec_out_W"])
        self.decoder.output_layer.b = self.adam.step("dec_out_b", self.decoder.output_layer.b, dec_grads["dec_out_b"])
        for i, layer in enumerate(self.decoder.layers):
            layer.W = self.adam.step(f"dec_{i}_W", layer.W, dec_grads[f"dec_{i}_W"])
            layer.b = self.adam.step(f"dec_{i}_b", layer.b, dec_grads[f"dec_{i}_b"])

        return {"total": total, "recon": recon, "kl": kl}

    # ---- full training loop ----
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32,
            verbose: bool = False) -> List[Dict[str, float]]:
        n = X.shape[0]
        self.history = []
        for epoch in range(epochs):
            indices = self.rng.permutation(n)
            epoch_loss = {"total": 0.0, "recon": 0.0, "kl": 0.0}
            n_batches = 0
            for start in range(0, n, batch_size):
                batch = X[indices[start:start + batch_size]]
                metrics = self.train_step(batch)
                for k in epoch_loss:
                    epoch_loss[k] += metrics[k]
                n_batches += 1
            for k in epoch_loss:
                epoch_loss[k] /= n_batches
            self.history.append(epoch_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"Epoch {epoch:4d} | loss={epoch_loss['total']:.5f} "
                      f"recon={epoch_loss['recon']:.5f} kl={epoch_loss['kl']:.5f}")
        return self.history

    # ---- encode ----
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.encoder.forward(x)

    # ---- decode ----
    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder.forward(z)

    # ---- reconstruct ----
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        mu, log_var = self.encode(x)
        z, _ = reparameterize(mu, log_var, self.rng)
        return self.decode(z)

    # ---- sample from prior ----
    def sample(self, n: int = 1) -> np.ndarray:
        z = self.rng.randn(n, self.latent_dim)
        return self.decode(z)

    # ---- reconstruction error per sample ----
    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        x_hat = self.reconstruct(x)
        return np.mean((x - x_hat) ** 2, axis=1)

    # ---- anomaly detection ----
    def detect_anomalies(self, x: np.ndarray,
                         threshold: Optional[float] = None,
                         quantile: float = 0.95) -> Tuple[np.ndarray, np.ndarray, float]:
        errors = self.reconstruction_error(x)
        if threshold is None:
            threshold = float(np.quantile(errors, quantile))
        anomalies = errors > threshold
        return anomalies, errors, threshold

# ---------------------------------------------------------------------------
# Conditional VAE
# ---------------------------------------------------------------------------

class ConditionalVAE:
    """
    Conditional Variational Autoencoder that conditions on a regime label.

    The label is one-hot encoded and concatenated with the input for the
    encoder and with the latent code for the decoder.
    """

    def __init__(self, input_dim: int, n_classes: int,
                 hidden_dims_enc: List[int], latent_dim: int,
                 hidden_dims_dec: Optional[List[int]] = None,
                 beta: float = 1.0, lr: float = 1e-3,
                 activation: str = "relu",
                 output_activation: str = "sigmoid",
                 seed: int = 42):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rng = np.random.RandomState(seed)
        self.beta = beta
        if hidden_dims_dec is None:
            hidden_dims_dec = list(reversed(hidden_dims_enc))

        enc_input = input_dim + n_classes
        dec_input = latent_dim + n_classes

        self.encoder = Encoder(enc_input, hidden_dims_enc, latent_dim,
                               activation, self.rng)
        self.decoder = Decoder(dec_input, hidden_dims_dec, input_dim,
                               activation, output_activation, self.rng)
        self.adam = AdamState(lr=lr)
        self.history: List[Dict[str, float]] = []

    def _one_hot(self, labels: np.ndarray) -> np.ndarray:
        oh = np.zeros((labels.shape[0], self.n_classes))
        oh[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
        return oh

    def _forward(self, x: np.ndarray, labels: np.ndarray):
        c = self._one_hot(labels)
        enc_input = np.concatenate([x, c], axis=1)
        mu, log_var = self.encoder.forward(enc_input)
        z, eps = reparameterize(mu, log_var, self.rng)
        dec_input = np.concatenate([z, c], axis=1)
        x_hat = self.decoder.forward(dec_input)
        return x_hat, mu, log_var, z, eps, c

    def train_step(self, x: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        x_hat, mu, log_var, z, eps, c = self._forward(x, labels)
        total, recon, kl = elbo_loss(x, x_hat, mu, log_var, self.beta)

        d_xhat = _grad_recon_mse(x, x_hat)
        d_dec_in, dec_grads = self.decoder.backward(d_xhat)

        d_z = d_dec_in[:, :self.latent_dim]

        std = np.exp(0.5 * log_var)
        d_mu = d_z + self.beta * _grad_kl_mu(mu)
        d_logvar = d_z * eps * 0.5 * std + self.beta * _grad_kl_logvar(log_var)

        enc_grads = self.encoder.backward(d_mu, d_logvar)

        # Update params (encoder)
        self.encoder.mu_layer.W = self.adam.step("mu_W", self.encoder.mu_layer.W, enc_grads["mu_W"])
        self.encoder.mu_layer.b = self.adam.step("mu_b", self.encoder.mu_layer.b, enc_grads["mu_b"])
        self.encoder.logvar_layer.W = self.adam.step("logvar_W", self.encoder.logvar_layer.W, enc_grads["logvar_W"])
        self.encoder.logvar_layer.b = self.adam.step("logvar_b", self.encoder.logvar_layer.b, enc_grads["logvar_b"])
        for i, layer in enumerate(self.encoder.layers):
            layer.W = self.adam.step(f"enc_{i}_W", layer.W, enc_grads[f"enc_{i}_W"])
            layer.b = self.adam.step(f"enc_{i}_b", layer.b, enc_grads[f"enc_{i}_b"])
        # Update params (decoder)
        self.decoder.output_layer.W = self.adam.step("dec_out_W", self.decoder.output_layer.W, dec_grads["dec_out_W"])
        self.decoder.output_layer.b = self.adam.step("dec_out_b", self.decoder.output_layer.b, dec_grads["dec_out_b"])
        for i, layer in enumerate(self.decoder.layers):
            layer.W = self.adam.step(f"dec_{i}_W", layer.W, dec_grads[f"dec_{i}_W"])
            layer.b = self.adam.step(f"dec_{i}_b", layer.b, dec_grads[f"dec_{i}_b"])

        return {"total": total, "recon": recon, "kl": kl}

    def fit(self, X: np.ndarray, labels: np.ndarray,
            epochs: int = 100, batch_size: int = 32,
            verbose: bool = False) -> List[Dict[str, float]]:
        n = X.shape[0]
        self.history = []
        for epoch in range(epochs):
            idx = self.rng.permutation(n)
            epoch_loss = {"total": 0.0, "recon": 0.0, "kl": 0.0}
            nb = 0
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                m = self.train_step(X[bi], labels[bi])
                for k in epoch_loss:
                    epoch_loss[k] += m[k]
                nb += 1
            for k in epoch_loss:
                epoch_loss[k] /= nb
            self.history.append(epoch_loss)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:4d} | loss={epoch_loss['total']:.5f}")
        return self.history

    def encode(self, x: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = self._one_hot(labels)
        return self.encoder.forward(np.concatenate([x, c], axis=1))

    def decode(self, z: np.ndarray, labels: np.ndarray) -> np.ndarray:
        c = self._one_hot(labels)
        return self.decoder.forward(np.concatenate([z, c], axis=1))

    def sample(self, labels: np.ndarray) -> np.ndarray:
        z = self.rng.randn(labels.shape[0], self.latent_dim)
        return self.decode(z, labels)

    def reconstruct(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        mu, lv = self.encode(x, labels)
        z, _ = reparameterize(mu, lv, self.rng)
        return self.decode(z, labels)

# ---------------------------------------------------------------------------
# Latent space visualization utilities
# ---------------------------------------------------------------------------

class LatentSpaceVisualizer:
    """Utilities for analyzing and visualizing the VAE latent space."""

    @staticmethod
    def compute_latent_embeddings(vae: VariationalAutoencoder,
                                  X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, log_var = vae.encode(X)
        return mu, log_var

    @staticmethod
    def pca_2d(mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project latent means to 2D via PCA. Returns (projected, components, explained_var)."""
        centered = mu - mu.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        components = eigvecs[:, :2]
        projected = centered @ components
        total_var = eigvals.sum()
        explained = eigvals[:2] / (total_var + 1e-12)
        return projected, components, explained

    @staticmethod
    def latent_interpolation(vae: VariationalAutoencoder,
                             z_start: np.ndarray, z_end: np.ndarray,
                             n_steps: int = 10) -> np.ndarray:
        """Linear interpolation in latent space, decoded back to input space."""
        alphas = np.linspace(0, 1, n_steps)
        interps = []
        for a in alphas:
            z = (1 - a) * z_start + a * z_end
            interps.append(vae.decode(z.reshape(1, -1)))
        return np.concatenate(interps, axis=0)

    @staticmethod
    def latent_grid_decode(vae: VariationalAutoencoder,
                           dim1: int = 0, dim2: int = 1,
                           n_grid: int = 10,
                           range_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Decode a 2D grid in latent space (varying dim1 and dim2)."""
        vals = np.linspace(-range_std, range_std, n_grid)
        z_base = np.zeros((1, vae.latent_dim))
        decoded = []
        coords = []
        for v1 in vals:
            for v2 in vals:
                z = z_base.copy()
                z[0, dim1] = v1
                z[0, dim2] = v2
                decoded.append(vae.decode(z))
                coords.append([v1, v2])
        return np.concatenate(decoded, axis=0), np.array(coords)

    @staticmethod
    def latent_cluster_stats(mu: np.ndarray,
                             labels: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """Compute per-cluster mean and covariance in latent space."""
        stats = {}
        for c in np.unique(labels):
            mask = labels == c
            cluster_mu = mu[mask]
            stats[int(c)] = {
                "mean": cluster_mu.mean(axis=0),
                "std": cluster_mu.std(axis=0),
                "cov": np.cov(cluster_mu.T) if cluster_mu.shape[0] > 1 else np.zeros((mu.shape[1], mu.shape[1])),
                "count": int(mask.sum()),
            }
        return stats

# ---------------------------------------------------------------------------
# Anomaly detection utilities
# ---------------------------------------------------------------------------

class RegimeAnomalyDetector:
    """
    Anomaly detection wrapper around a trained VAE.

    High reconstruction error indicates an anomalous regime not well
    represented by the training data.
    """

    def __init__(self, vae: VariationalAutoencoder, calibration_data: np.ndarray,
                 quantile: float = 0.95):
        self.vae = vae
        errors = vae.reconstruction_error(calibration_data)
        self.threshold = float(np.quantile(errors, quantile))
        self.calibration_mean = float(np.mean(errors))
        self.calibration_std = float(np.std(errors))

    def score(self, x: np.ndarray) -> np.ndarray:
        """Return anomaly score (reconstruction error) per sample."""
        return self.vae.reconstruction_error(x)

    def z_score(self, x: np.ndarray) -> np.ndarray:
        """Return z-score of reconstruction error relative to calibration set."""
        errors = self.score(x)
        return (errors - self.calibration_mean) / (self.calibration_std + 1e-12)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return boolean anomaly flags."""
        return self.score(x) > self.threshold

    def predict_with_scores(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        errors = self.score(x)
        return errors > self.threshold, errors, self.threshold

# ---------------------------------------------------------------------------
# Market state encoder application
# ---------------------------------------------------------------------------

class MarketStateEncoder:
    """
    Encode market state (returns, volatility, correlations) into a
    low-dimensional latent representation using a VAE.

    Provides preprocessing (normalize), training, encoding, and regime
    detection as a single pipeline.
    """

    def __init__(self, n_assets: int, lookback: int = 20,
                 latent_dim: int = 4, hidden_dims: Optional[List[int]] = None,
                 beta: float = 1.0, lr: float = 1e-3, seed: int = 42):
        self.n_assets = n_assets
        self.lookback = lookback
        self.latent_dim = latent_dim
        n_corr = n_assets * (n_assets - 1) // 2
        self.feature_dim = n_assets * 2 + n_corr  # returns + vols + upper-tri corr
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.vae = VariationalAutoencoder(
            self.feature_dim, hidden_dims, latent_dim,
            beta=beta, lr=lr, seed=seed)
        self._mean = None
        self._std = None

    def _extract_features(self, returns: np.ndarray) -> np.ndarray:
        """Extract mean return, vol, and upper-tri correlation from windows."""
        n_windows = returns.shape[0] - self.lookback + 1
        features = []
        for i in range(n_windows):
            window = returns[i:i + self.lookback]
            mean_ret = window.mean(axis=0)
            vol = window.std(axis=0)
            corr = np.corrcoef(window.T)
            upper = corr[np.triu_indices(self.n_assets, k=1)]
            features.append(np.concatenate([mean_ret, vol, upper]))
        return np.array(features)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-8
        return (X - self._mean) / self._std

    def _denormalize(self, X: np.ndarray) -> np.ndarray:
        return X * self._std + self._mean

    def fit(self, returns: np.ndarray, epochs: int = 100,
            batch_size: int = 32, verbose: bool = False) -> List[Dict[str, float]]:
        X = self._extract_features(returns)
        X_norm = self._normalize(X, fit=True)
        return self.vae.fit(X_norm, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def encode(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._extract_features(returns)
        X_norm = self._normalize(X)
        return self.vae.encode(X_norm)

    def detect_anomalies(self, returns: np.ndarray,
                         quantile: float = 0.95) -> Tuple[np.ndarray, np.ndarray, float]:
        X = self._extract_features(returns)
        X_norm = self._normalize(X)
        return self.vae.detect_anomalies(X_norm, quantile=quantile)

    def latent_distance(self, returns: np.ndarray) -> np.ndarray:
        """Mahalanobis-like distance of each encoded state from the prior mean (0)."""
        mu, log_var = self.encode(returns)
        var = np.exp(log_var)
        return np.sqrt(np.sum(mu ** 2 / (var + 1e-8), axis=1))


# ---------------------------------------------------------------------------
# Utility: training curve analysis
# ---------------------------------------------------------------------------

def analyze_training_curve(history: List[Dict[str, float]]) -> Dict[str, float]:
    """Summarize a VAE training history."""
    total = [h["total"] for h in history]
    recon = [h["recon"] for h in history]
    kl = [h["kl"] for h in history]
    return {
        "final_total": total[-1],
        "final_recon": recon[-1],
        "final_kl": kl[-1],
        "min_total": min(total),
        "total_improvement": total[0] - total[-1],
        "convergence_epoch": int(np.argmin(total)),
        "n_epochs": len(history),
    }
