"""
Normalizing flows for return distribution modeling.

Numpy-only implementation of planar flows, radial flows, and
simplified RealNVP affine coupling layers.  Supports density
estimation, sampling, and tail probability estimation for
fat-tailed financial return distributions.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict

# ---------------------------------------------------------------------------
# Base distribution
# ---------------------------------------------------------------------------

class StandardNormal:
    """Multivariate standard normal as the base distribution."""

    def __init__(self, dim: int):
        self.dim = dim

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        """Log probability under N(0, I). Shape (batch,)."""
        return -0.5 * (self.dim * np.log(2 * np.pi) + np.sum(z ** 2, axis=1))

    def sample(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        return rng.randn(n, self.dim)

# ---------------------------------------------------------------------------
# Abstract flow layer
# ---------------------------------------------------------------------------

class FlowLayer:
    """Base class for a single invertible transformation."""

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform z -> z', return (z', log_det_jacobian)."""
        raise NotImplementedError

    def backward(self, z_prime: np.ndarray) -> np.ndarray:
        """Inverse transform z' -> z (approximate if no closed form)."""
        raise NotImplementedError

    def parameters(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def set_parameters(self, params: Dict[str, np.ndarray]):
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Planar Flow
# ---------------------------------------------------------------------------

class PlanarFlow(FlowLayer):
    """
    Planar flow: z' = z + u * tanh(w^T z + b)
    log|det dz'/dz| = log|1 + u^T * h'(w^T z + b) * w|
    """

    def __init__(self, dim: int, rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.dim = dim
        self.w = rng.randn(dim) * 0.1
        self.u = rng.randn(dim) * 0.1
        self.b = np.zeros(1)
        self._cache_z = None
        self._cache_act = None
        self._cache_tanh = None

    def _get_u_hat(self) -> np.ndarray:
        """Enforce invertibility constraint on u."""
        wtu = np.dot(self.w, self.u)
        m_wtu = -1 + np.log(1 + np.exp(wtu))
        u_hat = self.u + (m_wtu - wtu) * self.w / (np.dot(self.w, self.w) + 1e-12)
        return u_hat

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u_hat = self._get_u_hat()
        act = z @ self.w + self.b  # (batch,)
        tanh_act = np.tanh(act)
        z_prime = z + np.outer(tanh_act, u_hat)
        dtanh = 1.0 - tanh_act ** 2
        psi = np.outer(dtanh, self.w)  # (batch, dim)
        log_det = np.log(np.abs(1.0 + psi @ u_hat) + 1e-12)
        self._cache_z = z
        self._cache_act = act
        self._cache_tanh = tanh_act
        return z_prime, log_det

    def backward(self, z_prime: np.ndarray) -> np.ndarray:
        """Approximate inverse via fixed-point iteration."""
        z = z_prime.copy()
        u_hat = self._get_u_hat()
        for _ in range(20):
            act = z @ self.w + self.b
            z = z_prime - np.outer(np.tanh(act), u_hat)
        return z

    def grad(self, d_out: np.ndarray, d_logdet: np.ndarray):
        """Compute gradients w.r.t. w, u, b given upstream gradients."""
        u_hat = self._get_u_hat()
        tanh_act = self._cache_tanh
        dtanh = 1.0 - tanh_act ** 2
        z = self._cache_z
        batch = z.shape[0]

        # grad from z' = z + tanh(w^Tz+b) * u_hat
        # d_loss/d_u_hat += d_out^T * tanh_act
        d_u = np.mean(d_out * tanh_act[:, None], axis=0)
        # d_loss/d_w through tanh
        d_tanh = np.sum(d_out * u_hat[None, :], axis=1)  # (batch,)
        d_act_from_zprime = d_tanh * dtanh  # (batch,)
        d_w = np.mean(d_act_from_zprime[:, None] * z, axis=0)
        d_b = np.mean(d_act_from_zprime)

        # grad from log_det
        psi_u = dtanh * (z @ self.w) # simplified
        # Just aggregate d_logdet contribution
        abs_det = np.abs(1.0 + np.outer(dtanh, self.w) @ u_hat) + 1e-12
        sign_det = np.sign(1.0 + np.outer(dtanh, self.w) @ u_hat)
        d2tanh = -2.0 * tanh_act * dtanh

        d_w_ld = np.zeros_like(self.w)
        d_u_ld = np.zeros_like(self.u)
        d_b_ld = 0.0
        for i in range(batch):
            coeff = d_logdet[i] * sign_det[i] / abs_det[i]
            d_u_ld += coeff * dtanh[i] * self.w
            d_w_ld += coeff * (dtanh[i] * u_hat + d2tanh[i] * (self.w @ z[i].T + self.b) * u_hat)  # approximate
            d_b_ld += coeff * d2tanh[i] * np.dot(self.w, u_hat)
        d_w_ld /= batch
        d_u_ld /= batch
        d_b_ld /= batch

        return {"w": d_w + d_w_ld, "u": d_u + d_u_ld, "b": np.array([d_b + d_b_ld])}

    def parameters(self) -> Dict[str, np.ndarray]:
        return {"w": self.w, "u": self.u, "b": self.b}

    def set_parameters(self, params: Dict[str, np.ndarray]):
        self.w = params["w"]
        self.u = params["u"]
        self.b = params["b"]

# ---------------------------------------------------------------------------
# Radial Flow
# ---------------------------------------------------------------------------

class RadialFlow(FlowLayer):
    """
    Radial flow: z' = z + beta * h(alpha, r) * (z - z0)
    where r = ||z - z0||, h(alpha, r) = 1 / (alpha + r)
    """

    def __init__(self, dim: int, rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.dim = dim
        self.z0 = rng.randn(dim) * 0.1
        self.log_alpha = np.zeros(1)
        self.beta_raw = np.zeros(1)
        self._cache_z = None
        self._cache_r = None
        self._cache_h = None

    @property
    def alpha(self) -> float:
        return float(np.exp(self.log_alpha[0]))

    @property
    def beta(self) -> float:
        """Enforce beta > -alpha to ensure invertibility."""
        return -self.alpha + np.log(1 + np.exp(self.beta_raw[0]))

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        diff = z - self.z0
        r = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True) + 1e-12)
        h = 1.0 / (self.alpha + r)
        beta = self.beta
        z_prime = z + beta * h * diff

        h_prime = -1.0 / (self.alpha + r) ** 2
        log_det = (self.dim - 1) * np.log(np.abs(1 + beta * h).ravel() + 1e-12) + \
                  np.log(np.abs(1 + beta * h + beta * h_prime * r).ravel() + 1e-12)

        self._cache_z = z
        self._cache_r = r
        self._cache_h = h
        return z_prime, log_det

    def backward(self, z_prime: np.ndarray) -> np.ndarray:
        """Approximate inverse via fixed-point iteration."""
        z = z_prime.copy()
        for _ in range(20):
            diff = z - self.z0
            r = np.sqrt(np.sum(diff ** 2, axis=1, keepdims=True) + 1e-12)
            h = 1.0 / (self.alpha + r)
            z = z_prime - self.beta * h * diff
        return z

    def parameters(self) -> Dict[str, np.ndarray]:
        return {"z0": self.z0, "log_alpha": self.log_alpha, "beta_raw": self.beta_raw}

    def set_parameters(self, params: Dict[str, np.ndarray]):
        self.z0 = params["z0"]
        self.log_alpha = params["log_alpha"]
        self.beta_raw = params["beta_raw"]

# ---------------------------------------------------------------------------
# Affine Coupling Layer (simplified RealNVP)
# ---------------------------------------------------------------------------

def _mlp_forward(x: np.ndarray, weights: List[Tuple[np.ndarray, np.ndarray]],
                 activation: str = "relu") -> Tuple[np.ndarray, List[np.ndarray]]:
    """Simple MLP forward pass returning activations cache."""
    cache = [x]
    h = x
    for i, (W, b) in enumerate(weights):
        h = h @ W + b
        if i < len(weights) - 1:
            if activation == "relu":
                h = np.maximum(0, h)
            else:
                h = np.tanh(h)
        cache.append(h)
    return h, cache


def _mlp_init(dims: List[int], rng: np.random.RandomState) -> List[Tuple[np.ndarray, np.ndarray]]:
    weights = []
    for i in range(len(dims) - 1):
        std = np.sqrt(2.0 / dims[i])
        W = rng.randn(dims[i], dims[i + 1]) * std
        b = np.zeros((1, dims[i + 1]))
        weights.append((W, b))
    return weights


class AffineCouplingLayer(FlowLayer):
    """
    Simplified RealNVP affine coupling layer.

    Split input into (z_a, z_b). Compute s, t = NN(z_a).
    z_a' = z_a (unchanged), z_b' = z_b * exp(s) + t.
    """

    def __init__(self, dim: int, hidden_dim: int = 32,
                 split_idx: Optional[int] = None,
                 rng: Optional[np.random.RandomState] = None):
        rng = rng or np.random.RandomState(42)
        self.dim = dim
        self.split = split_idx or dim // 2
        d_a = self.split
        d_b = dim - self.split

        self.s_net = _mlp_init([d_a, hidden_dim, hidden_dim, d_b], rng)
        self.t_net = _mlp_init([d_a, hidden_dim, hidden_dim, d_b], rng)
        self._cache = {}

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_a = z[:, :self.split]
        z_b = z[:, self.split:]

        s, s_cache = _mlp_forward(z_a, self.s_net)
        s = np.tanh(s) * 2.0  # bound log-scale
        t, t_cache = _mlp_forward(z_a, self.t_net)

        z_b_prime = z_b * np.exp(s) + t
        z_prime = np.concatenate([z_a, z_b_prime], axis=1)
        log_det = np.sum(s, axis=1)

        self._cache = {"z_a": z_a, "z_b": z_b, "s": s, "t": t,
                       "s_cache": s_cache, "t_cache": t_cache}
        return z_prime, log_det

    def backward(self, z_prime: np.ndarray) -> np.ndarray:
        """Exact inverse."""
        z_a = z_prime[:, :self.split]
        z_b_prime = z_prime[:, self.split:]
        s, _ = _mlp_forward(z_a, self.s_net)
        s = np.tanh(s) * 2.0
        t, _ = _mlp_forward(z_a, self.t_net)
        z_b = (z_b_prime - t) * np.exp(-s)
        return np.concatenate([z_a, z_b], axis=1)

    def parameters(self) -> Dict[str, np.ndarray]:
        params = {}
        for i, (W, b) in enumerate(self.s_net):
            params[f"s_W{i}"] = W
            params[f"s_b{i}"] = b
        for i, (W, b) in enumerate(self.t_net):
            params[f"t_W{i}"] = W
            params[f"t_b{i}"] = b
        return params

    def set_parameters(self, params: Dict[str, np.ndarray]):
        for i in range(len(self.s_net)):
            self.s_net[i] = (params[f"s_W{i}"], params[f"s_b{i}"])
        for i in range(len(self.t_net)):
            self.t_net[i] = (params[f"t_W{i}"], params[f"t_b{i}"])

# ---------------------------------------------------------------------------
# Flow chain
# ---------------------------------------------------------------------------

class NormalizingFlow:
    """
    Chain of flow transformations with a standard normal base.

    Supports training via maximum likelihood, density evaluation,
    and sampling.
    """

    def __init__(self, dim: int, flow_layers: Optional[List[FlowLayer]] = None,
                 lr: float = 1e-3, seed: int = 42):
        self.dim = dim
        self.base = StandardNormal(dim)
        self.rng = np.random.RandomState(seed)
        self.flows: List[FlowLayer] = flow_layers or []
        self.lr = lr
        self.history: List[float] = []
        # Adam state per parameter
        self._adam_m: Dict[str, np.ndarray] = {}
        self._adam_v: Dict[str, np.ndarray] = {}
        self._adam_t = 0

    def add_flow(self, flow: FlowLayer):
        self.flows.append(flow)

    def forward(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform base sample through all flows. Returns (x, total_log_det)."""
        total_log_det = np.zeros(z.shape[0])
        for flow in self.flows:
            z, ld = flow.forward(z)
            total_log_det += ld
        return z, total_log_det

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Map from data space back to base space."""
        z = x
        for flow in reversed(self.flows):
            z = flow.backward(z)
        return z

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Evaluate log p(x) = log p_base(z) + sum of log|det|."""
        z = x
        total_log_det = np.zeros(x.shape[0])
        # Need to go through inverse and accumulate log det of inverse
        # For flows with tractable inverse log-det, we can accumulate forward
        # Alternatively: transform from base to x and use change of variables
        # Here we use the forward direction: sample z, get x, accumulate ld
        # But for evaluation we need the inverse direction.
        # Use inverse to get z, then compute forward log_det for consistency.
        z_base = self.inverse(x)
        # Re-run forward to get log_det
        _, total_log_det = self.forward(z_base)
        return self.base.log_prob(z_base) + total_log_det

    def sample(self, n: int) -> np.ndarray:
        z = self.base.sample(n, self.rng)
        x, _ = self.forward(z)
        return x

    def _adam_update(self, key: str, param: np.ndarray, grad: np.ndarray,
                     lr: float = None, beta1: float = 0.9,
                     beta2: float = 0.999, eps: float = 1e-8) -> np.ndarray:
        lr = lr or self.lr
        if key not in self._adam_m:
            self._adam_m[key] = np.zeros_like(param)
            self._adam_v[key] = np.zeros_like(param)
        self._adam_m[key] = beta1 * self._adam_m[key] + (1 - beta1) * grad
        self._adam_v[key] = beta2 * self._adam_v[key] + (1 - beta2) * grad ** 2
        t = self._adam_t
        m_hat = self._adam_m[key] / (1 - beta1 ** t)
        v_hat = self._adam_v[key] / (1 - beta2 ** t)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)

    def _numerical_grad(self, x_batch: np.ndarray, flow_idx: int,
                        param_name: str, eps: float = 1e-5) -> np.ndarray:
        """Compute gradient of negative log-likelihood via finite differences."""
        flow = self.flows[flow_idx]
        params = flow.parameters()
        p = params[param_name]
        grad = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = p[idx]
            p[idx] = old + eps
            params[param_name] = p
            flow.set_parameters(params)
            lp_plus = np.mean(self.log_prob(x_batch))

            p[idx] = old - eps
            params[param_name] = p
            flow.set_parameters(params)
            lp_minus = np.mean(self.log_prob(x_batch))

            grad[idx] = -(lp_plus - lp_minus) / (2 * eps)  # negative for NLL

            p[idx] = old
            params[param_name] = p
            flow.set_parameters(params)
            it.iternext()
        return grad

    def train_step(self, x_batch: np.ndarray) -> float:
        """Single gradient step minimizing negative log-likelihood."""
        self._adam_t += 1
        nll = -float(np.mean(self.log_prob(x_batch)))

        for fi, flow in enumerate(self.flows):
            params = flow.parameters()
            new_params = {}
            for pname, pval in params.items():
                grad = self._numerical_grad(x_batch, fi, pname)
                new_params[pname] = self._adam_update(
                    f"flow{fi}_{pname}", pval, grad)
            flow.set_parameters(new_params)
        return nll

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 64,
            verbose: bool = False) -> List[float]:
        n = X.shape[0]
        self.history = []
        for epoch in range(epochs):
            idx = self.rng.permutation(n)
            epoch_nll = 0.0
            nb = 0
            for s in range(0, n, batch_size):
                batch = X[idx[s:s + batch_size]]
                nll = self.train_step(batch)
                epoch_nll += nll
                nb += 1
            epoch_nll /= nb
            self.history.append(epoch_nll)
            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch:4d} | NLL={epoch_nll:.4f}")
        return self.history

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_planar_flow(dim: int, n_flows: int = 8, lr: float = 1e-3,
                      seed: int = 42) -> NormalizingFlow:
    """Build a normalizing flow with a chain of planar transformations."""
    rng = np.random.RandomState(seed)
    layers = [PlanarFlow(dim, rng) for _ in range(n_flows)]
    return NormalizingFlow(dim, layers, lr=lr, seed=seed)


def build_radial_flow(dim: int, n_flows: int = 8, lr: float = 1e-3,
                      seed: int = 42) -> NormalizingFlow:
    """Build a normalizing flow with a chain of radial transformations."""
    rng = np.random.RandomState(seed)
    layers = [RadialFlow(dim, rng) for _ in range(n_flows)]
    return NormalizingFlow(dim, layers, lr=lr, seed=seed)


def build_realnvp_flow(dim: int, n_coupling: int = 4, hidden_dim: int = 32,
                       lr: float = 1e-3, seed: int = 42) -> NormalizingFlow:
    """Build a RealNVP-style flow with affine coupling layers."""
    rng = np.random.RandomState(seed)
    layers = []
    for i in range(n_coupling):
        # Alternate split direction
        split = dim // 2 if i % 2 == 0 else dim - dim // 2
        layers.append(AffineCouplingLayer(dim, hidden_dim, split, rng))
    return NormalizingFlow(dim, layers, lr=lr, seed=seed)

# ---------------------------------------------------------------------------
# Density estimation utilities
# ---------------------------------------------------------------------------

class DensityEstimator:
    """Wrapper for using a trained flow as a density estimator."""

    def __init__(self, flow: NormalizingFlow):
        self.flow = flow

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return self.flow.log_prob(x)

    def density(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.log_density(x))

    def tail_probability(self, threshold: float, n_samples: int = 10000,
                         direction: str = "left") -> float:
        """Estimate P(X < threshold) or P(X > threshold) via sampling."""
        samples = self.flow.sample(n_samples)
        if samples.shape[1] == 1:
            s = samples.ravel()
        else:
            s = np.sum(samples, axis=1)
        if direction == "left":
            return float(np.mean(s < threshold))
        return float(np.mean(s > threshold))

    def value_at_risk(self, alpha: float = 0.05,
                      n_samples: int = 50000) -> float:
        """Estimate VaR at level alpha from the flow model."""
        samples = self.flow.sample(n_samples)
        if samples.shape[1] == 1:
            s = samples.ravel()
        else:
            s = np.sum(samples, axis=1)
        return float(np.quantile(s, alpha))

    def expected_shortfall(self, alpha: float = 0.05,
                           n_samples: int = 50000) -> float:
        """Estimate CVaR (expected shortfall) at level alpha."""
        samples = self.flow.sample(n_samples)
        if samples.shape[1] == 1:
            s = samples.ravel()
        else:
            s = np.sum(samples, axis=1)
        var = np.quantile(s, alpha)
        return float(np.mean(s[s <= var]))

# ---------------------------------------------------------------------------
# Application: fat-tailed return modeling
# ---------------------------------------------------------------------------

class ReturnDistributionModel:
    """
    Model fat-tailed return distributions using normalizing flows.

    Preprocesses returns, fits a flow model, and provides risk metrics.
    """

    def __init__(self, flow_type: str = "realnvp", n_flows: int = 6,
                 hidden_dim: int = 32, lr: float = 5e-4, seed: int = 42):
        self.flow_type = flow_type
        self.n_flows = n_flows
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.seed = seed
        self.flow: Optional[NormalizingFlow] = None
        self.estimator: Optional[DensityEstimator] = None
        self._mean = None
        self._std = None

    def _normalize(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._mean = x.mean(axis=0)
            self._std = x.std(axis=0) + 1e-8
        return (x - self._mean) / self._std

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self._std + self._mean

    def fit(self, returns: np.ndarray, epochs: int = 50,
            batch_size: int = 64, verbose: bool = False) -> List[float]:
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        dim = returns.shape[1]
        X = self._normalize(returns, fit=True)

        if self.flow_type == "planar":
            self.flow = build_planar_flow(dim, self.n_flows, self.lr, self.seed)
        elif self.flow_type == "radial":
            self.flow = build_radial_flow(dim, self.n_flows, self.lr, self.seed)
        else:
            self.flow = build_realnvp_flow(dim, self.n_flows, self.hidden_dim,
                                           self.lr, self.seed)
        self.estimator = DensityEstimator(self.flow)
        return self.flow.fit(X, epochs, batch_size, verbose)

    def log_prob(self, returns: np.ndarray) -> np.ndarray:
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        X = self._normalize(returns)
        log_p = self.flow.log_prob(X)
        # Adjust for normalization: log|dx/dx_norm| = -sum(log(std))
        log_p -= np.sum(np.log(self._std))
        return log_p

    def sample(self, n: int) -> np.ndarray:
        X = self.flow.sample(n)
        return self._denormalize(X)

    def var(self, alpha: float = 0.05, n_samples: int = 50000) -> float:
        samples = self.sample(n_samples)
        if samples.shape[1] == 1:
            return float(np.quantile(samples.ravel(), alpha))
        return float(np.quantile(np.sum(samples, axis=1), alpha))

    def cvar(self, alpha: float = 0.05, n_samples: int = 50000) -> float:
        samples = self.sample(n_samples)
        s = samples.ravel() if samples.shape[1] == 1 else np.sum(samples, axis=1)
        cutoff = np.quantile(s, alpha)
        return float(np.mean(s[s <= cutoff]))

    def tail_index_estimate(self, n_samples: int = 50000,
                            tail_fraction: float = 0.05) -> Dict[str, float]:
        """Estimate tail index using Hill estimator on flow samples."""
        samples = self.sample(n_samples).ravel()
        # Right tail
        sorted_abs = np.sort(np.abs(samples))[::-1]
        k = max(int(n_samples * tail_fraction), 10)
        top_k = sorted_abs[:k]
        threshold = sorted_abs[k]
        if threshold > 0:
            hill = np.mean(np.log(top_k / threshold))
            alpha_hill = 1.0 / (hill + 1e-12)
        else:
            alpha_hill = np.inf
        return {
            "hill_estimator": float(hill) if threshold > 0 else 0.0,
            "tail_index": float(alpha_hill),
            "k": k,
            "threshold": float(threshold),
        }

# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_flow_models(X: np.ndarray, flow_types: List[str] = None,
                        n_flows: int = 6, epochs: int = 30,
                        seed: int = 42) -> Dict[str, Dict[str, float]]:
    """Compare different flow architectures on the same data."""
    if flow_types is None:
        flow_types = ["planar", "radial", "realnvp"]
    results = {}
    for ft in flow_types:
        model = ReturnDistributionModel(flow_type=ft, n_flows=n_flows, seed=seed)
        history = model.fit(X, epochs=epochs)
        final_nll = history[-1] if history else float("inf")
        lp = model.log_prob(X[:min(500, len(X))])
        results[ft] = {
            "final_nll": final_nll,
            "mean_log_prob": float(np.mean(lp)),
            "std_log_prob": float(np.std(lp)),
        }
    return results


def gaussian_baseline_log_prob(X: np.ndarray) -> float:
    """Log-likelihood under a fitted multivariate Gaussian."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    if X.shape[1] == 1:
        cov = np.atleast_2d(cov)
    d = X.shape[1]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("-inf")
    diff = X - mu
    inv_cov = np.linalg.inv(cov)
    mahal = np.sum(diff @ inv_cov * diff, axis=1)
    lp = -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)
    return float(np.mean(lp))
