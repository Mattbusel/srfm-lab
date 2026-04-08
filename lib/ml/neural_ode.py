"""
Neural ODEs for continuous-time financial dynamics.

ODE solvers (RK4, adaptive RK45), Neural ODE with adjoint method,
Continuous Normalizing Flows, Latent ODE, Augmented Neural ODE,
and applications to volatility modeling and irregular time series.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable


# ---------------------------------------------------------------------------
# Helpers: activation functions and their derivatives
# ---------------------------------------------------------------------------

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _tanh_deriv(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -20, 20)))

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# 1. ODE Solver: RK4 (fixed step)
# ---------------------------------------------------------------------------

class RK4Solver:
    """Classical 4th-order Runge-Kutta solver."""

    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def solve(self, f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
              t_eval: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        if t_eval is None:
            n_steps = max(int(np.ceil((tf - t0) / self.dt)), 1)
            t_eval = np.linspace(t0, tf, n_steps + 1)
        ys = [y0.copy()]
        y = y0.copy()
        t_curr = t0
        eval_idx = 1
        dt = self.dt
        while eval_idx < len(t_eval):
            t_next = t_eval[eval_idx]
            while t_curr < t_next - 1e-12:
                h = min(dt, t_next - t_curr)
                k1 = f(t_curr, y)
                k2 = f(t_curr + h / 2, y + h / 2 * k1)
                k3 = f(t_curr + h / 2, y + h / 2 * k2)
                k4 = f(t_curr + h, y + h * k3)
                y = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                t_curr += h
            ys.append(y.copy())
            eval_idx += 1
        return t_eval, np.array(ys)


# ---------------------------------------------------------------------------
# 2. Adaptive RK45 (Dormand-Prince)
# ---------------------------------------------------------------------------

class RK45Solver:
    """Adaptive step-size Dormand-Prince RK45 solver."""

    # Dormand-Prince coefficients
    A = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    B = [
        np.array([]),
        np.array([1/5]),
        np.array([3/40, 9/40]),
        np.array([44/45, -56/15, 32/9]),
        np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
        np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]),
        np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]),
    ]
    C4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    C5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])

    def __init__(self, atol: float = 1e-6, rtol: float = 1e-3,
                 max_step: float = 1.0, min_step: float = 1e-10):
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.min_step = min_step

    def solve(self, f: Callable, y0: np.ndarray, t_span: Tuple[float, float],
              t_eval: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        if t_eval is None:
            t_eval = np.linspace(t0, tf, 101)
        y = y0.copy()
        t = t0
        h = min(self.max_step, (tf - t0) / 10)
        ts_out = [t0]
        ys_out = [y0.copy()]
        ts_dense = [t0]
        ys_dense = [y0.copy()]
        max_iters = 100000
        iters = 0
        while t < tf - 1e-14 and iters < max_iters:
            iters += 1
            h = min(h, tf - t)
            h = max(h, self.min_step)
            k = [None] * 7
            k[0] = f(t, y)
            for i in range(1, 7):
                yi = y.copy()
                for j in range(i):
                    yi = yi + h * self.B[i][j] * k[j]
                k[i] = f(t + self.A[i] * h, yi)
            # 5th order solution
            y5 = y.copy()
            for i in range(7):
                y5 = y5 + h * self.C5[i] * k[i]
            # 4th order solution
            y4 = y.copy()
            for i in range(7):
                y4 = y4 + h * self.C4[i] * k[i]
            # Error
            err = np.max(np.abs(y5 - y4) / (self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y5))))
            if err <= 1.0 or h <= self.min_step:
                t += h
                y = y5
                ts_dense.append(t)
                ys_dense.append(y.copy())
            # Adjust step
            if err > 1e-30:
                h_new = h * min(5.0, max(0.2, 0.84 * (1.0 / err) ** 0.2))
            else:
                h_new = h * 5.0
            h = np.clip(h_new, self.min_step, self.max_step)
        # Interpolate to t_eval
        ts_dense = np.array(ts_dense)
        ys_dense = np.array(ys_dense)
        ys_eval = np.zeros((len(t_eval), len(y0)))
        for i, te in enumerate(t_eval):
            idx = np.searchsorted(ts_dense, te, side='right') - 1
            idx = np.clip(idx, 0, len(ts_dense) - 1)
            ys_eval[i] = ys_dense[idx]
        return t_eval, ys_eval


# ---------------------------------------------------------------------------
# 3. Simple Neural Network (for dynamics parameterization)
# ---------------------------------------------------------------------------

class MLP:
    """Multi-layer perceptron with numpy. Supports forward and backward."""

    def __init__(self, layer_sizes: List[int], activation: str = "tanh",
                 rng: Optional[np.random.Generator] = None):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        rng = rng or np.random.default_rng(42)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(rng.standard_normal((fan_in, fan_out)) * scale)
            self.biases.append(np.zeros(fan_out))
        if activation == "tanh":
            self.act, self.act_d = _tanh, _tanh_deriv
        elif activation == "relu":
            self.act, self.act_d = _relu, _relu_deriv
        else:
            self.act, self.act_d = _tanh, _tanh_deriv
        self._cache: List[Dict[str, np.ndarray]] = []

    def forward(self, x: np.ndarray, store: bool = False) -> np.ndarray:
        self._cache = []
        h = x
        for i in range(self.n_layers):
            z = h @ self.weights[i] + self.biases[i]
            if store:
                self._cache.append({"h_in": h.copy(), "z": z.copy()})
            if i < self.n_layers - 1:
                h = self.act(z)
            else:
                h = z  # linear output
        return h

    def backward(self, grad_out: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """Backprop. Returns (grad_weights, grad_biases, grad_input)."""
        gw, gb = [], []
        g = grad_out
        for i in reversed(range(self.n_layers)):
            cache = self._cache[i]
            if i < self.n_layers - 1:
                g = g * self.act_d(cache["z"])
            h_in = cache["h_in"]
            if h_in.ndim == 1:
                gw.insert(0, np.outer(h_in, g))
                gb.insert(0, g.copy())
            else:
                gw.insert(0, h_in.T @ g / h_in.shape[0])
                gb.insert(0, g.mean(axis=0))
            g = g @ self.weights[i].T
        return gw, gb, g

    def get_flat_params(self) -> np.ndarray:
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts)

    def set_flat_params(self, flat: np.ndarray) -> None:
        idx = 0
        for i in range(self.n_layers):
            n_w = self.weights[i].size
            self.weights[i] = flat[idx:idx + n_w].reshape(self.weights[i].shape)
            idx += n_w
            n_b = self.biases[i].size
            self.biases[i] = flat[idx:idx + n_b].reshape(self.biases[i].shape)
            idx += n_b

    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))


# ---------------------------------------------------------------------------
# 4. Neural ODE
# ---------------------------------------------------------------------------

class NeuralODE:
    """Neural ODE: dy/dt = f_theta(y, t) with f parameterized by MLP."""

    def __init__(self, state_dim: int, hidden_dim: int = 64, n_hidden: int = 2,
                 solver: str = "rk4", dt: float = 0.01):
        layers = [state_dim + 1] + [hidden_dim] * n_hidden + [state_dim]
        self.net = MLP(layers, activation="tanh")
        self.state_dim = state_dim
        self.solver = RK4Solver(dt=dt) if solver == "rk4" else RK45Solver()

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            inp = np.concatenate([y, [t]])
        else:
            t_col = np.full((y.shape[0], 1), t)
            inp = np.concatenate([y, t_col], axis=1)
        return self.net.forward(inp)

    def forward(self, y0: np.ndarray, t_span: Tuple[float, float],
                t_eval: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(self.dynamics, y0, t_span, t_eval)

    def predict_trajectory(self, y0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
        _, ys = self.forward(y0, (t_eval[0], t_eval[-1]), t_eval)
        return ys


# ---------------------------------------------------------------------------
# 5. Adjoint Method for Memory-Efficient Backprop
# ---------------------------------------------------------------------------

class AdjointNeuralODE:
    """Neural ODE with adjoint method for gradient computation."""

    def __init__(self, state_dim: int, hidden_dim: int = 64, n_hidden: int = 2,
                 dt: float = 0.01):
        layers = [state_dim + 1] + [hidden_dim] * n_hidden + [state_dim]
        self.net = MLP(layers, activation="tanh")
        self.state_dim = state_dim
        self.dt = dt

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        inp = np.concatenate([y, [t]])
        return self.net.forward(inp)

    def forward(self, y0: np.ndarray, t0: float, t1: float) -> np.ndarray:
        solver = RK4Solver(dt=self.dt)
        _, ys = solver.solve(self.dynamics, y0, (t0, t1))
        return ys[-1]

    def compute_gradients(self, y0: np.ndarray, t0: float, t1: float,
                          loss_grad: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Adjoint method: solve augmented ODE backward in time.
        Returns: grad_y0, grad_weights, grad_biases.
        """
        y1 = self.forward(y0, t0, t1)
        a = loss_grad.copy()  # adjoint state = dL/dy(t1)
        n_p = self.net.n_params()
        grad_params = np.zeros(n_p)
        # Backward Euler integration of adjoint ODE
        n_steps = max(int((t1 - t0) / self.dt), 1)
        dt_back = (t1 - t0) / n_steps
        y_curr = y1.copy()
        t_curr = t1
        for _ in range(n_steps):
            inp = np.concatenate([y_curr, [t_curr]])
            f_val = self.net.forward(inp, store=True)
            # df/dy via backprop
            grad_out = np.eye(self.state_dim)
            for row in range(self.state_dim):
                e = np.zeros(self.state_dim)
                e[row] = 1.0
                gw, gb, g_inp = self.net.backward(e)
                jac_row = g_inp[:self.state_dim]
                if row == 0:
                    jacobian = np.zeros((self.state_dim, self.state_dim))
                jacobian[row] = jac_row
            # Adjoint update: da/dt = -a^T df/dy => a(t-dt) = a(t) + dt * a^T @ jacobian
            a = a + dt_back * (a @ jacobian)
            # Parameter gradients: -a^T df/dtheta
            f_val2 = self.net.forward(inp, store=True)
            gw, gb, _ = self.net.backward(-a * dt_back)
            param_grad = []
            for w, b in zip(gw, gb):
                param_grad.append(w.ravel())
                param_grad.append(b.ravel())
            grad_params += np.concatenate(param_grad)
            # Step backward
            y_curr = y_curr - dt_back * f_val
            t_curr -= dt_back
        # Unpack grad_params into weight/bias lists
        gw_list, gb_list = [], []
        idx = 0
        for w, b in zip(self.net.weights, self.net.biases):
            nw = w.size
            gw_list.append(grad_params[idx:idx + nw].reshape(w.shape))
            idx += nw
            nb = b.size
            gb_list.append(grad_params[idx:idx + nb].reshape(b.shape))
            idx += nb
        return a, gw_list, gb_list


# ---------------------------------------------------------------------------
# 6. Continuous Normalizing Flow
# ---------------------------------------------------------------------------

class ContinuousNormalizingFlow:
    """Density estimation via continuous normalizing flow (ODE-based)."""

    def __init__(self, dim: int, hidden_dim: int = 32, n_hidden: int = 2,
                 dt: float = 0.05):
        self.dim = dim
        self.dt = dt
        layers = [dim + 1] + [hidden_dim] * n_hidden + [dim]
        self.net = MLP(layers, activation="tanh")

    def dynamics(self, t: float, y: np.ndarray) -> np.ndarray:
        inp = np.concatenate([y, [t]])
        return self.net.forward(inp)

    def log_density(self, x: np.ndarray, t0: float = 0.0, t1: float = 1.0,
                    n_steps: int = 50) -> float:
        """
        Compute log density via change of variables:
        log p(x) = log p(z) - integral trace(df/dy) dt
        where z = ODE_backward(x).
        """
        dt = (t1 - t0) / n_steps
        y = x.copy()
        log_det = 0.0
        for step in range(n_steps):
            t = t1 - step * dt
            inp = np.concatenate([y, [t]])
            f = self.net.forward(inp, store=True)
            # Approximate trace of Jacobian via Hutchinson estimator
            trace_est = 0.0
            eps = 1e-5
            for d in range(self.dim):
                y_p = y.copy()
                y_p[d] += eps
                inp_p = np.concatenate([y_p, [t]])
                f_p = self.net.forward(inp_p)
                trace_est += (f_p[d] - f[d]) / eps
            log_det -= trace_est * dt
            y = y - dt * f
        # log p(z) under standard normal
        log_pz = -0.5 * np.sum(y ** 2) - 0.5 * self.dim * np.log(2 * np.pi)
        return log_pz + log_det

    def sample(self, n_samples: int, t0: float = 0.0, t1: float = 1.0,
               n_steps: int = 50, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng(42)
        z = rng.standard_normal((n_samples, self.dim))
        dt = (t1 - t0) / n_steps
        for step in range(n_steps):
            t = t0 + step * dt
            for i in range(n_samples):
                inp = np.concatenate([z[i], [t]])
                z[i] = z[i] + dt * self.net.forward(inp)
        return z


# ---------------------------------------------------------------------------
# 7. Latent ODE
# ---------------------------------------------------------------------------

class LatentODE:
    """Latent ODE model: encode observations, evolve latent state, decode."""

    def __init__(self, obs_dim: int, latent_dim: int = 16,
                 hidden_dim: int = 32, dt: float = 0.05):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        # Encoder: obs -> latent mean + log_var
        self.encoder = MLP([obs_dim, hidden_dim, hidden_dim, latent_dim * 2], activation="tanh")
        # Decoder: latent -> obs
        self.decoder = MLP([latent_dim, hidden_dim, hidden_dim, obs_dim], activation="tanh")
        # ODE dynamics in latent space
        self.ode_net = MLP([latent_dim + 1, hidden_dim, hidden_dim, latent_dim], activation="tanh")
        self.dt = dt

    def encode(self, obs_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode observation sequence into latent distribution. Uses last obs."""
        last_obs = obs_sequence[-1] if obs_sequence.ndim > 1 else obs_sequence
        out = self.encoder.forward(last_obs)
        mu = out[:self.latent_dim]
        log_var = out[self.latent_dim:]
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        std = np.exp(0.5 * log_var)
        eps = rng.standard_normal(mu.shape)
        return mu + std * eps

    def latent_dynamics(self, t: float, z: np.ndarray) -> np.ndarray:
        inp = np.concatenate([z, [t]])
        return self.ode_net.forward(inp)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder.forward(z)

    def forward(self, obs_sequence: np.ndarray, t_obs: np.ndarray,
                t_pred: np.ndarray) -> Dict[str, Any]:
        mu, log_var = self.encode(obs_sequence)
        z0 = self.reparameterize(mu, log_var)
        solver = RK4Solver(dt=self.dt)
        _, z_traj = solver.solve(self.latent_dynamics, z0, (t_pred[0], t_pred[-1]), t_pred)
        predictions = np.array([self.decode(z_traj[i]) for i in range(len(t_pred))])
        kl = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var))
        return {
            "predictions": predictions,
            "latent_trajectory": z_traj,
            "mu": mu,
            "log_var": log_var,
            "kl_divergence": float(kl),
        }


# ---------------------------------------------------------------------------
# 8. Augmented Neural ODE
# ---------------------------------------------------------------------------

class AugmentedNeuralODE:
    """Augmented Neural ODE with extra dimensions for expressiveness."""

    def __init__(self, state_dim: int, aug_dim: int = 4, hidden_dim: int = 64,
                 n_hidden: int = 2, dt: float = 0.01):
        self.state_dim = state_dim
        self.aug_dim = aug_dim
        total_dim = state_dim + aug_dim
        layers = [total_dim + 1] + [hidden_dim] * n_hidden + [total_dim]
        self.net = MLP(layers, activation="tanh")
        self.dt = dt

    def dynamics(self, t: float, y_aug: np.ndarray) -> np.ndarray:
        inp = np.concatenate([y_aug, [t]])
        return self.net.forward(inp)

    def forward(self, y0: np.ndarray, t_span: Tuple[float, float],
                t_eval: Optional[np.ndarray] = None) -> np.ndarray:
        y0_aug = np.concatenate([y0, np.zeros(self.aug_dim)])
        solver = RK4Solver(dt=self.dt)
        _, ys = solver.solve(self.dynamics, y0_aug, t_span, t_eval)
        return ys[:, :self.state_dim]


# ---------------------------------------------------------------------------
# 9. Adam Optimizer
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Adam optimizer for training neural ODE parameters."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t: int = 0

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        if self.weight_decay > 0:
            grads = grads + self.weight_decay * params
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# 10. Training Loop
# ---------------------------------------------------------------------------

def train_neural_ode(node: NeuralODE, data: np.ndarray, t_data: np.ndarray,
                     n_epochs: int = 100, lr: float = 1e-3,
                     batch_size: int = 16, verbose: bool = False) -> Dict[str, List[float]]:
    """
    Train Neural ODE on time-series data.
    data: (T, D) observations at t_data times.
    """
    T, D = data.shape
    optimizer = AdamOptimizer(lr=lr)
    params = node.net.get_flat_params()
    losses = []
    eps = 1e-5
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, T - 1, batch_size):
            end = min(start + batch_size, T - 1)
            # Forward
            y0 = data[start]
            t_span = (t_data[start], t_data[end])
            t_eval = t_data[start:end + 1]
            _, pred = node.forward(y0, t_span, t_eval)
            target = data[start:end + 1]
            loss = np.mean((pred - target) ** 2)
            epoch_loss += loss
            n_batches += 1
            # Numerical gradient (for simplicity; adjoint is above for production)
            grad = np.zeros_like(params)
            for i in range(len(params)):
                params_p = params.copy()
                params_p[i] += eps
                node.net.set_flat_params(params_p)
                _, pred_p = node.forward(y0, t_span, t_eval)
                loss_p = np.mean((pred_p - target) ** 2)
                grad[i] = (loss_p - loss) / eps
            params = optimizer.step(params, grad)
            node.net.set_flat_params(params)
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")
    return {"losses": losses}


# ---------------------------------------------------------------------------
# 11. Application: Continuous-Time Volatility Model
# ---------------------------------------------------------------------------

class ContinuousVolatilityModel:
    """Neural ODE for modeling continuous-time volatility dynamics."""

    def __init__(self, hidden_dim: int = 32, dt: float = 0.005):
        # State: [log_price, log_vol]
        self.node = NeuralODE(state_dim=2, hidden_dim=hidden_dim, n_hidden=2, dt=dt)
        self.dt = dt

    def fit(self, prices: np.ndarray, times: np.ndarray, n_epochs: int = 50,
            lr: float = 1e-3) -> Dict[str, List[float]]:
        log_prices = np.log(prices + 1e-10)
        # Estimate realized vol via rolling window
        returns = np.diff(log_prices)
        window = min(20, len(returns) // 4)
        realized_vol = np.zeros(len(log_prices))
        for i in range(len(log_prices)):
            start = max(0, i - window)
            end = min(len(returns), i + 1)
            if end > start:
                realized_vol[i] = np.std(returns[start:end]) * np.sqrt(252)
            else:
                realized_vol[i] = 0.15
        data = np.column_stack([log_prices, np.log(realized_vol + 1e-10)])
        return train_neural_ode(self.node, data, times, n_epochs=n_epochs, lr=lr)

    def predict(self, initial_price: float, initial_vol: float,
                t_eval: np.ndarray) -> Dict[str, np.ndarray]:
        y0 = np.array([np.log(initial_price), np.log(initial_vol)])
        _, ys = self.node.forward(y0, (t_eval[0], t_eval[-1]), t_eval)
        return {
            "prices": np.exp(ys[:, 0]),
            "volatilities": np.exp(ys[:, 1]),
            "log_prices": ys[:, 0],
            "log_vols": ys[:, 1],
        }


# ---------------------------------------------------------------------------
# 12. Application: Irregular Time Series (unevenly spaced ticks)
# ---------------------------------------------------------------------------

class IrregularTimeSeriesModel:
    """Handle unevenly spaced tick data using Latent ODE."""

    def __init__(self, obs_dim: int, latent_dim: int = 8, hidden_dim: int = 32,
                 dt: float = 0.01):
        self.latent_ode = LatentODE(obs_dim, latent_dim, hidden_dim, dt)
        self.obs_dim = obs_dim

    def fit_and_predict(self, obs_times: np.ndarray, obs_values: np.ndarray,
                        pred_times: np.ndarray) -> Dict[str, Any]:
        """
        obs_times: (N,) irregular observation times
        obs_values: (N, D) observations
        pred_times: (M,) prediction times (can include gaps)
        """
        result = self.latent_ode.forward(obs_values, obs_times, pred_times)
        return {
            "pred_times": pred_times,
            "predictions": result["predictions"],
            "latent_trajectory": result["latent_trajectory"],
            "kl_divergence": result["kl_divergence"],
        }

    def interpolate(self, obs_times: np.ndarray, obs_values: np.ndarray,
                    target_times: np.ndarray) -> np.ndarray:
        """Interpolate to regular grid using ODE dynamics."""
        result = self.fit_and_predict(obs_times, obs_values, target_times)
        return result["predictions"]

    @staticmethod
    def simulate_irregular_ticks(n_ticks: int = 500, base_dt: float = 1.0,
                                  rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate irregular tick data (price + volume)."""
        rng = rng or np.random.default_rng(42)
        inter_arrival = rng.exponential(base_dt, n_ticks)
        times = np.cumsum(inter_arrival)
        log_price = np.cumsum(rng.standard_normal(n_ticks) * 0.01)
        prices = 100 * np.exp(log_price)
        volumes = rng.exponential(1000, n_ticks)
        obs = np.column_stack([prices, volumes])
        return times, obs
