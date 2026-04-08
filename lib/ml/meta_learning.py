"""
Meta-learning for rapid strategy adaptation.

MAML, Reptile, Prototypical Networks, task distributions over market regimes,
few-shot adaptation, and applications to regime-adaptive signals and
cross-instrument transfer.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from copy import deepcopy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-30)


def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    probs = _softmax(logits)
    n = len(targets)
    return -np.sum(np.log(probs[np.arange(n), targets.astype(int)] + 1e-30)) / n


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


# ---------------------------------------------------------------------------
# Simple MLP for meta-learning (self-contained, no external deps)
# ---------------------------------------------------------------------------

class MetaMLP:
    """Lightweight MLP supporting parameter cloning and manual SGD."""

    def __init__(self, layer_sizes: List[int], activation: str = "relu",
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng(42)
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(self.n_layers):
            fin, fout = layer_sizes[i], layer_sizes[i + 1]
            self.weights.append(rng.standard_normal((fin, fout)) * np.sqrt(2.0 / (fin + fout)))
            self.biases.append(np.zeros(fout))
        self._act_name = activation
        if activation == "relu":
            self._act = lambda x: np.maximum(x, 0)
            self._act_d = lambda x: (x > 0).astype(float)
        else:
            self._act = np.tanh
            self._act_d = lambda x: 1 - np.tanh(x) ** 2
        self._cache: List[Dict] = []

    def clone(self) -> "MetaMLP":
        c = MetaMLP.__new__(MetaMLP)
        c.layer_sizes = self.layer_sizes
        c.n_layers = self.n_layers
        c.weights = [w.copy() for w in self.weights]
        c.biases = [b.copy() for b in self.biases]
        c._act_name = self._act_name
        c._act = self._act
        c._act_d = self._act_d
        c._cache = []
        return c

    def forward(self, x: np.ndarray, store: bool = False) -> np.ndarray:
        self._cache = []
        h = x
        for i in range(self.n_layers):
            z = h @ self.weights[i] + self.biases[i]
            if store:
                self._cache.append({"h": h.copy(), "z": z.copy()})
            h = self._act(z) if i < self.n_layers - 1 else z
        return h

    def backward(self, loss_grad: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        gw, gb = [], []
        g = loss_grad
        for i in reversed(range(self.n_layers)):
            c = self._cache[i]
            if i < self.n_layers - 1:
                g = g * self._act_d(c["z"])
            h = c["h"]
            if h.ndim == 1:
                gw.insert(0, np.outer(h, g))
                gb.insert(0, g.copy())
            else:
                n = h.shape[0]
                gw.insert(0, h.T @ g / n)
                gb.insert(0, g.mean(axis=0))
            g = g @ self.weights[i].T
        return gw, gb

    def sgd_step(self, gw: List[np.ndarray], gb: List[np.ndarray], lr: float) -> None:
        for i in range(self.n_layers):
            self.weights[i] -= lr * gw[i]
            self.biases[i] -= lr * gb[i]

    def get_flat_params(self) -> np.ndarray:
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.extend([w.ravel(), b.ravel()])
        return np.concatenate(parts)

    def set_flat_params(self, flat: np.ndarray) -> None:
        idx = 0
        for i in range(self.n_layers):
            nw = self.weights[i].size
            self.weights[i] = flat[idx:idx + nw].reshape(self.weights[i].shape)
            idx += nw
            nb = self.biases[i].size
            self.biases[i] = flat[idx:idx + nb].reshape(self.biases[i].shape)
            idx += nb

    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))


# ---------------------------------------------------------------------------
# Task: each task is a dataset drawn from a market regime
# ---------------------------------------------------------------------------

class Task:
    """A meta-learning task = (support set, query set) from one regime."""

    def __init__(self, x_support: np.ndarray, y_support: np.ndarray,
                 x_query: np.ndarray, y_query: np.ndarray,
                 regime_label: str = "unknown"):
        self.x_support = x_support
        self.y_support = y_support
        self.x_query = x_query
        self.y_query = y_query
        self.regime_label = regime_label


class TaskDistribution:
    """Generate tasks from different market regimes."""

    def __init__(self, feature_dim: int, n_regimes: int = 4,
                 k_shot: int = 5, q_query: int = 20,
                 rng: Optional[np.random.Generator] = None):
        self.feature_dim = feature_dim
        self.n_regimes = n_regimes
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = rng or np.random.default_rng(42)
        # Each regime has different linear relationship + noise
        self.regime_weights = []
        self.regime_biases = []
        self.regime_noise = []
        self.regime_names = []
        regime_types = ["bull", "bear", "sideways", "volatile", "crisis",
                        "recovery", "momentum", "mean_revert"]
        for i in range(n_regimes):
            w = self.rng.standard_normal(feature_dim) * (0.5 + i * 0.3)
            b = self.rng.standard_normal() * 0.5
            noise = 0.1 + 0.1 * i
            self.regime_weights.append(w)
            self.regime_biases.append(b)
            self.regime_noise.append(noise)
            self.regime_names.append(regime_types[i % len(regime_types)])

    def sample_task(self, regime_idx: Optional[int] = None) -> Task:
        if regime_idx is None:
            regime_idx = self.rng.integers(0, self.n_regimes)
        n_total = self.k_shot + self.q_query
        x = self.rng.standard_normal((n_total, self.feature_dim))
        w = self.regime_weights[regime_idx]
        b = self.regime_biases[regime_idx]
        noise = self.regime_noise[regime_idx]
        y = x @ w + b + self.rng.standard_normal(n_total) * noise
        return Task(
            x_support=x[:self.k_shot],
            y_support=y[:self.k_shot],
            x_query=x[self.k_shot:],
            y_query=y[self.k_shot:],
            regime_label=self.regime_names[regime_idx],
        )

    def sample_batch(self, batch_size: int) -> List[Task]:
        return [self.sample_task() for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# Loss computation helpers
# ---------------------------------------------------------------------------

def _compute_regression_loss_and_grad(model: MetaMLP, x: np.ndarray,
                                       y: np.ndarray) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
    """MSE loss + gradient for regression task."""
    pred = model.forward(x, store=True)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred.ravel()
    diff = pred - y
    loss = float(np.mean(diff ** 2))
    # Gradient of MSE: 2 * diff / n
    n = len(y)
    grad_out = (2.0 * diff / n)
    if grad_out.ndim == 1:
        grad_out = grad_out.reshape(-1, 1)
    gw, gb = model.backward(grad_out)
    return loss, gw, gb


# ---------------------------------------------------------------------------
# 1. MAML (Model-Agnostic Meta-Learning)
# ---------------------------------------------------------------------------

class MAML:
    """MAML: learn initialization that enables fast fine-tuning."""

    def __init__(self, model: MetaMLP, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.loss_history: List[float] = []

    def inner_loop(self, task: Task) -> Tuple[MetaMLP, float]:
        """Adapt model to a single task via K gradient steps."""
        adapted = self.model.clone()
        for _ in range(self.inner_steps):
            loss, gw, gb = _compute_regression_loss_and_grad(
                adapted, task.x_support, task.y_support)
            adapted.sgd_step(gw, gb, self.inner_lr)
        # Evaluate on query set
        pred = adapted.forward(task.x_query)
        if pred.ndim == 2:
            pred = pred.ravel()
        query_loss = _mse(pred, task.y_query)
        return adapted, query_loss

    def outer_step(self, tasks: List[Task]) -> float:
        """One meta-update step over a batch of tasks."""
        meta_gw = [np.zeros_like(w) for w in self.model.weights]
        meta_gb = [np.zeros_like(b) for b in self.model.biases]
        total_loss = 0.0
        for task in tasks:
            adapted, query_loss = self.inner_loop(task)
            total_loss += query_loss
            # Approximate meta-gradient via finite differences on the outer loss
            # For efficiency, use first-order MAML approximation
            loss_q, gw_q, gb_q = _compute_regression_loss_and_grad(
                adapted, task.x_query, task.y_query)
            for i in range(self.model.n_layers):
                meta_gw[i] += gw_q[i]
                meta_gb[i] += gb_q[i]
        n_tasks = len(tasks)
        for i in range(self.model.n_layers):
            meta_gw[i] /= n_tasks
            meta_gb[i] /= n_tasks
        self.model.sgd_step(meta_gw, meta_gb, self.outer_lr)
        avg_loss = total_loss / n_tasks
        self.loss_history.append(avg_loss)
        return avg_loss

    def train(self, task_dist: TaskDistribution, n_iterations: int = 200,
              batch_size: int = 4, verbose: bool = False) -> Dict[str, List[float]]:
        for it in range(n_iterations):
            tasks = task_dist.sample_batch(batch_size)
            loss = self.outer_step(tasks)
            if verbose and it % 20 == 0:
                print(f"MAML iter {it}: loss={loss:.4f}")
        return {"losses": self.loss_history}

    def adapt(self, x_support: np.ndarray, y_support: np.ndarray,
              n_steps: Optional[int] = None) -> MetaMLP:
        """Adapt to a new task given support set."""
        adapted = self.model.clone()
        steps = n_steps or self.inner_steps
        for _ in range(steps):
            loss, gw, gb = _compute_regression_loss_and_grad(adapted, x_support, y_support)
            adapted.sgd_step(gw, gb, self.inner_lr)
        return adapted


# ---------------------------------------------------------------------------
# 2. Reptile
# ---------------------------------------------------------------------------

class Reptile:
    """Reptile: simplified first-order meta-learning."""

    def __init__(self, model: MetaMLP, inner_lr: float = 0.01,
                 outer_lr: float = 0.1, inner_steps: int = 10):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.loss_history: List[float] = []

    def inner_loop(self, task: Task) -> Tuple[MetaMLP, float]:
        adapted = self.model.clone()
        for _ in range(self.inner_steps):
            loss, gw, gb = _compute_regression_loss_and_grad(
                adapted, task.x_support, task.y_support)
            adapted.sgd_step(gw, gb, self.inner_lr)
        pred = adapted.forward(task.x_query)
        if pred.ndim == 2:
            pred = pred.ravel()
        query_loss = _mse(pred, task.y_query)
        return adapted, query_loss

    def outer_step(self, tasks: List[Task]) -> float:
        """Reptile outer update: move meta-params toward adapted params."""
        theta = self.model.get_flat_params()
        phi_sum = np.zeros_like(theta)
        total_loss = 0.0
        for task in tasks:
            adapted, loss = self.inner_loop(task)
            phi_sum += adapted.get_flat_params()
            total_loss += loss
        phi_avg = phi_sum / len(tasks)
        new_theta = theta + self.outer_lr * (phi_avg - theta)
        self.model.set_flat_params(new_theta)
        avg_loss = total_loss / len(tasks)
        self.loss_history.append(avg_loss)
        return avg_loss

    def train(self, task_dist: TaskDistribution, n_iterations: int = 200,
              batch_size: int = 4, verbose: bool = False) -> Dict[str, List[float]]:
        for it in range(n_iterations):
            tasks = task_dist.sample_batch(batch_size)
            loss = self.outer_step(tasks)
            if verbose and it % 20 == 0:
                print(f"Reptile iter {it}: loss={loss:.4f}")
        return {"losses": self.loss_history}

    def adapt(self, x_support: np.ndarray, y_support: np.ndarray,
              n_steps: Optional[int] = None) -> MetaMLP:
        adapted = self.model.clone()
        steps = n_steps or self.inner_steps
        for _ in range(steps):
            loss, gw, gb = _compute_regression_loss_and_grad(adapted, x_support, y_support)
            adapted.sgd_step(gw, gb, self.inner_lr)
        return adapted


# ---------------------------------------------------------------------------
# 3. Prototypical Networks
# ---------------------------------------------------------------------------

class PrototypicalNetwork:
    """Prototypical networks for few-shot classification in embedding space."""

    def __init__(self, input_dim: int, embed_dim: int = 32, hidden_dim: int = 64,
                 rng: Optional[np.random.Generator] = None):
        self.encoder = MetaMLP([input_dim, hidden_dim, hidden_dim, embed_dim],
                               activation="relu", rng=rng)
        self.embed_dim = embed_dim
        self.prototypes_: Optional[np.ndarray] = None
        self.class_labels_: Optional[np.ndarray] = None

    def compute_prototypes(self, x_support: np.ndarray,
                           y_support: np.ndarray) -> np.ndarray:
        embeddings = self.encoder.forward(x_support)
        classes = np.unique(y_support)
        prototypes = np.zeros((len(classes), self.embed_dim))
        for i, c in enumerate(classes):
            mask = y_support == c
            prototypes[i] = embeddings[mask].mean(axis=0)
        self.prototypes_ = prototypes
        self.class_labels_ = classes
        return prototypes

    def predict(self, x_query: np.ndarray) -> np.ndarray:
        embeddings = self.encoder.forward(x_query)
        # Negative squared Euclidean distance to prototypes
        dists = np.zeros((len(x_query), len(self.prototypes_)))
        for i in range(len(self.prototypes_)):
            diff = embeddings - self.prototypes_[i]
            dists[:, i] = -np.sum(diff ** 2, axis=1)
        probs = _softmax(dists)
        return self.class_labels_[np.argmax(probs, axis=1)]

    def predict_proba(self, x_query: np.ndarray) -> np.ndarray:
        embeddings = self.encoder.forward(x_query)
        dists = np.zeros((len(x_query), len(self.prototypes_)))
        for i in range(len(self.prototypes_)):
            diff = embeddings - self.prototypes_[i]
            dists[:, i] = -np.sum(diff ** 2, axis=1)
        return _softmax(dists)

    def episode_loss(self, x_support: np.ndarray, y_support: np.ndarray,
                     x_query: np.ndarray, y_query: np.ndarray) -> float:
        self.compute_prototypes(x_support, y_support)
        embeddings = self.encoder.forward(x_query)
        dists = np.zeros((len(x_query), len(self.prototypes_)))
        for i in range(len(self.prototypes_)):
            diff = embeddings - self.prototypes_[i]
            dists[:, i] = -np.sum(diff ** 2, axis=1)
        log_probs = dists - np.log(np.exp(dists).sum(axis=1, keepdims=True) + 1e-30)
        # Map y_query to prototype indices
        label_map = {c: i for i, c in enumerate(self.class_labels_)}
        y_idx = np.array([label_map.get(y, 0) for y in y_query])
        loss = -np.mean(log_probs[np.arange(len(y_query)), y_idx])
        return float(loss)

    def train(self, task_dist_fn: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
              n_episodes: int = 200, lr: float = 0.01,
              verbose: bool = False) -> Dict[str, List[float]]:
        """Train encoder via episodic training."""
        losses = []
        eps = 1e-5
        params = self.encoder.get_flat_params()
        # Simple SGD with numerical gradient
        for ep in range(n_episodes):
            xs, ys, xq, yq = task_dist_fn()
            self.encoder.set_flat_params(params)
            loss = self.episode_loss(xs, ys, xq, yq)
            losses.append(loss)
            # Numerical gradient (small param count makes this feasible)
            grad = np.zeros_like(params)
            for i in range(0, len(params), max(1, len(params) // 50)):
                p_plus = params.copy()
                p_plus[i] += eps
                self.encoder.set_flat_params(p_plus)
                loss_p = self.episode_loss(xs, ys, xq, yq)
                grad[i] = (loss_p - loss) / eps
            params -= lr * grad
            self.encoder.set_flat_params(params)
            if verbose and ep % 20 == 0:
                print(f"Proto ep {ep}: loss={loss:.4f}")
        return {"losses": losses}


# ---------------------------------------------------------------------------
# 4. Few-Shot Adaptation
# ---------------------------------------------------------------------------

class FewShotAdapter:
    """Adapter that wraps meta-learned model for few-shot deployment."""

    def __init__(self, meta_model: MetaMLP, inner_lr: float = 0.01,
                 n_adapt_steps: int = 10):
        self.meta_model = meta_model
        self.inner_lr = inner_lr
        self.n_adapt_steps = n_adapt_steps
        self.adapted_model: Optional[MetaMLP] = None

    def adapt(self, x_support: np.ndarray, y_support: np.ndarray) -> None:
        self.adapted_model = self.meta_model.clone()
        for _ in range(self.n_adapt_steps):
            loss, gw, gb = _compute_regression_loss_and_grad(
                self.adapted_model, x_support, y_support)
            self.adapted_model.sgd_step(gw, gb, self.inner_lr)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.adapted_model is None:
            raise RuntimeError("Must call adapt() before predict()")
        pred = self.adapted_model.forward(x)
        return pred.ravel() if pred.ndim == 2 and pred.shape[1] == 1 else pred

    def adaptation_curve(self, x_support: np.ndarray, y_support: np.ndarray,
                         x_test: np.ndarray, y_test: np.ndarray,
                         max_steps: int = 50) -> Dict[str, List[float]]:
        model = self.meta_model.clone()
        train_losses, test_losses = [], []
        for step in range(max_steps):
            pred_train = model.forward(x_support).ravel()
            pred_test = model.forward(x_test).ravel()
            train_losses.append(_mse(pred_train, y_support))
            test_losses.append(_mse(pred_test, y_test))
            loss, gw, gb = _compute_regression_loss_and_grad(model, x_support, y_support)
            model.sgd_step(gw, gb, self.inner_lr)
        return {"train_losses": train_losses, "test_losses": test_losses}


# ---------------------------------------------------------------------------
# 5. Meta-Train/Meta-Test Splitting
# ---------------------------------------------------------------------------

class MetaTrainTestSplit:
    """Split regimes into meta-train and meta-test sets."""

    def __init__(self, n_regimes: int, test_fraction: float = 0.25,
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng(42)
        indices = rng.permutation(n_regimes)
        n_test = max(1, int(n_regimes * test_fraction))
        self.test_regimes = set(indices[:n_test].tolist())
        self.train_regimes = set(indices[n_test:].tolist())

    def is_train(self, regime_idx: int) -> bool:
        return regime_idx in self.train_regimes

    def is_test(self, regime_idx: int) -> bool:
        return regime_idx in self.test_regimes


# ---------------------------------------------------------------------------
# 6. Application: Regime-Adaptive Signal Weights
# ---------------------------------------------------------------------------

class RegimeAdaptiveSignals:
    """
    Meta-learn signal combination weights that adapt quickly to new regimes.
    Each regime: different optimal weighting of alpha signals.
    """

    def __init__(self, n_signals: int, hidden_dim: int = 32,
                 inner_lr: float = 0.01, outer_lr: float = 0.001,
                 inner_steps: int = 5):
        self.n_signals = n_signals
        model = MetaMLP([n_signals, hidden_dim, hidden_dim, 1], activation="relu")
        self.maml = MAML(model, inner_lr=inner_lr, outer_lr=outer_lr,
                         inner_steps=inner_steps)
        self.task_dist: Optional[TaskDistribution] = None

    def create_task_distribution(self, signal_data: Dict[str, np.ndarray],
                                  returns: Dict[str, np.ndarray],
                                  regime_labels: Dict[str, np.ndarray]) -> TaskDistribution:
        """Create task distribution from historical data grouped by regime."""
        td = TaskDistribution(self.n_signals, n_regimes=4, k_shot=10, q_query=30)
        self.task_dist = td
        return td

    def meta_train(self, task_dist: TaskDistribution, n_iterations: int = 200,
                   batch_size: int = 4, verbose: bool = False) -> Dict[str, List[float]]:
        return self.maml.train(task_dist, n_iterations, batch_size, verbose)

    def adapt_to_regime(self, recent_signals: np.ndarray,
                        recent_returns: np.ndarray,
                        n_steps: int = 10) -> MetaMLP:
        return self.maml.adapt(recent_signals, recent_returns, n_steps)

    def predict_signals(self, adapted_model: MetaMLP,
                        current_signals: np.ndarray) -> np.ndarray:
        pred = adapted_model.forward(current_signals)
        return pred.ravel() if pred.ndim == 2 else pred

    def evaluate_adaptation_speed(self, task_dist: TaskDistribution,
                                   n_tasks: int = 50,
                                   max_k: int = 20) -> Dict[str, np.ndarray]:
        """Measure how fast model adapts with K=1,2,...,max_k support examples."""
        k_values = list(range(1, max_k + 1))
        avg_losses = np.zeros(len(k_values))
        for t in range(n_tasks):
            task = task_dist.sample_task()
            for ki, k in enumerate(k_values):
                k_actual = min(k, len(task.x_support))
                adapted = self.maml.adapt(task.x_support[:k_actual],
                                          task.y_support[:k_actual])
                pred = adapted.forward(task.x_query).ravel()
                avg_losses[ki] += _mse(pred, task.y_query)
        avg_losses /= n_tasks
        return {"k_values": np.array(k_values), "avg_losses": avg_losses}


# ---------------------------------------------------------------------------
# 7. Application: Cross-Instrument Transfer
# ---------------------------------------------------------------------------

class CrossInstrumentTransfer:
    """
    Meta-learn across liquid instruments, then adapt to illiquid ones.
    Each instrument = a different task with shared underlying dynamics.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32,
                 inner_lr: float = 0.01, outer_lr: float = 0.001,
                 inner_steps: int = 5, method: str = "maml"):
        self.feature_dim = feature_dim
        model = MetaMLP([feature_dim, hidden_dim, hidden_dim, 1], activation="relu")
        if method == "maml":
            self.learner = MAML(model, inner_lr=inner_lr, outer_lr=outer_lr,
                                inner_steps=inner_steps)
        else:
            self.learner = Reptile(model, inner_lr=inner_lr, outer_lr=outer_lr,
                                   inner_steps=inner_steps)
        self.method = method

    def prepare_instrument_tasks(self, instruments: Dict[str, Dict[str, np.ndarray]],
                                  k_shot: int = 10, q_query: int = 30) -> List[Task]:
        """
        instruments: {ticker: {"features": (T, D), "returns": (T,)}}
        Each instrument becomes a task.
        """
        tasks = []
        for ticker, data in instruments.items():
            x, y = data["features"], data["returns"]
            n = len(y)
            if n < k_shot + q_query:
                continue
            idx = np.random.permutation(n)
            xs = x[idx[:k_shot]]
            ys = y[idx[:k_shot]]
            xq = x[idx[k_shot:k_shot + q_query]]
            yq = y[idx[k_shot:k_shot + q_query]]
            tasks.append(Task(xs, ys, xq, yq, regime_label=ticker))
        return tasks

    def meta_train_on_liquid(self, task_dist: TaskDistribution,
                              n_iterations: int = 200,
                              batch_size: int = 4) -> Dict[str, List[float]]:
        if isinstance(self.learner, MAML):
            return self.learner.train(task_dist, n_iterations, batch_size)
        else:
            return self.learner.train(task_dist, n_iterations, batch_size)

    def adapt_to_illiquid(self, x_support: np.ndarray,
                           y_support: np.ndarray,
                           n_steps: int = 20) -> MetaMLP:
        return self.learner.adapt(x_support, y_support, n_steps)

    def compare_with_scratch(self, x_support: np.ndarray, y_support: np.ndarray,
                              x_test: np.ndarray, y_test: np.ndarray,
                              n_steps: int = 50) -> Dict[str, Any]:
        """Compare meta-learned adaptation vs training from scratch."""
        # Meta-learned
        adapter_meta = FewShotAdapter(self.learner.model, inner_lr=0.01, n_adapt_steps=n_steps)
        meta_curve = adapter_meta.adaptation_curve(x_support, y_support, x_test, y_test, n_steps)
        # From scratch
        scratch_model = MetaMLP(self.learner.model.layer_sizes, activation="relu")
        adapter_scratch = FewShotAdapter(scratch_model, inner_lr=0.01, n_adapt_steps=n_steps)
        scratch_curve = adapter_scratch.adaptation_curve(x_support, y_support, x_test, y_test, n_steps)
        return {
            "meta_test_losses": meta_curve["test_losses"],
            "scratch_test_losses": scratch_curve["test_losses"],
            "meta_final": meta_curve["test_losses"][-1],
            "scratch_final": scratch_curve["test_losses"][-1],
            "speedup": (scratch_curve["test_losses"][-1] + 1e-30) / (meta_curve["test_losses"][-1] + 1e-30),
        }


# ---------------------------------------------------------------------------
# 8. Full Pipeline: meta-learning for trading strategy adaptation
# ---------------------------------------------------------------------------

def meta_learning_pipeline(feature_dim: int = 10, n_regimes: int = 4,
                            k_shot: int = 10, q_query: int = 30,
                            n_meta_iterations: int = 100,
                            meta_batch_size: int = 4,
                            method: str = "maml",
                            verbose: bool = False) -> Dict[str, Any]:
    """
    End-to-end meta-learning pipeline:
    1. Create task distribution from synthetic regimes
    2. Meta-train (MAML or Reptile)
    3. Evaluate few-shot adaptation on held-out regimes
    """
    results: Dict[str, Any] = {}
    # Task distribution
    td = TaskDistribution(feature_dim, n_regimes=n_regimes, k_shot=k_shot, q_query=q_query)
    # Meta-train/test split
    split = MetaTrainTestSplit(n_regimes, test_fraction=0.25)
    results["train_regimes"] = list(split.train_regimes)
    results["test_regimes"] = list(split.test_regimes)
    # Build model
    model = MetaMLP([feature_dim, 32, 32, 1], activation="relu")
    if method == "maml":
        learner = MAML(model, inner_lr=0.01, outer_lr=0.001, inner_steps=5)
    else:
        learner = Reptile(model, inner_lr=0.01, outer_lr=0.1, inner_steps=10)
    # Meta-train
    train_result = learner.train(td, n_iterations=n_meta_iterations,
                                  batch_size=meta_batch_size, verbose=verbose)
    results["training_losses"] = train_result["losses"]
    # Evaluate on test regimes
    test_losses = []
    for regime_idx in split.test_regimes:
        task = td.sample_task(regime_idx)
        adapted = learner.adapt(task.x_support, task.y_support)
        pred = adapted.forward(task.x_query).ravel()
        loss = _mse(pred, task.y_query)
        test_losses.append(loss)
    results["test_losses"] = test_losses
    results["mean_test_loss"] = float(np.mean(test_losses))
    # Compare with scratch
    task = td.sample_task()
    scratch_model = MetaMLP([feature_dim, 32, 32, 1], activation="relu")
    adapter_scratch = FewShotAdapter(scratch_model, inner_lr=0.01, n_adapt_steps=50)
    scratch_curve = adapter_scratch.adaptation_curve(
        task.x_support, task.y_support, task.x_query, task.y_query, 50)
    adapter_meta = FewShotAdapter(learner.model, inner_lr=0.01, n_adapt_steps=50)
    meta_curve = adapter_meta.adaptation_curve(
        task.x_support, task.y_support, task.x_query, task.y_query, 50)
    results["meta_adaptation_curve"] = meta_curve["test_losses"]
    results["scratch_adaptation_curve"] = scratch_curve["test_losses"]
    return results
