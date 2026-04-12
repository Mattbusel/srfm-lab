"""
training.py — Training infrastructure for TensorNet MPS/TT models.

Provides:
- TensorNetTrainer: gradient descent training for MPS/TT parameters
- Loss functions: NLL, cross-entropy, reconstruction MSE
- Riemannian SGD on TT manifold
- Adam optimizer (via optax)
- Learning rate scheduling
- Training loop with validation and early stopping
- Hyperparameter sweep utilities
- Convergence diagnostics
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tensor_net.mps import (
    MatrixProductState,
    mps_compress,
    mps_to_dense,
    mps_norm,
    mps_bond_entropies,
    mps_random,
    mps_from_dense,
)
from tensor_net.tensor_train import (
    TensorTrain,
    tt_round,
    tt_to_dense,
    tt_norm,
    tt_riemannian_grad,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_mse(
    mps: MatrixProductState,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """
    Mean squared error between MPS reconstruction and dense target tensor.
    Loss = (1/N) ||mps_to_dense(mps) - target||^2_F
    """
    approx = mps_to_dense(mps).reshape(-1)
    target_flat = target.reshape(-1).astype(jnp.float32)
    return jnp.mean((approx - target_flat) ** 2)


def reconstruction_mse_tt(
    tt: TensorTrain,
    target: jnp.ndarray,
) -> jnp.ndarray:
    """MSE loss for TensorTrain reconstruction."""
    approx = tt_to_dense(tt).reshape(-1)
    target_flat = target.reshape(-1).astype(jnp.float32)
    return jnp.mean((approx - target_flat) ** 2)


def negative_log_likelihood(
    mps: MatrixProductState,
    X: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Negative log-likelihood for Born machine:
    NLL = -mean_x log(P(x)) = -mean_x log(|<x|psi>|^2) + log Z

    Parameters
    ----------
    mps : MatrixProductState representing the probability distribution
    X : array of shape (n_samples, n_sites) with integer indices
    eps : numerical stability constant
    """
    n_sites = mps.n_sites
    tensors = mps.tensors

    # Compute amplitudes for all samples
    def amplitude(x):
        """Compute <x|psi> for integer configuration x."""
        result = tensors[0][0, x[0], :]  # (chi_1,)
        for i in range(1, n_sites):
            result = result @ tensors[i][:, x[i], :]
        return result[0]

    log_amps_sq = jnp.array([
        jnp.log(amplitude(X[s]) ** 2 + eps)
        for s in range(X.shape[0])
    ])

    # Partition function Z = <psi|psi>
    Z = jnp.real(mps_norm(mps) ** 2)
    log_Z = jnp.log(Z + eps)

    return -jnp.mean(log_amps_sq) + log_Z


def cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """
    Cross-entropy loss between logits and one-hot targets.

    Parameters
    ----------
    logits : array of shape (batch, n_classes)
    targets : array of shape (batch,) with integer class indices
    """
    log_softmax = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    n_samples = logits.shape[0]
    # Gather log-softmax at target indices
    log_probs = log_softmax[jnp.arange(n_samples), targets]
    return -jnp.mean(log_probs)


def contrastive_loss(
    mps1: MatrixProductState,
    mps2: MatrixProductState,
    label: float,
    margin: float = 1.0,
) -> jnp.ndarray:
    """
    Contrastive loss for metric learning:
    L = label * ||mps1 - mps2||^2 + (1-label) * max(0, margin - ||mps1 - mps2||)^2

    label=1: similar pair (minimize distance)
    label=0: dissimilar pair (push apart)
    """
    # Approximate distance via inner products
    from tensor_net.mps import mps_inner_product
    ip = mps_inner_product(mps1, mps2)
    n1 = jnp.real(mps_inner_product(mps1, mps1))
    n2 = jnp.real(mps_inner_product(mps2, mps2))
    dist_sq = jnp.real(n1 + n2 - 2 * ip)
    dist_sq = jnp.maximum(dist_sq, 0.0)
    dist = jnp.sqrt(dist_sq + 1e-12)

    similar_loss = label * dist_sq
    dissimilar_loss = (1 - label) * jnp.maximum(margin - dist, 0.0) ** 2
    return similar_loss + dissimilar_loss


def kl_divergence_loss(
    mps_approx: MatrixProductState,
    target_probs: jnp.ndarray,
    phys_dims: List[int],
) -> jnp.ndarray:
    """
    KL divergence D_KL(target || P_MPS) where P_MPS is Born probability.

    Parameters
    ----------
    mps_approx : MPS encoding probability distribution
    target_probs : target probability array (dense, possibly log-form)
    phys_dims : physical dimensions per site
    """
    # Get MPS probabilities
    dense = mps_to_dense(mps_approx).reshape(-1)
    mps_probs = dense ** 2
    Z = jnp.sum(mps_probs)
    mps_probs = mps_probs / (Z + 1e-12)

    target = target_probs.reshape(-1).astype(jnp.float32)
    target = jnp.maximum(target, 1e-12)
    target = target / jnp.sum(target)

    kl = jnp.sum(target * (jnp.log(target + 1e-12) - jnp.log(mps_probs + 1e-12)))
    return kl


# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------

def cosine_schedule(
    init_lr: float,
    n_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
) -> Callable[[int], float]:
    """
    Cosine decay learning rate schedule with optional linear warmup.

    Returns a callable: step → learning rate.
    """
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return init_lr * (step + 1) / (warmup_steps + 1)
        progress = (step - warmup_steps) / max(n_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        lr = min_lr + (init_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return lr

    return schedule


def exponential_decay_schedule(
    init_lr: float,
    decay_rate: float = 0.95,
    decay_steps: int = 100,
) -> optax.Schedule:
    """Exponential decay schedule compatible with optax."""
    return optax.exponential_decay(
        init_value=init_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
    )


def warmup_cosine_schedule_optax(
    init_lr: float,
    n_steps: int,
    warmup_steps: int = 50,
    min_lr: float = 1e-6,
) -> optax.Schedule:
    """Warmup + cosine decay schedule for optax."""
    warmup = optax.linear_schedule(0.0, init_lr, warmup_steps)
    cosine = optax.cosine_decay_schedule(init_lr, n_steps - warmup_steps, min_lr)
    return optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])


# ---------------------------------------------------------------------------
# Riemannian SGD on TT manifold
# ---------------------------------------------------------------------------

def riemannian_sgd_step(
    tt: TensorTrain,
    euclidean_grad: TensorTrain,
    lr: float,
    max_rank: int,
) -> TensorTrain:
    """
    Single Riemannian SGD step on the TT manifold.

    1. Project Euclidean gradient onto tangent space of TT manifold
    2. Update: tt_new = retract(tt - lr * riemannian_grad)
    3. Retraction: round to closest rank-max_rank TT

    Parameters
    ----------
    tt : current TensorTrain on the manifold
    euclidean_grad : Euclidean gradient of the loss w.r.t. tt cores
    lr : learning rate
    max_rank : TT-rank for retraction
    """
    # Step 1: Project gradient onto tangent space
    riem_grad = tt_riemannian_grad(tt, euclidean_grad)

    # Step 2: Gradient step (Euclidean update in ambient space)
    new_cores = [c - lr * g for c, g in zip(tt.cores, riem_grad.cores)]
    tt_updated = TensorTrain(new_cores, tt.shape)

    # Step 3: Retract via rounding
    tt_retracted, _ = tt_round(tt_updated, max_rank)

    return tt_retracted


def mps_riemannian_update(
    mps: MatrixProductState,
    grad_tensors: List[jnp.ndarray],
    lr: float,
    max_bond: int,
) -> MatrixProductState:
    """
    Riemannian update for MPS:
    1. Gradient step in ambient space
    2. Retraction via SVD compression
    """
    new_tensors = [t - lr * g for t, g in zip(mps.tensors, grad_tensors)]
    new_mps = MatrixProductState(new_tensors, mps.phys_dims)
    new_mps_compressed, _ = mps_compress(new_mps, max_bond)
    return new_mps_compressed


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Container for training diagnostics."""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    bond_entropies: List[jnp.ndarray] = field(default_factory=list)
    bond_dims: List[List[int]] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    step_times: List[float] = field(default_factory=list)

    def best_val_loss(self) -> float:
        if not self.val_losses:
            return float("inf")
        return min(self.val_losses)

    def best_step(self) -> int:
        if not self.val_losses:
            return 0
        return int(np.argmin(self.val_losses))

    def is_converged(self, patience: int = 20, tol: float = 1e-6) -> bool:
        if len(self.val_losses) < patience:
            return False
        recent = self.val_losses[-patience:]
        return max(recent) - min(recent) < tol


# ---------------------------------------------------------------------------
# TensorNetTrainer
# ---------------------------------------------------------------------------

class TensorNetTrainer:
    """
    Trains MPS or TT parameters via gradient descent using JAX autodiff.

    Supports:
    - Euclidean gradient descent (Adam, SGD)
    - Riemannian gradient descent on TT/MPS manifold
    - Validation, early stopping, checkpointing
    - Learning rate scheduling
    - Gradient norm clipping

    Parameters
    ----------
    model_type : 'mps' or 'tt'
    loss_fn : callable(model, batch) → scalar loss
    optimizer_name : 'adam' or 'sgd' or 'riemannian'
    lr : initial learning rate
    max_bond : maximum bond dimension
    n_steps : maximum training steps
    batch_size : batch size for stochastic training
    patience : early stopping patience
    grad_clip : gradient clipping norm (None = no clipping)
    warmup_steps : number of warmup steps for LR schedule
    checkpoint_interval : save checkpoint every N steps
    """

    def __init__(
        self,
        model_type: str = "mps",
        loss_fn: Optional[Callable] = None,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        max_bond: int = 8,
        n_steps: int = 500,
        batch_size: int = 32,
        patience: int = 50,
        grad_clip: Optional[float] = 1.0,
        warmup_steps: int = 20,
        checkpoint_interval: int = 100,
        validate_every: int = 10,
        riemannian: bool = False,
    ):
        self.model_type = model_type
        self.loss_fn = loss_fn or reconstruction_mse
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.max_bond = max_bond
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.patience = patience
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.checkpoint_interval = checkpoint_interval
        self.validate_every = validate_every
        self.riemannian = riemannian

        self.metrics_ = TrainingMetrics()
        self.best_model_ = None
        self.checkpoints_ = {}
        self._build_optimizer()

    def _build_optimizer(self):
        """Build the optax optimizer with LR schedule."""
        schedule = warmup_cosine_schedule_optax(
            self.lr, self.n_steps, self.warmup_steps
        )
        if self.optimizer_name == "adam":
            opt = optax.adam(schedule)
        elif self.optimizer_name == "sgd":
            opt = optax.sgd(schedule, momentum=0.9)
        elif self.optimizer_name == "adamw":
            opt = optax.adamw(schedule, weight_decay=1e-4)
        else:
            opt = optax.adam(schedule)

        if self.grad_clip is not None:
            opt = optax.chain(
                optax.clip_by_global_norm(self.grad_clip),
                opt,
            )

        self.optimizer_ = opt

    def fit_mps(
        self,
        mps_init: MatrixProductState,
        target: jnp.ndarray,
        val_target: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.KeyArray] = None,
    ) -> Tuple[MatrixProductState, TrainingMetrics]:
        """
        Fit MPS to a target dense tensor via gradient descent.

        Parameters
        ----------
        mps_init : initial MPS
        target : dense target tensor
        val_target : optional validation target
        key : JAX random key

        Returns
        -------
        mps_final : optimized MatrixProductState
        metrics : TrainingMetrics object
        """
        target = jnp.array(target, dtype=jnp.float32)
        tensors = list(mps_init.tensors)
        phys_dims = mps_init.phys_dims

        opt_state = self.optimizer_.init(tensors)
        metrics = TrainingMetrics()
        best_val_loss = float("inf")
        best_tensors = [jnp.array(t) for t in tensors]
        steps_no_improve = 0

        def loss_fn_tensors(tensors):
            mps_temp = MatrixProductState(tensors, phys_dims)
            return reconstruction_mse(mps_temp, target)

        def val_loss_fn_tensors(tensors):
            mps_temp = MatrixProductState(tensors, phys_dims)
            return reconstruction_mse(mps_temp, val_target)

        for step in range(self.n_steps):
            t_start = time.time()

            # Compute loss and gradients
            loss_val, grads = jax.value_and_grad(loss_fn_tensors)(tensors)

            # Optimizer update
            updates, opt_state = self.optimizer_.update(grads, opt_state, tensors)
            tensors = optax.apply_updates(tensors, updates)

            # Riemannian retraction: compress back to max_bond
            if self.riemannian and step % 10 == 0:
                mps_temp = MatrixProductState(tensors, phys_dims)
                mps_comp, _ = mps_compress(mps_temp, self.max_bond)
                tensors = list(mps_comp.tensors)

            t_end = time.time()

            # Compute gradient norm
            grad_norm = float(jnp.sqrt(sum(
                jnp.sum(g ** 2) for g in grads
            )))

            metrics.train_losses.append(float(loss_val))
            metrics.gradient_norms.append(grad_norm)
            metrics.step_times.append(t_end - t_start)

            # Validation
            if step % self.validate_every == 0:
                if val_target is not None:
                    val_loss = float(val_loss_fn_tensors(tensors))
                else:
                    val_loss = float(loss_val)
                metrics.val_losses.append(val_loss)

                # Bond entropies
                mps_temp = MatrixProductState(tensors, phys_dims)
                try:
                    entropies = mps_bond_entropies(mps_temp)
                    metrics.bond_entropies.append(entropies)
                    metrics.bond_dims.append(mps_temp.bond_dims)
                except Exception:
                    pass

                # Early stopping
                if val_loss < best_val_loss - 1e-7:
                    best_val_loss = val_loss
                    best_tensors = [jnp.array(t) for t in tensors]
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1

                if steps_no_improve >= self.patience:
                    break

            # Checkpoint
            if step % self.checkpoint_interval == 0:
                self.checkpoints_[step] = [jnp.array(t) for t in tensors]

        self.metrics_ = metrics
        self.best_model_ = MatrixProductState(best_tensors, phys_dims)
        return self.best_model_, metrics

    def fit_tt(
        self,
        tt_init: TensorTrain,
        target: jnp.ndarray,
        val_target: Optional[jnp.ndarray] = None,
    ) -> Tuple[TensorTrain, TrainingMetrics]:
        """
        Fit TensorTrain to a target dense tensor via gradient descent.
        """
        target = jnp.array(target, dtype=jnp.float32)
        cores = list(tt_init.cores)
        shape = tt_init.shape

        opt_state = self.optimizer_.init(cores)
        metrics = TrainingMetrics()
        best_val_loss = float("inf")
        best_cores = [jnp.array(c) for c in cores]
        steps_no_improve = 0

        def loss_fn_cores(cores):
            tt_temp = TensorTrain(cores, shape)
            return reconstruction_mse_tt(tt_temp, target)

        for step in range(self.n_steps):
            t_start = time.time()

            loss_val, grads = jax.value_and_grad(loss_fn_cores)(cores)

            if self.riemannian:
                # Riemannian gradient
                grad_tt = TensorTrain(grads, shape)
                curr_tt = TensorTrain(cores, shape)
                riem_grad = tt_riemannian_grad(curr_tt, grad_tt)
                grads = riem_grad.cores

            updates, opt_state = self.optimizer_.update(grads, opt_state, cores)
            cores = optax.apply_updates(cores, updates)

            # Retract to manifold
            if self.riemannian and step % 5 == 0:
                tt_temp = TensorTrain(cores, shape)
                tt_rounded, _ = tt_round(tt_temp, self.max_bond)
                cores = list(tt_rounded.cores)

            t_end = time.time()

            grad_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads)))
            metrics.train_losses.append(float(loss_val))
            metrics.gradient_norms.append(grad_norm)
            metrics.step_times.append(t_end - t_start)

            if step % self.validate_every == 0:
                val_loss = float(loss_fn_cores(cores)) if val_target is None else \
                    float(reconstruction_mse_tt(TensorTrain(cores, shape), val_target))
                metrics.val_losses.append(val_loss)

                if val_loss < best_val_loss - 1e-7:
                    best_val_loss = val_loss
                    best_cores = [jnp.array(c) for c in cores]
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1

                if steps_no_improve >= self.patience:
                    break

            if step % self.checkpoint_interval == 0:
                self.checkpoints_[step] = [jnp.array(c) for c in cores]

        self.metrics_ = metrics
        self.best_model_ = TensorTrain(best_cores, shape)
        return self.best_model_, metrics

    def fit_generative(
        self,
        mps_init: MatrixProductState,
        X_train: jnp.ndarray,
        X_val: Optional[jnp.ndarray] = None,
        key: Optional[jax.random.KeyArray] = None,
    ) -> Tuple[MatrixProductState, TrainingMetrics]:
        """
        Train MPS Born Machine on discrete data.

        Parameters
        ----------
        mps_init : initial MPS
        X_train : discrete training data, shape (N, n_sites), values in [0, d)
        X_val : optional validation data
        key : JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        X_train = jnp.array(X_train, dtype=jnp.int32)
        N = X_train.shape[0]
        tensors = list(mps_init.tensors)
        phys_dims = mps_init.phys_dims

        opt_state = self.optimizer_.init(tensors)
        metrics = TrainingMetrics()
        best_val_loss = float("inf")
        best_tensors = [jnp.array(t) for t in tensors]
        steps_no_improve = 0

        for step in range(self.n_steps):
            t_start = time.time()

            # Random mini-batch
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (self.batch_size,), 0, N)
            X_batch = X_train[idx]

            loss_val, grads = jax.value_and_grad(
                lambda t: negative_log_likelihood(MatrixProductState(t, phys_dims), X_batch)
            )(tensors)

            updates, opt_state = self.optimizer_.update(grads, opt_state, tensors)
            tensors = optax.apply_updates(tensors, updates)

            t_end = time.time()
            metrics.train_losses.append(float(loss_val))
            metrics.step_times.append(t_end - t_start)

            if step % self.validate_every == 0:
                if X_val is not None:
                    val_loss = float(negative_log_likelihood(
                        MatrixProductState(tensors, phys_dims),
                        jnp.array(X_val, dtype=jnp.int32)
                    ))
                else:
                    val_loss = float(loss_val)
                metrics.val_losses.append(val_loss)

                if val_loss < best_val_loss - 1e-7:
                    best_val_loss = val_loss
                    best_tensors = [jnp.array(t) for t in tensors]
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1

                if steps_no_improve >= self.patience:
                    break

        self.metrics_ = metrics
        self.best_model_ = MatrixProductState(best_tensors, phys_dims)
        return self.best_model_, metrics


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """Configuration for a hyperparameter sweep."""
    bond_dims: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3, 5e-3, 1e-2])
    n_steps: int = 200
    n_trials: int = 3
    metric: str = "val_loss"


def hyperparameter_sweep(
    target: jnp.ndarray,
    phys_dims: List[int],
    config: SweepConfig,
    val_target: Optional[jnp.ndarray] = None,
    key: Optional[jax.random.KeyArray] = None,
) -> Dict:
    """
    Grid search over bond dimension and learning rate.

    Returns
    -------
    dict with:
    - results: list of result dicts
    - best_config: (bond_dim, lr) with lowest validation loss
    - best_mps: best MatrixProductState
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    target = jnp.array(target, dtype=jnp.float32)
    results = []
    best_loss = float("inf")
    best_config = None
    best_mps = None

    for bond_dim in config.bond_dims:
        for lr in config.learning_rates:
            trial_losses = []

            for trial in range(config.n_trials):
                key, subkey = jax.random.split(key)
                # Initialize random MPS
                mps_init = mps_random(
                    len(phys_dims), phys_dims, bond_dim, subkey
                )

                trainer = TensorNetTrainer(
                    model_type="mps",
                    lr=lr,
                    max_bond=bond_dim,
                    n_steps=config.n_steps,
                    patience=30,
                    validate_every=5,
                )
                mps_final, metrics = trainer.fit_mps(
                    mps_init, target, val_target
                )

                val_loss = metrics.best_val_loss()
                trial_losses.append(val_loss)

            mean_loss = float(np.mean(trial_losses))
            std_loss = float(np.std(trial_losses))

            result = {
                "bond_dim": bond_dim,
                "lr": lr,
                "mean_val_loss": mean_loss,
                "std_val_loss": std_loss,
                "trial_losses": trial_losses,
            }
            results.append(result)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_config = (bond_dim, lr)
                best_mps = mps_final

    return {
        "results": results,
        "best_config": best_config,
        "best_mps": best_mps,
        "best_val_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def compute_convergence_diagnostics(
    metrics: TrainingMetrics,
    mps: MatrixProductState,
) -> Dict:
    """
    Compute convergence diagnostics for a training run.

    Returns
    -------
    dict with:
    - converged: bool
    - final_train_loss: float
    - final_val_loss: float
    - best_step: int
    - loss_plateau: bool (true if loss didn't decrease in last 20%)
    - gradient_explosion: bool (true if any gradient norm > 100)
    - max_bond_used: int
    - avg_entropy: float (mean bond entropy)
    """
    train_losses = metrics.train_losses
    val_losses = metrics.val_losses
    grad_norms = metrics.gradient_norms

    final_train = train_losses[-1] if train_losses else float("inf")
    final_val = val_losses[-1] if val_losses else float("inf")
    best_step = metrics.best_step()

    # Check plateau: last 20% of training
    n = len(train_losses)
    if n > 10:
        recent = train_losses[int(0.8 * n):]
        plateau = (max(recent) - min(recent)) < 1e-6
    else:
        plateau = False

    # Gradient explosion
    grad_explosion = any(g > 100 for g in grad_norms) if grad_norms else False

    # Bond dimensions
    max_bond_used = mps.max_bond

    # Average entropy
    try:
        entropies = mps_bond_entropies(mps)
        avg_entropy = float(jnp.mean(entropies))
    except Exception:
        avg_entropy = 0.0

    converged = (
        len(val_losses) > 5 and
        not grad_explosion and
        final_val < float("inf")
    )

    return {
        "converged": converged,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "best_step": best_step,
        "n_steps_taken": n,
        "loss_plateau": plateau,
        "gradient_explosion": grad_explosion,
        "max_bond_used": max_bond_used,
        "avg_entropy": avg_entropy,
    }


def print_training_summary(
    metrics: TrainingMetrics,
    diagnostics: Dict,
):
    """Print a formatted training summary."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Steps taken:       {diagnostics['n_steps_taken']}")
    print(f"  Best step:         {diagnostics['best_step']}")
    print(f"  Final train loss:  {diagnostics['final_train_loss']:.6f}")
    print(f"  Final val loss:    {diagnostics['final_val_loss']:.6f}")
    print(f"  Max bond used:     {diagnostics['max_bond_used']}")
    print(f"  Avg bond entropy:  {diagnostics['avg_entropy']:.4f}")
    print(f"  Converged:         {diagnostics['converged']}")
    print(f"  Loss plateau:      {diagnostics['loss_plateau']}")
    print(f"  Grad explosion:    {diagnostics['gradient_explosion']}")
    if metrics.step_times:
        avg_time = np.mean(metrics.step_times)
        print(f"  Avg step time:     {avg_time*1000:.1f} ms")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Batch training utilities
# ---------------------------------------------------------------------------

def create_batches(
    X: jnp.ndarray,
    batch_size: int,
    key: jax.random.KeyArray,
    shuffle: bool = True,
) -> List[jnp.ndarray]:
    """Split data into batches, optionally shuffled."""
    N = X.shape[0]
    if shuffle:
        idx = jax.random.permutation(key, N)
        X = X[idx]

    batches = []
    for start in range(0, N, batch_size):
        batches.append(X[start:start + batch_size])
    return batches


def train_val_split(
    X: jnp.ndarray,
    val_frac: float = 0.2,
    key: Optional[jax.random.KeyArray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split data into train and validation sets."""
    N = X.shape[0]
    n_val = max(1, int(N * val_frac))
    n_train = N - n_val

    if key is not None:
        idx = jax.random.permutation(key, N)
        X = X[idx]

    return X[:n_train], X[n_train:]
