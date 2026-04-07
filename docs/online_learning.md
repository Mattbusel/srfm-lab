# Online Learning

## Overview

The `crates/online-learning/` Rust crate provides four families of online learning
algorithms used to adapt signal weights in real time as market conditions change.
Unlike batch retraining, every algorithm here updates its parameters on each
incoming observation with no stored history -- making them suitable for
low-latency, non-stationary financial data.

The crate is structured as a library exposed to Python via FFI or a file-based
weight handoff. The `ml/training/` directory contains the Python glue that reads
updated weights and distributes them to the signal combiner.

```
crates/online-learning/
  src/
    ftrl.rs
    pa_ii.rs
    hedge_algorithm.rs
    adaptive_learning_rate.rs
    bandit_explorer.rs
    lib.rs
  Cargo.toml
```

---

## FTRL-Proximal -- ftrl.rs

Follow The Regularized Leader with Proximal (L1 + L2) regularization. FTRL-
Proximal is the go-to algorithm for sparse, high-dimensional feature spaces such
as one-hot encoded market microstructure signals.

### Algorithm

At each step t, for feature i:

```
z_i  <- z_i + g_i - (sigma_i) * w_i
n_i  <- n_i + g_i^2

w_i  = 0                                           if |z_i| <= lambda1
     = -(1 / ((beta + sqrt(n_i)) / alpha + lambda2))
       * (z_i - sign(z_i) * lambda1)               otherwise
```

Where:

- `alpha` -- initial learning rate
- `beta` -- smoothing constant (default 1.0)
- `lambda1` -- L1 regularization coefficient (drives sparsity)
- `lambda2` -- L2 regularization coefficient (prevents large weights)
- `g_i` -- gradient for feature i on this step
- `z_i`, `n_i` -- per-feature accumulators

The per-feature adaptive learning rate means high-frequency features get smaller
updates automatically -- no global learning rate tuning is needed.

### Rust Interface

```rust
pub struct FtrlProximal {
    alpha: f64,
    beta: f64,
    lambda1: f64,
    lambda2: f64,
    z: Vec<f64>,   // per-feature z accumulator
    n: Vec<f64>,   // per-feature n accumulator
}

impl FtrlProximal {
    pub fn new(n_features: usize, alpha: f64, beta: f64,
               lambda1: f64, lambda2: f64) -> Self { /* ... */ }

    /// Predict and then update weights in one pass (online learning).
    pub fn update(&mut self, x: &[f64], grad: f64) -> f64 {
        // Returns predicted value before the update
        // ...
    }

    pub fn weights(&self) -> Vec<f64> { /* ... */ }
}
```

### When to Use

FTRL is the best choice when:

- The feature vector is sparse (most entries are zero)
- You want automatic feature selection via L1 sparsity
- Signal weights should drift slowly -- L2 keeps them bounded
- Used in `ml/training/` for live signal weight updates after each bar

---

## Passive-Aggressive II -- pa_ii.rs

A margin-based online classifier that updates weights aggressively when the
current prediction violates the margin and makes no update otherwise.

### Algorithm

PA-II variant uses a soft margin with aggressiveness parameter C:

```
loss  = max(0, 1 - y * dot(w, x))     (hinge loss)

tau   = loss / (||x||^2 + 1/(2*C))    (PA-II step size)

w     <- w + tau * y * x
```

Where `y in {-1, +1}` is the true label (e.g., regime direction).

PA-II is instantaneous -- there is no learning rate to tune and no history to
maintain. The parameter C trades off between large updates (high C) and stability
(low C).

### Rust Interface

```rust
pub struct PassiveAggressiveII {
    c: f64,
    weights: Vec<f64>,
}

impl PassiveAggressiveII {
    pub fn new(n_features: usize, c: f64) -> Self { /* ... */ }

    /// Update on a single labeled example. Returns the hinge loss.
    pub fn update(&mut self, x: &[f64], y: f64) -> f64 { /* ... */ }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.weights.iter().zip(x).map(|(w, xi)| w * xi).sum()
    }
}
```

### When to Use

PA-II excels at:

- Rapid concept drift -- regime transitions, macro events, opening gaps
- Binary classification tasks where you need zero-latency adaptation
- Situations where storing any history is too expensive
- Regime shift detection -- plugs directly into the regime classifier pipeline

---

## Hedge Algorithm -- hedge_algorithm.rs

The Hedge algorithm (also called Multiplicative Weights) maintains a probability
distribution over an ensemble of signals and achieves no-regret guarantees in the
adversarial online learning sense.

### Algorithm

At each step:

```
// Observe losses l_i for each expert i
w_i  <- w_i * exp(-eta * l_i)

// Normalize to get updated probabilities
p_i  = w_i / sum(w_j)
```

Where `eta` is the learning rate. The theoretical optimal setting is:

```
eta = sqrt(8 * ln(N) / T)
```

Where N is the number of experts and T is the time horizon.

Regret bound: After T rounds, the cumulative loss of Hedge exceeds the best
single expert's loss by at most `sqrt(T * ln(N) / 2)`.

### SignalEnsembleHedge Wrapper

`SignalEnsembleHedge` is a higher-level struct that wraps the raw Hedge algorithm
and manages signal indices, name mapping, and loss computation:

```rust
pub struct SignalEnsembleHedge {
    hedge: Hedge,
    signal_names: Vec<String>,
    eta: f64,
}

impl SignalEnsembleHedge {
    pub fn new(signal_names: Vec<String>, eta: f64) -> Self { /* ... */ }

    /// Update weights given realized losses for each signal this bar.
    pub fn update(&mut self, losses: &[f64]) { /* ... */ }

    /// Return the current blending probabilities (sum to 1.0).
    pub fn probabilities(&self) -> Vec<f64> { /* ... */ }

    /// Blend signal predictions using current weights.
    pub fn blend(&self, predictions: &[f64]) -> f64 {
        self.probabilities()
            .iter()
            .zip(predictions)
            .map(|(p, pred)| p * pred)
            .sum()
    }
}
```

### Loss Definition

For trading signals the loss for signal i on bar t is typically:

```
l_i = max(0, -direction_i * realized_return_t)
```

That is, the loss is the magnitude of a wrong-direction prediction. Correct
predictions incur zero loss.

### When to Use

Hedge is ideal for:

- Blending a fixed ensemble of diverse signals (momentum, mean-reversion, ML)
- Situations where no single signal dominates across regimes
- Provable worst-case regret bounds matter more than average-case performance

---

## Adaptive Learning Rates -- adaptive_learning_rate.rs

Four optimizer variants used as the inner update rule for FTRL, PA-II, and
gradient-based signal tuning.

### Adam

Adaptive Moment Estimation with bias correction:

```
m1  <- beta1 * m1 + (1 - beta1) * g         (first moment)
m2  <- beta2 * m2 + (1 - beta2) * g^2        (second moment)

m1_hat = m1 / (1 - beta1^t)                  (bias correction)
m2_hat = m2 / (1 - beta2^t)

w   <- w - lr * m1_hat / (sqrt(m2_hat) + eps)
```

Default hyperparameters: `beta1=0.9`, `beta2=0.999`, `eps=1e-8`.

### AdaGrad

Accumulated gradient squares as denominator -- good for sparse updates:

```
G   <- G + g^2
w   <- w - (lr / sqrt(G + eps)) * g
```

Learning rate decays monotonically per feature. Best when features arrive
at different frequencies and sparse features should get large updates early.

### RMSProp

Exponential moving average of squared gradients -- fixes AdaGrad's decaying
learning rate:

```
v   <- rho * v + (1 - rho) * g^2
w   <- w - (lr / sqrt(v + eps)) * g
```

Default: `rho=0.9`. Useful when gradients are non-stationary.

### CyclicLR

Triangular cyclic learning rate schedule. Useful for breaking out of sharp
local optima in intraday signal tuning:

```
cycle      = floor(1 + step / (2 * step_size))
x          = abs(step / step_size - 2 * cycle + 1)
lr_current = base_lr + (max_lr - base_lr) * max(0, 1 - x)
```

```rust
pub enum Optimizer {
    Adam    { lr: f64, beta1: f64, beta2: f64, eps: f64, m1: Vec<f64>, m2: Vec<f64>, t: u64 },
    AdaGrad { lr: f64, eps: f64, accum: Vec<f64> },
    RMSProp { lr: f64, rho: f64, eps: f64, v: Vec<f64> },
    CyclicLR { base_lr: f64, max_lr: f64, step_size: u64, step: u64 },
}

impl Optimizer {
    pub fn step(&mut self, weights: &mut [f64], grads: &[f64]) { /* ... */ }
}
```

---

## Bandit Explorer -- bandit_explorer.rs

Multi-armed bandit algorithms for signal selection under uncertainty. When the
system is uncertain which signal is best in the current regime, bandits trade off
exploration (trying underused signals) against exploitation (using the best known
signal).

### UCB1

Upper Confidence Bound -- select the arm with the highest optimistic estimate:

```
score_i = mu_i + sqrt(2 * ln(t) / n_i)
```

Where `mu_i` is the empirical mean reward for signal i, `t` is the total number
of steps, and `n_i` is the number of times signal i was selected. The exploration
bonus shrinks as a signal is used more, naturally balancing the tradeoff.

### Thompson Sampling -- Beta-Bernoulli

Maintains a Beta(alpha_i, beta_i) posterior over the success probability of each
signal. On each step:

```
theta_i ~ Beta(alpha_i + successes_i, beta_i + failures_i)
arm*    = argmax_i theta_i
```

After observing outcome r in {0, 1}:

```
alpha_i <- alpha_i + r        (if arm* was chosen)
beta_i  <- beta_i  + (1 - r)
```

Thompson sampling tends to outperform UCB1 empirically even though both have
O(sqrt(T log T)) expected regret.

### ContextualBandit -- Linear UCB

When contextual features are available (e.g., regime embedding, time of day,
volatility state), Linear UCB (LinUCB) uses ridge regression to model reward
as a linear function of context:

```
theta_i = (A_i)^-1 * b_i          (ridge regression solution)
score_i = x^T * theta_i + alpha * sqrt(x^T * (A_i)^-1 * x)
```

Where `x` is the context vector, `A_i = X_i^T X_i + I`, `b_i = X_i^T y_i`,
and `alpha` controls the exploration width.

```rust
pub enum BanditExplorer {
    UCB1 {
        counts: Vec<u64>,
        values: Vec<f64>,
        total: u64,
    },
    ThompsonBeta {
        alpha: Vec<f64>,
        beta:  Vec<f64>,
    },
    LinearUCB {
        n_arms:    usize,
        n_context: usize,
        alpha:     f64,
        a_inv:     Vec<Vec<f64>>,  // per-arm (A^-1) matrices
        b:         Vec<Vec<f64>>,  // per-arm b vectors
    },
}

impl BanditExplorer {
    pub fn select(&self, context: Option<&[f64]>) -> usize { /* ... */ }
    pub fn update(&mut self, arm: usize, reward: f64, context: Option<&[f64]>) { /* ... */ }
}
```

### When to Use

| Algorithm | Best scenario |
|---|---|
| UCB1 | Simple signal selection, no context available, stationary rewards |
| Thompson sampling | Non-stationary rewards, better empirical performance than UCB1 |
| Linear UCB | Regime-aware selection where a feature vector describes the current market state |

---

## Integration with Python

Updated weights flow from the Rust crate to Python via two mechanisms:

### File-Based Handoff (default)

The Rust binary writes a JSON weight file after each update cycle:

```json
{
  "signal_weights": {
    "momentum_5m": 0.312,
    "mean_rev_bbo": 0.187,
    "bh_condensate": 0.441,
    "garch_vol_break": 0.060
  },
  "updated_at": 1743897600
}
```

Python reads this file in `ml/signal_combiner.py`:

```python
import json
import os
from pathlib import Path

WEIGHTS_PATH = Path("config/online_weights.json")

class SignalCombiner:
    def __init__(self) -> None:
        self._weights: dict[str, float] = {}
        self._mtime: float = 0.0
        self._reload()

    def _reload(self) -> None:
        mtime = os.path.getmtime(WEIGHTS_PATH)
        if mtime != self._mtime:
            with open(WEIGHTS_PATH) as fh:
                data = json.load(fh)
            self._weights = data["signal_weights"]
            self._mtime = mtime

    def combine(self, signals: dict[str, float]) -> float:
        self._reload()
        total = sum(self._weights.get(k, 0.0) * v for k, v in signals.items())
        weight_sum = sum(self._weights.get(k, 0.0) for k in signals)
        return total / weight_sum if weight_sum > 0 else 0.0
```

### FFI (low-latency path)

For sub-millisecond weight updates the crate exposes a C ABI:

```rust
#[no_mangle]
pub extern "C" fn ftrl_update(handle: *mut FtrlProximal,
                               x_ptr: *const f64, x_len: usize,
                               grad: f64) -> f64 { /* ... */ }
```

Called from Python via `ctypes`:

```python
import ctypes
lib = ctypes.CDLL("target/release/libonline_learning.so")
lib.ftrl_update.restype = ctypes.c_double
```

---

## Algorithm Selection Guide

| Scenario | Recommended algorithm |
|---|---|
| Sparse, high-dimensional features | FTRL-Proximal |
| Fast concept drift -- regime breaks | Passive-Aggressive II |
| Blending a fixed signal ensemble | Hedge (SignalEnsembleHedge) |
| Signal selection, no context | Thompson sampling (BanditExplorer) |
| Regime-aware signal routing | Linear UCB (ContextualBandit) |
| Gradient-based weight tuning | Adam or RMSProp |

---

## Configuration Reference

```toml
# config/online_learning.toml

[ftrl]
alpha   = 0.1
beta    = 1.0
lambda1 = 0.001
lambda2 = 0.01

[pa_ii]
c = 1.0

[hedge]
eta = 0.05

[bandit]
type  = "thompson"   # ucb1 | thompson | linear_ucb
alpha = 0.5          # only used by linear_ucb

[adam]
lr    = 0.001
beta1 = 0.9
beta2 = 0.999
eps   = 1e-8

[output]
weights_path = "config/online_weights.json"
update_freq  = "per_bar"   # per_bar | per_minute | per_session
```
