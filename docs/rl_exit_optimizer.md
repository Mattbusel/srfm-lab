# RL Exit Optimizer

## Overview

The RL Exit Optimizer provides a reinforcement learning-based exit policy for open
positions. It uses a Q-table stored at `config/rl_exit_qtable.json` to decide, on
every bar, whether to hold or exit a position. The state space is defined by
5 features each binned into 5 quantile buckets, yielding 5^5 = 3,125 possible
states. The Q-table is trained offline by the Double DQN trainer in the Rust crate
`crates/rl-exit-optimizer/` and hot-reloaded into Python at runtime via
`lib/rl_exit_policy.py`.

Design goals:

- Sub-microsecond inference -- Q-table lookup is O(1) dict access, ~100 ns
- Offline training loop -- live transitions are collected, then a Rust trainer
  updates the Q-table in the background without interrupting execution
- Stateless lookup -- no recurrent state is needed; all context is captured in
  the 5 binned features

---

## State Representation

Each state is a tuple of 5 integer bin indices, one per feature. Bins are computed
from quantile boundaries fit on recent history, so they adapt to regime changes.

| Feature | Description |
|---|---|
| `unrealized_pnl_bps` | Current mark-to-market P&L in basis points |
| `bars_held` | Number of bars the position has been open |
| `bh_mass` | Bose-Hubbard condensate mass proxy -- measures order-book clustering |
| `garch_vol` | GARCH(1,1) conditional volatility estimate |
| `spread_bps` | Current bid-ask spread in basis points |

Each feature is mapped to a bin index in `{0, 1, 2, 3, 4}` using its quantile
boundaries. Bin 0 is the lowest quantile bucket; bin 4 is the highest.

```python
# Simplified binning example
import numpy as np

def compute_bin(value: float, quantile_edges: list[float]) -> int:
    """Map a scalar value to a 0-based bin index using precomputed edges."""
    return int(np.searchsorted(quantile_edges, value, side="right").clip(0, 4))
```

The 5-tuple is serialized as a comma-separated string to form the Q-table key:

```
state = (2, 3, 1, 4, 0)  ->  key = "2,3,1,4,0"
```

---

## Q-Table Format

The Q-table is a flat JSON object mapping state keys to float Q-values for each
action.

```json
{
  "0,0,0,0,0": [0.012, -0.034],
  "0,0,0,0,1": [0.005, 0.091],
  "2,3,1,4,0": [-0.018, 0.143],
  "4,4,4,4,4": [0.002, 0.187]
}
```

Layout:

- Key -- comma-separated bin string, e.g. `"2,3,1,4,0"`
- Value -- two-element array `[Q(hold), Q(exit)]`

States not present in the JSON fall back to `[0.0, 0.0]`. The Rust trainer only
writes entries where the Q-values have been updated at least once, keeping the
file sparse.

---

## Actions

| Action | Integer | Meaning |
|---|---|---|
| Hold | 0 | Keep the position open; do nothing this bar |
| Exit | 1 | Submit a market exit order immediately |

The policy always picks `argmax Q(state, *)`. No epsilon is applied at inference
time -- the epsilon-greedy strategy is used only during training.

---

## Python -- RLExitPolicy

`lib/rl_exit_policy.py` wraps the Q-table and exposes a single `should_exit()`
method called by the live trader on every bar for each open position.

```python
# lib/rl_exit_policy.py  (abbreviated)
import json
import os
from pathlib import Path
from typing import Optional

QTABLE_PATH = Path("config/rl_exit_qtable.json")
N_BINS = 5

class RLExitPolicy:
    """
    Q-table-based exit policy.

    Reloads the Q-table from disk whenever the file modification time changes,
    enabling hot-reload after an offline training run without restarting the
    trader process.
    """

    def __init__(self, qtable_path: Path = QTABLE_PATH) -> None:
        self._path = qtable_path
        self._qtable: dict[str, list[float]] = {}
        self._mtime: float = 0.0
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        mtime = os.path.getmtime(self._path)
        if mtime != self._mtime:
            with open(self._path) as fh:
                self._qtable = json.load(fh)
            self._mtime = mtime

    def _state_key(
        self,
        unrealized_pnl_bps: float,
        bars_held: int,
        bh_mass: float,
        garch_vol: float,
        spread_bps: float,
        edges: dict[str, list[float]],
    ) -> str:
        import numpy as np

        def _bin(val: float, feature: str) -> int:
            return int(
                np.searchsorted(edges[feature], val, side="right").clip(0, N_BINS - 1)
            )

        bins = (
            _bin(unrealized_pnl_bps, "unrealized_pnl_bps"),
            _bin(bars_held,          "bars_held"),
            _bin(bh_mass,            "bh_mass"),
            _bin(garch_vol,          "garch_vol"),
            _bin(spread_bps,         "spread_bps"),
        )
        return ",".join(map(str, bins))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_exit(
        self,
        unrealized_pnl_bps: float,
        bars_held: int,
        bh_mass: float,
        garch_vol: float,
        spread_bps: float,
        edges: dict[str, list[float]],
    ) -> bool:
        """Return True if the policy recommends exiting the position."""
        self._load()  # no-op unless file changed
        key = self._state_key(
            unrealized_pnl_bps, bars_held, bh_mass, garch_vol, spread_bps, edges
        )
        q = self._qtable.get(key, [0.0, 0.0])
        return q[1] > q[0]  # exit wins if Q(exit) > Q(hold)
```

The `edges` dict is built once per session from a rolling window of recent
observations and passed in from the live trader; it is not stored inside the policy
object so it can be refreshed independently.

---

## Rust Crate -- crates/rl-exit-optimizer/

The Rust crate handles the compute-intensive parts of the training loop. It is
invoked as a standalone binary between live trading sessions (or in a background
thread). Source files:

```
crates/rl-exit-optimizer/
  src/
    experience_replay.rs
    dqn_trainer.rs
    reward_shaping.rs
    main.rs
  Cargo.toml
```

### experience_replay.rs

Implements a `PrioritizedReplayBuffer` using a sum-tree data structure for
O(log N) priority updates.

Key features:

- Importance sampling weights -- `w_i = (1 / (N * P(i)))^beta` to correct for
  non-uniform sampling bias
- N-step returns -- transitions store the discounted sum of rewards over N future
  steps before bootstrapping from the target network
- Beta annealing -- beta starts at `beta_start` (e.g. 0.4) and linearly increases
  to 1.0 over training so the IS correction strengthens as the buffer fills

```rust
// Simplified interface
pub struct PrioritizedReplayBuffer {
    capacity: usize,
    alpha: f32,       // priority exponent, typically 0.6
    beta: f32,        // IS exponent, annealed toward 1.0
    beta_increment: f32,
    // ... sum-tree internals ...
}

impl PrioritizedReplayBuffer {
    pub fn push(&mut self, transition: Transition, priority: f32) { /* ... */ }

    pub fn sample(&mut self, batch_size: usize) -> (Vec<Transition>, Vec<f32>, Vec<usize>) {
        // returns (transitions, is_weights, tree_indices)
        // ...
    }

    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) { /* ... */ }
}
```

### dqn_trainer.rs

Implements Double DQN with separate online and target networks.

Key parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `tau` | 0.005 | Polyak averaging coefficient for target network soft update |
| `gamma` | 0.99 | Discount factor |
| `lr` | 1e-4 | Adam learning rate for online network |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Final exploration rate |
| `epsilon_decay` | 5000 | Steps over which epsilon decays exponentially |
| `target_update_freq` | 100 | Hard update frequency (fallback if Polyak disabled) |

Double DQN update rule -- action selection uses the online network, value
estimation uses the target network:

```
a* = argmax_a Q_online(s', a)
y  = r + gamma * Q_target(s', a*)
loss = (y - Q_online(s, a))^2
```

After training, the Q-table is serialized to JSON:

```rust
pub fn export_qtable(&self, path: &Path) -> anyhow::Result<()> {
    let mut map = serde_json::Map::new();
    for state_idx in 0..N_STATES {
        let key = idx_to_key(state_idx);          // e.g. "2,3,1,4,0"
        let q_vals = self.online_net.forward(state_idx);
        map.insert(key, json!([q_vals[0], q_vals[1]]));
    }
    let json = serde_json::Value::Object(map);
    std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}
```

### reward_shaping.rs

Computes the shaped reward for each transition before it is inserted into the
replay buffer.

Reward formula:

```
r_shaped = r_pnl - lambda_time * bars_held - lambda_spread * spread_bps
```

Where:

- `r_pnl` -- realized or mark-to-market P&L in basis points (positive = profit)
- `lambda_time` -- time penalty per bar, penalizes tying up capital (default 0.5 bps)
- `lambda_spread` -- spread cost multiplier, penalizes wide-spread exits (default 0.3)

Generalized Advantage Estimation (GAE) is also available for policy gradient
variants. It uses the lambda-weighted sum of TD residuals:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = sum_{k=0}^{T} (gamma * lambda)^k * delta_{t+k}
```

GAE is not used in the default Double DQN path but is exposed for future actor-
critic extensions.

---

## Training Workflow

```
1. Live trader collects transitions:
      (state, action, reward, next_state, done)
   and appends them to data/rl_transitions.jsonl

2. After the session (or on a schedule), invoke the Rust trainer:
      cargo run -p rl-exit-optimizer -- \
        --transitions data/rl_transitions.jsonl \
        --qtable config/rl_exit_qtable.json \
        --epochs 50

3. Trainer reads transitions, fills the replay buffer, runs Double DQN
   updates, and writes the new Q-table to config/rl_exit_qtable.json.

4. Python RLExitPolicy detects the changed mtime on next bar and reloads
   the Q-table transparently -- no process restart needed.
```

File handoff format (`data/rl_transitions.jsonl`):

```json
{"s": [2,3,1,4,0], "a": 0, "r": 1.2, "s_next": [2,3,2,4,0], "done": false}
{"s": [1,2,0,3,1], "a": 1, "r": 8.7, "s_next": null,          "done": true}
```

---

## Latency Profile

| Operation | Cost |
|---|---|
| Q-table dict lookup (Python) | ~100 ns |
| Feature binning (5 features, numpy) | ~2 us |
| File mtime check (os.path.getmtime) | ~500 ns |
| Full should_exit() call (no reload) | ~3 us |
| Q-table reload from disk (~3125 entries) | ~1 ms |

The reload path is taken at most once per training cycle, which occurs between
sessions. During live trading the mtime check is a no-op and the hot path stays
under 5 us end to end.

---

## Integration with the Live Trader

The live trader calls `should_exit()` on every incoming bar for each open position:

```python
# Inside the main bar handler
for position in open_positions:
    features = compute_features(position, market_snapshot)
    if rl_exit_policy.should_exit(**features, edges=quantile_edges):
        order_manager.submit_exit(position.symbol, reason="rl_exit")
```

The policy object is instantiated once at startup and shared across all positions.
Thread safety is not a concern in the single-threaded async event loop, but if
the trader is multi-threaded the Q-table dict should be replaced under a lock
during hot-reload.

---

## Configuration Reference

```toml
# config/rl_exit_optimizer.toml

[qtable]
path = "config/rl_exit_qtable.json"

[features]
n_bins = 5

[trainer]
epochs          = 50
batch_size      = 256
gamma           = 0.99
tau             = 0.005
lr              = 0.0001
epsilon_start   = 1.0
epsilon_end     = 0.01
epsilon_decay   = 5000
n_step          = 3
buffer_capacity = 100000
alpha           = 0.6
beta_start      = 0.4

[reward]
lambda_time   = 0.5
lambda_spread = 0.3
```
