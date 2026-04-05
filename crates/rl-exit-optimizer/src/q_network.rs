use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::action::{Action, NUM_ACTIONS};
use crate::state::{StateVector, STATE_DIM, NUM_BINS};

/// A compact key derived from discretized state features.
/// Encodes NUM_BINS^STATE_DIM possible states using a mixed-radix integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateKey(pub u64);

impl StateKey {
    /// Build a `StateKey` from a discretized bin array.
    pub fn from_bins(bins: &[usize; STATE_DIM]) -> Self {
        let mut key: u64 = 0;
        let base = NUM_BINS as u64;
        for &b in bins.iter() {
            key = key * base + b as u64;
        }
        StateKey(key)
    }

    /// Build a `StateKey` directly from a `StateVector`.
    pub fn from_state(sv: &StateVector) -> Self {
        let bins = sv.discretize();
        Self::from_bins(&bins)
    }
}

/// Tabular Q-network mapping (state, action) -> Q-value.
///
/// The Q-table stores a pair `[q_hold, q_exit]` per visited state.
/// Unvisited states return an optimistic initial value of 0.0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNetwork {
    /// Q[state_key] = [Q(s,HOLD), Q(s,EXIT)]
    table: HashMap<StateKey, [f64; NUM_ACTIONS]>,

    /// Learning rate α
    pub alpha: f64,

    /// Discount factor γ
    pub gamma: f64,

    /// Optimistic initial Q-value for unvisited states
    pub init_value: f64,

    /// Total number of Q-table updates performed
    pub update_count: u64,
}

impl QNetwork {
    pub fn new(alpha: f64, gamma: f64) -> Self {
        QNetwork {
            table: HashMap::new(),
            alpha,
            gamma,
            init_value: 0.0,
            update_count: 0,
        }
    }

    /// Retrieve Q-values for a state. Returns `[init_value; 2]` for unseen states.
    pub fn get_q(&self, key: StateKey) -> [f64; NUM_ACTIONS] {
        self.table
            .get(&key)
            .copied()
            .unwrap_or([self.init_value; NUM_ACTIONS])
    }

    /// Best (greedy) Q-value across actions at `key`.
    pub fn max_q(&self, key: StateKey) -> f64 {
        let qs = self.get_q(key);
        qs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Greedy action at `key`.
    pub fn greedy_action(&self, key: StateKey) -> Action {
        let qs = self.get_q(key);
        if qs[Action::Exit.index()] > qs[Action::Hold.index()] {
            Action::Exit
        } else {
            Action::Hold
        }
    }

    /// Bellman update:
    /// Q(s,a) ← Q(s,a) + α · (r + γ · max_a' Q(s',a') − Q(s,a))
    ///
    /// If `done` is true, the next-state value is not bootstrapped (episode ended).
    pub fn update(
        &mut self,
        state_key: StateKey,
        action: Action,
        reward: f64,
        next_key: StateKey,
        done: bool,
    ) {
        let current_q = self.get_q(state_key);
        let next_max = if done { 0.0 } else { self.max_q(next_key) };

        let target = reward + self.gamma * next_max;
        let a = action.index();
        let old_q = current_q[a];
        let new_q = old_q + self.alpha * (target - old_q);

        let entry = self
            .table
            .entry(state_key)
            .or_insert([self.init_value; NUM_ACTIONS]);
        entry[a] = new_q;

        self.update_count += 1;
    }

    /// Number of unique states in the Q-table.
    pub fn num_states(&self) -> usize {
        self.table.len()
    }

    /// Serialize the Q-table to a JSON string.
    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Deserialize a Q-table from JSON.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// Save to a file path.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from a file path.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    /// Decay the learning rate (called periodically during training).
    pub fn decay_alpha(&mut self, factor: f64) {
        self.alpha = (self.alpha * factor).max(1e-4);
    }

    /// Return an iterator over (StateKey, [q_hold, q_exit]) for inspection.
    pub fn iter(&self) -> impl Iterator<Item = (&StateKey, &[f64; NUM_ACTIONS])> {
        self.table.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TradeStateRaw;

    fn make_key(pnl: f64, bars: u32, bh_active: bool) -> StateKey {
        let raw = TradeStateRaw {
            position_pnl_pct: pnl,
            bars_held: bars,
            bh_active,
            bh_mass: 0.5,
            atr_ratio: 1.0,
            market_return_since_entry: 0.0,
            momentum_15m: 0.0,
            utc_hour: 12.0,
            drawdown_from_peak: 0.0,
            pnl_acceleration: 0.0,
        };
        StateKey::from_state(&StateVector::from_raw(&raw))
    }

    #[test]
    fn test_unvisited_state_returns_init() {
        let q = QNetwork::new(0.1, 0.95);
        let key = make_key(0.01, 5, true);
        let qs = q.get_q(key);
        assert_eq!(qs, [0.0, 0.0]);
    }

    #[test]
    fn test_single_update() {
        let mut q = QNetwork::new(0.5, 0.9);
        let s = make_key(0.0, 5, true);
        let s2 = make_key(0.02, 6, true);
        // reward=1.0, next max Q=0 (unseen), done=false
        q.update(s, Action::Exit, 1.0, s2, false);
        let qs = q.get_q(s);
        // Q(s,EXIT) = 0 + 0.5*(1.0 + 0.9*0 - 0) = 0.5
        assert!((qs[Action::Exit.index()] - 0.5).abs() < 1e-9);
        assert_eq!(qs[Action::Hold.index()], 0.0);
    }

    #[test]
    fn test_greedy_action_prefers_higher_q() {
        let mut q = QNetwork::new(0.5, 0.9);
        let s = make_key(0.05, 10, false);
        let s2 = make_key(0.06, 11, false);
        q.update(s, Action::Exit, 0.5, s2, true);
        assert_eq!(q.greedy_action(s), Action::Exit);
    }

    #[test]
    fn test_qtable_grows() {
        let mut q = QNetwork::new(0.1, 0.95);
        let keys: Vec<StateKey> = (0..5)
            .map(|i| make_key(i as f64 * 0.01, i as u32, true))
            .collect();
        for k in &keys {
            q.update(*k, Action::Hold, -0.001, *k, false);
        }
        assert_eq!(q.num_states(), keys.iter().collect::<std::collections::HashSet<_>>().len());
    }

    #[test]
    fn test_json_roundtrip() {
        let mut q = QNetwork::new(0.1, 0.95);
        let s = make_key(0.03, 15, true);
        q.update(s, Action::Hold, -0.001, s, false);
        let json = q.to_json().unwrap();
        let q2 = QNetwork::from_json(&json).unwrap();
        let qs_orig = q.get_q(s);
        let qs_loaded = q2.get_q(s);
        assert!((qs_orig[0] - qs_loaded[0]).abs() < 1e-9);
    }

    #[test]
    fn test_done_no_bootstrap() {
        let mut q = QNetwork::new(1.0, 0.9); // alpha=1 so update is full replacement
        let s = make_key(0.0, 5, true);
        let s2 = make_key(0.05, 10, false);
        // done=true: target = reward + 0 (no bootstrap)
        q.update(s, Action::Exit, 2.0, s2, true);
        let qs = q.get_q(s);
        // Q = 0 + 1.0*(2.0 + 0 - 0) = 2.0
        assert!((qs[Action::Exit.index()] - 2.0).abs() < 1e-9);
    }
}
