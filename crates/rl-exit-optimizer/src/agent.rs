use rand::Rng;

use crate::action::Action;
use crate::q_network::{QNetwork, StateKey};
use crate::replay_buffer::ReplayBuffer;
use crate::state::StateVector;

/// Hyperparameters governing the agent's learning and exploration behaviour.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Starting epsilon (exploration probability).
    pub epsilon_start: f64,
    /// Final epsilon after full decay.
    pub epsilon_end: f64,
    /// Number of training steps over which epsilon decays linearly.
    pub epsilon_decay_steps: u64,
    /// Q-learning rate α.
    pub alpha: f64,
    /// Discount factor γ.
    pub gamma: f64,
    /// Mini-batch size for experience replay.
    pub batch_size: usize,
    /// Replay buffer capacity.
    pub buffer_capacity: usize,
    /// Train from replay every N environment steps.
    pub train_every: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 200_000,
            alpha: 0.1,
            gamma: 0.95,
            batch_size: 64,
            buffer_capacity: 50_000,
            train_every: 4,
        }
    }
}

/// ε-greedy tabular Q-learning agent for exit timing.
pub struct RLExitAgent {
    pub q_net: QNetwork,
    pub replay: ReplayBuffer,
    pub config: AgentConfig,

    /// Current exploration rate.
    pub epsilon: f64,

    /// Total environment steps taken (used for epsilon decay).
    pub total_steps: u64,

    /// Steps counter within the current train_every cycle.
    steps_since_train: usize,

    /// Cumulative training loss proxy (mean |TD error|).
    pub mean_td_error: f64,
    td_error_count: u64,
}

impl RLExitAgent {
    /// Construct a new agent with the given hyperparameters.
    pub fn new(config: AgentConfig) -> Self {
        let q_net = QNetwork::new(config.alpha, config.gamma);
        let replay = ReplayBuffer::new(config.buffer_capacity);
        let epsilon = config.epsilon_start;
        RLExitAgent {
            q_net,
            replay,
            config,
            epsilon,
            total_steps: 0,
            steps_since_train: 0,
            mean_td_error: 0.0,
            td_error_count: 0,
        }
    }

    /// Select an action using ε-greedy policy.
    ///
    /// With probability ε, picks a random action (exploration).
    /// Otherwise, picks the action with the highest Q-value (exploitation).
    pub fn select_action<R: Rng>(&self, state: &StateVector, rng: &mut R) -> Action {
        if rng.gen::<f64>() < self.epsilon {
            // Random action
            if rng.gen_bool(0.5) {
                Action::Hold
            } else {
                Action::Exit
            }
        } else {
            let key = StateKey::from_state(state);
            self.q_net.greedy_action(key)
        }
    }

    /// Observe a transition and optionally trigger a training step.
    /// Returns the TD error magnitude if training occurred.
    pub fn observe<R: Rng>(
        &mut self,
        state: StateVector,
        action: Action,
        reward: f64,
        next_state: StateVector,
        done: bool,
        rng: &mut R,
    ) -> Option<f64> {
        self.replay.store(state, action, reward, next_state, done);
        self.total_steps += 1;
        self.steps_since_train += 1;

        // Decay epsilon linearly
        self.update_epsilon();

        // Train every N steps
        if self.steps_since_train >= self.config.train_every
            && self.replay.ready(self.config.batch_size)
        {
            self.steps_since_train = 0;
            let td = self.train_step(rng);
            return Some(td);
        }
        None
    }

    /// Perform one mini-batch training step from the replay buffer.
    /// Returns mean absolute TD error for diagnostics.
    pub fn train_step<R: Rng>(&mut self, rng: &mut R) -> f64 {
        let batch = match self.replay.sample_owned(self.config.batch_size, rng) {
            Some(b) => b,
            None => return 0.0,
        };

        let mut total_td = 0.0;

        for exp in &batch {
            let s_key = StateKey::from_state(&exp.state);
            let s2_key = StateKey::from_state(&exp.next_state);

            // Compute TD error before update (for logging)
            let q_old = self.q_net.get_q(s_key)[exp.action.index()];
            let next_max = if exp.done {
                0.0
            } else {
                self.q_net.max_q(s2_key)
            };
            let target = exp.reward + self.q_net.gamma * next_max;
            let td_error = (target - q_old).abs();
            total_td += td_error;

            self.q_net
                .update(s_key, exp.action, exp.reward, s2_key, exp.done);
        }

        let mean_td = total_td / batch.len() as f64;

        // Exponential moving average of TD error
        let alpha_ema = 0.01;
        self.mean_td_error =
            (1.0 - alpha_ema) * self.mean_td_error + alpha_ema * mean_td;
        self.td_error_count += 1;

        mean_td
    }

    /// Update epsilon using linear decay schedule.
    fn update_epsilon(&mut self) {
        let t = self.total_steps as f64;
        let t_max = self.config.epsilon_decay_steps as f64;
        let frac = (t / t_max).min(1.0);
        self.epsilon = self.config.epsilon_start
            + frac * (self.config.epsilon_end - self.config.epsilon_start);
    }

    /// Force epsilon to a specific value (e.g., for evaluation mode).
    pub fn set_epsilon(&mut self, eps: f64) {
        self.epsilon = eps.clamp(0.0, 1.0);
    }

    /// Save the Q-table to a JSON file.
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        self.q_net.save(path)
    }

    /// Load a Q-table from a JSON file and replace the agent's network.
    pub fn load(&mut self, path: &str) -> anyhow::Result<()> {
        self.q_net = QNetwork::load(path)?;
        Ok(())
    }

    /// Return the number of unique states in the Q-table.
    pub fn qtable_size(&self) -> usize {
        self.q_net.num_states()
    }

    /// Return the current greedy action for a state (evaluation mode).
    pub fn predict(&self, state: &StateVector) -> Action {
        let key = StateKey::from_state(state);
        self.q_net.greedy_action(key)
    }

    /// Training statistics summary string.
    pub fn stats_line(&self) -> String {
        format!(
            "steps={} epsilon={:.4} qtable_states={} buffer={} mean_td={:.5}",
            self.total_steps,
            self.epsilon,
            self.q_net.num_states(),
            self.replay.len(),
            self.mean_td_error,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use crate::state::TradeStateRaw;

    fn dummy_state(pnl: f64) -> StateVector {
        StateVector::from_raw(&TradeStateRaw {
            position_pnl_pct: pnl,
            bars_held: 5,
            bh_active: true,
            bh_mass: 0.6,
            atr_ratio: 1.0,
            ..Default::default()
        })
    }

    #[test]
    fn test_epsilon_decay() {
        let mut cfg = AgentConfig::default();
        cfg.epsilon_decay_steps = 10;
        let mut agent = RLExitAgent::new(cfg);
        assert!((agent.epsilon - 1.0).abs() < 1e-9);
        let mut rng = SmallRng::seed_from_u64(0);
        // Push 10 steps
        for i in 0..10 {
            let s = dummy_state(i as f64 * 0.001);
            agent.observe(s.clone(), Action::Hold, -0.001, s, false, &mut rng);
        }
        // After 10 steps (= decay_steps), epsilon should be at or near epsilon_end
        assert!(agent.epsilon <= 0.1);
    }

    #[test]
    fn test_predict_returns_valid_action() {
        let agent = RLExitAgent::new(AgentConfig::default());
        let s = dummy_state(0.02);
        let a = agent.predict(&s);
        assert!(a == Action::Hold || a == Action::Exit);
    }

    #[test]
    fn test_observe_fills_buffer() {
        let mut agent = RLExitAgent::new(AgentConfig::default());
        let mut rng = SmallRng::seed_from_u64(1);
        for i in 0..10 {
            let s = dummy_state(i as f64 * 0.005);
            agent.observe(s.clone(), Action::Hold, -0.001, s, false, &mut rng);
        }
        assert_eq!(agent.replay.len(), 10);
    }

    #[test]
    fn test_train_step_grows_qtable() {
        let mut cfg = AgentConfig::default();
        cfg.batch_size = 4;
        let mut agent = RLExitAgent::new(cfg);
        let mut rng = SmallRng::seed_from_u64(7);
        // Fill with varied states
        for i in 0..20 {
            let s = dummy_state(i as f64 * 0.01 - 0.1);
            let ns = dummy_state(i as f64 * 0.01 - 0.09);
            agent.observe(s, Action::Exit, 0.1, ns, i == 19, &mut rng);
        }
        assert!(agent.qtable_size() > 0);
    }
}
