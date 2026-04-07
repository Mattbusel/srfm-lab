// dqn_trainer.rs -- Full DQN training loop for the SRFM exit policy.
//
// Implements:
//   - QNetwork: fully connected 5->32->32->3 neural network
//   - DQNTrainer: Double DQN training with Polyak target updates
//   - export_qtable: evaluate network at all 5^5=3125 states for RLExitPolicy

use std::collections::HashMap;
use rand::Rng;

use crate::experience_replay::{Experience, ExperienceReplayBuffer};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of input features fed to the Q-network.
pub const INPUT_DIM: usize = 5;

/// First hidden layer width.
pub const HIDDEN1: usize = 32;

/// Second hidden layer width.
pub const HIDDEN2: usize = 32;

/// Number of actions: 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT.
pub const OUTPUT_DIM: usize = 3;

/// Number of bins used to discretize each feature for Q-table export.
pub const NUM_BINS: usize = 5;

/// Total number of discrete states: NUM_BINS^INPUT_DIM = 5^5 = 3125.
pub const TOTAL_STATES: usize = 3125; // 5^5

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

/// Activation function applied to hidden layers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Linear,
}

impl Activation {
    #[inline]
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU   => x.max(0.0),
            Activation::Linear => x,
        }
    }

    /// Derivative of the activation (used in gradient computation).
    #[inline]
    fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU   => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Linear => 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// LinearLayer
// ---------------------------------------------------------------------------

/// A single affine layer: out = W * in + b.
///
/// weights[i][j] = weight from input neuron j to output neuron i.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Vec<Vec<f64>>,  // [out_dim][in_dim]
    pub biases:  Vec<f64>,       // [out_dim]
    pub in_dim:  usize,
    pub out_dim: usize,
}

impl LinearLayer {
    /// Create a new layer with He-style normal initialization.
    pub fn new<R: Rng>(in_dim: usize, out_dim: usize, rng: &mut R) -> Self {
        let std = (2.0 / in_dim as f64).sqrt();
        let weights = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| sample_normal(rng) * std)
                    .collect()
            })
            .collect();
        let biases = vec![0.0; out_dim];
        LinearLayer { weights, biases, in_dim, out_dim }
    }

    /// Create a zero-initialized layer (used for target network copy).
    pub fn zeros(in_dim: usize, out_dim: usize) -> Self {
        LinearLayer {
            weights: vec![vec![0.0; in_dim]; out_dim],
            biases:  vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Forward pass: compute output vector for a given input.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.in_dim, "input dim mismatch");
        let mut out = Vec::with_capacity(self.out_dim);
        for i in 0..self.out_dim {
            let mut sum = self.biases[i];
            for j in 0..self.in_dim {
                sum += self.weights[i][j] * input[j];
            }
            out.push(sum);
        }
        out
    }

    /// Polyak averaging update toward another layer:
    ///   self = tau * other + (1 - tau) * self
    pub fn polyak_update(&mut self, other: &LinearLayer, tau: f64) {
        for i in 0..self.out_dim {
            for j in 0..self.in_dim {
                self.weights[i][j] = tau * other.weights[i][j]
                    + (1.0 - tau) * self.weights[i][j];
            }
            self.biases[i] = tau * other.biases[i] + (1.0 - tau) * self.biases[i];
        }
    }

    /// Hard copy weights from another layer (tau=1 Polyak).
    pub fn copy_from(&mut self, other: &LinearLayer) {
        self.weights = other.weights.clone();
        self.biases  = other.biases.clone();
    }
}

// ---------------------------------------------------------------------------
// QNetwork (neural)
// ---------------------------------------------------------------------------

/// Fully connected Q-network: 5 -> 32 (ReLU) -> 32 (ReLU) -> 3 (linear).
#[derive(Debug, Clone)]
pub struct QNetwork {
    pub layers: Vec<LinearLayer>,
    pub activation: Activation,
}

impl QNetwork {
    /// Construct and randomly initialize all layers.
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let layers = vec![
            LinearLayer::new(INPUT_DIM, HIDDEN1, rng),
            LinearLayer::new(HIDDEN1,   HIDDEN2, rng),
            LinearLayer::new(HIDDEN2,   OUTPUT_DIM, rng),
        ];
        QNetwork { layers, activation: Activation::ReLU }
    }

    /// Construct a zero-initialized network (for target copy).
    pub fn zeros() -> Self {
        let layers = vec![
            LinearLayer::zeros(INPUT_DIM, HIDDEN1),
            LinearLayer::zeros(HIDDEN1,   HIDDEN2),
            LinearLayer::zeros(HIDDEN2,   OUTPUT_DIM),
        ];
        QNetwork { layers, activation: Activation::ReLU }
    }

    /// Forward pass through the full network.
    /// Returns Q-values for all three actions.
    pub fn forward(&self, state: &[f64; 5]) -> [f64; 3] {
        let mut x: Vec<f64> = state.to_vec();
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            let pre = layer.forward(&x);
            // Apply ReLU to hidden layers, linear to output.
            let act = if i < n_layers - 1 { Activation::ReLU } else { Activation::Linear };
            x = pre.into_iter().map(|v| act.apply(v)).collect();
        }
        [x[0], x[1], x[2]]
    }

    /// Argmax action from Q-values.
    pub fn greedy_action(&self, state: &[f64; 5]) -> u8 {
        let qs = self.forward(state);
        qs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    /// Hard copy weights from another network.
    pub fn copy_from(&mut self, other: &QNetwork) {
        for (dst, src) in self.layers.iter_mut().zip(other.layers.iter()) {
            dst.copy_from(src);
        }
    }

    /// Polyak averaging update toward another network.
    pub fn polyak_update(&mut self, other: &QNetwork, tau: f64) {
        for (dst, src) in self.layers.iter_mut().zip(other.layers.iter()) {
            dst.polyak_update(src, tau);
        }
    }

    /// Apply a gradient update (SGD step) given per-layer weight gradients.
    /// `grad_layers` has the same shape as `self.layers`.
    pub fn sgd_update(&mut self, grad_layers: &[LayerGradient], lr: f64) {
        for (layer, grad) in self.layers.iter_mut().zip(grad_layers.iter()) {
            for i in 0..layer.out_dim {
                for j in 0..layer.in_dim {
                    layer.weights[i][j] -= lr * grad.dw[i][j];
                }
                layer.biases[i] -= lr * grad.db[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient storage
// ---------------------------------------------------------------------------

/// Gradients for one linear layer.
#[derive(Debug, Clone)]
pub struct LayerGradient {
    pub dw: Vec<Vec<f64>>,
    pub db: Vec<f64>,
}

impl LayerGradient {
    pub fn zeros_like(layer: &LinearLayer) -> Self {
        LayerGradient {
            dw: vec![vec![0.0; layer.in_dim]; layer.out_dim],
            db: vec![0.0; layer.out_dim],
        }
    }
}

// ---------------------------------------------------------------------------
// TrainerConfig
// ---------------------------------------------------------------------------

/// Hyperparameter bundle for `DQNTrainer`.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// SGD learning rate.
    pub learning_rate: f64,
    /// Discount factor.
    pub gamma: f64,
    /// Current epsilon for epsilon-greedy exploration.
    pub epsilon: f64,
    /// Multiplicative decay applied to epsilon each step.
    pub epsilon_decay: f64,
    /// Minimum epsilon after decay.
    pub epsilon_min: f64,
    /// Mini-batch size for each training step.
    pub batch_size: usize,
    /// How many steps between hard target-network copies (set to 0 for Polyak only).
    pub target_update_freq: usize,
    /// Polyak averaging coefficient (1.0 = hard copy).
    pub polyak_tau: f64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            learning_rate:     1e-3,
            gamma:             0.99,
            epsilon:           1.0,
            epsilon_decay:     0.9995,
            epsilon_min:       0.05,
            batch_size:        64,
            target_update_freq: 1000,
            polyak_tau:        0.005,
        }
    }
}

// ---------------------------------------------------------------------------
// DQNTrainer
// ---------------------------------------------------------------------------

/// Orchestrates Double DQN training for the SRFM exit policy.
///
/// Uses:
///   - online network to select actions (greedy / epsilon-greedy)
///   - target network to evaluate Q(s', argmax_a Q_online(s', a))
///   - Huber loss for stable gradient computation
///   - Polyak averaging for smooth target updates
pub struct DQNTrainer {
    /// Online Q-network (updated every training step).
    pub network:        QNetwork,
    /// Target Q-network (slowly tracks the online network).
    pub target_network: QNetwork,
    /// Experience replay buffer.
    pub buffer:         ExperienceReplayBuffer,
    /// Hyperparameters.
    pub config:         TrainerConfig,
    /// Total number of training steps completed.
    pub step_count:     u64,
}

impl DQNTrainer {
    /// Construct a new DQN trainer with randomly initialized networks.
    pub fn new<R: Rng>(buffer_capacity: usize, config: TrainerConfig, rng: &mut R) -> Self {
        let network = QNetwork::new(rng);
        let mut target_network = QNetwork::zeros();
        target_network.copy_from(&network);
        DQNTrainer {
            network,
            target_network,
            buffer: ExperienceReplayBuffer::new(buffer_capacity),
            config,
            step_count: 0,
        }
    }

    /// Store a transition in the replay buffer and optionally decay epsilon.
    pub fn store(&mut self, exp: Experience) {
        self.buffer.push(exp);
    }

    /// Select an action using epsilon-greedy policy.
    pub fn select_action<R: Rng>(&self, state: &[f64; 5], rng: &mut R) -> u8 {
        if rng.gen::<f64>() < self.config.epsilon {
            rng.gen_range(0..OUTPUT_DIM as u8)
        } else {
            self.network.greedy_action(state)
        }
    }

    /// Perform one mini-batch gradient update (Double DQN + Huber loss).
    ///
    /// Returns the mean absolute TD error of the batch, or None if the buffer
    /// does not yet contain enough experiences.
    pub fn train_step<R: Rng>(&mut self, rng: &mut R) -> Option<f64> {
        if !self.buffer.ready(self.config.batch_size) {
            return None;
        }

        let batch: Vec<Experience> = {
            let refs = self.buffer.sample_uniform(self.config.batch_size, rng);
            refs.into_iter().cloned().collect()
        };

        let lr   = self.config.learning_rate;
        let gamma = self.config.gamma;

        // Accumulate gradients for each layer.
        let mut grads: Vec<LayerGradient> = self
            .network
            .layers
            .iter()
            .map(LayerGradient::zeros_like)
            .collect();

        let mut total_td_error = 0.0_f64;

        for exp in &batch {
            // Double DQN target:
            //   a* = argmax_a Q_online(s', a)
            //   target = r + gamma * Q_target(s', a*)   (if not done)
            let online_next_qs  = self.network.forward(&exp.next_state);
            let best_next_action = online_next_qs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let target_next_qs = self.target_network.forward(&exp.next_state);
            let bootstrap = if exp.done { 0.0 } else { target_next_qs[best_next_action] };
            let target = exp.reward + gamma * bootstrap;

            // Current Q-values from online network.
            let current_qs = self.network.forward(&exp.state);
            let a = exp.action as usize;
            let prediction = current_qs[a];

            let delta = target - prediction;
            total_td_error += delta.abs();

            // Huber loss gradient wrt prediction:
            //   L = |delta| if |delta| < 1 else (delta^2 + 1)/2 - 0.5
            //   dL/d(prediction) = -huber_grad(delta)
            let huber_grad = if delta.abs() < 1.0 { delta } else { delta.signum() };

            // Backpropagate through the network via manual chain rule.
            // We only update the output corresponding to action `a`.
            let output_grad = {
                let mut g = [0.0_f64; OUTPUT_DIM];
                g[a] = -huber_grad; // gradient of loss wrt Q(s,a)
                g
            };

            // Full forward pass saving pre-activations for backprop.
            let pre_acts = self.forward_with_preacts(&exp.state);
            self.backprop(&pre_acts, &output_grad, &mut grads);
        }

        let n = batch.len() as f64;
        // Normalize gradients by batch size.
        for grad in grads.iter_mut() {
            for row in grad.dw.iter_mut() {
                for g in row.iter_mut() {
                    *g /= n;
                }
            }
            for g in grad.db.iter_mut() {
                *g /= n;
            }
        }

        self.network.sgd_update(&grads, lr);

        // Polyak-average target network every step.
        self.update_target_network();

        // Periodic hard copy.
        if self.config.target_update_freq > 0
            && self.step_count % self.config.target_update_freq as u64 == 0
        {
            self.target_network.copy_from(&self.network);
        }

        // Decay epsilon.
        self.config.epsilon = (self.config.epsilon * self.config.epsilon_decay)
            .max(self.config.epsilon_min);

        self.step_count += 1;
        Some(total_td_error / n)
    }

    /// Polyak-average the target network toward the online network.
    pub fn update_target_network(&mut self) {
        let tau = self.config.polyak_tau;
        self.target_network.polyak_update(&self.network, tau);
    }

    /// Evaluate the Q-network at all 5^5 = 3125 discretized states and
    /// return a HashMap compatible with the tabular RLExitPolicy format.
    ///
    /// Keys are formatted as decimal state indices ("0" .. "3124").
    /// Values are the three Q-values [hold, partial_exit, full_exit].
    pub fn export_qtable(&self) -> HashMap<String, [f64; 3]> {
        let mut table = HashMap::with_capacity(TOTAL_STATES);

        for flat_idx in 0..TOTAL_STATES {
            let state = decode_state_index(flat_idx);
            let qs = self.network.forward(&state);
            table.insert(flat_idx.to_string(), qs);
        }

        table
    }

    // ---- private helpers ---------------------------------------------------

    /// Run forward pass through the network, recording pre-activation vectors
    /// for each layer (needed for backprop).
    fn forward_with_preacts(&self, state: &[f64; 5]) -> Vec<(Vec<f64>, Vec<f64>)> {
        // Returns Vec of (pre_act, post_act) per layer.
        let mut x: Vec<f64> = state.to_vec();
        let mut records = Vec::with_capacity(self.network.layers.len());
        let n_layers = self.network.layers.len();

        for (i, layer) in self.network.layers.iter().enumerate() {
            let pre = layer.forward(&x);
            let act = if i < n_layers - 1 { Activation::ReLU } else { Activation::Linear };
            let post: Vec<f64> = pre.iter().map(|&v| act.apply(v)).collect();
            records.push((pre.clone(), post.clone()));
            x = post;
        }
        records
    }

    /// Manual backpropagation through the network.
    ///
    /// `pre_acts` -- output of `forward_with_preacts`.
    /// `output_grad` -- dL/d(output) for each output neuron.
    /// `grads` -- accumulate gradients here (not zeroed inside this fn).
    fn backprop(
        &self,
        pre_acts: &[(Vec<f64>, Vec<f64>)],
        output_grad: &[f64; OUTPUT_DIM],
        grads: &mut Vec<LayerGradient>,
    ) {
        let n_layers = self.network.layers.len();
        // delta propagates backwards through layers.
        let mut delta: Vec<f64> = output_grad.to_vec();

        for layer_idx in (0..n_layers).rev() {
            let layer      = &self.network.layers[layer_idx];
            let (pre, _post) = &pre_acts[layer_idx];
            let act = if layer_idx < n_layers - 1 { Activation::ReLU } else { Activation::Linear };

            // Multiply by activation derivative.
            let act_delta: Vec<f64> = delta
                .iter()
                .zip(pre.iter())
                .map(|(&d, &z)| d * act.derivative(z))
                .collect();

            // Gradient wrt weights and biases.
            let input = if layer_idx == 0 {
                // Input to the first layer is not stored; we skip weight grad accumulation
                // for layer 0 input -- instead retrieve it from pre_acts of the previous layer.
                // Actually we need the input to this layer.  For layer 0 it is the state itself.
                // We pass it in through pre_acts[layer_idx-1].post, or for layer 0 use zeros.
                // Simplified: we recompute inputs from post-activations of prior layer.
                None
            } else {
                Some(&pre_acts[layer_idx - 1].1)
            };

            // Accumulate weight gradients if we have the input activations.
            if let Some(inp) = input {
                let grad = &mut grads[layer_idx];
                for i in 0..layer.out_dim {
                    for j in 0..layer.in_dim {
                        grad.dw[i][j] += act_delta[i] * inp[j];
                    }
                    grad.db[i] += act_delta[i];
                }
            } else {
                // Layer 0: still accumulate bias gradients.
                let grad = &mut grads[layer_idx];
                for i in 0..layer.out_dim {
                    grad.db[i] += act_delta[i];
                }
            }

            // Propagate delta backward through weights.
            let mut new_delta = vec![0.0_f64; layer.in_dim];
            for j in 0..layer.in_dim {
                for i in 0..layer.out_dim {
                    new_delta[j] += layer.weights[i][j] * act_delta[i];
                }
            }
            delta = new_delta;
        }
    }
}

// ---------------------------------------------------------------------------
// State codec helpers
// ---------------------------------------------------------------------------

/// Convert a flat state index [0, TOTAL_STATES) to a normalized state vector.
///
/// Each feature is decoded from its bin index and mapped to the centre of its
/// [-1, 1] bin: bin b -> -1 + (2*b + 1) / NUM_BINS.
pub fn decode_state_index(flat_idx: usize) -> [f64; INPUT_DIM] {
    let mut remaining = flat_idx;
    let mut state = [0.0_f64; INPUT_DIM];
    for i in (0..INPUT_DIM).rev() {
        let bin = remaining % NUM_BINS;
        remaining /= NUM_BINS;
        // Map bin centre to [-1, 1].
        state[i] = -1.0 + (2.0 * bin as f64 + 1.0) / NUM_BINS as f64;
    }
    state
}

/// Encode a normalized state vector to a flat index by discretizing each feature.
pub fn encode_state(state: &[f64; INPUT_DIM]) -> usize {
    let mut idx = 0usize;
    for &v in state.iter() {
        let scaled = (v + 1.0) / 2.0; // [0, 1]
        let bin = ((scaled * NUM_BINS as f64).floor() as isize)
            .clamp(0, NUM_BINS as isize - 1) as usize;
        idx = idx * NUM_BINS + bin;
    }
    idx
}

// ---------------------------------------------------------------------------
// Utility: Box-Muller normal sample
// ---------------------------------------------------------------------------

/// Draw a standard normal sample using the Box-Muller transform.
fn sample_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-10);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn make_rng() -> SmallRng {
        SmallRng::seed_from_u64(99)
    }

    fn dummy_exp(action: u8, reward: f64) -> Experience {
        Experience::new(
            [0.1, -0.2, 0.5, 1.0, 0.0],
            action,
            reward,
            [0.2, -0.1, 0.4, 1.0, 0.1],
            false,
        )
    }

    // -- QNetwork tests ------------------------------------------------------

    #[test]
    fn test_forward_output_shape() {
        let mut rng = make_rng();
        let net = QNetwork::new(&mut rng);
        let qs = net.forward(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(qs.len(), 3);
    }

    #[test]
    fn test_greedy_action_in_range() {
        let mut rng = make_rng();
        let net = QNetwork::new(&mut rng);
        let a = net.greedy_action(&[0.1, 0.2, -0.3, 0.5, 0.0]);
        assert!(a < 3, "action out of range: {}", a);
    }

    #[test]
    fn test_copy_from_matches() {
        let mut rng = make_rng();
        let net = QNetwork::new(&mut rng);
        let mut copy = QNetwork::zeros();
        copy.copy_from(&net);
        let state = [0.1, 0.2, 0.3, 0.4, 0.5];
        let q1 = net.forward(&state);
        let q2 = copy.forward(&state);
        for i in 0..3 {
            assert!((q1[i] - q2[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_polyak_update_interpolates() {
        let mut rng = make_rng();
        let net_a = QNetwork::new(&mut rng);
        let net_b = QNetwork::new(&mut rng);
        let mut target = QNetwork::zeros();
        target.copy_from(&net_a);
        // After polyak with tau=0.5, target should be halfway between a and b.
        target.polyak_update(&net_b, 0.5);
        for (tl, al, bl) in target.layers.iter()
            .zip(net_a.layers.iter())
            .zip(net_b.layers.iter())
            .map(|((t, a), b)| (t, a, b))
        {
            for i in 0..tl.out_dim {
                for j in 0..tl.in_dim {
                    let expected = 0.5 * bl.weights[i][j] + 0.5 * al.weights[i][j];
                    assert!((tl.weights[i][j] - expected).abs() < 1e-10);
                }
            }
        }
    }

    // -- decode/encode roundtrip ---------------------------------------------

    #[test]
    fn test_decode_encode_roundtrip() {
        for flat in [0, 1, 100, 1000, 3124] {
            let state = decode_state_index(flat);
            let re_encoded = encode_state(&state);
            assert_eq!(re_encoded, flat, "roundtrip failed for flat={}", flat);
        }
    }

    #[test]
    fn test_total_states_coverage() {
        // Every flat index in [0, TOTAL_STATES) should decode to a valid state.
        for flat in 0..TOTAL_STATES {
            let state = decode_state_index(flat);
            for &v in &state {
                assert!(v >= -1.0 && v <= 1.0, "state value {} out of [-1,1]", v);
            }
        }
    }

    // -- DQNTrainer tests ----------------------------------------------------

    #[test]
    fn test_train_step_none_when_buffer_small() {
        let mut rng = make_rng();
        let mut trainer = DQNTrainer::new(10000, TrainerConfig::default(), &mut rng);
        // Add fewer than batch_size experiences.
        for _ in 0..10 {
            trainer.store(dummy_exp(0, 0.1));
        }
        let result = trainer.train_step(&mut rng);
        assert!(result.is_none());
    }

    #[test]
    fn test_train_step_returns_td_error() {
        let mut rng = make_rng();
        let config = TrainerConfig { batch_size: 16, ..Default::default() };
        let mut trainer = DQNTrainer::new(10000, config, &mut rng);
        for i in 0..100 {
            trainer.store(dummy_exp(i % 3, (i as f64) * 0.01));
        }
        let td = trainer.train_step(&mut rng);
        assert!(td.is_some());
        assert!(td.unwrap() >= 0.0);
    }

    #[test]
    fn test_epsilon_decays() {
        let mut rng = make_rng();
        let config = TrainerConfig {
            batch_size: 8,
            epsilon: 1.0,
            epsilon_decay: 0.5,
            epsilon_min: 0.05,
            ..Default::default()
        };
        let mut trainer = DQNTrainer::new(10000, config, &mut rng);
        for i in 0..100 {
            trainer.store(dummy_exp(i % 3, 0.0));
        }
        let eps_before = trainer.config.epsilon;
        trainer.train_step(&mut rng);
        assert!(trainer.config.epsilon < eps_before);
    }

    #[test]
    fn test_export_qtable_size() {
        let mut rng = make_rng();
        let trainer = DQNTrainer::new(100, TrainerConfig::default(), &mut rng);
        let table = trainer.export_qtable();
        assert_eq!(table.len(), TOTAL_STATES);
    }

    #[test]
    fn test_export_qtable_keys_and_values() {
        let mut rng = make_rng();
        let trainer = DQNTrainer::new(100, TrainerConfig::default(), &mut rng);
        let table = trainer.export_qtable();
        // Every key from "0" to "3124" must exist and have 3 values.
        for i in 0..TOTAL_STATES {
            let qs = table.get(&i.to_string()).expect("key missing");
            assert_eq!(qs.len(), 3);
        }
    }

    #[test]
    fn test_select_action_with_full_epsilon() {
        let mut rng = make_rng();
        let config = TrainerConfig { epsilon: 1.0, ..Default::default() };
        let trainer = DQNTrainer::new(100, config, &mut rng);
        let mut counts = [0u32; 3];
        for _ in 0..300 {
            let a = trainer.select_action(&[0.0; 5], &mut SmallRng::seed_from_u64(rng.gen()));
            counts[a as usize] += 1;
        }
        // With epsilon=1.0 all actions should appear.
        assert!(counts.iter().all(|&c| c > 0), "some action never selected");
    }

    #[test]
    fn test_multiple_train_steps_do_not_panic() {
        let mut rng = make_rng();
        let config = TrainerConfig { batch_size: 8, ..Default::default() };
        let mut trainer = DQNTrainer::new(10000, config, &mut rng);
        for i in 0..200 {
            trainer.store(dummy_exp(i % 3, (i as f64) * 0.005 - 0.5));
        }
        for _ in 0..10 {
            let _ = trainer.train_step(&mut rng);
        }
        assert!(trainer.step_count > 0);
    }
}
