// experience_replay.rs -- Experience replay buffer for DQN training.
//
// Provides:
//   - ExperienceReplayBuffer: uniform and prioritized experience replay (PER)
//   - NStepBuffer: n-step return accumulation before storing to the main buffer

use std::collections::VecDeque;
use rand::Rng;
use rand::seq::SliceRandom;

// ---------------------------------------------------------------------------
// Experience tuple
// ---------------------------------------------------------------------------

/// A single (s, a, r, s', done) transition with a compact 5-feature state.
///
/// State layout matches the 5-dimensional discretized export used by RLExitPolicy:
///   [pnl_pct_norm, bars_held_norm, bh_mass_norm, bh_active, atr_ratio_norm]
#[derive(Debug, Clone)]
pub struct Experience {
    pub state:      [f64; 5],
    pub action:     u8,       // 0=HOLD, 1=PARTIAL_EXIT, 2=FULL_EXIT
    pub reward:     f64,
    pub next_state: [f64; 5],
    pub done:       bool,
}

impl Experience {
    pub fn new(
        state:      [f64; 5],
        action:     u8,
        reward:     f64,
        next_state: [f64; 5],
        done:       bool,
    ) -> Self {
        Experience { state, action, reward, next_state, done }
    }
}

// ---------------------------------------------------------------------------
// ExperienceReplayBuffer
// ---------------------------------------------------------------------------

/// Fixed-capacity replay buffer supporting both uniform and prioritized sampling.
///
/// When capacity is exceeded the oldest experience is evicted (FIFO).
/// Priorities are maintained in a parallel Vec and updated via `update_priorities`.
pub struct ExperienceReplayBuffer {
    pub capacity:   usize,
    buffer:         VecDeque<Experience>,
    priorities:     Vec<f64>,
    /// Small positive constant added to every priority to prevent zero-sampling.
    pub priority_eps: f64,
}

impl ExperienceReplayBuffer {
    /// Create an empty buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        ExperienceReplayBuffer {
            capacity,
            buffer:       VecDeque::with_capacity(capacity),
            priorities:   Vec::with_capacity(capacity),
            priority_eps: 1e-6,
        }
    }

    /// Current number of stored experiences.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// True if no experiences are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// True when the buffer holds at least `n` experiences.
    #[inline]
    pub fn ready(&self, n: usize) -> bool {
        self.buffer.len() >= n
    }

    /// Push a new experience.  When the buffer is full the oldest entry is
    /// evicted (FIFO).  New experiences receive the maximum current priority
    /// so they are sampled at least once before being down-weighted.
    pub fn push(&mut self, exp: Experience) {
        let max_priority = self.priorities.iter().cloned().fold(1.0_f64, f64::max);

        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
            self.priorities.remove(0);
        }

        self.buffer.push_back(exp);
        self.priorities.push(max_priority);
    }

    /// Sample `batch_size` experiences uniformly at random without replacement.
    ///
    /// Returns fewer than `batch_size` only when the buffer is smaller.
    pub fn sample_uniform<R: Rng>(
        &self,
        batch_size: usize,
        rng: &mut R,
    ) -> Vec<&Experience> {
        let n = batch_size.min(self.buffer.len());
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        let chosen: Vec<usize> = indices.choose_multiple(rng, n).cloned().collect();
        chosen.iter().map(|&i| &self.buffer[i]).collect()
    }

    /// Prioritized experience replay (PER) sampling.
    ///
    /// Sampling probability: P(i) = p_i^alpha / sum(p_j^alpha)
    /// Returns (sampled experiences, importance-sampling weights).
    /// IS weights: w_i = (N * P(i))^{-beta} normalized by max weight.
    /// Here beta defaults to 1.0; callers should anneal it externally.
    pub fn sample_prioritized<R: Rng>(
        &self,
        batch_size: usize,
        alpha: f64,
        rng: &mut R,
    ) -> (Vec<&Experience>, Vec<f64>) {
        let n = self.buffer.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let k = batch_size.min(n);

        // Compute scaled priorities.
        let scaled: Vec<f64> = self
            .priorities
            .iter()
            .map(|&p| (p + self.priority_eps).powf(alpha))
            .collect();
        let sum: f64 = scaled.iter().sum();

        // Build CDF for inverse transform sampling.
        let mut cdf: Vec<f64> = Vec::with_capacity(n);
        let mut acc = 0.0_f64;
        for &s in &scaled {
            acc += s / sum;
            cdf.push(acc);
        }

        // Sample k indices proportional to priority (with replacement for simplicity).
        let mut chosen_indices: Vec<usize> = Vec::with_capacity(k);
        for _ in 0..k {
            let u: f64 = rng.gen::<f64>();
            let idx = cdf.partition_point(|&c| c < u).min(n - 1);
            chosen_indices.push(idx);
        }

        // Compute IS weights: w_i = (N * P(i))^{-1} normalized.
        let mut weights: Vec<f64> = chosen_indices
            .iter()
            .map(|&i| {
                let p_i = scaled[i] / sum;
                (n as f64 * p_i).recip()
            })
            .collect();

        let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1e-10);
        for w in weights.iter_mut() {
            *w /= max_w;
        }

        let experiences: Vec<&Experience> = chosen_indices
            .iter()
            .map(|&i| &self.buffer[i])
            .collect();

        (experiences, weights)
    }

    /// Update priorities after computing new TD errors.
    ///
    /// `indices` must correspond to the buffer positions returned by a prior
    /// sample call.  Out-of-range indices are silently ignored.
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f64]) {
        for (&idx, &err) in indices.iter().zip(td_errors.iter()) {
            if idx < self.priorities.len() {
                self.priorities[idx] = err.abs() + self.priority_eps;
            }
        }
    }

    /// Return the mean priority (useful for diagnostics).
    pub fn mean_priority(&self) -> f64 {
        if self.priorities.is_empty() {
            return 0.0;
        }
        self.priorities.iter().sum::<f64>() / self.priorities.len() as f64
    }

    /// Iterate over all stored experiences (in order from oldest to newest).
    pub fn iter(&self) -> impl Iterator<Item = &Experience> {
        self.buffer.iter()
    }
}

// ---------------------------------------------------------------------------
// NStepBuffer
// ---------------------------------------------------------------------------

/// Accumulates a short episode prefix and computes n-step discounted returns.
///
/// When the internal buffer reaches depth `n`, the oldest experience is popped
/// and its reward is replaced with the n-step return:
///
///   R_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1}
///
/// The `next_state` of the returned experience points to s_{t+n} so that a
/// downstream Q-network can bootstrap the tail value.
/// The terminal flag is set if *any* step in the window was terminal.
pub struct NStepBuffer {
    n:      usize,
    gamma:  f64,
    buffer: VecDeque<Experience>,
}

impl NStepBuffer {
    /// Create a new n-step buffer.
    ///
    /// * `n`     -- number of steps to accumulate
    /// * `gamma` -- discount factor applied between consecutive steps
    pub fn new(n: usize, gamma: f64) -> Self {
        assert!(n >= 1, "n must be >= 1");
        NStepBuffer {
            n,
            gamma,
            buffer: VecDeque::with_capacity(n),
        }
    }

    /// Push one experience into the accumulation window.
    ///
    /// Returns `Some(Experience)` when the window contains exactly `n` steps,
    /// where the returned experience carries the n-step discounted return and
    /// the next_state from the end of the window.
    ///
    /// Returns `None` while still filling the window.
    pub fn push(&mut self, exp: Experience) -> Option<Experience> {
        self.buffer.push_back(exp);

        if self.buffer.len() < self.n {
            return None;
        }

        // Compute n-step return starting from the oldest experience.
        let mut n_step_return = 0.0_f64;
        let mut discount = 1.0_f64;
        let mut any_done = false;

        for step in self.buffer.iter() {
            n_step_return += discount * step.reward;
            discount *= self.gamma;
            if step.done {
                any_done = true;
                break; // reward beyond a terminal is zero
            }
        }

        // The "next state" is the state at the end of the n-step window.
        let tail = self.buffer.back().expect("buffer non-empty");
        let next_state = tail.next_state;

        // Pop the oldest experience to slide the window.
        let oldest = self.buffer.pop_front().expect("buffer non-empty");

        Some(Experience {
            state:      oldest.state,
            action:     oldest.action,
            reward:     n_step_return,
            next_state,
            done:       any_done,
        })
    }

    /// Flush any remaining experiences in the window, computing their partial
    /// n-step returns.  Call this at the end of each episode.
    pub fn flush(&mut self) -> Vec<Experience> {
        let mut out = Vec::new();
        while !self.buffer.is_empty() {
            let mut n_step_return = 0.0_f64;
            let mut discount = 1.0_f64;
            let mut any_done = false;

            for step in self.buffer.iter() {
                n_step_return += discount * step.reward;
                discount *= self.gamma;
                if step.done {
                    any_done = true;
                    break;
                }
            }

            let tail_next = self.buffer.back().expect("non-empty").next_state;
            let oldest = self.buffer.pop_front().expect("non-empty");

            out.push(Experience {
                state:      oldest.state,
                action:     oldest.action,
                reward:     n_step_return,
                next_state: tail_next,
                done:       any_done,
            });
        }
        out
    }

    /// Current fill level of the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// True if no experiences are buffered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn make_exp(reward: f64, done: bool) -> Experience {
        Experience::new([0.1, 0.2, 0.3, 0.4, 0.5], 0, reward, [0.2, 0.3, 0.4, 0.5, 0.6], done)
    }

    fn make_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    // -- ExperienceReplayBuffer tests ----------------------------------------

    #[test]
    fn test_push_and_len() {
        let mut buf = ExperienceReplayBuffer::new(100);
        assert_eq!(buf.len(), 0);
        buf.push(make_exp(1.0, false));
        buf.push(make_exp(2.0, false));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_fifo_eviction() {
        let mut buf = ExperienceReplayBuffer::new(3);
        for i in 0..5 {
            buf.push(make_exp(i as f64, false));
        }
        // Only 3 entries should remain.
        assert_eq!(buf.len(), 3);
        // The oldest 2 were evicted; remaining rewards should be 2.0, 3.0, 4.0.
        let rewards: Vec<f64> = buf.iter().map(|e| e.reward).collect();
        assert_eq!(rewards, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sample_uniform_count() {
        let mut buf = ExperienceReplayBuffer::new(1000);
        for i in 0..100 {
            buf.push(make_exp(i as f64, false));
        }
        let mut rng = make_rng();
        let batch = buf.sample_uniform(32, &mut rng);
        assert_eq!(batch.len(), 32);
    }

    #[test]
    fn test_sample_uniform_smaller_than_batch() {
        let mut buf = ExperienceReplayBuffer::new(10);
        for i in 0..5 {
            buf.push(make_exp(i as f64, false));
        }
        let mut rng = make_rng();
        // Should return only 5 even though we asked for 32.
        let batch = buf.sample_uniform(32, &mut rng);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_sample_prioritized_returns_weights() {
        let mut buf = ExperienceReplayBuffer::new(500);
        for i in 0..100 {
            buf.push(make_exp(i as f64, false));
        }
        let mut rng = make_rng();
        let (exps, weights) = buf.sample_prioritized(32, 0.6, &mut rng);
        assert_eq!(exps.len(), 32);
        assert_eq!(weights.len(), 32);
        // IS weights should be in (0, 1].
        for &w in &weights {
            assert!(w > 0.0 && w <= 1.0 + 1e-9, "weight out of range: {}", w);
        }
    }

    #[test]
    fn test_update_priorities_changes_values() {
        let mut buf = ExperienceReplayBuffer::new(100);
        for _ in 0..10 {
            buf.push(make_exp(1.0, false));
        }
        let initial_mean = buf.mean_priority();
        buf.update_priorities(&[0, 1, 2], &[10.0, 20.0, 30.0]);
        let new_mean = buf.mean_priority();
        assert!(new_mean > initial_mean, "priorities should increase after large TD errors");
    }

    #[test]
    fn test_ready() {
        let mut buf = ExperienceReplayBuffer::new(100);
        for i in 0..31 {
            buf.push(make_exp(i as f64, false));
        }
        assert!(!buf.ready(32));
        buf.push(make_exp(31.0, false));
        assert!(buf.ready(32));
    }

    #[test]
    fn test_priorities_parallel_length() {
        let mut buf = ExperienceReplayBuffer::new(50);
        for i in 0..50 {
            buf.push(make_exp(i as f64, false));
        }
        assert_eq!(buf.priorities.len(), buf.len());
        // Overfill -- still consistent.
        buf.push(make_exp(99.0, false));
        assert_eq!(buf.priorities.len(), buf.len());
    }

    // -- NStepBuffer tests ---------------------------------------------------

    #[test]
    fn test_nstep_none_while_filling() {
        let mut nb = NStepBuffer::new(3, 0.99);
        assert!(nb.push(make_exp(1.0, false)).is_none());
        assert!(nb.push(make_exp(1.0, false)).is_none());
    }

    #[test]
    fn test_nstep_returns_on_nth_push() {
        let mut nb = NStepBuffer::new(3, 0.99);
        nb.push(make_exp(1.0, false));
        nb.push(make_exp(1.0, false));
        let result = nb.push(make_exp(1.0, false));
        assert!(result.is_some());
    }

    #[test]
    fn test_nstep_correct_discounting() {
        // n=3, gamma=0.5, rewards = [1, 1, 1]
        // R = 1 + 0.5*1 + 0.25*1 = 1.75
        let mut nb = NStepBuffer::new(3, 0.5);
        nb.push(make_exp(1.0, false));
        nb.push(make_exp(1.0, false));
        let result = nb.push(make_exp(1.0, false)).expect("should pop");
        assert!((result.reward - 1.75).abs() < 1e-9, "got {}", result.reward);
    }

    #[test]
    fn test_nstep_terminal_truncates_sum() {
        // n=3, gamma=0.9, first step is terminal -- subsequent rewards should be excluded.
        let mut nb = NStepBuffer::new(3, 0.9);
        nb.push(make_exp(5.0, true));
        nb.push(make_exp(3.0, false));
        let result = nb.push(make_exp(2.0, false)).expect("should pop");
        // After terminal, sum stops: R = 5.0 (only first step).
        assert!((result.reward - 5.0).abs() < 1e-9, "got {}", result.reward);
        assert!(result.done, "done flag should be set");
    }

    #[test]
    fn test_nstep_flush_drains_buffer() {
        let mut nb = NStepBuffer::new(5, 0.99);
        for i in 0..3 {
            nb.push(make_exp(i as f64, false));
        }
        assert_eq!(nb.len(), 3);
        let flushed = nb.flush();
        assert_eq!(flushed.len(), 3);
        assert_eq!(nb.len(), 0);
    }

    #[test]
    fn test_nstep_next_state_is_tail() {
        let mut nb = NStepBuffer::new(2, 0.9);
        let mut e1 = make_exp(1.0, false);
        e1.state = [0.1; 5];
        let mut e2 = make_exp(2.0, false);
        e2.next_state = [0.9; 5];

        nb.push(e1);
        let result = nb.push(e2).expect("should pop");
        // next_state should come from the tail of the window (e2).
        assert_eq!(result.next_state, [0.9; 5]);
    }
}
