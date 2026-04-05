use rand::seq::SliceRandom;
use rand::Rng;

use crate::action::Action;
use crate::state::StateVector;

/// A single experience tuple stored in the replay buffer.
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: StateVector,
    pub action: Action,
    pub reward: f64,
    pub next_state: StateVector,
    pub done: bool,
}

impl Experience {
    pub fn new(
        state: StateVector,
        action: Action,
        reward: f64,
        next_state: StateVector,
        done: bool,
    ) -> Self {
        Experience {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Fixed-capacity circular replay buffer for experience replay.
///
/// When the buffer is full, the oldest experiences are overwritten.
pub struct ReplayBuffer {
    data: Vec<Experience>,
    capacity: usize,
    head: usize,   // next write position
    size: usize,   // current number of stored experiences
}

impl ReplayBuffer {
    /// Create a new empty buffer with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        ReplayBuffer {
            data: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            size: 0,
        }
    }

    /// Push a new experience into the buffer.
    /// Overwrites the oldest experience when capacity is reached.
    pub fn push(&mut self, exp: Experience) {
        if self.size < self.capacity {
            self.data.push(exp);
            self.size += 1;
        } else {
            self.data[self.head] = exp;
        }
        self.head = (self.head + 1) % self.capacity;
    }

    /// Convenience wrapper to push individual fields.
    pub fn store(
        &mut self,
        state: StateVector,
        action: Action,
        reward: f64,
        next_state: StateVector,
        done: bool,
    ) {
        self.push(Experience::new(state, action, reward, next_state, done));
    }

    /// Current number of experiences stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// True if the buffer contains no experiences.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// True if the buffer has at least `n` experiences (enough for a mini-batch).
    #[inline]
    pub fn ready(&self, n: usize) -> bool {
        self.size >= n
    }

    /// Sample `n` experiences uniformly at random (without replacement).
    /// Returns `None` if there are fewer than `n` experiences stored.
    pub fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Option<Vec<&Experience>> {
        if self.size < n {
            return None;
        }
        let indices: Vec<usize> = (0..self.size).collect();
        let sampled: Vec<usize> = indices
            .choose_multiple(rng, n)
            .cloned()
            .collect();
        Some(sampled.iter().map(|&i| &self.data[i]).collect())
    }

    /// Sample and clone `n` experiences (owned).
    pub fn sample_owned<R: Rng>(&self, n: usize, rng: &mut R) -> Option<Vec<Experience>> {
        self.sample(n, rng)
            .map(|batch| batch.into_iter().cloned().collect())
    }

    /// Capacity of the buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drain all experiences (useful for on-policy variants or reset).
    pub fn clear(&mut self) {
        self.data.clear();
        self.head = 0;
        self.size = 0;
    }

    /// Statistics: mean reward of all stored experiences.
    pub fn mean_reward(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        self.data.iter().map(|e| e.reward).sum::<f64>() / self.size as f64
    }

    /// Fraction of stored experiences where action == EXIT.
    pub fn exit_fraction(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let exits = self
            .data
            .iter()
            .filter(|e| e.action == Action::Exit)
            .count();
        exits as f64 / self.size as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::TradeStateRaw;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn dummy_state(v: f64) -> StateVector {
        let raw = TradeStateRaw {
            position_pnl_pct: v,
            ..Default::default()
        };
        StateVector::from_raw(&raw)
    }

    fn fill(buf: &mut ReplayBuffer, n: usize) {
        for i in 0..n {
            buf.store(
                dummy_state(i as f64 * 0.01),
                if i % 2 == 0 { Action::Hold } else { Action::Exit },
                i as f64 * 0.1,
                dummy_state(i as f64 * 0.01 + 0.001),
                false,
            );
        }
    }

    #[test]
    fn test_push_and_len() {
        let mut buf = ReplayBuffer::new(10);
        assert_eq!(buf.len(), 0);
        fill(&mut buf, 5);
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_circular_overwrite() {
        let mut buf = ReplayBuffer::new(5);
        fill(&mut buf, 10); // overfill 2x
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_sample_returns_correct_count() {
        let mut buf = ReplayBuffer::new(50000);
        fill(&mut buf, 100);
        let mut rng = SmallRng::seed_from_u64(42);
        let batch = buf.sample(32, &mut rng).expect("should have enough");
        assert_eq!(batch.len(), 32);
    }

    #[test]
    fn test_sample_returns_none_when_insufficient() {
        let mut buf = ReplayBuffer::new(50);
        fill(&mut buf, 10);
        let mut rng = SmallRng::seed_from_u64(1);
        assert!(buf.sample(32, &mut rng).is_none());
    }

    #[test]
    fn test_ready() {
        let mut buf = ReplayBuffer::new(100);
        fill(&mut buf, 31);
        assert!(!buf.ready(32));
        fill(&mut buf, 1);
        assert!(buf.ready(32));
    }

    #[test]
    fn test_clear() {
        let mut buf = ReplayBuffer::new(50);
        fill(&mut buf, 20);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_exit_fraction() {
        let mut buf = ReplayBuffer::new(100);
        fill(&mut buf, 10); // alternating hold/exit -> 50%
        let frac = buf.exit_fraction();
        assert!((frac - 0.5).abs() < 0.01);
    }
}
