/// Welford's online algorithm for mean, variance, skewness, kurtosis.
/// All updates are O(1). No data is stored beyond the running accumulators.

use std::collections::VecDeque;

/// Online statistics accumulator using Welford's method.
/// Computes mean, variance, skewness, and kurtosis without storing data.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    n:  u64,
    m1: f64,   // mean
    m2: f64,   // sum of squared deviations
    m3: f64,   // sum of cubed deviations (for skewness)
    m4: f64,   // sum of 4th power deviations (for kurtosis)
}

impl StreamingStats {
    pub fn new() -> Self { Self::default() }

    pub fn update(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n  = self.n as f64;

        let delta   = x - self.m1;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1   = delta * delta_n * n1;

        self.m1 += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
                   + 6.0 * delta_n2 * self.m2
                   - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    pub fn count(&self)    -> u64  { self.n }
    pub fn mean(&self)     -> f64  { self.m1 }

    pub fn variance(&self) -> f64 {
        if self.n < 2 { return 0.0; }
        self.m2 / (self.n as f64 - 1.0)
    }

    pub fn std(&self)      -> f64  { self.variance().sqrt() }

    /// Fisher-Pearson standardised skewness coefficient
    pub fn skewness(&self) -> f64 {
        if self.n < 3 { return 0.0; }
        let n = self.n as f64;
        let variance = self.m2 / (n - 1.0);
        if variance == 0.0 { return 0.0; }
        (self.m3 / n) / variance.powf(1.5)
    }

    /// Excess kurtosis (0 = normal distribution)
    pub fn kurtosis(&self) -> f64 {
        if self.n < 4 { return 0.0; }
        let n = self.n as f64;
        let variance = self.m2 / n;
        if variance == 0.0 { return 0.0; }
        (self.m4 / n) / (variance * variance) - 3.0
    }

    pub fn reset(&mut self) { *self = Self::default(); }

    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            n:        self.n,
            mean:     self.mean(),
            std:      self.std(),
            skewness: self.skewness(),
            kurtosis: self.kurtosis(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StatsSnapshot {
    pub n:        u64,
    pub mean:     f64,
    pub std:      f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Fixed-capacity rolling window backed by a VecDeque.
#[derive(Debug, Clone)]
pub struct RollingWindow<T: Clone> {
    data:     VecDeque<T>,
    capacity: usize,
}

impl<T: Clone> RollingWindow<T> {
    pub fn new(capacity: usize) -> Self {
        Self { data: VecDeque::with_capacity(capacity), capacity }
    }

    /// Push a value; if full, the oldest is dropped. Returns the evicted value if any.
    pub fn push(&mut self, val: T) -> Option<T> {
        let evicted = if self.data.len() >= self.capacity {
            self.data.pop_front()
        } else {
            None
        };
        self.data.push_back(val);
        evicted
    }

    pub fn len(&self)        -> usize  { self.data.len() }
    pub fn is_full(&self)    -> bool   { self.data.len() == self.capacity }
    pub fn is_empty(&self)   -> bool   { self.data.is_empty() }
    pub fn capacity(&self)   -> usize  { self.capacity }
    pub fn iter(&self)       -> impl Iterator<Item = &T> { self.data.iter() }
    pub fn latest(&self)     -> Option<&T> { self.data.back() }
    pub fn oldest(&self)     -> Option<&T> { self.data.front() }

    pub fn as_slice_newest_last(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }
}

impl RollingWindow<f64> {
    /// Compute rolling mean of f64 window. O(n) but simple.
    pub fn mean(&self) -> Option<f64> {
        if self.is_empty() { return None; }
        Some(self.data.iter().sum::<f64>() / self.len() as f64)
    }

    pub fn std(&self) -> Option<f64> {
        let n = self.len();
        if n < 2 { return None; }
        let mean = self.mean().unwrap();
        let var = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        Some(var.sqrt())
    }

    /// Returns data sorted ascending for quantile computation
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if self.is_empty() { return None; }
        let mut v: Vec<f64> = self.data.iter().cloned().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p * (v.len() - 1) as f64).round() as usize;
        Some(v[idx])
    }
}

/// Exponential moving average and variance tracker
#[derive(Debug, Clone)]
pub struct ExponentialMovingStats {
    alpha:    f64,
    mean:     f64,
    variance: f64,
    n:        u64,
}

impl ExponentialMovingStats {
    pub fn new(alpha: f64) -> Self {
        assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0,1]");
        Self { alpha, mean: 0.0, variance: 0.0, n: 0 }
    }

    /// Half-life constructor: alpha = 1 - exp(-ln(2)/half_life)
    pub fn from_half_life(half_life: f64) -> Self {
        let alpha = 1.0 - (-std::f64::consts::LN_2 / half_life).exp();
        Self::new(alpha)
    }

    pub fn update(&mut self, x: f64) {
        if self.n == 0 {
            self.mean = x;
            self.n    = 1;
            return;
        }
        self.n += 1;
        let diff       = x - self.mean;
        self.mean     += self.alpha * diff;
        self.variance  = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff);
    }

    pub fn mean(&self)     -> f64 { self.mean }
    pub fn variance(&self) -> f64 { self.variance }
    pub fn std(&self)      -> f64 { self.variance.sqrt() }
    pub fn count(&self)    -> u64 { self.n }

    /// Z-score of a new observation against the running distribution
    pub fn zscore(&self, x: f64) -> f64 {
        let s = self.std();
        if s < 1e-12 { return 0.0; }
        (x - self.mean) / s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_stats_basic() {
        let mut s = StreamingStats::new();
        for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            s.update(x);
        }
        assert!((s.mean() - 5.0).abs() < 1e-10);
        assert!((s.std()  - 2.0).abs() < 1e-10);
    }

    #[test]
    fn rolling_window_eviction() {
        let mut w: RollingWindow<f64> = RollingWindow::new(3);
        w.push(1.0); w.push(2.0); w.push(3.0);
        assert!(w.is_full());
        let evicted = w.push(4.0);
        assert_eq!(evicted, Some(1.0));
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn ema_stats_convergence() {
        let mut e = ExponentialMovingStats::new(0.1);
        for _ in 0..1000 { e.update(5.0); }
        assert!((e.mean() - 5.0).abs() < 1e-6);
    }
}
