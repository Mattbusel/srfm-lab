// streaming_stats.rs — Online statistical estimators for real-time analytics.
//
// All algorithms are numerically stable online updates requiring only O(1) state
// (except RollingWindow which is O(capacity) by design).

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

// ─── Stats Snapshot ──────────────────────────────────────────────────────────

/// Snapshot of StreamingStats state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub count: u64,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub min: f64,
    pub max: f64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

// ─── StreamingStats ───────────────────────────────────────────────────────────

/// Online mean, variance, skewness, and kurtosis via Welford / West's algorithm.
///
/// Uses the numerically stable central moment update from:
/// B.P. Welford (1962) and T.B. West (1979).
pub struct StreamingStats {
    count: u64,
    m1: f64, // mean
    m2: f64, // sum of squared deviations
    m3: f64, // sum of cubed deviations
    m4: f64, // sum of fourth-power deviations
    min: f64,
    max: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Incorporate a new observation.
    pub fn update<T: Into<f64> + Copy>(&mut self, x: T) {
        let x = x.into();
        self.count += 1;
        let n = self.count as f64;
        let delta = x - self.m1;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        self.m1 += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;

        if x < self.min { self.min = x; }
        if x > self.max { self.max = x; }
    }

    /// Reset all accumulated state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Population variance (divide by n).
    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        self.m2 / self.count as f64
    }

    /// Sample variance (divide by n-1).
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        self.m2 / (self.count - 1) as f64
    }

    /// Sample standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    /// Sample skewness (Fisher's definition).
    pub fn skewness(&self) -> f64 {
        if self.count < 3 || self.m2 < 1e-14 { return 0.0; }
        let n = self.count as f64;
        let s = self.sample_variance().sqrt();
        (n / ((n - 1.0) * (n - 2.0))) * (self.m3 / self.m2) * (n / s)
    }

    /// Excess kurtosis (Fisher, normal = 0).
    pub fn kurtosis(&self) -> f64 {
        if self.count < 4 || self.m2 < 1e-14 { return 0.0; }
        let _n = self.count as f64;
        let var = self.variance();
        if var < 1e-14 { return 0.0; }
        (self.m4 / self.count as f64) / (var * var) - 3.0
    }

    pub fn mean(&self) -> f64 { self.m1 }
    pub fn count(&self) -> u64 { self.count }
    pub fn min(&self) -> f64 { self.min }
    pub fn max(&self) -> f64 { self.max }

    /// Return full stats snapshot.
    pub fn snapshot(&self) -> Stats {
        Stats {
            count: self.count,
            mean: self.mean(),
            variance: self.sample_variance(),
            std_dev: self.std_dev(),
            skewness: self.skewness(),
            kurtosis: self.kurtosis(),
            min: self.min,
            max: self.max,
        }
    }
}

// ─── RollingWindow ────────────────────────────────────────────────────────────

/// Fixed-capacity circular buffer over recent observations.
///
/// Provides O(1) push/evict and derives streaming stats from Welford over the
/// window population. Full recompute on eviction keeps variance exact.
pub struct RollingWindow<T> {
    buf: VecDeque<T>,
    capacity: usize,
    stats: StreamingStats,
}

impl<T: Into<f64> + Copy> RollingWindow<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RollingWindow capacity must be > 0");
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
            stats: StreamingStats::new(),
        }
    }

    /// Push a new value, evicting the oldest if at capacity.
    pub fn update(&mut self, x: T) {
        if self.buf.len() == self.capacity {
            // Recompute from scratch after eviction for numerical stability.
            self.buf.pop_front();
            self.buf.push_back(x);
            self.recompute();
        } else {
            self.buf.push_back(x);
            self.stats.update(x);
        }
    }

    fn recompute(&mut self) {
        self.stats.reset();
        for &v in &self.buf {
            self.stats.update(v);
        }
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.stats.reset();
    }

    pub fn snapshot(&self) -> Stats { self.stats.snapshot() }
    pub fn len(&self) -> usize { self.buf.len() }
    pub fn is_full(&self) -> bool { self.buf.len() == self.capacity }
    pub fn is_empty(&self) -> bool { self.buf.is_empty() }
    pub fn capacity(&self) -> usize { self.capacity }

    /// Access the underlying deque for iteration.
    pub fn values(&self) -> &VecDeque<T> { &self.buf }
}

// ─── ExponentialMovingStats ───────────────────────────────────────────────────

/// EWMA mean and variance with configurable smoothing factor α ∈ (0, 1].
///
/// Mean:     μ_t = α·x_t + (1−α)·μ_{t−1}
/// Variance: σ²_t = (1−α)·(σ²_{t−1} + α·(x_t − μ_{t−1})²)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialMovingStats {
    pub alpha: f64,
    pub mean: f64,
    pub variance: f64,
    pub count: u64,
    initialized: bool,
}

impl ExponentialMovingStats {
    /// Create with given smoothing factor. alpha = 2/(N+1) for N-period EMA.
    pub fn new(alpha: f64) -> Self {
        assert!((0.0..=1.0).contains(&alpha), "alpha must be in (0, 1]");
        Self { alpha, mean: 0.0, variance: 0.0, count: 0, initialized: false }
    }

    /// Convenience: create from equivalent EMA period N.
    pub fn from_period(n: u32) -> Self {
        Self::new(2.0 / (n as f64 + 1.0))
    }

    pub fn update<T: Into<f64> + Copy>(&mut self, x: T) {
        let x = x.into();
        self.count += 1;
        if !self.initialized {
            self.mean = x;
            self.variance = 0.0;
            self.initialized = true;
            return;
        }
        let diff = x - self.mean;
        self.mean += self.alpha * diff;
        self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff);
    }

    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.variance = 0.0;
        self.count = 0;
        self.initialized = false;
    }

    pub fn std_dev(&self) -> f64 { self.variance.sqrt() }

    pub fn snapshot(&self) -> ExponentialStats {
        ExponentialStats {
            alpha: self.alpha,
            mean: self.mean,
            variance: self.variance,
            std_dev: self.std_dev(),
            count: self.count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialStats {
    pub alpha: f64,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub count: u64,
}

// ─── StreamingQuantile (P² Algorithm) ────────────────────────────────────────

/// Online quantile estimator using the P² algorithm (Jain & Chlamtac 1985).
///
/// Estimates a single quantile p ∈ (0, 1) in O(1) space without storing data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2Quantile {
    p: f64,
    q: [f64; 5],   // marker heights
    n: [f64; 5],   // marker positions (desired, float)
    dn: [f64; 5],  // desired positions increments
    count: u64,
}

impl P2Quantile {
    pub fn new(p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "p must be in (0, 1)");
        Self {
            p,
            q: [0.0; 5],
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            count: 0,
        }
    }

    pub fn update(&mut self, x: f64) {
        self.count += 1;
        if self.count <= 5 {
            let idx = (self.count - 1) as usize;
            self.q[idx] = x;
            if self.count == 5 {
                // Sort initial 5 observations.
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            return;
        }

        // Find cell k where x falls.
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x < self.q[1] { 0 }
        else if x < self.q[2] { 1 }
        else if x < self.q[3] { 2 }
        else if x < self.q[4] { 3 }
        else {
            self.q[4] = x;
            3
        };

        // Increment positions of markers k+1..4.
        for i in (k + 1)..5 {
            self.n[i] += 1.0;
        }
        // Update desired positions.
        for i in 0..5 {
            self.dn[i] += if i == 0 { 0.0 }
                else if i == 1 { self.p / 2.0 }
                else if i == 2 { self.p }
                else if i == 3 { (1.0 + self.p) / 2.0 }
                else { 1.0 };
        }

        // Adjust marker heights.
        for i in 1..4 {
            let d = self.dn[i] - self.n[i];
            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1.0)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1.0)
            {
                let sign = if d > 0.0 { 1.0_f64 } else { -1.0_f64 };
                let q_new = self.parabolic(i, sign);
                if self.q[i - 1] < q_new && q_new < self.q[i + 1] {
                    self.q[i] = q_new;
                } else {
                    // Linear interpolation fallback.
                    let j = (i as isize + sign as isize) as usize;
                    self.q[i] += sign * (self.q[j] - self.q[i]) / (self.n[j] - self.n[i]);
                }
                self.n[i] += sign;
            }
        }
    }

    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let (qi, qp, qn) = (self.q[i], self.q[i - 1], self.q[i + 1]);
        let (ni, np, nn) = (self.n[i], self.n[i - 1], self.n[i + 1]);
        qi + d / (nn - np)
            * ((ni - np + d) * (qn - qi) / (nn - ni)
                + (nn - ni - d) * (qi - qp) / (ni - np))
    }

    /// Current estimate of the p-th quantile. Returns None until 5 samples seen.
    pub fn quantile(&self) -> Option<f64> {
        if self.count < 5 { None } else { Some(self.q[2]) }
    }

    pub fn reset(&mut self) {
        self.q = [0.0; 5];
        self.n = [1.0, 2.0, 3.0, 4.0, 5.0];
        self.count = 0;
    }
}

/// Tracks p50, p90, p95, p99 simultaneously via four P² estimators.
pub struct StreamingQuantile {
    pub p50: P2Quantile,
    pub p90: P2Quantile,
    pub p95: P2Quantile,
    pub p99: P2Quantile,
    count: u64,
}

impl StreamingQuantile {
    pub fn new() -> Self {
        Self {
            p50: P2Quantile::new(0.50),
            p90: P2Quantile::new(0.90),
            p95: P2Quantile::new(0.95),
            p99: P2Quantile::new(0.99),
            count: 0,
        }
    }

    pub fn update<T: Into<f64> + Copy>(&mut self, x: T) {
        let x = x.into();
        self.count += 1;
        self.p50.update(x);
        self.p90.update(x);
        self.p95.update(x);
        self.p99.update(x);
    }

    pub fn reset(&mut self) {
        self.p50.reset();
        self.p90.reset();
        self.p95.reset();
        self.p99.reset();
        self.count = 0;
    }

    pub fn snapshot(&self) -> QuantileSnapshot {
        QuantileSnapshot {
            count: self.count,
            p50: self.p50.quantile(),
            p90: self.p90.quantile(),
            p95: self.p95.quantile(),
            p99: self.p99.quantile(),
        }
    }
}

impl Default for StreamingQuantile {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileSnapshot {
    pub count: u64,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
}

// ─── CorrelationTracker ───────────────────────────────────────────────────────

/// Online Pearson correlation between two simultaneous streams X and Y.
///
/// Uses Chan's parallel algorithm (equivalent to Welford for bivariate case).
pub struct CorrelationTracker {
    count: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,   // sum of (x - mean_x)²
    m2_y: f64,   // sum of (y - mean_y)²
    c_xy: f64,   // sum of (x - mean_x)(y - mean_y)
}

impl Default for CorrelationTracker {
    fn default() -> Self { Self::new() }
}

impl CorrelationTracker {
    pub fn new() -> Self {
        Self { count: 0, mean_x: 0.0, mean_y: 0.0, m2_x: 0.0, m2_y: 0.0, c_xy: 0.0 }
    }

    /// Update with a simultaneous (x, y) pair.
    pub fn update<T: Into<f64> + Copy>(&mut self, x: T, y: T) {
        let (x, y) = (x.into(), y.into());
        self.count += 1;
        let n = self.count as f64;
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;
        self.mean_x += dx / n;
        self.mean_y += dy / n;
        let dx2 = x - self.mean_x;
        let dy2 = y - self.mean_y;
        self.m2_x += dx * dx2;
        self.m2_y += dy * dy2;
        self.c_xy += dx * dy2;
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Pearson r ∈ [-1, 1]. Returns None if fewer than 2 observations.
    pub fn correlation(&self) -> Option<f64> {
        if self.count < 2 { return None; }
        let denom = (self.m2_x * self.m2_y).sqrt();
        if denom < 1e-14 { return None; }
        Some((self.c_xy / denom).clamp(-1.0, 1.0))
    }

    /// Covariance (sample).
    pub fn covariance(&self) -> Option<f64> {
        if self.count < 2 { return None; }
        Some(self.c_xy / (self.count - 1) as f64)
    }

    pub fn snapshot(&self) -> CorrelationSnapshot {
        CorrelationSnapshot {
            count: self.count,
            mean_x: self.mean_x,
            mean_y: self.mean_y,
            std_x: (self.m2_x / self.count.saturating_sub(1).max(1) as f64).sqrt(),
            std_y: (self.m2_y / self.count.saturating_sub(1).max(1) as f64).sqrt(),
            covariance: self.covariance(),
            correlation: self.correlation(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationSnapshot {
    pub count: u64,
    pub mean_x: f64,
    pub mean_y: f64,
    pub std_x: f64,
    pub std_y: f64,
    pub covariance: Option<f64>,
    pub correlation: Option<f64>,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_stats_mean_variance() {
        let mut s = StreamingStats::new();
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        for &x in &data { s.update(x); }
        assert!((s.mean() - 5.0).abs() < 1e-10);
        assert!((s.sample_variance() - 4.571428).abs() < 1e-4);
    }

    #[test]
    fn test_rolling_window_eviction() {
        let mut w: RollingWindow<f64> = RollingWindow::new(3);
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] { w.update(x); }
        assert_eq!(w.len(), 3);
        let snap = w.snapshot();
        assert!((snap.mean - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ewma() {
        let mut e = ExponentialMovingStats::from_period(10);
        for x in [10.0_f64; 100] { e.update(x); }
        assert!((e.mean - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_tracker_perfect() {
        let mut c = CorrelationTracker::new();
        for i in 0..100 { c.update(i as f64, i as f64 * 2.0); }
        assert!((c.correlation().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_quantile_normal() {
        use std::f64::consts::PI;
        let mut q = StreamingQuantile::new();
        // Feed values from 0..1000; p50 ≈ 500.
        for i in 0..1000 { q.update(i as f64); }
        let snap = q.snapshot();
        let p50 = snap.p50.unwrap();
        // P² is approximate; allow ±5% relative error.
        assert!((p50 - 499.5).abs() / 499.5 < 0.05, "p50={p50}");
        let _ = PI; // suppress unused import warning
    }
}
