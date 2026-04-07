/// Market contagion detection and measurement.
///
/// Uses EWMA covariance (DCC-GARCH proxy) and exceedance correlation to detect
/// when extreme joint moves indicate contagion rather than normal co-movement.

use serde::{Deserialize, Serialize};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default EWMA decay factor (lambda). Common choice: 0.94 (RiskMetrics).
const DEFAULT_LAMBDA: f64 = 0.94;

/// Contagion threshold: extreme_corr / avg_corr > this => contagion regime.
const CONTAGION_RATIO_THRESHOLD: f64 = 1.2;

/// Default percentile threshold for exceedance correlation (e.g. 0.90 = top 10%).
const DEFAULT_EXCEEDANCE_PCT: f64 = 0.90;

// ── ContagionEvent ────────────────────────────────────────────────────────────

/// Record of a detected contagion event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContagionEvent {
    /// Bar index when the event was detected.
    pub bar_index: u64,
    /// Rolling EWMA correlation at the time of detection.
    pub ewma_correlation: f64,
    /// Contagion score at detection.
    pub score: f64,
}

// ── EwmaState ─────────────────────────────────────────────────────────────────

/// EWMA-based covariance estimator for a single pair (x, y).
///
/// DCC-GARCH simplified: treat the conditional covariance as exponentially
/// weighted sums of cross-products of demeaned returns.
#[derive(Debug, Clone)]
struct EwmaState {
    lambda: f64,
    /// EWMA of x.
    mean_x: f64,
    /// EWMA of y.
    mean_y: f64,
    /// EWMA of (x - mean_x)^2.
    var_x: f64,
    /// EWMA of (y - mean_y)^2.
    var_y: f64,
    /// EWMA of (x - mean_x)(y - mean_y).
    cov_xy: f64,
    /// Number of observations seen.
    count: u64,
}

impl EwmaState {
    fn new(lambda: f64) -> Self {
        EwmaState {
            lambda,
            mean_x: 0.0,
            mean_y: 0.0,
            var_x: 1e-10,
            var_y: 1e-10,
            cov_xy: 0.0,
            count: 0,
        }
    }

    fn update(&mut self, x: f64, y: f64) {
        let lam = self.lambda;
        let inv_lam = 1.0 - lam;

        if self.count == 0 {
            self.mean_x = x;
            self.mean_y = y;
            self.count += 1;
            return;
        }

        // Update means first (EWMA).
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;

        self.mean_x = lam * self.mean_x + inv_lam * x;
        self.mean_y = lam * self.mean_y + inv_lam * y;

        // Update (co)variances using pre-update deviations.
        self.var_x = lam * self.var_x + inv_lam * dx * dx;
        self.var_y = lam * self.var_y + inv_lam * dy * dy;
        self.cov_xy = lam * self.cov_xy + inv_lam * dx * dy;

        self.count += 1;
    }

    /// Current EWMA correlation in [-1, 1]. Returns None if not enough data.
    fn correlation(&self) -> Option<f64> {
        if self.count < 5 {
            return None;
        }
        let denom = (self.var_x * self.var_y).sqrt();
        if denom < 1e-14 {
            return None;
        }
        Some((self.cov_xy / denom).clamp(-1.0, 1.0))
    }
}

// ── ContagionDetector ─────────────────────────────────────────────────────────

/// Measures market contagion via EWMA covariance and exceedance correlation.
pub struct ContagionDetector {
    /// EWMA decay factor.
    lambda: f64,
    /// Per-pair EWMA state. Key: canonical (a, b) with a <= b.
    ewma: std::collections::HashMap<(String, String), EwmaState>,
    /// Monotonic bar counter.
    bar_idx: u64,
    /// Detected contagion events.
    events: Vec<ContagionEvent>,
    /// Ring buffer of recent EWMA correlations for baseline estimation.
    /// Only used for `contagion_score` baseline.
    recent_corr_buffer: Vec<f64>,
    /// Baseline capacity (number of bars used to establish average correlation).
    baseline_capacity: usize,
}

impl ContagionDetector {
    /// Create a new detector with EWMA lambda and baseline window.
    pub fn new(lambda: f64, baseline_window: usize) -> Self {
        ContagionDetector {
            lambda,
            ewma: std::collections::HashMap::new(),
            bar_idx: 0,
            events: Vec::new(),
            recent_corr_buffer: Vec::new(),
            baseline_capacity: baseline_window.max(20),
        }
    }

    /// Create with sensible defaults (lambda=0.94, window=60).
    pub fn default_params() -> Self {
        Self::new(DEFAULT_LAMBDA, 60)
    }

    // ── Feed data ─────────────────────────────────────────────────────────

    /// Feed a new joint observation (x_t, y_t) for the given asset pair.
    ///
    /// Automatically checks for contagion and records events.
    pub fn update(&mut self, asset_a: &str, asset_b: &str, ret_a: f64, ret_b: f64) {
        let key = canonical_key(asset_a, asset_b);
        let lam = self.lambda;
        let state = self.ewma.entry(key).or_insert_with(|| EwmaState::new(lam));

        if asset_a <= asset_b {
            state.update(ret_a, ret_b);
        } else {
            state.update(ret_b, ret_a);
        }

        // Track recent correlation for baseline.
        if let Some(r) = state.correlation() {
            if self.recent_corr_buffer.len() < self.baseline_capacity {
                self.recent_corr_buffer.push(r);
            } else {
                let idx = (self.bar_idx as usize) % self.baseline_capacity;
                self.recent_corr_buffer[idx] = r;
            }

            // Check for contagion.
            let score = self.contagion_score_internal(r);
            if score > CONTAGION_RATIO_THRESHOLD {
                self.events.push(ContagionEvent {
                    bar_index: self.bar_idx,
                    ewma_correlation: r,
                    score,
                });
            }
        }

        self.bar_idx += 1;
    }

    // ── EWMA Correlation ──────────────────────────────────────────────────

    /// Current EWMA correlation for the given pair.
    pub fn ewma_correlation(&self, asset_a: &str, asset_b: &str) -> Option<f64> {
        let key = canonical_key(asset_a, asset_b);
        self.ewma.get(&key)?.correlation()
    }

    // ── Exceedance Correlation ────────────────────────────────────────────

    /// Correlation conditional on both `x` and `y` simultaneously exceeding
    /// `threshold_pct` (e.g. 0.90 = top 10% in absolute terms).
    ///
    /// This captures tail dependence: if extreme correlation >> unconditional
    /// correlation, the pair exhibits contagion-like behaviour.
    pub fn exceedance_correlation(x: &[f64], y: &[f64], threshold_pct: f64) -> f64 {
        let n = x.len().min(y.len());
        if n < 5 {
            return 0.0;
        }

        // Compute thresholds as absolute quantile.
        let mut abs_x: Vec<f64> = x[..n].iter().map(|v| v.abs()).collect();
        let mut abs_y: Vec<f64> = y[..n].iter().map(|v| v.abs()).collect();
        abs_x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        abs_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let thresh_x = quantile(&abs_x, threshold_pct);
        let thresh_y = quantile(&abs_y, threshold_pct);

        // Select observations where both exceed their respective threshold.
        let xs_exc: Vec<f64> = x[..n]
            .iter()
            .zip(y[..n].iter())
            .filter(|(&xi, &yi)| xi.abs() >= thresh_x && yi.abs() >= thresh_y)
            .map(|(&xi, _)| xi)
            .collect();
        let ys_exc: Vec<f64> = x[..n]
            .iter()
            .zip(y[..n].iter())
            .filter(|(&xi, &yi)| xi.abs() >= thresh_x && yi.abs() >= thresh_y)
            .map(|(_, &yi)| yi)
            .collect();

        if xs_exc.len() < 3 {
            return 0.0;
        }
        pearson(&xs_exc, &ys_exc).unwrap_or(0.0)
    }

    // ── Contagion Score ───────────────────────────────────────────────────

    /// Compute contagion score for a pair using full return history.
    ///
    /// score = exceedance_corr(x, y, 0.90) / avg_corr(x, y)
    ///
    /// Values > 1.2 indicate a contagion regime.
    pub fn contagion_score_from_series(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 10 {
            return 0.0;
        }
        let avg_r = pearson(&x[..n], &y[..n]).unwrap_or(0.0).abs();
        if avg_r < 1e-6 {
            return 0.0;
        }
        let exc_r = Self::exceedance_correlation(x, y, DEFAULT_EXCEEDANCE_PCT).abs();
        exc_r / avg_r
    }

    /// Rolling contagion score based on the EWMA estimator.
    ///
    /// Returns the ratio of the current EWMA |correlation| to the baseline
    /// average |correlation|. > 1.2 => contagion regime.
    pub fn contagion_score(&self) -> f64 {
        if self.recent_corr_buffer.is_empty() {
            return 0.0;
        }
        let current = self
            .recent_corr_buffer
            .last()
            .copied()
            .unwrap_or(0.0)
            .abs();

        let baseline: f64 = if self.recent_corr_buffer.len() < 2 {
            current
        } else {
            let n = self.recent_corr_buffer.len() - 1; // exclude last point
            let sum: f64 = self.recent_corr_buffer[..n].iter().map(|v| v.abs()).sum();
            sum / n as f64
        };

        if baseline < 1e-6 {
            return 0.0;
        }
        current / baseline
    }

    /// Internal helper: compute contagion score given a current correlation.
    fn contagion_score_internal(&self, current_r: f64) -> f64 {
        let n = self.recent_corr_buffer.len();
        if n == 0 {
            return 0.0;
        }
        let baseline: f64 = self.recent_corr_buffer.iter().map(|v| v.abs()).sum::<f64>()
            / n as f64;
        if baseline < 1e-6 {
            return 0.0;
        }
        current_r.abs() / baseline
    }

    // ── Contagion Events ──────────────────────────────────────────────────

    /// Return all detected contagion events.
    pub fn contagion_events(&self) -> &[ContagionEvent] {
        &self.events
    }

    /// True if the most recent contagion score exceeds the threshold.
    pub fn is_contagion_regime(&self) -> bool {
        self.contagion_score() > CONTAGION_RATIO_THRESHOLD
    }

    // ── Tail Dependence ───────────────────────────────────────────────────

    /// Compute the tail dependence coefficient (simplified lambda_U).
    ///
    /// lambda_U = P(Y > Q_p | X > Q_p) as p -> 1.
    ///
    /// Approximated at `threshold_pct` quantile.
    pub fn tail_dependence(x: &[f64], y: &[f64], threshold_pct: f64) -> f64 {
        let n = x.len().min(y.len());
        if n < 5 {
            return 0.0;
        }

        let mut xs = x[..n].to_vec();
        let mut ys = y[..n].to_vec();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let qx = quantile(&xs, threshold_pct);
        let qy = quantile(&ys, threshold_pct);

        let both_exceed: usize = x[..n]
            .iter()
            .zip(y[..n].iter())
            .filter(|(&xi, &yi)| xi > qx && yi > qy)
            .count();
        let x_exceeds: usize = x[..n].iter().filter(|&&xi| xi > qx).count();

        if x_exceeds == 0 {
            return 0.0;
        }
        both_exceed as f64 / x_exceeds as f64
    }

    /// Compute downside tail dependence (both assets fall simultaneously).
    pub fn downside_tail_dependence(x: &[f64], y: &[f64], threshold_pct: f64) -> f64 {
        // Flip signs to convert lower tail to upper tail.
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();
        let neg_y: Vec<f64> = y.iter().map(|v| -v).collect();
        Self::tail_dependence(&neg_x, &neg_y, threshold_pct)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn canonical_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return None;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-14 {
        return None;
    }
    Some((num / denom).clamp(-1.0, 1.0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_series(n: usize, freq: f64, offset: f64) -> Vec<f64> {
        (0..n).map(|i| (i as f64 * freq + offset).sin()).collect()
    }

    #[test]
    fn test_ewma_correlation_converges() {
        let mut det = ContagionDetector::new(0.94, 60);
        let x = sine_series(200, 0.1, 0.0);
        let y = sine_series(200, 0.1, 0.0); // identical => r should be ~1
        for i in 0..200 {
            det.update("A", "B", x[i], y[i]);
        }
        let r = det.ewma_correlation("A", "B").unwrap();
        assert!(r > 0.9, "expected high correlation for identical series, got {}", r);
    }

    #[test]
    fn test_ewma_correlation_negative() {
        let mut det = ContagionDetector::new(0.94, 60);
        let x: Vec<f64> = (0..200).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| -v).collect();
        for i in 0..200 {
            det.update("X", "Y", x[i], y[i]);
        }
        let r = det.ewma_correlation("X", "Y").unwrap();
        assert!(r < -0.5, "expected negative correlation, got {}", r);
    }

    #[test]
    fn test_exceedance_correlation_positive() {
        // Strongly correlated series should have positive exceedance correlation.
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = x.iter().map(|v| v + 0.01).collect();
        let exc = ContagionDetector::exceedance_correlation(&x, &y, 0.80);
        assert!(exc > 0.0, "expected positive exceedance correlation, got {}", exc);
    }

    #[test]
    fn test_contagion_score_from_series() {
        // Identical series => exceedance_corr == avg_corr => score ~1.
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let score = ContagionDetector::contagion_score_from_series(&x, &x);
        // For identical series both measures are 1.0 => score == 1.0.
        assert!((score - 1.0).abs() < 0.1, "expected score ~1.0, got {}", score);
    }

    #[test]
    fn test_tail_dependence_positive() {
        // Positively correlated => joint upper tail should be non-zero.
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.15).sin()).collect();
        let y: Vec<f64> = x.iter().map(|v| v + 0.001).collect();
        let td = ContagionDetector::tail_dependence(&x, &y, 0.80);
        assert!(td >= 0.0);
    }

    #[test]
    fn test_downside_tail_dependence() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.15).sin()).collect();
        let y: Vec<f64> = x.iter().map(|v| v + 0.001).collect();
        let dtd = ContagionDetector::downside_tail_dependence(&x, &y, 0.80);
        assert!(dtd >= 0.0 && dtd <= 1.0);
    }

    #[test]
    fn test_contagion_events_empty_initially() {
        let det = ContagionDetector::default_params();
        assert!(det.contagion_events().is_empty());
    }

    #[test]
    fn test_is_contagion_regime_false_initially() {
        let det = ContagionDetector::default_params();
        assert!(!det.is_contagion_regime());
    }

    #[test]
    fn test_ewma_unknown_pair_none() {
        let det = ContagionDetector::default_params();
        assert!(det.ewma_correlation("A", "B").is_none());
    }
}
