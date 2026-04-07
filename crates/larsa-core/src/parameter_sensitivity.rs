/// Runtime parameter sensitivity analysis for SRFM.
///
/// Tracks how Sharpe varies with each parameter value over a rolling
/// 30-day window. Used by the IAE to prioritize parameter exploration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Rolling window in "days" (each record is one period).
const WINDOW_DAYS: usize = 30;

// ---------------------------------------------------------------------------
// ParamRecord
// ---------------------------------------------------------------------------

/// One observation of a parameter value and the resulting Sharpe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamRecord {
    pub param_name: String,
    pub param_value: f64,
    pub sharpe: f64,
    /// Unix timestamp in seconds.
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// OLS helpers
// ---------------------------------------------------------------------------

/// Simple OLS slope: sum((x-mx)(y-my)) / sum((x-mx)^2).
/// Returns None when there are fewer than two points or no x-variance.
fn ols_slope(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len();
    if n < 2 || ys.len() != n {
        return None;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let num: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|&x| (x - mx).powi(2)).sum();
    if den.abs() < 1e-12 {
        None
    } else {
        Some(num / den)
    }
}

/// Variance of a slice.
fn variance(vals: &[f64]) -> f64 {
    let n = vals.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = vals.iter().sum::<f64>() / nf;
    vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / nf
}

// ---------------------------------------------------------------------------
// ParameterWindow -- rolling buffer for one parameter
// ---------------------------------------------------------------------------

/// Rolling 30-day buffer of (param_value, sharpe) observations.
#[derive(Debug, Clone)]
struct ParameterWindow {
    records: std::collections::VecDeque<ParamRecord>,
}

impl ParameterWindow {
    fn new() -> Self {
        Self {
            records: std::collections::VecDeque::with_capacity(WINDOW_DAYS + 1),
        }
    }

    fn push(&mut self, rec: ParamRecord) {
        self.records.push_back(rec);
        if self.records.len() > WINDOW_DAYS {
            self.records.pop_front();
        }
    }

    fn values(&self) -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = self.records.iter().map(|r| r.param_value).collect();
        let ys: Vec<f64> = self.records.iter().map(|r| r.sharpe).collect();
        (xs, ys)
    }

    fn sharpes(&self) -> Vec<f64> {
        self.records.iter().map(|r| r.sharpe).collect()
    }
}

// ---------------------------------------------------------------------------
// SensitivityTracker
// ---------------------------------------------------------------------------

/// Tracks how Sharpe varies with parameter values across a rolling window.
///
/// For each parameter, stores a 30-observation window of (value, sharpe) pairs.
/// Provides:
/// - `sensitivity`: d(sharpe)/d(param) via OLS slope.
/// - `optimal_value`: the parameter value with the highest mean sharpe.
/// - `instability_score`: variance of sharpe across observed parameter values.
pub struct SensitivityTracker {
    windows: HashMap<String, ParameterWindow>,
}

impl SensitivityTracker {
    pub fn new() -> Self {
        Self {
            windows: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /// Record a parameter observation.
    pub fn record(&mut self, param_name: &str, value: f64, sharpe: f64) {
        self.record_with_ts(param_name, value, sharpe, 0);
    }

    /// Record a parameter observation with an explicit Unix timestamp.
    pub fn record_with_ts(
        &mut self,
        param_name: &str,
        value: f64,
        sharpe: f64,
        timestamp: i64,
    ) {
        let win = self
            .windows
            .entry(param_name.to_string())
            .or_insert_with(ParameterWindow::new);
        win.push(ParamRecord {
            param_name: param_name.to_string(),
            param_value: value,
            sharpe,
            timestamp,
        });
    }

    // -----------------------------------------------------------------------
    // Analytics
    // -----------------------------------------------------------------------

    /// OLS slope of sharpe w.r.t. parameter value: d(sharpe)/d(param).
    /// Returns None when there are fewer than 2 observations or no variance
    /// in the parameter values.
    pub fn sensitivity(&self, param_name: &str) -> Option<f64> {
        let win = self.windows.get(param_name)?;
        let (xs, ys) = win.values();
        ols_slope(&xs, &ys)
    }

    /// The observed parameter value associated with the highest mean sharpe
    /// in the current window.
    ///
    /// When multiple records share the same value, their sharpes are averaged.
    /// Returns None when no records exist for this parameter.
    pub fn optimal_value(&self, param_name: &str) -> Option<f64> {
        let win = self.windows.get(param_name)?;
        if win.records.is_empty() {
            return None;
        }

        // Group sharpes by (rounded) parameter value.
        let mut groups: HashMap<u64, (f64, usize)> = HashMap::new();
        for rec in &win.records {
            // Use a hash-safe key by quantizing to 6 decimal places.
            let key = (rec.param_value * 1_000_000.0).round() as u64;
            let entry = groups.entry(key).or_insert((0.0, 0));
            entry.0 += rec.sharpe;
            entry.1 += 1;
        }

        // Find the value with the highest mean sharpe.
        let mut best_val = win.records[0].param_value;
        let mut best_mean = f64::NEG_INFINITY;
        for rec in &win.records {
            let key = (rec.param_value * 1_000_000.0).round() as u64;
            if let Some(&(sum, count)) = groups.get(&key) {
                let mean = sum / count as f64;
                if mean > best_mean {
                    best_mean = mean;
                    best_val = rec.param_value;
                }
            }
        }
        Some(best_val)
    }

    /// Variance of sharpe across all observed values for this parameter.
    /// High variance means the Sharpe is unstable across parameter settings.
    pub fn instability_score(&self, param_name: &str) -> f64 {
        match self.windows.get(param_name) {
            None => 0.0,
            Some(win) => {
                let sharpes = win.sharpes();
                variance(&sharpes)
            }
        }
    }

    /// Return all current records for a parameter (newest last).
    pub fn records(&self, param_name: &str) -> Vec<&ParamRecord> {
        match self.windows.get(param_name) {
            None => vec![],
            Some(win) => win.records.iter().collect(),
        }
    }

    /// List all tracked parameter names.
    pub fn param_names(&self) -> Vec<&str> {
        self.windows.keys().map(|s| s.as_str()).collect()
    }

    /// Ranked list of parameters by instability (most unstable first).
    /// Used by the IAE to decide which parameters to explore next.
    pub fn ranked_by_instability(&self) -> Vec<(String, f64)> {
        let mut scores: Vec<(String, f64)> = self
            .windows
            .keys()
            .map(|name| (name.clone(), self.instability_score(name)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Serialize the full state to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        // Build a Vec<ParamRecord> from all windows for serialization.
        let all: Vec<&ParamRecord> = self
            .windows
            .values()
            .flat_map(|w| w.records.iter())
            .collect();
        serde_json::to_string(&all)
    }
}

impl Default for SensitivityTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_retrieve() {
        let mut tracker = SensitivityTracker::new();
        tracker.record("cf", 0.01, 1.2);
        tracker.record("cf", 0.02, 1.5);
        tracker.record("cf", 0.03, 1.1);
        let recs = tracker.records("cf");
        assert_eq!(recs.len(), 3);
    }

    #[test]
    fn test_sensitivity_positive_slope() {
        let mut tracker = SensitivityTracker::new();
        // Sharpe rises linearly with param: slope should be positive.
        for i in 1..=10 {
            tracker.record("bh_form", i as f64, i as f64 * 0.5);
        }
        let slope = tracker.sensitivity("bh_form").expect("slope must exist");
        assert!(slope > 0.0, "expected positive slope, got {slope}");
    }

    #[test]
    fn test_sensitivity_negative_slope() {
        let mut tracker = SensitivityTracker::new();
        for i in 1..=10 {
            tracker.record("bh_decay", i as f64 * 0.1, -(i as f64) * 0.3);
        }
        let slope = tracker.sensitivity("bh_decay").expect("slope must exist");
        assert!(slope < 0.0, "expected negative slope, got {slope}");
    }

    #[test]
    fn test_sensitivity_missing_param_returns_none() {
        let tracker = SensitivityTracker::new();
        assert!(tracker.sensitivity("nonexistent").is_none());
    }

    #[test]
    fn test_optimal_value_returns_best() {
        let mut tracker = SensitivityTracker::new();
        tracker.record("cf", 0.01, 0.5);
        tracker.record("cf", 0.02, 1.8); // best
        tracker.record("cf", 0.03, 0.3);
        let opt = tracker.optimal_value("cf").expect("optimal must exist");
        assert!((opt - 0.02).abs() < 1e-6, "expected 0.02 as optimal, got {opt}");
    }

    #[test]
    fn test_instability_zero_for_constant_sharpe() {
        let mut tracker = SensitivityTracker::new();
        for i in 0..5 {
            tracker.record("bh_form", i as f64 * 0.1, 1.5);
        }
        let instab = tracker.instability_score("bh_form");
        assert!(instab < 1e-9, "constant sharpe => zero instability, got {instab}");
    }

    #[test]
    fn test_instability_nonzero_for_varying_sharpe() {
        let mut tracker = SensitivityTracker::new();
        tracker.record("ctl_req", 3.0, -0.5);
        tracker.record("ctl_req", 5.0, 1.5);
        tracker.record("ctl_req", 7.0, 0.2);
        let instab = tracker.instability_score("ctl_req");
        assert!(instab > 0.0, "varying sharpe => positive instability");
    }

    #[test]
    fn test_rolling_window_evicts_old_records() {
        let mut tracker = SensitivityTracker::new();
        // Push WINDOW_DAYS + 5 records; only WINDOW_DAYS should remain.
        for i in 0..(WINDOW_DAYS + 5) {
            tracker.record("cf", i as f64 * 0.001, 1.0);
        }
        let recs = tracker.records("cf");
        assert_eq!(recs.len(), WINDOW_DAYS, "window must evict old records");
    }

    #[test]
    fn test_ranked_by_instability_ordering() {
        let mut tracker = SensitivityTracker::new();
        // "cf" has high instability, "bh_form" has low.
        for i in 0..5 {
            tracker.record("cf", i as f64, (i as f64 - 2.0) * 3.0);
            tracker.record("bh_form", i as f64, 1.0);
        }
        let ranked = tracker.ranked_by_instability();
        assert_eq!(ranked[0].0, "cf", "cf should be most unstable");
    }

    #[test]
    fn test_to_json_produces_valid_json() {
        let mut tracker = SensitivityTracker::new();
        tracker.record("cf", 0.02, 1.1);
        tracker.record_with_ts("cf", 0.03, 1.3, 1_700_000_000);
        let json = tracker.to_json().expect("serialization must succeed");
        // Must be a valid JSON array.
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_ols_slope_flat_line_returns_none() {
        // All x values are identical -- no x-variance.
        let xs = vec![1.0, 1.0, 1.0];
        let ys = vec![0.5, 1.0, 1.5];
        assert!(ols_slope(&xs, &ys).is_none());
    }
}
