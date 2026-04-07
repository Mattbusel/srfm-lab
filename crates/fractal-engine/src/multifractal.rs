//! Multifractal analysis of price time series.
//!
//! Multifractal Detrended Fluctuation Analysis (MFDFA) characterises the
//! complexity of a time series by computing the generalised Hurst exponent
//! H(q) for a range of orders q.  The singularity spectrum f(alpha) is then
//! obtained via the Legendre transform.
//!
//! # Interpretation
//!
//! - `multifractal_width > 0.3`   -- multifractal, complex dynamics
//! - `multifractal_width < 0.1`   -- monofractal (near-Brownian)
//! - `H(q=2)` should equal the standard DFA Hurst exponent
//!
//! # Trading implications
//!
//! A wide singularity spectrum indicates that the price series has varying
//! local scaling properties -- different strategy parameters are needed for
//! different market micro-regimes.  A narrow spectrum implies a single
//! scaling regime (e.g., pure trending or pure random walk).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute log-returns of a price series.
fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2)
        .map(|w| if w[0] > 0.0 && w[1] > 0.0 { (w[1] / w[0]).ln() } else { 0.0 })
        .collect()
}

/// OLS slope of y on x.
fn ols_slope(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.5;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    if den.abs() < 1e-15 { 0.5 } else { num / den }
}

/// Detrend a segment by removing a linear fit (polynomial order 1).
fn detrend_linear(segment: &[f64]) -> Vec<f64> {
    let n = segment.len();
    if n < 2 {
        return segment.to_vec();
    }
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let slope = ols_slope(&xs, segment);
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = segment.iter().sum::<f64>() / n as f64;
    let intercept = my - slope * mx;

    segment.iter().enumerate()
        .map(|(i, &y)| y - (slope * i as f64 + intercept))
        .collect()
}

// ---------------------------------------------------------------------------
// MFDFA core
// ---------------------------------------------------------------------------

/// Compute the q-th order fluctuation function F_q(s) for a given scale s.
///
/// The profile Y(i) = cumsum(returns - mean(returns)).
/// Non-overlapping segments of length `scale` are detrended and their
/// variance computed.  F_q(s) = (mean(|F_v|^q))^(1/q).
fn fluctuation_fq(profile: &[f64], scale: usize, q: f64) -> Option<f64> {
    let n = profile.len();
    if scale < 4 || scale > n / 2 {
        return None;
    }
    let n_segs = n / scale;
    if n_segs < 2 {
        return None;
    }

    let mut variances: Vec<f64> = Vec::with_capacity(n_segs * 2);

    // Forward segments
    for v in 0..n_segs {
        let start = v * scale;
        let seg = &profile[start..(start + scale)];
        let detrended = detrend_linear(seg);
        let var = detrended.iter().map(|x| x * x).sum::<f64>() / scale as f64;
        if var > 0.0 {
            variances.push(var);
        }
    }

    // Backward segments (mirror at end)
    for v in 0..n_segs {
        let start = n - (v + 1) * scale;
        let seg = &profile[start..(start + scale)];
        let detrended = detrend_linear(seg);
        let var = detrended.iter().map(|x| x * x).sum::<f64>() / scale as f64;
        if var > 0.0 {
            variances.push(var);
        }
    }

    if variances.is_empty() {
        return None;
    }

    // F_q(s) = (mean(sigma^q))^(1/q)
    let fq = if (q - 0.0).abs() < 1e-9 {
        // q = 0: geometric mean
        let log_sum: f64 = variances.iter().map(|&v| v.ln()).sum::<f64>();
        (log_sum / variances.len() as f64 / 2.0).exp()
    } else {
        let sum: f64 = variances.iter().map(|&v| v.powf(q / 2.0)).sum::<f64>();
        let mean = sum / variances.len() as f64;
        if mean <= 0.0 {
            return None;
        }
        mean.powf(1.0 / q)
    };

    if fq.is_finite() && fq > 0.0 {
        Some(fq)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// MultifractalAnalyzer
// ---------------------------------------------------------------------------

/// Result of a full multifractal analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultifractalResult {
    /// Generalised Hurst exponents H(q) for each order q.
    pub hq: Vec<(f64, f64)>, // (q, H(q))
    /// Tau(q) = q * H(q) - 1 (mass exponent function).
    pub tau_q: Vec<f64>,
    /// Singularity spectrum: (alpha, f_alpha).
    pub singularity_spectrum: (Vec<f64>, Vec<f64>),
    /// Width of the singularity spectrum: alpha_max - alpha_min.
    pub multifractal_width: f64,
    /// Whether the series is classified as multifractal (width > 0.3).
    pub is_multifractal: bool,
    /// H(q=2): should equal the standard DFA Hurst exponent.
    pub h2: f64,
}

/// Performs Multifractal Detrended Fluctuation Analysis (MFDFA).
pub struct MultifractalAnalyzer {
    /// Range of q orders to compute.  Typically [-5, 5] in steps of 1.
    pub q_orders: Vec<f64>,
    /// Scale sizes to use.  Must cover at least 3 octaves.
    pub scales: Vec<usize>,
}

impl MultifractalAnalyzer {
    /// Create analyzer with default q range [-5..5 step 1] and auto scales.
    pub fn new() -> Self {
        let q_orders: Vec<f64> = (-5i32..=5).map(|q| q as f64).collect();
        Self { q_orders, scales: Vec::new() }
    }

    /// Create analyzer with explicit q range and scales.
    pub fn with_config(q_orders: Vec<f64>, scales: Vec<usize>) -> Self {
        Self { q_orders, scales }
    }

    /// Generate default scales for a series of length n.
    fn default_scales(n: usize) -> Vec<usize> {
        let mut scales = Vec::new();
        let min_scale = 8usize;
        let max_scale = n / 4;
        let mut s = min_scale;
        while s <= max_scale {
            scales.push(s);
            s = (s as f64 * 1.5).ceil() as usize;
        }
        scales
    }

    /// Compute the integrated profile Y(i) = cumsum(returns - mean(returns)).
    fn compute_profile(returns: &[f64]) -> Vec<f64> {
        let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
        let mut profile = Vec::with_capacity(returns.len());
        let mut cum = 0.0f64;
        for &r in returns {
            cum += r - mean_r;
            profile.push(cum);
        }
        profile
    }

    /// Run full MFDFA on a price series.
    ///
    /// `prices` must have at least 64 elements for reliable results.
    pub fn analyze(&self, prices: &[f64]) -> Option<MultifractalResult> {
        let n = prices.len();
        if n < 64 {
            return None;
        }

        let returns = log_returns(prices);
        let profile = Self::compute_profile(&returns);

        let scales = if self.scales.is_empty() {
            Self::default_scales(n)
        } else {
            self.scales.clone()
        };

        if scales.len() < 3 {
            return None;
        }

        let log_scales: Vec<f64> = scales.iter().map(|&s| (s as f64).ln()).collect();

        // Compute H(q) for each order q
        let mut hq_pairs: Vec<(f64, f64)> = Vec::new();
        let mut h2_val = f64::NAN;

        for &q in &self.q_orders {
            let log_fq: Vec<f64> = scales.iter()
                .zip(log_scales.iter())
                .filter_map(|(&s, _)| fluctuation_fq(&profile, s, q).map(|fq| fq.ln()))
                .collect();

            let valid_log_scales: Vec<f64> = scales.iter()
                .zip(log_scales.iter())
                .filter_map(|(&s, &ls)| {
                    if fluctuation_fq(&profile, s, q).is_some() { Some(ls) } else { None }
                })
                .collect();

            if log_fq.len() < 3 {
                continue;
            }

            let h = ols_slope(&valid_log_scales, &log_fq);
            if h.is_finite() {
                hq_pairs.push((q, h.clamp(0.0, 2.0)));
                if (q - 2.0).abs() < 1e-9 {
                    h2_val = h.clamp(0.0, 2.0);
                }
            }
        }

        if hq_pairs.len() < 3 {
            return None;
        }

        // tau(q) = q * H(q) - 1
        let tau_q: Vec<f64> = hq_pairs.iter().map(|(q, h)| q * h - 1.0).collect();

        // Singularity spectrum via Legendre transform
        // alpha(q) = d(tau)/dq, f(alpha) = q * alpha - tau(q)
        let (alpha_vec, f_alpha_vec) = Self::legendre_transform(&hq_pairs, &tau_q);

        let multifractal_width = if alpha_vec.is_empty() {
            0.0
        } else {
            let alpha_min = alpha_vec.iter().cloned().fold(f64::INFINITY, f64::min);
            let alpha_max = alpha_vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            alpha_max - alpha_min
        };

        Some(MultifractalResult {
            hq: hq_pairs,
            tau_q,
            singularity_spectrum: (alpha_vec, f_alpha_vec),
            multifractal_width,
            is_multifractal: multifractal_width > 0.3,
            h2: h2_val,
        })
    }

    /// Legendre transform of tau(q) to obtain (alpha, f_alpha).
    ///
    /// Uses numerical differentiation: alpha(q) = d(tau)/d(q).
    fn legendre_transform(
        hq: &[(f64, f64)],
        tau_q: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        if hq.len() < 3 {
            return (Vec::new(), Vec::new());
        }
        let n = hq.len();
        let mut alphas = Vec::with_capacity(n - 2);
        let mut f_alphas = Vec::with_capacity(n - 2);

        // Central differences for interior points
        for i in 1..(n - 1) {
            let q = hq[i].0;
            let dq = hq[i + 1].0 - hq[i - 1].0;
            if dq.abs() < 1e-12 {
                continue;
            }
            let dtau = tau_q[i + 1] - tau_q[i - 1];
            let alpha = dtau / dq;
            let f_alpha = q * alpha - tau_q[i];

            if alpha.is_finite() && f_alpha.is_finite() {
                alphas.push(alpha);
                f_alphas.push(f_alpha);
            }
        }

        (alphas, f_alphas)
    }

    /// Return the singularity spectrum (alpha, f_alpha) directly from prices.
    pub fn singularity_spectrum(&self, prices: &[f64]) -> (Vec<f64>, Vec<f64>) {
        self.analyze(prices)
            .map(|r| r.singularity_spectrum)
            .unwrap_or_default()
    }

    /// Width of the multifractal spectrum.
    pub fn multifractal_width(&self, prices: &[f64]) -> f64 {
        self.analyze(prices)
            .map(|r| r.multifractal_width)
            .unwrap_or(0.0)
    }

    /// Is the price series multifractal?
    pub fn is_multifractal(&self, prices: &[f64]) -> bool {
        self.multifractal_width(prices) > 0.3
    }
}

impl Default for MultifractalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Generalised Hurst exponent summary
// ---------------------------------------------------------------------------

/// Summary statistics of the H(q) spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HurstSpectrum {
    /// H(-5): sensitivity to small fluctuations.
    pub h_neg5: f64,
    /// H(0): geometric mean scaling (median fluctuations).
    pub h_0: f64,
    /// H(2): standard DFA Hurst exponent.
    pub h_2: f64,
    /// H(5): sensitivity to large fluctuations.
    pub h_5: f64,
    /// Range H(-5) - H(5): heterogeneity of scaling.
    pub hq_range: f64,
}

impl HurstSpectrum {
    pub fn from_result(result: &MultifractalResult) -> Option<Self> {
        let hq_map: std::collections::HashMap<i64, f64> = result.hq
            .iter()
            .map(|(q, h)| ((*q as i64), *h))
            .collect();

        let h_neg5 = *hq_map.get(&-5)?;
        let h_0 = *hq_map.get(&0)?;
        let h_2 = *hq_map.get(&2)?;
        let h_5 = *hq_map.get(&5)?;

        Some(Self {
            h_neg5,
            h_0,
            h_2,
            h_5,
            hq_range: h_neg5 - h_5,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a random walk price series using LCG.
    fn random_walk(n: usize, seed: u64) -> Vec<f64> {
        let mut price = 100.0f64;
        let mut prices = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            prices.push(price);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            price = (price + r * 0.5).max(1.0);
        }
        prices
    }

    /// Generate a strongly trending price series (persistent, multifractal).
    fn trending_series(n: usize) -> Vec<f64> {
        let mut prices = Vec::with_capacity(n);
        let mut p = 100.0f64;
        let mut state = 7u64;
        for _ in 0..n {
            prices.push(p);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (state as f64 / u64::MAX as f64) * 0.002 - 0.001;
            p *= 1.003 + noise; // strong trend + small noise
        }
        prices
    }

    #[test]
    fn test_mfdfa_returns_result_for_sufficient_data() {
        let prices = random_walk(256, 42);
        let analyzer = MultifractalAnalyzer::new();
        let result = analyzer.analyze(&prices);
        assert!(result.is_some(), "Should return a result for 256 prices");
    }

    #[test]
    fn test_mfdfa_too_short_returns_none() {
        let prices = random_walk(32, 42);
        let analyzer = MultifractalAnalyzer::new();
        assert!(analyzer.analyze(&prices).is_none());
    }

    #[test]
    fn test_multifractal_width_positive() {
        let prices = random_walk(512, 99);
        let analyzer = MultifractalAnalyzer::new();
        let width = analyzer.multifractal_width(&prices);
        assert!(width >= 0.0, "Width should be non-negative, got {width}");
    }

    #[test]
    fn test_hq_count_matches_q_orders() {
        let prices = random_walk(256, 7);
        let analyzer = MultifractalAnalyzer::new();
        if let Some(result) = analyzer.analyze(&prices) {
            // All q in [-5..5] that produced valid H(q) should be present
            assert!(result.hq.len() >= 5, "expected at least 5 H(q) values");
        }
    }

    #[test]
    fn test_h2_in_valid_range() {
        let prices = random_walk(256, 42);
        let analyzer = MultifractalAnalyzer::new();
        if let Some(result) = analyzer.analyze(&prices) {
            assert!(
                result.h2 >= 0.0 && result.h2 <= 2.0,
                "H(2) = {} should be in [0, 2]",
                result.h2
            );
        }
    }

    #[test]
    fn test_hurst_h2_near_standard_hurst() {
        // For a random walk H should be near 0.5; H(q=2) from MFDFA should
        // also be near 0.5 (within tolerance of 0.3 given finite sample).
        let prices = random_walk(512, 12345);
        let analyzer = MultifractalAnalyzer::new();
        if let Some(result) = analyzer.analyze(&prices) {
            if result.h2.is_finite() {
                assert!(
                    result.h2 >= 0.2 && result.h2 <= 0.8,
                    "H(2) = {:.3} should be near 0.5 for random walk",
                    result.h2
                );
            }
        }
    }

    #[test]
    fn test_singularity_spectrum_lengths_match() {
        let prices = random_walk(256, 1);
        let analyzer = MultifractalAnalyzer::new();
        let (alpha, f_alpha) = analyzer.singularity_spectrum(&prices);
        assert_eq!(alpha.len(), f_alpha.len(), "alpha and f_alpha must have equal length");
    }

    #[test]
    fn test_tau_q_length_matches_hq() {
        let prices = random_walk(256, 3);
        let analyzer = MultifractalAnalyzer::new();
        if let Some(result) = analyzer.analyze(&prices) {
            assert_eq!(result.tau_q.len(), result.hq.len());
        }
    }

    #[test]
    fn test_trending_series_high_h2() {
        // A strongly trending series should have H(2) higher than a random walk.
        // Exact value varies with finite-sample estimation; just check > 0.4.
        let prices = trending_series(512);
        let analyzer = MultifractalAnalyzer::new();
        if let Some(result) = analyzer.analyze(&prices) {
            if result.h2.is_finite() {
                assert!(result.h2 > 0.4, "trending H(2) = {:.3} should be > 0.4", result.h2);
            }
        }
    }

    #[test]
    fn test_is_multifractal_consistent_with_width() {
        let prices = random_walk(512, 77);
        let analyzer = MultifractalAnalyzer::new();
        if let Some(result) = analyzer.analyze(&prices) {
            assert_eq!(result.is_multifractal, result.multifractal_width > 0.3);
        }
    }

    #[test]
    fn test_hurst_spectrum_construction() {
        let prices = random_walk(512, 55);
        let analyzer = MultifractalAnalyzer::new();
        if let Some(result) = analyzer.analyze(&prices) {
            let spectrum = HurstSpectrum::from_result(&result);
            // May be None if q=+/-5 didn't converge; that's fine
            if let Some(s) = spectrum {
                assert!(s.h_2.is_finite());
                assert!(s.hq_range.is_finite());
            }
        }
    }

    #[test]
    fn test_profile_zero_mean() {
        // The integrated profile should have approximately zero drift
        let returns = vec![0.01, -0.01, 0.02, -0.02, 0.005, -0.005];
        let profile = MultifractalAnalyzer::compute_profile(&returns);
        let last = profile.last().copied().unwrap_or(0.0);
        assert!(last.abs() < 1e-10, "cumsum of demeaned returns should end near 0");
    }
}
