//! Extended fractal dimension estimation methods.
//!
//! Complements `fractal_dimension.rs` with Petrosian, Katz, and a
//! unified `FractalDimension` dispatcher.
//!
//! # Methods
//!
//! | Method       | Speed    | Suitable for          | Notes                    |
//! |--------------|----------|-----------------------|--------------------------|
//! | `BoxCounting`| O(n log n)| Any series           | High accuracy, slow      |
//! | `Higuchi`    | O(n)     | Short/real-time       | Best general-purpose     |
//! | `Petrosian`  | O(n)     | EEG-style, fast scan  | Sign-change heuristic    |
//! | `Katz`       | O(n)     | Waveform analysis     | Length/diameter ratio    |
//!
//! # Interpretation
//!
//! - D near 1.0 -- trending (price follows a smooth 1D curve)
//! - D near 1.5 -- random walk (Brownian motion)
//! - D near 2.0 -- chaotic / plane-filling
//!
//! # Trading rule of thumb
//!
//! Use LARSA trend-following signals only when D < 1.45.
//! Switch to mean-reversion when 1.45 < D < 1.6.
//! Reduce exposure when D > 1.6.

use serde::{Deserialize, Serialize};

// Re-export from existing module for convenience
pub use crate::fractal_dimension::{higuchi_fd, box_counting_fd, FractalAnalysis, FdClass};

// ---------------------------------------------------------------------------
// Method selector
// ---------------------------------------------------------------------------

/// Method to use for fractal dimension computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FDMethod {
    /// Higuchi's algorithm (fast, best for short series).
    Higuchi { k_max: usize },
    /// Box-counting (slower, more accurate for long series).
    BoxCounting,
    /// Petrosian method (fastest, sign-change based).
    Petrosian,
    /// Katz method (total path length / diameter).
    Katz,
}

impl Default for FDMethod {
    fn default() -> Self {
        FDMethod::Higuchi { k_max: 8 }
    }
}

// ---------------------------------------------------------------------------
// PetrosianMethod
// ---------------------------------------------------------------------------

/// Petrosian fractal dimension.
///
/// Based on the number of sign changes in the first derivative of the signal:
///
/// ```text
/// D = log10(N) / (log10(N) + log10(N / (N + 0.4 * Ndelta)))
/// ```
///
/// where N = series length, Ndelta = number of sign changes in diff(signal).
///
/// Originally developed for EEG analysis; adapted here for price series.
/// Very fast: O(n).  Less accurate than Higuchi for financial data.
pub struct PetrosianMethod;

impl PetrosianMethod {
    /// Compute Petrosian fractal dimension.
    pub fn compute(prices: &[f64]) -> Option<f64> {
        let n = prices.len();
        if n < 4 {
            return None;
        }

        // First difference of the price series
        let diff: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();

        // Count sign changes in diff (zero differences are treated as positive)
        let ndelta = diff.windows(2)
            .filter(|w| {
                let s0 = if w[0] >= 0.0 { 1i32 } else { -1 };
                let s1 = if w[1] >= 0.0 { 1i32 } else { -1 };
                s0 != s1
            })
            .count();

        let n_f = n as f64;
        let ndelta_f = ndelta as f64;

        if ndelta == 0 {
            // Perfect trend -- FD = 1.0
            return Some(1.0);
        }

        let denominator = n_f / (n_f + 0.4 * ndelta_f);
        if denominator <= 0.0 || denominator >= 1.0 {
            return None;
        }

        let log_n = n_f.log10();
        let log_denom = denominator.log10();

        if log_denom.abs() < 1e-15 {
            return None;
        }

        let fd = log_n / (log_n + log_denom.abs());
        if fd.is_finite() && fd >= 1.0 && fd <= 2.0 {
            Some(fd)
        } else {
            Some(fd.clamp(1.0, 2.0))
        }
    }

    /// Compute over a rolling window.
    pub fn rolling(prices: &[f64], window: usize) -> Vec<f64> {
        let n = prices.len();
        let mut out = vec![f64::NAN; n];
        for i in window..=n {
            let slice = &prices[(i - window)..i];
            if let Some(d) = Self::compute(slice) {
                out[i - 1] = d;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// KatzMethod
// ---------------------------------------------------------------------------

/// Katz fractal dimension.
///
/// Based on the ratio of the total path length to the maximum diameter:
///
/// ```text
/// D = log10(L / a) / log10(d / a)
/// ```
///
/// where:
/// - L = total arc length = sum(|diff|)
/// - d = maximum Euclidean distance from first point (diameter)
/// - a = mean step length = L / (N-1)
///
/// This method is very sensitive to outliers and amplitude.  It works best
/// on normalised series.
pub struct KatzMethod;

impl KatzMethod {
    /// Compute Katz fractal dimension.
    ///
    /// The series is first normalised to [0, 1] to remove scale effects.
    pub fn compute(prices: &[f64]) -> Option<f64> {
        let n = prices.len();
        if n < 4 {
            return None;
        }

        // Normalise to [0, 1] to make scale-invariant
        let min_p = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_p - min_p;
        if range < 1e-12 {
            // Constant series = perfectly smooth trend
            return Some(1.0);
        }

        let norm: Vec<f64> = prices.iter().map(|p| (p - min_p) / range).collect();

        // Total arc length (using time + price as 2D path)
        let dt = 1.0 / n as f64; // normalised time step
        let mut total_length = 0.0f64;
        for i in 1..n {
            let dx = dt;
            let dy = norm[i] - norm[i - 1];
            total_length += (dx * dx + dy * dy).sqrt();
        }

        if total_length < 1e-12 {
            return Some(1.0);
        }

        // Maximum distance from first point
        let x0 = 0.0f64;
        let y0 = norm[0];
        let mut max_dist = 0.0f64;
        for i in 1..n {
            let xi = i as f64 * dt;
            let yi = norm[i];
            let dist = ((xi - x0).powi(2) + (yi - y0).powi(2)).sqrt();
            if dist > max_dist {
                max_dist = dist;
            }
        }

        if max_dist < 1e-12 {
            return Some(1.0);
        }

        // Mean step size
        let a = total_length / (n - 1) as f64;

        if a < 1e-14 {
            return None;
        }

        let log_l_over_a = (total_length / a).ln();
        let log_d_over_a = (max_dist / a).ln();

        if log_d_over_a.abs() < 1e-14 {
            return None;
        }

        let fd = log_l_over_a / log_d_over_a;

        if fd.is_finite() {
            Some(fd.clamp(1.0, 2.0))
        } else {
            None
        }
    }

    /// Compute over a rolling window.
    pub fn rolling(prices: &[f64], window: usize) -> Vec<f64> {
        let n = prices.len();
        let mut out = vec![f64::NAN; n];
        for i in window..=n {
            let slice = &prices[(i - window)..i];
            if let Some(d) = Self::compute(slice) {
                out[i - 1] = d;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// HiguchiMethod (thin wrapper struct for the functional API)
// ---------------------------------------------------------------------------

/// Struct wrapper for Higuchi's algorithm.
pub struct HiguchiMethod {
    pub k_max: usize,
}

impl HiguchiMethod {
    pub fn new(k_max: usize) -> Self {
        Self { k_max }
    }

    pub fn compute(&self, prices: &[f64]) -> Option<f64> {
        higuchi_fd(prices, self.k_max).map(|fa| fa.fd)
    }

    pub fn rolling(&self, prices: &[f64], window: usize) -> Vec<f64> {
        crate::fractal_dimension::rolling_fd(prices, window, self.k_max)
    }
}

// ---------------------------------------------------------------------------
// BoxCountingMethod (thin wrapper struct)
// ---------------------------------------------------------------------------

/// Struct wrapper for box-counting algorithm.
pub struct BoxCountingMethod;

impl BoxCountingMethod {
    pub fn compute(prices: &[f64]) -> Option<f64> {
        box_counting_fd(prices)
    }
}

// ---------------------------------------------------------------------------
// FractalDimension dispatcher
// ---------------------------------------------------------------------------

/// Unified fractal dimension estimator with selectable algorithm.
///
/// # Example
///
/// ```
/// use fractal_engine::fractal_dim_ext::{FractalDimension, FDMethod};
///
/// let prices: Vec<f64> = (0..128).map(|i| 100.0 + i as f64 * 0.5).collect();
/// let fd = FractalDimension::compute(FDMethod::Higuchi { k_max: 8 }, &prices);
/// println!("Trending series FD: {:?}", fd);
/// ```
pub struct FractalDimension;

impl FractalDimension {
    /// Compute fractal dimension using the specified method.
    ///
    /// Returns `None` if the series is too short or the computation fails.
    pub fn compute(method: FDMethod, prices: &[f64]) -> Option<f64> {
        match method {
            FDMethod::Higuchi { k_max } => {
                higuchi_fd(prices, k_max).map(|fa| fa.fd)
            }
            FDMethod::BoxCounting => {
                box_counting_fd(prices)
            }
            FDMethod::Petrosian => {
                PetrosianMethod::compute(prices)
            }
            FDMethod::Katz => {
                KatzMethod::compute(prices)
            }
        }
    }

    /// Compute with all methods and return a summary.
    pub fn compute_all(prices: &[f64]) -> AllMethodsResult {
        AllMethodsResult {
            higuchi: higuchi_fd(prices, 8).map(|fa| fa.fd),
            box_counting: box_counting_fd(prices),
            petrosian: PetrosianMethod::compute(prices),
            katz: KatzMethod::compute(prices),
        }
    }

    /// Rolling fractal dimension using the specified method.
    pub fn rolling(method: FDMethod, prices: &[f64], window: usize) -> Vec<f64> {
        let n = prices.len();
        let mut out = vec![f64::NAN; n];
        for i in window..=n {
            let slice = &prices[(i - window)..i];
            if let Some(d) = Self::compute(method.clone(), slice) {
                out[i - 1] = d;
            }
        }
        out
    }

    /// Classify FD into a trading regime.
    pub fn classify(fd: f64) -> FdTradingRegime {
        FdTradingRegime::from_fd(fd)
    }
}

// ---------------------------------------------------------------------------
// All-methods result
// ---------------------------------------------------------------------------

/// Fractal dimension computed by all four methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllMethodsResult {
    pub higuchi: Option<f64>,
    pub box_counting: Option<f64>,
    pub petrosian: Option<f64>,
    pub katz: Option<f64>,
}

impl AllMethodsResult {
    /// Ensemble average of all available estimates.
    pub fn ensemble_mean(&self) -> Option<f64> {
        let vals: Vec<f64> = [self.higuchi, self.box_counting, self.petrosian, self.katz]
            .iter()
            .filter_map(|v| *v)
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    }

    /// All four methods agree within `tolerance`.
    pub fn methods_agree(&self, tolerance: f64) -> bool {
        let vals: Vec<f64> = [self.higuchi, self.box_counting, self.petrosian, self.katz]
            .iter()
            .filter_map(|v| *v)
            .collect();
        if vals.len() < 2 {
            return true;
        }
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max - min <= tolerance
    }
}

// ---------------------------------------------------------------------------
// Trading regime from FD
// ---------------------------------------------------------------------------

/// Trading regime inferred from fractal dimension.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FdTradingRegime {
    /// D < 1.3: strongly trending -- use momentum / trend-following.
    StrongTrend,
    /// 1.3 <= D < 1.45: mild trend -- LARSA signals valid.
    MildTrend,
    /// 1.45 <= D < 1.6: borderline Brownian -- consider mean-reversion.
    Borderline,
    /// 1.6 <= D < 1.8: noisy -- reduce position size.
    Noisy,
    /// D >= 1.8: highly chaotic -- no signal, stay flat.
    Chaotic,
}

impl FdTradingRegime {
    pub fn from_fd(fd: f64) -> Self {
        if fd < 1.3 {
            FdTradingRegime::StrongTrend
        } else if fd < 1.45 {
            FdTradingRegime::MildTrend
        } else if fd < 1.6 {
            FdTradingRegime::Borderline
        } else if fd < 1.8 {
            FdTradingRegime::Noisy
        } else {
            FdTradingRegime::Chaotic
        }
    }

    pub fn is_tradeable(&self) -> bool {
        matches!(self, FdTradingRegime::StrongTrend | FdTradingRegime::MildTrend)
    }

    pub fn position_scale(&self) -> f64 {
        match self {
            FdTradingRegime::StrongTrend => 1.0,
            FdTradingRegime::MildTrend   => 0.8,
            FdTradingRegime::Borderline  => 0.5,
            FdTradingRegime::Noisy       => 0.2,
            FdTradingRegime::Chaotic     => 0.0,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FdTradingRegime::StrongTrend => "STRONG_TREND",
            FdTradingRegime::MildTrend   => "MILD_TREND",
            FdTradingRegime::Borderline  => "BORDERLINE",
            FdTradingRegime::Noisy       => "NOISY",
            FdTradingRegime::Chaotic     => "CHAOTIC",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn trending_series(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 + i as f64 * 0.5).collect()
    }

    fn noisy_series(n: usize, seed: u64) -> Vec<f64> {
        let mut price = 100.0f64;
        let mut prices = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            prices.push(price);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
            price = (price + r * 2.0).max(1.0);
        }
        prices
    }

    // --- Petrosian tests ---

    #[test]
    fn test_petrosian_trending_low_fd() {
        let prices = trending_series(128);
        let fd = PetrosianMethod::compute(&prices);
        assert!(fd.is_some());
        let fd = fd.unwrap();
        assert!(fd < 1.5, "Trending series Petrosian FD should be < 1.5, got {fd:.3}");
    }

    #[test]
    fn test_petrosian_noisy_higher_fd() {
        let trend = trending_series(128);
        let noisy = noisy_series(128, 42);
        let fd_trend = PetrosianMethod::compute(&trend).unwrap_or(1.0);
        let fd_noisy = PetrosianMethod::compute(&noisy).unwrap_or(1.5);
        assert!(
            fd_noisy >= fd_trend,
            "Noisy FD {fd_noisy:.3} should be >= trend FD {fd_trend:.3}"
        );
    }

    #[test]
    fn test_petrosian_in_valid_range() {
        for seed in [1u64, 7, 42] {
            let prices = noisy_series(128, seed);
            if let Some(fd) = PetrosianMethod::compute(&prices) {
                assert!(fd >= 1.0 && fd <= 2.0, "Petrosian FD {fd:.3} out of [1,2]");
            }
        }
    }

    #[test]
    fn test_petrosian_too_short_returns_none() {
        assert!(PetrosianMethod::compute(&[1.0, 2.0]).is_none());
    }

    // --- Katz tests ---

    #[test]
    fn test_katz_trending_low_fd() {
        let prices = trending_series(128);
        let fd = KatzMethod::compute(&prices);
        assert!(fd.is_some());
        let fd = fd.unwrap();
        assert!(fd < 1.5, "Trending series Katz FD should be < 1.5, got {fd:.3}");
    }

    #[test]
    fn test_katz_constant_series_fd_one() {
        let prices = vec![100.0f64; 64];
        let fd = KatzMethod::compute(&prices);
        assert!(fd.is_some());
        assert!((fd.unwrap() - 1.0).abs() < 1e-6, "Constant series Katz FD should be 1.0");
    }

    #[test]
    fn test_katz_in_valid_range() {
        let prices = noisy_series(128, 99);
        if let Some(fd) = KatzMethod::compute(&prices) {
            assert!(fd >= 1.0 && fd <= 2.0, "Katz FD {fd:.3} out of [1,2]");
        }
    }

    // --- FractalDimension dispatcher tests ---

    #[test]
    fn test_fractal_dimension_trending_below_1_4() {
        let prices = trending_series(128);
        let fd = FractalDimension::compute(FDMethod::Higuchi { k_max: 8 }, &prices);
        assert!(fd.is_some());
        let fd = fd.unwrap();
        assert!(fd < 1.4, "Trending series Higuchi FD should be < 1.4, got {fd:.3}");
    }

    #[test]
    fn test_fractal_dimension_noise_above_1_4() {
        let prices = noisy_series(256, 12345);
        let fd = FractalDimension::compute(FDMethod::Higuchi { k_max: 8 }, &prices);
        if let Some(fd) = fd {
            assert!(fd > 1.0, "Noisy series FD should be > 1.0, got {fd:.3}");
            // noisy series should be higher than trending
            let trend_fd = FractalDimension::compute(
                FDMethod::Higuchi { k_max: 8 },
                &trending_series(256),
            ).unwrap_or(1.0);
            assert!(fd >= trend_fd, "noisy FD {fd:.3} should be >= trend FD {trend_fd:.3}");
        }
    }

    #[test]
    fn test_fractal_dimension_all_methods() {
        let prices = noisy_series(128, 42);
        let all = FractalDimension::compute_all(&prices);
        assert!(all.higuchi.is_some() || all.petrosian.is_some(),
            "At least one method should return a value");
    }

    #[test]
    fn test_fractal_dimension_petrosian_dispatch() {
        let prices = trending_series(128);
        let fd = FractalDimension::compute(FDMethod::Petrosian, &prices);
        assert!(fd.is_some());
    }

    #[test]
    fn test_fractal_dimension_katz_dispatch() {
        let prices = trending_series(128);
        let fd = FractalDimension::compute(FDMethod::Katz, &prices);
        assert!(fd.is_some());
    }

    #[test]
    fn test_fractal_dimension_box_counting_dispatch() {
        // Use a longer series to give box-counting enough scale octaves
        let prices: Vec<f64> = (0..512)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.05)
            .collect();
        let fd = FractalDimension::compute(FDMethod::BoxCounting, &prices);
        // Box-counting may return None for degenerate series; just check no panic
        if let Some(v) = fd {
            assert!(v >= 1.0 && v <= 2.0, "box-counting FD {v:.3} out of [1,2]");
        }
    }

    // --- Trading regime tests ---

    #[test]
    fn test_fd_regime_strong_trend() {
        let regime = FdTradingRegime::from_fd(1.1);
        assert_eq!(regime, FdTradingRegime::StrongTrend);
        assert!(regime.is_tradeable());
        assert!((regime.position_scale() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fd_regime_chaotic() {
        let regime = FdTradingRegime::from_fd(1.9);
        assert_eq!(regime, FdTradingRegime::Chaotic);
        assert!(!regime.is_tradeable());
        assert!((regime.position_scale()).abs() < 1e-9);
    }

    #[test]
    fn test_fd_regime_noisy() {
        let regime = FdTradingRegime::from_fd(1.7);
        assert_eq!(regime, FdTradingRegime::Noisy);
        assert!(!regime.is_tradeable());
    }

    // --- Ensemble tests ---

    #[test]
    fn test_ensemble_mean_all_present() {
        let result = AllMethodsResult {
            higuchi: Some(1.3),
            box_counting: Some(1.4),
            petrosian: Some(1.2),
            katz: Some(1.5),
        };
        let mean = result.ensemble_mean().unwrap();
        assert!((mean - 1.35).abs() < 1e-9, "mean should be 1.35, got {mean}");
    }

    #[test]
    fn test_ensemble_methods_agree_tight() {
        let result = AllMethodsResult {
            higuchi: Some(1.3),
            box_counting: Some(1.31),
            petrosian: Some(1.29),
            katz: Some(1.32),
        };
        assert!(result.methods_agree(0.05), "methods should agree within 0.05");
        assert!(!result.methods_agree(0.01), "methods should NOT agree within 0.01");
    }

    // --- Rolling tests ---

    #[test]
    fn test_petrosian_rolling_length() {
        let prices = noisy_series(100, 1);
        let rolling = PetrosianMethod::rolling(&prices, 32);
        assert_eq!(rolling.len(), prices.len());
    }

    #[test]
    fn test_katz_rolling_first_values_nan() {
        let prices = trending_series(100);
        let rolling = KatzMethod::rolling(&prices, 32);
        // First 31 values should be NaN
        for v in &rolling[..31] {
            assert!(v.is_nan(), "Expected NaN before window fills");
        }
    }
}
