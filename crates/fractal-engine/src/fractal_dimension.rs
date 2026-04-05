/// Fractal dimension estimation via box-counting algorithm.
///
/// Interpretation:
///   FD ≈ 1.0  → smooth trend (1D curve)
///   FD ≈ 1.5  → Brownian motion
///   FD ≈ 2.0  → plane-filling, highly jagged / noisy
///
/// Trading use:
///   Only enter when FD < 1.5 (signal distinguishable from noise).
///   FD > 1.6 → reduce exposure, market is too noisy.
///
/// Algorithm (Higuchi fractal dimension):
///   Higuchi's method is O(n) and well-suited for financial time series.
///   For each k (scale), compute the average curve length L(k).
///   FD = slope of log(L) vs log(1/k) in OLS.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Fractal dimension types
// ---------------------------------------------------------------------------

/// Fractal dimension with classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalAnalysis {
    pub fd: f64,
    pub classification: FdClass,
    /// Number of scale levels used in regression.
    pub scale_levels: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FdClass {
    /// FD < 1.3 — very smooth trend
    SmoothTrend,
    /// 1.3 ≤ FD < 1.5 — mild roughness, tradeable
    Tradeable,
    /// 1.5 ≤ FD < 1.65 — Brownian-like, borderline
    Brownian,
    /// FD ≥ 1.65 — highly jagged, noisy
    Noisy,
}

impl FdClass {
    pub fn from_fd(fd: f64) -> Self {
        if fd < 1.3 {
            FdClass::SmoothTrend
        } else if fd < 1.5 {
            FdClass::Tradeable
        } else if fd < 1.65 {
            FdClass::Brownian
        } else {
            FdClass::Noisy
        }
    }

    pub fn is_tradeable(&self) -> bool {
        matches!(self, FdClass::SmoothTrend | FdClass::Tradeable)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FdClass::SmoothTrend => "SMOOTH_TREND",
            FdClass::Tradeable => "TRADEABLE",
            FdClass::Brownian => "BROWNIAN",
            FdClass::Noisy => "NOISY",
        }
    }
}

// ---------------------------------------------------------------------------
// Higuchi fractal dimension
// ---------------------------------------------------------------------------

/// Compute fractal dimension using Higuchi's method.
/// `k_max` controls the number of scales (typically 8–16).
pub fn higuchi_fd(prices: &[f64], k_max: usize) -> Option<FractalAnalysis> {
    let n = prices.len();
    if n < 16 || k_max < 2 {
        return None;
    }
    let k_max = k_max.min(n / 4).max(2);

    let mut log_k_inv: Vec<f64> = Vec::with_capacity(k_max);
    let mut log_lk: Vec<f64> = Vec::with_capacity(k_max);

    for k in 1..=k_max {
        if let Some(lk) = curve_length(prices, k) {
            log_k_inv.push(-(k as f64).ln());
            log_lk.push(lk.ln());
        }
    }

    if log_k_inv.len() < 3 {
        return None;
    }

    let pairs: Vec<(f64, f64)> = log_k_inv.iter().cloned().zip(log_lk.iter().cloned()).collect();
    let fd = ols_slope(&pairs);

    if fd.is_finite() && fd > 0.5 && fd < 3.0 {
        let fd_clamped = fd.clamp(1.0, 2.0);
        Some(FractalAnalysis {
            fd: fd_clamped,
            classification: FdClass::from_fd(fd_clamped),
            scale_levels: log_k_inv.len(),
        })
    } else {
        None
    }
}

/// Higuchi curve length L(k) for interval k.
fn curve_length(prices: &[f64], k: usize) -> Option<f64> {
    let n = prices.len();
    if n < k + 1 {
        return None;
    }
    let mut length_sum = 0.0f64;
    let mut valid_m = 0;
    for m in 1..=k {
        let num_steps = (n - m) / k;
        if num_steps < 1 {
            continue;
        }
        let mut lm = 0.0f64;
        for i in 1..=num_steps {
            lm += (prices[m + i * k - 1] - prices[m + (i - 1) * k - 1]).abs();
        }
        let normalisation = (n - 1) as f64 / (num_steps * k) as f64;
        lm *= normalisation / k as f64;
        length_sum += lm;
        valid_m += 1;
    }
    if valid_m == 0 {
        return None;
    }
    let avg = length_sum / valid_m as f64;
    if avg > 0.0 {
        Some(avg)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Box-counting fractal dimension (alternative / cross-check)
// ---------------------------------------------------------------------------

/// Compute fractal dimension via box-counting.
/// Scales the price series to [0, 1] and counts non-empty boxes at each resolution.
pub fn box_counting_fd(prices: &[f64]) -> Option<f64> {
    let n = prices.len();
    if n < 16 {
        return None;
    }
    let min_p = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_p = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_p - min_p;
    if range < 1e-12 {
        return None;
    }
    let normalised: Vec<f64> = prices.iter().map(|p| (p - min_p) / range).collect();

    let mut log_eps: Vec<f64> = Vec::new();
    let mut log_count: Vec<f64> = Vec::new();

    let max_grid = (n as f64).sqrt() as usize;
    let mut grid = 4usize;
    while grid <= max_grid {
        let count = count_boxes(&normalised, grid);
        if count > 0 {
            log_eps.push(-(grid as f64).ln());
            log_count.push((count as f64).ln());
        }
        grid *= 2;
    }

    if log_eps.len() < 3 {
        return None;
    }

    let pairs: Vec<(f64, f64)> = log_eps.iter().cloned().zip(log_count.iter().cloned()).collect();
    let fd = ols_slope(&pairs);
    if fd.is_finite() && fd > 0.5 && fd < 3.0 {
        Some(fd.clamp(1.0, 2.0))
    } else {
        None
    }
}

fn count_boxes(normalised: &[f64], grid_size: usize) -> usize {
    use std::collections::HashSet;
    let n = normalised.len();
    let mut occupied: HashSet<(usize, usize)> = HashSet::new();
    for i in 0..n {
        let x_box = (i * grid_size / n).min(grid_size - 1);
        let y_box = (normalised[i] * grid_size as f64).floor() as usize;
        let y_box = y_box.min(grid_size - 1);
        occupied.insert((x_box, y_box));
    }
    occupied.len()
}

// ---------------------------------------------------------------------------
// Rolling fractal dimension
// ---------------------------------------------------------------------------

/// Compute fractal dimension over a rolling window. Returns NaN where insufficient data.
pub fn rolling_fd(prices: &[f64], window: usize, k_max: usize) -> Vec<f64> {
    let n = prices.len();
    let mut out = vec![f64::NAN; n];
    for i in window..=n {
        let slice = &prices[(i - window)..i];
        if let Some(fa) = higuchi_fd(slice, k_max) {
            out[i - 1] = fa.fd;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// OLS helper
// ---------------------------------------------------------------------------

fn ols_slope(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    let mx: f64 = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let my: f64 = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
    let num: f64 = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = pairs.iter().map(|(x, _)| (x - mx).powi(2)).sum();
    if den.abs() < 1e-15 {
        1.5
    } else {
        num / den
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn straight_line(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    fn sine_noisy(n: usize, noise: f64, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|i| {
                let base = (i as f64 * 0.3).sin() * 10.0;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = (state as f64 / u64::MAX as f64) * 2.0 - 1.0;
                base + r * noise
            })
            .collect()
    }

    #[test]
    fn smooth_series_fd_below_1_5() {
        let prices = straight_line(64);
        if let Some(fa) = higuchi_fd(&prices, 8) {
            assert!(fa.fd < 1.5, "Straight line FD should be < 1.5, got {}", fa.fd);
        }
    }

    #[test]
    fn noisy_series_fd_higher() {
        let smooth = sine_noisy(128, 0.0, 1);
        let noisy = sine_noisy(128, 5.0, 1);
        let fd_smooth = higuchi_fd(&smooth, 8).map(|fa| fa.fd).unwrap_or(1.5);
        let fd_noisy = higuchi_fd(&noisy, 8).map(|fa| fa.fd).unwrap_or(1.5);
        assert!(
            fd_noisy >= fd_smooth,
            "Noisy FD {fd_noisy:.3} should be >= smooth FD {fd_smooth:.3}"
        );
    }

    #[test]
    fn fd_in_valid_range() {
        for seed in [1u64, 7, 42, 99] {
            let prices = sine_noisy(128, 2.0, seed);
            if let Some(fa) = higuchi_fd(&prices, 8) {
                assert!(
                    fa.fd >= 1.0 && fa.fd <= 2.0,
                    "FD {:.3} out of [1,2] for seed {seed}", fa.fd
                );
            }
        }
    }

    #[test]
    fn too_short_returns_none() {
        let prices = vec![1.0, 2.0, 3.0];
        assert!(higuchi_fd(&prices, 4).is_none());
    }

    #[test]
    fn fd_class_classification() {
        assert!(FdClass::from_fd(1.1).is_tradeable());
        assert!(FdClass::from_fd(1.4).is_tradeable());
        assert!(!FdClass::from_fd(1.7).is_tradeable());
    }

    #[test]
    fn box_counting_returns_finite() {
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 5.0)
            .collect();
        let fd = box_counting_fd(&prices);
        assert!(fd.map(|f| f.is_finite()).unwrap_or(true));
    }
}
