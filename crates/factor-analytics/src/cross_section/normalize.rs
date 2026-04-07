//! Cross-sectional normalization: z-score, rank normalization,
//! winsorization (3-sigma), and MAD robust scaling.

use ndarray::{Array1, Array2, Axis};
use crate::error::{FactorError, Result};

/// Method for cross-sectional normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Standard z-score: (x - mean) / std
    ZScore,
    /// Rank normalization: convert to uniform [0, 1] ranks
    Rank,
    /// Winsorize at k sigma then z-score
    WinsorizedZScore { sigma: f64 },
    /// MAD scaling: (x - median) / (1.4826 * MAD)
    MadRobust,
}

/// Compute cross-sectional z-score.
///
/// NaN values are preserved; computation uses only finite values for mean/std.
pub fn zscore_cross_section(values: &[f64]) -> Vec<f64> {
    let valid: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.len() < 2 {
        return vec![f64::NAN; values.len()];
    }
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();

    if std < 1e-12 {
        return vec![0.0; values.len()];
    }

    values
        .iter()
        .map(|&v| if v.is_finite() { (v - mean) / std } else { f64::NAN })
        .collect()
}

/// Compute cross-sectional rank normalization.
///
/// Converts values to uniform [0, 1] scores using fractional ranks.
/// Ties are broken by averaging ranks.
/// NaN values receive NaN score.
pub fn rank_normalize(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![f64::NAN; n];

    // Collect finite values with original indices
    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
        .collect();

    if indexed.is_empty() {
        return result;
    }

    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let m = indexed.len();

    let mut i = 0;
    while i < m {
        // Find the extent of ties at this value
        let val = indexed[i].1;
        let mut j = i + 1;
        while j < m && (indexed[j].1 - val).abs() < 1e-14 {
            j += 1;
        }
        // Average rank for tied values: ranks are 1-based, normalized to [0,1]
        let avg_rank = (i + j - 1) as f64 / 2.0;
        let normalized = avg_rank / (m - 1).max(1) as f64;

        for k in i..j {
            result[indexed[k].0] = normalized;
        }
        i = j;
    }

    result
}

/// Winsorize values at k standard deviations from the mean.
///
/// Modifies extreme values by capping at mean +/- k*sigma.
pub fn winsorize_sigma(values: &[f64], k: f64) -> Vec<f64> {
    let valid: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.len() < 2 {
        return values.to_vec();
    }
    let n = valid.len() as f64;
    let mean = valid.iter().sum::<f64>() / n;
    let std = (valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

    let lo = mean - k * std;
    let hi = mean + k * std;

    values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                f64::NAN
            } else {
                v.max(lo).min(hi)
            }
        })
        .collect()
}

/// Winsorize at k sigma, then z-score.
pub fn winsorized_zscore(values: &[f64], k: f64) -> Vec<f64> {
    let winsorized = winsorize_sigma(values, k);
    zscore_cross_section(&winsorized)
}

/// Compute the median of finite values.
pub fn median(values: &[f64]) -> f64 {
    let mut valid: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        return f64::NAN;
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = valid.len();
    if n % 2 == 0 {
        (valid[n / 2 - 1] + valid[n / 2]) / 2.0
    } else {
        valid[n / 2]
    }
}

/// Median Absolute Deviation (MAD).
///
/// MAD = median(|x_i - median(x)|)
pub fn mad(values: &[f64]) -> f64 {
    let med = median(values);
    if !med.is_finite() {
        return f64::NAN;
    }
    let deviations: Vec<f64> = values
        .iter()
        .filter_map(|&v| if v.is_finite() { Some((v - med).abs()) } else { None })
        .collect();
    median(&deviations)
}

/// MAD robust scaling: (x - median) / (1.4826 * MAD).
///
/// The constant 1.4826 makes MAD consistent with normal distribution std.
pub fn mad_robust_scale(values: &[f64]) -> Vec<f64> {
    let med = median(values);
    let mad_val = mad(values);

    if !med.is_finite() || !mad_val.is_finite() {
        return vec![f64::NAN; values.len()];
    }

    let scale = 1.4826 * mad_val;
    if scale < 1e-12 {
        // All values are identical -- return zeros
        return values
            .iter()
            .map(|&v| if v.is_finite() { 0.0 } else { f64::NAN })
            .collect();
    }

    values
        .iter()
        .map(|&v| if v.is_finite() { (v - med) / scale } else { f64::NAN })
        .collect()
}

/// Apply normalization method to a single factor column.
pub fn normalize_factor(values: &[f64], method: NormalizationMethod) -> Vec<f64> {
    match method {
        NormalizationMethod::ZScore => zscore_cross_section(values),
        NormalizationMethod::Rank => rank_normalize(values),
        NormalizationMethod::WinsorizedZScore { sigma } => winsorized_zscore(values, sigma),
        NormalizationMethod::MadRobust => mad_robust_scale(values),
    }
}

/// Normalize all columns of a factor matrix in place.
///
/// Each column is normalized independently as a cross-section.
///
/// # Arguments
/// * `factor_matrix` -- shape (n_assets, n_factors)
/// * `method` -- normalization method to apply
pub fn normalize_factor_matrix(
    factor_matrix: &Array2<f64>,
    method: NormalizationMethod,
) -> Result<Array2<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();
    let mut result = Array2::<f64>::from_elem((n_assets, n_factors), f64::NAN);

    for k in 0..n_factors {
        let col: Vec<f64> = factor_matrix.column(k).to_vec();
        let normalized = normalize_factor(&col, method);
        for i in 0..n_assets {
            result[[i, k]] = normalized[i];
        }
    }

    Ok(result)
}

/// Percentile-based normalization: convert to quantile ranks in [0, 100].
pub fn percentile_rank(values: &[f64]) -> Vec<f64> {
    rank_normalize(values)
        .iter()
        .map(|&r| if r.is_finite() { r * 100.0 } else { f64::NAN })
        .collect()
}

/// Quantile normalization: map each value to the corresponding normal quantile.
///
/// This approximates the inverse normal CDF applied to rank-normalized values.
/// Uses the rational approximation to the inverse normal.
pub fn quantile_normalize(values: &[f64]) -> Vec<f64> {
    let ranks = rank_normalize(values);
    ranks
        .iter()
        .map(|&r| {
            if !r.is_finite() {
                f64::NAN
            } else {
                // Map uniform [0,1] to normal quantile
                // Clamp to avoid infinities
                let p = r.max(0.001).min(0.999);
                inverse_normal_cdf(p)
            }
        })
        .collect()
}

/// Rational approximation of the inverse normal CDF (Abramowitz & Stegun).
/// Accurate to about 4.5e-4.
fn inverse_normal_cdf(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Compute cross-sectional information ratio between two factor versions.
///
/// Useful to compare raw vs. normalized factor predictive power.
pub fn cross_sectional_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let pairs: Vec<(f64, f64)> = (0..n)
        .filter_map(|i| {
            if x[i].is_finite() && y[i].is_finite() {
                Some((x[i], y[i]))
            } else {
                None
            }
        })
        .collect();

    let m = pairs.len();
    if m < 3 {
        return f64::NAN;
    }

    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / m as f64;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / m as f64;

    let cov = pairs.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / (m - 1) as f64;
    let sx = (pairs.iter().map(|p| (p.0 - mx).powi(2)).sum::<f64>() / (m - 1) as f64).sqrt();
    let sy = (pairs.iter().map(|p| (p.1 - my).powi(2)).sum::<f64>() / (m - 1) as f64).sqrt();

    if sx < 1e-12 || sy < 1e-12 {
        return 0.0;
    }

    cov / (sx * sy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let zs = zscore_cross_section(&vals);
        let mean_z: f64 = zs.iter().sum::<f64>() / 5.0;
        let std_z = (zs.iter().map(|v| v.powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!(mean_z.abs() < 1e-10);
        assert!((std_z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_normalize() {
        let vals = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = rank_normalize(&vals);
        // All should be in [0, 1]
        for r in &ranks {
            assert!(*r >= 0.0 && *r <= 1.0);
        }
        // Min value should have min rank
        assert!(ranks[1] < ranks[0]);
    }

    #[test]
    fn test_rank_normalize_with_nan() {
        let vals = vec![1.0, f64::NAN, 3.0, 2.0];
        let ranks = rank_normalize(&vals);
        assert!(ranks[1].is_nan());
        assert!(ranks[0].is_finite());
    }

    #[test]
    fn test_mad_robust_scale() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let scaled = mad_robust_scale(&vals);
        // Median = 49.5, MAD = 25, scale = 1.4826 * 25 = 37.065
        // vals[0] should be (0 - 49.5) / (1.4826 * 25)
        assert!(scaled[0].is_finite());
        // Scaled mean should be close to 0
        let finite: Vec<f64> = scaled.iter().copied().filter(|v| v.is_finite()).collect();
        let mean_scaled = finite.iter().sum::<f64>() / finite.len() as f64;
        assert!(mean_scaled.abs() < 0.01);
    }

    #[test]
    fn test_winsorize_sigma() {
        let mut vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        vals.push(1000.0); // outlier
        let ws = winsorize_sigma(&vals, 3.0);
        // 1000.0 should be winsorized -- the outlier is the last element
        // After winsorization at 3-sigma, any value beyond mean +/- 3*std is capped
        let max_ws = ws.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let unwinsorized_max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_ws < unwinsorized_max, "max after winsorize={}, original max={}", max_ws, unwinsorized_max);
    }

    #[test]
    fn test_inverse_normal_cdf() {
        // Standard normal quantiles
        assert!((inverse_normal_cdf(0.5)).abs() < 0.001);
        assert!((inverse_normal_cdf(0.8413) - 1.0).abs() < 0.01);
        assert!((inverse_normal_cdf(0.1587) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_normalize_matrix() {
        let m = Array2::from_shape_fn((20, 3), |(i, j)| i as f64 + j as f64 * 0.5);
        let norm = normalize_factor_matrix(&m, NormalizationMethod::ZScore).unwrap();
        assert_eq!(norm.dim(), (20, 3));
        for k in 0..3 {
            let col: Vec<f64> = norm.column(k).to_vec();
            let mean: f64 = col.iter().sum::<f64>() / 20.0;
            assert!(mean.abs() < 1e-10);
        }
    }
}
