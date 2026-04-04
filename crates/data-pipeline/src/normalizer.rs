/// Feature normalisation: z-score, rank, min-max, robust scaling.

fn mean_f(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_f(v: &[f64]) -> f64 {
    if v.len() < 2 { return 1.0; }
    let m = mean_f(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt().max(1e-12)
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ── Z-score Rolling ───────────────────────────────────────────────────────────

/// Rolling z-score normalisation: (x - rolling_mean) / rolling_std.
pub fn z_score_rolling(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    if n < window { return vec![0.0; n]; }
    let mut out = vec![0.0_f64; n];
    for i in window - 1..n {
        let slice = &series[i + 1 - window..=i];
        let m = mean_f(slice);
        let s = std_f(slice);
        out[i] = (series[i] - m) / s;
    }
    out
}

// ── Rank Normalisation ────────────────────────────────────────────────────────

/// Cross-sectional rank normalisation: maps values to [0, 1] based on rank.
/// Ties are broken by position.
pub fn rank_normalize(series: &[f64]) -> Vec<f64> {
    let n = series.len();
    if n == 0 { return vec![]; }
    let mut indexed: Vec<(usize, f64)> = series.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0.0_f64; n];
    for (rank, (orig_idx, _)) in indexed.into_iter().enumerate() {
        out[orig_idx] = rank as f64 / (n - 1).max(1) as f64;
    }
    out
}

/// Rolling rank normalisation: rank each value within its window.
pub fn rank_normalize_rolling(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    if n < window { return vec![0.5; n]; }
    let mut out = vec![0.5_f64; n];
    for i in window - 1..n {
        let slice = &series[i + 1 - window..=i];
        let val = series[i];
        let rank = slice.iter().filter(|&&x| x < val).count();
        out[i] = rank as f64 / (window - 1) as f64;
    }
    out
}

// ── Min-Max Rolling ───────────────────────────────────────────────────────────

/// Rolling min-max normalisation: (x - rolling_min) / (rolling_max - rolling_min).
pub fn min_max_rolling(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    if n < window { return vec![0.5; n]; }
    let mut out = vec![0.5_f64; n];
    for i in window - 1..n {
        let slice = &series[i + 1 - window..=i];
        let mn = slice.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        out[i] = if (mx - mn) < 1e-12 { 0.5 } else { (series[i] - mn) / (mx - mn) };
    }
    out
}

// ── Robust Scaling ────────────────────────────────────────────────────────────

/// Robust scaling using quantiles: (x - median) / IQR.
/// `quantile_low` and `quantile_high` typically = 0.25 and 0.75 (IQR).
pub fn robust_scale(series: &[f64], quantile_low: f64, quantile_high: f64) -> Vec<f64> {
    if series.is_empty() { return vec![]; }
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = quantile(&sorted, 0.5);
    let q_lo = quantile(&sorted, quantile_low);
    let q_hi = quantile(&sorted, quantile_high);
    let iqr = (q_hi - q_lo).max(1e-12);
    series.iter().map(|x| (x - median) / iqr).collect()
}

/// Rolling robust scale.
pub fn robust_scale_rolling(series: &[f64], window: usize, q_low: f64, q_high: f64) -> Vec<f64> {
    let n = series.len();
    if n < window { return vec![0.0; n]; }
    let mut out = vec![0.0_f64; n];
    for i in window - 1..n {
        let slice = series[i + 1 - window..=i].to_vec();
        let scaled = robust_scale(&slice, q_low, q_high);
        out[i] = *scaled.last().unwrap_or(&0.0);
    }
    out
}

// ── Winsorisation ─────────────────────────────────────────────────────────────

/// Winsorise a series at the given quantile bounds (in-place clipping).
pub fn winsorize(series: &[f64], lower_q: f64, upper_q: f64) -> Vec<f64> {
    if series.is_empty() { return vec![]; }
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = quantile(&sorted, lower_q);
    let hi = quantile(&sorted, upper_q);
    series.iter().map(|x| x.clamp(lo, hi)).collect()
}

// ── Standardisation of Matrix Columns ────────────────────────────────────────

/// Z-score standardise each column of a matrix.
pub fn standardize_columns(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_rows = matrix.len();
    if n_rows == 0 { return vec![]; }
    let n_cols = matrix[0].len();
    let mut result = matrix.to_vec();

    for j in 0..n_cols {
        let col: Vec<f64> = matrix.iter().map(|row| row[j]).collect();
        let m = mean_f(&col);
        let s = std_f(&col);
        for i in 0..n_rows {
            result[i][j] = (matrix[i][j] - m) / s;
        }
    }
    result
}

/// Min-max normalise each column of a matrix to [0, 1].
pub fn normalize_columns(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_rows = matrix.len();
    if n_rows == 0 { return vec![]; }
    let n_cols = matrix[0].len();
    let mut result = matrix.to_vec();
    for j in 0..n_cols {
        let col: Vec<f64> = matrix.iter().map(|row| row[j]).collect();
        let mn = col.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = (mx - mn).max(1e-12);
        for i in 0..n_rows {
            result[i][j] = (matrix[i][j] - mn) / range;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_score_near_zero_mean() {
        let series: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let z = z_score_rolling(&series, 10);
        let last = z.last().unwrap();
        // For a linear series the last z-score should be positive.
        assert!(*last > 0.0, "last z={last}");
    }

    #[test]
    fn rank_normalize_bounds() {
        let series = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        let ranks = rank_normalize(&series);
        let min_rank = ranks.iter().copied().fold(f64::INFINITY, f64::min);
        let max_rank = ranks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(min_rank >= 0.0, "min rank {min_rank}");
        assert!(max_rank <= 1.0, "max rank {max_rank}");
    }

    #[test]
    fn min_max_rolling_in_range() {
        let series: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let mm = min_max_rolling(&series, 10);
        for &v in &mm[9..] {
            assert!(v >= 0.0 && v <= 1.0, "v={v}");
        }
    }
}
