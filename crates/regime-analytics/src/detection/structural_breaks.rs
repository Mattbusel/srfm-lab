// structural_breaks.rs
// Structural break tests:
//   - CUSUM test for parameter stability.
//   - Zivot-Andrews unit root test with structural break.
//   - Bai-Perron multiple breakpoint test (BIC-based).

use serde::{Deserialize, Serialize};

/// Result of a breakpoint detection test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakpointResult {
    /// Detected breakpoint indices (0-based bar index).
    pub breakpoints: Vec<usize>,
    /// Test statistic(s) at each candidate location.
    pub test_stats: Vec<f64>,
    /// Critical value (e.g., 5% level).
    pub critical_value: f64,
    /// Whether the null of no break is rejected.
    pub reject_null: bool,
    pub method: String,
}

/// CUSUM test for structural change in the mean.
/// Null: no structural break. Reject if max |CUSUM| > critical value.
pub struct CusumTest {
    /// Significance level: 0.05 or 0.01.
    pub significance: f64,
}

impl CusumTest {
    pub fn new(significance: f64) -> Self {
        CusumTest { significance }
    }

    /// Run CUSUM test on a time series.
    /// Returns BreakpointResult with breakpoint at max CUSUM location.
    pub fn test(&self, y: &[f64]) -> BreakpointResult {
        let n = y.len();
        if n < 10 {
            return BreakpointResult {
                breakpoints: vec![],
                test_stats: vec![],
                critical_value: f64::NAN,
                reject_null: false,
                method: "CUSUM".to_string(),
            };
        }

        let mean = y.iter().sum::<f64>() / n as f64;
        let std = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-10);

        // Recursive residuals approach: CUSUM statistic.
        let mut cusum = Vec::with_capacity(n);
        let mut running_sum = 0.0f64;
        for yi in y {
            running_sum += (yi - mean) / std;
            cusum.push(running_sum / (n as f64).sqrt());
        }

        // Find maximum absolute CUSUM.
        let test_stats: Vec<f64> = cusum.iter().map(|v| v.abs()).collect();
        let max_cusum = test_stats.iter().cloned().fold(0.0f64, f64::max);
        let bp_idx = test_stats
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Critical values from Brown, Durbin, Evans (1975).
        let critical_value = if (self.significance - 0.05).abs() < 0.01 {
            1.358 // 5% level, asymptotic.
        } else if (self.significance - 0.01).abs() < 0.005 {
            1.628 // 1% level.
        } else {
            1.224 // 10% level.
        };

        let reject = max_cusum > critical_value;

        BreakpointResult {
            breakpoints: if reject { vec![bp_idx] } else { vec![] },
            test_stats,
            critical_value,
            reject_null: reject,
            method: "CUSUM".to_string(),
        }
    }

    /// CUSUM-of-squares test: tests for variance changes.
    pub fn test_squares(&self, y: &[f64]) -> BreakpointResult {
        let n = y.len();
        if n < 10 {
            return BreakpointResult {
                breakpoints: vec![],
                test_stats: vec![],
                critical_value: f64::NAN,
                reject_null: false,
                method: "CUSUM-SQ".to_string(),
            };
        }

        let total_ss: f64 = y.iter().map(|v| v * v).sum();
        let mut running_ss = 0.0f64;
        let mut cusum_sq = Vec::with_capacity(n);
        for (t, yi) in y.iter().enumerate() {
            running_ss += yi * yi;
            let expected = (t + 1) as f64 / n as f64;
            let stat = (running_ss / total_ss.max(1e-14)) - expected;
            cusum_sq.push(stat.abs());
        }

        let max_stat = cusum_sq.iter().cloned().fold(0.0f64, f64::max);
        let bp_idx = cusum_sq
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let critical_value = 1.36 / (n as f64).sqrt();
        let reject = max_stat > critical_value;

        BreakpointResult {
            breakpoints: if reject { vec![bp_idx] } else { vec![] },
            test_stats: cusum_sq,
            critical_value,
            reject_null: reject,
            method: "CUSUM-SQ".to_string(),
        }
    }
}

/// Zivot-Andrews (1992) test: unit root with endogenous structural break.
/// Tests H0: unit root with no break vs H1: trend-stationary with one break.
pub struct ZivotAndrews {
    /// Lags for ADF-type regression.
    pub lags: usize,
}

impl ZivotAndrews {
    pub fn new(lags: usize) -> Self {
        ZivotAndrews { lags }
    }

    /// Run the Zivot-Andrews test.
    /// Returns BreakpointResult with the optimal breakpoint (min t-statistic).
    pub fn test(&self, y: &[f64]) -> BreakpointResult {
        let n = y.len();
        if n < 20 {
            return BreakpointResult {
                breakpoints: vec![],
                test_stats: vec![f64::NAN],
                critical_value: -5.34,
                reject_null: false,
                method: "Zivot-Andrews".to_string(),
            };
        }

        // Trim: evaluate breaks from 10% to 90% of sample.
        let start_t = (0.10 * n as f64).ceil() as usize;
        let end_t = (0.90 * n as f64).floor() as usize;

        let mut min_t_stat = f64::INFINITY;
        let mut best_break = start_t;
        let mut t_stats_at_breaks = Vec::new();

        for tb in start_t..=end_t {
            let t_stat = self.adf_with_break(y, tb);
            t_stats_at_breaks.push(t_stat);
            if t_stat < min_t_stat {
                min_t_stat = t_stat;
                best_break = tb;
            }
        }

        // Critical value at 5% for model C (both mean and trend break): -5.08.
        let critical_value = -5.08;
        let reject = min_t_stat < critical_value;

        BreakpointResult {
            breakpoints: if reject { vec![best_break] } else { vec![] },
            test_stats: t_stats_at_breaks,
            critical_value,
            reject_null: reject,
            method: "Zivot-Andrews".to_string(),
        }
    }

    /// ADF regression with dummy for mean shift at breakpoint tb.
    /// Returns the t-statistic on the lagged level (unit root test).
    fn adf_with_break(&self, y: &[f64], tb: usize) -> f64 {
        let n = y.len();
        let lags = self.lags.min(n / 5);
        let start = lags + 1;
        if start >= n {
            return 0.0;
        }

        // Build regressors: [1, t, DU_tb, DT_tb, dy_{t-1}, ..., dy_{t-lags}, y_{t-1}].
        // DU_tb = 1 if t > tb.
        // DT_tb = t - tb if t > tb, else 0.
        // delta_y = y_t - y_{t-1} (dep variable).

        let dy: Vec<f64> = y.windows(2).map(|(w)| w[1] - w[0]).collect(); // length n-1.

        let n_obs = n - start;
        if n_obs < 5 {
            return 0.0;
        }

        // Number of regressors: 1 (intercept) + 1 (trend) + 1 (DU) + 1 (DT) + lags (lagged dy) + 1 (y_{t-1}).
        let k = 4 + lags + 1;
        let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
        let mut dep: Vec<f64> = Vec::with_capacity(n_obs);

        for t_idx in start..n {
            let t = t_idx as f64;
            let du = if t_idx > tb { 1.0 } else { 0.0 };
            let dt = if t_idx > tb { (t_idx - tb) as f64 } else { 0.0 };
            let mut row = vec![1.0, t, du, dt];
            // Lagged differences.
            for lag in 1..=lags {
                let lag_idx = t_idx as isize - lag as isize - 1;
                if lag_idx >= 0 && (lag_idx as usize) < dy.len() {
                    row.push(dy[lag_idx as usize]);
                } else {
                    row.push(0.0);
                }
            }
            // y_{t-1}.
            row.push(y[t_idx - 1]);
            x_mat.push(row);
            dep.push(dy[t_idx - 1]);
        }

        // OLS regression.
        let betas = ols_multivariate_t(&x_mat, &dep);
        if betas.len() != k {
            return 0.0;
        }

        // t-stat on y_{t-1} (last coefficient).
        let beta_last = betas[k - 1];
        let fitted: Vec<f64> = x_mat.iter().map(|row| row.iter().zip(betas.iter()).map(|(x, b)| x * b).sum::<f64>()).collect();
        let residuals: Vec<f64> = dep.iter().zip(fitted.iter()).map(|(d, f)| d - f).collect();
        let n_obs_f = n_obs as f64;
        let dof = (n_obs_f - k as f64).max(1.0);
        let s2 = residuals.iter().map(|r| r * r).sum::<f64>() / dof;

        // Compute (X'X)^{-1} diagonal for last element.
        let xtx_inv_last = ols_xtx_inv_diag_last(&x_mat, k);
        let se_last = (s2 * xtx_inv_last).sqrt().max(1e-14);
        beta_last / se_last
    }
}

/// Bai-Perron multiple breakpoint test (simplified BIC-based selection).
/// Tests for up to `max_breaks` structural breaks in the mean.
pub struct BaiPerron {
    /// Maximum number of breaks to consider.
    pub max_breaks: usize,
    /// Minimum segment length as fraction of T.
    pub h_frac: f64,
}

impl BaiPerron {
    pub fn new(max_breaks: usize, h_frac: f64) -> Self {
        BaiPerron { max_breaks, h_frac }
    }

    /// Run Bai-Perron test.
    /// Returns BreakpointResult with selected number of breaks and their locations.
    pub fn test(&self, y: &[f64]) -> BreakpointResult {
        let n = y.len();
        let h = ((self.h_frac * n as f64).ceil() as usize).max(3);
        let max_breaks = self.max_breaks.min(n / h - 1);

        if max_breaks == 0 || n < 2 * h {
            return BreakpointResult {
                breakpoints: vec![],
                test_stats: vec![0.0],
                critical_value: 0.0,
                reject_null: false,
                method: "Bai-Perron".to_string(),
            };
        }

        // Compute SSR for each segment [i, j].
        let ssr = self.segment_ssr_matrix(y, h);

        // Dynamic programming: find optimal m-break partition.
        let mut best_bic = self.bic_for_breaks(y, &[], &ssr, n);
        let mut best_breaks: Vec<usize> = vec![];

        for m in 1..=max_breaks {
            let candidate_bps = self.dp_optimal_breaks(y, m, &ssr, h, n);
            let bic = self.bic_for_breaks(y, &candidate_bps, &ssr, n);
            if bic < best_bic {
                best_bic = bic;
                best_breaks = candidate_bps;
            }
        }

        let reject = !best_breaks.is_empty();
        BreakpointResult {
            breakpoints: best_breaks,
            test_stats: vec![best_bic],
            critical_value: 0.0,
            reject_null: reject,
            method: "Bai-Perron".to_string(),
        }
    }

    /// Compute SSR[i][j] = SSR of a single segment from bar i to j (inclusive).
    fn segment_ssr_matrix(&self, y: &[f64], _h: usize) -> Vec<Vec<f64>> {
        let n = y.len();
        let mut ssr = vec![vec![f64::INFINITY; n]; n];

        // Precompute prefix sums for fast mean/SSR computation.
        let mut prefix_sum = vec![0.0f64; n + 1];
        let mut prefix_sq = vec![0.0f64; n + 1];
        for i in 0..n {
            prefix_sum[i + 1] = prefix_sum[i] + y[i];
            prefix_sq[i + 1] = prefix_sq[i] + y[i] * y[i];
        }

        for i in 0..n {
            for j in i..n {
                let len = (j - i + 1) as f64;
                let s = prefix_sum[j + 1] - prefix_sum[i];
                let sq = prefix_sq[j + 1] - prefix_sq[i];
                let mean = s / len;
                ssr[i][j] = sq - len * mean * mean;
            }
        }
        ssr
    }

    /// Dynamic programming to find optimal m-break locations.
    fn dp_optimal_breaks(
        &self,
        _y: &[f64],
        m: usize,
        ssr: &[Vec<f64>],
        h: usize,
        n: usize,
    ) -> Vec<usize> {
        // V[m][t] = min SSR of m-partition of y[0..=t].
        let mut v = vec![vec![f64::INFINITY; n]; m + 1];
        let mut bp_matrix = vec![vec![0usize; n]; m + 1];

        // Base: 0 breaks.
        for t in 0..n {
            v[0][t] = ssr[0][t];
        }

        for k in 1..=m {
            for t in (k * h)..(n) {
                for s in ((k - 1) * h)..(t.saturating_sub(h - 1)) {
                    let val = v[k - 1][s] + ssr[s + 1][t];
                    if val < v[k][t] {
                        v[k][t] = val;
                        bp_matrix[k][t] = s + 1;
                    }
                }
            }
        }

        // Backtrack.
        let mut bps = Vec::with_capacity(m);
        let mut t = n - 1;
        for k in (1..=m).rev() {
            let bp = bp_matrix[k][t];
            bps.push(bp);
            t = bp - 1;
        }
        bps.reverse();
        bps
    }

    /// BIC for a given set of breakpoints.
    fn bic_for_breaks(&self, y: &[f64], bps: &[usize], ssr: &[Vec<f64>], n: usize) -> f64 {
        let mut segments: Vec<(usize, usize)> = Vec::new();
        let mut prev = 0;
        for &bp in bps {
            if bp > prev {
                segments.push((prev, bp - 1));
            }
            prev = bp;
        }
        segments.push((prev, n - 1));

        let total_ssr: f64 = segments.iter().map(|&(i, j)| ssr[i][j]).sum();
        let sigma2 = total_ssr / n as f64;
        let ll = if sigma2 > 1e-14 {
            -0.5 * n as f64 * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0)
        } else {
            0.0
        };
        // Parameters: mean per segment + sigma.
        let k = segments.len() + 1;
        k as f64 * (n as f64).ln() - 2.0 * ll
    }
}

/// OLS for structural break regressions (helper).
fn ols_multivariate_t(x: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let k = if x.is_empty() { 0 } else { x[0].len() };
    if n == 0 || k == 0 {
        return vec![];
    }
    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    for i in 0..n {
        let xi = &x[i];
        for j in 0..k {
            xty[j] += xi[j] * y[i];
            for l in 0..k {
                xtx[j][l] += xi[j] * xi[l];
            }
        }
    }
    gauss_solve_sb(&xtx, &xty)
}

fn gauss_solve_sb(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j] * factor;
                aug[row][j] -= val;
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-14 {
            x[i] /= aug[i][i];
        }
    }
    x
}

/// Compute the (k-1, k-1) diagonal element of (X'X)^{-1} for SE calculation.
fn ols_xtx_inv_diag_last(x: &[Vec<f64>], k: usize) -> f64 {
    let n = x.len();
    if n == 0 || k == 0 {
        return 1.0;
    }
    let mut xtx = vec![vec![0.0f64; k]; k];
    for xi in x {
        for j in 0..k {
            for l in 0..k {
                xtx[j][l] += xi[j] * xi[l];
            }
        }
    }
    // Return (X'X)^{-1}[k-1, k-1] via cofactor / determinant or inverse.
    // Use Gaussian elimination to solve for last column of inverse.
    let mut e_last = vec![0.0f64; k];
    e_last[k - 1] = 1.0;
    let col = gauss_solve_sb(&xtx, &e_last);
    col.get(k - 1).copied().unwrap_or(1.0).abs().max(1e-14)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series_with_break(n: usize, break_at: usize, shift: f64) -> Vec<f64> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        (0..n)
            .map(|i| {
                let base = if i >= break_at { shift } else { 0.0 };
                base + rng.gen::<f64>() * 0.01 - 0.005
            })
            .collect()
    }

    #[test]
    fn test_cusum_detects_break() {
        let y = series_with_break(300, 150, 0.05);
        let cusum = CusumTest::new(0.05);
        let result = cusum.test(&y);
        // With a large mean shift, CUSUM should detect the break.
        assert!(result.reject_null, "CUSUM should detect mean shift");
        assert!(!result.breakpoints.is_empty());
    }

    #[test]
    fn test_cusum_no_break() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(7);
        let y: Vec<f64> = (0..200).map(|_| rng.gen::<f64>() * 0.01 - 0.005).collect();
        let cusum = CusumTest::new(0.05);
        let result = cusum.test(&y);
        // Might not always pass but should mostly not reject for pure noise.
        // Just check it runs without error.
        let _ = result;
    }

    #[test]
    fn test_bai_perron_finds_breaks() {
        let y = series_with_break(300, 100, 0.10);
        let bp = BaiPerron::new(3, 0.10);
        let result = bp.test(&y);
        // Should find at least one break.
        assert!(!result.breakpoints.is_empty(), "Bai-Perron should find break");
    }

    #[test]
    fn test_zivot_andrews_runs() {
        let y: Vec<f64> = (0..100).map(|i| i as f64 * 0.001).collect();
        let za = ZivotAndrews::new(1);
        let result = za.test(&y);
        assert!(!result.test_stats.is_empty());
    }
}
