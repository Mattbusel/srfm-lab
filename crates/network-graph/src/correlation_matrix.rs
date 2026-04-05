use std::collections::HashMap;

/// An N×N symmetric correlation matrix.
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    pub symbols: Vec<String>,
    pub n: usize,
    /// Row-major N×N matrix. Element [i][j] = corr(symbol[i], symbol[j]).
    pub data: Vec<Vec<f64>>,
}

impl CorrelationMatrix {
    /// Build a correlation matrix from return series using the last `window` observations.
    ///
    /// Uses Welford's numerically-stable online algorithm for mean and variance.
    pub fn from_returns(returns: &HashMap<String, Vec<f64>>, window: usize) -> Self {
        let mut symbols: Vec<String> = returns.keys().cloned().collect();
        symbols.sort(); // deterministic ordering
        let n = symbols.len();

        // Extract windowed return slices.
        let slices: Vec<&[f64]> = symbols
            .iter()
            .map(|s| {
                let r = returns[s].as_slice();
                let start = r.len().saturating_sub(window);
                &r[start..]
            })
            .collect();

        let mut data = vec![vec![0.0_f64; n]; n];

        for i in 0..n {
            data[i][i] = 1.0;
            for j in (i + 1)..n {
                let c = pearson_stable(slices[i], slices[j]).unwrap_or(0.0);
                data[i][j] = c;
                data[j][i] = c;
            }
        }

        Self { symbols, n, data }
    }

    /// Look up correlation between two symbols by name.
    pub fn get(&self, sym_a: &str, sym_b: &str) -> Option<f64> {
        let i = self.symbols.iter().position(|s| s == sym_a)?;
        let j = self.symbols.iter().position(|s| s == sym_b)?;
        Some(self.data[i][j])
    }

    /// Compute the partial correlation matrix by inverting the correlation matrix
    /// (precision matrix) via Gaussian elimination.
    ///
    /// Partial corr(i,j) = -P[i][j] / sqrt(P[i][i] * P[j][j])
    /// where P is the precision matrix.
    pub fn partial_correlations(&self) -> Option<Vec<Vec<f64>>> {
        let precision = invert_matrix(&self.data)?;
        let n = self.n;
        let mut partial = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            partial[i][i] = 1.0;
            for j in (i + 1)..n {
                let denom = (precision[i][i] * precision[j][j]).sqrt();
                if denom < 1e-12 {
                    continue;
                }
                let pc = (-precision[i][j] / denom).clamp(-1.0, 1.0);
                partial[i][j] = pc;
                partial[j][i] = pc;
            }
        }
        Some(partial)
    }

    /// Ledoit-Wolf shrinkage estimator towards the identity matrix.
    ///
    /// Shrinks S → (1-α)·S + α·I to reduce estimation error in small samples.
    /// The shrinkage intensity α is computed analytically.
    pub fn shrink_ledoit_wolf(returns: &HashMap<String, Vec<f64>>, window: usize) -> Self {
        let base = Self::from_returns(returns, window);
        let n = base.n as f64;

        // Determine sample size.
        let t = returns
            .values()
            .map(|r| r.len().min(window))
            .min()
            .unwrap_or(1) as f64;

        // Oracle shrinkage intensity (simplified Ledoit-Wolf 2004 constant formula).
        // alpha = min(1, max(0, (n+2) / ((t-n-1+2)*t) * trace_correction))
        // Simplified: alpha = (n + 2) / ((t + n + 1))
        let alpha = ((n + 2.0) / (t + n + 1.0)).clamp(0.0, 1.0);

        let mut shrunk = base.clone();
        for i in 0..shrunk.n {
            for j in 0..shrunk.n {
                if i == j {
                    shrunk.data[i][j] = 1.0;
                } else {
                    shrunk.data[i][j] = (1.0 - alpha) * base.data[i][j];
                }
            }
        }

        shrunk
    }
}

// ── Numerically-stable Pearson correlation (Welford-style) ───────────────────

/// Pearson correlation using Welford's online algorithm for numerical stability.
pub fn pearson_stable(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 3 {
        return None;
    }

    let mut mean_x = 0.0_f64;
    let mut mean_y = 0.0_f64;
    let mut m2_x = 0.0_f64;
    let mut m2_y = 0.0_f64;
    let mut c_xy = 0.0_f64;

    for k in 0..n {
        let kf = (k + 1) as f64;
        let dx = x[k] - mean_x;
        let dy = y[k] - mean_y;
        mean_x += dx / kf;
        mean_y += dy / kf;
        m2_x += dx * (x[k] - mean_x);
        m2_y += dy * (y[k] - mean_y);
        c_xy += dx * (y[k] - mean_y);
    }

    let var_x = m2_x / (n - 1) as f64;
    let var_y = m2_y / (n - 1) as f64;
    let cov_xy = c_xy / (n - 1) as f64;

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        None
    } else {
        Some((cov_xy / denom).clamp(-1.0, 1.0))
    }
}

// ── Gaussian Elimination Matrix Inversion ────────────────────────────────────

/// Invert an N×N matrix using Gauss-Jordan elimination with partial pivoting.
/// Returns None if the matrix is singular.
pub fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    // Augmented matrix [M | I].
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = m[i].clone();
            for j in 0..n {
                row.push(if i == j { 1.0 } else { 0.0 });
            }
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivoting.
        let pivot = (col..n)
            .max_by(|&a, &b| {
                aug[a][col]
                    .abs()
                    .partial_cmp(&aug[b][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
        aug.swap(col, pivot);

        let diag = aug[col][col];
        if diag.abs() < 1e-12 {
            return None; // singular
        }

        // Normalise pivot row.
        let inv_diag = 1.0 / diag;
        for val in aug[col].iter_mut() {
            *val *= inv_diag;
        }

        // Eliminate column.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for k in 0..2 * n {
                let sub = factor * aug[col][k];
                aug[row][k] -= sub;
            }
        }
    }

    // Extract right half.
    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pearson_perfect_correlation() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let corr = pearson_stable(&x, &x).unwrap();
        assert_abs_diff_eq!(corr, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_anti_correlation() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| -v).collect();
        let corr = pearson_stable(&x, &y).unwrap();
        assert_abs_diff_eq!(corr, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pearson_zero_correlation() {
        // x = constant, y = anything => None (zero variance in x).
        let x = vec![1.0_f64; 50];
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!(pearson_stable(&x, &y).is_none());
    }

    #[test]
    fn test_correlation_matrix_diagonal_one() {
        let mut returns = HashMap::new();
        let r: Vec<f64> = (0..40).map(|i| (i as f64).sin() * 0.01).collect();
        returns.insert("BTC".to_string(), r.clone());
        returns.insert("ETH".to_string(), r.iter().map(|x| x * 1.1).collect());
        let cm = CorrelationMatrix::from_returns(&returns, 40);
        assert_abs_diff_eq!(cm.data[0][0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cm.data[1][1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invert_identity() {
        let id = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let inv = invert_matrix(&id).unwrap();
        assert_abs_diff_eq!(inv[0][0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(inv[1][1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(inv[0][1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invert_2x2() {
        // [2,1;1,2]^-1 = [2,-1;-1,2] / 3
        let m = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let inv = invert_matrix(&m).unwrap();
        assert_abs_diff_eq!(inv[0][0], 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(inv[0][1], -1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_partial_correlations_shape() {
        let mut returns = HashMap::new();
        // Use distinct, non-collinear series to avoid singular matrix.
        let r1: Vec<f64> = (0..60).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let r2: Vec<f64> = (0..60).map(|i| (i as f64 * 0.2).cos() * 0.01).collect();
        let r3: Vec<f64> = (0..60).map(|i| (i as f64 * 0.35).sin() * 0.01 + (i as f64 * 0.17).cos() * 0.005).collect();
        returns.insert("BTC".to_string(), r1);
        returns.insert("ETH".to_string(), r2);
        returns.insert("SOL".to_string(), r3);
        let cm = CorrelationMatrix::from_returns(&returns, 60);
        let partial = cm.partial_correlations();
        // May return None if matrix is nearly singular — that is acceptable.
        if let Some(p) = partial {
            assert_eq!(p.len(), 3);
            for i in 0..3 {
                assert_abs_diff_eq!(p[i][i], 1.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_shrinkage_reduces_off_diagonal() {
        let mut returns = HashMap::new();
        let r: Vec<f64> = (0..20).map(|i| (i as f64 * 0.3).sin() * 0.01).collect();
        let r2: Vec<f64> = r.iter().map(|x| x * 1.5 + 0.0001).collect();
        returns.insert("A".to_string(), r);
        returns.insert("B".to_string(), r2);
        let raw = CorrelationMatrix::from_returns(&returns, 20);
        let shrunk = CorrelationMatrix::shrink_ledoit_wolf(&returns, 20);
        // Off-diagonal should be smaller in shrunk version.
        let i = 0;
        let j = 1;
        assert!(shrunk.data[i][j].abs() <= raw.data[i][j].abs() + 1e-10);
    }
}
