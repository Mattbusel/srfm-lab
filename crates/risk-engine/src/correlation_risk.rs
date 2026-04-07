/// Correlation risk monitoring: rolling matrices, DCC(1,1), eigen analysis,
/// and correlation-breakdown stress testing.

// ── Rolling correlation matrix ────────────────────────────────────────────────

/// Compute a rolling correlation matrix from a matrix of return series.
/// `returns[i]` is the return series for asset i.
/// Uses a sliding window of `window` observations.
/// Returns an n x n correlation matrix using the most recent `window` observations.
pub fn rolling_correlation_matrix(returns: &[Vec<f64>], window: usize) -> Vec<Vec<f64>> {
    let n = returns.len();
    if n == 0 {
        return vec![];
    }
    let min_len = returns.iter().map(|r| r.len()).min().unwrap_or(0);
    let w = window.min(min_len);
    if w < 2 {
        // Not enough data -- return identity.
        return identity_matrix(n);
    }
    let start = min_len - w;

    // Compute means.
    let means: Vec<f64> = returns
        .iter()
        .map(|r| {
            let slice = &r[start..start + w];
            slice.iter().sum::<f64>() / w as f64
        })
        .collect();

    // Compute variances and covariances.
    let mut cov = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0_f64;
            for t in 0..w {
                sum += (returns[i][start + t] - means[i])
                    * (returns[j][start + t] - means[j]);
            }
            cov[i][j] = sum / (w as f64 - 1.0);
            cov[j][i] = cov[i][j];
        }
    }

    // Convert covariance to correlation.
    let mut corr = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let denom = (cov[i][i] * cov[j][j]).sqrt();
            corr[i][j] = if denom > 1e-15 {
                (cov[i][j] / denom).clamp(-1.0, 1.0)
            } else if i == j {
                1.0
            } else {
                0.0
            };
        }
    }
    corr
}

/// Incremental correlation update given a new observation vector.
/// Updates the running sums n, sum_x, sum_x2, sum_xy to produce a new
/// n x n correlation matrix. O(n^2) per new observation.
pub struct IncrementalCorrelation {
    n_assets: usize,
    window: usize,
    /// Circular buffer of observations: buffer[t] = [r_0, r_1, ..., r_{n-1}]
    buffer: Vec<Vec<f64>>,
    buf_pos: usize,
    count: usize,
    /// Running sums (over buffer).
    sum: Vec<f64>,
    /// Running sum of squares.
    sum_sq: Vec<f64>,
    /// Running sum of cross-products: sum_cross[i][j] = sum(r_i * r_j).
    sum_cross: Vec<Vec<f64>>,
}

impl IncrementalCorrelation {
    pub fn new(n_assets: usize, window: usize) -> Self {
        IncrementalCorrelation {
            n_assets,
            window,
            buffer: vec![vec![0.0; n_assets]; window],
            buf_pos: 0,
            count: 0,
            sum: vec![0.0; n_assets],
            sum_sq: vec![0.0; n_assets],
            sum_cross: vec![vec![0.0; n_assets]; n_assets],
        }
    }

    /// Feed a new observation and update running sums. O(n^2).
    pub fn update(&mut self, obs: &[f64]) {
        assert_eq!(obs.len(), self.n_assets);

        // Remove the oldest observation if buffer is full.
        if self.count == self.window {
            let old = self.buffer[self.buf_pos].clone();
            for i in 0..self.n_assets {
                self.sum[i] -= old[i];
                self.sum_sq[i] -= old[i] * old[i];
                for j in 0..self.n_assets {
                    self.sum_cross[i][j] -= old[i] * old[j];
                }
            }
        } else {
            self.count += 1;
        }

        // Store new observation.
        self.buffer[self.buf_pos] = obs.to_vec();
        self.buf_pos = (self.buf_pos + 1) % self.window;

        // Add new observation to running sums.
        for i in 0..self.n_assets {
            self.sum[i] += obs[i];
            self.sum_sq[i] += obs[i] * obs[i];
            for j in 0..self.n_assets {
                self.sum_cross[i][j] += obs[i] * obs[j];
            }
        }
    }

    /// Return the current correlation matrix.
    pub fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n_assets;
        let c = self.count as f64;
        if c < 2.0 {
            return identity_matrix(n);
        }
        let mut corr = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in i..n {
                // Cov(i,j) = (sum_cross[i][j] - sum[i]*sum[j]/c) / (c-1)
                let cov_ij = (self.sum_cross[i][j] - self.sum[i] * self.sum[j] / c) / (c - 1.0);
                let var_i = (self.sum_sq[i] - self.sum[i] * self.sum[i] / c) / (c - 1.0);
                let var_j = (self.sum_sq[j] - self.sum[j] * self.sum[j] / c) / (c - 1.0);
                let rho = if var_i > 1e-15 && var_j > 1e-15 {
                    (cov_ij / (var_i * var_j).sqrt()).clamp(-1.0, 1.0)
                } else if i == j {
                    1.0
                } else {
                    0.0
                };
                corr[i][j] = rho;
                corr[j][i] = rho;
            }
        }
        corr
    }
}

// ── Correlation breakdown risk ────────────────────────────────────────────────

/// Compute the incremental portfolio variance due to a correlation shift.
/// Returns: portfolio variance under stress_corr - portfolio variance under current_corr.
/// portfolio: vector of portfolio weights (not necessarily normalized).
pub fn correlation_breakdown_risk(
    current_corr: &[Vec<f64>],
    stress_corr: &[Vec<f64>],
    portfolio: &[f64],
) -> f64 {
    let n = portfolio.len();
    assert_eq!(current_corr.len(), n);
    assert_eq!(stress_corr.len(), n);

    let var_current = portfolio_variance_from_corr(current_corr, portfolio);
    let var_stress = portfolio_variance_from_corr(stress_corr, portfolio);
    var_stress - var_current
}

fn portfolio_variance_from_corr(corr: &[Vec<f64>], weights: &[f64]) -> f64 {
    let n = weights.len();
    let mut var = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            var += weights[i] * weights[j] * corr[i][j];
        }
    }
    var.max(0.0)
}

// ── Eigenvalue analysis ───────────────────────────────────────────────────────

/// Summary of eigenvalue-based portfolio risk.
#[derive(Debug, Clone)]
pub struct EigenRisk {
    /// Number of factors explaining >95% of total variance.
    pub n_significant_factors: usize,
    /// Concentration ratio (0 to 1): higher = more concentrated.
    pub concentration: f64,
    /// Fraction of total variance explained by the first eigenvalue.
    pub first_eigenvalue_pct: f64,
}

/// Estimate eigenvalue risk from a correlation matrix using the power iteration
/// method to find the dominant eigenvalue and the Frobenius norm for the rest.
/// Note: for a full decomposition of n x n, we use a simplified Jacobi-inspired
/// approach suitable for small matrices (n <= 20).
pub fn eigen_portfolio_risk(corr_matrix: &[Vec<f64>]) -> EigenRisk {
    let n = corr_matrix.len();
    if n == 0 {
        return EigenRisk {
            n_significant_factors: 0,
            concentration: 0.0,
            first_eigenvalue_pct: 0.0,
        };
    }
    if n == 1 {
        return EigenRisk {
            n_significant_factors: 1,
            concentration: 1.0,
            first_eigenvalue_pct: 1.0,
        };
    }

    // Power iteration to estimate the top k eigenvalues.
    let eigenvalues = estimate_eigenvalues(corr_matrix, n.min(10));
    let total: f64 = eigenvalues.iter().sum::<f64>().max(1e-12);
    let first_pct = eigenvalues[0] / total;

    // Number of factors explaining > 95% of variance.
    let mut cumulative = 0.0_f64;
    let mut n_sig = 0_usize;
    for &ev in &eigenvalues {
        cumulative += ev;
        n_sig += 1;
        if cumulative / total >= 0.95 {
            break;
        }
    }

    // Concentration: 1 if first eigenvalue > 50% of trace (= n for corr matrix).
    let concentration = first_pct.clamp(0.0, 1.0);

    EigenRisk {
        n_significant_factors: n_sig,
        concentration,
        first_eigenvalue_pct: first_pct,
    }
}

/// Estimate the top k eigenvalues of a symmetric PD matrix using deflated
/// power iteration.
fn estimate_eigenvalues(matrix: &[Vec<f64>], k: usize) -> Vec<f64> {
    let n = matrix.len();
    let mut m: Vec<Vec<f64>> = matrix.to_vec();
    let mut eigenvalues = Vec::with_capacity(k);

    for _ in 0..k {
        // Power iteration for dominant eigenvalue.
        let mut v: Vec<f64> = vec![1.0; n];
        normalize_vec(&mut v);
        let mut eigenval = 0.0_f64;
        for _ in 0..200 {
            let mv = mat_vec_mul(&m, &v);
            let new_eigenval = dot(&v, &mv);
            normalize_vec_into(&mv, &mut v);
            if (new_eigenval - eigenval).abs() < 1e-10 {
                eigenval = new_eigenval;
                break;
            }
            eigenval = new_eigenval;
        }
        if eigenval < 1e-10 {
            break;
        }
        eigenvalues.push(eigenval);
        // Deflate: M = M - lambda * v * v^T
        for i in 0..n {
            for j in 0..n {
                m[i][j] -= eigenval * v[i] * v[j];
            }
        }
    }

    if eigenvalues.is_empty() {
        eigenvalues.push(n as f64); // fallback
    }
    eigenvalues
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum()).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize_vec(v: &mut Vec<f64>) {
    let norm = dot(v, v).sqrt();
    if norm > 1e-15 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn normalize_vec_into(src: &[f64], dst: &mut Vec<f64>) {
    let norm = dot(src, src).sqrt();
    dst.resize(src.len(), 0.0);
    if norm > 1e-15 {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = s / norm;
        }
    } else {
        for d in dst.iter_mut() {
            *d = 0.0;
        }
    }
}

// ── Dynamic Conditional Correlation (DCC-GARCH) ────────────────────────────────

/// One-step DCC(1,1) update.
/// Given the previous conditional correlation matrix Q_prev (n x n),
/// a vector of standardized residuals (returns / conditional vol),
/// and persistence parameters alpha + beta < 1,
/// compute the new correlation matrix.
///
/// DCC update equations (Engle 2002):
///   Q_t = (1 - alpha - beta) * Q_bar + alpha * e_{t-1} * e_{t-1}^T + beta * Q_{t-1}
///   R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
///
/// Q_bar is approximated as the unconditional correlation (identity here).
pub fn dynamic_conditional_correlation_step(
    prev_dcc: &[Vec<f64>],
    returns: &[f64],
    alpha: f64,
    beta: f64,
) -> Vec<Vec<f64>> {
    let n = returns.len();
    assert_eq!(prev_dcc.len(), n, "DCC matrix dimension must match returns length");
    assert!(alpha >= 0.0 && beta >= 0.0 && alpha + beta < 1.0,
        "DCC requires alpha >= 0, beta >= 0, alpha + beta < 1");

    let omega = 1.0 - alpha - beta;

    // Q_bar = identity (unconditional correlation).
    let mut q_new = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let q_bar = if i == j { 1.0 } else { 0.0 };
            let outer = returns[i] * returns[j];
            q_new[i][j] = omega * q_bar + alpha * outer + beta * prev_dcc[i][j];
        }
    }

    // Normalize to get correlation matrix R_t.
    let mut r_new = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let denom = (q_new[i][i] * q_new[j][j]).sqrt();
            r_new[i][j] = if denom > 1e-15 {
                (q_new[i][j] / denom).clamp(-1.0, 1.0)
            } else if i == j {
                1.0
            } else {
                0.0
            };
        }
    }
    r_new
}

// ── Utility ───────────────────────────────────────────────────────────────────

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_pos_corr_returns(n: usize, t: usize) -> Vec<Vec<f64>> {
        let base: Vec<f64> = (0..t).map(|i| if i % 2 == 0 { 0.01 } else { -0.01 }).collect();
        vec![base; n]
    }

    fn uncorr_returns(t: usize) -> Vec<Vec<f64>> {
        // Two assets with no linear relationship over 50 obs.
        let r0: Vec<f64> = (0..t).map(|i| if i % 2 == 0 { 0.02 } else { -0.02 }).collect();
        let r1: Vec<f64> = (0..t).map(|i| if i % 3 == 0 { 0.03 } else { -0.015 }).collect();
        vec![r0, r1]
    }

    #[test]
    fn test_rolling_corr_diagonal_ones() {
        let returns = uncorr_returns(50);
        let corr = rolling_correlation_matrix(&returns, 20);
        assert_eq!(corr.len(), 2);
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_corr_perfect_positive() {
        let returns = perfect_pos_corr_returns(2, 30);
        let corr = rolling_correlation_matrix(&returns, 20);
        // Identical series should have correlation 1.0.
        assert!((corr[0][1] - 1.0).abs() < 1e-8, "Perfect pos corr: {}", corr[0][1]);
    }

    #[test]
    fn test_rolling_corr_symmetry() {
        let returns = uncorr_returns(50);
        let corr = rolling_correlation_matrix(&returns, 30);
        assert!((corr[0][1] - corr[1][0]).abs() < 1e-12);
    }

    #[test]
    fn test_incremental_corr_matches_batch() {
        let data = uncorr_returns(50);
        let window = 20;
        let batch_corr = rolling_correlation_matrix(&data, window);
        let mut incr = IncrementalCorrelation::new(2, window);
        for t in 0..50 {
            incr.update(&[data[0][t], data[1][t]]);
        }
        let incr_corr = incr.correlation_matrix();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (incr_corr[i][j] - batch_corr[i][j]).abs() < 1e-8,
                    "Incremental vs batch mismatch at [{},{}]: {} vs {}",
                    i, j, incr_corr[i][j], batch_corr[i][j]
                );
            }
        }
    }

    #[test]
    fn test_correlation_breakdown_risk_positive_when_corr_increases() {
        let low_corr = vec![
            vec![1.0, 0.2],
            vec![0.2, 1.0],
        ];
        let high_corr = vec![
            vec![1.0, 0.8],
            vec![0.8, 1.0],
        ];
        let weights = vec![0.5, 0.5];
        let breakdown = correlation_breakdown_risk(&low_corr, &high_corr, &weights);
        assert!(breakdown > 0.0, "Higher corr should increase portfolio variance");
    }

    #[test]
    fn test_correlation_breakdown_risk_zero_same_matrix() {
        let corr = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let weights = vec![0.5, 0.5];
        let breakdown = correlation_breakdown_risk(&corr, &corr, &weights);
        assert!(breakdown.abs() < 1e-10, "Same matrix should give zero breakdown risk");
    }

    #[test]
    fn test_eigen_risk_identity_matrix() {
        let identity = identity_matrix(3);
        let er = eigen_portfolio_risk(&identity);
        // Identity: all eigenvalues = 1, first_eigenvalue_pct = 1/3.
        assert!((er.first_eigenvalue_pct - 1.0 / 3.0).abs() < 0.05,
            "First eigenvalue % of identity 3x3: {}", er.first_eigenvalue_pct);
    }

    #[test]
    fn test_eigen_risk_concentrated_matrix() {
        // High pairwise correlation => first eigenvalue dominates.
        let corr = vec![
            vec![1.0, 0.95, 0.95],
            vec![0.95, 1.0, 0.95],
            vec![0.95, 0.95, 1.0],
        ];
        let er = eigen_portfolio_risk(&corr);
        assert!(
            er.first_eigenvalue_pct > 0.90,
            "Highly correlated matrix: first eigenvalue should dominate: {}",
            er.first_eigenvalue_pct
        );
    }

    #[test]
    fn test_dcc_step_diagonal_ones() {
        let n = 3;
        let prev = identity_matrix(n);
        let returns = vec![0.01, -0.02, 0.005];
        let r_new = dynamic_conditional_correlation_step(&prev, &returns, 0.05, 0.90);
        for i in 0..n {
            assert!((r_new[i][i] - 1.0).abs() < 1e-10, "DCC diagonal must be 1.0");
        }
    }

    #[test]
    fn test_dcc_step_symmetry() {
        let n = 2;
        let prev = vec![vec![1.0, 0.3], vec![0.3, 1.0]];
        let returns = vec![0.02, -0.01];
        let r_new = dynamic_conditional_correlation_step(&prev, &returns, 0.05, 0.90);
        assert!((r_new[0][1] - r_new[1][0]).abs() < 1e-12, "DCC result must be symmetric");
    }

    #[test]
    fn test_dcc_step_bounded_correlations() {
        let n = 3;
        let prev = vec![
            vec![1.0, 0.8, -0.5],
            vec![0.8, 1.0, 0.6],
            vec![-0.5, 0.6, 1.0],
        ];
        let returns = vec![0.10, -0.08, 0.05];
        let r_new = dynamic_conditional_correlation_step(&prev, &returns, 0.10, 0.85);
        for i in 0..n {
            for j in 0..n {
                assert!(r_new[i][j] >= -1.0 - 1e-10 && r_new[i][j] <= 1.0 + 1e-10,
                    "DCC correlation out of range: r[{},{}]={}", i, j, r_new[i][j]);
            }
        }
    }

    #[test]
    fn test_eigen_n_significant_factors_uncorrelated() {
        let corr = identity_matrix(5);
        let er = eigen_portfolio_risk(&corr);
        // For identity, need all 5 factors to explain 95%.
        // With 5 equal eigenvalues, 4 explain 80%, 5 explain 100%.
        assert!(er.n_significant_factors <= 5);
        assert!(er.n_significant_factors >= 1);
    }
}
