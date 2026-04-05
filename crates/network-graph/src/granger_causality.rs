use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a pairwise Granger causality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerResult {
    /// Potential cause (Y Granger-causes X).
    pub cause: String,
    /// Potential effect.
    pub effect: String,
    /// VAR lag order used.
    pub lag_order: usize,
    /// F-statistic from the F-test.
    pub f_stat: f64,
    /// Approximate p-value (F distribution with (p, T-2p-1) df).
    pub p_value: f64,
    /// True if Granger causality is significant at 5% level.
    pub significant: bool,
}

/// Directed Granger causality graph: edges from cause → effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerDAGEdge {
    pub cause: String,
    pub effect: String,
    pub lag_order: usize,
    pub f_stat: f64,
    pub p_value: f64,
}

/// Build pairwise Granger causality tests for all pairs in the return series.
///
/// # Arguments
/// * `returns` — map from symbol to return series.
/// * `max_lag` — maximum VAR lag order to test (typically 1–5).
///
/// Returns all significant Granger causality edges (p < 0.05).
pub fn granger_causality_dag(
    returns: &HashMap<String, Vec<f64>>,
    max_lag: usize,
) -> Vec<GrangerDAGEdge> {
    let symbols: Vec<String> = {
        let mut v: Vec<String> = returns.keys().cloned().collect();
        v.sort();
        v
    };

    let mut edges = Vec::new();

    for i in 0..symbols.len() {
        for j in 0..symbols.len() {
            if i == j {
                continue;
            }
            let x = &returns[&symbols[i]]; // potential effect
            let y = &returns[&symbols[j]]; // potential cause

            if let Some(result) = granger_test(&symbols[j], &symbols[i], y, x, max_lag) {
                if result.significant {
                    edges.push(GrangerDAGEdge {
                        cause: result.cause,
                        effect: result.effect,
                        lag_order: result.lag_order,
                        f_stat: result.f_stat,
                        p_value: result.p_value,
                    });
                }
            }
        }
    }

    // Sort by F-statistic descending (strongest causal links first).
    edges.sort_unstable_by(|a, b| {
        b.f_stat
            .partial_cmp(&a.f_stat)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    edges
}

/// Test whether Y Granger-causes X at lag order `p`.
///
/// Procedure:
/// 1. Fit restricted AR(p) of X on X only: RSS_restricted.
/// 2. Fit unrestricted VAR(p) of X on X and Y lags: RSS_unrestricted.
/// 3. F-stat = ((RSS_r - RSS_u) / p) / (RSS_u / (T - 2p - 1))
/// 4. Approximate p-value from F(p, T-2p-1) distribution.
pub fn granger_test(
    cause_sym: &str,
    effect_sym: &str,
    y: &[f64],
    x: &[f64],
    max_lag: usize,
) -> Option<GrangerResult> {
    let t = x.len().min(y.len());
    let p = max_lag.clamp(1, 5);

    if t < 2 * p + 10 {
        return None;
    }

    // Build design matrices.
    // Restricted: X_t ~ X_{t-1}, ..., X_{t-p}  (intercept + p regressors)
    // Unrestricted: X_t ~ X_{t-1}, ..., X_{t-p}, Y_{t-1}, ..., Y_{t-p}

    let n_obs = t - p;

    // Dependent variable: x[p..t]
    let dep: Vec<f64> = x[p..t].to_vec();

    // Restricted regressors: intercept + p lags of x.
    let mut restr_x: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let mut row = vec![1.0_f64]; // intercept
        for lag in 1..=p {
            row.push(x[p + i - lag]);
        }
        restr_x.push(row);
    }

    // Unrestricted regressors: intercept + p lags of x + p lags of y.
    let mut unres_x: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    for i in 0..n_obs {
        let mut row = vec![1.0_f64]; // intercept
        for lag in 1..=p {
            row.push(x[p + i - lag]);
        }
        for lag in 1..=p {
            row.push(y[p + i - lag]);
        }
        unres_x.push(row);
    }

    let rss_r = ols_rss(&dep, &restr_x)?;
    let rss_u = ols_rss(&dep, &unres_x)?;

    if rss_u < 1e-15 {
        return None; // perfect fit, degenerate
    }

    let df1 = p as f64;
    let df2 = (n_obs as f64 - 2.0 * p as f64 - 1.0).max(1.0);

    let f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2);

    // Approximate p-value using the F-distribution CDF.
    let p_value = f_pvalue(f_stat, df1, df2);
    let significant = p_value < 0.05;

    Some(GrangerResult {
        cause: cause_sym.to_string(),
        effect: effect_sym.to_string(),
        lag_order: p,
        f_stat,
        p_value,
        significant,
    })
}

/// OLS regression: return the residual sum of squares.
/// Design matrix `x_mat` is n_obs × n_cols; `y` is n_obs.
fn ols_rss(y: &[f64], x_mat: &[Vec<f64>]) -> Option<f64> {
    let n = y.len();
    let k = x_mat.first().map(|r| r.len()).unwrap_or(0);
    if n <= k || k == 0 {
        return None;
    }

    // X^T X (k×k) and X^T y (k×1)
    let mut xtx = vec![vec![0.0_f64; k]; k];
    let mut xty = vec![0.0_f64; k];

    for i in 0..n {
        let row = &x_mat[i];
        for a in 0..k {
            xty[a] += row[a] * y[i];
            for b in 0..k {
                xtx[a][b] += row[a] * row[b];
            }
        }
    }

    // Solve via Gaussian elimination: XtX * beta = Xty.
    let beta = solve_linear(&xtx, &xty)?;

    // Compute RSS.
    let rss = (0..n)
        .map(|i| {
            let fitted: f64 = (0..k).map(|j| beta[j] * x_mat[i][j]).sum();
            (y[i] - fitted).powi(2)
        })
        .sum::<f64>();

    Some(rss)
}

/// Solve Ax = b via Gaussian elimination. Returns None if singular.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    // Augmented matrix.
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivoting.
        let pivot = (col..n).max_by(|&r1, &r2| {
            aug[r1][col]
                .abs()
                .partial_cmp(&aug[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        aug.swap(col, pivot);

        let diag = aug[col][col];
        if diag.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / diag;
        for val in aug[col].iter_mut() {
            *val *= inv;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = aug[row][col];
            for k in 0..=n {
                let sub = f * aug[col][k];
                aug[row][k] -= sub;
            }
        }
    }

    Some(aug.into_iter().map(|row| row[n]).collect())
}

/// Approximate p-value for F(df1, df2) distribution using a numerical approximation.
///
/// Uses a regularised incomplete beta function approximation.
fn f_pvalue(f: f64, df1: f64, df2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    // Transform F to Beta variable: x = df1*F / (df1*F + df2)
    let x = (df1 * f) / (df1 * f + df2);
    // P(F > f) = I_{1-x}(df2/2, df1/2) ≈ 1 - I_x(df1/2, df2/2)
    1.0 - regularised_incomplete_beta(x, df1 / 2.0, df2 / 2.0)
}

/// Regularised incomplete beta function I_x(a,b) via continued fraction (Lentz method).
/// Sufficient accuracy for statistical p-value computation.
fn regularised_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation for numerical stability.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularised_incomplete_beta(1.0 - x, b, a);
    }

    // Log of the beta function prefix.
    let log_prefix = a * x.ln() + b * (1.0 - x).ln() - log_beta(a, b);

    // Continued fraction via modified Lentz.
    let cf = continued_fraction_beta(x, a, b);
    (log_prefix.exp() * cf / a).clamp(0.0, 1.0)
}

fn continued_fraction_beta(x: f64, a: f64, b: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-7;
    let fpmin = 1e-30;

    let mut c = 1.0_f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < fpmin { d = fpmin; }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        // Even step.
        let aa = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + aa * d;
        if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c;
        if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        h *= d * c;

        // Odd step.
        let aa = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + aa * d;
        if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c;
        if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

fn log_beta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Stirling-series approximation for the log-gamma function.
fn lgamma(x: f64) -> f64 {
    // Use Lanczos approximation.
    let coeffs = [
        76.18009172947146_f64,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let x = x;
    let mut y = x;
    let tmp = x + 5.5;
    let tmp2 = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015_f64;
    for c in &coeffs {
        y += 1.0;
        ser += c / y;
    }
    -tmp2 + (2.5066282746310005 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ar1_series(n: usize, phi: f64, sigma: f64) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut x = vec![0.0_f64; n];
        for i in 1..n {
            x[i] = phi * x[i - 1] + rng.gen::<f64>() * sigma;
        }
        x
    }

    #[test]
    fn test_granger_no_causality() {
        // Two independent AR(1) series should not show Granger causality.
        let x = ar1_series(200, 0.5, 0.01);
        let y = ar1_series(200, 0.5, 0.01);
        let result = granger_test("Y", "X", &y, &x, 2);
        // Should return a result without panic.
        assert!(result.is_some());
    }

    #[test]
    fn test_granger_with_causality() {
        // Y causes X: x[t] = 0.7 * y[t-1] + noise
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = 300;
        let y: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 0.01).collect();
        let mut x = vec![0.0_f64; n];
        for i in 1..n {
            x[i] = 0.7 * y[i - 1] + rng.gen::<f64>() * 0.001;
        }
        let result = granger_test("Y", "X", &y, &x, 2).unwrap();
        // Expect significant causality.
        assert!(result.significant, "expected significant Granger causality, f={:.2} p={:.4}", result.f_stat, result.p_value);
    }

    #[test]
    fn test_granger_dag_returns_vec() {
        let mut returns = HashMap::new();
        returns.insert("A".to_string(), ar1_series(100, 0.3, 0.01));
        returns.insert("B".to_string(), ar1_series(100, 0.3, 0.01));
        let dag = granger_causality_dag(&returns, 2);
        // Just check it doesn't panic and returns a Vec.
        assert!(dag.len() <= 2); // at most A->B and B->A
    }

    #[test]
    fn test_f_pvalue_large_f_small_p() {
        let p = f_pvalue(20.0, 2.0, 100.0);
        assert!(p < 0.05, "large F should give small p, got {}", p);
    }

    #[test]
    fn test_f_pvalue_small_f_large_p() {
        let p = f_pvalue(0.1, 2.0, 100.0);
        assert!(p > 0.5, "small F should give large p, got {}", p);
    }
}
