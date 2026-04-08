use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Linear Regression (OLS)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct OlsResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub std_error: f64,
    pub slope_std_error: f64,
    pub intercept_std_error: f64,
    pub t_stat_slope: f64,
    pub t_stat_intercept: f64,
    pub residuals: Vec<f64>,
}

pub fn ols_regression(x: &[f64], y: &[f64]) -> OlsResult {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let syy: f64 = y.iter().map(|&yi| yi * yi).sum();

    let denom = n * sxx - sx * sx;
    let slope = if denom.abs() > 1e-15 { (n * sxy - sx * sy) / denom } else { 0.0 };
    let intercept = (sy - slope * sx) / n;

    let mut residuals = Vec::with_capacity(x.len());
    let mut ss_res = 0.0;
    let y_mean = sy / n;
    let mut ss_tot = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let pred = slope * xi + intercept;
        let res = yi - pred;
        residuals.push(res);
        ss_res += res * res;
        ss_tot += (yi - y_mean) * (yi - y_mean);
    }

    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let mse = if n > 2.0 { ss_res / (n - 2.0) } else { 0.0 };
    let std_error = mse.sqrt();

    let x_mean = sx / n;
    let sxx_centered: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
    let slope_se = if sxx_centered > 1e-15 { (mse / sxx_centered).sqrt() } else { 0.0 };
    let intercept_se = (mse * (1.0 / n + x_mean * x_mean / sxx_centered.max(1e-15))).sqrt();

    let t_slope = if slope_se > 1e-15 { slope / slope_se } else { 0.0 };
    let t_intercept = if intercept_se > 1e-15 { intercept / intercept_se } else { 0.0 };

    OlsResult {
        slope,
        intercept,
        r_squared,
        std_error,
        slope_std_error: slope_se,
        intercept_std_error: intercept_se,
        t_stat_slope: t_slope,
        t_stat_intercept: t_intercept,
        residuals,
    }
}

pub fn ols_predict(result: &OlsResult, x: f64) -> f64 {
    result.slope * x + result.intercept
}

pub fn ols_predict_batch(result: &OlsResult, x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| result.slope * xi + result.intercept).collect()
}

// ---------------------------------------------------------------------------
// Theil-Sen Estimator (robust median-based)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct TheilSenResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

pub fn theil_sen_regression(x: &[f64], y: &[f64]) -> TheilSenResult {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    // Compute all pairwise slopes
    let mut slopes = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx.abs() > 1e-15 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }

    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let slope = if slopes.is_empty() {
        0.0
    } else {
        median_sorted(&slopes)
    };

    // Intercept: median of y_i - slope * x_i
    let mut intercepts: Vec<f64> = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| yi - slope * xi)
        .collect();
    intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let intercept = median_sorted(&intercepts);

    // R-squared
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (yi - slope * xi - intercept).powi(2))
        .sum();
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };

    TheilSenResult { slope, intercept, r_squared }
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

// ---------------------------------------------------------------------------
// RANSAC Regression
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RansacResult {
    pub slope: f64,
    pub intercept: f64,
    pub inlier_count: usize,
    pub inlier_indices: Vec<usize>,
    pub r_squared: f64,
}

pub fn ransac_regression(
    x: &[f64],
    y: &[f64],
    threshold: f64,
    max_iterations: usize,
    min_inliers: usize,
) -> RansacResult {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    assert!(n >= 2);

    let mut best_inliers = Vec::new();
    let mut best_slope = 0.0;
    let mut best_intercept = 0.0;

    // Simple deterministic RANSAC using all pairs (or random-like via stride)
    let stride = (n as f64 / max_iterations as f64).max(1.0) as usize;

    for iter in 0..max_iterations {
        // Pick two points
        let i1 = (iter * 7 + 3) % n;
        let i2 = (iter * 13 + 7) % n;
        if i1 == i2 { continue; }

        let dx = x[i2] - x[i1];
        if dx.abs() < 1e-15 { continue; }

        let slope = (y[i2] - y[i1]) / dx;
        let intercept = y[i1] - slope * x[i1];

        // Count inliers
        let mut inliers = Vec::new();
        for j in 0..n {
            let pred = slope * x[j] + intercept;
            if (y[j] - pred).abs() < threshold {
                inliers.push(j);
            }
        }

        if inliers.len() > best_inliers.len() && inliers.len() >= min_inliers {
            best_inliers = inliers;
            best_slope = slope;
            best_intercept = intercept;
        }
    }

    // Refit on inliers
    if best_inliers.len() >= 2 {
        let inlier_x: Vec<f64> = best_inliers.iter().map(|&i| x[i]).collect();
        let inlier_y: Vec<f64> = best_inliers.iter().map(|&i| y[i]).collect();
        let result = ols_regression(&inlier_x, &inlier_y);
        best_slope = result.slope;
        best_intercept = result.intercept;
    }

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (yi - best_slope * xi - best_intercept).powi(2))
        .sum();
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };

    RansacResult {
        slope: best_slope,
        intercept: best_intercept,
        inlier_count: best_inliers.len(),
        inlier_indices: best_inliers,
        r_squared,
    }
}

// ---------------------------------------------------------------------------
// Polynomial Regression
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PolynomialResult {
    pub coefficients: Vec<f64>, // [a0, a1, a2, ...] where y = a0 + a1*x + a2*x^2 + ...
    pub r_squared: f64,
    pub std_error: f64,
    pub degree: usize,
}

pub fn polynomial_regression(x: &[f64], y: &[f64], degree: usize) -> PolynomialResult {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let m = degree + 1;

    // Build normal equations: (X^T X) * beta = X^T y
    // X is n x m Vandermonde matrix
    let mut xtx = vec![0.0; m * m];
    let mut xty = vec![0.0; m];

    for i in 0..n {
        let mut xi_powers = vec![1.0; m];
        for k in 1..m {
            xi_powers[k] = xi_powers[k - 1] * x[i];
        }
        for r in 0..m {
            for c in 0..m {
                xtx[r * m + c] += xi_powers[r] * xi_powers[c];
            }
            xty[r] += xi_powers[r] * y[i];
        }
    }

    // Solve via Gaussian elimination
    let coefficients = solve_linear_system(&xtx, &xty, m);

    // Compute R-squared
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let mut pred = 0.0;
        let mut xp = 1.0;
        for k in 0..m {
            pred += coefficients[k] * xp;
            xp *= x[i];
        }
        ss_res += (y[i] - pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let df = if n > m { n - m } else { 1 };
    let std_error = (ss_res / df as f64).sqrt();

    PolynomialResult { coefficients, r_squared, std_error, degree }
}

pub fn polynomial_predict(result: &PolynomialResult, x: f64) -> f64 {
    let mut y = 0.0;
    let mut xp = 1.0;
    for &c in &result.coefficients {
        y += c * xp;
        xp *= x;
    }
    y
}

pub fn polynomial_predict_batch(result: &PolynomialResult, x: &[f64]) -> Vec<f64> {
    x.iter().map(|&xi| polynomial_predict(result, xi)).collect()
}

fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a[r * n + c];
        }
        aug[r * (n + 1) + n] = b[r];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for c in 0..=n {
                let tmp = aug[col * (n + 1) + c];
                aug[col * (n + 1) + c] = aug[max_row * (n + 1) + c];
                aug[max_row * (n + 1) + c] = tmp;
            }
        }
        let pivot = aug[col * (n + 1) + col];
        if pivot.abs() < 1e-15 { continue; }
        for c in col..=n {
            aug[col * (n + 1) + c] /= pivot;
        }
        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row * (n + 1) + col];
            for c in col..=n {
                aug[row * (n + 1) + c] -= factor * aug[col * (n + 1) + c];
            }
        }
    }

    let mut result = vec![0.0; n];
    for r in 0..n {
        result[r] = aug[r * (n + 1) + n];
    }
    result
}

// ---------------------------------------------------------------------------
// Ridge Regression
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RidgeResult {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub r_squared: f64,
    pub lambda: f64,
}

pub fn ridge_regression(x_matrix: &[Vec<f64>], y: &[f64], lambda: f64) -> RidgeResult {
    let n = y.len();
    let p = if n > 0 { x_matrix[0].len() } else { 0 };

    // Center data
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut x_means = vec![0.0; p];
    for j in 0..p {
        for i in 0..n {
            x_means[j] += x_matrix[i][j];
        }
        x_means[j] /= n as f64;
    }

    // Build X^T X + lambda * I
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];

    for i in 0..n {
        for r in 0..p {
            let xr = x_matrix[i][r] - x_means[r];
            let yc = y[i] - y_mean;
            xty[r] += xr * yc;
            for c in 0..p {
                let xc = x_matrix[i][c] - x_means[c];
                xtx[r * p + c] += xr * xc;
            }
        }
    }

    // Add lambda * I
    for j in 0..p {
        xtx[j * p + j] += lambda;
    }

    let coefficients = solve_linear_system(&xtx, &xty, p);

    // Compute intercept
    let mut intercept = y_mean;
    for j in 0..p {
        intercept -= coefficients[j] * x_means[j];
    }

    // R-squared
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for i in 0..n {
        let mut pred = intercept;
        for j in 0..p {
            pred += coefficients[j] * x_matrix[i][j];
        }
        ss_res += (y[i] - pred).powi(2);
        ss_tot += (y[i] - y_mean).powi(2);
    }
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };

    RidgeResult { coefficients, intercept, r_squared, lambda }
}

// ---------------------------------------------------------------------------
// Exponential Regression: y = a * exp(b * x)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ExponentialResult {
    pub a: f64,
    pub b: f64,
    pub r_squared: f64,
}

pub fn exponential_regression(x: &[f64], y: &[f64]) -> ExponentialResult {
    assert_eq!(x.len(), y.len());
    // Linearize: ln(y) = ln(a) + b*x
    let log_y: Vec<f64> = y.iter().map(|&yi| {
        if yi > 0.0 { yi.ln() } else { 1e-15_f64.ln() }
    }).collect();

    let result = ols_regression(x, &log_y);
    let a = result.intercept.exp();
    let b = result.slope;

    // Compute actual R-squared
    let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let pred = a * (b * xi).exp();
        ss_res += (yi - pred).powi(2);
        ss_tot += (yi - y_mean).powi(2);
    }
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };

    ExponentialResult { a, b, r_squared }
}

pub fn exponential_predict(result: &ExponentialResult, x: f64) -> f64 {
    result.a * (result.b * x).exp()
}

// ---------------------------------------------------------------------------
// Logarithmic Regression: y = a + b * ln(x)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct LogarithmicResult {
    pub a: f64,
    pub b: f64,
    pub r_squared: f64,
}

pub fn logarithmic_regression(x: &[f64], y: &[f64]) -> LogarithmicResult {
    assert_eq!(x.len(), y.len());
    let log_x: Vec<f64> = x.iter().map(|&xi| {
        if xi > 0.0 { xi.ln() } else { 1e-15_f64.ln() }
    }).collect();

    let result = ols_regression(&log_x, y);
    LogarithmicResult {
        a: result.intercept,
        b: result.slope,
        r_squared: result.r_squared,
    }
}

pub fn logarithmic_predict(result: &LogarithmicResult, x: f64) -> f64 {
    result.a + result.b * x.max(1e-15).ln()
}

// ---------------------------------------------------------------------------
// Power Regression: y = a * x^b
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PowerResult {
    pub a: f64,
    pub b: f64,
    pub r_squared: f64,
}

pub fn power_regression(x: &[f64], y: &[f64]) -> PowerResult {
    assert_eq!(x.len(), y.len());
    let log_x: Vec<f64> = x.iter().map(|&xi| xi.max(1e-15).ln()).collect();
    let log_y: Vec<f64> = y.iter().map(|&yi| yi.max(1e-15).ln()).collect();

    let result = ols_regression(&log_x, &log_y);
    let a = result.intercept.exp();
    let b = result.slope;

    let y_mean: f64 = y.iter().sum::<f64>() / y.len() as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let pred = a * xi.max(1e-15).powf(b);
        ss_res += (yi - pred).powi(2);
        ss_tot += (yi - y_mean).powi(2);
    }
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };

    PowerResult { a, b, r_squared }
}

pub fn power_predict(result: &PowerResult, x: f64) -> f64 {
    result.a * x.max(1e-15).powf(result.b)
}

// ---------------------------------------------------------------------------
// Rolling Regression
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RollingRegressionOutput {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub std_error: f64,
}

#[derive(Debug, Clone)]
pub struct RollingRegression {
    period: usize,
    x_buffer: VecDeque<f64>,
    y_buffer: VecDeque<f64>,
    x_counter: f64,
}

impl RollingRegression {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            x_buffer: VecDeque::with_capacity(period + 1),
            y_buffer: VecDeque::with_capacity(period + 1),
            x_counter: 0.0,
        }
    }

    pub fn update(&mut self, y: f64) -> Option<RollingRegressionOutput> {
        self.x_buffer.push_back(self.x_counter);
        self.y_buffer.push_back(y);
        self.x_counter += 1.0;

        if self.x_buffer.len() > self.period {
            self.x_buffer.pop_front();
            self.y_buffer.pop_front();
        }

        if self.x_buffer.len() == self.period {
            let n = self.period as f64;
            let x: Vec<f64> = self.x_buffer.iter().cloned().collect();
            let y: Vec<f64> = self.y_buffer.iter().cloned().collect();

            let sx: f64 = x.iter().sum();
            let sy: f64 = y.iter().sum();
            let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
            let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();

            let denom = n * sxx - sx * sx;
            let slope = if denom.abs() > 1e-15 { (n * sxy - sx * sy) / denom } else { 0.0 };
            let intercept = (sy - slope * sx) / n;

            let y_mean = sy / n;
            let mut ss_res = 0.0;
            let mut ss_tot = 0.0;
            for (&xi, &yi) in x.iter().zip(y.iter()) {
                let pred = slope * xi + intercept;
                ss_res += (yi - pred).powi(2);
                ss_tot += (yi - y_mean).powi(2);
            }
            let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
            let std_error = if n > 2.0 { (ss_res / (n - 2.0)).sqrt() } else { 0.0 };

            Some(RollingRegressionOutput { slope, intercept, r_squared, std_error })
        } else {
            None
        }
    }

    pub fn update_xy(&mut self, x: f64, y: f64) -> Option<RollingRegressionOutput> {
        self.x_buffer.push_back(x);
        self.y_buffer.push_back(y);

        if self.x_buffer.len() > self.period {
            self.x_buffer.pop_front();
            self.y_buffer.pop_front();
        }

        if self.x_buffer.len() == self.period {
            let xv: Vec<f64> = self.x_buffer.iter().cloned().collect();
            let yv: Vec<f64> = self.y_buffer.iter().cloned().collect();
            let res = ols_regression(&xv, &yv);
            Some(RollingRegressionOutput {
                slope: res.slope,
                intercept: res.intercept,
                r_squared: res.r_squared,
                std_error: res.std_error,
            })
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.x_buffer.clear();
        self.y_buffer.clear();
        self.x_counter = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Deming Regression (Orthogonal / Total Least Squares)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct DemingResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

pub fn deming_regression(x: &[f64], y: &[f64], lambda: f64) -> DemingResult {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - x_mean;
        let dy = yi - y_mean;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }
    sxx /= n - 1.0;
    syy /= n - 1.0;
    sxy /= n - 1.0;

    let diff = syy - lambda * sxx;
    let slope = (diff + (diff * diff + 4.0 * lambda * sxy * sxy).sqrt()) / (2.0 * sxy);
    let intercept = y_mean - slope * x_mean;

    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (yi - slope * xi - intercept).powi(2))
        .sum();
    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };

    DemingResult { slope, intercept, r_squared }
}

// ---------------------------------------------------------------------------
// Passing-Bablok Regression
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct PassingBablokResult {
    pub slope: f64,
    pub intercept: f64,
    pub slope_ci: (f64, f64),
    pub intercept_ci: (f64, f64),
}

pub fn passing_bablok_regression(x: &[f64], y: &[f64]) -> PassingBablokResult {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    // Compute all pairwise slopes S_ij = (y_j - y_i) / (x_j - x_i) for i < j
    let mut slopes = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            if dx.abs() > 1e-15 {
                slopes.push((y[j] - y[i]) / dx);
            }
        }
    }

    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Count slopes < -1
    let k = slopes.iter().filter(|&&s| s < -1.0).count();

    // Shift index
    let nn = slopes.len();
    let slope = if nn == 0 {
        1.0
    } else {
        let idx = k + nn / 2;
        if nn % 2 == 0 && idx > 0 {
            (slopes[idx.min(nn - 1)] + slopes[(idx - 1).min(nn - 1)]) / 2.0
        } else {
            slopes[idx.min(nn - 1)]
        }
    };

    // Intercept: median of y_i - slope * x_i
    let mut intercepts: Vec<f64> = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| yi - slope * xi)
        .collect();
    intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let intercept = median_sorted(&intercepts);

    // Confidence intervals (approximate using Jackknife)
    let z = 1.96; // 95% CI
    let c_alpha = z * ((n * (n - 1) * (2 * n + 5)) as f64 / 18.0).sqrt();
    let m1 = ((nn as f64 - c_alpha) / 2.0).round() as usize;
    let m2 = (nn as f64 / 2.0 + c_alpha / 2.0).round() as usize;

    let slope_ci = if nn > 0 {
        (slopes[m1.min(nn - 1)], slopes[m2.min(nn - 1)])
    } else {
        (slope, slope)
    };

    // Intercept CI
    let intercept_lo = intercepts.first().copied().unwrap_or(intercept);
    let intercept_hi = intercepts.last().copied().unwrap_or(intercept);

    PassingBablokResult {
        slope,
        intercept,
        slope_ci,
        intercept_ci: (intercept_lo, intercept_hi),
    }
}

// ---------------------------------------------------------------------------
// Quantile Regression (simplex method approximation)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct QuantileResult {
    pub slope: f64,
    pub intercept: f64,
    pub quantile: f64,
}

pub fn quantile_regression(x: &[f64], y: &[f64], tau: f64) -> QuantileResult {
    assert_eq!(x.len(), y.len());
    assert!(tau > 0.0 && tau < 1.0);
    let n = x.len();

    // Iteratively reweighted least squares (IRLS) approach
    let mut slope = 0.0;
    let mut intercept = {
        let mut sorted_y = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((n as f64 * tau) as usize).min(n - 1);
        sorted_y[idx]
    };

    for _iter in 0..100 {
        // Compute weights
        let mut w_sum_xx = 0.0;
        let mut w_sum_xy = 0.0;
        let mut w_sum_x = 0.0;
        let mut w_sum_y = 0.0;
        let mut w_sum = 0.0;

        for i in 0..n {
            let residual = y[i] - slope * x[i] - intercept;
            let weight = if residual.abs() > 1e-10 {
                if residual > 0.0 { tau } else { 1.0 - tau }
            } else {
                0.5
            };
            let w = weight / residual.abs().max(1e-10);

            w_sum += w;
            w_sum_x += w * x[i];
            w_sum_y += w * y[i];
            w_sum_xx += w * x[i] * x[i];
            w_sum_xy += w * x[i] * y[i];
        }

        let denom = w_sum * w_sum_xx - w_sum_x * w_sum_x;
        if denom.abs() > 1e-15 {
            slope = (w_sum * w_sum_xy - w_sum_x * w_sum_y) / denom;
            intercept = (w_sum_y - slope * w_sum_x) / w_sum;
        }
    }

    QuantileResult { slope, intercept, quantile: tau }
}

pub fn quantile_predict(result: &QuantileResult, x: f64) -> f64 {
    result.slope * x + result.intercept
}

// ---------------------------------------------------------------------------
// Multi-variate OLS
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MultiOlsResult {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub std_error: f64,
    pub residuals: Vec<f64>,
}

pub fn multi_ols_regression(x_matrix: &[Vec<f64>], y: &[f64]) -> MultiOlsResult {
    let n = y.len();
    let p = if n > 0 { x_matrix[0].len() } else { 0 };
    let m = p + 1; // including intercept

    // Build X^T X and X^T y (with intercept column)
    let mut xtx = vec![0.0; m * m];
    let mut xty = vec![0.0; m];

    for i in 0..n {
        let mut row = vec![1.0]; // intercept
        row.extend_from_slice(&x_matrix[i]);

        for r in 0..m {
            for c in 0..m {
                xtx[r * m + c] += row[r] * row[c];
            }
            xty[r] += row[r] * y[i];
        }
    }

    let beta = solve_linear_system(&xtx, &xty, m);
    let intercept = beta[0];
    let coefficients: Vec<f64> = beta[1..].to_vec();

    // Residuals and R-squared
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let mut pred = intercept;
        for j in 0..p {
            pred += coefficients[j] * x_matrix[i][j];
        }
        let res = y[i] - pred;
        residuals.push(res);
        ss_res += res * res;
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }

    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let adj_r_squared = if n > m && ss_tot > 1e-15 {
        1.0 - (1.0 - r_squared) * (n as f64 - 1.0) / (n as f64 - m as f64)
    } else {
        r_squared
    };
    let std_error = if n > m { (ss_res / (n - m) as f64).sqrt() } else { 0.0 };

    MultiOlsResult {
        coefficients,
        intercept,
        r_squared,
        adj_r_squared,
        std_error,
        residuals,
    }
}

// ---------------------------------------------------------------------------
// Batch convenience functions
// ---------------------------------------------------------------------------
pub fn linear_regression_from_y(y: &[f64]) -> OlsResult {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    ols_regression(&x, y)
}

pub fn rolling_regression_batch(y: &[f64], period: usize) -> Vec<Option<RollingRegressionOutput>> {
    let mut rr = RollingRegression::new(period);
    y.iter().map(|&v| rr.update(v)).collect()
}

pub fn rolling_slope(y: &[f64], period: usize) -> Vec<Option<f64>> {
    rolling_regression_batch(y, period)
        .into_iter()
        .map(|opt| opt.map(|r| r.slope))
        .collect()
}

pub fn rolling_r_squared(y: &[f64], period: usize) -> Vec<Option<f64>> {
    rolling_regression_batch(y, period)
        .into_iter()
        .map(|opt| opt.map(|r| r.r_squared))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = ols_regression(&x, &y);
        assert!((result.slope - 2.0).abs() < 1e-10);
        assert!((result.intercept - 0.0).abs() < 1e-10);
        assert!((result.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ols_with_noise() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let result = ols_regression(&x, &y);
        assert!((result.slope - 2.0).abs() < 0.2);
        assert!(result.r_squared > 0.99);
    }

    #[test]
    fn test_theil_sen() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = theil_sen_regression(&x, &y);
        assert!((result.slope - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ransac() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 3.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 100.0]; // outlier
        let result = ransac_regression(&x, &y, 1.0, 100, 3);
        assert!((result.slope - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_polynomial() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let result = polynomial_regression(&x, &y, 2);
        assert!(result.r_squared > 0.99);
        let pred = polynomial_predict(&result, 3.0);
        assert!((pred - 9.0).abs() < 0.1);
    }

    #[test]
    fn test_exponential() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (0.5 * xi).exp()).collect();
        let result = exponential_regression(&x, &y);
        assert!((result.a - 2.0).abs() < 0.1);
        assert!((result.b - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_logarithmic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| 3.0 + 2.0 * xi.ln()).collect();
        let result = logarithmic_regression(&x, &y);
        assert!((result.a - 3.0).abs() < 0.1);
        assert!((result.b - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_power() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi.powf(3.0)).collect();
        let result = power_regression(&x, &y);
        assert!((result.b - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_rolling_regression() {
        let y: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
        let results = rolling_regression_batch(&y, 5);
        for r in results.iter().flatten() {
            assert!((r.slope - 2.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_deming() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = deming_regression(&x, &y, 1.0);
        assert!((result.slope - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_quantile() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = quantile_regression(&x, &y, 0.5);
        assert!((result.slope - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_ridge() {
        let x_matrix = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
        ];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = ridge_regression(&x_matrix, &y, 0.01);
        assert!((result.coefficients[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_multi_ols() {
        let x_matrix = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
            vec![5.0, 5.0],
        ];
        let y = vec![4.0, 8.0, 12.0, 16.0, 20.0]; // y = 2*x1 + 2*x2
        let result = multi_ols_regression(&x_matrix, &y);
        assert!(result.r_squared > 0.99);
    }

    #[test]
    fn test_passing_bablok() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 4.0, 5.9, 8.1, 9.9];
        let result = passing_bablok_regression(&x, &y);
        assert!((result.slope - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_linear_from_y() {
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let result = linear_regression_from_y(&y);
        assert!((result.slope - 2.0).abs() < 1e-10);
        assert!((result.intercept - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_slope() {
        let y: Vec<f64> = (0..20).map(|i| 3.0 * i as f64).collect();
        let slopes = rolling_slope(&y, 5);
        for s in slopes.iter().flatten() {
            assert!((s - 3.0).abs() < 1e-6);
        }
    }
}
