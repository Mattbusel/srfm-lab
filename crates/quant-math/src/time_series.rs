// time_series.rs — ARMA, GARCH, exponential smoothing, Holt-Winters, ACF, PACF, tests, Kalman

/// Simple Moving Average
pub fn sma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut result = Vec::with_capacity(n);
    let mut sum = 0.0;
    for i in 0..n {
        sum += data[i];
        if i >= window { sum -= data[i - window]; }
        if i + 1 >= window {
            result.push(sum / window as f64);
        } else {
            result.push(f64::NAN);
        }
    }
    result
}

/// Exponential Moving Average
pub fn ema(data: &[f64], alpha: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut current = data[0];
    result.push(current);
    for i in 1..data.len() {
        current = alpha * data[i] + (1.0 - alpha) * current;
        result.push(current);
    }
    result
}

/// EMA with span parameter
pub fn ema_span(data: &[f64], span: f64) -> Vec<f64> {
    ema(data, 2.0 / (span + 1.0))
}

/// Weighted Moving Average
pub fn wma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let denom = (window * (window + 1) / 2) as f64;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 < window {
            result.push(f64::NAN);
        } else {
            let mut sum = 0.0;
            for j in 0..window {
                sum += data[i - window + 1 + j] * (j + 1) as f64;
            }
            result.push(sum / denom);
        }
    }
    result
}

/// Double Exponential Moving Average (DEMA)
pub fn dema(data: &[f64], alpha: f64) -> Vec<f64> {
    let e1 = ema(data, alpha);
    let e2 = ema(&e1.iter().copied().filter(|x| !x.is_nan()).collect::<Vec<_>>(), alpha);
    let offset = e1.len() - e2.len();
    let mut result = vec![f64::NAN; offset];
    for i in 0..e2.len() {
        result.push(2.0 * e1[i + offset] - e2[i]);
    }
    result
}

/// Triple Exponential Moving Average (TEMA)
pub fn tema(data: &[f64], alpha: f64) -> Vec<f64> {
    let e1 = ema(data, alpha);
    let valid1: Vec<f64> = e1.iter().copied().filter(|x| !x.is_nan()).collect();
    let e2 = ema(&valid1, alpha);
    let valid2: Vec<f64> = e2.iter().copied().filter(|x| !x.is_nan()).collect();
    let e3 = ema(&valid2, alpha);

    let off1 = e1.len() - valid1.len();
    let off2 = valid1.len() - valid2.len();
    let off3 = valid2.len() - e3.len();
    let total_offset = off1 + off2 + off3;
    let len = e3.len();

    let mut result = vec![f64::NAN; total_offset];
    for i in 0..len {
        let i1 = i + total_offset;
        let i2 = i + off2 + off3;
        let i3 = i;
        if i1 < e1.len() && i2 < e2.len() && i3 < e3.len() {
            result.push(3.0 * e1[i1] - 3.0 * e2[i2] + e3[i3]);
        }
    }
    result
}

/// Autocorrelation function (ACF)
pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    if var < 1e-30 { return vec![1.0; max_lag + 1]; }

    let mut result = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in lag..n {
            sum += (data[i] - mean) * (data[i - lag] - mean);
        }
        result.push(sum / var);
    }
    result
}

/// Partial autocorrelation function (PACF) via Durbin-Levinson
pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let ac = acf(data, max_lag);
    let mut result = vec![1.0]; // lag 0
    let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];

    for k in 1..=max_lag {
        let mut num = ac[k];
        for j in 1..k {
            num -= phi[k - 1][j] * ac[k - j];
        }
        let mut den = 1.0;
        for j in 1..k {
            den -= phi[k - 1][j] * ac[j];
        }
        phi[k][k] = if den.abs() > 1e-30 { num / den } else { 0.0 };

        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }

        result.push(phi[k][k]);
    }
    result
}

/// Ljung-Box test statistic
pub fn ljung_box(data: &[f64], max_lag: usize) -> (f64, f64) {
    let n = data.len() as f64;
    let ac = acf(data, max_lag);
    let mut q = 0.0;
    for k in 1..=max_lag {
        q += ac[k] * ac[k] / (n - k as f64);
    }
    q *= n * (n + 2.0);
    // p-value from chi-squared with max_lag degrees of freedom
    let p_value = 1.0 - chi2_cdf_approx(q, max_lag as f64);
    (q, p_value)
}

fn chi2_cdf_approx(x: f64, k: f64) -> f64 {
    // Wilson-Hilferty approximation
    let z = ((x / k).powf(1.0 / 3.0) - 1.0 + 2.0 / (9.0 * k)) / (2.0 / (9.0 * k)).sqrt();
    norm_cdf_approx(z)
}

fn norm_cdf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327;
    let p = d * (-x * x / 2.0).exp()
        * (0.3193815 * t - 0.3565638 * t * t + 1.7814779 * t.powi(3)
            - 1.8212560 * t.powi(4) + 1.3302744 * t.powi(5));
    if x >= 0.0 { 1.0 - p } else { p }
}

/// Augmented Dickey-Fuller test (simplified)
pub fn adf_test(data: &[f64], max_lag: usize) -> (f64, bool) {
    let n = data.len();
    if n < max_lag + 3 { return (0.0, false); }

    // Δyₜ = α + β yₜ₋₁ + Σ γᵢ Δyₜ₋ᵢ + εₜ
    let dy: Vec<f64> = (1..n).map(|i| data[i] - data[i - 1]).collect();
    let nd = dy.len();

    // OLS regression
    let p = max_lag + 2; // intercept + y_{t-1} + max_lag lags of Δy
    if nd < p + 1 { return (0.0, false); }

    let nobs = nd - max_lag;
    let mut x_mat = vec![vec![0.0; p]; nobs];
    let mut y_vec = vec![0.0; nobs];

    for t in 0..nobs {
        let tt = t + max_lag;
        y_vec[t] = dy[tt];
        x_mat[t][0] = 1.0; // intercept
        x_mat[t][1] = data[tt]; // y_{t-1}
        for lag in 1..=max_lag {
            x_mat[t][1 + lag] = dy[tt - lag];
        }
    }

    // (X^T X)^{-1} X^T y
    let xtx = mat_mul_ata(&x_mat);
    let xty = mat_mul_atb(&x_mat, &y_vec);
    let beta = solve_symmetric(&xtx, &xty);

    if beta.is_empty() { return (0.0, false); }

    // Residuals
    let mut rss = 0.0;
    for t in 0..nobs {
        let mut pred = 0.0;
        for j in 0..p { pred += x_mat[t][j] * beta[j]; }
        rss += (y_vec[t] - pred).powi(2);
    }
    let sigma2 = rss / (nobs - p) as f64;

    // Standard error of β₁ (coefficient on y_{t-1})
    let inv_xtx = invert_symmetric(&xtx);
    let se = if inv_xtx.len() > 1 { (sigma2 * inv_xtx[1][1]).sqrt() } else { 1.0 };

    let t_stat = if se > 1e-30 { beta[1] / se } else { 0.0 };

    // Critical values (approximate for intercept, no trend)
    // 1%: -3.43, 5%: -2.86, 10%: -2.57
    let reject_5pct = t_stat < -2.86;

    (t_stat, reject_5pct)
}

fn mat_mul_ata(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..m { s += a[k][i] * a[k][j]; }
            result[i][j] = s;
            result[j][i] = s;
        }
    }
    result
}

fn mat_mul_atb(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a[0].len();
    let m = a.len();
    let mut result = vec![0.0; n];
    for j in 0..n {
        for i in 0..m { result[j] += a[i][j] * b[i]; }
    }
    result
}

fn solve_symmetric(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = a[i][j]; }
        aug[i][n] = b[i];
    }
    // Gaussian elimination
    for k in 0..n {
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val { max_val = aug[i][k].abs(); max_row = i; }
        }
        aug.swap(k, max_row);
        if aug[k][k].abs() < 1e-14 { continue; }
        for i in (k + 1)..n {
            let f = aug[i][k] / aug[k][k];
            for j in k..=n { aug[i][j] -= f * aug[k][j]; }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i][n];
        for j in (i + 1)..n { s -= aug[i][j] * x[j]; }
        x[i] = if aug[i][i].abs() > 1e-14 { s / aug[i][i] } else { 0.0 };
    }
    x
}

fn invert_symmetric(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = a[i][j]; }
        aug[i][n + i] = 1.0;
    }
    for k in 0..n {
        let mut max_row = k;
        let mut max_val = aug[k][k].abs();
        for i in (k + 1)..n {
            if aug[i][k].abs() > max_val { max_val = aug[i][k].abs(); max_row = i; }
        }
        aug.swap(k, max_row);
        if aug[k][k].abs() < 1e-14 { continue; }
        let d = aug[k][k];
        for j in 0..(2 * n) { aug[k][j] /= d; }
        for i in 0..n {
            if i == k { continue; }
            let f = aug[i][k];
            for j in 0..(2 * n) { aug[i][j] -= f * aug[k][j]; }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n { inv[i][j] = aug[i][n + j]; }
    }
    inv
}

/// AR model via Burg method
pub struct ArBurg {
    pub coeffs: Vec<f64>,
    pub error_variance: f64,
}

impl ArBurg {
    pub fn fit(data: &[f64], order: usize) -> Self {
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = data.iter().map(|x| x - mean).collect();

        let mut ef: Vec<f64> = centered.clone();
        let mut eb: Vec<f64> = centered.clone();
        let mut a = vec![0.0; order];
        let mut e = centered.iter().map(|x| x * x).sum::<f64>() / n as f64;

        for m in 0..order {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in (m + 1)..n {
                num += 2.0 * ef[i] * eb[i - 1];
                den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
            }
            let k = if den.abs() > 1e-30 { num / den } else { 0.0 };

            // Update coefficients
            let mut new_a = a.clone();
            new_a[m] = k;
            for i in 0..m {
                new_a[i] = a[i] - k * a[m - 1 - i];
            }
            a = new_a;

            // Update forward/backward errors
            let mut new_ef = vec![0.0; n];
            let mut new_eb = vec![0.0; n];
            for i in (m + 1)..n {
                new_ef[i] = ef[i] - k * eb[i - 1];
                new_eb[i] = eb[i - 1] - k * ef[i];
            }
            ef = new_ef;
            eb = new_eb;

            e *= 1.0 - k * k;
        }

        Self { coeffs: a, error_variance: e }
    }

    pub fn predict(&self, data: &[f64], steps: usize) -> Vec<f64> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let p = self.coeffs.len();
        let mut history: Vec<f64> = data.iter().map(|x| x - mean).collect();
        let mut predictions = Vec::with_capacity(steps);

        for _ in 0..steps {
            let n = history.len();
            let mut pred = 0.0;
            for j in 0..p {
                if n > j { pred += self.coeffs[j] * history[n - 1 - j]; }
            }
            predictions.push(pred + mean);
            history.push(pred);
        }
        predictions
    }
}

/// ARMA(p,q) model via OLS (conditional)
pub struct Arma {
    pub ar_coeffs: Vec<f64>,
    pub ma_coeffs: Vec<f64>,
    pub intercept: f64,
    pub residual_variance: f64,
}

impl Arma {
    /// Fit ARMA(p,q) via iterative OLS
    pub fn fit(data: &[f64], p: usize, q: usize) -> Self {
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        let y: Vec<f64> = data.iter().map(|x| x - mean).collect();
        let start = p.max(q);
        if n <= start + 1 {
            return Self { ar_coeffs: vec![0.0; p], ma_coeffs: vec![0.0; q], intercept: mean, residual_variance: 0.0 };
        }

        // Initialize residuals to zero
        let mut residuals = vec![0.0; n];

        let max_iter = 10;
        let mut ar = vec![0.0; p];
        let mut ma = vec![0.0; q];

        for _ in 0..max_iter {
            let nobs = n - start;
            let ncols = p + q;
            if ncols == 0 { break; }
            let mut x_mat = vec![vec![0.0; ncols]; nobs];
            let mut y_vec = vec![0.0; nobs];

            for t in 0..nobs {
                let tt = t + start;
                y_vec[t] = y[tt];
                for j in 0..p {
                    if tt > j { x_mat[t][j] = y[tt - 1 - j]; }
                }
                for j in 0..q {
                    if tt > j { x_mat[t][p + j] = residuals[tt - 1 - j]; }
                }
            }

            let xtx = mat_mul_ata(&x_mat);
            let xty = mat_mul_atb(&x_mat, &y_vec);
            let beta = solve_symmetric(&xtx, &xty);

            for j in 0..p { ar[j] = beta[j]; }
            for j in 0..q { ma[j] = beta[p + j]; }

            // Update residuals
            for t in start..n {
                let mut pred = 0.0;
                for j in 0..p {
                    if t > j { pred += ar[j] * y[t - 1 - j]; }
                }
                for j in 0..q {
                    if t > j { pred += ma[j] * residuals[t - 1 - j]; }
                }
                residuals[t] = y[t] - pred;
            }
        }

        let nobs = n - start;
        let var = residuals[start..].iter().map(|r| r * r).sum::<f64>() / nobs as f64;

        Self { ar_coeffs: ar, ma_coeffs: ma, intercept: mean, residual_variance: var }
    }

    pub fn predict(&self, data: &[f64], steps: usize) -> Vec<f64> {
        let n = data.len();
        let y: Vec<f64> = data.iter().map(|x| x - self.intercept).collect();
        let p = self.ar_coeffs.len();
        let q = self.ma_coeffs.len();

        // Compute residuals for existing data
        let start = p.max(q);
        let mut residuals = vec![0.0; n];
        for t in start..n {
            let mut pred = 0.0;
            for j in 0..p { if t > j { pred += self.ar_coeffs[j] * y[t - 1 - j]; } }
            for j in 0..q { if t > j { pred += self.ma_coeffs[j] * residuals[t - 1 - j]; } }
            residuals[t] = y[t] - pred;
        }

        let mut history = y;
        let mut res_history = residuals;
        let mut predictions = Vec::with_capacity(steps);

        for _ in 0..steps {
            let t = history.len();
            let mut pred = 0.0;
            for j in 0..p { if t > j { pred += self.ar_coeffs[j] * history[t - 1 - j]; } }
            for j in 0..q { if t > j { pred += self.ma_coeffs[j] * res_history[t - 1 - j]; } }
            predictions.push(pred + self.intercept);
            history.push(pred);
            res_history.push(0.0); // future residuals are zero
        }
        predictions
    }
}

/// GARCH(1,1) model
pub struct Garch11 {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub log_likelihood: f64,
}

impl Garch11 {
    /// Fit GARCH(1,1) via simplified MLE
    pub fn fit(returns: &[f64]) -> Self {
        let n = returns.len();
        let mean = returns.iter().sum::<f64>() / n as f64;
        let residuals: Vec<f64> = returns.iter().map(|r| r - mean).collect();
        let var = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;

        // Grid search + refinement for omega, alpha, beta
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_params = (var * 0.05, 0.1, 0.85);

        for a_idx in 0..20 {
            let alpha = 0.01 + 0.02 * a_idx as f64;
            for b_idx in 0..20 {
                let beta = 0.5 + 0.025 * b_idx as f64;
                if alpha + beta >= 0.999 { continue; }
                let omega = var * (1.0 - alpha - beta);
                if omega <= 0.0 { continue; }

                let ll = Self::log_likelihood_eval(&residuals, omega, alpha, beta);
                if ll > best_ll {
                    best_ll = ll;
                    best_params = (omega, alpha, beta);
                }
            }
        }

        // Newton refinement (simplified)
        let (mut omega, mut alpha, mut beta) = best_params;
        let step = 1e-4;
        for _ in 0..100 {
            let ll0 = Self::log_likelihood_eval(&residuals, omega, alpha, beta);
            let dll_omega = (Self::log_likelihood_eval(&residuals, omega + step, alpha, beta) - ll0) / step;
            let dll_alpha = (Self::log_likelihood_eval(&residuals, omega, alpha + step, beta) - ll0) / step;
            let dll_beta = (Self::log_likelihood_eval(&residuals, omega, alpha, beta + step) - ll0) / step;

            let lr = 1e-6;
            omega = (omega + lr * dll_omega).max(1e-10);
            alpha = (alpha + lr * dll_alpha).clamp(1e-6, 0.5);
            beta = (beta + lr * dll_beta).clamp(0.01, 0.999 - alpha);
        }

        let ll = Self::log_likelihood_eval(&residuals, omega, alpha, beta);
        Self { omega, alpha, beta, log_likelihood: ll }
    }

    fn log_likelihood_eval(residuals: &[f64], omega: f64, alpha: f64, beta: f64) -> f64 {
        let n = residuals.len();
        let mut sigma2 = residuals.iter().map(|r| r * r).sum::<f64>() / n as f64;
        let mut ll = 0.0;
        let log_2pi = (2.0 * std::f64::consts::PI).ln();

        for i in 0..n {
            sigma2 = omega + alpha * residuals[i.max(1) - if i > 0 { 0 } else { 0 }].powi(2) + beta * sigma2;
            let r2 = residuals[i];
            sigma2 = omega + alpha * r2 * r2 + beta * sigma2;
            sigma2 = sigma2.max(1e-20);
            ll += -0.5 * (log_2pi + sigma2.ln() + r2 * r2 / sigma2);
        }
        ll
    }

    /// Forecast conditional variance h steps ahead
    pub fn forecast_variance(&self, returns: &[f64], steps: usize) -> Vec<f64> {
        let n = returns.len();
        let mean = returns.iter().sum::<f64>() / n as f64;

        // Compute last sigma^2
        let mut sigma2 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
        for i in 1..n {
            let r = returns[i] - mean;
            sigma2 = self.omega + self.alpha * r * r + self.beta * sigma2;
        }

        let last_r2 = (returns[n - 1] - mean).powi(2);
        let mut forecasts = Vec::with_capacity(steps);
        let unconditional = self.omega / (1.0 - self.alpha - self.beta).max(1e-10);
        let persist = self.alpha + self.beta;

        let mut h = self.omega + self.alpha * last_r2 + self.beta * sigma2;
        for _ in 0..steps {
            forecasts.push(h);
            h = self.omega + persist * h;
        }
        forecasts
    }

    pub fn unconditional_variance(&self) -> f64 {
        self.omega / (1.0 - self.alpha - self.beta).max(1e-10)
    }

    pub fn half_life(&self) -> f64 {
        let persist = self.alpha + self.beta;
        if persist >= 1.0 || persist <= 0.0 { return f64::INFINITY; }
        0.693147 / (1.0 - persist).ln().abs()
    }
}

/// Simple exponential smoothing
pub fn ses(data: &[f64], alpha: f64) -> Vec<f64> {
    ema(data, alpha)
}

/// Double exponential smoothing (Holt's method)
pub struct HoltSmoothing {
    pub alpha: f64,
    pub beta: f64,
}

impl HoltSmoothing {
    pub fn new(alpha: f64, beta: f64) -> Self { Self { alpha, beta } }

    pub fn fit_predict(&self, data: &[f64], forecast_steps: usize) -> Vec<f64> {
        let n = data.len();
        let mut level = data[0];
        let mut trend = if n > 1 { data[1] - data[0] } else { 0.0 };
        let mut fitted = Vec::with_capacity(n + forecast_steps);
        fitted.push(level);

        for i in 1..n {
            let prev_level = level;
            level = self.alpha * data[i] + (1.0 - self.alpha) * (prev_level + trend);
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
            fitted.push(level + trend);
        }

        for h in 1..=forecast_steps {
            fitted.push(level + h as f64 * trend);
        }
        fitted
    }
}

/// Triple exponential smoothing (Holt-Winters)
pub struct HoltWinters {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub period: usize,
    pub multiplicative: bool,
}

impl HoltWinters {
    pub fn new(alpha: f64, beta: f64, gamma: f64, period: usize, multiplicative: bool) -> Self {
        Self { alpha, beta, gamma, period, multiplicative }
    }

    pub fn fit_predict(&self, data: &[f64], forecast_steps: usize) -> Vec<f64> {
        let n = data.len();
        let p = self.period;
        assert!(n >= 2 * p, "Need at least 2 full periods of data");

        // Initialize level, trend, seasonal
        let mut level = data[..p].iter().sum::<f64>() / p as f64;
        let mut trend = (data[p..2 * p].iter().sum::<f64>() - data[..p].iter().sum::<f64>()) / (p * p) as f64;

        let mut seasonal = vec![0.0; p];
        if self.multiplicative {
            for i in 0..p {
                seasonal[i] = data[i] / level;
            }
        } else {
            for i in 0..p {
                seasonal[i] = data[i] - level;
            }
        }

        let mut fitted = Vec::with_capacity(n + forecast_steps);
        for i in 0..p { fitted.push(data[i]); } // initial period

        for i in p..n {
            let s_idx = i % p;
            let prev_level = level;

            if self.multiplicative {
                level = self.alpha * (data[i] / seasonal[s_idx]) + (1.0 - self.alpha) * (prev_level + trend);
                trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
                seasonal[s_idx] = self.gamma * (data[i] / level) + (1.0 - self.gamma) * seasonal[s_idx];
                fitted.push((level + trend) * seasonal[(i + 1) % p]);
            } else {
                level = self.alpha * (data[i] - seasonal[s_idx]) + (1.0 - self.alpha) * (prev_level + trend);
                trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
                seasonal[s_idx] = self.gamma * (data[i] - level) + (1.0 - self.gamma) * seasonal[s_idx];
                fitted.push(level + trend + seasonal[(i + 1) % p]);
            }
        }

        for h in 1..=forecast_steps {
            let s_idx = (n + h - 1) % p;
            if self.multiplicative {
                fitted.push((level + h as f64 * trend) * seasonal[s_idx]);
            } else {
                fitted.push(level + h as f64 * trend + seasonal[s_idx]);
            }
        }
        fitted
    }
}

/// 1D Kalman Filter
pub struct KalmanFilter1D {
    pub state: f64,
    pub variance: f64,
    pub process_noise: f64,
    pub measurement_noise: f64,
}

impl KalmanFilter1D {
    pub fn new(initial_state: f64, initial_variance: f64, q: f64, r: f64) -> Self {
        Self { state: initial_state, variance: initial_variance, process_noise: q, measurement_noise: r }
    }

    pub fn predict(&mut self) {
        self.variance += self.process_noise;
    }

    pub fn update(&mut self, measurement: f64) {
        let k = self.variance / (self.variance + self.measurement_noise);
        self.state += k * (measurement - self.state);
        self.variance *= 1.0 - k;
    }

    pub fn filter(&mut self, measurements: &[f64]) -> Vec<f64> {
        let mut filtered = Vec::with_capacity(measurements.len());
        for &z in measurements {
            self.predict();
            self.update(z);
            filtered.push(self.state);
        }
        filtered
    }

    pub fn smooth(&mut self, measurements: &[f64]) -> Vec<f64> {
        let n = measurements.len();
        let mut states = Vec::with_capacity(n);
        let mut variances = Vec::with_capacity(n);
        let mut predicted_vars = Vec::with_capacity(n);

        // Forward pass
        for &z in measurements {
            self.predict();
            predicted_vars.push(self.variance);
            self.update(z);
            states.push(self.state);
            variances.push(self.variance);
        }

        // Backward pass (RTS smoother)
        let mut smoothed = states.clone();
        for i in (0..n - 1).rev() {
            let gain = variances[i] / predicted_vars[i + 1].max(1e-30);
            smoothed[i] += gain * (smoothed[i + 1] - states[i]);
        }
        smoothed
    }
}

/// 2D Kalman Filter (constant velocity model)
pub struct KalmanFilter2D {
    pub state: [f64; 4], // [x, vx, y, vy]
    pub p: [[f64; 4]; 4], // covariance
    pub q: f64, // process noise scalar
    pub r: f64, // measurement noise scalar
}

impl KalmanFilter2D {
    pub fn new(x: f64, y: f64, q: f64, r: f64) -> Self {
        let mut p = [[0.0; 4]; 4];
        for i in 0..4 { p[i][i] = 1.0; }
        Self { state: [x, 0.0, y, 0.0], p, q, r }
    }

    pub fn predict(&mut self, dt: f64) {
        // State transition: x = x + vx*dt, vx = vx, etc
        self.state[0] += self.state[1] * dt;
        self.state[2] += self.state[3] * dt;

        // Update covariance: P = F P F^T + Q
        let f = [[1.0, dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, dt], [0.0, 0.0, 0.0, 1.0]];
        let mut fp = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 { fp[i][j] += f[i][k] * self.p[k][j]; }
            }
        }
        let mut new_p = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 { new_p[i][j] += fp[i][k] * f[j][k]; }
            }
        }
        for i in 0..4 { new_p[i][i] += self.q; }
        self.p = new_p;
    }

    pub fn update(&mut self, mx: f64, my: f64) {
        // Measurement: z = H x + noise, H = [[1,0,0,0],[0,0,1,0]]
        let innovation = [mx - self.state[0], my - self.state[2]];

        // S = H P H^T + R
        let s = [[self.p[0][0] + self.r, self.p[0][2]],
                  [self.p[2][0], self.p[2][2] + self.r]];
        let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
        if det.abs() < 1e-30 { return; }
        let s_inv = [[s[1][1] / det, -s[0][1] / det],
                      [-s[1][0] / det, s[0][0] / det]];

        // K = P H^T S^{-1}
        // P H^T: columns 0 and 2 of P
        let mut k = [[0.0; 2]; 4];
        for i in 0..4 {
            let ph = [self.p[i][0], self.p[i][2]];
            for j in 0..2 {
                k[i][j] = ph[0] * s_inv[0][j] + ph[1] * s_inv[1][j];
            }
        }

        // Update state
        for i in 0..4 {
            self.state[i] += k[i][0] * innovation[0] + k[i][1] * innovation[1];
        }

        // Update covariance: P = (I - KH) P
        let mut kh = [[0.0; 4]; 4];
        for i in 0..4 {
            kh[i][0] = k[i][0];
            kh[i][2] = k[i][1];
        }
        let mut new_p = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                let mut ikh = if i == j { 1.0 } else { 0.0 };
                ikh -= kh[i][j];
                for l in 0..4 {
                    let ikhl = if i == l { 1.0 } else { 0.0 } - kh[i][l];
                    new_p[i][j] += ikhl * self.p[l][j];
                }
            }
        }
        self.p = new_p;
    }

    pub fn filter(&mut self, measurements: &[(f64, f64)], dt: f64) -> Vec<(f64, f64)> {
        let mut result = Vec::with_capacity(measurements.len());
        for &(mx, my) in measurements {
            self.predict(dt);
            self.update(mx, my);
            result.push((self.state[0], self.state[2]));
        }
        result
    }
}

/// Detect changepoints using CUSUM
pub fn cusum(data: &[f64], threshold: f64) -> Vec<usize> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
    if std < 1e-15 { return vec![]; }

    let mut s_pos = 0.0;
    let mut s_neg = 0.0;
    let mut changepoints = Vec::new();

    for (i, &x) in data.iter().enumerate() {
        let z = (x - mean) / std;
        s_pos = (s_pos + z - 0.5).max(0.0);
        s_neg = (s_neg - z - 0.5).max(0.0);
        if s_pos > threshold || s_neg > threshold {
            changepoints.push(i);
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }
    changepoints
}

/// Hurst exponent estimation via R/S analysis
pub fn hurst_exponent(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 20 { return 0.5; }

    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();

    let min_size = 10;
    let mut size = min_size;
    while size <= n / 2 {
        let n_blocks = n / size;
        let mut rs_sum = 0.0;
        for b in 0..n_blocks {
            let block = &data[b * size..(b + 1) * size];
            let mean = block.iter().sum::<f64>() / size as f64;
            let std = (block.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / size as f64).sqrt();
            if std < 1e-15 { continue; }

            // Cumulative deviations
            let mut cum = vec![0.0; size];
            cum[0] = block[0] - mean;
            for i in 1..size { cum[i] = cum[i - 1] + block[i] - mean; }
            let range = cum.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - cum.iter().copied().fold(f64::INFINITY, f64::min);
            rs_sum += range / std;
        }
        if n_blocks > 0 {
            log_n.push((size as f64).ln());
            log_rs.push((rs_sum / n_blocks as f64).ln());
        }
        size = (size as f64 * 1.5) as usize;
    }

    if log_n.len() < 2 { return 0.5; }

    // Linear regression: log(R/S) = H * log(n) + c
    let n_pts = log_n.len() as f64;
    let sum_x: f64 = log_n.iter().sum();
    let sum_y: f64 = log_rs.iter().sum();
    let sum_xy: f64 = log_n.iter().zip(&log_rs).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_n.iter().map(|x| x * x).sum();
    let denom = n_pts * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-15 { return 0.5; }
    (n_pts * sum_xy - sum_x * sum_y) / denom
}

/// Detrend data (remove linear trend)
pub fn detrend(data: &[f64]) -> Vec<f64> {
    let n = data.len() as f64;
    let sum_x: f64 = (0..data.len()).map(|i| i as f64).sum();
    let sum_y: f64 = data.iter().sum();
    let sum_xy: f64 = data.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..data.len()).map(|i| (i as f64).powi(2)).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    data.iter().enumerate().map(|(i, &y)| y - (intercept + slope * i as f64)).collect()
}

/// Difference a time series (d-th order)
pub fn difference(data: &[f64], d: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..d {
        let new: Vec<f64> = result.windows(2).map(|w| w[1] - w[0]).collect();
        result = new;
    }
    result
}

/// Log returns
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}

/// Simple returns
pub fn simple_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect()
}

/// Realized volatility from intraday returns
pub fn realized_volatility(returns: &[f64]) -> f64 {
    returns.iter().map(|r| r * r).sum::<f64>().sqrt()
}

/// Realized variance with subsampling
pub fn realized_variance_subsample(returns: &[f64], q: usize) -> f64 {
    let n = returns.len();
    let mut total = 0.0;
    for offset in 0..q {
        let mut rv = 0.0;
        let mut i = offset;
        while i + q <= n {
            let r: f64 = returns[i..i + q].iter().sum();
            rv += r * r;
            i += q;
        }
        total += rv;
    }
    total / q as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&data, 3);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&data, 0.5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_acf() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let ac = acf(&data, 10);
        assert!((ac[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kalman_1d() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
        let measurements: Vec<f64> = (0..50).map(|i| i as f64 + 0.5).collect(); // noisy linear
        let filtered = kf.filter(&measurements);
        assert!(filtered.len() == 50);
    }
}
