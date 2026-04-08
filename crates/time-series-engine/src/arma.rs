use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Autocovariance and ACF/PACF helpers
// ---------------------------------------------------------------------------
pub fn autocovariance(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let mut acov = vec![0.0; max_lag + 1];
    for lag in 0..=max_lag.min(n - 1) {
        let mut s = 0.0;
        for i in 0..n - lag {
            s += (data[i] - mean) * (data[i + lag] - mean);
        }
        acov[lag] = s / n as f64;
    }
    acov
}

pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let acov = autocovariance(data, max_lag);
    if acov[0].abs() < 1e-30 {
        return vec![0.0; max_lag + 1];
    }
    acov.iter().map(|&g| g / acov[0]).collect()
}

pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let acf_vals = acf(data, max_lag);
    let mut pacf_vals = vec![0.0; max_lag + 1];
    pacf_vals[0] = 1.0;

    if max_lag == 0 { return pacf_vals; }

    // Levinson-Durbin recursion
    let mut phi = vec![vec![0.0; max_lag + 1]; max_lag + 1];
    phi[1][1] = acf_vals[1];
    pacf_vals[1] = acf_vals[1];

    for k in 2..=max_lag {
        let mut num = acf_vals[k];
        for j in 1..k {
            num -= phi[k - 1][j] * acf_vals[k - j];
        }
        let mut den = 1.0;
        for j in 1..k {
            den -= phi[k - 1][j] * acf_vals[j];
        }
        if den.abs() < 1e-15 { break; }
        phi[k][k] = num / den;
        pacf_vals[k] = phi[k][k];

        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
    }

    pacf_vals
}

// ---------------------------------------------------------------------------
// AR(p) Estimation: Yule-Walker
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ArModel {
    pub order: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub variance: f64,
    pub aic: f64,
    pub bic: f64,
}

pub fn ar_yule_walker(data: &[f64], order: usize) -> ArModel {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();
    let acov = autocovariance(&centered, order);

    // Build Toeplitz system: R * phi = r
    let p = order;
    let mut r_matrix = vec![0.0; p * p];
    let mut r_vec = vec![0.0; p];

    for i in 0..p {
        for j in 0..p {
            let lag = if i >= j { i - j } else { j - i };
            r_matrix[i * p + j] = acov[lag];
        }
        r_vec[i] = acov[i + 1];
    }

    let coefficients = solve_system(&r_matrix, &r_vec, p);

    // Residual variance
    let mut var = acov[0];
    for i in 0..p {
        var -= coefficients[i] * acov[i + 1];
    }
    var = var.max(1e-15);

    let intercept = mean * (1.0 - coefficients.iter().sum::<f64>());

    // Information criteria
    let log_likelihood = -(n as f64) / 2.0 * (2.0 * PI * var).ln() - (n as f64) / 2.0;
    let k = (p + 1) as f64; // p coeffs + variance
    let aic = -2.0 * log_likelihood + 2.0 * k;
    let bic = -2.0 * log_likelihood + k * (n as f64).ln();

    ArModel { order, coefficients, intercept, variance: var, aic, bic }
}

// ---------------------------------------------------------------------------
// AR(p) Estimation: Burg's method
// ---------------------------------------------------------------------------
pub fn ar_burg(data: &[f64], order: usize) -> ArModel {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    let mut a = vec![0.0; order + 1];
    a[0] = 1.0;

    let mut ef: Vec<f64> = centered.clone(); // forward error
    let mut eb: Vec<f64> = centered.clone(); // backward error

    let mut variance = centered.iter().map(|&x| x * x).sum::<f64>() / n as f64;

    let mut coefficients = vec![0.0; order];

    for k in 1..=order {
        // Compute reflection coefficient
        let mut num = 0.0;
        let mut den = 0.0;
        for i in k..n {
            num += ef[i] * eb[i - 1];
            den += ef[i] * ef[i] + eb[i - 1] * eb[i - 1];
        }
        let rc = if den.abs() > 1e-15 { -2.0 * num / den } else { 0.0 };

        // Update AR coefficients
        let mut a_new = vec![0.0; order + 1];
        a_new[0] = 1.0;
        for i in 1..k {
            a_new[i] = a[i] + rc * a[k - i];
        }
        a_new[k] = rc;
        a = a_new;

        // Update forward/backward errors
        let mut ef_new = vec![0.0; n];
        let mut eb_new = vec![0.0; n];
        for i in k..n {
            ef_new[i] = ef[i] + rc * eb[i - 1];
            eb_new[i] = eb[i - 1] + rc * ef[i];
        }
        ef = ef_new;
        eb = eb_new;

        variance *= 1.0 - rc * rc;
    }

    for i in 0..order {
        coefficients[i] = -a[i + 1]; // convention: y_t = sum(phi_i * y_{t-i})
    }

    let intercept = mean * (1.0 - coefficients.iter().sum::<f64>());
    let log_likelihood = -(n as f64) / 2.0 * (2.0 * PI * variance).ln() - (n as f64) / 2.0;
    let kp = (order + 1) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * kp;
    let bic = -2.0 * log_likelihood + kp * (n as f64).ln();

    ArModel { order, coefficients, intercept, variance, aic, bic }
}

// ---------------------------------------------------------------------------
// AR(p) Estimation: OLS
// ---------------------------------------------------------------------------
pub fn ar_ols(data: &[f64], order: usize) -> ArModel {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;

    let p = order;
    let m = n - p;

    // Build X matrix and y vector
    let mut xtx = vec![0.0; (p + 1) * (p + 1)];
    let mut xty = vec![0.0; p + 1];

    for t in p..n {
        let mut row = vec![1.0]; // intercept
        for j in 1..=p {
            row.push(data[t - j]);
        }
        let yi = data[t];

        for r in 0..=p {
            for c in 0..=p {
                xtx[r * (p + 1) + c] += row[r] * row[c];
            }
            xty[r] += row[r] * yi;
        }
    }

    let beta = solve_system(&xtx, &xty, p + 1);
    let intercept = beta[0];
    let coefficients: Vec<f64> = beta[1..].to_vec();

    // Residual variance
    let mut ss_res = 0.0;
    for t in p..n {
        let mut pred = intercept;
        for j in 0..p {
            pred += coefficients[j] * data[t - 1 - j];
        }
        ss_res += (data[t] - pred).powi(2);
    }
    let variance = ss_res / m as f64;

    let log_likelihood = -(m as f64) / 2.0 * (2.0 * PI * variance).ln() - (m as f64) / 2.0;
    let kp = (p + 2) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * kp;
    let bic = -2.0 * log_likelihood + kp * (m as f64).ln();

    ArModel { order, coefficients, intercept, variance, aic, bic }
}

// ---------------------------------------------------------------------------
// MA(q) Estimation via Innovations Algorithm
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct MaModel {
    pub order: usize,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub variance: f64,
    pub aic: f64,
    pub bic: f64,
}

pub fn ma_innovations(data: &[f64], order: usize) -> MaModel {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    let max_lag = order + 1;
    let acov = autocovariance(&centered, max_lag.min(n - 1));

    // Innovations algorithm
    let m = order;
    let num_steps = n.min(m + 20);
    let mut v = vec![0.0; num_steps + 1];
    let mut theta = vec![vec![0.0; num_steps + 1]; num_steps + 1];

    v[0] = acov[0];

    for i in 1..=num_steps.min(n - 1) {
        for k in 0..i {
            let mut s = acov[i - k];
            for j in 0..k {
                s -= theta[k][k - 1 - j] * theta[i][i - 1 - j] * v[j];
            }
            if v[k].abs() > 1e-15 {
                theta[i][i - 1 - k] = s / v[k];
            }
        }
        v[i] = acov[0];
        for j in 0..i {
            v[i] -= theta[i][i - 1 - j].powi(2) * v[j];
        }
        v[i] = v[i].max(1e-15);
    }

    // Extract MA coefficients from last row
    let idx = num_steps.min(n - 1);
    let mut coefficients = vec![0.0; m];
    for j in 0..m {
        if idx > j {
            coefficients[j] = theta[idx][idx - 1 - j];
        }
    }

    let variance = v[idx.min(v.len() - 1)].max(1e-15);

    let log_likelihood = -(n as f64) / 2.0 * (2.0 * PI * variance).ln() - (n as f64) / 2.0;
    let kp = (m + 1) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * kp;
    let bic = -2.0 * log_likelihood + kp * (n as f64).ln();

    MaModel { order: m, coefficients, intercept: mean, variance, aic, bic }
}

// ---------------------------------------------------------------------------
// ARMA(p,q) Estimation via Conditional MLE
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ArmaModel {
    pub ar_order: usize,
    pub ma_order: usize,
    pub ar_coefficients: Vec<f64>,
    pub ma_coefficients: Vec<f64>,
    pub intercept: f64,
    pub variance: f64,
    pub aic: f64,
    pub bic: f64,
}

pub fn arma_conditional_mle(data: &[f64], p: usize, q: usize) -> ArmaModel {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Initialize AR coefficients from Yule-Walker
    let mut ar_coeff = if p > 0 {
        let ar = ar_yule_walker(&centered, p);
        ar.coefficients
    } else {
        vec![]
    };

    let mut ma_coeff = vec![0.0; q];

    // Iterative conditional MLE (simplified Gauss-Newton)
    let start = p.max(q);
    let m = n - start;

    for _iteration in 0..50 {
        // Compute residuals
        let mut residuals = vec![0.0; n];
        for t in start..n {
            let mut pred = 0.0;
            for j in 0..p {
                pred += ar_coeff[j] * centered[t - 1 - j];
            }
            for j in 0..q {
                pred += ma_coeff[j] * residuals[t - 1 - j];
            }
            residuals[t] = centered[t] - pred;
        }

        // Update AR coefficients using pseudo-OLS on residuals
        if p > 0 {
            let mut xtx = vec![0.0; p * p];
            let mut xty = vec![0.0; p];
            for t in start..n {
                let mut row = Vec::with_capacity(p);
                for j in 0..p {
                    row.push(centered[t - 1 - j]);
                }
                let yi = centered[t];
                let mut ma_part = 0.0;
                for j in 0..q {
                    ma_part += ma_coeff[j] * residuals[t - 1 - j];
                }
                let target = yi - ma_part;
                for r in 0..p {
                    for c in 0..p {
                        xtx[r * p + c] += row[r] * row[c];
                    }
                    xty[r] += row[r] * target;
                }
            }
            ar_coeff = solve_system(&xtx, &xty, p);
        }

        // Update MA coefficients
        if q > 0 {
            // Recompute residuals with updated AR
            for t in start..n {
                let mut pred = 0.0;
                for j in 0..p {
                    pred += ar_coeff[j] * centered[t - 1 - j];
                }
                for j in 0..q {
                    pred += ma_coeff[j] * residuals[t - 1 - j];
                }
                residuals[t] = centered[t] - pred;
            }

            let mut xtx = vec![0.0; q * q];
            let mut xty = vec![0.0; q];
            for t in start..n {
                let mut row = Vec::with_capacity(q);
                for j in 0..q {
                    row.push(residuals[t - 1 - j]);
                }
                let mut ar_part = 0.0;
                for j in 0..p {
                    ar_part += ar_coeff[j] * centered[t - 1 - j];
                }
                let target = centered[t] - ar_part;
                for r in 0..q {
                    for c in 0..q {
                        xtx[r * q + c] += row[r] * row[c];
                    }
                    xty[r] += row[r] * target;
                }
            }
            ma_coeff = solve_system(&xtx, &xty, q);
        }
    }

    // Final residual variance
    let mut residuals = vec![0.0; n];
    let mut ss = 0.0;
    for t in start..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += ar_coeff[j] * centered[t - 1 - j];
        }
        for j in 0..q {
            pred += ma_coeff[j] * residuals[t - 1 - j];
        }
        residuals[t] = centered[t] - pred;
        ss += residuals[t] * residuals[t];
    }
    let variance = (ss / m as f64).max(1e-15);

    let intercept = mean * (1.0 - ar_coeff.iter().sum::<f64>());
    let log_likelihood = -(m as f64) / 2.0 * (2.0 * PI * variance).ln() - (m as f64) / 2.0;
    let k = (p + q + 1) as f64;
    let aic = -2.0 * log_likelihood + 2.0 * k;
    let bic = -2.0 * log_likelihood + k * (m as f64).ln();

    ArmaModel {
        ar_order: p,
        ma_order: q,
        ar_coefficients: ar_coeff,
        ma_coefficients: ma_coeff,
        intercept,
        variance,
        aic,
        bic,
    }
}

// ---------------------------------------------------------------------------
// ARIMA(p,d,q)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ArimaModel {
    pub arma: ArmaModel,
    pub d: usize,
    pub original_tail: Vec<f64>, // last d values for undifferencing
}

pub fn difference(data: &[f64], d: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..d {
        let n = result.len();
        if n <= 1 { break; }
        let mut diff = Vec::with_capacity(n - 1);
        for i in 1..n {
            diff.push(result[i] - result[i - 1]);
        }
        result = diff;
    }
    result
}

pub fn undifference(forecasts: &[f64], tail: &[f64], d: usize) -> Vec<f64> {
    let mut result = forecasts.to_vec();
    for di in (0..d).rev() {
        let last = tail[di];
        let mut new_result = Vec::with_capacity(result.len());
        let mut prev = last;
        for &f in &result {
            prev += f;
            new_result.push(prev);
        }
        result = new_result;
    }
    result
}

pub fn arima_fit(data: &[f64], p: usize, d: usize, q: usize) -> ArimaModel {
    let mut tail = Vec::new();
    let mut current = data.to_vec();
    for _ in 0..d {
        if current.len() > 0 {
            tail.push(*current.last().unwrap());
        }
        current = difference(&current, 1);
    }

    let arma = arma_conditional_mle(&current, p, q);

    ArimaModel { arma, d, original_tail: tail }
}

// ---------------------------------------------------------------------------
// Seasonal ARIMA
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct SarimaModel {
    pub arima: ArimaModel,
    pub seasonal_period: usize,
    pub seasonal_ar: Vec<f64>,
    pub seasonal_ma: Vec<f64>,
    pub seasonal_d: usize,
}

pub fn seasonal_difference(data: &[f64], period: usize, d: usize) -> Vec<f64> {
    let mut result = data.to_vec();
    for _ in 0..d {
        let n = result.len();
        if n <= period { break; }
        let mut diff = Vec::with_capacity(n - period);
        for i in period..n {
            diff.push(result[i] - result[i - period]);
        }
        result = diff;
    }
    result
}

pub fn sarima_fit(
    data: &[f64],
    p: usize, d: usize, q: usize,
    sp: usize, sd: usize, sq: usize,
    period: usize,
) -> SarimaModel {
    // Apply seasonal differencing first
    let after_seasonal = seasonal_difference(data, period, sd);
    // Then regular differencing and fit ARIMA
    let arima = arima_fit(&after_seasonal, p, d, q);

    // Fit seasonal AR/MA on residuals (simplified: use Yule-Walker)
    let residuals = arima_residuals(&after_seasonal, &arima);
    let seasonal_ar = if sp > 0 {
        let mut ar_coeffs = vec![0.0; sp];
        let acov = autocovariance(&residuals, sp * period);
        // Seasonal autocorrelation at multiples of period
        for i in 0..sp {
            let lag = (i + 1) * period;
            if lag < acov.len() && acov[0].abs() > 1e-15 {
                ar_coeffs[i] = acov[lag] / acov[0];
            }
        }
        ar_coeffs
    } else {
        vec![]
    };

    let seasonal_ma = vec![0.0; sq]; // Simplified

    SarimaModel {
        arima,
        seasonal_period: period,
        seasonal_ar,
        seasonal_ma,
        seasonal_d: sd,
    }
}

fn arima_residuals(data: &[f64], model: &ArimaModel) -> Vec<f64> {
    let diffed = difference(data, model.d);
    let p = model.arma.ar_order;
    let q = model.arma.ma_order;
    let start = p.max(q);
    let n = diffed.len();
    let centered: Vec<f64> = diffed.iter().map(|&x| x - model.arma.intercept).collect();

    let mut residuals = vec![0.0; n];
    for t in start..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += model.arma.ar_coefficients[j] * centered[t - 1 - j];
        }
        for j in 0..q {
            pred += model.arma.ma_coefficients[j] * residuals[t - 1 - j];
        }
        residuals[t] = centered[t] - pred;
    }
    residuals
}

// ---------------------------------------------------------------------------
// Forecasting
// ---------------------------------------------------------------------------
pub fn arma_forecast(model: &ArmaModel, data: &[f64], steps: usize) -> Vec<f64> {
    let n = data.len();
    let mean = model.intercept / (1.0 - model.ar_coefficients.iter().sum::<f64>());
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();
    let p = model.ar_order;
    let q = model.ma_order;

    // Compute residuals for MA component
    let start = p.max(q);
    let mut residuals = vec![0.0; n];
    for t in start..n {
        let mut pred = 0.0;
        for j in 0..p {
            pred += model.ar_coefficients[j] * centered[t - 1 - j];
        }
        for j in 0..q {
            if t > j {
                pred += model.ma_coefficients[j] * residuals[t - 1 - j];
            }
        }
        residuals[t] = centered[t] - pred;
    }

    let mut forecasts = Vec::with_capacity(steps);
    let mut extended = centered.clone();
    let mut extended_res = residuals.clone();

    for h in 0..steps {
        let t = n + h;
        let mut pred = 0.0;
        for j in 0..p {
            if t > j && t - 1 - j < extended.len() {
                pred += model.ar_coefficients[j] * extended[t - 1 - j];
            }
        }
        for j in 0..q {
            if t > j && t - 1 - j < extended_res.len() {
                pred += model.ma_coefficients[j] * extended_res[t - 1 - j];
            }
        }
        forecasts.push(pred + mean);
        extended.push(pred);
        extended_res.push(0.0); // future residuals = 0
    }

    forecasts
}

pub fn arima_forecast(model: &ArimaModel, data: &[f64], steps: usize) -> Vec<f64> {
    let diffed = difference(data, model.d);
    let fc_diffed = arma_forecast(&model.arma, &diffed, steps);

    // Undifference
    let mut tail = Vec::new();
    let mut current = data.to_vec();
    for _ in 0..model.d {
        tail.push(*current.last().unwrap_or(&0.0));
        current = difference(&current, 1);
    }

    undifference(&fc_diffed, &tail, model.d)
}

// ---------------------------------------------------------------------------
// Confidence Intervals
// ---------------------------------------------------------------------------
pub fn forecast_confidence_intervals(
    model: &ArmaModel,
    steps: usize,
    confidence: f64,
) -> Vec<(f64, f64)> {
    // MA representation coefficients (psi weights)
    let mut psi = vec![0.0; steps];
    if steps > 0 { psi[0] = 1.0; }

    let p = model.ar_order;
    let q = model.ma_order;

    for j in 1..steps {
        let mut val = 0.0;
        for i in 0..p {
            if j > i {
                val += model.ar_coefficients[i] * psi[j - 1 - i];
            }
        }
        if j <= q {
            val += model.ma_coefficients[j - 1];
        }
        psi[j] = val;
    }

    // z-score for confidence level
    let z = normal_quantile((1.0 + confidence) / 2.0);

    let mut intervals = Vec::with_capacity(steps);
    let mut cumsum = 0.0;
    for h in 0..steps {
        cumsum += psi[h] * psi[h];
        let se = (model.variance * cumsum).sqrt();
        intervals.push((-z * se, z * se));
    }
    intervals
}

// ---------------------------------------------------------------------------
// Model Selection (AIC, BIC, HQIC)
// ---------------------------------------------------------------------------
pub fn hqic(log_likelihood: f64, k: usize, n: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * k as f64 * (n as f64).ln().ln()
}

pub fn select_ar_order(data: &[f64], max_order: usize) -> usize {
    let mut best_aic = f64::INFINITY;
    let mut best_order = 0;
    for p in 0..=max_order {
        let model = ar_burg(data, p);
        if model.aic < best_aic {
            best_aic = model.aic;
            best_order = p;
        }
    }
    best_order
}

pub fn select_arma_order(data: &[f64], max_p: usize, max_q: usize) -> (usize, usize) {
    let mut best_aic = f64::INFINITY;
    let mut best_p = 0;
    let mut best_q = 0;
    for p in 0..=max_p {
        for q in 0..=max_q {
            if p == 0 && q == 0 { continue; }
            let model = arma_conditional_mle(data, p, q);
            if model.aic < best_aic {
                best_aic = model.aic;
                best_p = p;
                best_q = q;
            }
        }
    }
    (best_p, best_q)
}

// ---------------------------------------------------------------------------
// Residual Diagnostics
// ---------------------------------------------------------------------------
pub fn ljung_box_test(residuals: &[f64], max_lag: usize, num_params: usize) -> (f64, f64) {
    let n = residuals.len();
    let acf_vals = acf(residuals, max_lag);
    let mut q_stat = 0.0;
    for k in 1..=max_lag {
        q_stat += acf_vals[k] * acf_vals[k] / (n - k) as f64;
    }
    q_stat *= n as f64 * (n as f64 + 2.0);

    // Degrees of freedom
    let df = max_lag.saturating_sub(num_params);
    let p_value = if df > 0 {
        1.0 - chi_squared_cdf(q_stat, df)
    } else {
        1.0
    };

    (q_stat, p_value)
}

pub fn durbin_watson(residuals: &[f64]) -> f64 {
    let n = residuals.len();
    if n < 2 { return 2.0; }
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        den += residuals[i] * residuals[i];
        if i > 0 {
            num += (residuals[i] - residuals[i - 1]).powi(2);
        }
    }
    if den.abs() > 1e-15 { num / den } else { 2.0 }
}

pub fn jarque_bera_test(residuals: &[f64]) -> (f64, f64) {
    let n = residuals.len() as f64;
    let mean = residuals.iter().sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;
    for &r in residuals {
        let d = r - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;

    let skewness = if m2 > 1e-15 { m3 / m2.powf(1.5) } else { 0.0 };
    let kurtosis = if m2 > 1e-15 { m4 / (m2 * m2) } else { 3.0 };

    let jb = n / 6.0 * (skewness * skewness + (kurtosis - 3.0).powi(2) / 4.0);
    let p_value = 1.0 - chi_squared_cdf(jb, 2);
    (jb, p_value)
}

// ---------------------------------------------------------------------------
// Helper: solve linear system via Gaussian elimination
// ---------------------------------------------------------------------------
fn solve_system(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    if n == 0 { return vec![]; }
    let mut aug = vec![0.0; n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            aug[r * (n + 1) + c] = a[r * n + c];
        }
        aug[r * (n + 1) + n] = b[r];
    }

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let v = aug[row * (n + 1) + col].abs();
            if v > max_val { max_val = v; max_row = row; }
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
        for c in col..=n { aug[col * (n + 1) + c] /= pivot; }
        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row * (n + 1) + col];
            for c in col..=n { aug[row * (n + 1) + c] -= factor * aug[col * (n + 1) + c]; }
        }
    }

    (0..n).map(|r| aug[r * (n + 1) + n]).collect()
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation (Abramowitz & Stegun)
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-15 { return 0.0; }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let q = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -q } else { q }
}

fn chi_squared_cdf(x: f64, k: usize) -> f64 {
    if x <= 0.0 { return 0.0; }
    // Regularized lower incomplete gamma function approximation
    let a = k as f64 / 2.0;
    let z = x / 2.0;
    regularized_gamma_p(a, z)
}

fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }

    // Series expansion for small x
    if x < a + 1.0 {
        let mut sum = 1.0 / a;
        let mut term = 1.0 / a;
        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-15 * sum.abs() { break; }
        }
        let log_gamma_a = ln_gamma(a);
        (a * x.ln() - x - log_gamma_a + sum.ln()).exp().min(1.0)
    } else {
        // Continued fraction for large x
        1.0 - regularized_gamma_q_cf(a, x)
    }
}

fn regularized_gamma_q_cf(a: f64, x: f64) -> f64 {
    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 1.0 / (x + 1.0 - a);
    let mut h = d;

    for n in 1..200 {
        let an = -(n as f64) * (n as f64 - a);
        let bn = x + 2.0 * n as f64 + 1.0 - a;
        d = 1.0 / (bn + an * d);
        c = bn + an / c;
        let delta = c * d;
        h *= delta;
        if (delta - 1.0).abs() < 1e-15 { break; }
    }

    let log_gamma_a = ln_gamma(a);
    (a * x.ln() - x - log_gamma_a + h.ln()).exp().min(1.0)
}

fn ln_gamma(x: f64) -> f64 {
    // Stirling approximation with Lanczos coefficients
    let g = 7.0;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let v = PI / (PI * x).sin();
        v.abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        let t = x + g + 0.5;
        for i in 1..9 {
            a += c[i] / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (t.ln()) * (x + 0.5) - t + a.ln()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_ar1_data(n: usize, phi: f64) -> Vec<f64> {
        let mut data = vec![0.0; n];
        let mut state = 0.0;
        for i in 0..n {
            // Deterministic pseudo-noise
            let noise = ((i as f64 * 1.618033).sin() * 43758.5453).fract() * 0.5;
            state = phi * state + noise;
            data[i] = state;
        }
        data
    }

    #[test]
    fn test_acf() {
        let data = make_ar1_data(200, 0.7);
        let ac = acf(&data, 10);
        assert!((ac[0] - 1.0).abs() < 1e-10);
        assert!(ac[1].abs() > 0.3); // should show autocorrelation
    }

    #[test]
    fn test_pacf() {
        let data = make_ar1_data(200, 0.8);
        let pc = pacf(&data, 10);
        assert!((pc[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ar_yule_walker() {
        let data = make_ar1_data(500, 0.6);
        let model = ar_yule_walker(&data, 1);
        assert!((model.coefficients[0] - 0.6).abs() < 0.15);
    }

    #[test]
    fn test_ar_burg() {
        let data = make_ar1_data(500, 0.7);
        let model = ar_burg(&data, 1);
        assert!((model.coefficients[0] - 0.7).abs() < 0.15);
    }

    #[test]
    fn test_ar_ols() {
        let data = make_ar1_data(500, 0.5);
        let model = ar_ols(&data, 1);
        assert!((model.coefficients[0] - 0.5).abs() < 0.15);
    }

    #[test]
    fn test_ma_innovations() {
        let data = make_ar1_data(200, 0.0);
        let model = ma_innovations(&data, 1);
        assert_eq!(model.order, 1);
    }

    #[test]
    fn test_arma() {
        let data = make_ar1_data(300, 0.5);
        let model = arma_conditional_mle(&data, 1, 1);
        assert_eq!(model.ar_order, 1);
        assert_eq!(model.ma_order, 1);
    }

    #[test]
    fn test_difference() {
        let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let d1 = difference(&data, 1);
        assert_eq!(d1, vec![2.0, 3.0, 4.0, 5.0]);
        let d2 = difference(&data, 2);
        assert_eq!(d2, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_arima() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 + make_ar1_data(1, 0.0)[0]).collect();
        let model = arima_fit(&data, 1, 1, 0);
        assert_eq!(model.d, 1);
    }

    #[test]
    fn test_forecast() {
        let data = make_ar1_data(100, 0.8);
        let model = arma_conditional_mle(&data, 1, 0);
        let fc = arma_forecast(&model, &data, 5);
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_confidence_intervals() {
        let data = make_ar1_data(100, 0.5);
        let model = arma_conditional_mle(&data, 1, 0);
        let ci = forecast_confidence_intervals(&model, 5, 0.95);
        assert_eq!(ci.len(), 5);
        for &(lo, hi) in &ci {
            assert!(lo < 0.0);
            assert!(hi > 0.0);
        }
    }

    #[test]
    fn test_ljung_box() {
        let data = make_ar1_data(200, 0.0);
        let (q, p) = ljung_box_test(&data, 10, 0);
        // White noise should not reject
        assert!(q >= 0.0);
    }

    #[test]
    fn test_durbin_watson() {
        let residuals: Vec<f64> = (0..50).map(|i| ((i as f64 * 2.718).sin())).collect();
        let dw = durbin_watson(&residuals);
        assert!(dw > 0.0 && dw < 4.0);
    }

    #[test]
    fn test_select_ar_order() {
        let data = make_ar1_data(200, 0.7);
        let order = select_ar_order(&data, 5);
        assert!(order <= 5);
    }

    #[test]
    fn test_normal_quantile() {
        let z = normal_quantile(0.975);
        assert!((z - 1.96).abs() < 0.02);
    }

    #[test]
    fn test_seasonal_difference() {
        let data: Vec<f64> = (0..24).map(|i| (i % 12) as f64 + i as f64 * 0.1).collect();
        let sd = seasonal_difference(&data, 12, 1);
        assert_eq!(sd.len(), 12);
    }
}
