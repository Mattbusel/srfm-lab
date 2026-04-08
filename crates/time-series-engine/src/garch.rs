use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// GARCH(1,1) Model
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Garch11 {
    pub omega: f64,   // constant
    pub alpha: f64,   // ARCH coefficient (lagged squared return)
    pub beta: f64,    // GARCH coefficient (lagged variance)
    pub mean: f64,
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub unconditional_var: f64,
    pub persistence: f64,
}

pub fn garch11_fit(returns: &[f64], max_iter: usize) -> Garch11 {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();

    let sample_var = demean.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    // Initialize parameters
    let mut omega = sample_var * 0.05;
    let mut alpha = 0.1;
    let mut beta = 0.85;

    // MLE via gradient ascent (simplified)
    let lr = 1e-6;

    for _iter in 0..max_iter {
        // Compute conditional variances and log-likelihood
        let mut sigma2 = vec![sample_var; n];
        for t in 1..n {
            sigma2[t] = omega + alpha * demean[t - 1] * demean[t - 1] + beta * sigma2[t - 1];
            sigma2[t] = sigma2[t].max(1e-15);
        }

        // Numerical gradients
        let eps = 1e-8;
        let ll_base = garch_log_likelihood(&demean, &sigma2);

        // omega gradient
        let mut sigma2_p = sigma2.clone();
        let omega_p = omega + eps;
        sigma2_p[0] = sample_var;
        for t in 1..n {
            sigma2_p[t] = omega_p + alpha * demean[t - 1] * demean[t - 1] + beta * sigma2_p[t - 1];
            sigma2_p[t] = sigma2_p[t].max(1e-15);
        }
        let grad_omega = (garch_log_likelihood(&demean, &sigma2_p) - ll_base) / eps;

        // alpha gradient
        let alpha_p = alpha + eps;
        for t in 1..n {
            sigma2_p[t] = omega + alpha_p * demean[t - 1] * demean[t - 1] + beta * sigma2_p[t - 1];
            sigma2_p[t] = sigma2_p[t].max(1e-15);
        }
        let grad_alpha = (garch_log_likelihood(&demean, &sigma2_p) - ll_base) / eps;

        // beta gradient
        let beta_p = beta + eps;
        for t in 1..n {
            sigma2_p[t] = omega + alpha * demean[t - 1] * demean[t - 1] + beta_p * sigma2_p[t - 1];
            sigma2_p[t] = sigma2_p[t].max(1e-15);
        }
        let grad_beta = (garch_log_likelihood(&demean, &sigma2_p) - ll_base) / eps;

        // Update with projection onto constraints
        omega = (omega + lr * grad_omega).max(1e-10);
        alpha = (alpha + lr * grad_alpha).max(1e-8).min(0.999);
        beta = (beta + lr * grad_beta).max(1e-8).min(0.999);

        // Ensure stationarity: alpha + beta < 1
        let persist = alpha + beta;
        if persist >= 0.999 {
            let scale = 0.998 / persist;
            alpha *= scale;
            beta *= scale;
        }
    }

    let mut sigma2 = vec![sample_var; n];
    for t in 1..n {
        sigma2[t] = omega + alpha * demean[t - 1] * demean[t - 1] + beta * sigma2[t - 1];
        sigma2[t] = sigma2[t].max(1e-15);
    }
    let ll = garch_log_likelihood(&demean, &sigma2);
    let persistence = alpha + beta;
    let unconditional_var = if persistence < 1.0 { omega / (1.0 - persistence) } else { sample_var };

    let k = 3.0;
    let aic = -2.0 * ll + 2.0 * k;
    let bic = -2.0 * ll + k * (n as f64).ln();

    Garch11 {
        omega, alpha, beta, mean, log_likelihood: ll,
        aic, bic, unconditional_var, persistence,
    }
}

fn garch_log_likelihood(demean: &[f64], sigma2: &[f64]) -> f64 {
    let n = demean.len();
    let mut ll = 0.0;
    for t in 0..n {
        ll += -0.5 * (2.0 * PI).ln() - 0.5 * sigma2[t].ln() - 0.5 * demean[t] * demean[t] / sigma2[t];
    }
    ll
}

pub fn garch11_conditional_variance(model: &Garch11, returns: &[f64]) -> Vec<f64> {
    let n = returns.len();
    let sample_var = returns.iter().map(|&r| (r - model.mean).powi(2)).sum::<f64>() / n as f64;
    let mut sigma2 = vec![sample_var; n];
    for t in 1..n {
        let e = returns[t - 1] - model.mean;
        sigma2[t] = model.omega + model.alpha * e * e + model.beta * sigma2[t - 1];
        sigma2[t] = sigma2[t].max(1e-15);
    }
    sigma2
}

pub fn garch11_forecast(model: &Garch11, last_return: f64, last_var: f64, steps: usize) -> Vec<f64> {
    let mut forecasts = Vec::with_capacity(steps);
    let e2 = (last_return - model.mean).powi(2);
    let mut prev_var = model.omega + model.alpha * e2 + model.beta * last_var;
    forecasts.push(prev_var);

    for _ in 1..steps {
        prev_var = model.omega + (model.alpha + model.beta) * prev_var;
        forecasts.push(prev_var);
    }
    forecasts
}

// ---------------------------------------------------------------------------
// EGARCH(1,1)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Egarch11 {
    pub omega: f64,
    pub alpha: f64,
    pub gamma: f64, // leverage/asymmetry
    pub beta: f64,
    pub mean: f64,
    pub log_likelihood: f64,
}

pub fn egarch11_fit(returns: &[f64], max_iter: usize) -> Egarch11 {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var = demean.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    let mut omega = sample_var.ln() * 0.05;
    let mut alpha = 0.1;
    let mut gamma = -0.05; // leverage
    let mut beta = 0.9;

    let lr = 1e-6;
    let eps = 1e-8;

    for _iter in 0..max_iter {
        let log_sigma2 = egarch_log_var(&demean, omega, alpha, gamma, beta, sample_var);
        let sigma2: Vec<f64> = log_sigma2.iter().map(|&ls| ls.exp().max(1e-15)).collect();
        let ll_base = garch_log_likelihood(&demean, &sigma2);

        // Gradient for omega
        let ls_p = egarch_log_var(&demean, omega + eps, alpha, gamma, beta, sample_var);
        let s2_p: Vec<f64> = ls_p.iter().map(|&ls| ls.exp().max(1e-15)).collect();
        let g_omega = (garch_log_likelihood(&demean, &s2_p) - ll_base) / eps;

        let ls_p = egarch_log_var(&demean, omega, alpha + eps, gamma, beta, sample_var);
        let s2_p: Vec<f64> = ls_p.iter().map(|&ls| ls.exp().max(1e-15)).collect();
        let g_alpha = (garch_log_likelihood(&demean, &s2_p) - ll_base) / eps;

        let ls_p = egarch_log_var(&demean, omega, alpha, gamma + eps, beta, sample_var);
        let s2_p: Vec<f64> = ls_p.iter().map(|&ls| ls.exp().max(1e-15)).collect();
        let g_gamma = (garch_log_likelihood(&demean, &s2_p) - ll_base) / eps;

        let ls_p = egarch_log_var(&demean, omega, alpha, gamma, beta + eps, sample_var);
        let s2_p: Vec<f64> = ls_p.iter().map(|&ls| ls.exp().max(1e-15)).collect();
        let g_beta = (garch_log_likelihood(&demean, &s2_p) - ll_base) / eps;

        omega += lr * g_omega;
        alpha = (alpha + lr * g_alpha).max(0.0);
        gamma += lr * g_gamma;
        beta = (beta + lr * g_beta).max(0.0).min(0.999);
    }

    let log_sigma2 = egarch_log_var(&demean, omega, alpha, gamma, beta, sample_var);
    let sigma2: Vec<f64> = log_sigma2.iter().map(|&ls| ls.exp().max(1e-15)).collect();
    let ll = garch_log_likelihood(&demean, &sigma2);

    Egarch11 { omega, alpha, gamma, beta, mean, log_likelihood: ll }
}

fn egarch_log_var(demean: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64, init_var: f64) -> Vec<f64> {
    let n = demean.len();
    let mut log_sigma2 = vec![init_var.ln(); n];
    let sqrt_2_pi = (2.0 / PI).sqrt();

    for t in 1..n {
        let sigma_prev = log_sigma2[t - 1].exp().max(1e-15).sqrt();
        let z = demean[t - 1] / sigma_prev;
        log_sigma2[t] = omega + alpha * (z.abs() - sqrt_2_pi) + gamma * z + beta * log_sigma2[t - 1];
    }
    log_sigma2
}

// ---------------------------------------------------------------------------
// GJR-GARCH(1,1) (Glosten-Jagannathan-Runkle)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct GjrGarch11 {
    pub omega: f64,
    pub alpha: f64,
    pub gamma: f64, // asymmetric leverage
    pub beta: f64,
    pub mean: f64,
    pub log_likelihood: f64,
}

pub fn gjr_garch11_fit(returns: &[f64], max_iter: usize) -> GjrGarch11 {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var = demean.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    let mut omega = sample_var * 0.05;
    let mut alpha = 0.05;
    let mut gamma = 0.1; // extra effect for negative shocks
    let mut beta = 0.85;

    let lr = 1e-6;
    let eps = 1e-8;

    for _iter in 0..max_iter {
        let sigma2 = gjr_variance(&demean, omega, alpha, gamma, beta, sample_var);
        let ll_base = garch_log_likelihood(&demean, &sigma2);

        let s2 = gjr_variance(&demean, omega + eps, alpha, gamma, beta, sample_var);
        let g_o = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        let s2 = gjr_variance(&demean, omega, alpha + eps, gamma, beta, sample_var);
        let g_a = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        let s2 = gjr_variance(&demean, omega, alpha, gamma + eps, beta, sample_var);
        let g_g = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        let s2 = gjr_variance(&demean, omega, alpha, gamma, beta + eps, sample_var);
        let g_b = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        omega = (omega + lr * g_o).max(1e-10);
        alpha = (alpha + lr * g_a).max(0.0).min(0.999);
        gamma = (gamma + lr * g_g).max(0.0).min(0.999);
        beta = (beta + lr * g_b).max(0.0).min(0.999);

        if alpha + gamma / 2.0 + beta >= 0.999 {
            let s = 0.998 / (alpha + gamma / 2.0 + beta);
            alpha *= s;
            gamma *= s;
            beta *= s;
        }
    }

    let sigma2 = gjr_variance(&demean, omega, alpha, gamma, beta, sample_var);
    let ll = garch_log_likelihood(&demean, &sigma2);

    GjrGarch11 { omega, alpha, gamma, beta, mean, log_likelihood: ll }
}

fn gjr_variance(demean: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64, init: f64) -> Vec<f64> {
    let n = demean.len();
    let mut sigma2 = vec![init; n];
    for t in 1..n {
        let e2 = demean[t - 1] * demean[t - 1];
        let indicator = if demean[t - 1] < 0.0 { 1.0 } else { 0.0 };
        sigma2[t] = omega + alpha * e2 + gamma * indicator * e2 + beta * sigma2[t - 1];
        sigma2[t] = sigma2[t].max(1e-15);
    }
    sigma2
}

// ---------------------------------------------------------------------------
// TGARCH(1,1) (Threshold GARCH - Zakoian)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Tgarch11 {
    pub omega: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub beta: f64,
    pub mean: f64,
    pub log_likelihood: f64,
}

pub fn tgarch11_fit(returns: &[f64], max_iter: usize) -> Tgarch11 {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_std = (demean.iter().map(|&r| r * r).sum::<f64>() / n as f64).sqrt();

    let mut omega = sample_std * 0.05;
    let mut alpha = 0.05;
    let mut gamma = 0.1;
    let mut beta = 0.85;

    // TGARCH models conditional standard deviation
    let lr = 1e-7;

    for _iter in 0..max_iter {
        let sigma = tgarch_sigma(&demean, omega, alpha, gamma, beta, sample_std);
        let sigma2: Vec<f64> = sigma.iter().map(|&s| (s * s).max(1e-15)).collect();
        let ll_base = garch_log_likelihood(&demean, &sigma2);

        let eps = 1e-8;

        let s = tgarch_sigma(&demean, omega + eps, alpha, gamma, beta, sample_std);
        let s2: Vec<f64> = s.iter().map(|&v| (v * v).max(1e-15)).collect();
        let g_o = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        let s = tgarch_sigma(&demean, omega, alpha + eps, gamma, beta, sample_std);
        let s2: Vec<f64> = s.iter().map(|&v| (v * v).max(1e-15)).collect();
        let g_a = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        let s = tgarch_sigma(&demean, omega, alpha, gamma + eps, beta, sample_std);
        let s2: Vec<f64> = s.iter().map(|&v| (v * v).max(1e-15)).collect();
        let g_g = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        let s = tgarch_sigma(&demean, omega, alpha, gamma, beta + eps, sample_std);
        let s2: Vec<f64> = s.iter().map(|&v| (v * v).max(1e-15)).collect();
        let g_b = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        omega = (omega + lr * g_o).max(1e-10);
        alpha = (alpha + lr * g_a).max(0.0);
        gamma = (gamma + lr * g_g).max(0.0);
        beta = (beta + lr * g_b).max(0.0).min(0.999);
    }

    let sigma = tgarch_sigma(&demean, omega, alpha, gamma, beta, sample_std);
    let sigma2: Vec<f64> = sigma.iter().map(|&s| (s * s).max(1e-15)).collect();
    let ll = garch_log_likelihood(&demean, &sigma2);

    Tgarch11 { omega, alpha, gamma, beta, mean, log_likelihood: ll }
}

fn tgarch_sigma(demean: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64, init: f64) -> Vec<f64> {
    let n = demean.len();
    let mut sigma = vec![init; n];
    for t in 1..n {
        let e = demean[t - 1].abs();
        let neg = if demean[t - 1] < 0.0 { demean[t - 1].abs() } else { 0.0 };
        sigma[t] = omega + alpha * e + gamma * neg + beta * sigma[t - 1];
        sigma[t] = sigma[t].max(1e-10);
    }
    sigma
}

// ---------------------------------------------------------------------------
// APARCH(1,1) (Asymmetric Power ARCH)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Aparch11 {
    pub omega: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub beta: f64,
    pub delta: f64, // power parameter
    pub mean: f64,
    pub log_likelihood: f64,
}

pub fn aparch11_fit(returns: &[f64], max_iter: usize) -> Aparch11 {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var = demean.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    let mut omega = sample_var.sqrt() * 0.05;
    let mut alpha = 0.1;
    let mut gamma = 0.0;
    let mut beta = 0.85;
    let mut delta = 2.0; // start at GARCH

    let lr = 1e-7;
    let eps = 1e-8;

    for _iter in 0..max_iter {
        let sigma_d = aparch_sigma_delta(&demean, omega, alpha, gamma, beta, delta, sample_var);
        let sigma2: Vec<f64> = sigma_d.iter().map(|&sd| sd.powf(2.0 / delta).max(1e-15)).collect();
        let ll_base = garch_log_likelihood(&demean, &sigma2);

        let sd = aparch_sigma_delta(&demean, omega + eps, alpha, gamma, beta, delta, sample_var);
        let s2: Vec<f64> = sd.iter().map(|&v| v.powf(2.0 / delta).max(1e-15)).collect();
        let g_o = (garch_log_likelihood(&demean, &s2) - ll_base) / eps;

        omega = (omega + lr * g_o).max(1e-10);
        alpha = (alpha).max(0.0).min(0.999);
        beta = (beta).max(0.0).min(0.999);
    }

    let sigma_d = aparch_sigma_delta(&demean, omega, alpha, gamma, beta, delta, sample_var);
    let sigma2: Vec<f64> = sigma_d.iter().map(|&sd| sd.powf(2.0 / delta).max(1e-15)).collect();
    let ll = garch_log_likelihood(&demean, &sigma2);

    Aparch11 { omega, alpha, gamma, beta, delta, mean, log_likelihood: ll }
}

fn aparch_sigma_delta(demean: &[f64], omega: f64, alpha: f64, gamma: f64, beta: f64, delta: f64, init_var: f64) -> Vec<f64> {
    let n = demean.len();
    let init_sd = init_var.powf(delta / 2.0);
    let mut sigma_d = vec![init_sd; n];
    for t in 1..n {
        let e = demean[t - 1].abs();
        let asym = (e - gamma * demean[t - 1]).abs().powf(delta);
        sigma_d[t] = omega + alpha * asym + beta * sigma_d[t - 1];
        sigma_d[t] = sigma_d[t].max(1e-15);
    }
    sigma_d
}

// ---------------------------------------------------------------------------
// FIGARCH(1,d,1) (Fractionally Integrated GARCH)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct Figarch {
    pub omega: f64,
    pub beta: f64,
    pub phi: f64,
    pub d: f64, // fractional integration parameter
    pub mean: f64,
    pub log_likelihood: f64,
}

pub fn figarch_fit(returns: &[f64], max_iter: usize) -> Figarch {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var = demean.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    // Simplified: use GARCH(1,1) and estimate d from ACF of squared returns
    let e2: Vec<f64> = demean.iter().map(|&r| r * r).collect();
    let acf_e2 = acf_simple(&e2, 50);

    // Estimate d from decay rate of ACF (Geweke-Porter-Hudak style)
    let mut d = 0.3; // default
    let mut sum_log = 0.0;
    let mut sum_log_acf = 0.0;
    let mut sum_log2 = 0.0;
    let mut count = 0;
    for k in 1..acf_e2.len().min(20) {
        if acf_e2[k] > 0.01 {
            let lk = (k as f64).ln();
            let la = acf_e2[k].ln();
            sum_log += lk;
            sum_log_acf += la;
            sum_log2 += lk * lk;
            count += 1;
        }
    }
    if count > 1 {
        let n_c = count as f64;
        let slope = (n_c * (0..count).zip(1..).map(|(i, k)| {
            let lk = (k as f64).ln();
            if acf_e2[k] > 0.01 { lk * acf_e2[k].ln() } else { 0.0 }
        }).sum::<f64>() - sum_log * sum_log_acf) / (n_c * sum_log2 - sum_log * sum_log);
        d = ((-slope + 1.0) / 2.0).max(0.0).min(0.5);
    }

    let garch = garch11_fit(returns, max_iter / 2);

    Figarch {
        omega: garch.omega,
        beta: garch.beta,
        phi: garch.alpha,
        d,
        mean,
        log_likelihood: garch.log_likelihood,
    }
}

fn acf_simple(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    let mean = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    if var < 1e-15 { return vec![1.0; max_lag + 1]; }
    let mut result = vec![0.0; max_lag + 1];
    result[0] = 1.0;
    for lag in 1..=max_lag.min(n - 1) {
        let mut s = 0.0;
        for i in 0..n - lag {
            s += (data[i] - mean) * (data[i + lag] - mean);
        }
        result[lag] = s / (n as f64 * var);
    }
    result
}

// ---------------------------------------------------------------------------
// Component GARCH
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct ComponentGarch {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,    // trend decay
    pub phi: f64,    // trend AR
    pub mean: f64,
    pub log_likelihood: f64,
}

pub fn component_garch_fit(returns: &[f64], max_iter: usize) -> ComponentGarch {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var = demean.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    // Fit standard GARCH first
    let garch = garch11_fit(returns, max_iter);

    // Decompose into permanent (trend) and transitory components
    let sigma2 = garch11_conditional_variance(&garch, returns);

    // Trend component: slow-moving average of variance
    let mut trend = vec![sample_var; n];
    let rho = 0.99;
    let phi = 0.05;
    for t in 1..n {
        trend[t] = sample_var + rho * (trend[t - 1] - sample_var) + phi * (demean[t - 1] * demean[t - 1] - sigma2[t - 1]);
        trend[t] = trend[t].max(1e-15);
    }

    ComponentGarch {
        omega: garch.omega,
        alpha: garch.alpha,
        beta: garch.beta,
        rho,
        phi,
        mean,
        log_likelihood: garch.log_likelihood,
    }
}

// ---------------------------------------------------------------------------
// DCC-GARCH (simplified bivariate)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct DccGarch {
    pub garch1: Garch11,
    pub garch2: Garch11,
    pub dcc_a: f64,
    pub dcc_b: f64,
    pub unconditional_corr: f64,
}

pub fn dcc_garch_fit(returns1: &[f64], returns2: &[f64], max_iter: usize) -> DccGarch {
    assert_eq!(returns1.len(), returns2.len());
    let n = returns1.len();

    // Fit univariate GARCH to each series
    let g1 = garch11_fit(returns1, max_iter);
    let g2 = garch11_fit(returns2, max_iter);

    // Standardized residuals
    let sigma2_1 = garch11_conditional_variance(&g1, returns1);
    let sigma2_2 = garch11_conditional_variance(&g2, returns2);

    let z1: Vec<f64> = returns1.iter().zip(sigma2_1.iter())
        .map(|(&r, &s2)| (r - g1.mean) / s2.max(1e-15).sqrt())
        .collect();
    let z2: Vec<f64> = returns2.iter().zip(sigma2_2.iter())
        .map(|(&r, &s2)| (r - g2.mean) / s2.max(1e-15).sqrt())
        .collect();

    // Unconditional correlation
    let rho_bar = z1.iter().zip(z2.iter()).map(|(&a, &b)| a * b).sum::<f64>() / n as f64;

    // DCC parameters via grid search
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_a = 0.01;
    let mut best_b = 0.95;

    for a_i in 0..10 {
        for b_i in 0..10 {
            let a = 0.01 + a_i as f64 * 0.03;
            let b = 0.5 + b_i as f64 * 0.05;
            if a + b >= 0.999 { continue; }

            let mut qt = rho_bar;
            let mut ll = 0.0;
            for t in 1..n {
                qt = rho_bar * (1.0 - a - b) + a * z1[t - 1] * z2[t - 1] + b * qt;
                let rho_t = qt / (1.0 + 1e-15); // simplified normalization
                let rho_t = rho_t.max(-0.999).min(0.999);
                let det = 1.0 - rho_t * rho_t;
                if det > 1e-15 {
                    ll += -0.5 * (det.ln() + (z1[t] * z1[t] - 2.0 * rho_t * z1[t] * z2[t] + z2[t] * z2[t]) / det);
                }
            }

            if ll > best_ll {
                best_ll = ll;
                best_a = a;
                best_b = b;
            }
        }
    }

    DccGarch {
        garch1: g1,
        garch2: g2,
        dcc_a: best_a,
        dcc_b: best_b,
        unconditional_corr: rho_bar,
    }
}

pub fn dcc_dynamic_correlation(model: &DccGarch, returns1: &[f64], returns2: &[f64]) -> Vec<f64> {
    let n = returns1.len();
    let sigma2_1 = garch11_conditional_variance(&model.garch1, returns1);
    let sigma2_2 = garch11_conditional_variance(&model.garch2, returns2);

    let z1: Vec<f64> = returns1.iter().zip(sigma2_1.iter())
        .map(|(&r, &s2)| (r - model.garch1.mean) / s2.max(1e-15).sqrt())
        .collect();
    let z2: Vec<f64> = returns2.iter().zip(sigma2_2.iter())
        .map(|(&r, &s2)| (r - model.garch2.mean) / s2.max(1e-15).sqrt())
        .collect();

    let mut corrs = vec![model.unconditional_corr; n];
    let mut qt = model.unconditional_corr;
    for t in 1..n {
        qt = model.unconditional_corr * (1.0 - model.dcc_a - model.dcc_b)
            + model.dcc_a * z1[t - 1] * z2[t - 1]
            + model.dcc_b * qt;
        corrs[t] = qt.max(-0.999).min(0.999);
    }
    corrs
}

// ---------------------------------------------------------------------------
// Realized GARCH
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RealizedGarch {
    pub omega: f64,
    pub beta: f64,
    pub gamma: f64,  // realized measure coefficient
    pub xi: f64,     // measurement equation intercept
    pub phi: f64,    // measurement equation slope
    pub tau1: f64,   // leverage
    pub tau2: f64,
    pub sigma2_r: f64,
    pub mean: f64,
    pub log_likelihood: f64,
}

pub fn realized_garch_fit(returns: &[f64], realized_var: &[f64], max_iter: usize) -> RealizedGarch {
    assert_eq!(returns.len(), realized_var.len());
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let demean: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var = demean.iter().map(|&r| r * r).sum::<f64>() / n as f64;

    // Simple estimation via regression
    let log_rv: Vec<f64> = realized_var.iter().map(|&rv| rv.max(1e-15).ln()).collect();

    // Variance equation: log(h_t) = omega + beta*log(h_{t-1}) + gamma*log(x_{t-1})
    // Measurement: log(x_t) = xi + phi*log(h_t) + tau1*z_t + tau2*(z_t^2 - 1) + u_t

    let mut log_h = vec![sample_var.ln(); n];
    let omega = sample_var.ln() * 0.05;
    let beta = 0.9;
    let gamma_coeff = 0.1;

    for t in 1..n {
        log_h[t] = omega + beta * log_h[t - 1] + gamma_coeff * log_rv[t - 1];
    }

    // Measurement equation regression
    let z: Vec<f64> = demean.iter().zip(log_h.iter())
        .map(|(&e, &lh)| e / lh.exp().max(1e-15).sqrt())
        .collect();

    let mut xi = 0.0;
    let mut phi = 1.0;
    let mut tau1 = 0.0;
    let mut tau2 = 0.0;

    // Simple OLS for measurement equation
    if n > 4 {
        let mut sx = 0.0; let mut sy = 0.0; let mut sxy = 0.0; let mut sxx = 0.0;
        for t in 0..n {
            let x = log_h[t];
            let y = log_rv[t];
            sx += x; sy += y; sxy += x * y; sxx += x * x;
        }
        let nn = n as f64;
        let denom = nn * sxx - sx * sx;
        if denom.abs() > 1e-15 {
            phi = (nn * sxy - sx * sy) / denom;
            xi = (sy - phi * sx) / nn;
        }
    }

    let sigma2_r = 0.1; // residual variance of measurement equation

    let sigma2: Vec<f64> = log_h.iter().map(|&lh| lh.exp().max(1e-15)).collect();
    let ll = garch_log_likelihood(&demean, &sigma2);

    RealizedGarch {
        omega, beta, gamma: gamma_coeff, xi, phi, tau1, tau2, sigma2_r,
        mean, log_likelihood: ll,
    }
}

// ---------------------------------------------------------------------------
// GARCH Multi-step Forecasting
// ---------------------------------------------------------------------------
pub fn garch_multistep_forecast(
    omega: f64, alpha: f64, beta: f64,
    last_e2: f64, last_sigma2: f64,
    steps: usize,
) -> Vec<f64> {
    let mut forecasts = Vec::with_capacity(steps);
    let mut prev_sigma2 = omega + alpha * last_e2 + beta * last_sigma2;
    forecasts.push(prev_sigma2);

    let ab = alpha + beta;
    for _ in 1..steps {
        prev_sigma2 = omega + ab * prev_sigma2;
        forecasts.push(prev_sigma2);
    }
    forecasts
}

// ---------------------------------------------------------------------------
// VaR from GARCH
// ---------------------------------------------------------------------------
pub fn garch_var(
    model: &Garch11,
    last_return: f64,
    last_var: f64,
    confidence: f64,
) -> f64 {
    let e2 = (last_return - model.mean).powi(2);
    let next_var = model.omega + model.alpha * e2 + model.beta * last_var;
    let z = normal_quantile(1.0 - confidence);
    model.mean + z * next_var.sqrt()
}

pub fn garch_var_series(
    model: &Garch11,
    returns: &[f64],
    confidence: f64,
) -> Vec<f64> {
    let sigma2 = garch11_conditional_variance(model, returns);
    let z = normal_quantile(1.0 - confidence);
    sigma2.iter().map(|&s2| model.mean + z * s2.sqrt()).collect()
}

// ---------------------------------------------------------------------------
// News Impact Curve
// ---------------------------------------------------------------------------
pub fn news_impact_curve(model: &Garch11, n_points: usize, range: f64) -> Vec<(f64, f64)> {
    let unconditional = model.unconditional_var;
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let shock = -range + 2.0 * range * i as f64 / (n_points - 1).max(1) as f64;
        let sigma2 = model.omega + model.alpha * shock * shock + model.beta * unconditional;
        points.push((shock, sigma2));
    }
    points
}

pub fn gjr_news_impact_curve(model: &GjrGarch11, n_points: usize, range: f64) -> Vec<(f64, f64)> {
    let sample_var = model.omega / (1.0 - model.alpha - model.gamma / 2.0 - model.beta).max(0.01);
    let mut points = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let shock = -range + 2.0 * range * i as f64 / (n_points - 1).max(1) as f64;
        let indicator = if shock < 0.0 { 1.0 } else { 0.0 };
        let sigma2 = model.omega + model.alpha * shock * shock + model.gamma * indicator * shock * shock + model.beta * sample_var;
        points.push((shock, sigma2));
    }
    points
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    let t = if p < 0.5 { (-2.0 * p.ln()).sqrt() } else { (-2.0 * (1.0 - p).ln()).sqrt() };
    let c0 = 2.515517; let c1 = 0.802853; let c2 = 0.010328;
    let d1 = 1.432788; let d2 = 0.189269; let d3 = 0.001308;
    let q = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -q } else { q }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_garch_data(n: usize) -> Vec<f64> {
        let mut data = vec![0.0; n];
        let mut sigma2 = 0.01;
        for i in 0..n {
            let noise = ((i as f64 * 1.618033).sin() * 43758.5453).fract() * 2.0 - 1.0;
            let r = noise * sigma2.sqrt();
            sigma2 = 0.00001 + 0.1 * r * r + 0.85 * sigma2;
            data[i] = r;
        }
        data
    }

    #[test]
    fn test_garch11_fit() {
        let data = make_garch_data(500);
        let model = garch11_fit(&data, 100);
        assert!(model.omega > 0.0);
        assert!(model.alpha > 0.0 && model.alpha < 1.0);
        assert!(model.beta > 0.0 && model.beta < 1.0);
        assert!(model.persistence < 1.0);
    }

    #[test]
    fn test_garch_forecast() {
        let data = make_garch_data(200);
        let model = garch11_fit(&data, 50);
        let sigma2 = garch11_conditional_variance(&model, &data);
        let fc = garch11_forecast(&model, *data.last().unwrap(), *sigma2.last().unwrap(), 10);
        assert_eq!(fc.len(), 10);
        for &v in &fc { assert!(v > 0.0); }
    }

    #[test]
    fn test_egarch() {
        let data = make_garch_data(300);
        let model = egarch11_fit(&data, 50);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_gjr_garch() {
        let data = make_garch_data(300);
        let model = gjr_garch11_fit(&data, 50);
        assert!(model.gamma >= 0.0);
    }

    #[test]
    fn test_tgarch() {
        let data = make_garch_data(200);
        let model = tgarch11_fit(&data, 50);
        assert!(model.omega > 0.0);
    }

    #[test]
    fn test_aparch() {
        let data = make_garch_data(200);
        let model = aparch11_fit(&data, 50);
        assert!(model.delta > 0.0);
    }

    #[test]
    fn test_component_garch() {
        let data = make_garch_data(200);
        let model = component_garch_fit(&data, 50);
        assert!(model.rho > 0.0);
    }

    #[test]
    fn test_dcc_garch() {
        let data1 = make_garch_data(200);
        let data2 = make_garch_data(200);
        let model = dcc_garch_fit(&data1, &data2, 50);
        assert!(model.dcc_a > 0.0);
        let corrs = dcc_dynamic_correlation(&model, &data1, &data2);
        assert_eq!(corrs.len(), 200);
    }

    #[test]
    fn test_var() {
        let data = make_garch_data(200);
        let model = garch11_fit(&data, 50);
        let sigma2 = garch11_conditional_variance(&model, &data);
        let var = garch_var(&model, *data.last().unwrap(), *sigma2.last().unwrap(), 0.95);
        assert!(var < 0.0); // VaR is typically negative
    }

    #[test]
    fn test_news_impact() {
        let data = make_garch_data(200);
        let model = garch11_fit(&data, 50);
        let nic = news_impact_curve(&model, 21, 0.1);
        assert_eq!(nic.len(), 21);
        // Should be symmetric for standard GARCH
        let mid = nic[10].1;
        assert!(nic[0].1 > mid);
        assert!(nic[20].1 > mid);
    }

    #[test]
    fn test_multistep_forecast() {
        let fc = garch_multistep_forecast(0.00001, 0.1, 0.85, 0.01, 0.02, 10);
        assert_eq!(fc.len(), 10);
        // Should converge to unconditional variance
    }

    #[test]
    fn test_figarch() {
        let data = make_garch_data(200);
        let model = figarch_fit(&data, 50);
        assert!(model.d >= 0.0 && model.d <= 0.5);
    }

    #[test]
    fn test_realized_garch() {
        let returns = make_garch_data(200);
        let rv: Vec<f64> = returns.iter().map(|&r| r * r + 0.001).collect();
        let model = realized_garch_fit(&returns, &rv, 50);
        assert!(model.log_likelihood.is_finite());
    }
}
