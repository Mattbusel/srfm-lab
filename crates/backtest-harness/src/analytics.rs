// analytics.rs — Performance metrics, drawdown, rolling, regime stats, Monte Carlo, CPCV, Deflated Sharpe
use std::collections::HashMap;

/// Core performance metrics
#[derive(Clone, Debug)]
pub struct PerformanceMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub omega_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub ulcer_index: f64,
    pub tail_ratio: f64,
    pub information_ratio: f64,
    pub treynor_ratio: f64,
    pub beta: f64,
    pub alpha: f64,
}

/// Compute all metrics from return series
pub fn compute_metrics(returns: &[f64], risk_free_rate: f64, periods_per_year: f64) -> PerformanceMetrics {
    let n = returns.len();
    if n < 2 {
        return PerformanceMetrics {
            total_return: 0.0, annualized_return: 0.0, annualized_volatility: 0.0,
            sharpe_ratio: 0.0, sortino_ratio: 0.0, calmar_ratio: 0.0, omega_ratio: 1.0,
            max_drawdown: 0.0, max_drawdown_duration: 0, win_rate: 0.0, profit_factor: 0.0,
            expectancy: 0.0, skewness: 0.0, kurtosis: 0.0, var_95: 0.0, cvar_95: 0.0,
            ulcer_index: 0.0, tail_ratio: 1.0, information_ratio: 0.0, treynor_ratio: 0.0,
            beta: 0.0, alpha: 0.0,
        };
    }

    let total_return = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
    let mean_ret = returns.iter().sum::<f64>() / n as f64;
    let var = returns.iter().map(|&r| (r - mean_ret).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = var.sqrt();
    let ann_ret = mean_ret * periods_per_year;
    let ann_vol = std * periods_per_year.sqrt();
    let rf_period = risk_free_rate / periods_per_year;

    let sharpe = if ann_vol > 1e-15 { (ann_ret - risk_free_rate) / ann_vol } else { 0.0 };

    // Sortino
    let downside_var = returns.iter()
        .map(|&r| if r < rf_period { (r - rf_period).powi(2) } else { 0.0 })
        .sum::<f64>() / n as f64;
    let downside_dev = downside_var.sqrt() * periods_per_year.sqrt();
    let sortino = if downside_dev > 1e-15 { (ann_ret - risk_free_rate) / downside_dev } else { 0.0 };

    // Drawdown
    let (mdd, mdd_dur) = max_drawdown_with_duration(returns);
    let calmar = if mdd > 1e-15 { ann_ret / mdd } else { 0.0 };

    // Omega ratio at threshold 0
    let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    let omega = if losses > 1e-15 { 1.0 + gains / losses } else { f64::INFINITY };

    let winning = returns.iter().filter(|&&r| r > 0.0).count();
    let win_rate = winning as f64 / n as f64;
    let profit_factor = if losses > 1e-15 { gains / losses } else { f64::INFINITY };
    let avg_win = if winning > 0 { returns.iter().filter(|&&r| r > 0.0).sum::<f64>() / winning as f64 } else { 0.0 };
    let losing = n - winning;
    let avg_loss = if losing > 0 { returns.iter().filter(|&&r| r <= 0.0).sum::<f64>() / losing as f64 } else { 0.0 };
    let expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss;

    // Higher moments
    let skewness = if n > 2 && std > 1e-15 {
        let m3 = returns.iter().map(|&r| ((r - mean_ret) / std).powi(3)).sum::<f64>();
        m3 * n as f64 / ((n - 1) as f64 * (n - 2) as f64)
    } else { 0.0 };

    let kurtosis = if n > 3 && std > 1e-15 {
        let m4 = returns.iter().map(|&r| ((r - mean_ret) / std).powi(4)).sum::<f64>();
        m4 / n as f64 - 3.0
    } else { 0.0 };

    // VaR and CVaR
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx_95 = ((n as f64 * 0.05) as usize).max(0);
    let var_95 = if idx_95 < n { -sorted[idx_95] } else { 0.0 };
    let cvar_95 = if idx_95 > 0 { -sorted[..idx_95].iter().sum::<f64>() / idx_95 as f64 } else { var_95 };

    // Ulcer index
    let ui = ulcer_index(returns);

    // Tail ratio: 95th percentile / 5th percentile
    let idx_hi = ((n as f64 * 0.95) as usize).min(n - 1);
    let tail_ratio = if sorted[idx_95].abs() > 1e-15 { sorted[idx_hi].abs() / sorted[idx_95].abs() } else { 1.0 };

    PerformanceMetrics {
        total_return, annualized_return: ann_ret, annualized_volatility: ann_vol,
        sharpe_ratio: sharpe, sortino_ratio: sortino, calmar_ratio: calmar, omega_ratio: omega,
        max_drawdown: mdd, max_drawdown_duration: mdd_dur,
        win_rate, profit_factor, expectancy, skewness, kurtosis,
        var_95, cvar_95, ulcer_index: ui, tail_ratio,
        information_ratio: 0.0, treynor_ratio: 0.0, beta: 0.0, alpha: 0.0,
    }
}

/// Compute metrics with benchmark
pub fn compute_metrics_with_benchmark(
    returns: &[f64], benchmark: &[f64], risk_free_rate: f64, periods_per_year: f64,
) -> PerformanceMetrics {
    let mut metrics = compute_metrics(returns, risk_free_rate, periods_per_year);
    let n = returns.len().min(benchmark.len());
    if n < 2 { return metrics; }

    // Excess returns
    let excess: Vec<f64> = returns[..n].iter().zip(benchmark[..n].iter())
        .map(|(&r, &b)| r - b).collect();
    let mean_excess = excess.iter().sum::<f64>() / n as f64;
    let std_excess = (excess.iter().map(|&e| (e - mean_excess).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    metrics.information_ratio = if std_excess > 1e-15 {
        mean_excess * periods_per_year.sqrt() / (std_excess * periods_per_year.sqrt())
    } else { 0.0 };

    // Beta
    let mean_r = returns[..n].iter().sum::<f64>() / n as f64;
    let mean_b = benchmark[..n].iter().sum::<f64>() / n as f64;
    let cov: f64 = returns[..n].iter().zip(benchmark[..n].iter())
        .map(|(&r, &b)| (r - mean_r) * (b - mean_b)).sum::<f64>() / (n - 1) as f64;
    let var_b: f64 = benchmark[..n].iter().map(|&b| (b - mean_b).powi(2)).sum::<f64>() / (n - 1) as f64;
    metrics.beta = if var_b > 1e-15 { cov / var_b } else { 0.0 };
    metrics.alpha = metrics.annualized_return - risk_free_rate - metrics.beta * (mean_b * periods_per_year - risk_free_rate);
    metrics.treynor_ratio = if metrics.beta.abs() > 1e-15 {
        (metrics.annualized_return - risk_free_rate) / metrics.beta
    } else { 0.0 };

    metrics
}

/// Max drawdown with duration
pub fn max_drawdown_with_duration(returns: &[f64]) -> (f64, usize) {
    let mut peak = 1.0;
    let mut equity = 1.0;
    let mut mdd = 0.0;
    let mut dd_start = 0;
    let mut mdd_dur = 0usize;
    let mut current_dur = 0usize;

    for &r in returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
            current_dur = 0;
        } else {
            current_dur += 1;
        }
        let dd = (peak - equity) / peak;
        if dd > mdd {
            mdd = dd;
            mdd_dur = current_dur;
        }
    }
    (mdd, mdd_dur)
}

/// Drawdown series
pub fn drawdown_series(returns: &[f64]) -> Vec<f64> {
    let mut peak = 1.0;
    let mut equity = 1.0;
    let mut dd = Vec::with_capacity(returns.len());
    for &r in returns {
        equity *= 1.0 + r;
        if equity > peak { peak = equity; }
        dd.push((peak - equity) / peak);
    }
    dd
}

/// Underwater equity curve
pub fn underwater_curve(returns: &[f64]) -> Vec<f64> {
    drawdown_series(returns).iter().map(|d| -d).collect()
}

/// Ulcer index
pub fn ulcer_index(returns: &[f64]) -> f64 {
    let dd = drawdown_series(returns);
    if dd.is_empty() { return 0.0; }
    let sum_sq: f64 = dd.iter().map(|&d| d * d).sum();
    (sum_sq / dd.len() as f64).sqrt()
}

/// Rolling Sharpe ratio
pub fn rolling_sharpe(returns: &[f64], window: usize, risk_free_rate: f64, periods_per_year: f64) -> Vec<f64> {
    if returns.len() < window { return vec![]; }
    let rf_period = risk_free_rate / periods_per_year;
    (0..=returns.len() - window).map(|i| {
        let w = &returns[i..i + window];
        let mean = w.iter().sum::<f64>() / window as f64;
        let std = (w.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (window - 1) as f64).sqrt();
        if std > 1e-15 { (mean - rf_period) * periods_per_year.sqrt() / (std * periods_per_year.sqrt()) } else { 0.0 }
    }).collect()
}

/// Rolling Sortino
pub fn rolling_sortino(returns: &[f64], window: usize, risk_free_rate: f64, periods_per_year: f64) -> Vec<f64> {
    if returns.len() < window { return vec![]; }
    let rf_period = risk_free_rate / periods_per_year;
    (0..=returns.len() - window).map(|i| {
        let w = &returns[i..i + window];
        let mean = w.iter().sum::<f64>() / window as f64;
        let dd = w.iter().map(|&r| if r < rf_period { (r - rf_period).powi(2) } else { 0.0 })
            .sum::<f64>() / window as f64;
        let ds = dd.sqrt() * periods_per_year.sqrt();
        if ds > 1e-15 { (mean * periods_per_year - risk_free_rate) / ds } else { 0.0 }
    }).collect()
}

/// Rolling volatility
pub fn rolling_volatility(returns: &[f64], window: usize, periods_per_year: f64) -> Vec<f64> {
    if returns.len() < window { return vec![]; }
    (0..=returns.len() - window).map(|i| {
        let w = &returns[i..i + window];
        let mean = w.iter().sum::<f64>() / window as f64;
        let var = w.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (window - 1) as f64;
        var.sqrt() * periods_per_year.sqrt()
    }).collect()
}

/// Rolling max drawdown
pub fn rolling_max_drawdown(returns: &[f64], window: usize) -> Vec<f64> {
    if returns.len() < window { return vec![]; }
    (0..=returns.len() - window).map(|i| {
        max_drawdown_with_duration(&returns[i..i + window]).0
    }).collect()
}

/// Rolling beta
pub fn rolling_beta(returns: &[f64], benchmark: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len().min(benchmark.len());
    if n < window { return vec![]; }
    (0..=n - window).map(|i| {
        let r = &returns[i..i + window];
        let b = &benchmark[i..i + window];
        let mr = r.iter().sum::<f64>() / window as f64;
        let mb = b.iter().sum::<f64>() / window as f64;
        let cov: f64 = r.iter().zip(b.iter()).map(|(&ri, &bi)| (ri - mr) * (bi - mb)).sum::<f64>() / window as f64;
        let var: f64 = b.iter().map(|&bi| (bi - mb).powi(2)).sum::<f64>() / window as f64;
        if var > 1e-15 { cov / var } else { 0.0 }
    }).collect()
}

/// Regime-conditional metrics
pub fn regime_conditional_metrics(
    returns: &[f64], regimes: &[usize], risk_free_rate: f64, periods_per_year: f64,
) -> HashMap<usize, PerformanceMetrics> {
    let mut by_regime: HashMap<usize, Vec<f64>> = HashMap::new();
    for (&r, &regime) in returns.iter().zip(regimes.iter()) {
        by_regime.entry(regime).or_default().push(r);
    }
    by_regime.into_iter().map(|(regime, rets)| {
        (regime, compute_metrics(&rets, risk_free_rate, periods_per_year))
    }).collect()
}

/// Equity curve from returns
pub fn equity_curve(returns: &[f64], initial: f64) -> Vec<f64> {
    let mut curve = Vec::with_capacity(returns.len() + 1);
    curve.push(initial);
    for &r in returns {
        curve.push(curve.last().unwrap() * (1.0 + r));
    }
    curve
}

/// Monthly returns table
pub fn monthly_returns(returns: &[f64], bars_per_month: usize) -> Vec<f64> {
    returns.chunks(bars_per_month).map(|chunk| {
        chunk.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0
    }).collect()
}

/// Annual returns
pub fn annual_returns(returns: &[f64], bars_per_year: usize) -> Vec<f64> {
    returns.chunks(bars_per_year).map(|chunk| {
        chunk.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0
    }).collect()
}

/// Monte Carlo bootstrap confidence intervals
pub fn monte_carlo_bootstrap(
    returns: &[f64], num_sims: usize, sim_length: usize, seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Returns (5th percentile, median, 95th percentile) equity curves
    let n = returns.len();
    if n == 0 { return (vec![], vec![], vec![]); }

    let mut all_final: Vec<f64> = Vec::with_capacity(num_sims);
    let mut state = seed;

    for _ in 0..num_sims {
        let mut equity = 1.0;
        for _ in 0..sim_length {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (state >> 32) as usize % n;
            equity *= 1.0 + returns[idx];
        }
        all_final.push(equity);
    }
    all_final.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p5 = all_final[(num_sims as f64 * 0.05) as usize];
    let p50 = all_final[num_sims / 2];
    let p95 = all_final[(num_sims as f64 * 0.95) as usize];

    // Build approximate curves (just endpoints for now)
    (vec![1.0, p5], vec![1.0, p50], vec![1.0, p95])
}

/// Full Monte Carlo paths
pub fn monte_carlo_equity_paths(
    returns: &[f64], num_paths: usize, path_length: usize, seed: u64,
) -> Vec<Vec<f64>> {
    let n = returns.len();
    if n == 0 { return vec![]; }
    let mut paths = Vec::with_capacity(num_paths);
    let mut state = seed;

    for _ in 0..num_paths {
        let mut path = Vec::with_capacity(path_length + 1);
        path.push(1.0);
        for _ in 0..path_length {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (state >> 32) as usize % n;
            path.push(path.last().unwrap() * (1.0 + returns[idx]));
        }
        paths.push(path);
    }
    paths
}

/// Combinatorial Purged Cross-Validation (CPCV)
pub fn cpcv_splits(num_samples: usize, num_groups: usize, num_test_groups: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let group_size = num_samples / num_groups;
    let groups: Vec<Vec<usize>> = (0..num_groups).map(|g| {
        let start = g * group_size;
        let end = if g == num_groups - 1 { num_samples } else { (g + 1) * group_size };
        (start..end).collect()
    }).collect();

    // Generate all combinations of num_test_groups from num_groups
    let mut splits = Vec::new();
    let mut combo = vec![0usize; num_test_groups];
    fn generate_combos(groups: &[Vec<usize>], combo: &mut Vec<usize>, start: usize, depth: usize, target: usize, splits: &mut Vec<(Vec<usize>, Vec<usize>)>) {
        if depth == target {
            let test: Vec<usize> = combo.iter().flat_map(|&g| groups[g].clone()).collect();
            let test_set: std::collections::HashSet<usize> = test.iter().cloned().collect();
            let train: Vec<usize> = groups.iter().enumerate()
                .filter(|(i, _)| !combo.contains(i))
                .flat_map(|(_, g)| g.clone())
                .collect();
            splits.push((train, test));
            return;
        }
        for i in start..groups.len() {
            combo[depth] = i;
            generate_combos(groups, combo, i + 1, depth + 1, target, splits);
        }
    }
    generate_combos(&groups, &mut combo, 0, 0, num_test_groups, &mut splits);
    splits
}

/// Deflated Sharpe Ratio (DSR)
/// Tests whether the observed Sharpe ratio is statistically significant
pub fn deflated_sharpe_ratio(
    observed_sharpe: f64,
    num_trials: usize,   // how many strategies were tried
    returns_length: usize,
    skewness: f64,
    kurtosis: f64,
) -> f64 {
    let n = returns_length as f64;
    if n < 2.0 { return 0.0; }

    // Expected max Sharpe under null (Euler-Mascheroni)
    let euler_gamma = 0.5772156649;
    let expected_max_sr = ((2.0 * (num_trials as f64).ln()).sqrt())
        - ((2.0 * (num_trials as f64).ln()).ln() + (4.0 * std::f64::consts::PI).ln())
        / (2.0 * (2.0 * (num_trials as f64).ln()).sqrt());

    // Standard error of Sharpe
    let se_sr = ((1.0 - skewness * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe * observed_sharpe) / (n - 1.0)).sqrt();

    if se_sr < 1e-15 { return 0.0; }

    // z-statistic
    let z = (observed_sharpe - expected_max_sr) / se_sr;

    // CDF of standard normal (approximate)
    standard_normal_cdf(z)
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let t = 1.0 / (1.0 + p * x.abs());
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Minimum Track Record Length (MinTRL)
pub fn min_track_record_length(observed_sharpe: f64, benchmark_sharpe: f64, skewness: f64, kurtosis: f64, confidence: f64) -> f64 {
    let z = standard_normal_cdf_inverse(confidence);
    let sr_diff = observed_sharpe - benchmark_sharpe;
    if sr_diff.abs() < 1e-15 { return f64::INFINITY; }
    let se_factor = 1.0 - skewness * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe * observed_sharpe;
    (z * z * se_factor) / (sr_diff * sr_diff)
}

fn standard_normal_cdf_inverse(p: f64) -> f64 {
    // Rational approximation (Beasley-Springer-Moro)
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    let t = if p < 0.5 { (-2.0 * p.ln()).sqrt() } else { (-2.0 * (1.0 - p).ln()).sqrt() };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -val } else { val }
}

/// Probabilistic Sharpe Ratio
pub fn probabilistic_sharpe_ratio(observed_sharpe: f64, benchmark_sharpe: f64, n: usize, skewness: f64, kurtosis: f64) -> f64 {
    let nf = n as f64;
    let se = ((1.0 - skewness * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe.powi(2)) / (nf - 1.0)).sqrt();
    if se < 1e-15 { return 0.5; }
    standard_normal_cdf((observed_sharpe - benchmark_sharpe) / se)
}

/// Hurst exponent estimation (R/S method)
pub fn hurst_exponent(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 20 { return 0.5; }

    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();

    let sizes = [10, 20, 40, 80, 160, 320];
    for &size in &sizes {
        if size > n / 2 { break; }
        let num_blocks = n / size;
        let mut rs_values = Vec::new();

        for b in 0..num_blocks {
            let block = &returns[b * size..(b + 1) * size];
            let mean = block.iter().sum::<f64>() / size as f64;
            let deviations: Vec<f64> = block.iter().map(|&r| r - mean).collect();
            let cumsum: Vec<f64> = deviations.iter().scan(0.0, |acc, &d| { *acc += d; Some(*acc) }).collect();
            let range = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - cumsum.iter().cloned().fold(f64::INFINITY, f64::min);
            let std = (block.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / size as f64).sqrt();
            if std > 1e-15 { rs_values.push(range / std); }
        }

        if !rs_values.is_empty() {
            let avg_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
            log_n.push((size as f64).ln());
            log_rs.push(avg_rs.ln());
        }
    }

    if log_n.len() < 2 { return 0.5; }

    // Linear regression
    let n_pts = log_n.len() as f64;
    let mean_x = log_n.iter().sum::<f64>() / n_pts;
    let mean_y = log_rs.iter().sum::<f64>() / n_pts;
    let cov: f64 = log_n.iter().zip(log_rs.iter()).map(|(&x, &y)| (x - mean_x) * (y - mean_y)).sum();
    let var: f64 = log_n.iter().map(|&x| (x - mean_x).powi(2)).sum();
    if var < 1e-15 { return 0.5; }
    cov / var
}

/// Stability of returns: fraction of positive rolling windows
pub fn stability_of_returns(returns: &[f64], window: usize) -> f64 {
    if returns.len() < window { return 0.0; }
    let total = returns.len() - window + 1;
    let positive = (0..total).filter(|&i| {
        returns[i..i + window].iter().sum::<f64>() > 0.0
    }).count();
    positive as f64 / total as f64
}

/// Gain-to-pain ratio
pub fn gain_to_pain(returns: &[f64]) -> f64 {
    let gains: f64 = returns.iter().sum();
    let pain: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
    if pain < 1e-15 { return f64::INFINITY; }
    gains / pain
}

/// Common Sense Ratio
pub fn common_sense_ratio(returns: &[f64]) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n < 20 { return 1.0; }
    let tail95 = sorted[(n as f64 * 0.95) as usize];
    let tail5 = sorted[(n as f64 * 0.05) as usize];
    let tail_ratio = if tail5.abs() > 1e-15 { tail95.abs() / tail5.abs() } else { 1.0 };
    let pf = {
        let gains: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        if losses > 1e-15 { gains / losses } else { 1.0 }
    };
    tail_ratio * pf
}

/// Drawdown recovery analysis
pub fn drawdown_periods(returns: &[f64]) -> Vec<(usize, usize, f64, usize)> {
    // returns (start, trough, max_dd, recovery_end)
    let mut periods = Vec::new();
    let mut peak = 1.0;
    let mut equity = 1.0;
    let mut in_dd = false;
    let mut dd_start = 0;
    let mut dd_trough = 0;
    let mut max_dd = 0.0;

    for (i, &r) in returns.iter().enumerate() {
        equity *= 1.0 + r;
        if equity > peak {
            if in_dd {
                periods.push((dd_start, dd_trough, max_dd, i));
                in_dd = false;
            }
            peak = equity;
            max_dd = 0.0;
        } else {
            if !in_dd {
                dd_start = i;
                in_dd = true;
            }
            let dd = (peak - equity) / peak;
            if dd > max_dd {
                max_dd = dd;
                dd_trough = i;
            }
        }
    }
    if in_dd {
        periods.push((dd_start, dd_trough, max_dd, returns.len()));
    }
    periods
}

/// Performance summary as formatted string
pub fn format_metrics(m: &PerformanceMetrics) -> String {
    let mut s = String::new();
    s.push_str(&format!("Total Return:       {:.2}%\n", m.total_return * 100.0));
    s.push_str(&format!("Annualized Return:  {:.2}%\n", m.annualized_return * 100.0));
    s.push_str(&format!("Annualized Vol:     {:.2}%\n", m.annualized_volatility * 100.0));
    s.push_str(&format!("Sharpe Ratio:       {:.3}\n", m.sharpe_ratio));
    s.push_str(&format!("Sortino Ratio:      {:.3}\n", m.sortino_ratio));
    s.push_str(&format!("Calmar Ratio:       {:.3}\n", m.calmar_ratio));
    s.push_str(&format!("Omega Ratio:        {:.3}\n", m.omega_ratio));
    s.push_str(&format!("Max Drawdown:       {:.2}%\n", m.max_drawdown * 100.0));
    s.push_str(&format!("Max DD Duration:    {} bars\n", m.max_drawdown_duration));
    s.push_str(&format!("Win Rate:           {:.2}%\n", m.win_rate * 100.0));
    s.push_str(&format!("Profit Factor:      {:.3}\n", m.profit_factor));
    s.push_str(&format!("Skewness:           {:.3}\n", m.skewness));
    s.push_str(&format!("Kurtosis:           {:.3}\n", m.kurtosis));
    s.push_str(&format!("VaR 95:             {:.4}\n", m.var_95));
    s.push_str(&format!("CVaR 95:            {:.4}\n", m.cvar_95));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.003, 0.012, 0.008, -0.002];
        let m = compute_metrics(&returns, 0.0, 252.0);
        assert!(m.sharpe_ratio.is_finite());
        assert!(m.annualized_return > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, -0.2, 0.05, -0.1, 0.15];
        let (mdd, _) = max_drawdown_with_duration(&returns);
        assert!(mdd > 0.0 && mdd < 1.0);
    }

    #[test]
    fn test_rolling_sharpe() {
        let returns: Vec<f64> = (0..100).map(|i| 0.001 * (i as f64 % 5.0 - 2.0)).collect();
        let rs = rolling_sharpe(&returns, 20, 0.0, 252.0);
        assert_eq!(rs.len(), 81);
    }

    #[test]
    fn test_deflated_sharpe() {
        let dsr = deflated_sharpe_ratio(2.0, 100, 252, 0.0, 3.0);
        assert!(dsr >= 0.0 && dsr <= 1.0);
    }

    #[test]
    fn test_cpcv() {
        let splits = cpcv_splits(100, 5, 1);
        assert_eq!(splits.len(), 5); // C(5,1) = 5
    }

    #[test]
    fn test_hurst() {
        let returns: Vec<f64> = (0..1000).map(|i| ((i as f64) * 0.1).sin() * 0.01).collect();
        let h = hurst_exponent(&returns);
        assert!(h > 0.0 && h < 1.0);
    }

    #[test]
    fn test_equity_curve() {
        let returns = vec![0.1, 0.05, -0.03];
        let curve = equity_curve(&returns, 100.0);
        assert_eq!(curve.len(), 4);
        assert!((curve[1] - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_monthly_returns() {
        let returns: Vec<f64> = vec![0.01; 60];
        let monthly = monthly_returns(&returns, 20);
        assert_eq!(monthly.len(), 3);
    }

    #[test]
    fn test_gain_to_pain() {
        let returns = vec![0.05, -0.02, 0.03, -0.01];
        let gp = gain_to_pain(&returns);
        assert!(gp > 0.0);
    }

    #[test]
    fn test_drawdown_periods() {
        let returns = vec![0.1, -0.15, -0.05, 0.2, 0.1, -0.3, 0.1];
        let periods = drawdown_periods(&returns);
        assert!(!periods.is_empty());
    }
}
