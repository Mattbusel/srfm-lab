// risk_analytics.rs — Real-time risk analytics for portfolio monitoring.
//
// Implements: VaR (3 methods), CVaR/ES, Cornish-Fisher adjustment,
// max drawdown tracker, tail risk decomposition, and stress testing.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use anyhow::{Result, anyhow};

// ─── Normal Distribution Helpers ─────────────────────────────────────────────

/// Inverse CDF of the standard normal (probit function).
/// Abramowitz & Stegun rational approximation — error < 4.5e-4.
pub fn inv_normal_cdf(p: f64) -> f64 {
    assert!((0.0..1.0).contains(&p), "p must be in (0, 1)");
    let p = p.clamp(1e-9, 1.0 - 1e-9);
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
    let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 { -z } else { z }
}

/// CDF of the standard normal Φ(x). Hart (1968) rational approximation.
pub fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530
        + t * (-0.356563782
            + t * (1.781477937
                + t * (-1.821255978
                    + t * 1.330274429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if x >= 0.0 { 1.0 - pdf * poly } else { pdf * poly }
}

/// Inverse CDF of Student's t-distribution (approximate, via Newton iteration).
pub fn inv_t_cdf(p: f64, df: f64) -> f64 {
    // Start from normal approximation, refine with Newton.
    let mut x = inv_normal_cdf(p);
    for _ in 0..10 {
        let cdf_x = t_cdf(x, df);
        let pdf_x = t_pdf(x, df);
        if pdf_x < 1e-14 { break; }
        x -= (cdf_x - p) / pdf_x;
    }
    x
}

fn t_pdf(x: f64, df: f64) -> f64 {
    use std::f64::consts::PI;
    let g_ratio = gamma_ratio(df);
    let base = 1.0 + x * x / df;
    g_ratio / (df * PI).sqrt() * base.powf(-(df + 1.0) / 2.0)
}

fn t_cdf(x: f64, df: f64) -> f64 {
    // Regularized incomplete beta function approximation.
    let t2 = x * x;
    let z = df / (df + t2);
    let ibeta = regularized_incomplete_beta(df / 2.0, 0.5, z);
    if x >= 0.0 { 1.0 - 0.5 * ibeta } else { 0.5 * ibeta }
}

fn gamma_ratio(df: f64) -> f64 {
    // Γ((df+1)/2) / Γ(df/2) approximation.
    let a = (df + 1.0) / 2.0;
    let b = df / 2.0;
    (lgamma(a) - lgamma(b)).exp()
}

fn lgamma(x: f64) -> f64 {
    // Lanczos approximation.
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        for i in 1..=8 { a += C[i] / (x + i as f64); }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    // Continued fraction approximation (Lentz's method, numerical recipes).
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    let lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta_ab).exp() / a;
    front * betacf(a, b, x)
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let eps = 3e-7;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps { break; }
    }
    h
}

// ─── VaR Method ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VaRMethod {
    /// Historical simulation: sort returns, take percentile.
    Historical,
    /// Parametric normal: μ - z_α·σ.
    ParametricNormal,
    /// Parametric Student-t.
    ParametricT { df_int: u32 },
    /// Filtered historical simulation: standardize by GARCH-like vol, then HS.
    FilteredHistorical,
}

// ─── VaREstimator ─────────────────────────────────────────────────────────────

/// Value at Risk estimator supporting multiple methodologies.
///
/// Maintains a rolling window of returns for historical / filtered HS methods.
pub struct VaREstimator {
    returns: VecDeque<f64>,
    window: usize,
    // Welford for parametric methods.
    count: u64,
    mean: f64,
    m2: f64,
    // EWMA volatility for filtered HS.
    ewma_var: f64,
    ewma_alpha: f64,
    standardized: VecDeque<f64>,
}

impl VaREstimator {
    pub fn new(window: usize) -> Self {
        Self {
            returns: VecDeque::with_capacity(window),
            window,
            count: 0,
            mean: 0.0,
            m2: 0.0,
            ewma_var: 0.0,
            ewma_alpha: 0.06, // λ = 0.94 (RiskMetrics)
            standardized: VecDeque::with_capacity(window),
        }
    }

    pub fn update(&mut self, r: f64) {
        // Update Welford.
        self.count += 1;
        let delta = r - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (r - self.mean);

        // Update EWMA variance.
        if self.count == 1 {
            self.ewma_var = r * r;
        } else {
            self.ewma_var = (1.0 - self.ewma_alpha) * self.ewma_var + self.ewma_alpha * r * r;
        }

        // Store standardized return.
        let vol = self.ewma_var.sqrt().max(1e-9);
        let std_r = r / vol;

        if self.returns.len() == self.window {
            self.returns.pop_front();
            self.standardized.pop_front();
        }
        self.returns.push_back(r);
        self.standardized.push_back(std_r);
    }

    pub fn reset(&mut self) {
        self.returns.clear();
        self.standardized.clear();
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
        self.ewma_var = 0.0;
    }

    pub fn variance(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        self.m2 / (self.count - 1) as f64
    }

    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }

    /// Estimate VaR at given confidence level (e.g., 0.95, 0.99).
    ///
    /// Returns the loss threshold: P(loss > VaR) = 1 - confidence.
    /// VaR is expressed as a positive number (loss).
    pub fn var(&self, confidence: f64, method: VaRMethod) -> Result<f64> {
        let alpha = 1.0 - confidence;
        if self.count < 5 {
            return Err(anyhow!("insufficient data for VaR (need ≥5 returns)"));
        }
        match method {
            VaRMethod::Historical => self.historical_var(alpha),
            VaRMethod::ParametricNormal => Ok(self.parametric_normal_var(alpha)),
            VaRMethod::ParametricT { df_int } => Ok(self.parametric_t_var(alpha, df_int as f64)),
            VaRMethod::FilteredHistorical => self.filtered_hs_var(alpha),
        }
    }

    fn historical_var(&self, alpha: f64) -> Result<f64> {
        if self.returns.is_empty() { return Err(anyhow!("no returns")); }
        let mut sorted: Vec<f64> = self.returns.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (alpha * sorted.len() as f64).floor() as usize;
        let idx = idx.min(sorted.len() - 1);
        Ok(-sorted[idx]) // VaR is positive (loss)
    }

    fn parametric_normal_var(&self, alpha: f64) -> f64 {
        let z = inv_normal_cdf(alpha);
        -(self.mean + z * self.std_dev())
    }

    fn parametric_t_var(&self, alpha: f64, df: f64) -> f64 {
        let t = inv_t_cdf(alpha, df);
        let scale = ((df - 2.0) / df).sqrt() * self.std_dev();
        -(self.mean + t * scale)
    }

    fn filtered_hs_var(&self, alpha: f64) -> Result<f64> {
        if self.standardized.is_empty() { return Err(anyhow!("no data")); }
        let mut sorted: Vec<f64> = self.standardized.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (alpha * sorted.len() as f64).floor() as usize;
        let idx = idx.min(sorted.len() - 1);
        let current_vol = self.ewma_var.sqrt();
        Ok(-sorted[idx] * current_vol)
    }
}

// ─── ExpectedShortfall ────────────────────────────────────────────────────────

/// Expected Shortfall (CVaR) at configurable confidence levels.
///
/// ES = E[loss | loss > VaR] = mean of the worst (1-confidence) fraction of returns.
pub struct ExpectedShortfall {
    var_estimator: VaREstimator,
}

impl ExpectedShortfall {
    pub fn new(window: usize) -> Self {
        Self { var_estimator: VaREstimator::new(window) }
    }

    pub fn update(&mut self, r: f64) { self.var_estimator.update(r); }
    pub fn reset(&mut self) { self.var_estimator.reset(); }

    /// CVaR at given confidence level.
    pub fn cvar(&self, confidence: f64, method: VaRMethod) -> Result<f64> {
        let alpha = 1.0 - confidence;
        match method {
            VaRMethod::Historical | VaRMethod::FilteredHistorical => {
                self.historical_cvar(alpha)
            }
            VaRMethod::ParametricNormal => Ok(self.normal_cvar(alpha)),
            VaRMethod::ParametricT { df_int } => Ok(self.t_cvar(alpha, df_int as f64)),
        }
    }

    fn historical_cvar(&self, alpha: f64) -> Result<f64> {
        let returns = &self.var_estimator.returns;
        if returns.is_empty() { return Err(anyhow!("no returns")); }
        let mut sorted: Vec<f64> = returns.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_tail = ((alpha * sorted.len() as f64).floor() as usize).max(1);
        let tail_mean = sorted[..n_tail].iter().sum::<f64>() / n_tail as f64;
        Ok(-tail_mean)
    }

    fn normal_cvar(&self, alpha: f64) -> f64 {
        // ES_normal = μ + σ * φ(z_α) / α
        let z = inv_normal_cdf(alpha);
        let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let mu = self.var_estimator.mean;
        let sigma = self.var_estimator.std_dev();
        -(mu - sigma * phi / alpha)
    }

    fn t_cvar(&self, alpha: f64, df: f64) -> f64 {
        // ES for Student-t (closed form).
        let t_alpha = inv_t_cdf(alpha, df);
        let factor = (1.0 + t_alpha * t_alpha / df).powf(-(df + 1.0) / 2.0) / alpha;
        let scale = ((df - 2.0) / df).sqrt() * self.var_estimator.std_dev();
        let coeff = factor * (df + t_alpha * t_alpha) / (df - 1.0);
        -(self.var_estimator.mean - scale * coeff)
    }
}

// ─── CornishFisher ────────────────────────────────────────────────────────────

/// Cornish-Fisher expansion: adjust VaR/CVaR for skewness and kurtosis.
///
/// Modified z-score: z_cf = z + (z²-1)·s/6 + (z³-3z)·k/24 - (2z³-5z)·s²/36
/// where s = skewness, k = excess kurtosis.
pub struct CornishFisher;

impl CornishFisher {
    /// Adjusted VaR incorporating higher moments.
    pub fn adjusted_var(
        mean: f64,
        std_dev: f64,
        skewness: f64,
        excess_kurtosis: f64,
        confidence: f64,
    ) -> f64 {
        let alpha = 1.0 - confidence;
        let z = inv_normal_cdf(alpha);
        let s = skewness;
        let k = excess_kurtosis;
        let z_cf = z
            + (z * z - 1.0) * s / 6.0
            + (z * z * z - 3.0 * z) * k / 24.0
            - (2.0 * z * z * z - 5.0 * z) * s * s / 36.0;
        -(mean + z_cf * std_dev)
    }

    /// Adjusted CVaR.
    pub fn adjusted_cvar(
        mean: f64,
        std_dev: f64,
        skewness: f64,
        excess_kurtosis: f64,
        confidence: f64,
    ) -> f64 {
        // Approximate CVaR via adjusted quantile.
        let var_cf = Self::adjusted_var(mean, std_dev, skewness, excess_kurtosis, confidence);
        // Add an ES adjustment factor.
        let alpha = 1.0 - confidence;
        let z = inv_normal_cdf(alpha);
        let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let es_normal = -(mean - std_dev * phi / alpha);
        // Scale by ratio of CF-VaR to normal VaR.
        let var_normal = Self::adjusted_var(mean, std_dev, 0.0, 0.0, confidence);
        if var_normal.abs() < 1e-12 { return var_cf; }
        es_normal * (var_cf / var_normal)
    }
}

// ─── MaxDrawdown ──────────────────────────────────────────────────────────────

/// Real-time maximum drawdown tracker.
///
/// Tracks: running peak, current trough, current drawdown (%), max drawdown,
/// and drawdown duration (number of periods below peak).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownState {
    pub peak: f64,
    pub trough: f64,
    pub current_equity: f64,
    pub current_drawdown_pct: f64,
    pub max_drawdown_pct: f64,
    pub max_drawdown_peak: f64,
    pub max_drawdown_trough: f64,
    pub drawdown_start_bar: u64,
    pub current_bar: u64,
    pub drawdown_duration_bars: u64,
    pub max_drawdown_duration_bars: u64,
    /// Whether currently in a drawdown.
    pub in_drawdown: bool,
}

pub struct MaxDrawdown {
    peak: f64,
    trough: f64,
    max_dd: f64,
    max_dd_peak: f64,
    max_dd_trough: f64,
    drawdown_start: u64,
    current_bar: u64,
    max_dd_duration: u64,
    in_drawdown: bool,
}

impl MaxDrawdown {
    pub fn new(initial_equity: f64) -> Self {
        Self {
            peak: initial_equity,
            trough: initial_equity,
            max_dd: 0.0,
            max_dd_peak: initial_equity,
            max_dd_trough: initial_equity,
            drawdown_start: 0,
            current_bar: 0,
            max_dd_duration: 0,
            in_drawdown: false,
        }
    }

    pub fn update(&mut self, equity: f64) {
        self.current_bar += 1;

        if equity > self.peak {
            // New high-water mark: drawdown recovered.
            if self.in_drawdown {
                let duration = self.current_bar - self.drawdown_start;
                if duration > self.max_dd_duration {
                    self.max_dd_duration = duration;
                }
            }
            self.peak = equity;
            self.trough = equity;
            self.in_drawdown = false;
        } else {
            // In drawdown territory.
            if !self.in_drawdown {
                self.in_drawdown = true;
                self.drawdown_start = self.current_bar;
            }
            if equity < self.trough {
                self.trough = equity;
            }
            let dd = (equity - self.peak) / self.peak;
            if dd < self.max_dd {
                self.max_dd = dd;
                self.max_dd_peak = self.peak;
                self.max_dd_trough = self.trough;
            }
        }
    }

    pub fn reset(&mut self, initial_equity: f64) {
        *self = Self::new(initial_equity);
    }

    pub fn current_drawdown_pct(&self) -> f64 {
        if self.peak < 1e-12 { return 0.0; }
        (self.trough - self.peak) / self.peak
    }

    pub fn snapshot(&self) -> DrawdownState {
        let duration = if self.in_drawdown {
            self.current_bar - self.drawdown_start
        } else {
            0
        };
        DrawdownState {
            peak: self.peak,
            trough: self.trough,
            current_equity: self.trough,
            current_drawdown_pct: self.current_drawdown_pct(),
            max_drawdown_pct: self.max_dd,
            max_drawdown_peak: self.max_dd_peak,
            max_drawdown_trough: self.max_dd_trough,
            drawdown_start_bar: self.drawdown_start,
            current_bar: self.current_bar,
            drawdown_duration_bars: duration,
            max_drawdown_duration_bars: self.max_dd_duration,
            in_drawdown: self.in_drawdown,
        }
    }
}

// ─── TailRiskDecomposition ────────────────────────────────────────────────────

/// Decompose portfolio tail risk into individual and systematic components.
///
/// Uses correlation structure to separate:
/// - Idiosyncratic tail (diversifiable): VaR contributions uncorrelated with portfolio
/// - Systematic tail (non-diversifiable): VaR contributions correlated with portfolio
pub struct TailRiskDecomposition {
    asset_returns: Vec<VecDeque<f64>>,
    portfolio_returns: VecDeque<f64>,
    window: usize,
    n_assets: usize,
}

impl TailRiskDecomposition {
    pub fn new(n_assets: usize, window: usize) -> Self {
        Self {
            asset_returns: vec![VecDeque::with_capacity(window); n_assets],
            portfolio_returns: VecDeque::with_capacity(window),
            window,
            n_assets,
        }
    }

    /// Update with an array of asset returns and the portfolio return.
    pub fn update(&mut self, asset_rets: &[f64], portfolio_ret: f64) -> Result<()> {
        if asset_rets.len() != self.n_assets {
            return Err(anyhow!(
                "expected {} asset returns, got {}", self.n_assets, asset_rets.len()
            ));
        }
        for (i, &r) in asset_rets.iter().enumerate() {
            if self.asset_returns[i].len() == self.window {
                self.asset_returns[i].pop_front();
            }
            self.asset_returns[i].push_back(r);
        }
        if self.portfolio_returns.len() == self.window {
            self.portfolio_returns.pop_front();
        }
        self.portfolio_returns.push_back(portfolio_ret);
        Ok(())
    }

    pub fn reset(&mut self) {
        for buf in &mut self.asset_returns { buf.clear(); }
        self.portfolio_returns.clear();
    }

    /// Compute beta of each asset to the portfolio.
    pub fn betas(&self) -> Vec<f64> {
        let n = self.portfolio_returns.len();
        if n < 5 { return vec![0.0; self.n_assets]; }
        let p_mean: f64 = self.portfolio_returns.iter().sum::<f64>() / n as f64;
        let p_var: f64 = self.portfolio_returns.iter()
            .map(|&r| (r - p_mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        if p_var < 1e-14 { return vec![0.0; self.n_assets]; }

        self.asset_returns.iter().map(|asset_buf| {
            let a_mean: f64 = asset_buf.iter().sum::<f64>() / n as f64;
            let cov: f64 = asset_buf.iter().zip(self.portfolio_returns.iter())
                .map(|(&a, &p)| (a - a_mean) * (p - p_mean))
                .sum::<f64>() / (n - 1) as f64;
            cov / p_var
        }).collect()
    }

    /// Per-asset systematic vs idiosyncratic tail risk breakdown.
    /// Returns Vec of (systematic_var, idiosyncratic_var) for each asset.
    pub fn decompose(&self, confidence: f64) -> Vec<TailRiskComponent> {
        let betas = self.betas();
        let n = self.portfolio_returns.len();
        if n < 5 { return vec![TailRiskComponent::zero(); self.n_assets]; }

        // Portfolio VaR (historical).
        let mut p_sorted: Vec<f64> = self.portfolio_returns.iter().cloned().collect();
        p_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = 1.0 - confidence;
        let idx = ((alpha * n as f64).floor() as usize).min(n - 1);
        let p_var_abs = -p_sorted[idx];

        betas.iter().enumerate().map(|(i, &beta)| {
            let systematic = beta * p_var_abs;
            // Asset total VaR.
            let mut a_sorted: Vec<f64> = self.asset_returns[i].iter().cloned().collect();
            a_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let a_var_abs = -a_sorted[idx.min(a_sorted.len().saturating_sub(1))];
            let idio = (a_var_abs - systematic.abs()).max(0.0);
            TailRiskComponent {
                asset_index: i,
                beta,
                systematic_var: systematic.max(0.0),
                idiosyncratic_var: idio,
                total_var: a_var_abs,
            }
        }).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TailRiskComponent {
    pub asset_index: usize,
    pub beta: f64,
    pub systematic_var: f64,
    pub idiosyncratic_var: f64,
    pub total_var: f64,
}

impl TailRiskComponent {
    fn zero() -> Self {
        Self { asset_index: 0, beta: 0.0, systematic_var: 0.0, idiosyncratic_var: 0.0, total_var: 0.0 }
    }
}

// ─── StressTest ───────────────────────────────────────────────────────────────

/// Historical shock scenario definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    pub name: String,
    /// Percentage shock applied to portfolio value.
    pub shock_pct: f64,
    /// Per-asset shocks (override shock_pct if provided).
    pub per_asset_shocks: Vec<(String, f64)>,
    /// Volatility multiplier (sigma scaling).
    pub vol_multiplier: f64,
    /// Correlation shift: add to all pairwise correlations.
    pub correlation_shift: f64,
}

impl StressScenario {
    pub fn new(name: impl Into<String>, shock_pct: f64) -> Self {
        Self {
            name: name.into(),
            shock_pct,
            per_asset_shocks: Vec::new(),
            vol_multiplier: 1.0,
            correlation_shift: 0.0,
        }
    }

    pub fn with_per_asset(mut self, symbol: impl Into<String>, shock: f64) -> Self {
        self.per_asset_shocks.push((symbol.into(), shock));
        self
    }

    pub fn with_vol_multiplier(mut self, mult: f64) -> Self {
        self.vol_multiplier = mult;
        self
    }
}

/// Named historical scenarios.
pub mod scenarios {
    use super::StressScenario;

    pub fn gfc_2008() -> StressScenario {
        StressScenario::new("2008 GFC", -0.50)
            .with_vol_multiplier(4.0)
    }

    pub fn covid_crash_2020() -> StressScenario {
        StressScenario::new("COVID-19 Crash 2020", -0.34)
            .with_vol_multiplier(3.5)
    }

    pub fn rate_hike_2022() -> StressScenario {
        StressScenario::new("2022 Rate Hike Cycle", -0.25)
            .with_vol_multiplier(1.8)
    }

    pub fn volmageddon_2018() -> StressScenario {
        StressScenario::new("Volmageddon 2018", -0.12)
            .with_vol_multiplier(5.0)
    }

    pub fn dot_com_crash() -> StressScenario {
        StressScenario::new("Dot-com Crash 2000-2002", -0.78)
            .with_vol_multiplier(2.0)
    }

    pub fn flash_crash_2010() -> StressScenario {
        StressScenario::new("Flash Crash 2010", -0.10)
            .with_vol_multiplier(10.0)
    }

    pub fn all() -> Vec<StressScenario> {
        vec![
            gfc_2008(), covid_crash_2020(), rate_hike_2022(),
            volmageddon_2018(), dot_com_crash(), flash_crash_2010(),
        ]
    }
}

/// Portfolio position for stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub market_value: f64,
    pub quantity: f64,
    pub price: f64,
}

/// Stress test result for one scenario applied to a portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressResult {
    pub scenario_name: String,
    pub portfolio_value_before: f64,
    pub portfolio_value_after: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub per_position: Vec<PositionStress>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionStress {
    pub symbol: String,
    pub value_before: f64,
    pub value_after: f64,
    pub pnl: f64,
    pub shock_applied: f64,
}

pub struct StressTest;

impl StressTest {
    /// Apply a scenario shock to a portfolio of positions.
    pub fn apply(positions: &[Position], scenario: &StressScenario) -> StressResult {
        let total_before: f64 = positions.iter().map(|p| p.market_value).sum();
        let mut total_after = 0.0;
        let mut per_position = Vec::with_capacity(positions.len());

        for pos in positions {
            // Look up per-asset shock, fallback to global.
            let shock = scenario.per_asset_shocks.iter()
                .find(|(sym, _)| sym == &pos.symbol)
                .map(|(_, s)| *s)
                .unwrap_or(scenario.shock_pct);

            let value_after = pos.market_value * (1.0 + shock);
            let pnl = value_after - pos.market_value;
            total_after += value_after;
            per_position.push(PositionStress {
                symbol: pos.symbol.clone(),
                value_before: pos.market_value,
                value_after,
                pnl,
                shock_applied: shock,
            });
        }

        let pnl = total_after - total_before;
        let pnl_pct = if total_before.abs() > 1e-9 { pnl / total_before } else { 0.0 };

        StressResult {
            scenario_name: scenario.name.clone(),
            portfolio_value_before: total_before,
            portfolio_value_after: total_after,
            pnl,
            pnl_pct,
            per_position,
        }
    }

    /// Run all named historical scenarios.
    pub fn run_all(positions: &[Position]) -> Vec<StressResult> {
        scenarios::all().iter().map(|s| Self::apply(positions, s)).collect()
    }

    /// Find worst-case scenario.
    pub fn worst_case(positions: &[Position]) -> StressResult {
        let results = Self::run_all(positions);
        results.into_iter()
            .min_by(|a, b| a.pnl.partial_cmp(&b.pnl).unwrap())
            .unwrap_or_else(|| Self::apply(positions, &scenarios::gfc_2008()))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inv_normal_cdf_symmetry() {
        let z = inv_normal_cdf(0.95);
        assert!((z - 1.6449).abs() < 0.01, "z_0.95={z}");
        let z99 = inv_normal_cdf(0.99);
        assert!((z99 - 2.3263).abs() < 0.01, "z_0.99={z99}");
    }

    #[test]
    fn test_var_historical() {
        let mut est = VaREstimator::new(252);
        for i in 0..200 {
            let r = (i as f64 % 10.0 - 5.0) * 0.01;
            est.update(r);
        }
        let var95 = est.var(0.95, VaRMethod::Historical).unwrap();
        let var99 = est.var(0.99, VaRMethod::Historical).unwrap();
        assert!(var99 >= var95, "VaR99 should be >= VaR95");
    }

    #[test]
    fn test_var_parametric_normal() {
        let mut est = VaREstimator::new(252);
        // N(0, 0.01): 99% VaR ≈ 2.33 * 0.01 = 0.0233
        for _ in 0..100 { est.update(0.01); }
        for _ in 0..100 { est.update(-0.01); }
        let var = est.var(0.99, VaRMethod::ParametricNormal).unwrap();
        assert!(var > 0.0);
    }

    #[test]
    fn test_cvar_gt_var() {
        let mut es = ExpectedShortfall::new(252);
        for i in 0..200 { es.update((i as f64 % 20.0 - 10.0) * 0.005); }
        let cvar = es.cvar(0.95, VaRMethod::Historical).unwrap();
        let var = es.var_estimator.var(0.95, VaRMethod::Historical).unwrap();
        assert!(cvar >= var, "CVaR {cvar} should be >= VaR {var}");
    }

    #[test]
    fn test_max_drawdown_tracking() {
        let mut dd = MaxDrawdown::new(100_000.0);
        let equities = [100_000.0, 105_000.0, 102_000.0, 95_000.0, 98_000.0, 110_000.0];
        for &e in &equities { dd.update(e); }
        let snap = dd.snapshot();
        assert!((snap.max_drawdown_pct - (-0.0952)).abs() < 0.01);
        assert!(!snap.in_drawdown);
    }

    #[test]
    fn test_cornish_fisher_fat_tails() {
        let normal_var = CornishFisher::adjusted_var(0.0, 0.01, 0.0, 0.0, 0.99);
        let fat_tail_var = CornishFisher::adjusted_var(0.0, 0.01, -0.5, 2.0, 0.99);
        assert!(fat_tail_var > normal_var, "Fat-tail VaR should exceed normal VaR");
    }

    #[test]
    fn test_stress_test() {
        let positions = vec![
            Position { symbol: "SPY".into(), market_value: 50_000.0, quantity: 100.0, price: 500.0 },
            Position { symbol: "QQQ".into(), market_value: 30_000.0, quantity: 80.0, price: 375.0 },
        ];
        let result = StressTest::apply(&positions, &scenarios::gfc_2008());
        assert!(result.pnl < 0.0);
        assert!((result.pnl_pct - (-0.50)).abs() < 0.01);
    }

    #[test]
    fn test_tail_risk_decomposition() {
        let mut trd = TailRiskDecomposition::new(2, 100);
        for i in 0..100 {
            let p = (i as f64 % 10.0 - 5.0) * 0.01;
            trd.update(&[p * 1.2, p * 0.8], p).unwrap();
        }
        let components = trd.decompose(0.95);
        assert_eq!(components.len(), 2);
    }
}
