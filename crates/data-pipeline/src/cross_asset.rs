/// Cross-asset feature computation for crypto assets.
///
/// Computes BTC-relative excess return, rolling OLS beta to BTC, a BTC
/// dominance proxy, and an optional correlation to an SPX proxy.

use serde::{Deserialize, Serialize};

// ── CrossAssetFV ──────────────────────────────────────────────────────────────

/// Cross-asset feature vector for a single bar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAssetFV {
    /// Excess return vs BTC: symbol_ret - btc_ret.
    pub excess_return_vs_btc: f64,
    /// Rolling 30-bar OLS beta to BTC.
    pub rolling_beta_btc: f64,
    /// Correlation to BTC over the rolling window.
    pub correlation_to_btc: f64,
    /// Estimated BTC dominance proxy: BTC vol / total vol.
    pub btc_dominance_proxy: f64,
    /// Signed excess return: excess / |btc_ret| (relative performance).
    pub relative_performance: f64,
    /// Correlation to SPX proxy (if available), else 0.
    pub corr_to_spx: f64,
    /// Risk-on/risk-off score: positive = risk-on (high beta, high excess return).
    pub risk_on_score: f64,
}

// ── CrossAssetHistory ─────────────────────────────────────────────────────────

/// Rolling history of returns for a single asset and BTC, with an optional SPX
/// proxy series.
///
/// Maintains ring buffers for symbol returns, BTC returns, and (optionally)
/// SPX proxy returns.
pub struct CrossAssetHistory {
    symbol: String,
    /// Ring buffers.
    sym_rets: Vec<f64>,
    btc_rets: Vec<f64>,
    btc_vols: Vec<f64>,  // |btc return| (vol proxy)
    sym_vols: Vec<f64>,  // |symbol return| (vol proxy)
    spx_rets: Vec<f64>,
    head: usize,
    count: usize,
    capacity: usize,
    /// Whether an SPX proxy is being tracked.
    has_spx: bool,
}

impl CrossAssetHistory {
    /// Create history with the given window (default: 30 bars).
    pub fn new(symbol: impl Into<String>, window: usize) -> Self {
        let cap = window.max(3);
        CrossAssetHistory {
            symbol: symbol.into(),
            sym_rets: vec![0.0; cap],
            btc_rets: vec![0.0; cap],
            btc_vols: vec![0.0; cap],
            sym_vols: vec![0.0; cap],
            spx_rets: vec![0.0; cap],
            head: 0,
            count: 0,
            capacity: cap,
            has_spx: false,
        }
    }

    /// Push a new observation (symbol return, BTC return).
    pub fn push(&mut self, sym_ret: f64, btc_ret: f64) {
        self.sym_rets[self.head] = sym_ret;
        self.btc_rets[self.head] = btc_ret;
        self.btc_vols[self.head] = btc_ret.abs();
        self.sym_vols[self.head] = sym_ret.abs();
        self.spx_rets[self.head] = 0.0; // no SPX by default
        self.advance();
    }

    /// Push with an SPX proxy return.
    pub fn push_with_spx(&mut self, sym_ret: f64, btc_ret: f64, spx_ret: f64) {
        self.sym_rets[self.head] = sym_ret;
        self.btc_rets[self.head] = btc_ret;
        self.btc_vols[self.head] = btc_ret.abs();
        self.sym_vols[self.head] = sym_ret.abs();
        self.spx_rets[self.head] = spx_ret;
        self.has_spx = true;
        self.advance();
    }

    fn advance(&mut self) {
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Number of observations stored.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Retrieve the last `n` symbol returns in chronological order.
    fn last_sym_rets(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.count);
        (0..n).rev().map(|i| {
            let slot = (self.head + self.capacity - 1 - i) % self.capacity;
            self.sym_rets[slot]
        }).collect()
    }

    /// Retrieve the last `n` BTC returns in chronological order.
    fn last_btc_rets(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.count);
        (0..n).rev().map(|i| {
            let slot = (self.head + self.capacity - 1 - i) % self.capacity;
            self.btc_rets[slot]
        }).collect()
    }

    fn last_spx_rets(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.count);
        (0..n).rev().map(|i| {
            let slot = (self.head + self.capacity - 1 - i) % self.capacity;
            self.spx_rets[slot]
        }).collect()
    }

    /// Rolling BTC dominance proxy: fraction of total (BTC + symbol) volume
    /// attributable to BTC over the last `window` bars.
    pub fn btc_dominance_proxy(&self, window: usize) -> f64 {
        let n = window.min(self.count);
        if n == 0 {
            return 0.5;
        }
        let mut btc_sum = 0.0_f64;
        let mut sym_sum = 0.0_f64;
        for i in 0..n {
            let slot = (self.head + self.capacity - 1 - i) % self.capacity;
            btc_sum += self.btc_vols[slot];
            sym_sum += self.sym_vols[slot];
        }
        let total = btc_sum + sym_sum;
        if total < 1e-15 { 0.5 } else { btc_sum / total }
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

// ── CrossAssetFeatures ────────────────────────────────────────────────────────

/// Computes cross-asset features for one bar.
pub struct CrossAssetFeatures {
    /// Rolling window for beta and correlation (bars).
    window: usize,
}

impl CrossAssetFeatures {
    pub fn new(window: usize) -> Self {
        CrossAssetFeatures { window: window.max(3) }
    }

    pub fn default_window() -> Self {
        Self::new(30)
    }

    // ── compute ───────────────────────────────────────────────────────────

    /// Compute cross-asset features for the current bar.
    ///
    /// `symbol_ret` and `btc_ret` are the current bar's log returns.
    /// `history` contains all PRIOR observations (push the current bar
    /// AFTER calling this if you want it in future computations).
    pub fn compute(
        &self,
        symbol_ret: f64,
        btc_ret: f64,
        history: &CrossAssetHistory,
    ) -> CrossAssetFV {
        let n = self.window.min(history.count());

        let sym_rets = history.last_sym_rets(n);
        let btc_rets = history.last_btc_rets(n);

        // Excess return vs BTC.
        let excess = symbol_ret - btc_ret;

        // Rolling OLS beta.
        let beta = ols_beta(&btc_rets, &sym_rets).unwrap_or(1.0);

        // Rolling Pearson correlation to BTC.
        let corr = pearson(&btc_rets, &sym_rets).unwrap_or(0.0);

        // BTC dominance proxy.
        let dominance = history.btc_dominance_proxy(self.window);

        // Relative performance: excess / |btc_ret| (capped).
        let rel_perf = if btc_ret.abs() > 1e-8 {
            (excess / btc_ret.abs()).clamp(-10.0, 10.0)
        } else {
            0.0
        };

        // SPX correlation.
        let corr_spx = if history.has_spx && n >= 3 {
            let spx = history.last_spx_rets(n);
            pearson(&spx, &sym_rets).unwrap_or(0.0)
        } else {
            0.0
        };

        // Risk-on score: combination of beta and excess return.
        // High beta + positive excess = strong risk-on.
        let risk_on = (beta * 0.5 + corr * 0.3 + excess.signum() * 0.2).clamp(-1.0, 1.0);

        CrossAssetFV {
            excess_return_vs_btc: excess,
            rolling_beta_btc: beta,
            correlation_to_btc: corr,
            btc_dominance_proxy: dominance,
            relative_performance: rel_perf,
            corr_to_spx: corr_spx,
            risk_on_score: risk_on,
        }
    }

    // ── Universe sweep ────────────────────────────────────────────────────

    /// Compute cross-asset features for every symbol in `histories`.
    ///
    /// `current_rets`: symbol -> current bar log return.
    /// `btc_ret`: BTC's current bar log return.
    pub fn compute_universe(
        &self,
        current_rets: &std::collections::HashMap<String, f64>,
        btc_ret: f64,
        histories: &std::collections::HashMap<String, CrossAssetHistory>,
    ) -> std::collections::HashMap<String, CrossAssetFV> {
        current_rets
            .iter()
            .filter_map(|(sym, &ret)| {
                let history = histories.get(sym)?;
                let fv = self.compute(ret, btc_ret, history);
                Some((sym.clone(), fv))
            })
            .collect()
    }
}

// ── Beta estimators ───────────────────────────────────────────────────────────

/// OLS beta of `y` regressed on `x` (no intercept in the market model form):
///   beta = cov(x, y) / var(x)
pub fn ols_beta(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return None;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        var_x += dx * dx;
    }
    if var_x < 1e-14 {
        return None;
    }
    Some(cov / var_x)
}

/// Rolling beta over the last `window` observations using EWMA weights.
///
/// More recent observations get higher weight (lambda = 0.97).
pub fn ewma_beta(x: &[f64], y: &[f64], lambda: f64) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return None;
    }
    let inv_lam = 1.0 - lambda;
    let mut wmx = x[0];
    let mut wmy = y[0];
    let mut wcov = 0.0_f64;
    let mut wvar = 0.0_f64;

    for i in 1..n {
        let dx = x[i] - wmx;
        let dy = y[i] - wmy;
        wmx = lambda * wmx + inv_lam * x[i];
        wmy = lambda * wmy + inv_lam * y[i];
        wcov = lambda * wcov + inv_lam * dx * dy;
        wvar = lambda * wvar + inv_lam * dx * dx;
    }

    if wvar < 1e-14 {
        return None;
    }
    Some(wcov / wvar)
}

// ── Pearson helper ────────────────────────────────────────────────────────────

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return None;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-14 {
        return None;
    }
    Some((num / denom).clamp(-1.0, 1.0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn warm_history(n: usize, sym_scale: f64) -> CrossAssetHistory {
        let mut h = CrossAssetHistory::new("ETH", 30);
        let btc: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
        for i in 0..n {
            h.push(btc[i] * sym_scale, btc[i]);
        }
        h
    }

    #[test]
    fn test_excess_return_positive() {
        let h = warm_history(30, 1.5);
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.03, 0.02, &h);
        assert!((fv.excess_return_vs_btc - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_beta_greater_than_one_for_high_beta_asset() {
        // ETH returns = 1.5 * BTC returns => beta ~1.5.
        let h = warm_history(30, 1.5);
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.03, 0.02, &h);
        assert!(fv.rolling_beta_btc > 1.0, "expected beta > 1.0, got {}", fv.rolling_beta_btc);
    }

    #[test]
    fn test_beta_less_than_one_for_low_beta_asset() {
        let h = warm_history(30, 0.5);
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.01, 0.02, &h);
        assert!(fv.rolling_beta_btc < 1.0, "expected beta < 1.0, got {}", fv.rolling_beta_btc);
    }

    #[test]
    fn test_correlation_range() {
        let h = warm_history(30, 1.2);
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.02, 0.015, &h);
        assert!(fv.correlation_to_btc >= -1.0 && fv.correlation_to_btc <= 1.0);
    }

    #[test]
    fn test_btc_dominance_range() {
        let h = warm_history(30, 1.0);
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.01, 0.01, &h);
        assert!(fv.btc_dominance_proxy >= 0.0 && fv.btc_dominance_proxy <= 1.0);
    }

    #[test]
    fn test_spx_correlation_without_spx_is_zero() {
        let h = warm_history(30, 1.0);
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.01, 0.01, &h);
        assert_eq!(fv.corr_to_spx, 0.0);
    }

    #[test]
    fn test_spx_correlation_with_spx() {
        let mut h = CrossAssetHistory::new("ETH", 30);
        let base: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin() * 0.02).collect();
        for i in 0..30 {
            h.push_with_spx(base[i], base[i], base[i]); // all identical
        }
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.01, 0.01, &h);
        // SPX = sym series => correlation = 1.
        assert!(fv.corr_to_spx > 0.5, "expected high SPX corr, got {}", fv.corr_to_spx);
    }

    #[test]
    fn test_ols_beta_exact() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
        let beta = ols_beta(&x, &y).unwrap();
        assert!((beta - 2.0).abs() < 1e-9, "expected beta=2, got {}", beta);
    }

    #[test]
    fn test_ols_beta_zero_variance_returns_none() {
        let x = vec![1.0; 20];
        let y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!(ols_beta(&x, &y).is_none());
    }

    #[test]
    fn test_ewma_beta_positive() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64 * 0.2).sin()).collect();
        let y: Vec<f64> = x.iter().map(|v| 1.5 * v).collect();
        let beta = ewma_beta(&x, &y, 0.94).unwrap();
        assert!(beta > 0.0, "expected positive EWMA beta, got {}", beta);
    }

    #[test]
    fn test_risk_on_score_range() {
        let h = warm_history(30, 1.0);
        let ca = CrossAssetFeatures::default_window();
        let fv = ca.compute(0.02, 0.01, &h);
        assert!(fv.risk_on_score >= -1.0 && fv.risk_on_score <= 1.0);
    }

    #[test]
    fn test_compute_universe() {
        use std::collections::HashMap;
        let ca = CrossAssetFeatures::default_window();
        let mut histories = HashMap::new();
        histories.insert("ETH".to_string(), warm_history(30, 1.2));
        histories.insert("SOL".to_string(), warm_history(30, 0.8));
        let mut current_rets = HashMap::new();
        current_rets.insert("ETH".to_string(), 0.02);
        current_rets.insert("SOL".to_string(), 0.015);
        let result = ca.compute_universe(&current_rets, 0.01, &histories);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("ETH"));
        assert!(result.contains_key("SOL"));
    }
}
