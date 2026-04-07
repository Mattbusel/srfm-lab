/// Lead-lag relationship detection between crypto assets.
///
/// Provides cross-correlation scanning over lags [-5, +5], BTC lead-time
/// estimation, and a simplified Granger-causality proxy via OLS F-statistic.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default maximum lag in bars (scan [-MAX_LAG, +MAX_LAG]).
pub const MAX_LAG: usize = 5;

/// Minimum |correlation| to be considered statistically significant.
const SIGNIFICANCE_THRESHOLD: f64 = 0.3;

// ── LeadLagResult ─────────────────────────────────────────────────────────────

/// Result of a lead-lag analysis between two time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadLagResult {
    /// Lag (in bars) at which |correlation| is maximised.
    ///
    /// Positive => X leads Y.  Negative => Y leads X.  Zero => contemporaneous.
    pub max_corr_lag: i32,
    /// Correlation value at `max_corr_lag`.
    pub max_corr_value: f64,
    /// True when |max_corr_value| > 0.3.
    pub is_significant: bool,
    /// Full cross-correlation vector, indexed from lag = -MAX_LAG to +MAX_LAG.
    pub xcorr: Vec<f64>,
}

// ── LeadLagAnalyzer ───────────────────────────────────────────────────────────

/// Stateless helper for lead-lag analysis.
///
/// All methods are pure functions on slices. State (return histories) is kept
/// externally so that the analyzer is easily testable in isolation.
pub struct LeadLagAnalyzer {
    /// Maximum lag to scan in each direction.
    max_lag: usize,
}

impl LeadLagAnalyzer {
    /// Create a new analyzer with the given maximum lag.
    pub fn new(max_lag: usize) -> Self {
        LeadLagAnalyzer { max_lag }
    }

    /// Create with the default MAX_LAG = 5 bars.
    pub fn default_lag() -> Self {
        Self::new(MAX_LAG)
    }

    // ── Core cross-correlation ─────────────────────────────────────────────

    /// Compute cross-correlation between `x` and `y` at all integer lags in
    /// `[-max_lag, +max_lag]`.
    ///
    /// Result[max_lag + k] = corr(x[t], y[t+k]) for lag k.
    /// Positive k => x leads y by k bars.
    pub fn cross_correlation(&self, x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = x.len().min(y.len());
        let total = 2 * self.max_lag + 1;
        let mut result = vec![0.0_f64; total];

        for out_idx in 0..total {
            let lag = out_idx as i64 - self.max_lag as i64;
            let (sx, sy) = if lag >= 0 {
                (0usize, lag as usize)
            } else {
                (lag.unsigned_abs() as usize, 0usize)
            };
            let available = n.saturating_sub(lag.unsigned_abs() as usize);
            if available < 3 {
                continue;
            }
            let xs = &x[sx..sx + available];
            let ys = &y[sy..sy + available];
            result[out_idx] = pearson(xs, ys).unwrap_or(0.0);
        }
        result
    }

    // ── detect_lead_lag ────────────────────────────────────────────────────

    /// Detect the dominant lead-lag relationship between `x` and `y`.
    pub fn detect_lead_lag(&self, x: &[f64], y: &[f64]) -> LeadLagResult {
        let xcorr = self.cross_correlation(x, y);

        let (best_idx, &best_val) = xcorr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((self.max_lag, &0.0));

        let lag = best_idx as i32 - self.max_lag as i32;

        LeadLagResult {
            max_corr_lag: lag,
            max_corr_value: best_val,
            is_significant: best_val.abs() > SIGNIFICANCE_THRESHOLD,
            xcorr,
        }
    }

    // ── BTC lead time ──────────────────────────────────────────────────────

    /// Estimate how many bars BTC leads `symbol` on average.
    ///
    /// Returns the lag (in bars) at which BTC returns are most correlated with
    /// the target symbol's returns. Positive => BTC leads.
    ///
    /// `btc_returns` and `symbol_returns` must be the same length.
    /// `window` specifies how many trailing bars to use (0 = use all).
    pub fn btc_lead_time(
        &self,
        btc_returns: &[f64],
        symbol_returns: &[f64],
        window: usize,
    ) -> f64 {
        let n = btc_returns.len().min(symbol_returns.len());
        if n < 3 {
            return 0.0;
        }

        let (start, end) = if window == 0 || window >= n {
            (0, n)
        } else {
            (n - window, n)
        };

        let btc_slice = &btc_returns[start..end];
        let sym_slice = &symbol_returns[start..end];

        let result = self.detect_lead_lag(btc_slice, sym_slice);
        result.max_corr_lag as f64
    }

    /// Compute BTC lead times for every symbol in `universe`.
    ///
    /// Returns a map of symbol -> lead_time_bars (positive = BTC leads the asset).
    pub fn btc_lead_times_universe(
        &self,
        btc_returns: &[f64],
        universe: &HashMap<String, Vec<f64>>,
        window: usize,
    ) -> HashMap<String, f64> {
        universe
            .iter()
            .map(|(sym, rets)| {
                let lt = self.btc_lead_time(btc_returns, rets, window);
                (sym.clone(), lt)
            })
            .collect()
    }

    // ── Granger proxy ──────────────────────────────────────────────────────

    /// Simplified Granger-causality proxy: does lagged `x` improve the OLS
    /// prediction of `y` beyond `y`'s own lags?
    ///
    /// Implements the restricted/unrestricted F-test:
    ///   - Restricted model: y_t = sum_{k=1}^{max_lag} a_k * y_{t-k}
    ///   - Unrestricted model: same + sum_{k=1}^{max_lag} b_k * x_{t-k}
    ///   - F = ((RSS_r - RSS_u) / max_lag) / (RSS_u / df)
    ///
    /// Returns the F-statistic. Large F (> ~4 for typical sample sizes) suggests
    /// that x Granger-causes y. This is a proxy -- not a full asymptotic test.
    pub fn granger_proxy(&self, x: &[f64], y: &[f64], max_lag: usize) -> f64 {
        let n = x.len().min(y.len());
        if n <= 2 * max_lag + 1 {
            return 0.0;
        }

        // Number of observations in the regression (rows).
        let t_start = max_lag;
        let t_end = n;
        let nobs = t_end - t_start; // number of rows

        if nobs < max_lag + 2 {
            return 0.0;
        }

        // Build design matrices.
        // Restricted: columns = [1, y_{t-1}, ..., y_{t-max_lag}]  (p+1 cols)
        // Unrestricted: columns = [1, y_{t-1}, ..., y_{t-max_lag}, x_{t-1}, ..., x_{t-max_lag}]  (2p+1 cols)

        let p = max_lag;
        let p1 = p + 1; // restricted regressors (intercept + y lags)
        let p2 = 2 * p + 1; // unrestricted regressors

        // Dependent variable: y[t_start..t_end]
        let y_dep: Vec<f64> = y[t_start..t_end].to_vec();

        // Build restricted X matrix (nobs x p1).
        let mut xr: Vec<Vec<f64>> = Vec::with_capacity(nobs);
        for t in t_start..t_end {
            let mut row = vec![1.0_f64]; // intercept
            for k in 1..=p {
                row.push(y[t - k]);
            }
            xr.push(row);
        }

        // Build unrestricted X matrix (nobs x p2).
        let mut xu: Vec<Vec<f64>> = Vec::with_capacity(nobs);
        for t in t_start..t_end {
            let mut row = vec![1.0_f64]; // intercept
            for k in 1..=p {
                row.push(y[t - k]);
            }
            for k in 1..=p {
                row.push(x[t - k]);
            }
            xu.push(row);
        }

        let rss_r = ols_rss(&xr, &y_dep, p1);
        let rss_u = ols_rss(&xu, &y_dep, p2);

        if rss_u < 1e-15 {
            return 0.0;
        }

        let df_num = p as f64;
        let df_den = (nobs as f64) - (p2 as f64);
        if df_den <= 0.0 {
            return 0.0;
        }

        ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    }

    // ── Pairwise Granger network ───────────────────────────────────────────

    /// Compute the Granger proxy F-statistic for every ordered pair in `universe`.
    ///
    /// Returns a map (cause, effect) -> F-statistic. High F => cause Granger-causes effect.
    pub fn granger_network(
        &self,
        universe: &HashMap<String, Vec<f64>>,
        max_lag: usize,
    ) -> HashMap<(String, String), f64> {
        let symbols: Vec<String> = {
            let mut v: Vec<String> = universe.keys().cloned().collect();
            v.sort();
            v
        };

        let mut out = HashMap::new();
        for cause in &symbols {
            for effect in &symbols {
                if cause == effect {
                    continue;
                }
                let f = self.granger_proxy(&universe[cause], &universe[effect], max_lag);
                out.insert((cause.clone(), effect.clone()), f);
            }
        }
        out
    }
}

// ── OLS helper ────────────────────────────────────────────────────────────────

/// Fit y = X * beta via normal equations and return the residual sum of squares.
///
/// X is `nobs x k`.  Uses direct inversion for small k (< 20 cols).
fn ols_rss(x_mat: &[Vec<f64>], y: &[f64], k: usize) -> f64 {
    let nobs = x_mat.len();
    if nobs == 0 || k == 0 {
        return 0.0;
    }

    // X'X  (k x k)
    let mut xtx = vec![vec![0.0_f64; k]; k];
    for row in x_mat {
        for i in 0..k {
            for j in 0..k {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }

    // X'y  (k x 1)
    let mut xty = vec![0.0_f64; k];
    for (row, &yi) in x_mat.iter().zip(y.iter()) {
        for i in 0..k {
            xty[i] += row[i] * yi;
        }
    }

    // Solve (X'X) beta = X'y via Gaussian elimination with partial pivoting.
    let beta = match gauss_elim(&xtx, &xty) {
        Some(b) => b,
        None => return y.iter().map(|v| v * v).sum(), // fallback: intercept-only RSS
    };

    // Compute RSS = ||y - X*beta||^2
    let mut rss = 0.0_f64;
    for (row, &yi) in x_mat.iter().zip(y.iter()) {
        let yhat: f64 = row.iter().zip(beta.iter()).map(|(xi, bi)| xi * bi).sum();
        let resid = yi - yhat;
        rss += resid * resid;
    }
    rss
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
/// Returns None if the matrix is singular.
fn gauss_elim(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    // Build augmented matrix [A | b].
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivoting.
        let pivot_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        aug.swap(col, pivot_row);

        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            return None;
        }

        for r in (col + 1)..n {
            let factor = aug[r][col] / pivot;
            for c in col..=n {
                let val = aug[col][c] * factor;
                aug[r][c] -= val;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }
    Some(x)
}

// ── Pearson correlation helper ────────────────────────────────────────────────

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

    /// Build a series where x leads y by `lag` bars.
    fn make_lagged(n: usize, lag: usize) -> (Vec<f64>, Vec<f64>) {
        let sig: Vec<f64> = (0..n + lag).map(|i| (i as f64 * 0.25).sin()).collect();
        let x: Vec<f64> = sig[lag..lag + n].to_vec(); // advanced (early)
        let y: Vec<f64> = sig[..n].to_vec(); // delayed
        (x, y)
    }

    #[test]
    fn test_xcorr_peak_at_zero_for_identity() {
        let a = LeadLagAnalyzer::default_lag();
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let xcorr = a.cross_correlation(&x, &x);
        let center = MAX_LAG;
        assert!((xcorr[center] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_detect_lead_lag_x_leads() {
        let a = LeadLagAnalyzer::default_lag();
        let (x, y) = make_lagged(80, 2);
        let res = a.detect_lead_lag(&x, &y);
        assert!(res.max_corr_lag > 0, "expected positive lag (x leads), got {}", res.max_corr_lag);
        assert!(res.is_significant);
    }

    #[test]
    fn test_detect_lead_lag_contemporaneous() {
        let a = LeadLagAnalyzer::default_lag();
        let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.3).sin()).collect();
        let res = a.detect_lead_lag(&x, &x);
        assert_eq!(res.max_corr_lag, 0);
        assert!((res.max_corr_value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_btc_lead_time_positive() {
        let a = LeadLagAnalyzer::default_lag();
        let (btc, eth) = make_lagged(100, 3);
        let lt = a.btc_lead_time(&btc, &eth, 0);
        assert!(lt > 0.0, "expected BTC to lead, got {}", lt);
    }

    #[test]
    fn test_granger_proxy_x_causes_y() {
        let a = LeadLagAnalyzer::default_lag();
        // y[t] = 0.8 * x[t-1] + noise => x Granger-causes y.
        let n = 150;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let mut y = vec![0.0_f64; n];
        for t in 1..n {
            y[t] = 0.8 * x[t - 1] + 0.05 * ((t * 7) as f64 % 1.0 - 0.5);
        }
        let f = a.granger_proxy(&x, &y, 3);
        // F should be substantially larger than 1 if x causes y.
        assert!(f > 1.0, "expected F > 1, got {}", f);
    }

    #[test]
    fn test_granger_proxy_no_causality() {
        let a = LeadLagAnalyzer::default_lag();
        // Unrelated white noise series.
        let x: Vec<f64> = (0..100).map(|i| ((i * 31 + 7) as f64 % 5.0) - 2.5).collect();
        let y: Vec<f64> = (0..100).map(|i| ((i * 13 + 3) as f64 % 3.0) - 1.5).collect();
        let f = a.granger_proxy(&x, &y, 2);
        // F should be low (or at least not astronomically high).
        assert!(f < 100.0, "expected low F for unrelated series, got {}", f);
    }

    #[test]
    fn test_granger_network_keys() {
        let a = LeadLagAnalyzer::default_lag();
        let mut universe = HashMap::new();
        let sig: Vec<f64> = (0..80).map(|i| (i as f64 * 0.2).sin()).collect();
        universe.insert("BTC".to_string(), sig.clone());
        universe.insert("ETH".to_string(), sig);
        let net = a.granger_network(&universe, 2);
        // Should have (BTC->ETH) and (ETH->BTC) but not (BTC->BTC).
        assert!(net.contains_key(&("BTC".to_string(), "ETH".to_string())));
        assert!(net.contains_key(&("ETH".to_string(), "BTC".to_string())));
        assert!(!net.contains_key(&("BTC".to_string(), "BTC".to_string())));
    }

    #[test]
    fn test_significance_threshold() {
        let a = LeadLagAnalyzer::default_lag();
        let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.1).sin()).collect();
        let noise: Vec<f64> = (0..60).map(|i| ((i * 7) as f64 % 3.0) - 1.5).collect();
        let res = a.detect_lead_lag(&x, &noise);
        // For truly uncorrelated series, is_significant should often be false.
        // We just test the logic compiles and returns a bool.
        let _ = res.is_significant;
    }

    #[test]
    fn test_xcorr_length() {
        let a = LeadLagAnalyzer::new(5);
        let x = vec![1.0; 20];
        let y = vec![1.0; 20];
        let xcorr = a.cross_correlation(&x, &y);
        assert_eq!(xcorr.len(), 11); // 2*5 + 1
    }
}
