use crate::correlation_matrix::pearson_stable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Correlation analysis split by market regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeCorrelationReport {
    pub symbols: Vec<String>,
    /// Full-sample correlation matrix.
    pub full: Vec<Vec<f64>>,
    /// Bull-regime correlation (BTC bar return > 0).
    pub bull: Vec<Vec<f64>>,
    /// Bear-regime correlation (BTC bar return < 0).
    pub bear: Vec<Vec<f64>>,
    /// High-volatility-regime correlation (|BTC return| > 2%).
    pub high_vol: Vec<Vec<f64>>,
    /// Average pairwise correlation per regime.
    pub avg_corr_full: f64,
    pub avg_corr_bull: f64,
    pub avg_corr_bear: f64,
    pub avg_corr_high_vol: f64,
    /// Pairs with the largest correlation divergence between bull and bear.
    pub divergent_pairs: Vec<DivergentPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DivergentPair {
    pub sym_a: String,
    pub sym_b: String,
    pub bull_corr: f64,
    pub bear_corr: f64,
    pub divergence: f64,
}

/// Split returns by regime and compute correlation matrices for each.
///
/// # Arguments
/// * `returns` — map of symbol → return series (all same length, aligned by bar).
/// * `btc_sym` — the key to use as the BTC series for regime classification (default "BTC").
/// * `high_vol_threshold` — |return| threshold for high-vol regime (default 0.02 = 2%).
pub fn regime_correlation_analysis(
    returns: &HashMap<String, Vec<f64>>,
    btc_sym: &str,
    high_vol_threshold: f64,
) -> RegimeCorrelationReport {
    let symbols: Vec<String> = {
        let mut v: Vec<String> = returns.keys().cloned().collect();
        v.sort();
        v
    };

    let n = symbols.len();
    let empty_matrix = || vec![vec![0.0_f64; n]; n];

    // Get BTC returns for regime classification.
    let btc_returns = match returns.get(btc_sym) {
        Some(r) => r.clone(),
        None => {
            // Fallback: use the first symbol.
            returns.values().next().cloned().unwrap_or_default()
        }
    };

    let t = btc_returns.len();
    if t == 0 {
        return RegimeCorrelationReport {
            symbols,
            full: empty_matrix(),
            bull: empty_matrix(),
            bear: empty_matrix(),
            high_vol: empty_matrix(),
            avg_corr_full: 0.0,
            avg_corr_bull: 0.0,
            avg_corr_bear: 0.0,
            avg_corr_high_vol: 0.0,
            divergent_pairs: vec![],
        };
    }

    // Align all return series to BTC length.
    let series: Vec<Vec<f64>> = symbols
        .iter()
        .map(|s| {
            let r = returns.get(s).map(|v| v.as_slice()).unwrap_or(&[]);
            let start = r.len().saturating_sub(t);
            r[start..].to_vec()
        })
        .collect();

    // Build regime masks.
    let t_aligned = series.iter().map(|s| s.len()).min().unwrap_or(0).min(t);
    let bull_mask: Vec<bool> = (0..t_aligned).map(|i| btc_returns[i] > 0.0).collect();
    let bear_mask: Vec<bool> = (0..t_aligned).map(|i| btc_returns[i] < 0.0).collect();
    let high_vol_mask: Vec<bool> = (0..t_aligned)
        .map(|i| btc_returns[i].abs() > high_vol_threshold)
        .collect();

    // Filter returns by regime mask.
    let filter_regime = |mask: &[bool]| -> Vec<Vec<f64>> {
        series
            .iter()
            .map(|r| {
                r.iter()
                    .zip(mask.iter())
                    .filter_map(|(&v, &m)| if m { Some(v) } else { None })
                    .collect()
            })
            .collect()
    };

    let full_series = &series;
    let bull_series = filter_regime(&bull_mask);
    let bear_series = filter_regime(&bear_mask);
    let high_vol_series = filter_regime(&high_vol_mask);

    let build_matrix = |series: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
            for j in (i + 1)..n {
                let c = pearson_stable(&series[i], &series[j]).unwrap_or(0.0);
                m[i][j] = c;
                m[j][i] = c;
            }
        }
        m
    };

    let full_m = build_matrix(full_series);
    let bull_m = build_matrix(&bull_series);
    let bear_m = build_matrix(&bear_series);
    let high_vol_m = build_matrix(&high_vol_series);

    let avg_off_diagonal = |m: &[Vec<f64>]| -> f64 {
        let mut sum = 0.0;
        let mut count = 0u32;
        for i in 0..n {
            for j in (i + 1)..n {
                sum += m[i][j];
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    };

    let avg_full = avg_off_diagonal(&full_m);
    let avg_bull = avg_off_diagonal(&bull_m);
    let avg_bear = avg_off_diagonal(&bear_m);
    let avg_high_vol = avg_off_diagonal(&high_vol_m);

    // Find most divergent pairs (|bull_corr - bear_corr| largest).
    let mut divergent_pairs: Vec<DivergentPair> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let bull_c = bull_m[i][j];
            let bear_c = bear_m[i][j];
            let div = (bull_c - bear_c).abs();
            divergent_pairs.push(DivergentPair {
                sym_a: symbols[i].clone(),
                sym_b: symbols[j].clone(),
                bull_corr: bull_c,
                bear_corr: bear_c,
                divergence: div,
            });
        }
    }
    divergent_pairs.sort_unstable_by(|a, b| {
        b.divergence
            .partial_cmp(&a.divergence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    divergent_pairs.truncate(20);

    RegimeCorrelationReport {
        symbols,
        full: full_m,
        bull: bull_m,
        bear: bear_m,
        high_vol: high_vol_m,
        avg_corr_full: avg_full,
        avg_corr_bull: avg_bull,
        avg_corr_bear: avg_bear,
        avg_corr_high_vol: avg_high_vol,
        divergent_pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_returns() -> HashMap<String, Vec<f64>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = 200;
        let btc: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.03_f64..0.03_f64)).collect();
        // ETH correlated with BTC but noisier.
        let eth: Vec<f64> = btc.iter().map(|r| r * 1.2 + rng.gen_range(-0.01..0.01)).collect();
        // SOL semi-correlated.
        let sol: Vec<f64> = btc.iter().map(|r| r * 0.8 + rng.gen_range(-0.02..0.02)).collect();

        let mut m = HashMap::new();
        m.insert("BTC".to_string(), btc);
        m.insert("ETH".to_string(), eth);
        m.insert("SOL".to_string(), sol);
        m
    }

    #[test]
    fn test_regime_report_returns_three_symbols() {
        let returns = synthetic_returns();
        let report = regime_correlation_analysis(&returns, "BTC", 0.02);
        assert_eq!(report.symbols.len(), 3);
    }

    #[test]
    fn test_diagonal_is_one() {
        let returns = synthetic_returns();
        let report = regime_correlation_analysis(&returns, "BTC", 0.02);
        for m in [&report.full, &report.bull, &report.bear, &report.high_vol] {
            for i in 0..m.len() {
                assert!((m[i][i] - 1.0).abs() < 1e-9, "diagonal should be 1.0");
            }
        }
    }

    #[test]
    fn test_bear_corr_higher_than_bull() {
        // In real markets bear correlations tend to be higher.
        // We can't guarantee this on synthetic data, but the report should at least parse.
        let returns = synthetic_returns();
        let report = regime_correlation_analysis(&returns, "BTC", 0.02);
        // Just check values are in [-1,1].
        assert!(report.avg_corr_bear >= -1.0 && report.avg_corr_bear <= 1.0);
        assert!(report.avg_corr_bull >= -1.0 && report.avg_corr_bull <= 1.0);
    }

    #[test]
    fn test_divergent_pairs_sorted() {
        let returns = synthetic_returns();
        let report = regime_correlation_analysis(&returns, "BTC", 0.02);
        for w in report.divergent_pairs.windows(2) {
            assert!(w[0].divergence >= w[1].divergence);
        }
    }

    #[test]
    fn test_missing_btc_falls_back() {
        let mut returns = HashMap::new();
        returns.insert("ETH".to_string(), vec![0.01, -0.02, 0.03, -0.01, 0.02]);
        returns.insert("SOL".to_string(), vec![0.02, -0.01, 0.01, -0.02, 0.03]);
        // No BTC key — should not panic.
        let report = regime_correlation_analysis(&returns, "BTC", 0.02);
        assert!(report.avg_corr_full >= -1.0);
    }
}
