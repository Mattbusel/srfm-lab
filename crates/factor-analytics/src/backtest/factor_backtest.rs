//! Factor backtest: quintile/decile portfolio backtest.
//!
//! Implements monthly rebalancing, computes IC, ICIR, turnover,
//! and annualized return per quintile/decile.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};
use crate::decay::ic_decay::compute_ic;
use crate::backtest::stats::compute_ic_statistics;

/// Backtest configuration.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Number of quantile buckets (5 = quintile, 10 = decile)
    pub n_buckets: usize,
    /// Rebalancing frequency in trading days (21 = monthly)
    pub rebalance_freq: usize,
    /// Number of trading days per year for annualization
    pub annualization: f64,
    /// Transaction cost per unit of turnover (one-way)
    pub transaction_cost: f64,
    /// Whether to compute long-short (top bucket - bottom bucket) portfolio
    pub long_short: bool,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            n_buckets: 5,
            rebalance_freq: 21,
            annualization: 252.0,
            transaction_cost: 0.001,
            long_short: true,
        }
    }
}

/// Per-bucket backtest statistics.
#[derive(Debug, Clone)]
pub struct BucketStats {
    /// Bucket index (0 = lowest factor score, n_buckets-1 = highest)
    pub bucket: usize,
    /// Cumulative return (gross)
    pub cumulative_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Annualized volatility
    pub annualized_vol: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Average number of stocks per rebalance
    pub avg_n_stocks: f64,
    /// Average monthly turnover
    pub avg_turnover: f64,
}

/// Full factor backtest results.
#[derive(Debug, Clone)]
pub struct FactorBacktestResult {
    /// Per-bucket statistics
    pub bucket_stats: Vec<BucketStats>,
    /// Long-short portfolio stats (top - bottom bucket)
    pub long_short_stats: Option<BucketStats>,
    /// IC time series (computed at each rebalance date)
    pub ic_series: Vec<f64>,
    /// IC statistics
    pub mean_ic: f64,
    pub icir: f64,
    pub t_stat_ic: f64,
    /// Long-short return spread (annualized top - bottom)
    pub return_spread: f64,
    /// Monotonicity: fraction of buckets where return increases from bucket i to i+1
    pub monotonicity: f64,
    /// Per-bucket return time series
    pub bucket_return_series: Vec<Vec<f64>>,
}

/// Assign assets to quantile buckets based on factor scores.
///
/// Returns a Vec of bucket assignments (0 to n_buckets-1) for each asset.
/// NaN factor scores are assigned to bucket `n_buckets` (excluded from analysis).
pub fn assign_buckets(factor_scores: &[f64], n_buckets: usize) -> Vec<Option<usize>> {
    let n = factor_scores.len();
    let mut sorted_valid: Vec<(usize, f64)> = factor_scores
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
        .collect();

    sorted_valid.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let n_valid = sorted_valid.len();

    let mut assignments = vec![None; n];

    for (rank, &(orig_idx, _)) in sorted_valid.iter().enumerate() {
        let bucket = (rank * n_buckets / n_valid).min(n_buckets - 1);
        assignments[orig_idx] = Some(bucket);
    }

    assignments
}

/// Compute equal-weight return for a set of asset indices.
fn equal_weight_return(returns: &[f64], indices: &[usize]) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let valid: Vec<f64> = indices
        .iter()
        .filter_map(|&i| if returns[i].is_finite() { Some(returns[i]) } else { None })
        .collect();
    if valid.is_empty() {
        return 0.0;
    }
    valid.iter().sum::<f64>() / valid.len() as f64
}

/// Compute turnover between two sets of holdings.
///
/// Turnover = fraction of portfolio that changes.
/// For equal-weight portfolios, turnover = 1 - |overlap| / max(|prev|, |curr|).
fn compute_turnover(prev_indices: &[usize], curr_indices: &[usize]) -> f64 {
    if prev_indices.is_empty() {
        return 1.0;
    }
    let prev_set: std::collections::HashSet<usize> = prev_indices.iter().copied().collect();
    let curr_set: std::collections::HashSet<usize> = curr_indices.iter().copied().collect();
    let overlap = prev_set.intersection(&curr_set).count();
    let max_size = prev_set.len().max(curr_set.len());
    1.0 - (overlap as f64 / max_size as f64)
}

/// Compute annualized return from a series of period returns.
fn annualized_return(period_returns: &[f64], periods_per_year: f64) -> f64 {
    if period_returns.is_empty() {
        return f64::NAN;
    }
    // Compound the period returns
    let total_return: f64 = period_returns.iter().map(|&r| 1.0 + r).product();
    let n_years = period_returns.len() as f64 / periods_per_year;
    total_return.powf(1.0 / n_years) - 1.0
}

/// Compute annualized volatility from period returns.
fn annualized_vol(period_returns: &[f64], periods_per_year: f64) -> f64 {
    let n = period_returns.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = period_returns.iter().sum::<f64>() / n as f64;
    let var = period_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt() * periods_per_year.sqrt()
}

/// Compute max drawdown from a series of period returns.
fn max_drawdown_from_returns(period_returns: &[f64]) -> f64 {
    let mut cumulative = 1.0f64;
    let mut peak = 1.0f64;
    let mut max_dd = 0.0f64;

    for &r in period_returns {
        cumulative *= 1.0 + r;
        if cumulative > peak {
            peak = cumulative;
        }
        let dd = (peak - cumulative) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Run the factor backtest.
///
/// # Arguments
/// * `factor_panel` -- (n_periods, n_assets) factor scores
/// * `returns_panel` -- (n_periods, n_assets) asset returns (period t = next-period return)
/// * `config` -- backtest configuration
pub fn run_factor_backtest(
    factor_panel: &Array2<f64>,
    returns_panel: &Array2<f64>,
    config: &BacktestConfig,
) -> Result<FactorBacktestResult> {
    let (n_periods, n_assets) = factor_panel.dim();
    if returns_panel.dim() != (n_periods, n_assets) {
        return Err(FactorError::ShapeMismatch {
            msg: "factor_panel and returns_panel must have same shape".into(),
        });
    }
    if n_periods < config.rebalance_freq * 2 {
        return Err(FactorError::InsufficientData {
            required: config.rebalance_freq * 2,
            got: n_periods,
        });
    }

    let rebalance_periods_per_year = config.annualization / config.rebalance_freq as f64;

    // Bucket return series: Vec<Vec<f64>> of length n_buckets
    let mut bucket_return_series: Vec<Vec<f64>> = vec![Vec::new(); config.n_buckets];
    let mut long_short_returns: Vec<f64> = Vec::new();
    let mut ic_series: Vec<f64> = Vec::new();
    let mut prev_bucket_indices: Vec<Vec<usize>> = vec![Vec::new(); config.n_buckets];
    let mut bucket_turnover: Vec<Vec<f64>> = vec![Vec::new(); config.n_buckets];

    // Step through time in rebalance_freq increments
    let mut t = 0;
    while t + config.rebalance_freq <= n_periods {
        let factor_t: Vec<f64> = factor_panel.row(t).to_vec();

        // Compute IC at this rebalance: factor_t vs forward return (next rebalance period)
        let fwd_returns: Vec<f64> = if t + config.rebalance_freq < n_periods {
            // Average of next rebalance_freq period returns
            let mut avg = vec![0.0f64; n_assets];
            let mut cnt = vec![0u32; n_assets];
            for dt in 0..config.rebalance_freq {
                for i in 0..n_assets {
                    let r = returns_panel[[t + dt, i]];
                    if r.is_finite() {
                        avg[i] += r;
                        cnt[i] += 1;
                    }
                }
            }
            avg.iter().zip(cnt.iter()).map(|(&s, &c)| if c > 0 { s / c as f64 } else { f64::NAN }).collect()
        } else {
            vec![f64::NAN; n_assets]
        };

        let ic = compute_ic(&factor_t, &fwd_returns);
        ic_series.push(ic);

        // Assign to buckets
        let assignments = assign_buckets(&factor_t, config.n_buckets);

        // Get indices per bucket
        let curr_bucket_indices: Vec<Vec<usize>> = (0..config.n_buckets)
            .map(|b| {
                (0..n_assets)
                    .filter(|&i| assignments[i] == Some(b))
                    .collect()
            })
            .collect();

        // Compute returns over the holding period
        for b in 0..config.n_buckets {
            let mut period_return = 0.0;
            let mut period_count = 0;
            for dt in 0..config.rebalance_freq {
                let t_dt = t + dt;
                if t_dt < n_periods {
                    let period_ret_slice: Vec<f64> = returns_panel.row(t_dt).to_vec();
                    let ret = equal_weight_return(&period_ret_slice, &curr_bucket_indices[b]);
                    if ret.is_finite() {
                        period_return += ret;
                        period_count += 1;
                    }
                }
            }
            if period_count > 0 {
                let avg_return = period_return / period_count as f64;
                // Apply transaction cost based on turnover
                let turnover = compute_turnover(&prev_bucket_indices[b], &curr_bucket_indices[b]);
                let cost = turnover * config.transaction_cost;
                bucket_return_series[b].push(avg_return * config.rebalance_freq as f64 - cost);
                bucket_turnover[b].push(turnover);
            }
        }

        // Long-short return: top bucket - bottom bucket
        if config.long_short && config.n_buckets >= 2 {
            let long_b = config.n_buckets - 1;
            let short_b = 0;
            if let (Some(&lr), Some(&sr)) = (
                bucket_return_series[long_b].last(),
                bucket_return_series[short_b].last(),
            ) {
                long_short_returns.push(lr - sr);
            }
        }

        prev_bucket_indices = curr_bucket_indices;
        t += config.rebalance_freq;
    }

    // Compute statistics per bucket
    let mut bucket_stats = Vec::with_capacity(config.n_buckets);
    for b in 0..config.n_buckets {
        let returns = &bucket_return_series[b];
        if returns.is_empty() {
            bucket_stats.push(BucketStats {
                bucket: b,
                cumulative_return: 0.0,
                annualized_return: f64::NAN,
                annualized_vol: f64::NAN,
                sharpe_ratio: f64::NAN,
                max_drawdown: f64::NAN,
                avg_n_stocks: f64::NAN,
                avg_turnover: f64::NAN,
            });
            continue;
        }

        let cum_ret: f64 = returns.iter().map(|&r| 1.0 + r).product::<f64>() - 1.0;
        let ann_ret = annualized_return(returns, rebalance_periods_per_year);
        let ann_vol = annualized_vol(returns, rebalance_periods_per_year);
        let sharpe = if ann_vol > 1e-10 { ann_ret / ann_vol } else { f64::NAN };
        let mdd = max_drawdown_from_returns(returns);
        let avg_turn = if !bucket_turnover[b].is_empty() {
            bucket_turnover[b].iter().sum::<f64>() / bucket_turnover[b].len() as f64
        } else {
            f64::NAN
        };

        bucket_stats.push(BucketStats {
            bucket: b,
            cumulative_return: cum_ret,
            annualized_return: ann_ret,
            annualized_vol: ann_vol,
            sharpe_ratio: sharpe,
            max_drawdown: mdd,
            avg_n_stocks: n_assets as f64 / config.n_buckets as f64,
            avg_turnover: avg_turn,
        });
    }

    // Long-short stats
    let long_short_stats = if config.long_short && !long_short_returns.is_empty() {
        let cum_ret: f64 = long_short_returns.iter().map(|&r| 1.0 + r).product::<f64>() - 1.0;
        let ann_ret = annualized_return(&long_short_returns, rebalance_periods_per_year);
        let ann_vol = annualized_vol(&long_short_returns, rebalance_periods_per_year);
        let sharpe = if ann_vol > 1e-10 { ann_ret / ann_vol } else { f64::NAN };
        let mdd = max_drawdown_from_returns(&long_short_returns);
        Some(BucketStats {
            bucket: config.n_buckets,
            cumulative_return: cum_ret,
            annualized_return: ann_ret,
            annualized_vol: ann_vol,
            sharpe_ratio: sharpe,
            max_drawdown: mdd,
            avg_n_stocks: n_assets as f64 * 2.0 / config.n_buckets as f64,
            avg_turnover: f64::NAN,
        })
    } else {
        None
    };

    let ic_stats = compute_ic_statistics(&ic_series);
    let mean_ic = ic_stats.ic;
    let icir = ic_stats.icir;
    let t_stat_ic = ic_stats.t_statistic;

    // Return spread: annualized top bucket - bottom bucket
    let return_spread = if config.n_buckets >= 2 {
        let top = bucket_stats[config.n_buckets - 1].annualized_return;
        let bot = bucket_stats[0].annualized_return;
        if top.is_finite() && bot.is_finite() { top - bot } else { f64::NAN }
    } else {
        f64::NAN
    };

    // Monotonicity: fraction of consecutive bucket pairs where top > bottom
    let monotonicity = if config.n_buckets >= 2 {
        let n_pairs = config.n_buckets - 1;
        let n_monotone = (0..n_pairs)
            .filter(|&b| {
                let r_lo = bucket_stats[b].annualized_return;
                let r_hi = bucket_stats[b + 1].annualized_return;
                r_lo.is_finite() && r_hi.is_finite() && r_hi > r_lo
            })
            .count();
        n_monotone as f64 / n_pairs as f64
    } else {
        f64::NAN
    };

    Ok(FactorBacktestResult {
        bucket_stats,
        long_short_stats,
        ic_series,
        mean_ic,
        icir,
        t_stat_ic,
        return_spread,
        monotonicity,
        bucket_return_series,
    })
}

/// Summary table for printing backtest results.
pub fn format_backtest_summary(result: &FactorBacktestResult) -> String {
    let mut s = String::new();
    s.push_str("Factor Backtest Summary\n");
    s.push_str("=======================\n");
    s.push_str(&format!("Mean IC:    {:.4}\n", result.mean_ic));
    s.push_str(&format!("ICIR:       {:.4}\n", result.icir));
    s.push_str(&format!("t(IC):      {:.4}\n", result.t_stat_ic));
    s.push_str(&format!("Spread:     {:.2}%\n", result.return_spread * 100.0));
    s.push_str(&format!("Monotone:   {:.1}%\n", result.monotonicity * 100.0));
    s.push_str("\nBucket Performance:\n");
    s.push_str(&format!("{:<8} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "Bucket", "AnnRet%", "AnnVol%", "Sharpe", "MaxDD%", "Turnover%"));
    s.push_str(&"-".repeat(60));
    s.push('\n');
    for stat in &result.bucket_stats {
        s.push_str(&format!(
            "{:<8} {:>10.2} {:>10.2} {:>10.3} {:>10.2} {:>10.2}\n",
            stat.bucket + 1,
            stat.annualized_return * 100.0,
            stat.annualized_vol * 100.0,
            stat.sharpe_ratio,
            stat.max_drawdown * 100.0,
            stat.avg_turnover * 100.0,
        ));
    }
    if let Some(ref ls) = result.long_short_stats {
        s.push_str("\nLong-Short Portfolio:\n");
        s.push_str(&format!(
            "  AnnRet: {:.2}%  AnnVol: {:.2}%  Sharpe: {:.3}  MaxDD: {:.2}%\n",
            ls.annualized_return * 100.0,
            ls.annualized_vol * 100.0,
            ls.sharpe_ratio,
            ls.max_drawdown * 100.0,
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_panels(n_periods: usize, n_assets: usize) -> (Array2<f64>, Array2<f64>) {
        // Factor with mild predictive power
        let factor = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            (i as f64) + 0.1 * (t as f64 * 0.1).sin()
        });
        let returns = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            0.0005 * i as f64 - 0.005 + 0.01 * ((t + i) as f64 * 0.3).sin()
        });
        (factor, returns)
    }

    #[test]
    fn test_assign_buckets() {
        let scores = vec![1.0, 5.0, 3.0, 2.0, 4.0, f64::NAN];
        let buckets = assign_buckets(&scores, 5);
        // 5 valid, 5 buckets: each gets exactly one
        let valid_buckets: Vec<usize> = buckets.iter().filter_map(|&b| b).collect();
        assert_eq!(valid_buckets.len(), 5);
        assert!(buckets[5].is_none()); // NaN
    }

    #[test]
    fn test_factor_backtest() {
        let config = BacktestConfig {
            n_buckets: 5,
            rebalance_freq: 21,
            ..Default::default()
        };
        let (factor, returns) = make_test_panels(252, 50);
        let result = run_factor_backtest(&factor, &returns, &config).unwrap();
        assert_eq!(result.bucket_stats.len(), 5);
        assert!(result.long_short_stats.is_some());
        assert!(!result.ic_series.is_empty());
    }

    #[test]
    fn test_decile_backtest() {
        let config = BacktestConfig {
            n_buckets: 10,
            rebalance_freq: 21,
            ..Default::default()
        };
        let (factor, returns) = make_test_panels(300, 100);
        let result = run_factor_backtest(&factor, &returns, &config).unwrap();
        assert_eq!(result.bucket_stats.len(), 10);
    }

    #[test]
    fn test_annualized_return() {
        // 1% per period x 12 periods = annual
        let returns = vec![0.01; 12];
        let ann = annualized_return(&returns, 12.0);
        // (1.01^12 - 1) ~= 0.12683
        assert!((ann - 0.12682503_f64).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown_from_returns() {
        let returns = vec![0.01, 0.02, -0.05, 0.01, 0.02];
        let mdd = max_drawdown_from_returns(&returns);
        assert!(mdd > 0.0 && mdd < 0.1);
    }
}
