//! Multi-horizon momentum factors.
//!
//! Implements: 1m/3m/6m/12m returns, risk-adjusted momentum,
//! residual momentum (CAPM-adjusted), 52-week high proximity, 1-week reversal.

use ndarray::{Array1, Array2, ArrayView1, Axis};
use crate::error::{FactorError, Result};

/// Configuration for momentum factor computation.
#[derive(Debug, Clone)]
pub struct MomentumConfig {
    /// Number of trading days in 1 month
    pub days_1m: usize,
    /// Number of trading days in 3 months
    pub days_3m: usize,
    /// Number of trading days in 6 months
    pub days_6m: usize,
    /// Number of trading days in 12 months
    pub days_12m: usize,
    /// Lookback for reversal (1 week)
    pub days_reversal: usize,
    /// Volatility window for risk adjustment
    pub vol_window: usize,
    /// Skip last N days for momentum (avoids short-term reversal)
    pub skip_days: usize,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            days_1m: 21,
            days_3m: 63,
            days_6m: 126,
            days_12m: 252,
            days_reversal: 5,
            vol_window: 63,
            skip_days: 21,
        }
    }
}

/// All momentum factors for a single asset computed at one point in time.
#[derive(Debug, Clone)]
pub struct MomentumFactors {
    /// 1-month cumulative return (skipping last `skip_days`)
    pub ret_1m: f64,
    /// 3-month cumulative return (skipping last `skip_days`)
    pub ret_3m: f64,
    /// 6-month cumulative return (skipping last `skip_days`)
    pub ret_6m: f64,
    /// 12-month cumulative return (skipping last `skip_days`)
    pub ret_12m: f64,
    /// Risk-adjusted 12m momentum: ret_12m / realized_vol
    pub risk_adj_mom: f64,
    /// Residual momentum: CAPM alpha cumulated over 12m
    pub residual_mom: f64,
    /// 52-week high proximity: price / max(price, 52w)
    pub high_52w_proximity: f64,
    /// 1-week reversal: negative of 1w return (captures mean reversion)
    pub reversal_1w: f64,
}

/// Compute cumulative log return over a slice of log returns.
///
/// Returns sum of log returns = log(P_t / P_{t-n}).
fn cumulative_return(log_returns: &[f64]) -> f64 {
    log_returns.iter().sum()
}

/// Compute realized volatility (annualized) from daily log returns.
fn realized_vol(log_returns: &[f64]) -> f64 {
    let n = log_returns.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = log_returns.iter().sum::<f64>() / n as f64;
    let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt() * (252.0_f64).sqrt()
}

/// OLS regression of y on [1, x]. Returns (alpha, beta).
pub fn ols_simple(y: &[f64], x: &[f64]) -> Result<(f64, f64)> {
    let n = y.len();
    if n != x.len() {
        return Err(FactorError::DimensionMismatch {
            expected: n,
            got: x.len(),
        });
    }
    if n < 3 {
        return Err(FactorError::InsufficientData { required: 3, got: n });
    }
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|v| v * v).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let n_f = n as f64;
    let denom = n_f * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return Err(FactorError::SingularMatrix {
            operation: "OLS simple regression".into(),
        });
    }
    let beta = (n_f * sum_xy - sum_x * sum_y) / denom;
    let alpha = (sum_y - beta * sum_x) / n_f;
    Ok((alpha, beta))
}

/// Compute CAPM residual momentum.
///
/// Regresses asset log returns on market log returns over the window,
/// then sums the residuals (cumulative alpha) excluding the skip period.
fn capm_residual_momentum(
    asset_returns: &[f64],
    market_returns: &[f64],
    skip_days: usize,
) -> f64 {
    let n = asset_returns.len();
    if n <= skip_days + 3 {
        return f64::NAN;
    }
    let end = n - skip_days;
    let asset_window = &asset_returns[..end];
    let market_window = &market_returns[..end];

    match ols_simple(asset_window, market_window) {
        Ok((alpha, beta)) => {
            // Compute cumulative residual
            asset_window
                .iter()
                .zip(market_window.iter())
                .map(|(r_a, r_m)| r_a - alpha - beta * r_m)
                .sum()
        }
        Err(_) => f64::NAN,
    }
}

/// Compute all momentum factors for a single asset given its price series and market returns.
///
/// # Arguments
/// * `log_returns` -- daily log returns of the asset, most recent last
/// * `market_returns` -- daily log returns of the market index, same length
/// * `config` -- momentum configuration
pub fn compute_momentum_factors(
    log_returns: &[f64],
    market_returns: &[f64],
    config: &MomentumConfig,
) -> Result<MomentumFactors> {
    let n = log_returns.len();
    let min_required = config.days_12m + config.skip_days;
    if n < min_required {
        return Err(FactorError::InsufficientData {
            required: min_required,
            got: n,
        });
    }
    if market_returns.len() != n {
        return Err(FactorError::DimensionMismatch {
            expected: n,
            got: market_returns.len(),
        });
    }

    let skip = config.skip_days;

    // Returns windows exclude last `skip` days
    let window_end = n - skip;

    let ret_1m = if window_end >= config.days_1m {
        cumulative_return(&log_returns[window_end - config.days_1m..window_end])
    } else {
        f64::NAN
    };

    let ret_3m = if window_end >= config.days_3m {
        cumulative_return(&log_returns[window_end - config.days_3m..window_end])
    } else {
        f64::NAN
    };

    let ret_6m = if window_end >= config.days_6m {
        cumulative_return(&log_returns[window_end - config.days_6m..window_end])
    } else {
        f64::NAN
    };

    let ret_12m = if window_end >= config.days_12m {
        cumulative_return(&log_returns[window_end - config.days_12m..window_end])
    } else {
        f64::NAN
    };

    // Risk-adjusted momentum: 12m return / realized vol over vol_window
    let vol_start = n.saturating_sub(config.vol_window);
    let vol = realized_vol(&log_returns[vol_start..n]);
    let risk_adj_mom = if vol.is_nan() || vol < 1e-10 {
        f64::NAN
    } else {
        ret_12m / vol
    };

    // Residual momentum using full 12m + skip window
    let window_start = n.saturating_sub(config.days_12m + skip);
    let residual_mom = capm_residual_momentum(
        &log_returns[window_start..],
        &market_returns[window_start..],
        skip,
    );

    // 52-week high proximity -- reconstruct price series from log returns
    // Use last 252 trading days; price_{t} = exp(sum of log returns up to t)
    let lookback_252 = n.saturating_sub(config.days_12m);
    let price_window = &log_returns[lookback_252..];
    let mut cum_price = 1.0_f64;
    let mut max_price = 1.0_f64;
    for &r in price_window {
        cum_price *= r.exp();
        if cum_price > max_price {
            max_price = cum_price;
        }
    }
    let high_52w_proximity = if max_price < 1e-10 {
        f64::NAN
    } else {
        cum_price / max_price
    };

    // 1-week reversal: negative of short-term return (mean reversion signal)
    let reversal_1w = if n >= config.days_reversal {
        -cumulative_return(&log_returns[n - config.days_reversal..])
    } else {
        f64::NAN
    };

    Ok(MomentumFactors {
        ret_1m,
        ret_3m,
        ret_6m,
        ret_12m,
        risk_adj_mom,
        residual_mom,
        high_52w_proximity,
        reversal_1w,
    })
}

/// Compute cross-sectional momentum factors for a panel of assets.
///
/// # Arguments
/// * `returns_matrix` -- shape (n_days, n_assets) of daily log returns
/// * `market_returns` -- shape (n_days,) market log returns
/// * `config` -- momentum config
///
/// Returns a matrix of shape (n_assets, 8) with the factor scores.
pub fn compute_panel_momentum(
    returns_matrix: &Array2<f64>,
    market_returns: &Array1<f64>,
    config: &MomentumConfig,
) -> Result<Array2<f64>> {
    let (n_days, n_assets) = returns_matrix.dim();
    if market_returns.len() != n_days {
        return Err(FactorError::DimensionMismatch {
            expected: n_days,
            got: market_returns.len(),
        });
    }

    let market_slice: Vec<f64> = market_returns.to_vec();
    let mut result = Array2::<f64>::from_elem((n_assets, 8), f64::NAN);

    for j in 0..n_assets {
        let asset_col: Vec<f64> = returns_matrix.column(j).to_vec();
        match compute_momentum_factors(&asset_col, &market_slice, config) {
            Ok(f) => {
                result[[j, 0]] = f.ret_1m;
                result[[j, 1]] = f.ret_3m;
                result[[j, 2]] = f.ret_6m;
                result[[j, 3]] = f.ret_12m;
                result[[j, 4]] = f.risk_adj_mom;
                result[[j, 5]] = f.residual_mom;
                result[[j, 6]] = f.high_52w_proximity;
                result[[j, 7]] = f.reversal_1w;
            }
            Err(_) => { /* leave as NAN */ }
        }
    }

    Ok(result)
}

/// Names of the momentum factor columns in panel output.
pub fn momentum_factor_names() -> Vec<&'static str> {
    vec![
        "ret_1m",
        "ret_3m",
        "ret_6m",
        "ret_12m",
        "risk_adj_mom",
        "residual_mom",
        "high_52w_proximity",
        "reversal_1w",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_returns(n: usize, drift: f64, seed_offset: f64) -> Vec<f64> {
        // Deterministic pseudo-random returns for testing
        (0..n)
            .map(|i| {
                let t = i as f64 + seed_offset;
                drift + 0.01 * (t * 0.3).sin()
            })
            .collect()
    }

    #[test]
    fn test_momentum_factors_basic() {
        let config = MomentumConfig::default();
        let n = config.days_12m + config.skip_days + 10;
        let returns = synthetic_returns(n, 0.0003, 0.0);
        let market = synthetic_returns(n, 0.0002, 1.0);

        let factors = compute_momentum_factors(&returns, &market, &config).unwrap();
        assert!(!factors.ret_12m.is_nan());
        assert!(!factors.risk_adj_mom.is_nan());
        assert!(factors.high_52w_proximity > 0.0 && factors.high_52w_proximity <= 1.0);
    }

    #[test]
    fn test_insufficient_data_error() {
        let config = MomentumConfig::default();
        let result = compute_momentum_factors(&[0.001; 10], &[0.001; 10], &config);
        assert!(matches!(result, Err(FactorError::InsufficientData { .. })));
    }

    #[test]
    fn test_ols_simple() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|xi| 2.0 * xi + 1.0).collect();
        let (alpha, beta) = ols_simple(&y, &x).unwrap();
        assert!((alpha - 1.0).abs() < 1e-9);
        assert!((beta - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_panel_momentum() {
        let config = MomentumConfig::default();
        let n = config.days_12m + config.skip_days + 10;
        let n_assets = 5;
        let returns = Array2::from_shape_fn((n, n_assets), |(i, j)| {
            0.0002 + 0.005 * ((i as f64 + j as f64) * 0.1).sin()
        });
        let market = Array1::from_shape_fn(n, |i| 0.0001 + 0.004 * (i as f64 * 0.1).sin());

        let panel = compute_panel_momentum(&returns, &market, &config).unwrap();
        assert_eq!(panel.dim(), (n_assets, 8));
    }
}
