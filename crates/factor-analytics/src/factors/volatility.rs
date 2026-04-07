//! Volatility factors: realized vol (22d, 63d), EWMA vol (lambda=0.94),
//! downside deviation, maximum drawdown, and vol-of-vol.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};

/// Configuration for volatility factor computation.
#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    /// Short-horizon realized vol window (trading days)
    pub short_window: usize,
    /// Long-horizon realized vol window (trading days)
    pub long_window: usize,
    /// EWMA decay factor (RiskMetrics: 0.94)
    pub ewma_lambda: f64,
    /// Minimum acceptable threshold for downside deviation
    pub mar: f64,
    /// Annualization factor (252 for daily)
    pub annualization: f64,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            short_window: 22,
            long_window: 63,
            ewma_lambda: 0.94,
            mar: 0.0, // minimum acceptable return = 0
            annualization: 252.0,
        }
    }
}

/// All volatility factors for a single asset.
#[derive(Debug, Clone)]
pub struct VolatilityFactors {
    /// Realized volatility over short window (annualized)
    pub realized_vol_22d: f64,
    /// Realized volatility over long window (annualized)
    pub realized_vol_63d: f64,
    /// EWMA volatility (annualized), lambda = 0.94
    pub ewma_vol: f64,
    /// Downside deviation relative to MAR (annualized)
    pub downside_deviation: f64,
    /// Maximum drawdown over the available history
    pub max_drawdown: f64,
    /// Vol-of-vol: rolling std of realized vol estimates
    pub vol_of_vol: f64,
    /// Vol ratio: short/long vol (term structure of vol)
    pub vol_ratio: f64,
}

/// Compute annualized realized volatility from log returns.
pub fn realized_vol(log_returns: &[f64], annualization: f64) -> f64 {
    let n = log_returns.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = log_returns.iter().sum::<f64>() / n as f64;
    let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt() * annualization.sqrt()
}

/// Compute EWMA variance recursively.
///
/// sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r_t^2
/// Initializes with the sample variance of the first 10 observations.
pub fn ewma_variance(log_returns: &[f64], lambda: f64) -> f64 {
    let n = log_returns.len();
    if n < 2 {
        return f64::NAN;
    }
    // Seed with variance of first min(10, n) observations
    let seed_n = n.min(10);
    let seed_mean = log_returns[..seed_n].iter().sum::<f64>() / seed_n as f64;
    let mut var = log_returns[..seed_n]
        .iter()
        .map(|r| (r - seed_mean).powi(2))
        .sum::<f64>()
        / seed_n as f64;

    for &r in &log_returns[seed_n..] {
        var = lambda * var + (1.0 - lambda) * r * r;
    }
    var
}

/// Compute downside deviation relative to a minimum acceptable return (MAR).
///
/// downside_dev = sqrt(mean of (r - MAR)^2 where r < MAR) * sqrt(annualization)
pub fn downside_deviation(log_returns: &[f64], mar_daily: f64, annualization: f64) -> f64 {
    let below_mar: Vec<f64> = log_returns
        .iter()
        .filter_map(|&r| if r < mar_daily { Some((r - mar_daily).powi(2)) } else { None })
        .collect();

    if below_mar.is_empty() {
        return 0.0;
    }

    let mean_sq = below_mar.iter().sum::<f64>() / log_returns.len() as f64;
    mean_sq.sqrt() * annualization.sqrt()
}

/// Compute maximum drawdown from a price series reconstructed from log returns.
///
/// Returns the maximum peak-to-trough decline as a positive number.
pub fn max_drawdown(log_returns: &[f64]) -> f64 {
    let n = log_returns.len();
    if n == 0 {
        return 0.0;
    }

    let mut cum_prices = Vec::with_capacity(n + 1);
    cum_prices.push(1.0_f64);
    let mut running = 1.0_f64;
    for &r in log_returns {
        running *= r.exp();
        cum_prices.push(running);
    }

    let mut max_dd = 0.0_f64;
    let mut peak = cum_prices[0];
    for &p in &cum_prices[1..] {
        if p > peak {
            peak = p;
        }
        let dd = (peak - p) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Compute vol-of-vol: rolling 22-day realized vol computed over sub-windows,
/// then take the std of those estimates.
///
/// Uses a 63-day lookback with 22-day sub-windows stepped daily.
pub fn vol_of_vol(log_returns: &[f64], sub_window: usize, n_sub_windows: usize) -> f64 {
    let n = log_returns.len();
    let required = sub_window + n_sub_windows - 1;
    if n < required {
        return f64::NAN;
    }

    let mut rolling_vols: Vec<f64> = Vec::with_capacity(n_sub_windows);
    for i in 0..n_sub_windows {
        let start = n - required + i;
        let end = start + sub_window;
        let rv = realized_vol(&log_returns[start..end], 252.0);
        if rv.is_finite() {
            rolling_vols.push(rv);
        }
    }

    if rolling_vols.len() < 2 {
        return f64::NAN;
    }

    let mean = rolling_vols.iter().sum::<f64>() / rolling_vols.len() as f64;
    let var = rolling_vols.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / (rolling_vols.len() - 1) as f64;
    var.sqrt()
}

/// Compute all volatility factors for a single asset.
///
/// # Arguments
/// * `log_returns` -- daily log returns, most recent last
/// * `config` -- volatility config
pub fn compute_volatility_factors(
    log_returns: &[f64],
    config: &VolatilityConfig,
) -> Result<VolatilityFactors> {
    let n = log_returns.len();
    let min_required = config.long_window + 30; // need some extra for vol-of-vol
    if n < min_required {
        return Err(FactorError::InsufficientData {
            required: min_required,
            got: n,
        });
    }

    let rv_22 = realized_vol(
        &log_returns[n - config.short_window..],
        config.annualization,
    );

    let rv_63 = realized_vol(
        &log_returns[n - config.long_window..],
        config.annualization,
    );

    let ewma_var = ewma_variance(&log_returns[n - config.long_window..], config.ewma_lambda);
    let ewma_vol_ann = ewma_var.sqrt() * config.annualization.sqrt();

    // Daily MAR from annualized MAR
    let daily_mar = config.mar / config.annualization;
    let dd = downside_deviation(
        &log_returns[n - config.long_window..],
        daily_mar,
        config.annualization,
    );

    let mdd = max_drawdown(log_returns);

    let vov = vol_of_vol(log_returns, config.short_window, 30);

    let vol_ratio = if rv_63 > 1e-10 { rv_22 / rv_63 } else { f64::NAN };

    Ok(VolatilityFactors {
        realized_vol_22d: rv_22,
        realized_vol_63d: rv_63,
        ewma_vol: ewma_vol_ann,
        downside_deviation: dd,
        max_drawdown: mdd,
        vol_of_vol: vov,
        vol_ratio,
    })
}

/// Compute cross-sectional volatility factor panel.
///
/// Returns Array2 of shape (n_assets, 7).
/// Note: volatility factors are typically negatively signed for long-only portfolios
/// (lower vol = better risk-adjusted performance) -- callers should decide sign convention.
pub fn compute_panel_volatility(
    returns_matrix: &Array2<f64>,
    config: &VolatilityConfig,
) -> Result<Array2<f64>> {
    let (n_days, n_assets) = returns_matrix.dim();
    let min_required = config.long_window + 30;
    if n_days < min_required {
        return Err(FactorError::InsufficientData {
            required: min_required,
            got: n_days,
        });
    }

    let mut result = Array2::<f64>::from_elem((n_assets, 7), f64::NAN);

    for j in 0..n_assets {
        let asset_returns: Vec<f64> = returns_matrix.column(j).to_vec();
        match compute_volatility_factors(&asset_returns, config) {
            Ok(f) => {
                result[[j, 0]] = f.realized_vol_22d;
                result[[j, 1]] = f.realized_vol_63d;
                result[[j, 2]] = f.ewma_vol;
                result[[j, 3]] = f.downside_deviation;
                result[[j, 4]] = f.max_drawdown;
                result[[j, 5]] = f.vol_of_vol;
                result[[j, 6]] = f.vol_ratio;
            }
            Err(_) => {}
        }
    }

    Ok(result)
}

/// Names of volatility factor columns.
pub fn volatility_factor_names() -> Vec<&'static str> {
    vec![
        "realized_vol_22d",
        "realized_vol_63d",
        "ewma_vol",
        "downside_deviation",
        "max_drawdown",
        "vol_of_vol",
        "vol_ratio",
    ]
}

/// Compute Sortino ratio: excess return / downside deviation.
pub fn sortino_ratio(log_returns: &[f64], mar_annual: f64, annualization: f64) -> f64 {
    if log_returns.is_empty() {
        return f64::NAN;
    }
    let n = log_returns.len() as f64;
    let mean_daily = log_returns.iter().sum::<f64>() / n;
    let ann_return = mean_daily * annualization;
    let excess = ann_return - mar_annual;
    let daily_mar = mar_annual / annualization;
    let dd = downside_deviation(log_returns, daily_mar, annualization);
    if dd < 1e-10 {
        f64::INFINITY
    } else {
        excess / dd
    }
}

/// Calmar ratio: annualized return / max drawdown.
pub fn calmar_ratio(log_returns: &[f64], annualization: f64) -> f64 {
    if log_returns.is_empty() {
        return f64::NAN;
    }
    let n = log_returns.len() as f64;
    let mean_daily = log_returns.iter().sum::<f64>() / n;
    let ann_return = mean_daily * annualization;
    let mdd = max_drawdown(log_returns);
    if mdd < 1e-10 {
        f64::INFINITY
    } else {
        ann_return / mdd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_returns(n: usize, drift: f64) -> Vec<f64> {
        (0..n)
            .map(|i| drift + 0.01 * ((i as f64) * 0.15).sin())
            .collect()
    }

    #[test]
    fn test_realized_vol() {
        let returns = make_returns(22, 0.0);
        let vol = realized_vol(&returns, 252.0);
        assert!(vol > 0.0 && vol < 1.0);
    }

    #[test]
    fn test_ewma_variance() {
        let returns = make_returns(63, 0.0);
        let var = ewma_variance(&returns, 0.94);
        assert!(var > 0.0);
    }

    #[test]
    fn test_max_drawdown_positive() {
        // Declining series should have positive drawdown
        let returns: Vec<f64> = (0..50).map(|_| -0.01).collect();
        let mdd = max_drawdown(&returns);
        assert!(mdd > 0.0 && mdd < 1.0);
    }

    #[test]
    fn test_max_drawdown_monotone_up() {
        // Monotonically rising: no drawdown
        let returns: Vec<f64> = (0..50).map(|_| 0.001).collect();
        let mdd = max_drawdown(&returns);
        assert!(mdd < 1e-10);
    }

    #[test]
    fn test_vol_of_vol() {
        let returns = make_returns(200, 0.0);
        let vov = vol_of_vol(&returns, 22, 30);
        assert!(vov.is_finite() && vov >= 0.0);
    }

    #[test]
    fn test_volatility_factors() {
        let config = VolatilityConfig::default();
        let n = config.long_window + 40;
        let returns = make_returns(n, 0.0002);
        let f = compute_volatility_factors(&returns, &config).unwrap();
        assert!(f.realized_vol_22d > 0.0);
        assert!(f.realized_vol_63d > 0.0);
        assert!(f.ewma_vol > 0.0);
        assert!(f.max_drawdown >= 0.0);
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = make_returns(252, 0.0005);
        let s = sortino_ratio(&returns, 0.02, 252.0);
        assert!(s.is_finite());
    }
}
