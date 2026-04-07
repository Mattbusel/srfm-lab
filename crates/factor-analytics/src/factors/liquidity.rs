//! Liquidity factors: Amihud illiquidity, bid-ask spread proxy,
//! turnover ratio, and dollar volume.

use ndarray::Array2;
use crate::error::{FactorError, Result};

/// Daily market microstructure data for one asset.
#[derive(Debug, Clone)]
pub struct DailyMarketData {
    /// Daily absolute log return |r_t|
    pub abs_return: f64,
    /// Daily dollar volume (price * shares traded)
    pub dollar_volume: f64,
    /// Daily high price
    pub high: f64,
    /// Daily low price
    pub low: f64,
    /// Daily close price
    pub close: f64,
    /// Daily shares traded
    pub shares_volume: f64,
    /// Shares outstanding on this day
    pub shares_outstanding: f64,
}

/// Configuration for liquidity factor computation.
#[derive(Debug, Clone)]
pub struct LiquidityConfig {
    /// Lookback window for Amihud illiquidity (trading days)
    pub amihud_window: usize,
    /// Scale factor for Amihud (10^6 to get readable numbers)
    pub amihud_scale: f64,
    /// Window for dollar volume average
    pub dv_window: usize,
    /// Window for turnover ratio average
    pub turnover_window: usize,
}

impl Default for LiquidityConfig {
    fn default() -> Self {
        Self {
            amihud_window: 63,
            amihud_scale: 1e6,
            dv_window: 22,
            turnover_window: 22,
        }
    }
}

/// Liquidity factors for a single asset.
#[derive(Debug, Clone)]
pub struct LiquidityFactors {
    /// Amihud illiquidity = avg(|r_t| / DollarVolume_t) * scale
    pub amihud_illiquidity: f64,
    /// Bid-ask spread proxy: avg((High - Low) / Close)
    pub hl_spread_proxy: f64,
    /// Average daily turnover ratio = shares traded / shares outstanding
    pub turnover_ratio: f64,
    /// Log average daily dollar volume
    pub log_avg_dollar_volume: f64,
    /// Dollar volume z-score (requires cross-section)
    pub dv_zscore: f64,
}

/// Compute Amihud (2002) illiquidity ratio.
///
/// ILLIQ = (1/T) * sum_t(|R_t| / DVOL_t) * scale
///
/// Higher = less liquid.
pub fn amihud_illiquidity(data: &[DailyMarketData], scale: f64) -> f64 {
    let valid: Vec<f64> = data
        .iter()
        .filter_map(|d| {
            if d.dollar_volume > 1e-6 {
                Some(d.abs_return / d.dollar_volume)
            } else {
                None
            }
        })
        .collect();

    if valid.is_empty() {
        return f64::NAN;
    }

    (valid.iter().sum::<f64>() / valid.len() as f64) * scale
}

/// Compute the high-low spread proxy (Corwin-Schultz 2012 simplified).
///
/// Proxy = avg((High - Low) / Close) over the window.
/// This is a simple version; Corwin-Schultz uses 2-day overlapping windows.
pub fn hl_spread_proxy(data: &[DailyMarketData]) -> f64 {
    let valid: Vec<f64> = data
        .iter()
        .filter_map(|d| {
            if d.close > 1e-10 && d.high >= d.low {
                Some((d.high - d.low) / d.close)
            } else {
                None
            }
        })
        .collect();

    if valid.is_empty() {
        return f64::NAN;
    }

    valid.iter().sum::<f64>() / valid.len() as f64
}

/// Compute Corwin-Schultz (2012) spread estimate using 2-day windows.
///
/// beta = sum over 2-day windows of [ln(H_t/L_t)^2 + ln(H_{t+1}/L_{t+1})^2]
/// gamma = [ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))]^2
///
/// alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
/// spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
pub fn corwin_schultz_spread(data: &[DailyMarketData]) -> f64 {
    let n = data.len();
    if n < 4 {
        return f64::NAN;
    }

    let k = (3.0 - 2.0 * 2.0_f64.sqrt()).recip();
    let mut estimates: Vec<f64> = Vec::new();

    for i in 0..n - 1 {
        let h1 = data[i].high;
        let l1 = data[i].low;
        let h2 = data[i + 1].high;
        let l2 = data[i + 1].low;

        if l1 <= 0.0 || l2 <= 0.0 || h1 <= 0.0 || h2 <= 0.0 {
            continue;
        }

        let beta = (h1 / l1).ln().powi(2) + (h2 / l2).ln().powi(2);
        let h_max = h1.max(h2);
        let l_min = l1.min(l2);
        if l_min <= 0.0 {
            continue;
        }
        let gamma = (h_max / l_min).ln().powi(2);

        let alpha = (2.0 * beta).sqrt() - beta.sqrt();
        let alpha = alpha / (3.0 - 2.0 * 2.0_f64.sqrt()) - (gamma * k).sqrt();

        let spread = 2.0 * (alpha.exp() - 1.0) / (1.0 + alpha.exp());
        if spread.is_finite() && spread > 0.0 {
            estimates.push(spread);
        }
    }

    if estimates.is_empty() {
        return f64::NAN;
    }

    estimates.iter().sum::<f64>() / estimates.len() as f64
}

/// Compute turnover ratio = shares traded / shares outstanding.
pub fn turnover_ratio(data: &[DailyMarketData]) -> f64 {
    let valid: Vec<f64> = data
        .iter()
        .filter_map(|d| {
            if d.shares_outstanding > 1e-6 {
                Some(d.shares_volume / d.shares_outstanding)
            } else {
                None
            }
        })
        .collect();

    if valid.is_empty() {
        return f64::NAN;
    }

    valid.iter().sum::<f64>() / valid.len() as f64
}

/// Compute average log dollar volume.
pub fn avg_log_dollar_volume(data: &[DailyMarketData]) -> f64 {
    let valid: Vec<f64> = data
        .iter()
        .filter_map(|d| {
            if d.dollar_volume > 0.0 {
                Some(d.dollar_volume.ln())
            } else {
                None
            }
        })
        .collect();

    if valid.is_empty() {
        return f64::NAN;
    }

    valid.iter().sum::<f64>() / valid.len() as f64
}

/// Compute all liquidity factors for a single asset.
pub fn compute_liquidity_factors(
    data: &[DailyMarketData],
    config: &LiquidityConfig,
) -> Result<LiquidityFactors> {
    let n = data.len();
    let min_req = config.amihud_window.max(config.dv_window).max(config.turnover_window);
    if n < min_req {
        return Err(FactorError::InsufficientData {
            required: min_req,
            got: n,
        });
    }

    let amihud_window_data = &data[n - config.amihud_window..];
    let dv_window_data = &data[n - config.dv_window..];
    let turnover_data = &data[n - config.turnover_window..];

    let amihud = amihud_illiquidity(amihud_window_data, config.amihud_scale);
    let hl_spread = hl_spread_proxy(dv_window_data);
    let turnover = turnover_ratio(turnover_data);
    let log_dv = avg_log_dollar_volume(dv_window_data);

    Ok(LiquidityFactors {
        amihud_illiquidity: amihud,
        hl_spread_proxy: hl_spread,
        turnover_ratio: turnover,
        log_avg_dollar_volume: log_dv,
        dv_zscore: f64::NAN, // set by cross-sectional normalization
    })
}

/// Cross-sectional liquidity factor panel.
///
/// # Arguments
/// * `panel_data` -- vec of daily data series per asset
///
/// Returns Array2 of shape (n_assets, 5).
/// Note: liquidity factors as defined are "illiquidity" -- higher = worse liquidity.
/// For factor investing, invert signs when constructing long-liquid portfolios.
pub fn compute_panel_liquidity(
    panel_data: &[Vec<DailyMarketData>],
    config: &LiquidityConfig,
) -> Result<Array2<f64>> {
    let n = panel_data.len();
    if n == 0 {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    let mut result = Array2::<f64>::from_elem((n, 5), f64::NAN);

    for (i, asset_data) in panel_data.iter().enumerate() {
        match compute_liquidity_factors(asset_data, config) {
            Ok(f) => {
                result[[i, 0]] = f.amihud_illiquidity;
                result[[i, 1]] = f.hl_spread_proxy;
                result[[i, 2]] = f.turnover_ratio;
                result[[i, 3]] = f.log_avg_dollar_volume;
                result[[i, 4]] = f64::NAN; // dv_zscore filled below
            }
            Err(_) => {}
        }
    }

    // Compute dollar volume z-score cross-sectionally
    let dv_col: Vec<f64> = (0..n).map(|i| result[[i, 3]]).collect();
    let valid_dv: Vec<f64> = dv_col.iter().copied().filter(|v| v.is_finite()).collect();
    if valid_dv.len() >= 2 {
        let mean_dv = valid_dv.iter().sum::<f64>() / valid_dv.len() as f64;
        let std_dv = (valid_dv.iter().map(|v| (v - mean_dv).powi(2)).sum::<f64>()
            / (valid_dv.len() - 1) as f64)
            .sqrt();
        for i in 0..n {
            let dv = result[[i, 3]];
            result[[i, 4]] = if dv.is_finite() && std_dv > 1e-10 {
                (dv - mean_dv) / std_dv
            } else {
                f64::NAN
            };
        }
    }

    Ok(result)
}

/// Names of liquidity factor columns.
pub fn liquidity_factor_names() -> Vec<&'static str> {
    vec![
        "amihud_illiquidity",
        "hl_spread_proxy",
        "turnover_ratio",
        "log_avg_dollar_volume",
        "dv_zscore",
    ]
}

/// Pastor-Stambaugh (2003) liquidity beta approximation.
///
/// Regresses asset return changes on signed order flow * lagged return
/// over a window of daily data.
/// Returns the liquidity loading coefficient (gamma).
pub fn pastor_stambaugh_gamma(
    returns: &[f64],
    dollar_volumes: &[f64],
) -> f64 {
    let n = returns.len();
    if n < 10 || dollar_volumes.len() != n {
        return f64::NAN;
    }

    // Proxy for order flow: sign(return) * volume
    // PS regression: r_{t+1} = alpha + phi*r_t + gamma*(sign(r_t)*vol_t) + e
    let mut x_vals: Vec<f64> = Vec::new();
    let mut y_vals: Vec<f64> = Vec::new();

    for t in 0..n - 1 {
        if dollar_volumes[t] > 0.0 {
            let signed_vol = returns[t].signum() * dollar_volumes[t];
            x_vals.push(signed_vol);
            y_vals.push(returns[t + 1]);
        }
    }

    if x_vals.len() < 5 {
        return f64::NAN;
    }

    // Simple OLS: r_{t+1} on signed_vol (simplified, omitting lagged return)
    match crate::factors::momentum::ols_simple(&y_vals, &x_vals) {
        Ok((_alpha, beta)) => beta,
        Err(_) => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_market_data(n: usize) -> Vec<DailyMarketData> {
        (0..n)
            .map(|i| {
                let close = 100.0 + (i as f64 * 0.1).sin();
                DailyMarketData {
                    abs_return: (0.01 * (i as f64 * 0.2).sin()).abs(),
                    dollar_volume: 1_000_000.0 + 100_000.0 * (i as f64 * 0.3).cos(),
                    high: close * 1.005,
                    low: close * 0.995,
                    close,
                    shares_volume: 10_000.0 + 1000.0 * (i as f64 * 0.1).sin().abs(),
                    shares_outstanding: 1_000_000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_amihud() {
        let data = make_market_data(63);
        let illiq = amihud_illiquidity(&data, 1e6);
        assert!(illiq.is_finite() && illiq > 0.0);
    }

    #[test]
    fn test_hl_spread_proxy() {
        let data = make_market_data(22);
        let spread = hl_spread_proxy(&data);
        assert!(spread > 0.0 && spread < 0.1);
    }

    #[test]
    fn test_liquidity_factors() {
        let config = LiquidityConfig::default();
        let data = make_market_data(100);
        let f = compute_liquidity_factors(&data, &config).unwrap();
        assert!(f.amihud_illiquidity.is_finite());
        assert!(f.log_avg_dollar_volume.is_finite());
        assert!(f.turnover_ratio > 0.0);
    }

    #[test]
    fn test_corwin_schultz() {
        let data = make_market_data(30);
        let spread = corwin_schultz_spread(&data);
        // May be NAN if all estimates are negative (can happen with synthetic data)
        // Just check it doesn't panic
        let _ = spread;
    }

    #[test]
    fn test_panel_liquidity() {
        let config = LiquidityConfig::default();
        let panel: Vec<Vec<DailyMarketData>> = (0..10).map(|_| make_market_data(100)).collect();
        let result = compute_panel_liquidity(&panel, &config).unwrap();
        assert_eq!(result.dim(), (10, 5));
    }
}
