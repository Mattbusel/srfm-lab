//! Value factors: earnings yield, book-to-market, sales-to-price,
//! FCF yield, dividend yield, and composite value score.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};

/// Fundamental data for a single asset at a point in time.
#[derive(Debug, Clone)]
pub struct FundamentalData {
    /// Market capitalization (price * shares outstanding)
    pub market_cap: f64,
    /// Trailing twelve-month earnings per share (can be negative)
    pub eps_ttm: f64,
    /// Book value per share (shareholders equity / shares)
    pub book_value_per_share: f64,
    /// TTM revenue per share
    pub sales_per_share: f64,
    /// TTM free cash flow per share (operating CF - capex) / shares
    pub fcf_per_share: f64,
    /// TTM dividends per share
    pub dividends_per_share: f64,
    /// Current share price
    pub price: f64,
    /// Total shares outstanding
    pub shares_outstanding: f64,
    /// Total debt
    pub total_debt: f64,
    /// Cash and equivalents
    pub cash: f64,
}

impl FundamentalData {
    /// Enterprise value = market cap + total debt - cash.
    pub fn enterprise_value(&self) -> f64 {
        self.market_cap + self.total_debt - self.cash
    }
}

/// All value factors for a single asset.
#[derive(Debug, Clone)]
pub struct ValueFactors {
    /// Earnings yield = EPS / price (inverse of P/E)
    pub earnings_yield: f64,
    /// Book-to-market ratio = book value / market cap per share
    pub book_to_market: f64,
    /// Sales-to-price = sales per share / price
    pub sales_to_price: f64,
    /// Free cash flow yield = FCF per share / price
    pub fcf_yield: f64,
    /// Dividend yield = dividends / price
    pub dividend_yield: f64,
    /// Composite value score (equal-weight average of z-scored individual factors)
    pub composite_value: f64,
}

/// Compute value factors from fundamental data.
///
/// composite_value is set to NAN here; use `compute_composite_value`
/// on a cross-section of assets to get a meaningful z-scored composite.
pub fn compute_value_factors(data: &FundamentalData) -> Result<ValueFactors> {
    if data.price <= 0.0 {
        return Err(FactorError::InvalidParameter {
            name: "price".into(),
            value: data.price.to_string(),
            constraint: "must be positive".into(),
        });
    }

    let earnings_yield = data.eps_ttm / data.price;

    let book_to_market = if data.price > 1e-10 {
        data.book_value_per_share / data.price
    } else {
        f64::NAN
    };

    let sales_to_price = if data.price > 1e-10 {
        data.sales_per_share / data.price
    } else {
        f64::NAN
    };

    let fcf_yield = data.fcf_per_share / data.price;

    let dividend_yield = if data.price > 1e-10 {
        data.dividends_per_share / data.price
    } else {
        0.0
    };

    Ok(ValueFactors {
        earnings_yield,
        book_to_market,
        sales_to_price,
        fcf_yield,
        dividend_yield,
        composite_value: f64::NAN, // requires cross-sectional z-scoring
    })
}

/// Compute cross-sectional composite value score for a panel.
///
/// For each of the 5 individual value factors, winsorize at 1%/99%,
/// z-score cross-sectionally, then average to form composite.
///
/// Returns Array2 of shape (n_assets, 6): [ey, btm, stp, fcfy, dy, composite].
pub fn compute_panel_value(fundamentals: &[FundamentalData]) -> Result<Array2<f64>> {
    let n = fundamentals.len();
    if n < 5 {
        return Err(FactorError::InsufficientData { required: 5, got: n });
    }

    let mut raw = Array2::<f64>::zeros((n, 5));
    for (i, data) in fundamentals.iter().enumerate() {
        if data.price <= 0.0 {
            for k in 0..5 {
                raw[[i, k]] = f64::NAN;
            }
            continue;
        }
        raw[[i, 0]] = data.eps_ttm / data.price;
        raw[[i, 1]] = data.book_value_per_share / data.price;
        raw[[i, 2]] = data.sales_per_share / data.price;
        raw[[i, 3]] = data.fcf_per_share / data.price;
        raw[[i, 4]] = data.dividends_per_share / data.price;
    }

    let mut result = Array2::<f64>::from_elem((n, 6), f64::NAN);

    // For each factor: winsorize then z-score
    let mut z_scores = Array2::<f64>::from_elem((n, 5), f64::NAN);
    for k in 0..5_usize {
        let col: Vec<f64> = (0..n).map(|i| raw[[i, k]]).collect();
        let zs = winsorize_zscore(&col, 0.01, 0.99);
        for i in 0..n {
            result[[i, k]] = col[i]; // store raw value
            z_scores[[i, k]] = zs[i];
        }
    }

    // Composite = equal-weight average of valid z-scores
    for i in 0..n {
        let mut sum = 0.0;
        let mut count = 0;
        for k in 0..5 {
            let v = z_scores[[i, k]];
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        result[[i, 5]] = if count > 0 { sum / count as f64 } else { f64::NAN };
    }

    Ok(result)
}

/// Winsorize a cross-section at given quantiles, then z-score.
pub fn winsorize_zscore(values: &[f64], lower_q: f64, upper_q: f64) -> Vec<f64> {
    let mut valid: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        return vec![f64::NAN; values.len()];
    }
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = valid.len();

    let lower_idx = ((n as f64) * lower_q).floor() as usize;
    let upper_idx = (((n as f64) * upper_q).ceil() as usize).min(n - 1);
    let lo = valid[lower_idx];
    let hi = valid[upper_idx];

    let winsorized: Vec<f64> = values
        .iter()
        .map(|&v| {
            if !v.is_finite() {
                f64::NAN
            } else {
                v.max(lo).min(hi)
            }
        })
        .collect();

    // Compute mean and std of winsorized finite values
    let ws_valid: Vec<f64> = winsorized.iter().copied().filter(|v| v.is_finite()).collect();
    let mean = ws_valid.iter().sum::<f64>() / ws_valid.len() as f64;
    let var = ws_valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ws_valid.len() as f64;
    let std = var.sqrt();

    if std < 1e-12 {
        return vec![0.0; values.len()];
    }

    winsorized
        .iter()
        .map(|&v| if v.is_finite() { (v - mean) / std } else { f64::NAN })
        .collect()
}

/// Names of the value factor columns.
pub fn value_factor_names() -> Vec<&'static str> {
    vec![
        "earnings_yield",
        "book_to_market",
        "sales_to_price",
        "fcf_yield",
        "dividend_yield",
        "composite_value",
    ]
}

/// Simple valuation metrics derived from enterprise value.
#[derive(Debug, Clone)]
pub struct EnterpriseValueMetrics {
    /// EV/EBIT ratio
    pub ev_to_ebit: f64,
    /// EV/Sales ratio
    pub ev_to_sales: f64,
    /// EV/EBITDA ratio
    pub ev_to_ebitda: f64,
}

/// Compute EV-based valuation metrics.
pub fn compute_ev_metrics(
    data: &FundamentalData,
    ebit: f64,
    ebitda: f64,
    total_sales: f64,
) -> Result<EnterpriseValueMetrics> {
    let ev = data.enterprise_value();
    if ev <= 0.0 {
        return Err(FactorError::InvalidParameter {
            name: "enterprise_value".into(),
            value: ev.to_string(),
            constraint: "must be positive for EV ratios".into(),
        });
    }

    let ev_to_ebit = if ebit.abs() > 1e-10 { ev / ebit } else { f64::NAN };
    let ev_to_sales = if total_sales > 1e-10 { ev / total_sales } else { f64::NAN };
    let ev_to_ebitda = if ebitda.abs() > 1e-10 { ev / ebitda } else { f64::NAN };

    Ok(EnterpriseValueMetrics {
        ev_to_ebit,
        ev_to_sales,
        ev_to_ebitda,
    })
}

/// Rank-based composite: sort each factor, convert to uniform [0,1] ranks, average.
pub fn composite_value_rank(panel: &Array2<f64>, factor_cols: &[usize]) -> Vec<f64> {
    let n = panel.nrows();
    let mut composites = vec![f64::NAN; n];

    // For each selected factor column, compute cross-sectional rank
    let mut rank_sum = vec![0.0f64; n];
    let mut rank_count = vec![0usize; n];

    for &col in factor_cols {
        let vals: Vec<(usize, f64)> = (0..n)
            .filter_map(|i| {
                let v = panel[[i, col]];
                if v.is_finite() {
                    Some((i, v))
                } else {
                    None
                }
            })
            .collect();

        let mut sorted_idx: Vec<usize> = (0..vals.len()).collect();
        sorted_idx.sort_by(|&a, &b| vals[a].1.partial_cmp(&vals[b].1).unwrap());

        let m = sorted_idx.len() as f64;
        for (rank, &idx) in sorted_idx.iter().enumerate() {
            let asset_idx = vals[idx].0;
            let normalized_rank = (rank as f64 + 0.5) / m; // uniform [0,1]
            rank_sum[asset_idx] += normalized_rank;
            rank_count[asset_idx] += 1;
        }
    }

    for i in 0..n {
        if rank_count[i] > 0 {
            composites[i] = rank_sum[i] / rank_count[i] as f64;
        }
    }

    composites
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fundamental(price: f64, eps: f64, bv: f64, sales: f64, fcf: f64, div: f64) -> FundamentalData {
        FundamentalData {
            market_cap: price * 1_000_000.0,
            eps_ttm: eps,
            book_value_per_share: bv,
            sales_per_share: sales,
            fcf_per_share: fcf,
            dividends_per_share: div,
            price,
            shares_outstanding: 1_000_000.0,
            total_debt: 500_000.0,
            cash: 100_000.0,
        }
    }

    #[test]
    fn test_value_factors_basic() {
        let data = make_fundamental(100.0, 5.0, 40.0, 150.0, 8.0, 2.0);
        let f = compute_value_factors(&data).unwrap();
        assert!((f.earnings_yield - 0.05).abs() < 1e-10);
        assert!((f.book_to_market - 0.4).abs() < 1e-10);
        assert!((f.dividend_yield - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_price() {
        let data = make_fundamental(0.0, 5.0, 40.0, 150.0, 8.0, 2.0);
        assert!(matches!(compute_value_factors(&data), Err(FactorError::InvalidParameter { .. })));
    }

    #[test]
    fn test_panel_value() {
        let assets: Vec<FundamentalData> = (0..20)
            .map(|i| make_fundamental(50.0 + i as f64, 2.0 + 0.1 * i as f64, 20.0, 100.0, 5.0, 1.0))
            .collect();
        let panel = compute_panel_value(&assets).unwrap();
        assert_eq!(panel.dim(), (20, 6));
        // Composite should be finite for all valid assets
        for i in 0..20 {
            assert!(panel[[i, 5]].is_finite(), "composite NAN for asset {}", i);
        }
    }

    #[test]
    fn test_winsorize_zscore() {
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let zs = winsorize_zscore(&vals, 0.05, 0.95);
        // After winsorizing, mean should be close to 0
        let mean: f64 = zs.iter().sum::<f64>() / zs.len() as f64;
        assert!(mean.abs() < 1e-10);
    }
}
