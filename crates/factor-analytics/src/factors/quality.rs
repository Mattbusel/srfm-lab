//! Quality factors: ROE, ROA, gross margin stability, earnings variability,
//! accruals ratio, and leverage ratio.

use ndarray::Array2;
use crate::error::{FactorError, Result};

/// Financial statement data for quality factor computation.
/// All values are per-share unless noted.
#[derive(Debug, Clone)]
pub struct FinancialStatements {
    /// Return series: net income for each period (trailing, most recent last)
    pub net_income: Vec<f64>,
    /// Shareholders equity for each period
    pub equity: Vec<f64>,
    /// Total assets for each period
    pub total_assets: Vec<f64>,
    /// Revenue/sales for each period
    pub revenue: Vec<f64>,
    /// Cost of goods sold for each period
    pub cogs: Vec<f64>,
    /// Total debt (most recent)
    pub total_debt: f64,
    /// Cash from operations (most recent TTM)
    pub operating_cash_flow: f64,
    /// Change in net working capital (most recent)
    pub delta_working_capital: f64,
    /// Depreciation and amortization (most recent)
    pub depreciation: f64,
}

/// Quality factors for a single asset.
#[derive(Debug, Clone)]
pub struct QualityFactors {
    /// Return on equity = net income / avg equity (most recent period)
    pub roe: f64,
    /// Return on assets = net income / avg total assets
    pub roa: f64,
    /// Gross margin = (revenue - COGS) / revenue
    pub gross_margin: f64,
    /// Gross margin stability = 1 / std(gross margins over periods)
    pub gross_margin_stability: f64,
    /// Earnings variability = coefficient of variation of net income
    pub earnings_variability: f64,
    /// Accruals ratio = (net income - operating cash flow) / avg total assets
    pub accruals_ratio: f64,
    /// Leverage ratio = total debt / equity (most recent)
    pub leverage_ratio: f64,
    /// Composite quality score (z-scored, set after cross-section)
    pub composite_quality: f64,
}

/// Compute quality factors for a single asset.
///
/// # Arguments
/// * `stmts` -- financial statement history (at least 4 periods recommended)
pub fn compute_quality_factors(stmts: &FinancialStatements) -> Result<QualityFactors> {
    let n = stmts.net_income.len();
    if n < 2 {
        return Err(FactorError::InsufficientData { required: 2, got: n });
    }

    // ROE -- use average equity of last two periods
    let avg_equity = (stmts.equity[n - 1] + stmts.equity[n - 2]) / 2.0;
    let roe = if avg_equity.abs() > 1e-10 {
        stmts.net_income[n - 1] / avg_equity
    } else {
        f64::NAN
    };

    // ROA -- use average total assets of last two periods
    let avg_assets = (stmts.total_assets[n - 1] + stmts.total_assets[n - 2]) / 2.0;
    let roa = if avg_assets.abs() > 1e-10 {
        stmts.net_income[n - 1] / avg_assets
    } else {
        f64::NAN
    };

    // Gross margins over all periods
    let gross_margins: Vec<f64> = stmts
        .revenue
        .iter()
        .zip(stmts.cogs.iter())
        .map(|(rev, cogs)| {
            if rev.abs() > 1e-10 {
                (rev - cogs) / rev
            } else {
                f64::NAN
            }
        })
        .collect();

    let valid_gm: Vec<f64> = gross_margins.iter().copied().filter(|v| v.is_finite()).collect();
    let gross_margin = if valid_gm.is_empty() {
        f64::NAN
    } else {
        *valid_gm.last().unwrap()
    };

    let gross_margin_stability = if valid_gm.len() >= 2 {
        let mean_gm = valid_gm.iter().sum::<f64>() / valid_gm.len() as f64;
        let std_gm = (valid_gm.iter().map(|v| (v - mean_gm).powi(2)).sum::<f64>()
            / (valid_gm.len() - 1) as f64)
            .sqrt();
        if std_gm > 1e-10 {
            1.0 / std_gm // higher = more stable
        } else {
            f64::INFINITY // constant margin -- maximally stable
        }
    } else {
        f64::NAN
    };

    // Earnings variability -- coefficient of variation of net income
    let ni = &stmts.net_income;
    let mean_ni = ni.iter().sum::<f64>() / n as f64;
    let std_ni = if n >= 2 {
        (ni.iter().map(|v| (v - mean_ni).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        f64::NAN
    };
    let earnings_variability = if mean_ni.abs() > 1e-10 {
        std_ni / mean_ni.abs()
    } else {
        f64::NAN
    };

    // Accruals ratio: measures quality of earnings
    // Sloan (1996): accruals = net income - operating cash flow
    // Normalized by average total assets
    let accruals = stmts.net_income[n - 1] - stmts.operating_cash_flow;
    let accruals_ratio = if avg_assets.abs() > 1e-10 {
        accruals / avg_assets
    } else {
        f64::NAN
    };

    // Leverage ratio: debt / equity
    let leverage_ratio = if stmts.equity[n - 1].abs() > 1e-10 {
        stmts.total_debt / stmts.equity[n - 1]
    } else {
        f64::NAN
    };

    Ok(QualityFactors {
        roe,
        roa,
        gross_margin,
        gross_margin_stability,
        earnings_variability,
        accruals_ratio,
        leverage_ratio,
        composite_quality: f64::NAN,
    })
}

/// Compute cross-sectional quality factor scores.
///
/// Signs adjusted so that higher = better quality:
/// * ROE, ROA, gross_margin, gross_margin_stability: positive is better
/// * earnings_variability, accruals_ratio, leverage_ratio: negative is better (inverted)
///
/// Returns Array2 of shape (n_assets, 8): [roe, roa, gm, gm_stab, ev, accruals, leverage, composite].
pub fn compute_panel_quality(statements: &[FinancialStatements]) -> Result<Array2<f64>> {
    let n = statements.len();
    if n < 5 {
        return Err(FactorError::InsufficientData { required: 5, got: n });
    }

    let mut raw = Array2::<f64>::from_elem((n, 7), f64::NAN);

    for (i, stmts) in statements.iter().enumerate() {
        match compute_quality_factors(stmts) {
            Ok(f) => {
                raw[[i, 0]] = f.roe;
                raw[[i, 1]] = f.roa;
                raw[[i, 2]] = f.gross_margin;
                raw[[i, 3]] = if f.gross_margin_stability.is_infinite() {
                    10.0 // cap for numerical stability
                } else {
                    f.gross_margin_stability
                };
                // Invert so higher = better quality
                raw[[i, 4]] = if f.earnings_variability.is_finite() {
                    -f.earnings_variability
                } else {
                    f64::NAN
                };
                raw[[i, 5]] = if f.accruals_ratio.is_finite() {
                    -f.accruals_ratio
                } else {
                    f64::NAN
                };
                raw[[i, 6]] = if f.leverage_ratio.is_finite() {
                    -f.leverage_ratio
                } else {
                    f64::NAN
                };
            }
            Err(_) => {} // leave as NAN
        }
    }

    let mut result = Array2::<f64>::from_elem((n, 8), f64::NAN);
    let mut z_scores = Array2::<f64>::from_elem((n, 7), f64::NAN);

    for k in 0..7 {
        let col: Vec<f64> = (0..n).map(|i| raw[[i, k]]).collect();
        let zs = crate::factors::value::winsorize_zscore(&col, 0.01, 0.99);
        for i in 0..n {
            result[[i, k]] = raw[[i, k]];
            z_scores[[i, k]] = zs[i];
        }
    }

    // Composite: equal-weight average of z-scores
    for i in 0..n {
        let mut sum = 0.0;
        let mut count = 0;
        for k in 0..7 {
            let v = z_scores[[i, k]];
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        result[[i, 7]] = if count > 0 { sum / count as f64 } else { f64::NAN };
    }

    Ok(result)
}

/// Names of quality factor columns in panel output.
pub fn quality_factor_names() -> Vec<&'static str> {
    vec![
        "roe",
        "roa",
        "gross_margin",
        "gross_margin_stability",
        "earnings_variability_neg",
        "accruals_ratio_neg",
        "leverage_ratio_neg",
        "composite_quality",
    ]
}

/// Piotroski F-Score: binary quality signals (0 or 1 each), summed 0--9.
///
/// Signals (from Piotroski 2000):
/// Profitability: ROA > 0, Delta ROA > 0, CFO > 0, Accruals < 0
/// Leverage/Liquidity: Delta Leverage < 0, Delta Liquidity > 0, No new shares
/// Operating efficiency: Delta Gross Margin > 0, Delta Asset Turnover > 0
#[derive(Debug, Clone)]
pub struct PiotroskiFScore {
    pub roa_positive: bool,
    pub delta_roa_positive: bool,
    pub cfo_positive: bool,
    pub accruals_negative: bool,
    pub delta_leverage_negative: bool,
    pub delta_liquidity_positive: bool,
    pub no_dilution: bool,
    pub delta_gross_margin_positive: bool,
    pub delta_asset_turnover_positive: bool,
    pub total_score: u8,
}

/// Compute Piotroski F-Score from two consecutive periods of financial data.
pub fn compute_piotroski(
    current: &FinancialStatements,
    prior: &FinancialStatements,
) -> Result<PiotroskiFScore> {
    let n_curr = current.net_income.len();
    let n_prior = prior.net_income.len();
    if n_curr < 1 || n_prior < 1 {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    let curr_assets = current.total_assets[n_curr - 1];
    let prior_assets = prior.total_assets[n_prior - 1];
    let curr_equity = current.equity[n_curr - 1];
    let prior_equity = prior.equity[n_prior - 1];

    let curr_roa = if curr_assets > 1e-10 {
        current.net_income[n_curr - 1] / curr_assets
    } else {
        f64::NAN
    };
    let prior_roa = if prior_assets > 1e-10 {
        prior.net_income[n_prior - 1] / prior_assets
    } else {
        f64::NAN
    };

    let curr_leverage = if curr_equity.abs() > 1e-10 {
        current.total_debt / curr_equity
    } else {
        f64::NAN
    };
    let prior_leverage = if prior_equity.abs() > 1e-10 {
        prior.total_debt / prior_equity
    } else {
        f64::NAN
    };

    // Current ratio proxy: current assets / current liabilities
    // We approximate with equity / total_debt as liquidity proxy
    let curr_liq = if current.total_debt > 1e-10 {
        curr_equity / current.total_debt
    } else {
        10.0 // no debt -- highly liquid
    };
    let prior_liq = if prior.total_debt > 1e-10 {
        prior_equity / prior.total_debt
    } else {
        10.0
    };

    let curr_rev = *current.revenue.last().unwrap_or(&0.0);
    let prior_rev = *prior.revenue.last().unwrap_or(&0.0);
    let curr_cogs = *current.cogs.last().unwrap_or(&0.0);
    let prior_cogs = *prior.cogs.last().unwrap_or(&0.0);

    let curr_gm = if curr_rev > 1e-10 { (curr_rev - curr_cogs) / curr_rev } else { f64::NAN };
    let prior_gm = if prior_rev > 1e-10 { (prior_rev - prior_cogs) / prior_rev } else { f64::NAN };

    let curr_turnover = if curr_assets > 1e-10 { curr_rev / curr_assets } else { f64::NAN };
    let prior_turnover = if prior_assets > 1e-10 { prior_rev / prior_assets } else { f64::NAN };

    let curr_accruals = if curr_assets > 1e-10 {
        (current.net_income[n_curr - 1] - current.operating_cash_flow) / curr_assets
    } else {
        f64::NAN
    };

    // Shares dilution: use equity as proxy (increased equity without income = dilution)
    let net_income_curr = current.net_income[n_curr - 1];
    let equity_delta = curr_equity - prior_equity;
    let no_dilution = equity_delta <= net_income_curr + 1e-10; // no excess equity issuance

    let roa_positive = curr_roa.is_finite() && curr_roa > 0.0;
    let delta_roa_positive = curr_roa.is_finite() && prior_roa.is_finite() && curr_roa > prior_roa;
    let cfo_positive = current.operating_cash_flow > 0.0;
    let accruals_negative = curr_accruals.is_finite() && curr_accruals < 0.0;
    let delta_leverage_negative =
        curr_leverage.is_finite() && prior_leverage.is_finite() && curr_leverage < prior_leverage;
    let delta_liquidity_positive = curr_liq > prior_liq;
    let delta_gross_margin_positive =
        curr_gm.is_finite() && prior_gm.is_finite() && curr_gm > prior_gm;
    let delta_asset_turnover_positive =
        curr_turnover.is_finite() && prior_turnover.is_finite() && curr_turnover > prior_turnover;

    let total_score = [
        roa_positive,
        delta_roa_positive,
        cfo_positive,
        accruals_negative,
        delta_leverage_negative,
        delta_liquidity_positive,
        no_dilution,
        delta_gross_margin_positive,
        delta_asset_turnover_positive,
    ]
    .iter()
    .filter(|&&b| b)
    .count() as u8;

    Ok(PiotroskiFScore {
        roa_positive,
        delta_roa_positive,
        cfo_positive,
        accruals_negative,
        delta_leverage_negative,
        delta_liquidity_positive,
        no_dilution,
        delta_gross_margin_positive,
        delta_asset_turnover_positive,
        total_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stmts(n_periods: usize, profitable: bool) -> FinancialStatements {
        let sign = if profitable { 1.0 } else { -1.0 };
        FinancialStatements {
            net_income: vec![sign * 100.0; n_periods],
            equity: (0..n_periods).map(|i| 1000.0 + i as f64 * 10.0).collect(),
            total_assets: (0..n_periods).map(|i| 5000.0 + i as f64 * 20.0).collect(),
            revenue: vec![2000.0; n_periods],
            cogs: vec![1200.0; n_periods],
            total_debt: 500.0,
            operating_cash_flow: sign * 120.0,
            delta_working_capital: 10.0,
            depreciation: 50.0,
        }
    }

    #[test]
    fn test_quality_factors_profitable() {
        let stmts = make_stmts(4, true);
        let f = compute_quality_factors(&stmts).unwrap();
        assert!(f.roe > 0.0);
        assert!(f.roa > 0.0);
        assert!(f.gross_margin > 0.0);
        // accruals_ratio should be negative (cash flow > net income)
        assert!(f.accruals_ratio < 0.0);
    }

    #[test]
    fn test_piotroski_high_quality() {
        let curr = make_stmts(4, true);
        let mut prior = make_stmts(4, true);
        prior.net_income = vec![80.0; 4]; // current has higher ROA
        let score = compute_piotroski(&curr, &prior).unwrap();
        assert!(score.total_score >= 5);
    }

    #[test]
    fn test_panel_quality() {
        let stmts: Vec<FinancialStatements> = (0..15)
            .map(|i| make_stmts(4, i % 3 != 0))
            .collect();
        let panel = compute_panel_quality(&stmts).unwrap();
        assert_eq!(panel.dim(), (15, 8));
    }
}
