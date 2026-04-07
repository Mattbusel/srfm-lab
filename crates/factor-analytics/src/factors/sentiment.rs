//! Sentiment factors: short interest ratio, analyst revision momentum,
//! and earnings surprise persistence.

use ndarray::Array2;
use crate::error::{FactorError, Result};

/// Short interest data for a single asset.
#[derive(Debug, Clone)]
pub struct ShortInterestData {
    /// Short interest (shares sold short) -- most recent first or last (see compute fn)
    pub short_interest_history: Vec<f64>,
    /// Float (shares available for trading)
    pub float_shares: f64,
    /// Average daily trading volume (for days-to-cover)
    pub avg_daily_volume: f64,
}

/// Analyst estimate data for a single asset.
#[derive(Debug, Clone)]
pub struct AnalystData {
    /// EPS consensus estimates -- one per revision period (monthly), most recent last
    pub eps_estimates: Vec<f64>,
    /// Number of analyst upgrades in recent period
    pub upgrades: u32,
    /// Number of analyst downgrades in recent period
    pub downgrades: u32,
    /// Price target consensus history (most recent last)
    pub price_targets: Vec<f64>,
    /// Current price for normalization
    pub current_price: f64,
}

/// Earnings surprise data.
#[derive(Debug, Clone)]
pub struct EarningsSurpriseData {
    /// Actual EPS for each quarter (most recent last)
    pub actual_eps: Vec<f64>,
    /// Consensus estimate EPS for each quarter (most recent last)
    pub estimated_eps: Vec<f64>,
}

/// All sentiment factors for a single asset.
#[derive(Debug, Clone)]
pub struct SentimentFactors {
    /// Short interest ratio = short_interest / float (most recent)
    pub short_interest_ratio: f64,
    /// Days to cover = short_interest / avg_daily_volume
    pub days_to_cover: f64,
    /// Change in short interest ratio (current vs 3m ago) -- higher = more bearish
    pub short_change_3m: f64,
    /// Analyst revision momentum: (upgrades - downgrades) / total analysts
    pub revision_momentum: f64,
    /// EPS estimate revision: (current estimate - 3m ago estimate) / |3m ago estimate|
    pub eps_revision_3m: f64,
    /// Price target implied return: (target - price) / price
    pub pt_implied_return: f64,
    /// Earnings surprise persistence: average standardized surprise (SUE) over recent quarters
    pub sue_persistence: f64,
    /// Most recent SUE (standardized unexpected earnings)
    pub sue_recent: f64,
}

/// Compute standardized unexpected earnings (SUE).
///
/// SUE_t = (Actual_t - Estimate_t) / |Estimate_t| if Estimate_t != 0
/// else (Actual_t - Estimate_t) / std(historical surprises)
pub fn compute_sue(actual: f64, estimated: f64, historical_std: f64) -> f64 {
    let surprise = actual - estimated;
    let denom = if estimated.abs() > 1e-10 {
        estimated.abs()
    } else if historical_std > 1e-10 {
        historical_std
    } else {
        return f64::NAN;
    };
    surprise / denom
}

/// Compute all sentiment factors for a single asset.
pub fn compute_sentiment_factors(
    short_data: &ShortInterestData,
    analyst_data: &AnalystData,
    earnings_data: &EarningsSurpriseData,
) -> Result<SentimentFactors> {
    // --- Short interest ---
    let n_si = short_data.short_interest_history.len();
    if n_si == 0 {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    let current_si = short_data.short_interest_history[n_si - 1];
    let short_interest_ratio = if short_data.float_shares > 1e-6 {
        current_si / short_data.float_shares
    } else {
        f64::NAN
    };

    let days_to_cover = if short_data.avg_daily_volume > 1e-6 {
        current_si / short_data.avg_daily_volume
    } else {
        f64::NAN
    };

    // 3-month change in short interest (roughly 6 bi-weekly reports back)
    let lookback_si = 3.min(n_si - 1);
    let short_change_3m = if n_si > lookback_si && short_data.float_shares > 1e-6 {
        let old_sir = short_data.short_interest_history[n_si - 1 - lookback_si]
            / short_data.float_shares;
        short_interest_ratio - old_sir
    } else {
        f64::NAN
    };

    // --- Analyst revisions ---
    let total_analysts = analyst_data.upgrades + analyst_data.downgrades;
    let revision_momentum = if total_analysts > 0 {
        (analyst_data.upgrades as f64 - analyst_data.downgrades as f64)
            / total_analysts as f64
    } else {
        0.0
    };

    let n_est = analyst_data.eps_estimates.len();
    let eps_revision_3m = if n_est >= 4 {
        let old_est = analyst_data.eps_estimates[n_est - 4];
        let new_est = analyst_data.eps_estimates[n_est - 1];
        if old_est.abs() > 1e-10 {
            (new_est - old_est) / old_est.abs()
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    };

    let n_pt = analyst_data.price_targets.len();
    let pt_implied_return = if n_pt > 0 && analyst_data.current_price > 1e-10 {
        let target = analyst_data.price_targets[n_pt - 1];
        (target - analyst_data.current_price) / analyst_data.current_price
    } else {
        f64::NAN
    };

    // --- Earnings surprise persistence ---
    let n_earn = earnings_data.actual_eps.len().min(earnings_data.estimated_eps.len());
    if n_earn == 0 {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    // Compute historical std of surprises for normalization
    let surprises: Vec<f64> = (0..n_earn)
        .map(|i| earnings_data.actual_eps[i] - earnings_data.estimated_eps[i])
        .collect();
    let mean_surprise = surprises.iter().sum::<f64>() / n_earn as f64;
    let std_surprise = if n_earn >= 2 {
        (surprises.iter().map(|s| (s - mean_surprise).powi(2)).sum::<f64>()
            / (n_earn - 1) as f64)
            .sqrt()
    } else {
        1.0
    };

    let sue_recent = compute_sue(
        earnings_data.actual_eps[n_earn - 1],
        earnings_data.estimated_eps[n_earn - 1],
        std_surprise,
    );

    // Use up to last 4 quarters for persistence
    let lookback_earn = n_earn.min(4);
    let sue_values: Vec<f64> = (n_earn - lookback_earn..n_earn)
        .map(|i| {
            compute_sue(
                earnings_data.actual_eps[i],
                earnings_data.estimated_eps[i],
                std_surprise,
            )
        })
        .filter(|v| v.is_finite())
        .collect();

    let sue_persistence = if sue_values.is_empty() {
        f64::NAN
    } else {
        sue_values.iter().sum::<f64>() / sue_values.len() as f64
    };

    Ok(SentimentFactors {
        short_interest_ratio,
        days_to_cover,
        short_change_3m,
        revision_momentum,
        eps_revision_3m,
        pt_implied_return,
        sue_persistence,
        sue_recent,
    })
}

/// Compute cross-sectional sentiment factor panel.
///
/// Returns Array2 of shape (n_assets, 8).
/// Sign convention for each factor:
/// * short_interest_ratio: negative for long (high short interest = bearish)
/// * days_to_cover: negative
/// * short_change_3m: negative
/// * revision_momentum, eps_revision_3m, pt_implied_return, sue_*: positive
pub fn compute_panel_sentiment(
    short_data: &[ShortInterestData],
    analyst_data: &[AnalystData],
    earnings_data: &[EarningsSurpriseData],
) -> Result<Array2<f64>> {
    let n = short_data.len();
    if n == 0 {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }
    if analyst_data.len() != n || earnings_data.len() != n {
        return Err(FactorError::DimensionMismatch {
            expected: n,
            got: analyst_data.len().min(earnings_data.len()),
        });
    }

    let mut result = Array2::<f64>::from_elem((n, 8), f64::NAN);

    for i in 0..n {
        match compute_sentiment_factors(&short_data[i], &analyst_data[i], &earnings_data[i]) {
            Ok(f) => {
                result[[i, 0]] = f.short_interest_ratio;
                result[[i, 1]] = f.days_to_cover;
                result[[i, 2]] = f.short_change_3m;
                result[[i, 3]] = f.revision_momentum;
                result[[i, 4]] = f.eps_revision_3m;
                result[[i, 5]] = f.pt_implied_return;
                result[[i, 6]] = f.sue_persistence;
                result[[i, 7]] = f.sue_recent;
            }
            Err(_) => {}
        }
    }

    Ok(result)
}

/// Names of sentiment factor columns.
pub fn sentiment_factor_names() -> Vec<&'static str> {
    vec![
        "short_interest_ratio",
        "days_to_cover",
        "short_change_3m",
        "revision_momentum",
        "eps_revision_3m",
        "pt_implied_return",
        "sue_persistence",
        "sue_recent",
    ]
}

/// Compute analyst dispersion: std of individual estimates / |mean estimate|.
///
/// High dispersion = high uncertainty = negative signal.
pub fn analyst_dispersion(estimates: &[f64]) -> f64 {
    let n = estimates.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = estimates.iter().sum::<f64>() / n as f64;
    let std = (estimates.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    if mean.abs() > 1e-10 {
        std / mean.abs()
    } else {
        f64::NAN
    }
}

/// Compute earnings announcement drift proxy.
///
/// Returns the average return in the 3 days following each announcement
/// (Post-Earnings Announcement Drift -- PEAD).
pub fn compute_pead(
    announcement_day_returns: &[f64],
    post_announcement_returns_3d: &[f64],
) -> f64 {
    let n = announcement_day_returns.len().min(post_announcement_returns_3d.len());
    if n == 0 {
        return f64::NAN;
    }

    let mut drift_sum = 0.0;
    let mut count = 0;

    for i in 0..n {
        let sign = announcement_day_returns[i].signum();
        let post_ret = post_announcement_returns_3d[i];
        if post_ret.is_finite() {
            drift_sum += sign * post_ret;
            count += 1;
        }
    }

    if count == 0 {
        f64::NAN
    } else {
        drift_sum / count as f64
    }
}

/// Short squeeze potential score.
///
/// Based on: high short interest, low days-to-cover,
/// recent stock momentum (requires external momentum data).
pub fn short_squeeze_score(
    short_interest_ratio: f64,
    days_to_cover: f64,
    recent_return_1m: f64,
) -> f64 {
    let mut score = 0.0;
    let mut components = 0;

    if short_interest_ratio.is_finite() {
        score += short_interest_ratio * 10.0; // scale up SIR
        components += 1;
    }
    // Penalize if days_to_cover is high (hard to cover quickly)
    if days_to_cover.is_finite() && days_to_cover > 0.0 {
        score += 1.0 / days_to_cover;
        components += 1;
    }
    // Momentum adds to squeeze potential
    if recent_return_1m.is_finite() {
        score += recent_return_1m.max(0.0) * 5.0;
        components += 1;
    }

    if components == 0 {
        f64::NAN
    } else {
        score / components as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (ShortInterestData, AnalystData, EarningsSurpriseData) {
        let si = ShortInterestData {
            short_interest_history: vec![
                5_000_000.0,
                5_500_000.0,
                6_000_000.0,
                5_800_000.0,
            ],
            float_shares: 100_000_000.0,
            avg_daily_volume: 2_000_000.0,
        };

        let analyst = AnalystData {
            eps_estimates: vec![1.20, 1.25, 1.30, 1.35, 1.40, 1.42],
            upgrades: 5,
            downgrades: 2,
            price_targets: vec![150.0, 155.0, 160.0],
            current_price: 140.0,
        };

        let earnings = EarningsSurpriseData {
            actual_eps: vec![1.22, 1.28, 1.35, 1.41],
            estimated_eps: vec![1.20, 1.25, 1.30, 1.38],
        };

        (si, analyst, earnings)
    }

    #[test]
    fn test_sentiment_factors() {
        let (si, analyst, earnings) = make_test_data();
        let f = compute_sentiment_factors(&si, &analyst, &earnings).unwrap();

        assert!((f.short_interest_ratio - 0.058).abs() < 0.001);
        assert!(f.days_to_cover > 0.0);
        assert!(f.revision_momentum > 0.0); // more upgrades than downgrades
        assert!(f.sue_recent > 0.0); // beat estimate
        assert!(f.pt_implied_return > 0.0); // target above current price
    }

    #[test]
    fn test_sue_computation() {
        let sue = compute_sue(1.35, 1.30, 0.05);
        assert!((sue - (1.35 - 1.30) / 1.30).abs() < 1e-10);
    }

    #[test]
    fn test_analyst_dispersion() {
        let estimates = vec![1.20, 1.25, 1.30, 1.35, 1.40];
        let disp = analyst_dispersion(&estimates);
        assert!(disp > 0.0 && disp < 1.0);
    }

    #[test]
    fn test_pead() {
        let ann_returns = vec![0.03, -0.02, 0.04, -0.01];
        let post_returns = vec![0.01, -0.005, 0.008, -0.002];
        let pead = compute_pead(&ann_returns, &post_returns);
        // Positive surprise days: returns 0.03, 0.04 -> post_ret positive avg
        assert!(pead.is_finite());
    }

    #[test]
    fn test_panel_sentiment() {
        let n = 10;
        let (si_templ, an_templ, ea_templ) = make_test_data();
        let si_vec: Vec<_> = (0..n).map(|_| si_templ.clone()).collect();
        let an_vec: Vec<_> = (0..n).map(|_| an_templ.clone()).collect();
        let ea_vec: Vec<_> = (0..n).map(|_| ea_templ.clone()).collect();

        let panel = compute_panel_sentiment(&si_vec, &an_vec, &ea_vec).unwrap();
        assert_eq!(panel.dim(), (n, 8));
    }
}
