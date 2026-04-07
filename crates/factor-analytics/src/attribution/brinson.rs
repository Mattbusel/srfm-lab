//! Brinson-Hood-Beebower (BHB) performance attribution.
//!
//! Decomposes portfolio active return into:
//! * Allocation effect: over/underweighting sectors vs benchmark
//! * Selection effect: stock selection within sectors
//! * Interaction effect: joint effect of allocation and selection

use ndarray::Array1;
use crate::error::{FactorError, Result};

/// Sector-level attribution inputs.
#[derive(Debug, Clone)]
pub struct SectorAttribution {
    /// Sector name or identifier
    pub sector: String,
    /// Portfolio weight in this sector
    pub portfolio_weight: f64,
    /// Benchmark weight in this sector
    pub benchmark_weight: f64,
    /// Portfolio return within this sector
    pub portfolio_sector_return: f64,
    /// Benchmark return within this sector
    pub benchmark_sector_return: f64,
}

/// BHB attribution results for a single sector.
#[derive(Debug, Clone)]
pub struct SectorAttributionResult {
    pub sector: String,
    /// Allocation effect = (w_p - w_b) * (r_b_sector - r_b_total)
    pub allocation_effect: f64,
    /// Selection effect = w_b * (r_p_sector - r_b_sector)
    pub selection_effect: f64,
    /// Interaction effect = (w_p - w_b) * (r_p_sector - r_b_sector)
    pub interaction_effect: f64,
    /// Total active contribution = allocation + selection + interaction
    pub total_active: f64,
}

/// Full BHB attribution report.
#[derive(Debug, Clone)]
pub struct BrinsonAttributionReport {
    pub sector_results: Vec<SectorAttributionResult>,
    /// Sum of allocation effects
    pub total_allocation: f64,
    /// Sum of selection effects
    pub total_selection: f64,
    /// Sum of interaction effects
    pub total_interaction: f64,
    /// Total active return = portfolio return - benchmark return
    pub total_active_return: f64,
    /// Portfolio total return
    pub portfolio_return: f64,
    /// Benchmark total return
    pub benchmark_return: f64,
    /// Attribution residual (should be ~0 if weights sum to 1)
    pub residual: f64,
}

/// Compute Brinson-Hood-Beebower attribution.
///
/// # Arguments
/// * `sectors` -- attribution inputs by sector
///
/// Returns full BHB attribution report.
pub fn brinson_attribution(sectors: &[SectorAttribution]) -> Result<BrinsonAttributionReport> {
    if sectors.is_empty() {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    // Validate weights sum to ~1
    let portfolio_weight_sum: f64 = sectors.iter().map(|s| s.portfolio_weight).sum();
    let benchmark_weight_sum: f64 = sectors.iter().map(|s| s.benchmark_weight).sum();

    if (portfolio_weight_sum - 1.0).abs() > 0.05 {
        return Err(FactorError::InvalidParameter {
            name: "portfolio_weights_sum".into(),
            value: portfolio_weight_sum.to_string(),
            constraint: "must sum to approximately 1.0 (tolerance 0.05)".into(),
        });
    }

    // Total portfolio and benchmark returns (weighted average)
    let portfolio_return: f64 = sectors
        .iter()
        .map(|s| s.portfolio_weight * s.portfolio_sector_return)
        .sum();
    let benchmark_return: f64 = sectors
        .iter()
        .map(|s| s.benchmark_weight * s.benchmark_sector_return)
        .sum();

    let total_active_return = portfolio_return - benchmark_return;

    let mut sector_results = Vec::with_capacity(sectors.len());
    let mut total_allocation = 0.0;
    let mut total_selection = 0.0;
    let mut total_interaction = 0.0;

    for s in sectors {
        let w_p = s.portfolio_weight;
        let w_b = s.benchmark_weight;
        let r_p = s.portfolio_sector_return;
        let r_b = s.benchmark_sector_return;

        // BHB decomposition
        // Allocation: over/underweight * (sector bench return - total bench return)
        let allocation_effect = (w_p - w_b) * (r_b - benchmark_return);

        // Selection: benchmark weight * (portfolio sector return - benchmark sector return)
        let selection_effect = w_b * (r_p - r_b);

        // Interaction: active weight * active return within sector
        let interaction_effect = (w_p - w_b) * (r_p - r_b);

        let total_active = allocation_effect + selection_effect + interaction_effect;

        total_allocation += allocation_effect;
        total_selection += selection_effect;
        total_interaction += interaction_effect;

        sector_results.push(SectorAttributionResult {
            sector: s.sector.clone(),
            allocation_effect,
            selection_effect,
            interaction_effect,
            total_active,
        });
    }

    let residual = total_active_return - (total_allocation + total_selection + total_interaction);

    Ok(BrinsonAttributionReport {
        sector_results,
        total_allocation,
        total_selection,
        total_interaction,
        total_active_return,
        portfolio_return,
        benchmark_return,
        residual,
    })
}

/// Compute BHB attribution from raw position data.
///
/// Aggregates to sector level before running BHB.
///
/// # Arguments
/// * `portfolio_weights` -- weight of each asset in portfolio
/// * `benchmark_weights` -- weight of each asset in benchmark
/// * `asset_returns` -- return of each asset over the period
/// * `sector_ids` -- sector assignment for each asset
pub fn brinson_from_positions(
    portfolio_weights: &[f64],
    benchmark_weights: &[f64],
    asset_returns: &[f64],
    sector_ids: &[usize],
) -> Result<BrinsonAttributionReport> {
    let n = portfolio_weights.len();
    if benchmark_weights.len() != n || asset_returns.len() != n || sector_ids.len() != n {
        return Err(FactorError::DimensionMismatch {
            expected: n,
            got: benchmark_weights.len().min(asset_returns.len()),
        });
    }

    // Collect unique sectors
    let mut unique_sectors: Vec<usize> = sector_ids.to_vec();
    unique_sectors.sort_unstable();
    unique_sectors.dedup();

    let mut sector_inputs = Vec::with_capacity(unique_sectors.len());

    for &sector in &unique_sectors {
        let indices: Vec<usize> = (0..n).filter(|&i| sector_ids[i] == sector).collect();

        let portfolio_sector_weight: f64 = indices.iter().map(|&i| portfolio_weights[i]).sum();
        let benchmark_sector_weight: f64 = indices.iter().map(|&i| benchmark_weights[i]).sum();

        // Portfolio return in sector = weighted average of asset returns by portfolio weights
        let portfolio_sector_return = if portfolio_sector_weight > 1e-10 {
            indices
                .iter()
                .map(|&i| portfolio_weights[i] * asset_returns[i])
                .sum::<f64>()
                / portfolio_sector_weight
        } else {
            0.0
        };

        // Benchmark return in sector
        let benchmark_sector_return = if benchmark_sector_weight > 1e-10 {
            indices
                .iter()
                .map(|&i| benchmark_weights[i] * asset_returns[i])
                .sum::<f64>()
                / benchmark_sector_weight
        } else {
            0.0
        };

        sector_inputs.push(SectorAttribution {
            sector: format!("sector_{}", sector),
            portfolio_weight: portfolio_sector_weight,
            benchmark_weight: benchmark_sector_weight,
            portfolio_sector_return,
            benchmark_sector_return,
        });
    }

    brinson_attribution(&sector_inputs)
}

/// Time-aggregated BHB attribution.
///
/// Aggregates attribution results across multiple periods using geometric linking.
///
/// # Arguments
/// * `period_reports` -- BHB reports for each time period
///
/// Returns a summary with geometrically linked effects.
pub fn aggregate_attribution(period_reports: &[BrinsonAttributionReport]) -> Result<BrinsonAttributionReport> {
    if period_reports.is_empty() {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    let n = period_reports.len() as f64;

    // For simplicity, use arithmetic average of effects
    // (True geometric linking requires Brinson-Carino or Menchero methodology)
    let total_allocation = period_reports.iter().map(|r| r.total_allocation).sum::<f64>();
    let total_selection = period_reports.iter().map(|r| r.total_selection).sum::<f64>();
    let total_interaction = period_reports.iter().map(|r| r.total_interaction).sum::<f64>();

    // Compound portfolio and benchmark returns
    let portfolio_return = period_reports
        .iter()
        .map(|r| 1.0 + r.portfolio_return)
        .product::<f64>()
        - 1.0;
    let benchmark_return = period_reports
        .iter()
        .map(|r| 1.0 + r.benchmark_return)
        .product::<f64>()
        - 1.0;

    let total_active_return = portfolio_return - benchmark_return;
    let residual = total_active_return - (total_allocation + total_selection + total_interaction);

    // Aggregate sector-level results (sum across periods)
    // Collect unique sector names
    let mut all_sectors: Vec<String> = period_reports
        .iter()
        .flat_map(|r| r.sector_results.iter().map(|s| s.sector.clone()))
        .collect();
    all_sectors.sort();
    all_sectors.dedup();

    let sector_results: Vec<SectorAttributionResult> = all_sectors
        .iter()
        .map(|sector_name| {
            let alloc: f64 = period_reports
                .iter()
                .flat_map(|r| r.sector_results.iter())
                .filter(|s| &s.sector == sector_name)
                .map(|s| s.allocation_effect)
                .sum();
            let sel: f64 = period_reports
                .iter()
                .flat_map(|r| r.sector_results.iter())
                .filter(|s| &s.sector == sector_name)
                .map(|s| s.selection_effect)
                .sum();
            let inter: f64 = period_reports
                .iter()
                .flat_map(|r| r.sector_results.iter())
                .filter(|s| &s.sector == sector_name)
                .map(|s| s.interaction_effect)
                .sum();
            SectorAttributionResult {
                sector: sector_name.clone(),
                allocation_effect: alloc,
                selection_effect: sel,
                interaction_effect: inter,
                total_active: alloc + sel + inter,
            }
        })
        .collect();

    Ok(BrinsonAttributionReport {
        sector_results,
        total_allocation,
        total_selection,
        total_interaction,
        total_active_return,
        portfolio_return,
        benchmark_return,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sectors() -> Vec<SectorAttribution> {
        vec![
            SectorAttribution {
                sector: "Technology".into(),
                portfolio_weight: 0.30,
                benchmark_weight: 0.25,
                portfolio_sector_return: 0.15,
                benchmark_sector_return: 0.12,
            },
            SectorAttribution {
                sector: "Financials".into(),
                portfolio_weight: 0.20,
                benchmark_weight: 0.22,
                portfolio_sector_return: 0.08,
                benchmark_sector_return: 0.07,
            },
            SectorAttribution {
                sector: "Healthcare".into(),
                portfolio_weight: 0.25,
                benchmark_weight: 0.25,
                portfolio_sector_return: 0.10,
                benchmark_sector_return: 0.09,
            },
            SectorAttribution {
                sector: "Energy".into(),
                portfolio_weight: 0.10,
                benchmark_weight: 0.15,
                portfolio_sector_return: -0.05,
                benchmark_sector_return: -0.03,
            },
            SectorAttribution {
                sector: "ConsumerStaples".into(),
                portfolio_weight: 0.15,
                benchmark_weight: 0.13,
                portfolio_sector_return: 0.06,
                benchmark_sector_return: 0.05,
            },
        ]
    }

    #[test]
    fn test_brinson_attribution_basic() {
        let sectors = make_sectors();
        let report = brinson_attribution(&sectors).unwrap();

        // Sum of effects should approximate total active return
        let sum_effects = report.total_allocation + report.total_selection + report.total_interaction;
        assert!(
            (sum_effects - report.total_active_return).abs() < 1e-10,
            "Attribution doesn't sum: effects={:.6}, active={:.6}",
            sum_effects,
            report.total_active_return
        );

        assert!(report.residual.abs() < 1e-10);
    }

    #[test]
    fn test_brinson_selection_effect() {
        // Single sector: overweight in outperforming sector
        let sectors = vec![
            SectorAttribution {
                sector: "A".into(),
                portfolio_weight: 0.6,
                benchmark_weight: 0.6,
                portfolio_sector_return: 0.10,
                benchmark_sector_return: 0.08,
            },
            SectorAttribution {
                sector: "B".into(),
                portfolio_weight: 0.4,
                benchmark_weight: 0.4,
                portfolio_sector_return: 0.05,
                benchmark_sector_return: 0.05,
            },
        ];
        let report = brinson_attribution(&sectors).unwrap();
        // No allocation (same weights), positive selection in A
        assert!(report.total_allocation.abs() < 1e-10);
        assert!(report.total_selection > 0.0);
    }

    #[test]
    fn test_brinson_from_positions() {
        let n = 20;
        // Portfolio: equal weight summing to 1.0
        let portfolio_weights: Vec<f64> = vec![1.0 / n as f64; n];
        // Benchmark: also equal weight summing to 1.0
        let benchmark_weights: Vec<f64> = vec![1.0 / n as f64; n];
        let asset_returns: Vec<f64> = (0..n).map(|i| 0.01 * i as f64).collect();
        let sector_ids: Vec<usize> = (0..n).map(|i| i / 5).collect(); // 4 sectors

        let report = brinson_from_positions(
            &portfolio_weights,
            &benchmark_weights,
            &asset_returns,
            &sector_ids,
        ).unwrap();

        assert_eq!(report.sector_results.len(), 4);
        let sum = report.total_allocation + report.total_selection + report.total_interaction;
        assert!((sum - report.total_active_return).abs() < 1e-10);
    }
}
