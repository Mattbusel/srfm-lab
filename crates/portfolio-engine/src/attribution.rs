/// Brinson-Hood-Beebower (BHB) performance attribution.

// ── BHB Attribution ───────────────────────────────────────────────────────────

/// Result of BHB attribution for a single period.
#[derive(Debug, Clone)]
pub struct BrinsonResult {
    /// Allocation effect: choosing overweight/underweight sectors vs benchmark.
    pub allocation_effect: f64,
    /// Selection effect: within-sector stock picking.
    pub selection_effect: f64,
    /// Interaction effect: joint allocation + selection.
    pub interaction_effect: f64,
    /// Total active return = allocation + selection + interaction.
    pub total_active_return: f64,
    /// Per-sector breakdown.
    pub sector_allocation: Vec<f64>,
    pub sector_selection: Vec<f64>,
    pub sector_interaction: Vec<f64>,
}

/// Brinson-Hood-Beebower attribution.
///
/// # Arguments
/// * `portfolio_weights` — N-vector of portfolio weights.
/// * `benchmark_weights` — N-vector of benchmark weights.
/// * `portfolio_returns` — N-vector of asset returns in the portfolio.
/// * `benchmark_returns` — N-vector of asset returns in the benchmark.
///
/// All vectors must have the same length N (assets or sectors).
pub fn brinson_hood_beebower(
    portfolio_weights: &[f64],
    benchmark_weights: &[f64],
    portfolio_returns: &[f64],
    benchmark_returns: &[f64],
) -> BrinsonResult {
    let n = portfolio_weights.len();
    assert_eq!(benchmark_weights.len(), n);
    assert_eq!(portfolio_returns.len(), n);
    assert_eq!(benchmark_returns.len(), n);

    // Benchmark total return.
    let r_b: f64 = benchmark_weights
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(w, r)| w * r)
        .sum();

    let mut allocation = vec![0.0_f64; n];
    let mut selection = vec![0.0_f64; n];
    let mut interaction = vec![0.0_f64; n];

    for i in 0..n {
        let wp = portfolio_weights[i];
        let wb = benchmark_weights[i];
        let rp = portfolio_returns[i];
        let rb = benchmark_returns[i];

        // Allocation: (w_p - w_b) * (r_b_i - R_B)
        allocation[i] = (wp - wb) * (rb - r_b);
        // Selection: w_b * (r_p_i - r_b_i)
        selection[i] = wb * (rp - rb);
        // Interaction: (w_p - w_b) * (r_p_i - r_b_i)
        interaction[i] = (wp - wb) * (rp - rb);
    }

    let total_allocation: f64 = allocation.iter().sum();
    let total_selection: f64 = selection.iter().sum();
    let total_interaction: f64 = interaction.iter().sum();

    BrinsonResult {
        allocation_effect: total_allocation,
        selection_effect: total_selection,
        interaction_effect: total_interaction,
        total_active_return: total_allocation + total_selection + total_interaction,
        sector_allocation: allocation,
        sector_selection: selection,
        sector_interaction: interaction,
    }
}

// ── Time-Period Attribution ────────────────────────────────────────────────────

/// Period return computed from a slice of the equity curve.
#[derive(Debug, Clone)]
pub struct PeriodReturn {
    pub start_idx: usize,
    pub end_idx: usize,
    pub total_return: f64,
    pub annualised_return: f64,
    pub n_bars: usize,
}

/// Compute returns for each time period defined by (start_idx, end_idx) pairs.
///
/// `equity_curve` is indexed from 0 to T (T+1 values).
pub fn time_period_attribution(
    equity_curve: &[f64],
    periods: &[(usize, usize)],
) -> Vec<PeriodReturn> {
    periods
        .iter()
        .map(|&(start, end)| {
            let start = start.min(equity_curve.len().saturating_sub(1));
            let end = end.min(equity_curve.len().saturating_sub(1));
            let e0 = equity_curve[start].max(1e-12);
            let e1 = equity_curve[end];
            let n_bars = (end - start).max(1);
            let total_ret = (e1 - e0) / e0;
            let ann_ret = (1.0 + total_ret).powf(252.0 / n_bars as f64) - 1.0;
            PeriodReturn {
                start_idx: start,
                end_idx: end,
                total_return: total_ret,
                annualised_return: ann_ret,
                n_bars,
            }
        })
        .collect()
}

// ── Rolling attribution ────────────────────────────────────────────────────────

/// Rolling BHB attribution over a window.
///
/// For each window of `window_bars` bars ending at bar t, compute BHB attribution.
/// Returns a Vec of (bar_index, BrinsonResult).
pub fn rolling_attribution(
    portfolio_weights_series: &[Vec<f64>], // T × N
    benchmark_weights_series: &[Vec<f64>], // T × N
    portfolio_returns_series: &[Vec<f64>], // T × N
    benchmark_returns_series: &[Vec<f64>], // T × N
) -> Vec<BrinsonResult> {
    let t = portfolio_weights_series.len()
        .min(benchmark_weights_series.len())
        .min(portfolio_returns_series.len())
        .min(benchmark_returns_series.len());

    (0..t)
        .map(|i| {
            brinson_hood_beebower(
                &portfolio_weights_series[i],
                &benchmark_weights_series[i],
                &portfolio_returns_series[i],
                &benchmark_returns_series[i],
            )
        })
        .collect()
}

// ── Cumulative Attribution ────────────────────────────────────────────────────

/// Aggregate rolling BHB results into cumulative effects.
pub fn cumulative_attribution(results: &[BrinsonResult]) -> (f64, f64, f64) {
    let total_alloc: f64 = results.iter().map(|r| r.allocation_effect).sum();
    let total_sel: f64 = results.iter().map(|r| r.selection_effect).sum();
    let total_int: f64 = results.iter().map(|r| r.interaction_effect).sum();
    (total_alloc, total_sel, total_int)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bhb_total_matches_active_return() {
        let pw = vec![0.4, 0.6];
        let bw = vec![0.5, 0.5];
        let pr = vec![0.10, 0.05];
        let br = vec![0.08, 0.06];
        let result = brinson_hood_beebower(&pw, &bw, &pr, &br);
        let expected_active = pw.iter().zip(pr.iter()).map(|(w, r)| w * r).sum::<f64>()
            - bw.iter().zip(br.iter()).map(|(w, r)| w * r).sum::<f64>();
        assert!(
            (result.total_active_return - expected_active).abs() < 1e-10,
            "bhb={} expected={}",
            result.total_active_return,
            expected_active
        );
    }
}
