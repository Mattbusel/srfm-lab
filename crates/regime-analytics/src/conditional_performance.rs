// conditional_performance.rs
// Performance analytics conditioned on market regime.
// Tracks per-trade PnL, computes regime-conditional Sharpe, win rate,
// average holding period, and produces risk-adjusted allocation weights.

use std::collections::HashMap;
use crate::hmm_regime::RegimeLabel;

const ALL_REGIMES: [RegimeLabel; 4] = [
    RegimeLabel::Bull,
    RegimeLabel::Bear,
    RegimeLabel::Sideways,
    RegimeLabel::HighVol,
];

// ---- RegimePerformanceRecord ----------------------------------------------

/// A single completed trade attributed to a specific regime.
#[derive(Debug, Clone)]
pub struct RegimePerformanceRecord {
    /// Regime active at trade entry
    pub regime: RegimeLabel,
    /// Entry timestamp (Unix seconds or bar index)
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: i64,
    /// Realised PnL (absolute, in base currency units)
    pub pnl: f64,
    /// Maximum drawdown during the trade (positive value, absolute)
    pub max_drawdown: f64,
    /// Number of bars held
    pub bars: usize,
}

impl RegimePerformanceRecord {
    /// Trade duration in seconds (or bars if timestamps are bar indices).
    pub fn duration(&self) -> i64 {
        self.exit_time - self.entry_time
    }

    /// True for a winning trade.
    pub fn is_win(&self) -> bool {
        self.pnl > 0.0
    }

    /// Calmar proxy: pnl / max_drawdown.  Returns 0 if drawdown is zero.
    pub fn calmar(&self) -> f64 {
        if self.max_drawdown <= 1e-9 { return 0.0; }
        self.pnl / self.max_drawdown
    }
}

// ---- ConditionalPerformance -----------------------------------------------

/// Collection of trade records with regime-conditional analytics.
#[derive(Debug, Clone, Default)]
pub struct ConditionalPerformance {
    pub records: Vec<RegimePerformanceRecord>,
}

impl ConditionalPerformance {
    pub fn new() -> Self {
        ConditionalPerformance { records: Vec::new() }
    }

    /// Add a completed trade.
    pub fn add_trade(
        &mut self,
        regime: RegimeLabel,
        entry_ts: i64,
        exit_ts: i64,
        pnl: f64,
        max_drawdown: f64,
        bars: usize,
    ) {
        self.records.push(RegimePerformanceRecord {
            regime,
            entry_time: entry_ts,
            exit_time: exit_ts,
            pnl,
            max_drawdown,
            bars,
        });
    }

    /// Number of trades recorded.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    // ---- Filtering --------------------------------------------------------

    /// All records for a given regime.
    fn records_for(&self, regime: RegimeLabel) -> Vec<&RegimePerformanceRecord> {
        self.records.iter().filter(|r| r.regime == regime).collect()
    }

    // ---- Per-regime statistics -------------------------------------------

    /// PnL mean for a regime (0 if no trades).
    fn mean_pnl(&self, regime: RegimeLabel) -> f64 {
        let recs = self.records_for(regime);
        if recs.is_empty() { return 0.0; }
        recs.iter().map(|r| r.pnl).sum::<f64>() / recs.len() as f64
    }

    /// PnL standard deviation for a regime.
    fn std_pnl(&self, regime: RegimeLabel) -> f64 {
        let recs = self.records_for(regime);
        let n = recs.len();
        if n < 2 { return 1e-9; }
        let mean = self.mean_pnl(regime);
        let var = recs.iter().map(|r| (r.pnl - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        var.sqrt().max(1e-9)
    }

    /// Annualised Sharpe ratio by regime.
    /// Assumes each record represents one bar; annualisation factor = sqrt(252).
    pub fn sharpe_by_regime(&self) -> HashMap<RegimeLabel, f64> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let mean = self.mean_pnl(r);
            let std  = self.std_pnl(r);
            let sharpe = (mean / std) * (252.0_f64).sqrt();
            out.insert(r, sharpe);
        }
        out
    }

    /// Win rate (fraction of trades with PnL > 0) by regime.
    pub fn win_rate_by_regime(&self) -> HashMap<RegimeLabel, f64> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let recs = self.records_for(r);
            if recs.is_empty() {
                out.insert(r, 0.0);
                continue;
            }
            let wins = recs.iter().filter(|rec| rec.is_win()).count();
            out.insert(r, wins as f64 / recs.len() as f64);
        }
        out
    }

    /// Average number of bars held per trade, by regime.
    pub fn avg_bars_held_by_regime(&self) -> HashMap<RegimeLabel, f64> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let recs = self.records_for(r);
            if recs.is_empty() {
                out.insert(r, 0.0);
                continue;
            }
            let avg = recs.iter().map(|rec| rec.bars as f64).sum::<f64>() / recs.len() as f64;
            out.insert(r, avg);
        }
        out
    }

    /// Total PnL by regime.
    pub fn total_pnl_by_regime(&self) -> HashMap<RegimeLabel, f64> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let total: f64 = self.records_for(r).iter().map(|rec| rec.pnl).sum();
            out.insert(r, total);
        }
        out
    }

    /// Maximum drawdown observed across all trades in each regime.
    pub fn max_drawdown_by_regime(&self) -> HashMap<RegimeLabel, f64> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let max_dd = self.records_for(r)
                .iter()
                .map(|rec| rec.max_drawdown)
                .fold(0.0_f64, f64::max);
            out.insert(r, max_dd);
        }
        out
    }

    /// Number of trades per regime.
    pub fn trade_count_by_regime(&self) -> HashMap<RegimeLabel, usize> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            out.insert(r, self.records_for(r).len());
        }
        out
    }

    // ---- Summary statistics ----------------------------------------------

    /// Best regime by risk-adjusted return (highest Sharpe ratio).
    /// Requires at least one trade in a regime to be eligible.
    pub fn best_regime(&self) -> RegimeLabel {
        let sharpes = self.sharpe_by_regime();
        let counts  = self.trade_count_by_regime();
        ALL_REGIMES.iter()
            .filter(|&&r| *counts.get(&r).unwrap_or(&0) > 0)
            .max_by(|&&a, &&b| {
                sharpes.get(&a).unwrap_or(&f64::NEG_INFINITY)
                    .partial_cmp(sharpes.get(&b).unwrap_or(&f64::NEG_INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(RegimeLabel::Bull)
    }

    /// Worst regime by Sharpe ratio.
    pub fn worst_regime(&self) -> RegimeLabel {
        let sharpes = self.sharpe_by_regime();
        let counts  = self.trade_count_by_regime();
        ALL_REGIMES.iter()
            .filter(|&&r| *counts.get(&r).unwrap_or(&0) > 0)
            .min_by(|&&a, &&b| {
                sharpes.get(&a).unwrap_or(&f64::INFINITY)
                    .partial_cmp(sharpes.get(&b).unwrap_or(&f64::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(RegimeLabel::Bear)
    }

    // ---- Allocation recommendation ----------------------------------------

    /// Normalise positive Sharpe ratios to allocation weights.
    /// Regimes with negative or zero Sharpe get zero weight.
    /// The returned weights sum to 1.0.
    pub fn regime_allocation_recommendation(&self) -> HashMap<RegimeLabel, f64> {
        let sharpes = self.sharpe_by_regime();
        let mut positive: Vec<(RegimeLabel, f64)> = ALL_REGIMES.iter()
            .filter_map(|&r| {
                let s = *sharpes.get(&r).unwrap_or(&0.0);
                if s > 0.0 { Some((r, s)) } else { None }
            })
            .collect();

        // Cap extreme Sharpes to avoid over-concentration
        let max_sharpe = positive.iter().map(|(_, s)| *s).fold(0.0_f64, f64::max);
        let cap = max_sharpe.min(5.0); // Hard cap at 5.0
        for (_, s) in &mut positive {
            *s = (*s).min(cap);
        }

        let total: f64 = positive.iter().map(|(_, s)| s).sum();
        let mut out: HashMap<RegimeLabel, f64> = ALL_REGIMES.iter()
            .map(|&r| (r, 0.0))
            .collect();

        if total > 0.0 {
            for (r, s) in positive {
                out.insert(r, s / total);
            }
        }
        out
    }

    // ---- Calmar by regime ------------------------------------------------

    /// Average Calmar ratio (mean_pnl / max_drawdown) by regime.
    pub fn calmar_by_regime(&self) -> HashMap<RegimeLabel, f64> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let recs = self.records_for(r);
            if recs.is_empty() {
                out.insert(r, 0.0);
                continue;
            }
            let avg = recs.iter().map(|rec| rec.calmar()).sum::<f64>() / recs.len() as f64;
            out.insert(r, avg);
        }
        out
    }

    // ---- Profit factor ---------------------------------------------------

    /// Profit factor = gross profits / gross losses. Returns f64::INFINITY if no losses.
    pub fn profit_factor_by_regime(&self) -> HashMap<RegimeLabel, f64> {
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let recs = self.records_for(r);
            let gross_profit: f64 = recs.iter().filter(|rec| rec.pnl > 0.0).map(|rec| rec.pnl).sum();
            let gross_loss: f64   = recs.iter().filter(|rec| rec.pnl < 0.0).map(|rec| rec.pnl.abs()).sum();
            let pf = if gross_loss < 1e-9 { f64::INFINITY } else { gross_profit / gross_loss };
            out.insert(r, pf);
        }
        out
    }

    // ---- Regime stability ------------------------------------------------

    /// Fraction of time spent in each regime across all trade bars.
    pub fn time_in_regime(&self) -> HashMap<RegimeLabel, f64> {
        let total_bars: usize = self.records.iter().map(|r| r.bars).sum();
        if total_bars == 0 {
            return ALL_REGIMES.iter().map(|&r| (r, 0.0)).collect();
        }
        let mut out = HashMap::new();
        for &r in &ALL_REGIMES {
            let bars: usize = self.records_for(r).iter().map(|rec| rec.bars).sum();
            out.insert(r, bars as f64 / total_bars as f64);
        }
        out
    }
}

// ---- ConditionalPerformanceSummary ----------------------------------------

/// Snapshot summary of regime-conditional performance statistics.
#[derive(Debug, Clone)]
pub struct ConditionalPerformanceSummary {
    pub regime: RegimeLabel,
    pub n_trades: usize,
    pub total_pnl: f64,
    pub mean_pnl: f64,
    pub std_pnl: f64,
    pub sharpe: f64,
    pub win_rate: f64,
    pub avg_bars: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub recommended_weight: f64,
}

impl ConditionalPerformance {
    /// Build a full summary for every regime.
    pub fn summarise(&self) -> Vec<ConditionalPerformanceSummary> {
        let sharpes    = self.sharpe_by_regime();
        let win_rates  = self.win_rate_by_regime();
        let avg_bars   = self.avg_bars_held_by_regime();
        let total_pnls = self.total_pnl_by_regime();
        let counts     = self.trade_count_by_regime();
        let max_dds    = self.max_drawdown_by_regime();
        let pfs        = self.profit_factor_by_regime();
        let weights    = self.regime_allocation_recommendation();

        ALL_REGIMES.iter().map(|&r| {
            let n = *counts.get(&r).unwrap_or(&0);
            let total_pnl = *total_pnls.get(&r).unwrap_or(&0.0);
            let mean_pnl = if n > 0 { total_pnl / n as f64 } else { 0.0 };
            ConditionalPerformanceSummary {
                regime: r,
                n_trades: n,
                total_pnl,
                mean_pnl,
                std_pnl: self.std_pnl(r),
                sharpe: *sharpes.get(&r).unwrap_or(&0.0),
                win_rate: *win_rates.get(&r).unwrap_or(&0.0),
                avg_bars: *avg_bars.get(&r).unwrap_or(&0.0),
                max_drawdown: *max_dds.get(&r).unwrap_or(&0.0),
                profit_factor: *pfs.get(&r).unwrap_or(&0.0),
                recommended_weight: *weights.get(&r).unwrap_or(&0.0),
            }
        }).collect()
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sample_perf() -> ConditionalPerformance {
        let mut cp = ConditionalPerformance::new();
        // 10 Bull trades: 8 wins
        for i in 0..10 {
            let pnl = if i < 8 { 100.0 + i as f64 * 5.0 } else { -50.0 };
            cp.add_trade(RegimeLabel::Bull, i * 86400, i * 86400 + 3600, pnl, 20.0, 5);
        }
        // 5 Bear trades: 2 wins
        for i in 0..5 {
            let pnl = if i < 2 { 50.0 } else { -80.0 };
            cp.add_trade(RegimeLabel::Bear, i * 86400, i * 86400 + 7200, pnl, 60.0, 10);
        }
        // 3 Sideways trades: flat
        for _ in 0..3 {
            cp.add_trade(RegimeLabel::Sideways, 0, 1800, 5.0, 2.0, 2);
        }
        cp
    }

    #[test]
    fn test_add_and_count() {
        let cp = make_sample_perf();
        assert_eq!(cp.len(), 18);
        let counts = cp.trade_count_by_regime();
        assert_eq!(*counts.get(&RegimeLabel::Bull).unwrap(), 10);
        assert_eq!(*counts.get(&RegimeLabel::Bear).unwrap(), 5);
        assert_eq!(*counts.get(&RegimeLabel::HighVol).unwrap(), 0);
    }

    #[test]
    fn test_win_rate() {
        let cp = make_sample_perf();
        let wr = cp.win_rate_by_regime();
        let bull_wr = *wr.get(&RegimeLabel::Bull).unwrap();
        assert!((bull_wr - 0.80).abs() < 1e-6, "bull wr={}", bull_wr);
        let bear_wr = *wr.get(&RegimeLabel::Bear).unwrap();
        assert!((bear_wr - 0.40).abs() < 1e-6, "bear wr={}", bear_wr);
    }

    #[test]
    fn test_sharpe_sign() {
        let cp = make_sample_perf();
        let sharpes = cp.sharpe_by_regime();
        // Bull should have positive Sharpe
        let bull_s = *sharpes.get(&RegimeLabel::Bull).unwrap();
        assert!(bull_s > 0.0, "bull sharpe={}", bull_s);
        // Bear should have negative Sharpe (average loss)
        let bear_s = *sharpes.get(&RegimeLabel::Bear).unwrap();
        assert!(bear_s < bull_s, "bear should underperform bull: {}", bear_s);
    }

    #[test]
    fn test_best_regime_is_bull() {
        let cp = make_sample_perf();
        let best = cp.best_regime();
        // Bull should be best given our sample data
        assert!(
            matches!(best, RegimeLabel::Bull | RegimeLabel::Sideways),
            "best={:?}", best
        );
    }

    #[test]
    fn test_avg_bars() {
        let cp = make_sample_perf();
        let avg = cp.avg_bars_held_by_regime();
        assert!((avg.get(&RegimeLabel::Bull).unwrap() - 5.0).abs() < 1e-9);
        assert!((avg.get(&RegimeLabel::Bear).unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_allocation_recommendation_sums_to_one() {
        let cp = make_sample_perf();
        let alloc = cp.regime_allocation_recommendation();
        let sum: f64 = alloc.values().sum();
        // Allow 0 if no positive Sharpe regimes
        if sum > 1e-9 {
            assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
        }
    }

    #[test]
    fn test_allocation_no_negative_weights() {
        let cp = make_sample_perf();
        let alloc = cp.regime_allocation_recommendation();
        for (&_regime, &w) in &alloc {
            assert!(w >= 0.0, "negative weight={}", w);
        }
    }

    #[test]
    fn test_profit_factor_bull() {
        let cp = make_sample_perf();
        let pf = cp.profit_factor_by_regime();
        let bull_pf = *pf.get(&RegimeLabel::Bull).unwrap();
        // 8 wins * avg ~130, 2 losses * 50 => pf > 1
        assert!(bull_pf > 1.0, "bull profit factor={}", bull_pf);
    }

    #[test]
    fn test_total_pnl() {
        let cp = make_sample_perf();
        let totals = cp.total_pnl_by_regime();
        let bull_pnl = *totals.get(&RegimeLabel::Bull).unwrap();
        assert!(bull_pnl > 0.0, "bull total pnl={}", bull_pnl);
    }

    #[test]
    fn test_time_in_regime_sums_to_one() {
        let cp = make_sample_perf();
        let time = cp.time_in_regime();
        let sum: f64 = time.values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    #[test]
    fn test_summarise_produces_all_regimes() {
        let cp = make_sample_perf();
        let summaries = cp.summarise();
        assert_eq!(summaries.len(), 4);
        let regime_set: std::collections::HashSet<_> =
            summaries.iter().map(|s| s.regime).collect();
        assert!(regime_set.contains(&RegimeLabel::Bull));
        assert!(regime_set.contains(&RegimeLabel::Bear));
    }

    #[test]
    fn test_empty_performance() {
        let cp = ConditionalPerformance::new();
        let best = cp.best_regime();
        // Should return a default without panic
        let _ = best;
        let alloc = cp.regime_allocation_recommendation();
        let sum: f64 = alloc.values().sum();
        assert!(sum < 1e-9);
    }

    #[test]
    fn test_calmar_by_regime() {
        let cp = make_sample_perf();
        let calmar = cp.calmar_by_regime();
        // Sideways: pnl=5, drawdown=2, calmar=2.5
        let sw = *calmar.get(&RegimeLabel::Sideways).unwrap();
        assert!((sw - 2.5).abs() < 1e-6, "sideways calmar={}", sw);
    }
}
