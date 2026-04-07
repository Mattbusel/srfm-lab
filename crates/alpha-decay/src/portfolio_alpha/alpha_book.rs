// alpha_book.rs
// Portfolio of alpha signals: track alpha, beta, IR for each.
// Allocate research capacity proportional to IR.
// Flag overcrowded signals (high pairwise correlation).

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::pearson_corr;

/// Crowding flag for a signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrowdingFlag {
    pub signal_name: String,
    /// Average pairwise IC correlation with all other signals in the book.
    pub avg_pairwise_corr: f64,
    /// True if average correlation exceeds the crowding threshold.
    pub is_crowded: bool,
    /// Which other signals it is most correlated with.
    pub correlated_with: Vec<(String, f64)>,
}

/// Entry for a single alpha signal in the book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphaSignalEntry {
    pub name: String,
    /// Running mean alpha (daily).
    pub mean_alpha: f64,
    /// Running std of daily alphas.
    pub std_alpha: f64,
    /// Information ratio = mean_alpha / std_alpha.
    pub ir: f64,
    /// Correlation of signal with market factor (beta proxy).
    pub market_beta: f64,
    /// Number of observations.
    pub n_obs: usize,
    /// Research allocation fraction (0 to 1).
    pub research_allocation: f64,
    /// Current crowding flag.
    pub crowding: Option<CrowdingFlag>,
    /// Rolling IC history used for correlation analysis.
    pub ic_history: Vec<f64>,
    /// Daily alpha P&L history.
    pub alpha_history: Vec<f64>,
}

impl AlphaSignalEntry {
    pub fn new(name: String) -> Self {
        AlphaSignalEntry {
            name,
            mean_alpha: 0.0,
            std_alpha: 0.0,
            ir: 0.0,
            market_beta: 0.0,
            n_obs: 0,
            research_allocation: 0.0,
            crowding: None,
            ic_history: Vec::new(),
            alpha_history: Vec::new(),
        }
    }

    /// Update alpha statistics given a new daily alpha observation and market return.
    pub fn push_alpha(&mut self, alpha: f64, _market_return: f64) {
        self.alpha_history.push(alpha);
        let n = self.alpha_history.len() as f64;
        self.mean_alpha = self.alpha_history.iter().sum::<f64>() / n;
        let var = self.alpha_history.iter().map(|v| (v - self.mean_alpha).powi(2)).sum::<f64>() / n;
        self.std_alpha = var.sqrt().max(1e-10);
        self.ir = self.mean_alpha / self.std_alpha;
        self.n_obs = self.alpha_history.len();
    }

    /// Push an IC observation.
    pub fn push_ic(&mut self, ic: f64) {
        self.ic_history.push(ic);
        if self.ic_history.len() > 252 {
            self.ic_history.remove(0);
        }
    }

    /// Compute market beta from alpha history vs market returns.
    pub fn compute_market_beta(&mut self, market_returns: &[f64]) {
        let n = self.alpha_history.len().min(market_returns.len());
        if n < 10 {
            return;
        }
        let alphas = &self.alpha_history[self.alpha_history.len() - n..];
        let mkt = &market_returns[market_returns.len() - n..];
        let (beta, _) = crate::ols_simple(mkt, alphas);
        self.market_beta = beta;
    }

    /// Rolling IR over the last `window` observations.
    pub fn rolling_ir(&self, window: usize) -> f64 {
        let n = self.alpha_history.len();
        if n < 2 {
            return 0.0;
        }
        let start = n.saturating_sub(window);
        let slice = &self.alpha_history[start..];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let std = (slice.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / slice.len() as f64)
            .sqrt()
            .max(1e-10);
        mean / std
    }

    /// Annualized IR assuming daily observations.
    pub fn annualized_ir(&self) -> f64 {
        self.ir * (252.0f64).sqrt()
    }
}

/// Book of all alpha signals.
pub struct AlphaBook {
    signals: HashMap<String, AlphaSignalEntry>,
    /// Crowding threshold: average pairwise IC correlation above this is flagged.
    crowding_threshold: f64,
    /// Minimum IR to receive positive research allocation.
    min_ir_threshold: f64,
}

impl AlphaBook {
    pub fn new(crowding_threshold: f64, min_ir_threshold: f64) -> Self {
        AlphaBook {
            signals: HashMap::new(),
            crowding_threshold,
            min_ir_threshold,
        }
    }

    /// Register a new alpha signal.
    pub fn register(&mut self, name: impl Into<String>) {
        let n = name.into();
        self.signals.entry(n.clone()).or_insert_with(|| AlphaSignalEntry::new(n));
    }

    /// Update a signal with a new alpha observation.
    pub fn update_alpha(&mut self, name: &str, alpha: f64, market_return: f64) {
        if let Some(entry) = self.signals.get_mut(name) {
            entry.push_alpha(alpha, market_return);
        }
    }

    /// Update a signal with a new IC observation.
    pub fn update_ic(&mut self, name: &str, ic: f64) {
        if let Some(entry) = self.signals.get_mut(name) {
            entry.push_ic(ic);
        }
    }

    /// Recompute research allocations proportional to IR^+ (positive IR only).
    /// Signals with IR < min_ir_threshold get 0 allocation.
    pub fn recompute_allocations(&mut self) {
        let ir_pos: HashMap<String, f64> = self
            .signals
            .iter()
            .map(|(name, entry)| {
                let ir = entry.ir;
                (name.clone(), if ir >= self.min_ir_threshold { ir } else { 0.0 })
            })
            .collect();
        let total_ir: f64 = ir_pos.values().sum();
        let n_signals = self.signals.len();
        for (name, entry) in self.signals.iter_mut() {
            entry.research_allocation = if total_ir > 1e-14 {
                ir_pos[name] / total_ir
            } else {
                1.0 / n_signals as f64
            };
        }
    }

    /// Compute pairwise IC correlations and flag overcrowded signals.
    pub fn recompute_crowding(&mut self) {
        let names: Vec<String> = self.signals.keys().cloned().collect();
        let n = names.len();
        if n < 2 {
            return;
        }

        // Build pairwise correlation matrix.
        let mut corr_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            corr_matrix[i][i] = 1.0;
            let hi = self.signals[&names[i]].ic_history.clone();
            for j in (i + 1)..n {
                let hj = self.signals[&names[j]].ic_history.clone();
                let min_len = hi.len().min(hj.len());
                if min_len < 5 {
                    corr_matrix[i][j] = 0.0;
                    corr_matrix[j][i] = 0.0;
                    continue;
                }
                let xi = &hi[hi.len() - min_len..];
                let xj = &hj[hj.len() - min_len..];
                let c = pearson_corr(xi, xj);
                corr_matrix[i][j] = c;
                corr_matrix[j][i] = c;
            }
        }

        // For each signal compute average pairwise correlation.
        for (i, name) in names.iter().enumerate() {
            let avg_corr = if n > 1 {
                let sum: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| corr_matrix[i][j].abs())
                    .sum();
                sum / (n - 1) as f64
            } else {
                0.0
            };

            let mut correlated_with: Vec<(String, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (names[j].clone(), corr_matrix[i][j]))
                .collect();
            correlated_with
                .sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
            correlated_with.truncate(3);

            if let Some(entry) = self.signals.get_mut(name) {
                entry.crowding = Some(CrowdingFlag {
                    signal_name: name.clone(),
                    avg_pairwise_corr: avg_corr,
                    is_crowded: avg_corr > self.crowding_threshold,
                    correlated_with,
                });
            }
        }
    }

    /// Get signals sorted by IR descending.
    pub fn signals_by_ir(&self) -> Vec<&AlphaSignalEntry> {
        let mut entries: Vec<&AlphaSignalEntry> = self.signals.values().collect();
        entries.sort_by(|a, b| b.ir.partial_cmp(&a.ir).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }

    /// Get crowded signals.
    pub fn crowded_signals(&self) -> Vec<&AlphaSignalEntry> {
        self.signals
            .values()
            .filter(|e| e.crowding.as_ref().map(|c| c.is_crowded).unwrap_or(false))
            .collect()
    }

    /// Get a specific signal entry.
    pub fn get(&self, name: &str) -> Option<&AlphaSignalEntry> {
        self.signals.get(name)
    }

    /// Total number of signals in the book.
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// Portfolio-level statistics: total IR assuming zero correlation (ideal).
    pub fn portfolio_ir_ideal(&self) -> f64 {
        let sum_ir_sq: f64 = self.signals.values().map(|e| e.ir.powi(2)).sum();
        sum_ir_sq.sqrt()
    }

    /// Active signals (research_allocation > 0).
    pub fn active_signals(&self) -> Vec<&AlphaSignalEntry> {
        self.signals
            .values()
            .filter(|e| e.research_allocation > 1e-6)
            .collect()
    }

    /// Remove a signal from the book.
    pub fn remove(&mut self, name: &str) -> Option<AlphaSignalEntry> {
        self.signals.remove(name)
    }

    /// Generate a summary report as a vector of formatted strings.
    pub fn summary_report(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!("AlphaBook Summary: {} signals", self.signals.len()));
        lines.push(format!("Portfolio IR (ideal): {:.3}", self.portfolio_ir_ideal()));
        lines.push(String::from(""));
        lines.push(format!(
            "{:<20} {:>8} {:>8} {:>8} {:>12} {:>8}",
            "Signal", "MeanAlpha", "StdAlpha", "IR", "Alloc%", "Crowded"
        ));
        for entry in self.signals_by_ir() {
            let crowded = entry
                .crowding
                .as_ref()
                .map(|c| if c.is_crowded { "YES" } else { "no" })
                .unwrap_or("?");
            lines.push(format!(
                "{:<20} {:>8.5} {:>8.5} {:>8.3} {:>12.1} {:>8}",
                entry.name,
                entry.mean_alpha,
                entry.std_alpha,
                entry.ir,
                entry.research_allocation * 100.0,
                crowded
            ));
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_book() -> AlphaBook {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(11);
        let mut book = AlphaBook::new(0.6, 0.1);

        for i in 0..5 {
            let name = format!("alpha_{}", i);
            book.register(&name);
            let base_alpha = 0.001 + i as f64 * 0.001;
            for _ in 0..100 {
                let alpha = base_alpha + rng.gen::<f64>() * 0.005 - 0.0025;
                let mkt = rng.gen::<f64>() * 0.02 - 0.01;
                book.update_alpha(&name, alpha, mkt);
                book.update_ic(&name, 0.05 + rng.gen::<f64>() * 0.10);
            }
        }
        book.recompute_allocations();
        book.recompute_crowding();
        book
    }

    #[test]
    fn test_allocations_sum_to_one() {
        let book = make_book();
        let total: f64 = book.signals.values().map(|e| e.research_allocation).sum();
        assert!((total - 1.0).abs() < 1e-9, "Allocations should sum to 1: {}", total);
    }

    #[test]
    fn test_highest_ir_gets_most_allocation() {
        let book = make_book();
        let by_ir = book.signals_by_ir();
        assert!(
            by_ir[0].research_allocation >= by_ir[by_ir.len() - 1].research_allocation,
            "Highest IR signal should have highest allocation"
        );
    }

    #[test]
    fn test_crowding_flags() {
        let book = make_book();
        // Just ensure crowding flags are populated.
        for entry in book.signals.values() {
            assert!(entry.crowding.is_some(), "Crowding should be computed for all signals");
        }
    }
}
