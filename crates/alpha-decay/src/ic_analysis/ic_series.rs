// ic_series.rs
// Rolling Information Coefficient (IC) computation.
// IC = Spearman rank correlation of predicted vs realized returns.
// Computes IC at multiple horizons: 1-bar, 5-bar, 20-bar, 60-bar.
// Newey-West HAC t-statistics for IC significance testing.

use crate::{spearman_rank_corr, newey_west_variance, newey_west_lags, MIN_IC_OBS};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Forecast horizon in bars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IcHorizon {
    H1,   // 1-bar ahead
    H5,   // 5-bar ahead
    H20,  // 20-bar ahead
    H60,  // 60-bar ahead
}

impl IcHorizon {
    pub fn bars(&self) -> usize {
        match self {
            IcHorizon::H1 => 1,
            IcHorizon::H5 => 5,
            IcHorizon::H20 => 20,
            IcHorizon::H60 => 60,
        }
    }

    pub fn all() -> [IcHorizon; 4] {
        [IcHorizon::H1, IcHorizon::H5, IcHorizon::H20, IcHorizon::H60]
    }
}

/// A single IC observation at a given bar index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcObservation {
    pub bar_index: usize,
    pub horizon: IcHorizon,
    pub ic: f64,
    /// Number of cross-section assets used.
    pub n_assets: usize,
}

/// Summary statistics for an IC series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcStats {
    pub horizon: IcHorizon,
    pub mean_ic: f64,
    pub std_ic: f64,
    /// IC information ratio = mean / std.
    pub icir: f64,
    /// Newey-West t-statistic for H0: mean_IC = 0.
    pub t_stat: f64,
    pub p_value: f64,
    /// Fraction of observations where IC > 0.
    pub hit_rate: f64,
    pub n_obs: usize,
}

impl IcStats {
    /// Two-tailed p-value from standard normal approximation.
    fn normal_pvalue(t: f64) -> f64 {
        // Using a simple rational approximation to the normal CDF.
        let z = t.abs();
        let p = if z > 10.0 {
            0.0
        } else {
            let a1 = 0.254829592f64;
            let a2 = -0.284496736f64;
            let a3 = 1.421413741f64;
            let a4 = -1.453152027f64;
            let a5 = 1.061405429f64;
            let p_val = 0.3275911f64;
            let x = 1.0 / (1.0 + p_val * z / std::f64::consts::SQRT_2);
            let poly = ((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x;
            poly * (-z * z / 2.0).exp()
        };
        // Two-tailed
        2.0 * p
    }

    pub fn compute(horizon: IcHorizon, ic_values: &[f64]) -> Self {
        let n = ic_values.len();
        if n == 0 {
            return IcStats {
                horizon,
                mean_ic: 0.0,
                std_ic: 0.0,
                icir: 0.0,
                t_stat: 0.0,
                p_value: 1.0,
                hit_rate: 0.0,
                n_obs: 0,
            };
        }
        let mean_ic = ic_values.iter().sum::<f64>() / n as f64;
        let variance = ic_values.iter().map(|v| (v - mean_ic).powi(2)).sum::<f64>() / n as f64;
        let std_ic = variance.sqrt();
        let icir = if std_ic > 1e-10 { mean_ic / std_ic } else { 0.0 };

        // Newey-West t-stat
        let lags = newey_west_lags(n);
        let nw_var = newey_west_variance(ic_values, lags);
        let se = (nw_var / n as f64).sqrt();
        let t_stat = if se > 1e-14 { mean_ic / se } else { 0.0 };
        let p_value = Self::normal_pvalue(t_stat);

        let hit_rate = ic_values.iter().filter(|&&v| v > 0.0).count() as f64 / n as f64;

        IcStats {
            horizon,
            mean_ic,
            std_ic,
            icir,
            t_stat,
            p_value,
            hit_rate,
            n_obs: n,
        }
    }
}

/// Cross-section observation at a single bar: signal values and future returns.
#[derive(Debug, Clone)]
struct CrossSection {
    bar_index: usize,
    signals: Vec<f64>,
    /// Realized future returns (filled in later when horizon elapses).
    realized: Option<Vec<f64>>,
}

/// Rolling IC computation engine.
///
/// Usage:
/// 1. Push signal vectors at each bar with `push_signals`.
/// 2. Push realized return vectors when the horizon elapses with `push_realized`.
/// 3. Query IC series with `ic_series_for` or statistics with `ic_stats`.
///
/// Internally maintains a rolling window of `window` IC observations.
pub struct IcSeries {
    /// Rolling window size for IC computation.
    window: usize,
    /// Pending cross-sections awaiting realized returns.
    pending: VecDeque<CrossSection>,
    /// Stored IC observations per horizon.
    observations: std::collections::HashMap<IcHorizon, VecDeque<IcObservation>>,
    /// Current bar counter.
    bar: usize,
}

impl IcSeries {
    /// Create a new IcSeries with rolling window size.
    pub fn new(window: usize) -> Self {
        let mut observations = std::collections::HashMap::new();
        for h in IcHorizon::all() {
            observations.insert(h, VecDeque::new());
        }
        IcSeries {
            window: window.max(MIN_IC_OBS),
            pending: VecDeque::new(),
            observations,
            bar: 0,
        }
    }

    /// Push a cross-section of signal scores at the current bar.
    /// `signals` has one entry per asset in the universe.
    pub fn push_signals(&mut self, signals: Vec<f64>) {
        self.pending.push_back(CrossSection {
            bar_index: self.bar,
            signals,
            realized: None,
        });
        // Trim pending beyond the longest horizon we need + buffer.
        while self.pending.len() > 200 {
            self.pending.pop_front();
        }
        self.bar += 1;
    }

    /// Push realized returns for the cross-section that occurred `horizon.bars()` ago.
    /// `returns` must match the asset count of the signals pushed `h` bars ago.
    pub fn push_realized(&mut self, horizon: IcHorizon, returns: Vec<f64>) {
        let h = horizon.bars();
        // Find the cross-section that was bar_index = current_bar - h.
        let target_bar = if self.bar >= h { self.bar - h } else { return };
        // Find matching pending entry.
        let pos = self.pending.iter().position(|cs| cs.bar_index == target_bar);
        if let Some(idx) = pos {
            let cs = &mut self.pending[idx];
            if cs.realized.is_none() {
                cs.realized = Some(returns.clone());
            }
            // Compute IC.
            if let Some(ref real) = self.pending[idx].realized.clone() {
                let signals = self.pending[idx].signals.clone();
                let bar_idx = self.pending[idx].bar_index;
                let n = signals.len().min(real.len());
                if n >= MIN_IC_OBS {
                    let s = &signals[..n];
                    let r = &real[..n];
                    let ic = spearman_rank_corr(s, r);
                    let obs = IcObservation {
                        bar_index: bar_idx,
                        horizon,
                        ic,
                        n_assets: n,
                    };
                    let series = self.observations.get_mut(&horizon).unwrap();
                    series.push_back(obs);
                    if series.len() > self.window {
                        series.pop_front();
                    }
                }
            }
        }
    }

    /// Push realized returns for ALL horizons at once.
    /// The `returns_by_horizon` slice has one entry per horizon (1, 5, 20, 60).
    /// Each inner vec is the cross-section of realized returns.
    pub fn push_realized_all(&mut self, returns: &[(IcHorizon, Vec<f64>)]) {
        for (h, r) in returns {
            self.push_realized(*h, r.clone());
        }
    }

    /// Get the IC time series for a given horizon.
    pub fn ic_series_for(&self, horizon: IcHorizon) -> Vec<&IcObservation> {
        self.observations[&horizon].iter().collect()
    }

    /// Get all IC values for a horizon as a plain Vec<f64>.
    pub fn ic_values(&self, horizon: IcHorizon) -> Vec<f64> {
        self.observations[&horizon].iter().map(|o| o.ic).collect()
    }

    /// Compute summary IC statistics for a given horizon.
    pub fn ic_stats(&self, horizon: IcHorizon) -> IcStats {
        let values = self.ic_values(horizon);
        IcStats::compute(horizon, &values)
    }

    /// Compute IC stats for all horizons.
    pub fn all_ic_stats(&self) -> Vec<IcStats> {
        IcHorizon::all().iter().map(|&h| self.ic_stats(h)).collect()
    }

    /// Compute rolling IC over a sub-window ending at the most recent observation.
    pub fn rolling_ic(&self, horizon: IcHorizon, sub_window: usize) -> Vec<f64> {
        let vals = self.ic_values(horizon);
        if vals.len() < sub_window {
            return vals;
        }
        let start = vals.len() - sub_window;
        vals[start..].to_vec()
    }

    /// Return current bar count.
    pub fn current_bar(&self) -> usize {
        self.bar
    }

    /// Total number of IC observations stored for a horizon.
    pub fn n_obs(&self, horizon: IcHorizon) -> usize {
        self.observations[&horizon].len()
    }

    /// Compute cross-horizon IC correlation matrix (4x4).
    /// Returns flattened row-major correlation values.
    pub fn cross_horizon_ic_corr(&self) -> Vec<Vec<f64>> {
        let horizons = IcHorizon::all();
        let series: Vec<Vec<f64>> = horizons.iter().map(|&h| self.ic_values(h)).collect();
        let min_len = series.iter().map(|v| v.len()).min().unwrap_or(0);
        let mut corr = vec![vec![1.0f64; 4]; 4];
        if min_len < 2 {
            return corr;
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                let xi: Vec<f64> = series[i][series[i].len() - min_len..].to_vec();
                let xj: Vec<f64> = series[j][series[j].len() - min_len..].to_vec();
                let c = crate::pearson_corr(&xi, &xj);
                corr[i][j] = c;
                corr[j][i] = c;
            }
        }
        corr
    }

    /// Compute the decay of mean IC across horizons.
    /// Returns Vec of (horizon_bars, mean_ic).
    pub fn ic_decay_profile(&self) -> Vec<(usize, f64)> {
        IcHorizon::all()
            .iter()
            .map(|&h| {
                let stats = self.ic_stats(h);
                (h.bars(), stats.mean_ic)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ic_stats_positive() {
        // Build IC series directly by supplying perfectly correlated signal/returns.
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        let mut series = IcSeries::new(252);
        let n_assets = 100;
        let base_ic = 0.40; // strong IC for test stability

        // We pair: push_signals at bar t, then immediately push_realized with correlated returns.
        // Because current_bar increments after push_signals, push_realized(H1) at bar t+1
        // looks for target_bar = (t+1) - 1 = t. This matches the pending entry.
        for _bar in 0..200 {
            let signals: Vec<f64> = (0..n_assets).map(|_| rng.gen::<f64>()).collect();
            series.push_signals(signals.clone());
            // Immediately after pushing signals at bar t, current_bar is t+1.
            // Realized for H1 targets current_bar - 1 = t, which we just pushed.
            let returns: Vec<f64> = signals
                .iter()
                .map(|s| base_ic * s + (1.0 - base_ic) * rng.gen::<f64>())
                .collect();
            series.push_realized(IcHorizon::H1, returns);
        }
        let stats = series.ic_stats(IcHorizon::H1);
        assert!(stats.n_obs > 0, "Should have IC observations");
        assert!(stats.mean_ic > 0.0, "Expected positive mean IC, got {}", stats.mean_ic);
    }

    #[test]
    fn test_rank_data() {
        let x = vec![3.0, 1.0, 2.0, 1.0];
        let r = crate::rank_data(&x);
        // sorted order: 1.0(idx1), 1.0(idx3), 2.0(idx2), 3.0(idx0)
        // 1.0 ties at ranks 1 and 2: average rank = 1.5.
        assert!((r[1] - 1.5).abs() < 1e-9, "r[1] = {}", r[1]);
        assert!((r[3] - 1.5).abs() < 1e-9, "r[3] = {}", r[3]);
        assert!((r[0] - 4.0).abs() < 1e-9, "r[0] = {}", r[0]); // 3.0 is rank 4
        assert!((r[2] - 3.0).abs() < 1e-9, "r[2] = {}", r[2]); // 2.0 is rank 3
    }

    #[test]
    fn test_spearman_uncorrelated() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..50).rev().map(|i| i as f64).collect();
        let ic = spearman_rank_corr(&x, &y);
        assert!((ic - (-1.0)).abs() < 0.01);
    }
}
