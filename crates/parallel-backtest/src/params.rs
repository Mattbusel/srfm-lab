use rand::Rng;
use serde::{Deserialize, Serialize};

/// All tunable parameters for the BH strategy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StrategyParams {
    /// Minimum bars to hold a position before exit is allowed.
    pub min_hold_bars: u32,
    /// Stale-price threshold on the 15-min frame (fractional move, e.g. 0.002).
    pub stale_15m_move: f64,
    /// Protect winners: close if unrealised PnL drops by this fraction from peak.
    pub winner_protection_pct: f64,
    /// GARCH target annualised volatility for position sizing.
    pub garch_target_vol: f64,
    /// Normal-regime pair correlation assumption.
    pub corr_normal: f64,
    /// Stress-regime pair correlation assumption.
    pub corr_stress: f64,
    /// Absolute BTC daily return that flips regime to stress.
    pub corr_stress_threshold: f64,
    /// UTC hours during which trading is blocked (e.g. low-liquidity windows).
    pub blocked_hours: Vec<u8>,
    /// UTC hours that receive a size boost.
    pub boost_hours: Vec<u8>,
    /// Multiplier applied to position size during boost hours.
    pub hour_boost_multiplier: f64,
    /// Symbols for which OU mean-reversion is disabled.
    pub ou_disabled_syms: Vec<String>,
}

impl Default for StrategyParams {
    fn default() -> Self {
        Self {
            min_hold_bars: 3,
            stale_15m_move: 0.002,
            winner_protection_pct: 0.30,
            garch_target_vol: 0.40,
            corr_normal: 0.60,
            corr_stress: 0.90,
            corr_stress_threshold: 0.03,
            blocked_hours: vec![2, 3, 4],
            boost_hours: vec![9, 10, 14, 15],
            hour_boost_multiplier: 1.25,
            ou_disabled_syms: vec![],
        }
    }
}

impl StrategyParams {
    /// Produce a neighbour by perturbing each numeric parameter by a small random delta.
    /// Used in simulated-annealing / local search.
    pub fn random_neighbor<R: Rng>(&self, rng: &mut R, temperature: f64) -> Self {
        let mut neighbor = self.clone();

        let perturb = |rng: &mut R, v: f64, scale: f64| -> f64 {
            let delta: f64 = rng.gen_range(-1.0..=1.0) * scale * temperature;
            v + delta
        };

        neighbor.min_hold_bars =
            (self.min_hold_bars as i64 + rng.gen_range(-2..=2_i64)).clamp(1, 20) as u32;
        neighbor.stale_15m_move = perturb(rng, self.stale_15m_move, 0.001).clamp(0.0001, 0.02);
        neighbor.winner_protection_pct =
            perturb(rng, self.winner_protection_pct, 0.05).clamp(0.05, 0.90);
        neighbor.garch_target_vol = perturb(rng, self.garch_target_vol, 0.05).clamp(0.05, 2.0);
        neighbor.corr_normal = perturb(rng, self.corr_normal, 0.05).clamp(0.0, 0.99);
        let cn = neighbor.corr_normal;
        neighbor.corr_stress = perturb(rng, self.corr_stress, 0.03).clamp(cn, 0.99);
        neighbor.corr_stress_threshold =
            perturb(rng, self.corr_stress_threshold, 0.005).clamp(0.005, 0.10);
        neighbor.hour_boost_multiplier =
            perturb(rng, self.hour_boost_multiplier, 0.1).clamp(1.0, 3.0);
        neighbor
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
}

/// Ranges defining the parameter space for sampling.
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    pub min_hold_bars: (u32, u32),
    pub stale_15m_move: (f64, f64),
    pub winner_protection_pct: (f64, f64),
    pub garch_target_vol: (f64, f64),
    pub corr_normal: (f64, f64),
    pub corr_stress: (f64, f64),
    pub corr_stress_threshold: (f64, f64),
    pub hour_boost_multiplier: (f64, f64),
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            min_hold_bars: (1, 12),
            stale_15m_move: (0.0005, 0.010),
            winner_protection_pct: (0.10, 0.70),
            garch_target_vol: (0.15, 1.00),
            corr_normal: (0.30, 0.80),
            corr_stress: (0.70, 0.99),
            corr_stress_threshold: (0.01, 0.06),
            hour_boost_multiplier: (1.0, 2.5),
        }
    }
}

impl ParameterSpace {
    /// Latin Hypercube Sampling: produce `n` well-spread samples across the parameter space.
    ///
    /// Each dimension is stratified into `n` equal-probability bins. One sample is drawn
    /// uniformly from each bin, then bins are shuffled independently per dimension.
    pub fn sample(&self, n: usize) -> Vec<StrategyParams> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        // Build per-dimension permuted strata (n strata per dimension).
        let strata: Vec<usize> = (0..n).collect();

        macro_rules! lhs_f64 {
            ($lo:expr, $hi:expr) => {{
                let mut perm = strata.clone();
                perm.shuffle(&mut rng);
                perm.iter()
                    .map(|&k| {
                        let u: f64 = rng.gen::<f64>();
                        $lo + ($hi - $lo) * ((k as f64 + u) / n as f64)
                    })
                    .collect::<Vec<f64>>()
            }};
        }

        macro_rules! lhs_u32 {
            ($lo:expr, $hi:expr) => {{
                let mut perm = strata.clone();
                perm.shuffle(&mut rng);
                perm.iter()
                    .map(|&k| {
                        let u: f64 = rng.gen::<f64>();
                        let v = $lo as f64 + ($hi as f64 - $lo as f64) * ((k as f64 + u) / n as f64);
                        v.round() as u32
                    })
                    .collect::<Vec<u32>>()
            }};
        }

        let min_hold = lhs_u32!(self.min_hold_bars.0, self.min_hold_bars.1);
        let stale = lhs_f64!(self.stale_15m_move.0, self.stale_15m_move.1);
        let winner = lhs_f64!(self.winner_protection_pct.0, self.winner_protection_pct.1);
        let garch_vol = lhs_f64!(self.garch_target_vol.0, self.garch_target_vol.1);
        let corr_n = lhs_f64!(self.corr_normal.0, self.corr_normal.1);
        let corr_s = lhs_f64!(self.corr_stress.0, self.corr_stress.1);
        let corr_thresh = lhs_f64!(self.corr_stress_threshold.0, self.corr_stress_threshold.1);
        let boost_mul = lhs_f64!(self.hour_boost_multiplier.0, self.hour_boost_multiplier.1);

        (0..n)
            .map(|i| {
                let cn = corr_n[i].clamp(0.0, 0.99);
                // Ensure stress corr >= normal corr.
                let cs = corr_s[i].clamp(cn, 0.99);
                StrategyParams {
                    min_hold_bars: min_hold[i],
                    stale_15m_move: stale[i],
                    winner_protection_pct: winner[i],
                    garch_target_vol: garch_vol[i],
                    corr_normal: cn,
                    corr_stress: cs,
                    corr_stress_threshold: corr_thresh[i],
                    blocked_hours: vec![2, 3, 4],
                    boost_hours: vec![9, 10, 14, 15],
                    hour_boost_multiplier: boost_mul[i],
                    ou_disabled_syms: vec![],
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params_roundtrip() {
        let p = StrategyParams::default();
        let json = p.to_json().unwrap();
        let p2 = StrategyParams::from_json(&json).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn test_lhs_sample_count() {
        let space = ParameterSpace::default();
        let samples = space.sample(100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_lhs_sample_ranges() {
        let space = ParameterSpace::default();
        let samples = space.sample(200);
        for s in &samples {
            assert!(s.min_hold_bars >= space.min_hold_bars.0);
            assert!(s.min_hold_bars <= space.min_hold_bars.1);
            assert!(s.stale_15m_move >= space.stale_15m_move.0);
            assert!(s.stale_15m_move <= space.stale_15m_move.1);
            assert!(s.corr_stress >= s.corr_normal, "stress corr must >= normal corr");
        }
    }

    #[test]
    fn test_random_neighbor_stays_valid() {
        let p = StrategyParams::default();
        let mut rng = rand::thread_rng();
        for _ in 0..50 {
            let n = p.random_neighbor(&mut rng, 0.5);
            assert!(n.min_hold_bars >= 1);
            assert!(n.garch_target_vol > 0.0);
            assert!(n.corr_stress >= n.corr_normal);
        }
    }
}
