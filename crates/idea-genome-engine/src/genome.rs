/// Genome: float vector encoding of all LARSA strategy parameters.
///
/// Parameter layout:
///   [0]  bh_form           1.5 ..  2.5
///   [1]  bh_decay          0.85 .. 0.99
///   [2]  bh_collapse       0.80 .. 1.0
///   [3]  bh_ctl_min        1.0 ..  6.0  (discretized to nearest int at use)
///   [4]  min_hold_bars     1.0 .. 12.0
///   [5]  stale_15m_move    0.0003 .. 0.005
///   [6]  delta_max_frac    0.20 .. 0.80
///   [7]  cf_scale_15m      0.5 ..  2.0
///   [8]  cf_scale_1h       0.5 ..  2.0
///   [9]  cf_scale_1d       0.5 ..  2.0
///  [10]  corr_factor       0.1 ..  0.8
///  [11]  garch_target_vol  0.5 ..  2.0
///  [12]  ou_frac           0.0 ..  0.15
///  [13]  pos_floor_ctl_min 3.0 .. 10.0
///  [14]  pos_floor_frac    0.5 ..  0.9

use anyhow::{Context, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::fitness::FitnessVec;

/// Number of parameters in a genome.
pub const N_PARAMS: usize = 15;

/// Static parameter metadata: (name, min, max).
pub const PARAM_META: [(&str, f64, f64); N_PARAMS] = [
    ("bh_form",           1.5,    2.5  ),
    ("bh_decay",          0.85,   0.99 ),
    ("bh_collapse",       0.80,   1.0  ),
    ("bh_ctl_min",        1.0,    6.0  ),
    ("min_hold_bars",     1.0,   12.0  ),
    ("stale_15m_move",    0.0003, 0.005),
    ("delta_max_frac",    0.20,   0.80 ),
    ("cf_scale_15m",      0.5,    2.0  ),
    ("cf_scale_1h",       0.5,    2.0  ),
    ("cf_scale_1d",       0.5,    2.0  ),
    ("corr_factor",       0.1,    0.8  ),
    ("garch_target_vol",  0.5,    2.0  ),
    ("ou_frac",           0.0,    0.15 ),
    ("pos_floor_ctl_min", 3.0,   10.0  ),
    ("pos_floor_frac",    0.5,    0.9  ),
];

/// A single individual in the genetic population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    /// Unique identifier (UUID v4).
    pub id: String,
    /// Encoded parameter values — length must equal `N_PARAMS`.
    pub parameters: Vec<f64>,
    /// Human-readable names, one per parameter (mirrors `PARAM_META`).
    pub param_names: Vec<String>,
    /// (min, max) bounds per parameter.
    pub bounds: Vec<(f64, f64)>,
    /// Multi-objective fitness scores, set after evaluation.
    pub fitness: Option<FitnessVec>,
    /// Generation in which this genome was created.
    pub generation: u32,
    /// IDs of the parent genomes (0 for seed, 1 for cloned, 2 for crossed).
    pub parent_ids: Vec<String>,
}

impl Genome {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a genome with random parameter values sampled uniformly within bounds.
    pub fn new_random(rng: &mut impl Rng) -> Self {
        let parameters: Vec<f64> = PARAM_META
            .iter()
            .map(|(_, lo, hi)| rng.gen_range(*lo..=*hi))
            .collect();

        let param_names: Vec<String> = PARAM_META.iter().map(|(n, _, _)| n.to_string()).collect();
        let bounds: Vec<(f64, f64)> = PARAM_META.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();

        Self {
            id: Uuid::new_v4().to_string(),
            parameters,
            param_names,
            bounds,
            fitness: None,
            generation: 0,
            parent_ids: vec![],
        }
    }

    /// Create a genome with explicit parameter values (used by crossover / mutation operators).
    /// Bounds are initialised from `PARAM_META`; caller should call `clamp()` afterwards.
    pub fn from_parameters(parameters: Vec<f64>, generation: u32, parent_ids: Vec<String>) -> Self {
        assert_eq!(
            parameters.len(),
            N_PARAMS,
            "parameters length must equal N_PARAMS ({})",
            N_PARAMS
        );

        let param_names: Vec<String> = PARAM_META.iter().map(|(n, _, _)| n.to_string()).collect();
        let bounds: Vec<(f64, f64)> = PARAM_META.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();

        Self {
            id: Uuid::new_v4().to_string(),
            parameters,
            param_names,
            bounds,
            fitness: None,
            generation,
            parent_ids,
        }
    }

    // ------------------------------------------------------------------
    // Bound enforcement
    // ------------------------------------------------------------------

    /// Clamp every parameter to its declared (min, max) bounds.
    pub fn clamp(&mut self) {
        for (i, param) in self.parameters.iter_mut().enumerate() {
            let (lo, hi) = self.bounds[i];
            *param = param.clamp(lo, hi);
        }
    }

    // ------------------------------------------------------------------
    // Serialisation
    // ------------------------------------------------------------------

    /// Serialise to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Genome serialisation is infallible")
    }

    /// Deserialise from a JSON string.
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("Failed to deserialise Genome from JSON")
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Return the index of a parameter by name, or `None` if not found.
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.param_names.iter().position(|n| n == name)
    }

    /// Get a parameter value by name.
    pub fn get_param(&self, name: &str) -> Option<f64> {
        self.param_index(name).map(|i| self.parameters[i])
    }

    /// Check whether all parameters lie strictly within bounds.
    pub fn is_within_bounds(&self) -> bool {
        self.parameters
            .iter()
            .zip(self.bounds.iter())
            .all(|(v, (lo, hi))| v >= lo && v <= hi)
    }

    /// Scalar fitness used for single-objective comparisons (Sharpe, or –∞ when unknown).
    pub fn sharpe(&self) -> f64 {
        self.fitness
            .as_ref()
            .map(|f| f.sharpe)
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Calmar ratio (or –∞ when unknown).
    pub fn calmar(&self) -> f64 {
        self.fitness
            .as_ref()
            .map(|f| f.calmar)
            .unwrap_or(f64::NEG_INFINITY)
    }

    /// Returns `true` if this genome dominates `other` in the Pareto sense.
    ///
    /// Dominance: at least as good on every objective and strictly better on one.
    /// Objectives (higher = better): sharpe, calmar, win_rate, profit_factor.
    /// Objectives (lower = better): max_dd, is_oos_spread.
    pub fn dominates(&self, other: &Genome) -> bool {
        let (Some(s), Some(o)) = (&self.fitness, &other.fitness) else {
            return false;
        };

        let self_vals: [f64; 6] = [
            s.sharpe,
            s.calmar,
            s.win_rate,
            s.profit_factor,
            -s.max_dd,
            -s.is_oos_spread,
        ];
        let other_vals: [f64; 6] = [
            o.sharpe,
            o.calmar,
            o.win_rate,
            o.profit_factor,
            -o.max_dd,
            -o.is_oos_spread,
        ];

        let at_least_as_good = self_vals
            .iter()
            .zip(other_vals.iter())
            .all(|(sv, ov)| sv >= ov);
        let strictly_better = self_vals
            .iter()
            .zip(other_vals.iter())
            .any(|(sv, ov)| sv > ov);

        at_least_as_good && strictly_better
    }

    /// Euclidean distance in parameter space between two genomes.
    pub fn parameter_distance(&self, other: &Genome) -> f64 {
        self.parameters
            .iter()
            .zip(other.parameters.iter())
            .zip(self.bounds.iter())
            .map(|((a, b), (lo, hi))| {
                let range = hi - lo;
                if range > 0.0 {
                    let diff = (a - b) / range;
                    diff * diff
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            .sqrt()
    }

    /// Return a named-parameter map suitable for writing to a strategy config JSON.
    pub fn to_param_map(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        for (name, value) in self.param_names.iter().zip(self.parameters.iter()) {
            map.insert(name.clone(), serde_json::json!(value));
        }
        serde_json::Value::Object(map)
    }
}

impl std::fmt::Display for Genome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Genome(id={}, gen={}, sharpe={:.3})",
            &self.id[..8],
            self.generation,
            self.sharpe()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded_rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn random_genome_within_bounds() {
        let mut rng = seeded_rng();
        let g = Genome::new_random(&mut rng);
        assert!(g.is_within_bounds(), "Random genome must lie within bounds");
        assert_eq!(g.parameters.len(), N_PARAMS);
    }

    #[test]
    fn clamp_restores_bounds() {
        let mut rng = seeded_rng();
        let mut g = Genome::new_random(&mut rng);
        // Force out-of-bounds
        for p in g.parameters.iter_mut() {
            *p = 1e9;
        }
        g.clamp();
        assert!(g.is_within_bounds());
    }

    #[test]
    fn serialise_round_trip() {
        let mut rng = seeded_rng();
        let g = Genome::new_random(&mut rng);
        let json = g.to_json();
        let g2 = Genome::from_json(&json).unwrap();
        assert_eq!(g.id, g2.id);
        for (a, b) in g.parameters.iter().zip(g2.parameters.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn param_index_lookup() {
        let mut rng = seeded_rng();
        let g = Genome::new_random(&mut rng);
        assert_eq!(g.param_index("bh_form"), Some(0));
        assert_eq!(g.param_index("pos_floor_frac"), Some(14));
        assert_eq!(g.param_index("nonexistent"), None);
    }
}
