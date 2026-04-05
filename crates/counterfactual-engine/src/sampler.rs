//! Parameter sampling strategies for the counterfactual engine.
//!
//! # Strategies
//!
//! | Type                    | Use case                                          |
//! |-------------------------|---------------------------------------------------|
//! | [`LatinHypercubeSampler`] | Broad coverage of parameter space, O(n·d)       |
//! | [`SobolSampler`]          | Low-discrepancy quasi-random sequences          |
//! | [`NeighborhoodSampler`]   | Gaussian perturbation around a known-good point |
//!
//! All samplers implement the [`Sampler`] trait and return [`ParameterSample`]s.

use std::collections::HashMap;

use rand::prelude::*;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Parameter bounds
// ---------------------------------------------------------------------------

/// Min/max bounds for a single parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamBound {
    pub name: String,
    pub min: f64,
    pub max: f64,
    /// If true, round the sampled value to the nearest integer.
    pub integer: bool,
}

impl ParamBound {
    pub fn new(name: &str, min: f64, max: f64) -> Self {
        Self { name: name.to_owned(), min, max, integer: false }
    }

    pub fn integer(name: &str, min: f64, max: f64) -> Self {
        Self { name: name.to_owned(), min, max, integer: true }
    }

    /// Map a value in [0, 1] to the bounded range.
    pub fn map_unit(&self, u: f64) -> f64 {
        let v = self.min + u * (self.max - self.min);
        if self.integer {
            v.round()
        } else {
            v.clamp(self.min, self.max)
        }
    }

    /// Map a raw value to [0, 1] (for inverse transforms).
    pub fn to_unit(&self, v: f64) -> f64 {
        let range = self.max - self.min;
        if range < f64::EPSILON {
            return 0.5;
        }
        ((v - self.min) / range).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// ParameterBounds collection
// ---------------------------------------------------------------------------

/// Complete set of parameter bounds for one sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBounds {
    pub bounds: Vec<ParamBound>,
}

impl ParameterBounds {
    pub fn new(bounds: Vec<ParamBound>) -> Self {
        Self { bounds }
    }

    /// Default bounds matching the 15 genome parameters used by the IAE.
    pub fn genome_defaults() -> Self {
        Self::new(vec![
            ParamBound::new("bh_form",          1.70,  1.98),
            ParamBound::new("bh_decay",          0.92,  0.99),
            ParamBound::new("bh_collapse",       0.60,  0.95),
            ParamBound::integer("bh_ctl_min",    1.0,   5.0),
            ParamBound::integer("min_hold_bars", 1.0,  20.0),
            ParamBound::new("stale_15m_move",    0.001, 0.020),
            ParamBound::new("delta_max_frac",    0.10,  0.60),
            ParamBound::new("corr_factor",       0.15,  0.80),
            ParamBound::new("garch_target_vol",  0.60,  2.00),
            ParamBound::new("ou_frac",           0.02,  0.20),
            ParamBound::new("pos_floor_scale",   0.001, 0.050),
            ParamBound::new("cf_scale_bull",     0.5,   2.0),
            ParamBound::new("cf_scale_bear",     0.5,   2.0),
            ParamBound::new("cf_scale_neutral",  0.5,   2.0),
            ParamBound::new("pos_size_cap",      0.01,  0.20),
        ])
    }

    pub fn dim(&self) -> usize {
        self.bounds.len()
    }

    pub fn param_names(&self) -> Vec<&str> {
        self.bounds.iter().map(|b| b.name.as_str()).collect()
    }

    /// Map a unit vector (length == dim()) to a parameter HashMap.
    pub fn map_unit_vec(&self, unit: &[f64]) -> HashMap<String, f64> {
        assert_eq!(unit.len(), self.bounds.len());
        self.bounds.iter()
            .zip(unit.iter())
            .map(|(b, &u)| (b.name.clone(), b.map_unit(u)))
            .collect()
    }

    /// Map a parameter HashMap to a unit vector.
    pub fn to_unit_vec(&self, params: &HashMap<String, f64>) -> Vec<f64> {
        self.bounds.iter()
            .map(|b| {
                let v = params.get(&b.name).copied().unwrap_or((b.min + b.max) / 2.0);
                b.to_unit(v)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// ParameterSample
// ---------------------------------------------------------------------------

/// A single sample from the parameter space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSample {
    pub params: HashMap<String, f64>,
    /// Bounds snapshot (for post-hoc validation).
    #[serde(skip)]
    pub _bounds: Option<ParameterBounds>,
}

impl ParameterSample {
    pub fn new(params: HashMap<String, f64>) -> Self {
        Self { params, _bounds: None }
    }

    pub fn with_bounds(params: HashMap<String, f64>, bounds: ParameterBounds) -> Self {
        Self { params, _bounds: Some(bounds) }
    }

    /// Retrieve (lo, hi) for a named parameter from the embedded bounds.
    pub fn bounds_for(&self, name: &str) -> Option<(f64, f64)> {
        self._bounds.as_ref()?.bounds.iter()
            .find(|b| b.name == name)
            .map(|b| (b.min, b.max))
    }
}

// ---------------------------------------------------------------------------
// Sampler trait
// ---------------------------------------------------------------------------

pub trait Sampler {
    /// Generate `n` parameter samples.
    fn sample(&mut self, n: usize) -> Vec<ParameterSample>;
}

// ---------------------------------------------------------------------------
// Latin Hypercube Sampler
// ---------------------------------------------------------------------------

/// Generates samples using Latin Hypercube Sampling.
///
/// Divides each dimension into `n` equal strata and places exactly one
/// sample per stratum (per dimension), with strata permuted independently.
/// This guarantees better coverage than pure random Monte Carlo.
pub struct LatinHypercubeSampler {
    bounds: ParameterBounds,
    rng: StdRng,
}

impl LatinHypercubeSampler {
    /// Create a new LHS sampler.
    ///
    /// # Arguments
    /// * `bounds` — parameter bounds
    /// * `seed`   — RNG seed (use `None` for time-seeded)
    pub fn new(bounds: ParameterBounds, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { bounds, rng }
    }

    /// Generate a single LHS sample matrix, shape (n, d), values in [0,1).
    fn lhs_unit_matrix(&mut self, n: usize) -> Vec<Vec<f64>> {
        let d = self.bounds.dim();
        let mut matrix = vec![vec![0.0f64; d]; n];

        for col in 0..d {
            // Create stratum indices [0, 1, ..., n-1] and shuffle
            let mut strata: Vec<usize> = (0..n).collect();
            strata.shuffle(&mut self.rng);

            for row in 0..n {
                let stratum = strata[row];
                // Sample uniformly within the stratum
                let jitter: f64 = self.rng.gen::<f64>() / n as f64;
                matrix[row][col] = (stratum as f64) / (n as f64) + jitter;
            }
        }
        matrix
    }
}

impl Sampler for LatinHypercubeSampler {
    fn sample(&mut self, n: usize) -> Vec<ParameterSample> {
        let unit_matrix = self.lhs_unit_matrix(n);
        unit_matrix
            .iter()
            .map(|unit_row| {
                let params = self.bounds.map_unit_vec(unit_row);
                ParameterSample::with_bounds(params, self.bounds.clone())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Sobol Sampler
// ---------------------------------------------------------------------------

/// Quasi-random low-discrepancy sampler using Van der Corput sequences.
///
/// Each dimension uses a distinct prime base.  For true Sobol sequences
/// (which require direction numbers), replace with a library call.
/// This implementation provides good low-discrepancy properties for up to
/// 15 dimensions.
pub struct SobolSampler {
    bounds: ParameterBounds,
    index: usize,
}

impl SobolSampler {
    const PRIMES: [u64; 20] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    ];

    pub fn new(bounds: ParameterBounds) -> Self {
        Self { bounds, index: 1 }
    }

    /// Van der Corput sequence: i-th element of base-p sequence.
    fn van_der_corput(i: usize, base: u64) -> f64 {
        let mut q = i as u64;
        let mut denom: f64 = 1.0;
        let mut result: f64 = 0.0;
        let base_f = base as f64;
        while q > 0 {
            denom *= base_f;
            result += (q % base) as f64 / denom;
            q /= base;
        }
        result
    }

    fn next_unit_vec(&mut self) -> Vec<f64> {
        let d = self.bounds.dim();
        let i = self.index;
        self.index += 1;
        (0..d)
            .map(|j| {
                let base = Self::PRIMES[j % Self::PRIMES.len()];
                Self::van_der_corput(i, base)
            })
            .collect()
    }
}

impl Sampler for SobolSampler {
    fn sample(&mut self, n: usize) -> Vec<ParameterSample> {
        (0..n)
            .map(|_| {
                let unit = self.next_unit_vec();
                let params = self.bounds.map_unit_vec(&unit);
                ParameterSample::with_bounds(params, self.bounds.clone())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Neighborhood Sampler
// ---------------------------------------------------------------------------

/// Gaussian perturbation around a center point in parameter space.
///
/// Useful for local refinement around a known-good genome.
pub struct NeighborhoodSampler {
    bounds: ParameterBounds,
    center: Vec<f64>,   // unit coordinates of center
    radius: f64,        // std-dev as fraction of each dimension's range
    rng: StdRng,
}

impl NeighborhoodSampler {
    /// Create a neighborhood sampler.
    ///
    /// # Arguments
    /// * `bounds`         — parameter bounds
    /// * `center_params`  — center of the neighborhood (genome params dict)
    /// * `radius`         — Gaussian std-dev as fraction of range (default 0.15)
    /// * `seed`           — RNG seed
    pub fn new(
        bounds: ParameterBounds,
        center_params: &HashMap<String, f64>,
        radius: f64,
        seed: Option<u64>,
    ) -> Self {
        let center = bounds.to_unit_vec(center_params);
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self { bounds, center, radius, rng }
    }

    fn perturb(&mut self) -> Vec<f64> {
        let d = self.bounds.dim();
        let mut unit = Vec::with_capacity(d);
        for i in 0..d {
            // Box-Muller transform for normal variate
            let u1: f64 = self.rng.gen::<f64>().max(f64::EPSILON);
            let u2: f64 = self.rng.gen::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let v = (self.center[i] + self.radius * z).clamp(0.0, 1.0);
            unit.push(v);
        }
        unit
    }
}

impl Sampler for NeighborhoodSampler {
    fn sample(&mut self, n: usize) -> Vec<ParameterSample> {
        (0..n)
            .map(|_| {
                let unit = self.perturb();
                let params = self.bounds.map_unit_vec(&unit);
                ParameterSample::with_bounds(params, self.bounds.clone())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bounds() -> ParameterBounds {
        ParameterBounds::genome_defaults()
    }

    #[test]
    fn lhs_stratification() {
        let bounds = default_bounds();
        let mut sampler = LatinHypercubeSampler::new(bounds.clone(), Some(7));
        let n = 30;
        let samples = sampler.sample(n);
        assert_eq!(samples.len(), n);

        // For continuous dimensions: each stratum [0,1/n) ... [(n-1)/n, 1) must be occupied exactly once.
        // Integer params (bh_ctl_min, min_hold_bars) are excluded — rounding can collapse strata.
        let integer_params: std::collections::HashSet<&str> = ["bh_ctl_min", "min_hold_bars"].iter().cloned().collect();
        let d = bounds.dim();
        let unit_matrix: Vec<Vec<f64>> = samples.iter()
            .map(|s| bounds.to_unit_vec(&s.params))
            .collect();
        for (col, bound) in bounds.bounds.iter().enumerate() {
            if integer_params.contains(bound.name.as_str()) {
                continue; // skip integer params
            }
            let mut counts = vec![0usize; n];
            for row in 0..n {
                let stratum = (unit_matrix[row][col] * n as f64).floor() as usize;
                counts[stratum.min(n - 1)] += 1;
            }
            for &c in &counts {
                assert_eq!(c, 1, "col {col} ({}): stratum count should be 1", bound.name);
            }
        }
    }

    #[test]
    fn sobol_count() {
        let bounds = default_bounds();
        let mut sampler = SobolSampler::new(bounds);
        let samples = sampler.sample(64);
        assert_eq!(samples.len(), 64);
    }

    #[test]
    fn neighborhood_within_bounds() {
        let bounds = default_bounds();
        let center: HashMap<String, f64> = bounds.bounds.iter()
            .map(|b| (b.name.clone(), (b.min + b.max) / 2.0))
            .collect();
        let mut sampler = NeighborhoodSampler::new(bounds.clone(), &center, 0.15, Some(42));
        let samples = sampler.sample(100);
        for s in &samples {
            for b in &bounds.bounds {
                let v = s.params[&b.name];
                assert!(v >= b.min && v <= b.max, "{} = {} out of [{}, {}]", b.name, v, b.min, b.max);
            }
        }
    }

    #[test]
    fn param_bound_unit_roundtrip() {
        let b = ParamBound::new("x", 1.0, 3.0);
        assert!((b.map_unit(0.0) - 1.0).abs() < 1e-9);
        assert!((b.map_unit(1.0) - 3.0).abs() < 1e-9);
        assert!((b.map_unit(0.5) - 2.0).abs() < 1e-9);
        assert!((b.to_unit(1.0) - 0.0).abs() < 1e-9);
        assert!((b.to_unit(3.0) - 1.0).abs() < 1e-9);
    }
}
