//! constraint_handler.rs -- Constraint handling for genome bounds and domain-specific rules.
//!
//! Provides both hard and soft constraint enforcement. Hard constraints either
//! clamp, repair, or kill infeasible genomes. Soft constraints add a penalty
//! term to the fitness objective.
//!
//! Usage pattern:
//!   1. Build a ConstraintSet with the desired constraints.
//!   2. Call apply_all(&mut genome) -> f64 (returns total penalty).
//!   3. Add the penalty to the genome fitness objective (or set to NEG_INFINITY for hard kill).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constraint trait
// ---------------------------------------------------------------------------

/// Common interface for all constraint operators.
///
/// `apply` modifies `genome` in place (repair strategy) and returns a
/// non-negative penalty value. A penalty of 0.0 means the constraint is satisfied.
/// For hard constraints that kill infeasible solutions, return f64::INFINITY.
pub trait Constraint: Send + Sync {
    fn apply(&self, genome: &mut Vec<f64>) -> f64;

    /// Human-readable constraint name for logging.
    fn name(&self) -> &str;

    /// Whether this constraint is hard (violation = instant death) or soft (penalty).
    fn is_hard(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// BoundsConstraint
// ---------------------------------------------------------------------------

/// Hard-clamp each gene to its declared [min, max] interval.
///
/// This is a repair strategy: infeasible values are projected onto the boundary.
/// Returns penalty proportional to the total clamped distance (useful for
/// gradient-based methods that need to see constraint violations).
pub struct BoundsConstraint {
    /// Per-gene (min, max) bounds. Length must match genome.
    pub bounds: Vec<(f64, f64)>,
}

impl BoundsConstraint {
    pub fn new(bounds: Vec<(f64, f64)>) -> Self {
        BoundsConstraint { bounds }
    }

    /// Construct from PARAM_META-style array of (name, lo, hi) triples.
    pub fn from_param_meta(meta: &[(&str, f64, f64)]) -> Self {
        let bounds = meta.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();
        BoundsConstraint { bounds }
    }
}

impl Constraint for BoundsConstraint {
    fn apply(&self, genome: &mut Vec<f64>) -> f64 {
        let n = genome.len().min(self.bounds.len());
        let mut penalty = 0.0f64;

        for i in 0..n {
            let (lo, hi) = self.bounds[i];
            let original = genome[i];
            genome[i] = genome[i].clamp(lo, hi);
            penalty += (original - genome[i]).abs();
        }

        penalty
    }

    fn name(&self) -> &str {
        "BoundsConstraint"
    }

    fn is_hard(&self) -> bool {
        false // We repair rather than kill
    }
}

// ---------------------------------------------------------------------------
// SumConstraint
// ---------------------------------------------------------------------------

/// Enforce that specified genes sum to a target value.
///
/// Repair strategy: scale all specified genes proportionally so their sum
/// equals `target`. If the sum is zero (all genes are zero), distribute
/// the target equally. Handles negative genes by shifting to positive before
/// normalization when `require_nonnegative` is true.
pub struct SumConstraint {
    /// Indices of genes that must sum to `target`.
    pub indices: Vec<usize>,
    /// Target sum value (e.g., 1.0 for portfolio weights).
    pub target: f64,
    /// If true, genes are shifted to [0, inf) before normalization.
    pub require_nonnegative: bool,
    /// Tolerance within which the sum is considered satisfied.
    pub tolerance: f64,
}

impl SumConstraint {
    pub fn new(indices: Vec<usize>, target: f64) -> Self {
        SumConstraint {
            indices,
            target,
            require_nonnegative: true,
            tolerance: 1e-9,
        }
    }

    pub fn all_genes(n_genes: usize, target: f64) -> Self {
        SumConstraint::new((0..n_genes).collect(), target)
    }
}

impl Constraint for SumConstraint {
    fn apply(&self, genome: &mut Vec<f64>) -> f64 {
        if self.indices.is_empty() {
            return 0.0;
        }

        let n_idx = self.indices.len();

        if self.require_nonnegative {
            // Shift negative values to zero first
            for &i in &self.indices {
                if i < genome.len() {
                    genome[i] = genome[i].max(0.0);
                }
            }
        }

        let current_sum: f64 = self
            .indices
            .iter()
            .filter(|&&i| i < genome.len())
            .map(|&i| genome[i])
            .sum();

        let violation = (current_sum - self.target).abs();
        if violation <= self.tolerance {
            return 0.0;
        }

        if current_sum.abs() < 1e-12 {
            // Distribute equally
            let equal_share = self.target / n_idx as f64;
            for &i in &self.indices {
                if i < genome.len() {
                    genome[i] = equal_share;
                }
            }
        } else {
            // Scale proportionally
            let scale = self.target / current_sum;
            for &i in &self.indices {
                if i < genome.len() {
                    genome[i] *= scale;
                }
            }
        }

        violation
    }

    fn name(&self) -> &str {
        "SumConstraint"
    }
}

// ---------------------------------------------------------------------------
// MonotonicityConstraint
// ---------------------------------------------------------------------------

/// Enforce gene[i] <= gene[i+1] for a specified range of gene indices.
///
/// Repair strategy: sort the affected genes in ascending order.
/// This is lossless (same gene values, reordered) and preserves the distribution.
pub struct MonotonicityConstraint {
    /// Contiguous range [start, end) of gene indices that must be non-decreasing.
    pub start: usize,
    pub end: usize,
}

impl MonotonicityConstraint {
    pub fn new(start: usize, end: usize) -> Self {
        assert!(end > start, "MonotonicityConstraint: end must be > start");
        MonotonicityConstraint { start, end }
    }
}

impl Constraint for MonotonicityConstraint {
    fn apply(&self, genome: &mut Vec<f64>) -> f64 {
        let end = self.end.min(genome.len());
        if end <= self.start {
            return 0.0;
        }

        // Measure violation before repair: sum of out-of-order differences
        let mut violation = 0.0f64;
        for i in self.start..(end - 1) {
            if genome[i] > genome[i + 1] {
                violation += genome[i] - genome[i + 1];
            }
        }

        if violation > 0.0 {
            // Sort the slice in ascending order (repair)
            genome[self.start..end].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        violation
    }

    fn name(&self) -> &str {
        "MonotonicityConstraint"
    }
}

// ---------------------------------------------------------------------------
// ConflictConstraint
// ---------------------------------------------------------------------------

/// Mutual exclusion constraint: if gene[trigger_idx] > threshold_trigger,
/// then gene[suppressed_idx] must be <= threshold_suppressed.
///
/// Repair: if the conflict is detected, clamp gene[suppressed_idx] down.
pub struct ConflictConstraint {
    pub trigger_idx: usize,
    pub threshold_trigger: f64,
    pub suppressed_idx: usize,
    pub threshold_suppressed: f64,
    /// Penalty multiplier per unit of violation.
    pub penalty_scale: f64,
}

impl ConflictConstraint {
    pub fn new(
        trigger_idx: usize,
        threshold_trigger: f64,
        suppressed_idx: usize,
        threshold_suppressed: f64,
    ) -> Self {
        ConflictConstraint {
            trigger_idx,
            threshold_trigger,
            suppressed_idx,
            threshold_suppressed,
            penalty_scale: 10.0,
        }
    }
}

impl Constraint for ConflictConstraint {
    fn apply(&self, genome: &mut Vec<f64>) -> f64 {
        let n = genome.len();
        if self.trigger_idx >= n || self.suppressed_idx >= n {
            return 0.0;
        }

        let trigger_active = genome[self.trigger_idx] > self.threshold_trigger;
        if !trigger_active {
            return 0.0;
        }

        let suppressed_val = genome[self.suppressed_idx];
        if suppressed_val <= self.threshold_suppressed {
            return 0.0; // constraint already satisfied
        }

        // Compute violation magnitude
        let violation = suppressed_val - self.threshold_suppressed;

        // Repair: clamp the suppressed gene
        genome[self.suppressed_idx] = self.threshold_suppressed;

        violation * self.penalty_scale
    }

    fn name(&self) -> &str {
        "ConflictConstraint"
    }
}

// ---------------------------------------------------------------------------
// PenaltyMethod
// ---------------------------------------------------------------------------

/// Soft constraint via additive penalty on fitness.
///
/// Evaluates a user-supplied predicate over the genome. If violated,
/// returns a penalty proportional to the violation magnitude.
///
/// The penalty function signature: fn(genome: &[f64]) -> f64
///   -- returns 0.0 if satisfied
///   -- returns positive value proportional to violation if not
///
/// The returned penalty is intended to be subtracted from Sharpe before
/// selection. PenaltyMethod does NOT repair the genome.
pub struct PenaltyMethod {
    pub label: String,
    /// Penalty coefficient (scales the raw violation into fitness units).
    pub coefficient: f64,
    /// Penalty evaluation function.
    penalty_fn: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
}

impl PenaltyMethod {
    pub fn new(
        label: impl Into<String>,
        coefficient: f64,
        penalty_fn: impl Fn(&[f64]) -> f64 + Send + Sync + 'static,
    ) -> Self {
        PenaltyMethod {
            label: label.into(),
            coefficient,
            penalty_fn: Box::new(penalty_fn),
        }
    }
}

impl Constraint for PenaltyMethod {
    fn apply(&self, genome: &mut Vec<f64>) -> f64 {
        let raw_violation = (self.penalty_fn)(genome);
        raw_violation * self.coefficient
    }

    fn name(&self) -> &str {
        &self.label
    }

    fn is_hard(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// DeathPenalty
// ---------------------------------------------------------------------------

/// Hard constraint: infeasible genomes receive fitness = f64::NEG_INFINITY.
///
/// The feasibility predicate returns true if the genome is feasible.
/// Infeasible genomes are NOT repaired -- they are marked for death and
/// excluded from selection by returning f64::INFINITY as the penalty.
/// The caller must treat penalty == f64::INFINITY as instant discard.
pub struct DeathPenalty {
    pub label: String,
    feasibility_fn: Box<dyn Fn(&[f64]) -> bool + Send + Sync>,
}

impl DeathPenalty {
    pub fn new(
        label: impl Into<String>,
        feasibility_fn: impl Fn(&[f64]) -> bool + Send + Sync + 'static,
    ) -> Self {
        DeathPenalty {
            label: label.into(),
            feasibility_fn: Box::new(feasibility_fn),
        }
    }
}

impl Constraint for DeathPenalty {
    fn apply(&self, genome: &mut Vec<f64>) -> f64 {
        if (self.feasibility_fn)(genome) {
            0.0
        } else {
            f64::INFINITY // caller interprets this as NEG_INFINITY fitness
        }
    }

    fn name(&self) -> &str {
        &self.label
    }

    fn is_hard(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// ConstraintSet
// ---------------------------------------------------------------------------

/// Collection of constraints applied in sequence.
///
/// Constraints are applied in insertion order. Hard constraints (is_hard() == true)
/// short-circuit on first violation: remaining constraints are not evaluated
/// and f64::INFINITY is returned immediately.
///
/// The returned penalty is the sum of all soft constraint penalties.
/// When any hard constraint fires, the returned penalty is f64::INFINITY.
pub struct ConstraintSet {
    constraints: Vec<Box<dyn Constraint>>,
    /// Accumulated penalty statistics per constraint name (for diagnostics).
    violation_counts: HashMap<String, usize>,
    applications: usize,
}

impl ConstraintSet {
    pub fn new() -> Self {
        ConstraintSet {
            constraints: Vec::new(),
            violation_counts: HashMap::new(),
            applications: 0,
        }
    }

    /// Add a constraint. Constraints are applied in order.
    pub fn add(&mut self, constraint: Box<dyn Constraint>) -> &mut Self {
        self.constraints.push(constraint);
        self
    }

    /// Apply all constraints to the genome, returning the total penalty.
    ///
    /// Returns f64::INFINITY if any hard constraint fires.
    /// Returns sum of soft penalties otherwise.
    /// Repairs genome in place where constraints use repair strategies.
    pub fn apply_all(&mut self, genome: &mut Vec<f64>) -> f64 {
        self.applications += 1;
        let mut total_penalty = 0.0f64;

        for constraint in &self.constraints {
            let penalty = constraint.apply(genome);

            if penalty > 0.0 {
                *self
                    .violation_counts
                    .entry(constraint.name().to_string())
                    .or_insert(0) += 1;
            }

            if constraint.is_hard() && penalty > 0.0 {
                return f64::INFINITY;
            }

            total_penalty += penalty;
        }

        total_penalty
    }

    /// Return true if the genome satisfies all constraints without repair.
    /// This is a read-only check -- it clones the genome to avoid mutation.
    pub fn is_feasible(&mut self, genome: &[f64]) -> bool {
        let mut clone = genome.to_vec();
        let penalty = self.apply_all(&mut clone);
        penalty == 0.0
    }

    /// Return violation rate per constraint (violations / total applications).
    pub fn violation_rates(&self) -> HashMap<String, f64> {
        if self.applications == 0 {
            return HashMap::new();
        }
        self.violation_counts
            .iter()
            .map(|(name, count)| (name.clone(), *count as f64 / self.applications as f64))
            .collect()
    }

    /// Number of times apply_all has been called.
    pub fn applications(&self) -> usize {
        self.applications
    }

    /// Number of constraints in this set.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

impl Default for ConstraintSet {
    fn default() -> Self {
        ConstraintSet::new()
    }
}

// ---------------------------------------------------------------------------
// Standard constraint set for LARSA genomes
// ---------------------------------------------------------------------------

/// Build the default constraint set for LARSA v18 genomes.
///
/// Applies:
///   1. BoundsConstraint clamping all 15 parameters to their declared ranges
///   2. ConflictConstraint: if bh_collapse > 0.95, bh_form must be <= 2.2
///      (prevents extremely greedy BH formation with very slow collapse)
///   3. MonotonicityConstraint on cf_scale indices [7,8,9] (15m <= 1h <= 1d)
///      to prevent scale inversions
pub fn default_larsa_constraints() -> ConstraintSet {
    use crate::genome::PARAM_META;

    let bounds: Vec<(f64, f64)> = PARAM_META.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();

    let mut cs = ConstraintSet::new();

    cs.add(Box::new(BoundsConstraint::new(bounds)));

    // ConflictConstraint: gene[2] (bh_collapse) > 0.95 => gene[0] (bh_form) <= 2.2
    cs.add(Box::new(ConflictConstraint::new(
        2,     // bh_collapse index
        0.95,  // threshold_trigger
        0,     // bh_form index
        2.2,   // threshold_suppressed
    )));

    // MonotonicityConstraint: cf_scale indices 7, 8, 9 (15m <= 1h <= 1d)
    cs.add(Box::new(MonotonicityConstraint::new(7, 10)));

    cs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_constraint_clamps_values() {
        let bounds = vec![(0.0, 1.0); 5];
        let bc = BoundsConstraint::new(bounds);
        let mut genome = vec![-1.0, 0.5, 2.0, 0.0, 1.0];
        let penalty = bc.apply(&mut genome);
        assert!(penalty > 0.0, "should have nonzero penalty for out-of-bounds values");
        for &g in &genome {
            assert!(g >= 0.0 && g <= 1.0, "gene {} out of [0,1]", g);
        }
    }

    #[test]
    fn sum_constraint_normalizes() {
        let sc = SumConstraint::all_genes(4, 1.0);
        let mut genome = vec![1.0, 1.0, 1.0, 1.0];
        sc.apply(&mut genome);
        let s: f64 = genome.iter().sum();
        assert!((s - 1.0).abs() < 1e-9, "sum should be 1.0 after repair, got {}", s);
    }

    #[test]
    fn monotonicity_constraint_sorts() {
        let mc = MonotonicityConstraint::new(0, 4);
        let mut genome = vec![4.0, 1.0, 3.0, 2.0, 9.9];
        let penalty = mc.apply(&mut genome);
        assert!(penalty > 0.0);
        // First 4 should be sorted
        assert!(genome[0] <= genome[1] && genome[1] <= genome[2] && genome[2] <= genome[3]);
        // Last gene untouched
        assert!((genome[4] - 9.9).abs() < 1e-10);
    }

    #[test]
    fn conflict_constraint_repairs() {
        // trigger_idx=0, threshold_trigger=0.5, suppressed_idx=1, threshold_suppressed=0.3
        let cc = ConflictConstraint::new(0, 0.5, 1, 0.3);
        let mut genome = vec![0.8, 0.9, 0.5]; // gene[0] > 0.5, gene[1] > 0.3 => conflict
        let penalty = cc.apply(&mut genome);
        assert!(penalty > 0.0, "should have penalty");
        assert!(genome[1] <= 0.3 + 1e-10, "gene[1] should be clamped to 0.3");
    }

    #[test]
    fn death_penalty_kills_infeasible() {
        let dp = DeathPenalty::new("test_death", |g: &[f64]| g[0] < 0.5);
        let mut genome = vec![0.8f64]; // infeasible (0.8 >= 0.5)
        let penalty = dp.apply(&mut genome);
        assert_eq!(penalty, f64::INFINITY);
    }

    #[test]
    fn constraint_set_short_circuits_on_hard() {
        let mut cs = ConstraintSet::new();
        cs.add(Box::new(DeathPenalty::new("always_kill", |_| false)));
        cs.add(Box::new(BoundsConstraint::new(vec![(0.0, 1.0); 3])));

        let mut genome = vec![0.5, 0.5, 0.5];
        let penalty = cs.apply_all(&mut genome);
        assert_eq!(penalty, f64::INFINITY);
    }

    #[test]
    fn constraint_set_accumulates_soft_penalties() {
        let mut cs = ConstraintSet::new();
        cs.add(Box::new(PenaltyMethod::new(
            "sum_penalty",
            5.0,
            |g: &[f64]| {
                let s: f64 = g.iter().sum();
                (s - 1.0).abs()
            },
        )));

        let mut genome = vec![0.5, 0.5, 0.5]; // sum = 1.5, violation = 0.5
        let penalty = cs.apply_all(&mut genome);
        assert!((penalty - 2.5).abs() < 1e-9, "expected penalty 2.5, got {}", penalty);
    }

    #[test]
    fn violation_rates_tracked() {
        let mut cs = ConstraintSet::new();
        cs.add(Box::new(BoundsConstraint::new(vec![(0.0, 1.0); 3])));

        let mut g1 = vec![2.0, 0.5, 0.5]; // violates
        let mut g2 = vec![0.5, 0.5, 0.5]; // ok
        cs.apply_all(&mut g1);
        cs.apply_all(&mut g2);

        let rates = cs.violation_rates();
        let bc_rate = rates.get("BoundsConstraint").copied().unwrap_or(0.0);
        assert!((bc_rate - 0.5).abs() < 1e-9, "expected 50% violation rate");
    }
}
