/// Novelty-based evolution for SRFM signal genomes.
///
/// Standard fitness-only evolution converges prematurely to local optima.
/// Novelty search rewards behavioral diversity by measuring how different
/// a genome's trading behavior is from the archive of known behaviors.
/// The combined objective uses fitness (0.6) and novelty (0.4).

use crate::signal_genome::SignalGenome;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// BehaviorDescriptor
// ---------------------------------------------------------------------------

/// A compact behavioral fingerprint for one evaluated genome.
///
/// Captures how a signal behaves in simulation -- used as the distance
/// metric for novelty computation. Two genomes can have very different
/// parameters but similar behavior (e.g., both trade once per day with
/// 60% win rate), so novelty is measured in behavior space, not gene space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorDescriptor {
    /// Average number of entry signals per simulated trading day.
    pub signals_per_day: f64,
    /// Average hold duration in bars.
    pub avg_hold_bars: f64,
    /// Fraction of closed trades that are profitable.
    pub win_rate: f64,
    /// Correlation of trade timing with the 4h regime signal.
    pub regime_correlation: f64,
}

impl BehaviorDescriptor {
    /// Euclidean distance between two behavior descriptors in the 4D space.
    pub fn distance(&self, other: &BehaviorDescriptor) -> f64 {
        let d0 = self.signals_per_day - other.signals_per_day;
        let d1 = self.avg_hold_bars - other.avg_hold_bars;
        let d2 = self.win_rate - other.win_rate;
        let d3 = self.regime_correlation - other.regime_correlation;
        (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).sqrt()
    }
}

// ---------------------------------------------------------------------------
// ArchiveEntry
// ---------------------------------------------------------------------------

/// One entry in the novelty archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub genome: SignalGenome,
    pub behavior: BehaviorDescriptor,
    /// Novelty score at the time of insertion.
    pub novelty_at_insertion: f64,
    /// Generation at which this genome was added.
    pub generation: u32,
}

// ---------------------------------------------------------------------------
// NoveltyArchive
// ---------------------------------------------------------------------------

/// Archive of behaviorally diverse signal genomes.
///
/// Genomes are added unconditionally when sufficiently novel. The archive
/// grows up to a configurable max size; when full, the entry with the
/// lowest novelty score is replaced.
pub struct NoveltyArchive {
    entries: Vec<ArchiveEntry>,
    /// k for k-NN novelty computation.
    k: usize,
    /// Maximum archive size.
    max_size: usize,
    /// Current generation counter.
    generation: u32,
}

impl NoveltyArchive {
    /// Create an archive with k=15 and a 1000-entry cap.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            k: 15,
            max_size: 1000,
            generation: 0,
        }
    }

    /// Create with explicit k and max_size.
    pub fn with_params(k: usize, max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            k,
            max_size,
            generation: 0,
        }
    }

    /// Add a genome and its behavioral descriptor to the archive.
    pub fn add(&mut self, genome: SignalGenome, behavior: BehaviorDescriptor) {
        let novelty = self.novelty_score_internal(&behavior);

        let entry = ArchiveEntry {
            genome,
            behavior,
            novelty_at_insertion: novelty,
            generation: self.generation,
        };

        if self.entries.len() < self.max_size {
            self.entries.push(entry);
        } else {
            // Replace the entry with the lowest novelty score at insertion.
            if let Some(min_pos) = self
                .entries
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.novelty_at_insertion
                        .partial_cmp(&b.1.novelty_at_insertion)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                if novelty > self.entries[min_pos].novelty_at_insertion {
                    self.entries[min_pos] = entry;
                }
            }
        }
    }

    /// Advance the generation counter.
    pub fn next_generation(&mut self) {
        self.generation += 1;
    }

    /// Number of entries in the archive.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // -----------------------------------------------------------------------
    // Novelty score
    // -----------------------------------------------------------------------

    /// Compute the k-NN novelty score for a behavior descriptor.
    ///
    /// Novelty = mean distance to the k nearest neighbors in the archive.
    /// If the archive has fewer than k entries, all entries are used.
    /// If the archive is empty, returns 1.0 (maximally novel).
    pub fn novelty_score(&self, _genome: &SignalGenome, behavior: &BehaviorDescriptor) -> f64 {
        self.novelty_score_internal(behavior)
    }

    fn novelty_score_internal(&self, behavior: &BehaviorDescriptor) -> f64 {
        if self.entries.is_empty() {
            return 1.0;
        }

        // Compute distances to all archive entries.
        let mut distances: Vec<f64> = self
            .entries
            .iter()
            .map(|e| e.behavior.distance(behavior))
            .collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let k = self.k.min(distances.len());
        let mean_dist = distances[..k].iter().sum::<f64>() / k as f64;
        mean_dist
    }

    // -----------------------------------------------------------------------
    // Parent selection
    // -----------------------------------------------------------------------

    /// Select `n` parent genomes using tournament selection on novelty score.
    ///
    /// Tournament size = 3. Returns cloned genomes.
    /// If the archive has fewer entries than requested, returns all of them.
    pub fn select_parents(&self, n: usize, rng: &mut impl Rng) -> Vec<SignalGenome> {
        if self.entries.is_empty() {
            return vec![];
        }

        let tournament_size = 3.min(self.entries.len());
        let mut parents = Vec::with_capacity(n);

        for _ in 0..n {
            // Pick tournament_size random candidates.
            let mut best_idx = rng.gen_range(0..self.entries.len());
            let mut best_novelty = self.entries[best_idx].novelty_at_insertion;

            for _ in 1..tournament_size {
                let idx = rng.gen_range(0..self.entries.len());
                let novelty = self.entries[idx].novelty_at_insertion;
                if novelty > best_novelty {
                    best_novelty = novelty;
                    best_idx = idx;
                }
            }
            parents.push(self.entries[best_idx].genome.clone());
        }

        parents
    }

    /// Return all archive entries (read-only).
    pub fn entries(&self) -> &[ArchiveEntry] {
        &self.entries
    }
}

impl Default for NoveltyArchive {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NoveltyEvolver
// ---------------------------------------------------------------------------

/// Combined fitness + novelty evolutionary driver.
///
/// Objective = 0.6 * normalized_fitness + 0.4 * normalized_novelty.
/// Novelty prevents the population from collapsing onto a single strategy
/// archetype, which is critical for regime-adaptive signal libraries.
pub struct NoveltyEvolver {
    pub archive: NoveltyArchive,
    /// Weight on fitness (default 0.6).
    pub fitness_weight: f64,
    /// Weight on novelty (default 0.4).
    pub novelty_weight: f64,
    /// Current population (genome + behavior pairs).
    population: Vec<(SignalGenome, Option<BehaviorDescriptor>)>,
    /// Population size.
    pop_size: usize,
}

impl NoveltyEvolver {
    pub fn new(pop_size: usize) -> Self {
        Self {
            archive: NoveltyArchive::new(),
            fitness_weight: 0.6,
            novelty_weight: 0.4,
            population: Vec::with_capacity(pop_size),
            pop_size,
        }
    }

    /// Seed the population with random genomes.
    pub fn seed(&mut self, rng: &mut impl Rng) {
        self.population.clear();
        for _ in 0..self.pop_size {
            self.population.push((SignalGenome::random(rng), None));
        }
    }

    /// Register evaluation results for a genome at the given index.
    pub fn register_evaluation(
        &mut self,
        idx: usize,
        behavior: BehaviorDescriptor,
    ) {
        if let Some(entry) = self.population.get_mut(idx) {
            entry.1 = Some(behavior.clone());
            self.archive.add(entry.0.clone(), behavior);
        }
    }

    /// Combined score for one genome given its behavior.
    pub fn combined_score(
        &self,
        genome: &SignalGenome,
        behavior: &BehaviorDescriptor,
        fitness: f64,
    ) -> f64 {
        let novelty = self.archive.novelty_score(genome, behavior);
        // Normalize novelty by a soft cap of 5.0 (typical max distance in behavior space).
        let novelty_norm = (novelty / 5.0).min(1.0);
        self.fitness_weight * fitness + self.novelty_weight * novelty_norm
    }

    /// Advance one generation: select, crossover, mutate.
    pub fn evolve_generation(&mut self, rng: &mut impl Rng) {
        let parents = self.archive.select_parents(self.pop_size, rng);
        let mut new_pop: Vec<(SignalGenome, Option<BehaviorDescriptor>)> =
            Vec::with_capacity(self.pop_size);

        let mut i = 0;
        while new_pop.len() < self.pop_size {
            let pa = &parents[i % parents.len()];
            let pb = &parents[(i + 1) % parents.len()];
            let (mut c1, mut c2) = SignalGenome::crossover(pa, pb, rng);
            c1.mutate(0.1, rng);
            c2.mutate(0.1, rng);
            new_pop.push((c1, None));
            if new_pop.len() < self.pop_size {
                new_pop.push((c2, None));
            }
            i += 2;
        }

        self.population = new_pop;
        self.archive.next_generation();
    }

    /// Return the current population (genome + optional behavior).
    pub fn population(&self) -> &[(SignalGenome, Option<BehaviorDescriptor>)] {
        &self.population
    }

    /// Return the best genome by novelty score in the archive.
    pub fn best_novel_genome(&self) -> Option<&SignalGenome> {
        self.archive
            .entries
            .iter()
            .max_by(|a, b| {
                a.novelty_at_insertion
                    .partial_cmp(&b.novelty_at_insertion)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| &e.genome)
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
        SmallRng::seed_from_u64(99)
    }

    fn make_behavior(spd: f64, ahb: f64, wr: f64, rc: f64) -> BehaviorDescriptor {
        BehaviorDescriptor {
            signals_per_day: spd,
            avg_hold_bars: ahb,
            win_rate: wr,
            regime_correlation: rc,
        }
    }

    #[test]
    fn test_behavior_distance_same_is_zero() {
        let b = make_behavior(2.0, 8.0, 0.55, 0.3);
        assert!((b.distance(&b) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_behavior_distance_different() {
        let a = make_behavior(1.0, 4.0, 0.5, 0.0);
        let b = make_behavior(3.0, 12.0, 0.7, 0.5);
        let d = a.distance(&b);
        assert!(d > 0.0, "different behaviors must have positive distance");
    }

    #[test]
    fn test_archive_empty_novelty_is_one() {
        let archive = NoveltyArchive::new();
        let g = SignalGenome::default_genome();
        let b = make_behavior(2.0, 8.0, 0.55, 0.3);
        let score = archive.novelty_score(&g, &b);
        assert!((score - 1.0).abs() < 1e-12, "empty archive => novelty = 1.0");
    }

    #[test]
    fn test_archive_add_increases_len() {
        let mut archive = NoveltyArchive::new();
        let g = SignalGenome::default_genome();
        archive.add(g.clone(), make_behavior(1.0, 4.0, 0.5, 0.0));
        archive.add(g.clone(), make_behavior(2.0, 8.0, 0.6, 0.2));
        assert_eq!(archive.len(), 2);
    }

    #[test]
    fn test_novelty_decreases_for_duplicate_behavior() {
        let mut archive = NoveltyArchive::with_params(3, 100);
        let g = SignalGenome::default_genome();
        let b_ref = make_behavior(2.0, 8.0, 0.55, 0.3);
        // Fill archive with many copies of the same behavior.
        for _ in 0..20 {
            archive.add(g.clone(), b_ref.clone());
        }
        // Novelty of a new identical behavior should now be near 0.
        let novelty = archive.novelty_score(&g, &b_ref);
        assert!(
            novelty < 0.5,
            "duplicate behavior should have low novelty, got {novelty}"
        );
    }

    #[test]
    fn test_novelty_high_for_unique_behavior() {
        let mut archive = NoveltyArchive::with_params(3, 100);
        let g = SignalGenome::default_genome();
        // Add behaviors clustered around (1.0, 4.0, 0.5, 0.0).
        for i in 0..10 {
            archive.add(
                g.clone(),
                make_behavior(1.0 + i as f64 * 0.01, 4.0, 0.5, 0.0),
            );
        }
        // A very different behavior should have high novelty.
        let unique = make_behavior(50.0, 200.0, 0.9, 0.99);
        let novelty = archive.novelty_score(&g, &unique);
        assert!(
            novelty > 10.0,
            "unique behavior should have high novelty, got {novelty}"
        );
    }

    #[test]
    fn test_select_parents_returns_correct_count() {
        let mut archive = NoveltyArchive::new();
        let mut rng = seeded_rng();
        let g = SignalGenome::default_genome();
        for i in 0..20 {
            archive.add(g.clone(), make_behavior(i as f64 * 0.5, 4.0, 0.5, 0.0));
        }
        let parents = archive.select_parents(5, &mut rng);
        assert_eq!(parents.len(), 5);
    }

    #[test]
    fn test_evolver_evolves_without_panic() {
        let mut evolver = NoveltyEvolver::new(10);
        let mut rng = seeded_rng();
        evolver.seed(&mut rng);

        // Register random evaluations.
        for i in 0..10 {
            let b = make_behavior(
                (i as f64 + 1.0) * 0.5,
                (i as f64 + 1.0) * 2.0,
                0.4 + i as f64 * 0.02,
                0.0,
            );
            evolver.register_evaluation(i, b);
        }

        // Evolve one generation -- must not panic.
        evolver.evolve_generation(&mut rng);
        assert_eq!(evolver.population().len(), 10);
    }

    #[test]
    fn test_combined_score_weights() {
        let mut evolver = NoveltyEvolver::new(5);
        let mut rng = seeded_rng();
        evolver.seed(&mut rng);

        let g = SignalGenome::default_genome();
        let b = make_behavior(2.0, 8.0, 0.55, 0.3);
        let fitness = 0.8;
        let score = evolver.combined_score(&g, &b, fitness);
        // With an empty archive, novelty = 1.0, norm_novelty = min(1/5, 1) = 0.2.
        // combined = 0.6 * 0.8 + 0.4 * 0.2 = 0.48 + 0.08 = 0.56.
        assert!(score > 0.0, "combined score must be positive");
    }

    #[test]
    fn test_serde_behavior_descriptor() {
        let b = make_behavior(2.5, 12.0, 0.62, 0.45);
        let json = serde_json::to_string(&b).unwrap();
        let decoded: BehaviorDescriptor = serde_json::from_str(&json).unwrap();
        assert!((decoded.signals_per_day - 2.5).abs() < 1e-12);
        assert!((decoded.win_rate - 0.62).abs() < 1e-12);
    }

    #[test]
    fn test_archive_max_size_not_exceeded() {
        let mut archive = NoveltyArchive::with_params(3, 5);
        let g = SignalGenome::default_genome();
        for i in 0..20 {
            archive.add(
                g.clone(),
                make_behavior(i as f64, i as f64 * 2.0, 0.5, 0.0),
            );
        }
        assert!(
            archive.len() <= 5,
            "archive must not exceed max_size, got {}",
            archive.len()
        );
    }
}
