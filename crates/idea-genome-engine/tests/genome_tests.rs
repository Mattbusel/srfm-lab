/// Integration-level tests for the genome engine.
///
/// These live in the `tests/` directory so they exercise the public API
/// exactly as an external consumer would.

use idea_genome_engine::{
    fitness::{EvaluatorConfig, FitnessEvaluator, FitnessVec},
    genome::{Genome, N_PARAMS, PARAM_META},
    operators::{
        crossover::{blx_alpha, sbx},
        mutation::{adaptive_mutate, gaussian_mutate},
        selection::tournament_select,
    },
    population::Population,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn seeded_rng(seed: u64) -> SmallRng {
    SmallRng::seed_from_u64(seed)
}

fn dry_evaluator() -> FitnessEvaluator {
    FitnessEvaluator::new(EvaluatorConfig {
        dry_run: true,
        ..Default::default()
    })
}

fn evaluated_population(size: usize, seed: u64) -> Population {
    let mut rng = seeded_rng(seed);
    let mut pop = Population::new(size, &mut rng);
    pop.evaluate_all_parallel(&dry_evaluator());
    pop
}

// ---------------------------------------------------------------------------
// Genome tests
// ---------------------------------------------------------------------------

#[test]
fn genome_random_correct_length() {
    let mut rng = seeded_rng(1);
    let g = Genome::new_random(&mut rng);
    assert_eq!(g.parameters.len(), N_PARAMS);
    assert_eq!(g.param_names.len(), N_PARAMS);
    assert_eq!(g.bounds.len(), N_PARAMS);
}

#[test]
fn genome_random_within_bounds() {
    let mut rng = seeded_rng(2);
    for _ in 0..50 {
        let g = Genome::new_random(&mut rng);
        assert!(g.is_within_bounds(), "Random genome violates bounds");
    }
}

#[test]
fn genome_clamp_enforces_bounds() {
    let mut rng = seeded_rng(3);
    let mut g = Genome::new_random(&mut rng);
    // Force every gene to a huge value.
    for p in g.parameters.iter_mut() {
        *p = 1e18;
    }
    g.clamp();
    for (i, (&v, (lo, hi))) in g.parameters.iter().zip(g.bounds.iter()).enumerate() {
        assert!(
            v >= *lo && v <= *hi,
            "gene {} = {} is outside [{}, {}]",
            i,
            v,
            lo,
            hi
        );
    }
}

#[test]
fn genome_serialisation_round_trip() {
    let mut rng = seeded_rng(4);
    let original = Genome::new_random(&mut rng);
    let json = original.to_json();
    let restored = Genome::from_json(&json).expect("deserialisation failed");

    assert_eq!(original.id, restored.id);
    assert_eq!(original.generation, restored.generation);
    assert_eq!(original.parameters.len(), restored.parameters.len());
    for (a, b) in original.parameters.iter().zip(restored.parameters.iter()) {
        assert!(
            (a - b).abs() < 1e-12,
            "parameter mismatch after round-trip: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn genome_param_index_all_names() {
    let mut rng = seeded_rng(5);
    let g = Genome::new_random(&mut rng);
    for (name, _, _) in PARAM_META.iter() {
        assert!(
            g.param_index(name).is_some(),
            "param_index returned None for '{}'",
            name
        );
    }
    assert_eq!(g.param_index("bh_form"), Some(0));
    assert_eq!(g.param_index("pos_floor_frac"), Some(14));
}

#[test]
fn genome_param_index_unknown_returns_none() {
    let mut rng = seeded_rng(6);
    let g = Genome::new_random(&mut rng);
    assert_eq!(g.param_index("not_a_real_param"), None);
}

#[test]
fn genome_get_param_values_in_range() {
    let mut rng = seeded_rng(7);
    let g = Genome::new_random(&mut rng);
    for (name, lo, hi) in PARAM_META.iter() {
        let v = g.get_param(name).expect("get_param should return Some");
        assert!(
            v >= *lo && v <= *hi,
            "param '{}' = {} outside [{}, {}]",
            name,
            v,
            lo,
            hi
        );
    }
}

#[test]
fn genome_to_param_map_has_all_keys() {
    let mut rng = seeded_rng(8);
    let g = Genome::new_random(&mut rng);
    let map = g.to_param_map();
    for (name, _, _) in PARAM_META.iter() {
        assert!(
            map.get(name).is_some(),
            "param_map missing key '{}'",
            name
        );
    }
}

#[test]
fn genome_dominance_ordering() {
    // Construct two genomes with deterministic synthetic fitness and verify dominance.
    let mut rng = seeded_rng(9);
    let eval = dry_evaluator();

    let mut g_good = Genome::new_random(&mut rng);
    let mut g_bad = Genome::new_random(&mut rng);

    g_good.fitness = Some(FitnessVec {
        sharpe: 2.0,
        max_dd: 0.05,
        calmar: 5.0,
        win_rate: 0.65,
        profit_factor: 2.5,
        n_trades: 100,
        is_oos_spread: 0.1,
    });
    g_bad.fitness = Some(FitnessVec {
        sharpe: 0.5,
        max_dd: 0.40,
        calmar: 0.8,
        win_rate: 0.45,
        profit_factor: 1.1,
        n_trades: 20,
        is_oos_spread: 0.6,
    });

    // Suppress unused warning — evaluator is needed for type consistency in some tests.
    let _ = eval;

    assert!(g_good.dominates(&g_bad), "g_good should dominate g_bad");
    assert!(!g_bad.dominates(&g_good), "g_bad should not dominate g_good");
}

// ---------------------------------------------------------------------------
// Crossover tests
// ---------------------------------------------------------------------------

#[test]
fn blx_children_parameters_in_bounds() {
    let mut rng = seeded_rng(20);
    let pa = Genome::new_random(&mut rng);
    let pb = Genome::new_random(&mut rng);

    for _ in 0..100 {
        let (ca, cb) = blx_alpha(&pa, &pb, 0.3, &mut rng);
        assert!(ca.is_within_bounds(), "BLX child A out of bounds");
        assert!(cb.is_within_bounds(), "BLX child B out of bounds");
    }
}

#[test]
fn blx_children_inherit_parent_ids() {
    let mut rng = seeded_rng(21);
    let pa = Genome::new_random(&mut rng);
    let pb = Genome::new_random(&mut rng);
    let (ca, cb) = blx_alpha(&pa, &pb, 0.3, &mut rng);

    assert!(
        ca.parent_ids.contains(&pa.id) && ca.parent_ids.contains(&pb.id),
        "child A missing parent IDs"
    );
    assert!(
        cb.parent_ids.contains(&pa.id) && cb.parent_ids.contains(&pb.id),
        "child B missing parent IDs"
    );
}

#[test]
fn sbx_children_parameters_in_bounds() {
    let mut rng = seeded_rng(22);
    let pa = Genome::new_random(&mut rng);
    let pb = Genome::new_random(&mut rng);

    for _ in 0..100 {
        let (ca, cb) = sbx(&pa, &pb, 2.0, &mut rng);
        assert!(ca.is_within_bounds(), "SBX child A out of bounds");
        assert!(cb.is_within_bounds(), "SBX child B out of bounds");
    }
}

#[test]
fn crossover_generation_incremented() {
    let mut rng = seeded_rng(23);
    let mut pa = Genome::new_random(&mut rng);
    let pb = Genome::new_random(&mut rng);
    pa.generation = 5;

    let (ca, _) = blx_alpha(&pa, &pb, 0.3, &mut rng);
    assert!(ca.generation > pa.generation, "child generation should exceed parent");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

#[test]
fn gaussian_mutation_preserves_bounds() {
    let mut rng = seeded_rng(30);
    for _ in 0..100 {
        let mut g = Genome::new_random(&mut rng);
        gaussian_mutate(&mut g, 0.05, 0.3, &mut rng);
        assert!(g.is_within_bounds(), "Gaussian mutation violated bounds");
    }
}

#[test]
fn gaussian_mutation_rate_zero_no_change() {
    let mut rng = seeded_rng(31);
    let mut g = Genome::new_random(&mut rng);
    let before = g.parameters.clone();
    gaussian_mutate(&mut g, 0.05, 0.0, &mut rng);
    assert_eq!(g.parameters, before, "rate=0 should not change parameters");
}

#[test]
fn adaptive_mutation_preserves_bounds() {
    let mut rng = seeded_rng(32);
    for gen in [0u32, 10, 25, 49, 50] {
        let mut g = Genome::new_random(&mut rng);
        adaptive_mutate(&mut g, gen, 50, &mut rng);
        assert!(
            g.is_within_bounds(),
            "adaptive mutation violated bounds at gen {}",
            gen
        );
    }
}

// ---------------------------------------------------------------------------
// Population diversity tests
// ---------------------------------------------------------------------------

#[test]
fn population_diversity_non_negative() {
    let pop = evaluated_population(20, 40);
    let d = pop.diversity_metric();
    assert!(d >= 0.0, "diversity must be non-negative");
}

#[test]
fn identical_population_has_zero_diversity() {
    let mut rng = seeded_rng(41);
    let template = Genome::new_random(&mut rng);
    let genomes: Vec<Genome> = (0..10).map(|_| template.clone()).collect();
    let pop = Population::from_genomes(genomes, 0);
    // Without evaluation there are no pairs to compare → 0.
    assert_eq!(pop.diversity_metric(), 0.0);
}

#[test]
fn large_population_has_positive_diversity() {
    let pop = evaluated_population(50, 42);
    let d = pop.diversity_metric();
    // A random population should have non-trivial diversity.
    assert!(d > 0.0, "large random population should have positive diversity");
}

// ---------------------------------------------------------------------------
// Tournament selection tests
// ---------------------------------------------------------------------------

#[test]
fn tournament_always_returns_evaluated_winner() {
    let pop = evaluated_population(30, 50);
    let mut rng = seeded_rng(99);
    for _ in 0..50 {
        let winner = tournament_select(&pop, 5, &mut rng);
        assert!(winner.fitness.is_some(), "tournament winner has no fitness");
    }
}

#[test]
fn tournament_size_1_is_random_selection() {
    // With tournament size 1, every genome has an equal chance of being selected.
    let pop = evaluated_population(20, 51);
    let mut rng = seeded_rng(123);
    let mut seen_ids = std::collections::HashSet::new();

    for _ in 0..500 {
        let winner = tournament_select(&pop, 1, &mut rng);
        seen_ids.insert(winner.id.clone());
    }

    // With 500 draws from 20 genomes, we expect to see most of them.
    assert!(
        seen_ids.len() >= 15,
        "tournament_size=1 should select diverse genomes (saw {} unique)",
        seen_ids.len()
    );
}

#[test]
fn tournament_prefers_higher_sharpe() {
    let pop = evaluated_population(30, 52);
    let mut rng = seeded_rng(200);

    let evaluated: Vec<&Genome> = pop.genomes.iter().filter(|g| g.fitness.is_some()).collect();
    let mut sharpes: Vec<f64> = evaluated.iter().map(|g| g.sharpe()).collect();
    sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sharpes[sharpes.len() / 2];

    let trials = 200;
    let above_median = (0..trials)
        .filter(|_| tournament_select(&pop, 5, &mut rng).sharpe() >= median)
        .count();

    assert!(
        above_median > trials * 6 / 10,
        "tournament should favour above-median genomes ({}/{} trials)",
        above_median,
        trials
    );
}

// ---------------------------------------------------------------------------
// Full-pipeline smoke test
// ---------------------------------------------------------------------------

#[test]
fn full_evolution_dry_run_completes() {
    use idea_genome_engine::evolution::GeneticEvolver;

    let eval = dry_evaluator();
    let mut evolver = GeneticEvolver::new(eval);
    evolver.population_size = 12;
    evolver.n_generations = 2;
    evolver.use_islands = false;

    let result = evolver.run();

    assert!(result.global_best.fitness.is_some(), "global_best has no fitness");
    assert!(!result.pareto_front.is_empty(), "pareto_front is empty");
    assert_eq!(result.island_results.len(), 1, "single-pop mode should have 1 island result");
}
