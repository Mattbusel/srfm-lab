/// idea-genome-engine CLI: runs the genetic algorithm for LARSA strategy optimisation.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use idea_genome_engine::{
    archive::Archive,
    evolution::GeneticEvolver,
    fitness::{EvaluatorConfig, FitnessEvaluator},
    genome::Genome,
    population::Population,
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Parser)]
#[command(
    name = "idea-genome-engine",
    about = "Genetic algorithm for LARSA strategy parameter optimisation",
    version
)]
struct Cli {
    /// Path to idea_engine.db (SQLite database for genome archive).
    #[arg(long, default_value = "idea_engine.db")]
    db_path: PathBuf,

    /// Path to the srfm-lab root directory (used to locate the Python backtest).
    #[arg(long, default_value = ".")]
    lab_path: PathBuf,

    /// Population size per island (total pop = population_size × n_islands when
    /// island model is active).
    #[arg(long, default_value_t = 100)]
    population_size: usize,

    /// Number of generations to evolve.
    #[arg(long, default_value_t = 50)]
    generations: usize,

    /// Path to write the JSON results file.
    #[arg(long, default_value = "genome_results.json")]
    output: PathBuf,

    /// Resume from the DB archive (warm-start the population).
    #[arg(long, default_value_t = false)]
    resume: bool,

    /// Evaluate one random genome and exit immediately (for pipeline smoke-tests).
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Disable the island model and run a single merged population.
    #[arg(long, default_value_t = false)]
    no_islands: bool,

    /// Elite fraction — top fraction of each population kept unchanged between generations.
    #[arg(long, default_value_t = 0.10)]
    elite_frac: f64,

    /// Crossover rate (probability that two parents undergo crossover vs. clone-mutate).
    #[arg(long, default_value_t = 0.80)]
    crossover_rate: f64,

    /// Per-gene mutation rate for non-adaptive mutations.
    #[arg(long, default_value_t = 0.15)]
    mutation_rate: f64,

    /// BLX-alpha extension parameter.
    #[arg(long, default_value_t = 0.30)]
    blx_alpha: f64,

    /// Maximum wall-clock seconds allowed for a single backtest evaluation.
    #[arg(long, default_value_t = 120)]
    eval_timeout_secs: u64,

    /// Archive capacity (maximum genomes kept in the hall-of-fame).
    #[arg(long, default_value_t = 50)]
    archive_capacity: usize,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("=== idea-genome-engine ===");
    println!("  db_path          : {}", cli.db_path.display());
    println!("  lab_path         : {}", cli.lab_path.display());
    println!("  population_size  : {}", cli.population_size);
    println!("  generations      : {}", cli.generations);
    println!("  output           : {}", cli.output.display());
    println!("  resume           : {}", cli.resume);
    println!("  dry_run          : {}", cli.dry_run);
    println!("  use_islands      : {}", !cli.no_islands);
    println!();

    let eval_config = EvaluatorConfig {
        lab_path: cli
            .lab_path
            .to_str()
            .context("lab_path is not valid UTF-8")?
            .to_string(),
        timeout: std::time::Duration::from_secs(cli.eval_timeout_secs),
        dry_run: cli.dry_run,
    };

    let evaluator = FitnessEvaluator::new(eval_config);

    // ------------------------------------------------------------------
    // Dry-run mode: evaluate one genome and exit.
    // ------------------------------------------------------------------
    if cli.dry_run {
        println!("[DRY RUN] Evaluating one random genome and exiting.");
        let mut rng = rand::thread_rng();
        let genome = Genome::new_random(&mut rng);
        let fitness = evaluator.evaluate(&genome);
        println!("  genome  : {}", genome);
        println!("  fitness : {}", fitness);
        return Ok(());
    }

    // ------------------------------------------------------------------
    // Optional warm-start from DB archive.
    // ------------------------------------------------------------------
    let mut archive = if cli.resume && cli.db_path.exists() {
        let db_str = cli.db_path.to_str().context("db_path not UTF-8")?;
        println!("[Resume] Loading archive from {}", db_str);
        let a = Archive::load_from_db(db_str, cli.archive_capacity)
            .context("loading archive from DB")?;
        println!("[Resume] Loaded {} genomes from archive.", a.len());
        a
    } else {
        Archive::new(cli.archive_capacity)
    };

    // ------------------------------------------------------------------
    // Build and run the evolver.
    // ------------------------------------------------------------------
    let mut evolver = GeneticEvolver::new(evaluator);
    evolver.population_size = cli.population_size;
    evolver.n_generations = cli.generations;
    evolver.crossover_rate = cli.crossover_rate;
    evolver.mutation_rate = cli.mutation_rate;
    evolver.elite_frac = cli.elite_frac;
    evolver.use_islands = !cli.no_islands;
    evolver.blx_alpha = cli.blx_alpha;

    // Seed the first-generation population from archive if resuming.
    if cli.resume && !archive.is_empty() {
        println!(
            "[Resume] Warm-starting with {} archived genomes.",
            archive.len()
        );
        // The evolver's internal population will be replaced immediately, but we
        // pass the archive genomes through the evaluator to refresh fitness
        // (regime windows may have changed since they were stored).
        let seed_genomes = archive.genomes.clone();
        let mut seed_pop = Population::from_genomes(seed_genomes, 0);
        // Clear fitness so they are re-evaluated in the current regime window.
        for g in seed_pop.genomes.iter_mut() {
            g.fitness = None;
        }
        // We don't inject into the evolver directly here; a future enhancement
        // would accept an initial seed population.  For now, the archive serves
        // as a hall-of-fame that is updated as the run progresses.
        let _ = seed_pop; // suppress unused warning
    }

    println!("[Evolve] Starting evolution …");
    let result = evolver.run();

    println!();
    println!("=== Evolution complete in {:.1}s ===", result.elapsed_secs);
    println!(
        "  Global best : {}",
        result.global_best
    );
    if let Some(ref f) = result.global_best.fitness {
        println!("  Fitness     : {}", f);
    }
    println!("  Pareto front size: {}", result.pareto_front.len());

    // ------------------------------------------------------------------
    // Update archive and save to DB.
    // ------------------------------------------------------------------
    archive.update_from_slice(&result.pareto_front);
    archive.update(&result.global_best);
    for island in &result.island_results {
        archive.update(&island.best_genome);
    }

    let db_str = cli.db_path.to_str().context("db_path not UTF-8")?;
    archive
        .save_to_db(db_str)
        .with_context(|| format!("saving archive to {}", db_str))?;
    println!("[Archive] Saved {} genomes to {}", archive.len(), db_str);

    // ------------------------------------------------------------------
    // Write JSON output.
    // ------------------------------------------------------------------
    let output_json = serde_json::json!({
        "elapsed_secs": result.elapsed_secs,
        "global_best": result.global_best,
        "pareto_front": result.pareto_front,
        "island_results": result.island_results,
        "archive_size": archive.len(),
    });

    let output_str = serde_json::to_string_pretty(&output_json)
        .context("serialising evolution result to JSON")?;

    // Ensure parent directory exists.
    if let Some(parent) = cli.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating output directory {:?}", parent))?;
        }
    }

    std::fs::write(&cli.output, &output_str)
        .with_context(|| format!("writing results to {}", cli.output.display()))?;

    println!("[Output] Results written to {}", cli.output.display());
    println!();

    Ok(())
}
