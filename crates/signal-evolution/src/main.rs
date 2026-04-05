/// signal-evolution CLI
///
/// Usage:
///   signal-evolution --data <csv> --generations <n> --population <n> --output <json>
///
/// Runs the genetic programming engine and writes discovered signals to JSON.

use anyhow::Result;
use clap::Parser;
use signal_evolution::data_loader::load_csv;
use signal_evolution::evolution::{run_evolution, EvolutionConfig};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[derive(Parser, Debug)]
#[command(
    name = "signal-evolution",
    about = "Genetic programming engine for synthesising trading signals"
)]
struct Args {
    /// Path to OHLCV CSV file
    #[arg(long, default_value = "data/crypto_trades.csv")]
    data: String,

    /// Number of evolution generations
    #[arg(long, default_value_t = 50)]
    generations: usize,

    /// Population size
    #[arg(long, default_value_t = 100)]
    population: usize,

    /// Output JSON path (stdout if not provided)
    #[arg(long)]
    output: Option<String>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Maximum tree depth
    #[arg(long, default_value_t = 6)]
    max_depth: usize,

    /// Crossover probability
    #[arg(long, default_value_t = 0.8)]
    crossover_prob: f64,

    /// Mutation probability
    #[arg(long, default_value_t = 0.3)]
    mutation_prob: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading bar data from: {}", args.data);
    let bars = load_csv(&args.data)?;
    eprintln!("Loaded {} bars", bars.len());

    if bars.len() < 50 {
        anyhow::bail!("Need at least 50 bars for meaningful evolution. Got {}", bars.len());
    }

    let config = EvolutionConfig {
        population_size: args.population,
        generations: args.generations,
        crossover_prob: args.crossover_prob,
        mutation_prob: args.mutation_prob,
        max_depth: args.max_depth,
        ..Default::default()
    };

    eprintln!(
        "Starting evolution: {} generations, {} individuals, max_depth={}",
        config.generations, config.population_size, config.max_depth
    );

    let mut rng = SmallRng::seed_from_u64(args.seed);
    let result = run_evolution(&bars, config, &mut rng);

    // Output
    let json = serde_json::to_string_pretty(&result)?;

    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("Results written to: {path}");
        }
        None => println!("{json}"),
    }

    // Print summary table to stderr
    eprintln!("\n=== Top Discovered Signals ===");
    eprintln!("{:<5} {:<8} {:<8} {:<8} {:<8} {}", "Rank", "IC", "ICIR", "Sharpe", "Cmplx", "Formula");
    for (rank, sig) in result.best_signals.iter().enumerate() {
        eprintln!(
            "{:<5} {:+.4} {:+.4} {:+.4} {:<8} {}",
            rank + 1,
            sig.ic,
            sig.icir,
            sig.sharpe_contrib,
            sig.complexity,
            &sig.formula[..sig.formula.len().min(60)]
        );
    }

    Ok(())
}
