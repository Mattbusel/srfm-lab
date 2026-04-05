//! Monte Carlo Engine CLI
//!
//! Provides three subcommands:
//!
//! ```text
//! monte-carlo-engine simulate --returns returns.json --n-paths 10000 --output results.json
//! monte-carlo-engine stress-test --returns returns.json --scenario crash2020 --output stress.json
//! monte-carlo-engine bootstrap --returns returns.json --method stationary --n 5000 --output bootstrap.json
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use monte_carlo_engine::{
    bootstrap::{
        Bootstrapper, CircularBlockBootstrap, MovingBlockBootstrap, StationaryBootstrap,
        optimal_block_length, slice_mean,
    },
    simulation::{fit_distribution, MonteCarloSimulator, SimulationConfig},
    stress_tests::{scenario_analysis, StressScenario, StressTestEngine},
};
use serde::Serialize;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "monte-carlo-engine",
    version = "0.1.0",
    about = "Fast Monte Carlo simulation for strategy return distributions",
    long_about = "Runs Monte Carlo simulations, bootstrap resampling, and stress tests\n\
                  on strategy return series. Output is always JSON."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run Monte Carlo simulation on a return series.
    Simulate(SimulateArgs),
    /// Run stress-test scenarios against a return series.
    StressTest(StressTestArgs),
    /// Bootstrap-resample a return series.
    Bootstrap(BootstrapArgs),
    /// Fit distribution parameters to a return series and print them.
    Fit(FitArgs),
}

// ---------------------------------------------------------------------------
// simulate
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct SimulateArgs {
    /// Path to JSON file containing array of daily returns (e.g. [0.01, -0.02, ...]).
    #[arg(short, long)]
    returns: PathBuf,

    /// Number of Monte Carlo paths to simulate.
    #[arg(long, default_value_t = 10_000)]
    n_paths: usize,

    /// Number of bars (periods) per path. Defaults to len(returns).
    #[arg(long)]
    n_bars: Option<usize>,

    /// Starting equity for each path.
    #[arg(long, default_value_t = 100_000.0)]
    initial_equity: f64,

    /// Enable fat-tail blending (Student-t + Normal mix).
    #[arg(long, default_value_t = true)]
    fat_tails: bool,

    /// Weight on Student-t draws when fat_tails is enabled.
    #[arg(long, default_value_t = 0.25)]
    fat_tail_weight: f64,

    /// Store full equity paths in output (memory-intensive).
    #[arg(long, default_value_t = false)]
    store_paths: bool,

    /// Random seed (0 = non-deterministic).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Output JSON file path.
    #[arg(short, long)]
    output: PathBuf,
}

// ---------------------------------------------------------------------------
// stress-test
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct StressTestArgs {
    /// Path to JSON file containing array of daily returns.
    #[arg(short, long)]
    returns: PathBuf,

    /// Stress scenario to apply.
    #[arg(long, value_enum, num_args = 1.., required = true)]
    scenario: Vec<ScenarioCli>,

    /// Number of paths for stress simulation.
    #[arg(long, default_value_t = 5_000)]
    n_paths: usize,

    /// Number of bars per path.
    #[arg(long, default_value_t = 252)]
    n_bars: usize,

    /// Initial equity.
    #[arg(long, default_value_t = 100_000.0)]
    initial_equity: f64,

    /// Random seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Output JSON file path.
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Clone, ValueEnum, Debug)]
enum ScenarioCli {
    Crash2020,
    CryptoWinter2022,
    CovidDip,
    Gfc2008,
    FlashCrash,
    All,
}

impl ScenarioCli {
    fn to_scenarios(&self) -> Vec<StressScenario> {
        match self {
            ScenarioCli::Crash2020 => vec![StressScenario::MarketCrash2020],
            ScenarioCli::CryptoWinter2022 => vec![StressScenario::CryptoWinter2022],
            ScenarioCli::CovidDip => vec![StressScenario::CovidDip],
            ScenarioCli::Gfc2008 => vec![StressScenario::Gfc2008],
            ScenarioCli::FlashCrash => vec![StressScenario::FlashCrash],
            ScenarioCli::All => vec![
                StressScenario::MarketCrash2020,
                StressScenario::CryptoWinter2022,
                StressScenario::CovidDip,
                StressScenario::Gfc2008,
                StressScenario::FlashCrash,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// bootstrap
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct BootstrapArgs {
    /// Path to JSON file containing array of daily returns.
    #[arg(short, long)]
    returns: PathBuf,

    /// Bootstrap method to use.
    #[arg(long, value_enum, default_value_t = BootstrapMethod::Stationary)]
    method: BootstrapMethod,

    /// Number of resampled observations to generate.
    #[arg(short, long, default_value_t = 5_000)]
    n: usize,

    /// Block size (auto if not set).
    #[arg(long)]
    block_size: Option<usize>,

    /// Random seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Output JSON file path.
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Clone, ValueEnum, Debug)]
enum BootstrapMethod {
    Stationary,
    Circular,
    Moving,
}

// ---------------------------------------------------------------------------
// fit
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct FitArgs {
    /// Path to JSON file containing array of daily returns.
    #[arg(short, long)]
    returns: PathBuf,

    /// Output JSON file path (optional; prints to stdout if omitted).
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Simulate(args) => cmd_simulate(args),
        Commands::StressTest(args) => cmd_stress_test(args),
        Commands::Bootstrap(args) => cmd_bootstrap(args),
        Commands::Fit(args) => cmd_fit(args),
    }
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

fn cmd_simulate(args: SimulateArgs) -> Result<()> {
    let returns = load_returns(&args.returns)?;
    if returns.is_empty() {
        bail!("returns array is empty");
    }

    let n_bars = args.n_bars.unwrap_or(returns.len());
    let dist = fit_distribution(&returns);

    eprintln!(
        "Fitted distribution: mean={:.6}, std={:.6}, skew={:.4}, kurtosis={:.4}, tail_alpha={:.4}",
        dist.mean, dist.std, dist.skew, dist.kurtosis, dist.tail_alpha
    );

    let config = SimulationConfig {
        n_paths: args.n_paths,
        n_bars,
        initial_equity: args.initial_equity,
        use_fat_tails: args.fat_tails,
        fat_tail_weight: args.fat_tail_weight,
        store_paths: args.store_paths,
        seed: args.seed,
    };

    eprintln!("Running {} paths × {} bars ...", config.n_paths, config.n_bars);

    let sim = MonteCarloSimulator::new();
    let results = sim.run(&config, &dist);

    eprintln!(
        "Done. Median final equity: {:.2}, P5: {:.2}, P95: {:.2}",
        results.median_final_equity, results.p5_final_equity, results.p95_final_equity
    );
    eprintln!(
        "  P95 max drawdown: {:.2}%  Prob ruin: {:.2}%  Median Sharpe: {:.3}",
        results.p95_max_drawdown * 100.0,
        results.prob_ruin * 100.0,
        results.median_sharpe
    );

    write_json(&args.output, &results)?;
    eprintln!("Results written to {}", args.output.display());
    Ok(())
}

fn cmd_stress_test(args: StressTestArgs) -> Result<()> {
    let returns = load_returns(&args.returns)?;
    if returns.is_empty() {
        bail!("returns array is empty");
    }

    // Collect unique scenarios from the CLI list
    let mut scenarios: Vec<StressScenario> = Vec::new();
    for s in &args.scenario {
        scenarios.extend(s.to_scenarios());
    }

    eprintln!("Running {} stress scenario(s) with {} paths each ...", scenarios.len(), args.n_paths);

    let engine = StressTestEngine {
        n_paths: args.n_paths,
        n_bars: args.n_bars,
        initial_equity: args.initial_equity,
        seed: args.seed,
    };

    let results = scenario_analysis(&engine, &returns, &scenarios)?;

    for r in &results {
        eprintln!(
            "  [{}] median_eq={:.2}, p95_dd={:.2}%, prob_ruin={:.2}%",
            r.scenario_name,
            r.median_final_equity,
            r.p95_max_drawdown * 100.0,
            r.prob_ruin * 100.0
        );
    }

    write_json(&args.output, &results)?;
    eprintln!("Results written to {}", args.output.display());
    Ok(())
}

fn cmd_bootstrap(args: BootstrapArgs) -> Result<()> {
    let returns = load_returns(&args.returns)?;
    if returns.is_empty() {
        bail!("returns array is empty");
    }

    let block_size = args
        .block_size
        .unwrap_or_else(|| optimal_block_length(returns.len()));

    eprintln!(
        "Bootstrap method={:?}, n={}, block_size={}, seed={}",
        args.method, args.n, block_size, args.seed
    );

    let sample: Vec<f64> = match args.method {
        BootstrapMethod::Stationary => {
            let mut bs = StationaryBootstrap::new(block_size as f64);
            bs.seed = args.seed;
            bs.resample(&returns, args.n)?
        }
        BootstrapMethod::Circular => {
            let mut bs = CircularBlockBootstrap::new(block_size);
            bs.seed = args.seed;
            bs.resample(&returns, args.n)?
        }
        BootstrapMethod::Moving => {
            let mut bs = MovingBlockBootstrap::new(block_size);
            bs.seed = args.seed;
            bs.resample(&returns, args.n)?
        }
    };

    let orig_mean = slice_mean(&returns);
    let boot_mean = slice_mean(&sample);
    eprintln!(
        "Original mean: {:.6}  Bootstrap mean: {:.6}  Δ: {:.6}",
        orig_mean,
        boot_mean,
        (boot_mean - orig_mean).abs()
    );

    #[derive(Serialize)]
    struct BootstrapOutput {
        method: String,
        n: usize,
        block_size: usize,
        original_mean: f64,
        bootstrap_mean: f64,
        sample: Vec<f64>,
    }

    let out = BootstrapOutput {
        method: format!("{:?}", args.method),
        n: args.n,
        block_size,
        original_mean: orig_mean,
        bootstrap_mean: boot_mean,
        sample,
    };

    write_json(&args.output, &out)?;
    eprintln!("Bootstrap sample written to {}", args.output.display());
    Ok(())
}

fn cmd_fit(args: FitArgs) -> Result<()> {
    let returns = load_returns(&args.returns)?;
    if returns.is_empty() {
        bail!("returns array is empty");
    }
    let dist = fit_distribution(&returns);
    let json = serde_json::to_string_pretty(&dist)?;

    if let Some(out) = args.output {
        std::fs::write(&out, &json)?;
        eprintln!("Distribution written to {}", out.display());
    } else {
        println!("{}", json);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// IO helpers
// ---------------------------------------------------------------------------

fn load_returns(path: &PathBuf) -> Result<Vec<f64>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Could not read {}", path.display()))?;
    let returns: Vec<f64> = serde_json::from_str(&content)
        .with_context(|| format!("Could not parse JSON in {}", path.display()))?;
    Ok(returns)
}

fn write_json<T: Serialize>(path: &PathBuf, value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json).with_context(|| format!("Could not write {}", path.display()))?;
    Ok(())
}
