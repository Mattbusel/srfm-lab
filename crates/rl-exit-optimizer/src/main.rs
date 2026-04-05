use clap::Parser;

use rl_exit_optimizer::agent::AgentConfig;
use rl_exit_optimizer::environment::{generate_synthetic_trades, load_trades_csv};
use rl_exit_optimizer::evaluator::{evaluate, print_comparison_table};
use rl_exit_optimizer::trainer::{Trainer, TrainerConfig};

/// RL Exit Optimizer -- train a Q-learning agent to find optimal trade exit timing.
#[derive(Parser, Debug)]
#[command(name = "rl-exit-optimizer", version, about)]
struct Cli {
    /// Path to input trades CSV. If omitted, synthetic data is generated.
    #[arg(long)]
    trades: Option<String>,

    /// Number of training episodes.
    #[arg(long, default_value_t = 10_000)]
    episodes: usize,

    /// Number of synthetic trades to generate (if --trades not provided).
    #[arg(long, default_value_t = 200)]
    synthetic_trades: usize,

    /// Output path for the trained Q-table JSON model.
    #[arg(long, default_value = "model.json")]
    output: String,

    /// Print per-trade comparison table after evaluation.
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Log training stats every N episodes.
    #[arg(long, default_value_t = 1000)]
    log_interval: usize,

    /// Q-learning rate alpha.
    #[arg(long, default_value_t = 0.1)]
    alpha: f64,

    /// Discount factor gamma.
    #[arg(long, default_value_t = 0.95)]
    gamma: f64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // ── Load or generate trade data ──────────────────────────────────────
    let trades = if let Some(ref path) = cli.trades {
        println!("Loading trades from: {}", path);
        load_trades_csv(path)?
    } else {
        println!(
            "No trades CSV provided. Generating {} synthetic trades...",
            cli.synthetic_trades
        );
        generate_synthetic_trades(cli.synthetic_trades, cli.seed)
    };

    println!("Loaded {} trades.", trades.len());

    // ── Configure and train ──────────────────────────────────────────────
    let agent_cfg = AgentConfig {
        epsilon_start: 1.0,
        epsilon_end: 0.05,
        epsilon_decay_steps: (cli.episodes as u64 * trades.len() as u64 / 2).max(1),
        alpha: cli.alpha,
        gamma: cli.gamma,
        batch_size: 64,
        buffer_capacity: 50_000,
        train_every: 4,
    };

    let trainer_cfg = TrainerConfig {
        num_episodes: cli.episodes,
        log_interval: cli.log_interval,
        seed: cli.seed,
        agent_config: agent_cfg,
    };

    println!(
        "Training for {} episodes (log every {})...",
        cli.episodes, cli.log_interval
    );

    let mut trainer = Trainer::new(trainer_cfg);
    trainer.train(&trades);

    println!("\nTraining complete.");
    println!("{}", trainer.agent.stats_line());

    // ── Evaluate ─────────────────────────────────────────────────────────
    println!("\nEvaluating agent vs BH baseline...");
    let report = evaluate(&trainer.agent, &trades);
    println!("{}", report);

    if cli.verbose {
        use rl_exit_optimizer::evaluator::compare_trade;
        let comparisons: Vec<_> = trades
            .iter()
            .take(50) // cap to 50 rows in verbose mode
            .map(|t| compare_trade(&trainer.agent, t))
            .collect();
        println!();
        print_comparison_table(&comparisons);
    }

    // ── Save model ───────────────────────────────────────────────────────
    trainer.agent.save(&cli.output)?;
    println!("\nModel saved to: {}", cli.output);

    Ok(())
}
