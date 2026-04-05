use clap::{Parser, ValueEnum};
use parallel_backtest::{
    bar_data::load_csv_dir,
    backtest::run_backtest,
    optimizer::multi_objective_optimize,
    params::{ParameterSpace, StrategyParams},
    sweep::{sweep, top_n_by_sharpe},
};
use std::path::PathBuf;

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    Sweep,
    Optimize,
    Single,
}

/// Massively parallel BH strategy parameter sweep engine.
#[derive(Debug, Parser)]
#[command(name = "parallel-backtest", version, about)]
struct Args {
    /// Operating mode: sweep (LHS parameter sweep), optimize (NSGA-II), single (one run).
    #[arg(long, default_value = "sweep")]
    mode: Mode,

    /// Directory containing per-symbol CSV files (one file per symbol).
    #[arg(long)]
    data: PathBuf,

    /// Path to JSON file with StrategyParams (only used in --mode single).
    #[arg(long)]
    params: Option<PathBuf>,

    /// Number of Latin Hypercube samples for sweep mode.
    #[arg(long, default_value_t = 1000)]
    n_samples: usize,

    /// Number of threads for parallel execution (0 = auto-detect).
    #[arg(long, default_value_t = 0)]
    n_threads: usize,

    /// Output file path for JSON results.
    #[arg(long, default_value = "results.json")]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Resolve thread count.
    let n_threads = if args.n_threads == 0 {
        num_cpus()
    } else {
        args.n_threads
    };

    eprintln!("[parallel-backtest] Loading data from {:?}", args.data);
    let data = load_csv_dir(&args.data)?;
    eprintln!("[parallel-backtest] Loaded {} symbols", data.len());
    for (sym, bars) in &data {
        eprintln!("  {} — {} bars", sym, bars.len());
    }

    match args.mode {
        Mode::Single => {
            let params = match &args.params {
                Some(path) => {
                    let json = std::fs::read_to_string(path)?;
                    StrategyParams::from_json(&json)?
                }
                None => {
                    eprintln!("[parallel-backtest] No --params file, using defaults");
                    StrategyParams::default()
                }
            };

            eprintln!("[parallel-backtest] Running single backtest...");
            let result = run_backtest(&data, &params);
            println!("=== Single Backtest Result ===");
            println!("  Sharpe:       {:.4}", result.sharpe);
            println!("  CAGR:         {:.2}%", result.cagr * 100.0);
            println!("  Max Drawdown: {:.2}%", result.max_drawdown * 100.0);
            println!("  Win Rate:     {:.2}%", result.win_rate * 100.0);
            println!("  Profit Factor:{:.3}", result.profit_factor);
            println!("  Total Trades: {}", result.total_trades);
            println!("  Final Equity: {:.4}", result.final_equity);

            let json = serde_json::to_string_pretty(&result)?;
            std::fs::write(&args.output, json)?;
            eprintln!("[parallel-backtest] Result saved to {:?}", args.output);
        }

        Mode::Sweep => {
            eprintln!(
                "[parallel-backtest] Generating {} LHS samples...",
                args.n_samples
            );
            let space = ParameterSpace::default();
            let params_list = space.sample(args.n_samples);

            eprintln!("[parallel-backtest] Starting sweep...");
            let results = sweep(&data, params_list, n_threads);

            let top10 = top_n_by_sharpe(results, 10);

            println!("=== Top 10 by Sharpe ===");
            for (rank, (params, res)) in top10.iter().enumerate() {
                println!(
                    "#{:>2}  Sharpe={:.4}  CAGR={:.1}%  DD={:.1}%  WinRate={:.1}%  min_hold={}  cf={}  garch_vol={:.2}",
                    rank + 1,
                    res.sharpe,
                    res.cagr * 100.0,
                    res.max_drawdown * 100.0,
                    res.win_rate * 100.0,
                    params.min_hold_bars,
                    params.stale_15m_move,
                    params.garch_target_vol,
                );
            }

            let json = serde_json::to_string_pretty(&top10)?;
            std::fs::write(&args.output, json)?;
            eprintln!("[parallel-backtest] Results saved to {:?}", args.output);
        }

        Mode::Optimize => {
            eprintln!("[parallel-backtest] Running NSGA-II optimisation...");
            let space = ParameterSpace::default();
            let pareto = multi_objective_optimize(&data, &space);

            println!("=== Pareto Front (top {}) ===", pareto.len());
            for (i, p) in pareto.iter().enumerate() {
                println!(
                    "#{:>2}  Sharpe={:.4}  DD={:.1}%  WinRate={:.1}%  CAGR={:.1}%",
                    i + 1,
                    p.sharpe,
                    p.max_drawdown * 100.0,
                    p.win_rate * 100.0,
                    p.cagr * 100.0,
                );
            }

            let json = serde_json::to_string_pretty(&pareto)?;
            std::fs::write(&args.output, json)?;
            eprintln!("[parallel-backtest] Pareto results saved to {:?}", args.output);
        }
    }

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
