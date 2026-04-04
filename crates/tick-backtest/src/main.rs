// main.rs — CLI binary for the tick-backtest engine
//
// Subcommands
// ───────────
//   backtest  — single-instrument backtest
//   sweep     — grid or random parameter sweep
//   multi     — multi-instrument parallel backtest
//   mc        — Monte Carlo simulation from a trades CSV

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use tick_backtest::{
    csv_loader::{load_barsset, load_trades_csv, save_trades_csv},
    engine::{BacktestConfig, BacktestEngine},
    monte_carlo::{run_mc, MCConfig},
    multi_engine::{MultiBacktestConfig, MultiBacktestEngine},
    param_sweep::{grid_search, random_search, save_sweep_csv, Metric, ParamBounds, ParamGrid},
};

// ---------------------------------------------------------------------------
// Top-level CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "tick-backtest",
    about = "High-performance BH physics backtest engine",
    version,
    author
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a single-instrument backtest
    Backtest(BacktestArgs),
    /// Parameter sweep (grid or random search)
    Sweep(SweepArgs),
    /// Multi-instrument parallel backtest
    Multi(MultiArgs),
    /// Monte Carlo simulation from a trades CSV
    Mc(McArgs),
}

// ---------------------------------------------------------------------------
// Backtest subcommand
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct BacktestArgs {
    /// Directory containing <SYM>_1d.csv, <SYM>_1h.csv, <SYM>_15m.csv
    #[arg(long)]
    data_dir: PathBuf,

    /// Instrument symbol (e.g. BTC, AAPL)
    #[arg(long)]
    sym: String,

    /// Speed-of-light parameter
    #[arg(long, default_value = "0.001")]
    cf: f64,

    /// BH formation mass threshold
    #[arg(long, default_value = "1.5")]
    bh_form: f64,

    /// BH collapse mass threshold
    #[arg(long, default_value = "1.2")]
    bh_collapse: f64,

    /// Per-bar mass decay for spacelike moves
    #[arg(long, default_value = "0.95")]
    bh_decay: f64,

    /// Starting equity in dollars
    #[arg(long, default_value = "1000000")]
    starting_equity: f64,

    /// Max fraction of equity per position
    #[arg(long, default_value = "0.10")]
    max_pos_frac: f64,

    /// Transaction cost in basis points (one-way)
    #[arg(long, default_value = "2.0")]
    tc_bps: f64,

    /// Output JSON results file
    #[arg(long)]
    output: Option<PathBuf>,

    /// Output trades CSV file
    #[arg(long)]
    trades_csv: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Sweep subcommand
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct SweepArgs {
    /// Directory containing bar CSVs
    #[arg(long)]
    data_dir: PathBuf,

    /// Instrument symbol
    #[arg(long)]
    sym: String,

    /// Number of random trials (0 = use default grid)
    #[arg(long, default_value = "0")]
    n_trials: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Optimisation metric: sharpe, cagr, maxdd, profit_factor, calmar
    #[arg(long, default_value = "sharpe")]
    metric: String,

    /// Starting equity for backtest configs
    #[arg(long, default_value = "1000000")]
    starting_equity: f64,

    /// Output CSV file
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Multi subcommand
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct MultiArgs {
    /// Directory containing bar CSVs
    #[arg(long)]
    data_dir: PathBuf,

    /// Comma-separated list of symbols (e.g. BTC,ETH,SOL)
    #[arg(long)]
    syms: String,

    /// Speed-of-light parameter (shared)
    #[arg(long, default_value = "0.001")]
    cf: f64,

    /// BH formation mass threshold (shared)
    #[arg(long, default_value = "1.5")]
    bh_form: f64,

    /// BH collapse mass threshold (shared)
    #[arg(long, default_value = "1.2")]
    bh_collapse: f64,

    /// BH decay (shared)
    #[arg(long, default_value = "0.95")]
    bh_decay: f64,

    /// Total starting equity (split equally across instruments)
    #[arg(long, default_value = "1000000")]
    starting_equity: f64,

    /// Output JSON file
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// MC subcommand
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct McArgs {
    /// Path to trades CSV file
    #[arg(long)]
    trades: PathBuf,

    /// Starting equity in dollars
    #[arg(long, default_value = "1000000")]
    starting_equity: f64,

    /// Number of simulation paths
    #[arg(long, default_value = "10000")]
    n_sims: usize,

    /// Months to simulate
    #[arg(long, default_value = "12")]
    months: usize,

    /// AR(1) serial correlation coefficient
    #[arg(long, default_value = "0.0")]
    serial_corr: f64,

    /// Blowup threshold (fraction of starting equity)
    #[arg(long, default_value = "0.5")]
    blowup_threshold: f64,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output JSON file
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Backtest(args) => cmd_backtest(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Multi(args) => cmd_multi(args),
        Commands::Mc(args) => cmd_mc(args),
    }
}

// ---------------------------------------------------------------------------
// cmd_backtest
// ---------------------------------------------------------------------------

fn cmd_backtest(args: BacktestArgs) -> Result<()> {
    let t0 = Instant::now();

    println!(
        "tick-backtest ▶ backtest: sym={} cf={} bh_form={} bh_collapse={} bh_decay={}",
        args.sym, args.cf, args.bh_form, args.bh_collapse, args.bh_decay
    );

    let bars = load_barsset(&args.data_dir, &args.sym)
        .with_context(|| format!("Loading bars for {} from {:?}", args.sym, args.data_dir))?;

    println!(
        "  Loaded: {} daily, {} hourly, {} 15-min bars",
        bars.bars_1d.len(),
        bars.bars_1h.len(),
        bars.bars_15m.len()
    );

    let mut config = BacktestConfig::default_for(&args.sym);
    config.cf = args.cf;
    config.bh_form = args.bh_form;
    config.bh_collapse = args.bh_collapse;
    config.bh_decay = args.bh_decay;
    config.starting_equity = args.starting_equity;
    config.max_position_frac = args.max_pos_frac;
    config.transaction_cost_bps = args.tc_bps;

    let mut engine = BacktestEngine::new(config);
    let result = engine.run_barsset(&bars)?;

    let elapsed = t0.elapsed();
    let m = &result.metrics;

    println!("\n  ── Results ───────────────────────────────────────────");
    println!("  Final equity   : ${:.2}  (started ${:.2})", result.final_equity, args.starting_equity);
    println!("  Trades         : {}", m.total_trades);
    println!("  Win rate       : {:.1}%", m.win_rate * 100.0);
    println!("  Profit factor  : {:.3}", m.profit_factor);
    println!("  Sharpe         : {:.3}", m.sharpe);
    println!("  Max drawdown   : {:.1}%", m.max_drawdown * 100.0);
    println!("  CAGR           : {:.1}%", m.cagr * 100.0);
    println!("  Calmar         : {:.3}", m.calmar_ratio);
    println!("  Avg hold bars  : {:.1}", m.avg_hold_bars);
    println!("  ─────────────────────────────────────────────────────");

    if !result.regime_breakdown.is_empty() {
        println!("\n  Regime breakdown:");
        let mut regimes: Vec<_> = result.regime_breakdown.iter().collect();
        regimes.sort_by_key(|(k, _)| k.as_str());
        for (regime, stats) in regimes {
            println!(
                "    {:10} trades={:4}  win={:.1}%  pnl=${:.0}",
                regime,
                stats.count,
                stats.win_rate() * 100.0,
                stats.total_pnl,
            );
        }
    }

    println!("\n  Elapsed: {:.2?}", elapsed);

    // Optional outputs
    if let Some(out_path) = &args.output {
        let json = serde_json::to_string_pretty(&result)
            .context("Serialising result to JSON")?;
        std::fs::write(out_path, json)
            .with_context(|| format!("Writing results to {:?}", out_path))?;
        println!("  Results written to {:?}", out_path);
    }

    if let Some(csv_path) = &args.trades_csv {
        save_trades_csv(csv_path, &result.trades)?;
        println!("  Trades CSV written to {:?}", csv_path);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// cmd_sweep
// ---------------------------------------------------------------------------

fn parse_metric(s: &str) -> Metric {
    match s.to_lowercase().as_str() {
        "sharpe" => Metric::Sharpe,
        "cagr" => Metric::CAGR,
        "maxdd" | "max_dd" => Metric::MaxDD,
        "profit_factor" | "pf" => Metric::ProfitFactor,
        "calmar" => Metric::CalmarRatio,
        other => {
            eprintln!("Unknown metric '{other}', defaulting to Sharpe");
            Metric::Sharpe
        }
    }
}

fn cmd_sweep(args: SweepArgs) -> Result<()> {
    let t0 = Instant::now();
    let metric = parse_metric(&args.metric);

    println!(
        "tick-backtest ▶ sweep: sym={} metric={:?} n_trials={}",
        args.sym, metric, args.n_trials
    );

    let bars = load_barsset(&args.data_dir, &args.sym)
        .with_context(|| format!("Loading bars for {} from {:?}", args.sym, args.data_dir))?;

    let mut template = BacktestConfig::default_for(&args.sym);
    template.starting_equity = args.starting_equity;

    let sweep_result = if args.n_trials == 0 {
        let grid = ParamGrid::default_grid();
        let n = grid.expand().len();
        println!("  Grid search: {} combinations", n);
        grid_search(&bars, &grid, &template, metric)?
    } else {
        println!("  Random search: {} Halton-sampled trials", args.n_trials);
        let bounds = ParamBounds::default();
        random_search(&bars, args.n_trials, args.seed, &bounds, &template, metric)?
    };

    let best = &sweep_result.best_config;
    let bm = &sweep_result.best_result.metrics;

    println!("\n  ── Best parameters ───────────────────────────────────");
    println!("  cf={:.6}  bh_form={:.4}  bh_collapse={:.4}  bh_decay={:.4}",
        best.cf, best.bh_form, best.bh_collapse, best.bh_decay);
    println!("  Metric ({:?}): {:.4}", metric, sweep_result.best_metric_value);
    println!("  Sharpe {:.3}  CAGR {:.1}%  MaxDD {:.1}%  WinRate {:.1}%",
        bm.sharpe, bm.cagr * 100.0, bm.max_drawdown * 100.0, bm.win_rate * 100.0);
    println!("  ─────────────────────────────────────────────────────");

    println!("  Evaluated {} configurations in {:.2?}", sweep_result.all_results.len(), t0.elapsed());

    if let Some(out_path) = &args.output {
        save_sweep_csv(out_path, &sweep_result)?;
        println!("  Sweep CSV written to {:?}", out_path);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// cmd_multi
// ---------------------------------------------------------------------------

fn cmd_multi(args: MultiArgs) -> Result<()> {
    let t0 = Instant::now();

    let syms: Vec<&str> = args.syms.split(',').map(str::trim).collect();
    println!(
        "tick-backtest ▶ multi: syms={} cf={} bh_form={} bh_decay={}",
        args.syms, args.cf, args.bh_form, args.bh_decay
    );

    // Load bars for every symbol
    let mut all_bars: HashMap<String, tick_backtest::csv_loader::BarsSet> = HashMap::new();
    for &sym in &syms {
        let bars = load_barsset(&args.data_dir, sym)
            .with_context(|| format!("Loading bars for {sym}"))?;
        println!("  {sym}: {} 15m bars", bars.bars_15m.len());
        all_bars.insert(sym.to_string(), bars);
    }

    let multi_cfg = MultiBacktestConfig::new(args.starting_equity)
        .with_default_params(&syms, args.cf, args.bh_form, args.bh_collapse, args.bh_decay);

    let engine = MultiBacktestEngine::new(multi_cfg);
    let result = engine.run_parallel(&all_bars)?;

    let elapsed = t0.elapsed();
    println!("\n  {}", result.summary());

    // Per-instrument summary
    let mut sym_list: Vec<&String> = result.per_instrument.keys().collect();
    sym_list.sort();
    println!("\n  Per-instrument:");
    for sym in sym_list {
        let r = &result.per_instrument[sym];
        println!(
            "    {:10} trades={:4}  Sharpe={:.2}  CAGR={:.1}%  MaxDD={:.1}%",
            sym,
            r.metrics.total_trades,
            r.metrics.sharpe,
            r.metrics.cagr * 100.0,
            r.metrics.max_drawdown * 100.0,
        );
    }

    println!("\n  Elapsed: {:.2?}", elapsed);

    if let Some(out_path) = &args.output {
        let json = serde_json::to_string_pretty(&result.portfolio_metrics)?;
        std::fs::write(out_path, json)?;
        println!("  Portfolio metrics written to {:?}", out_path);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// cmd_mc
// ---------------------------------------------------------------------------

fn cmd_mc(args: McArgs) -> Result<()> {
    let t0 = Instant::now();

    println!(
        "tick-backtest ▶ mc: n_sims={} months={} serial_corr={} seed={}",
        args.n_sims, args.months, args.serial_corr, args.seed
    );

    let trades = load_trades_csv(&args.trades)
        .with_context(|| format!("Loading trades from {:?}", args.trades))?;
    println!("  Loaded {} trades", trades.len());

    let cfg = MCConfig {
        n_sims: args.n_sims,
        months: args.months,
        serial_corr: args.serial_corr,
        blowup_threshold: args.blowup_threshold,
        seed: args.seed,
        geometric: true,
        trades_per_month: (trades.len() / 12).max(5),
    };

    let result = run_mc(&trades, args.starting_equity, &cfg)?;
    let elapsed = t0.elapsed();

    println!("\n  {}", result.summary());
    println!("  Mean equity    : ${:.2}", result.mean_equity);
    println!("  25th pct       : ${:.2}", result.pct_25);
    println!("  75th pct       : ${:.2}", result.pct_75);
    println!("  Elapsed: {:.2?}", elapsed);

    if let Some(out_path) = &args.output {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(out_path, json)?;
        println!("  MC results written to {:?}", out_path);
    }

    Ok(())
}
