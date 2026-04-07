// alpha_report.rs
// CLI binary: load signal CSV, output alpha decay report.
//
// CSV format:
//   bar,asset_id,signal,return_1,return_5,return_20,return_60
//
// Computes IC series at all horizons, fits decay curves, outputs report.

use alpha_decay::{
    ic_analysis::{
        decay_curve::{DecayCurve, DecayModel},
        ic_series::{IcHorizon, IcSeries},
    },
    simulation::{
        alpha_simulator::{AlphaSimulator, DecayScenario},
        capacity::CapacityModel,
    },
};
use anyhow::{Context, Result};
use clap::Parser;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "alpha_report", about = "Generate alpha decay report from signal CSV")]
struct Args {
    /// Path to input CSV file.
    #[arg(short, long)]
    input: PathBuf,

    /// Rolling window for IC computation (default: 252).
    #[arg(short, long, default_value = "252")]
    window: usize,

    /// Number of bootstrap replications for half-life CI (default: 500).
    #[arg(short, long, default_value = "500")]
    n_boot: usize,

    /// Output JSON report to this file (optional).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Verbose output.
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct CsvRow {
    bar: usize,
    asset_id: String,
    signal: f64,
    return_1: f64,
    return_5: f64,
    return_20: f64,
    return_60: f64,
}

#[derive(Debug, serde::Serialize)]
struct AlphaReport {
    n_bars: usize,
    n_assets: usize,
    ic_stats: Vec<IcHorizonReport>,
    decay_fit: DecayFitReport,
    simulation: SimReport,
}

#[derive(Debug, serde::Serialize)]
struct IcHorizonReport {
    horizon: String,
    mean_ic: f64,
    std_ic: f64,
    icir: f64,
    t_stat: f64,
    p_value: f64,
    hit_rate: f64,
    n_obs: usize,
    significant: bool,
}

#[derive(Debug, serde::Serialize)]
struct DecayFitReport {
    best_model: String,
    lambda: Option<f64>,
    beta: Option<f64>,
    half_life_bars: Option<f64>,
    half_life_ci_lower: Option<f64>,
    half_life_ci_upper: Option<f64>,
    r_squared: f64,
    aic: f64,
    bic: f64,
}

#[derive(Debug, serde::Serialize)]
struct SimReport {
    initial_ic: f64,
    long_run_ic: f64,
    optimal_holding_bars: usize,
    expected_net_value: f64,
    prob_positive: f64,
    break_even_ic: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read CSV.
    let mut reader = csv::Reader::from_path(&args.input)
        .with_context(|| format!("Cannot open CSV: {}", args.input.display()))?;

    // Group rows by bar.
    let mut bars: HashMap<usize, Vec<CsvRow>> = HashMap::new();
    for result in reader.deserialize::<CsvRow>() {
        let row: CsvRow = result.context("Failed to deserialize CSV row")?;
        bars.entry(row.bar).or_default().push(row);
    }

    let mut bar_indices: Vec<usize> = bars.keys().cloned().collect();
    bar_indices.sort();
    let n_bars = bar_indices.len();
    let n_assets = bars.values().map(|v| v.len()).max().unwrap_or(0);

    if args.verbose {
        println!("Loaded {} bars, {} assets per bar (max)", n_bars, n_assets);
    }

    // Build IcSeries.
    let mut ic_series = IcSeries::new(args.window);

    for &bar in &bar_indices {
        let rows = &bars[&bar];
        let signals: Vec<f64> = rows.iter().map(|r| r.signal).collect();
        ic_series.push_signals(signals);

        // Push realized returns for each horizon.
        let horizons = [
            (IcHorizon::H1, 1usize),
            (IcHorizon::H5, 5),
            (IcHorizon::H20, 20),
            (IcHorizon::H60, 60),
        ];
        for (h, lag) in &horizons {
            if bar >= *lag {
                let lag_bar = bar - lag;
                if let Some(lag_rows) = bars.get(&lag_bar) {
                    let returns: Vec<f64> = match h {
                        IcHorizon::H1 => rows.iter().map(|r| r.return_1).collect(),
                        IcHorizon::H5 => rows.iter().map(|r| r.return_5).collect(),
                        IcHorizon::H20 => rows.iter().map(|r| r.return_20).collect(),
                        IcHorizon::H60 => rows.iter().map(|r| r.return_60).collect(),
                    };
                    let _ = lag_rows; // lag_rows available for alignment if needed.
                    ic_series.push_realized(*h, returns);
                }
            }
        }
    }

    // Compute IC stats.
    let all_stats = ic_series.all_ic_stats();

    let ic_horizon_reports: Vec<IcHorizonReport> = all_stats
        .iter()
        .map(|s| IcHorizonReport {
            horizon: format!("{:?}", s.horizon),
            mean_ic: s.mean_ic,
            std_ic: s.std_ic,
            icir: s.icir,
            t_stat: s.t_stat,
            p_value: s.p_value,
            hit_rate: s.hit_rate,
            n_obs: s.n_obs,
            significant: s.p_value < 0.05,
        })
        .collect();

    // Build decay profile and fit curve.
    let decay_profile = ic_series.ic_decay_profile();
    let curve = DecayCurve::new(
        decay_profile
            .iter()
            .map(|(h, ic)| (*h as f64, *ic))
            .collect(),
    );
    let (exp_fit, pow_fit, best_model) = curve.select_best();

    let (fit, model_name) = match best_model {
        DecayModel::Exponential => (exp_fit, "Exponential"),
        DecayModel::PowerLaw => (pow_fit, "PowerLaw"),
    };

    let hl_estimate = fit.as_ref().and_then(|f| {
        curve.bootstrap_half_life(best_model, args.n_boot, 42)
    });

    let decay_fit_report = match &fit {
        Some(f) => DecayFitReport {
            best_model: model_name.to_string(),
            lambda: f.lambda,
            beta: f.beta,
            half_life_bars: f.half_life(),
            half_life_ci_lower: hl_estimate.as_ref().map(|h| h.ci_lower),
            half_life_ci_upper: hl_estimate.as_ref().map(|h| h.ci_upper),
            r_squared: f.r_squared,
            aic: f.aic,
            bic: f.bic,
        },
        None => DecayFitReport {
            best_model: "None".to_string(),
            lambda: None,
            beta: None,
            half_life_bars: None,
            half_life_ci_lower: None,
            half_life_ci_upper: None,
            r_squared: 0.0,
            aic: f64::INFINITY,
            bic: f64::INFINITY,
        },
    };

    // Simulation.
    let h1_stats = ic_series.ic_stats(IcHorizon::H1);
    let scenario = DecayScenario {
        initial_ic: h1_stats.mean_ic.abs(),
        long_run_ic: h1_stats.mean_ic.abs() * 0.3,
        half_life_bars: decay_fit_report.half_life_bars.unwrap_or(20.0),
        ic_vol: h1_stats.std_ic,
        signal_vol: 0.01,
        holding_cost_per_bar: 0.0001,
        n_bars: 60,
    };
    let sim = AlphaSimulator::new(42);
    let sim_result = sim.simulate(&scenario, 500);
    let opt_h = sim_result.optimal_holding_horizon;
    let ev = sim_result
        .expected_net_value
        .get(opt_h.saturating_sub(1))
        .copied()
        .unwrap_or(0.0);

    let sim_report = SimReport {
        initial_ic: scenario.initial_ic,
        long_run_ic: scenario.long_run_ic,
        optimal_holding_bars: opt_h,
        expected_net_value: ev,
        prob_positive: sim_result.prob_positive_at_optimal,
        break_even_ic: sim_result.break_even_ic,
    };

    let report = AlphaReport {
        n_bars,
        n_assets,
        ic_stats: ic_horizon_reports,
        decay_fit: decay_fit_report,
        simulation: sim_report,
    };

    // Print human-readable summary.
    println!("=== Alpha Decay Report ===");
    println!("Bars: {}, Assets (max/bar): {}", report.n_bars, report.n_assets);
    println!("");
    println!("IC Statistics:");
    println!(
        "  {:<8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Horizon", "MeanIC", "StdIC", "ICIR", "T-stat", "P-value", "Sig"
    );
    for stat in &report.ic_stats {
        println!(
            "  {:<8} {:>8.4} {:>8.4} {:>8.3} {:>8.3} {:>8.4} {:>8}",
            stat.horizon,
            stat.mean_ic,
            stat.std_ic,
            stat.icir,
            stat.t_stat,
            stat.p_value,
            if stat.significant { "*" } else { "" }
        );
    }
    println!("");
    println!("Decay Fit:");
    println!("  Model: {}", report.decay_fit.best_model);
    if let Some(hl) = report.decay_fit.half_life_bars {
        println!("  Half-life: {:.1} bars", hl);
        if let (Some(lo), Some(hi)) = (
            report.decay_fit.half_life_ci_lower,
            report.decay_fit.half_life_ci_upper,
        ) {
            println!("  95% CI: [{:.1}, {:.1}]", lo, hi);
        }
    }
    println!("  R-squared: {:.4}", report.decay_fit.r_squared);
    println!("  AIC: {:.2}", report.decay_fit.aic);
    println!("");
    println!("Simulation:");
    println!("  Initial IC: {:.4}", report.simulation.initial_ic);
    println!("  Optimal holding: {} bars", report.simulation.optimal_holding_bars);
    println!("  Expected net value: {:.6}", report.simulation.expected_net_value);
    println!("  P(positive): {:.3}", report.simulation.prob_positive);
    println!("  Break-even IC: {:.4}", report.simulation.break_even_ic);

    // Optionally write JSON.
    if let Some(out_path) = &args.output {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(out_path, json)
            .with_context(|| format!("Cannot write output: {}", out_path.display()))?;
        println!("");
        println!("Report written to {}", out_path.display());
    }

    Ok(())
}
