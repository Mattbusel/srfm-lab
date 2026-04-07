//! Factor analytics report CLI.
//!
//! Loads a CSV of returns and factor scores, runs the full analytics pipeline,
//! and outputs a structured factor analytics report.
//!
//! Expected CSV format:
//!   date,asset_id,return,factor_1,...,factor_N
//!
//! Example:
//!   factor_report --input data.csv --factors momentum,value --n-buckets 5 --output report.json

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use factor_analytics::{
    assign_buckets, compute_ic, compute_ic_statistics, compute_icir,
    fama_macbeth, format_backtest_summary, newey_west_se, run_factor_backtest,
    BacktestConfig, FactorError, NormalizationMethod, normalize_factor,
    rank_autocorrelation, zscore_cross_section, Result,
};

use ndarray::Array2;

/// CLI arguments.
#[derive(Parser, Debug)]
#[command(
    name = "factor_report",
    about = "Factor analytics report generator for multi-factor equity models",
    version = "0.1.0"
)]
struct Cli {
    /// Input CSV file path
    #[arg(short, long)]
    input: PathBuf,

    /// Comma-separated list of factor column names to analyze
    #[arg(short, long, value_delimiter = ',')]
    factors: Vec<String>,

    /// Number of quantile buckets (5 = quintile, 10 = decile)
    #[arg(long, default_value = "5")]
    n_buckets: usize,

    /// Rebalancing frequency in trading days (21 = monthly)
    #[arg(long, default_value = "21")]
    rebalance_freq: usize,

    /// Normalization method: zscore, rank, winsorized, mad
    #[arg(long, value_enum, default_value = "winsorized")]
    normalization: NormMethod,

    /// Winsorization sigma (only used with winsorized method)
    #[arg(long, default_value = "3.0")]
    winsor_sigma: f64,

    /// Newey-West lags for HAC standard errors
    #[arg(long, default_value = "4")]
    nw_lags: usize,

    /// Output file path for JSON report (stdout if not specified)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Transaction cost per unit of turnover (one-way, in decimal)
    #[arg(long, default_value = "0.001")]
    transaction_cost: f64,

    /// Whether to run Fama-MacBeth regressions
    #[arg(long, default_value = "true")]
    fama_macbeth: bool,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
enum NormMethod {
    Zscore,
    Rank,
    Winsorized,
    Mad,
}

impl NormMethod {
    fn to_normalization(&self, sigma: f64) -> NormalizationMethod {
        match self {
            NormMethod::Zscore => NormalizationMethod::ZScore,
            NormMethod::Rank => NormalizationMethod::Rank,
            NormMethod::Winsorized => NormalizationMethod::WinsorizedZScore { sigma },
            NormMethod::Mad => NormalizationMethod::MadRobust,
        }
    }
}

/// A single row from the input CSV.
#[derive(Debug, Deserialize)]
struct CsvRow {
    date: String,
    asset_id: String,
    #[serde(rename = "return")]
    ret: f64,
    #[serde(flatten)]
    factors: HashMap<String, f64>,
}

/// Parsed panel data.
struct PanelData {
    /// Unique sorted dates
    dates: Vec<String>,
    /// Unique sorted asset IDs
    assets: Vec<String>,
    /// Returns matrix (n_periods x n_assets)
    returns: Array2<f64>,
    /// Factor matrices keyed by factor name
    factor_matrices: HashMap<String, Array2<f64>>,
}

/// Load CSV into panel data structure.
fn load_csv(path: &PathBuf, factor_names: &[String]) -> anyhow::Result<PanelData> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut rows: Vec<CsvRow> = Vec::new();

    for result in reader.deserialize::<CsvRow>() {
        match result {
            Ok(row) => rows.push(row),
            Err(e) => {
                eprintln!("Warning: skipping malformed row: {}", e);
            }
        }
    }

    if rows.is_empty() {
        anyhow::bail!("No valid rows loaded from CSV");
    }

    // Collect unique dates and assets
    let mut date_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut asset_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for row in &rows {
        date_set.insert(row.date.clone());
        asset_set.insert(row.asset_id.clone());
    }

    let dates: Vec<String> = date_set.into_iter().collect();
    let assets: Vec<String> = asset_set.into_iter().collect();
    let n_periods = dates.len();
    let n_assets = assets.len();

    let date_idx: HashMap<&str, usize> = dates.iter().enumerate().map(|(i, d)| (d.as_str(), i)).collect();
    let asset_idx: HashMap<&str, usize> = assets.iter().enumerate().map(|(i, a)| (a.as_str(), i)).collect();

    let mut returns_mat = Array2::<f64>::from_elem((n_periods, n_assets), f64::NAN);
    let mut factor_mats: HashMap<String, Array2<f64>> = factor_names
        .iter()
        .map(|f| (f.clone(), Array2::<f64>::from_elem((n_periods, n_assets), f64::NAN)))
        .collect();

    for row in &rows {
        let t = date_idx[row.date.as_str()];
        let i = asset_idx[row.asset_id.as_str()];
        returns_mat[[t, i]] = row.ret;

        for factor in factor_names {
            if let Some(&val) = row.factors.get(factor.as_str()) {
                if let Some(mat) = factor_mats.get_mut(factor) {
                    mat[[t, i]] = val;
                }
            }
        }
    }

    Ok(PanelData {
        dates,
        assets,
        returns: returns_mat,
        factor_matrices: factor_mats,
    })
}

/// Per-factor analytics report.
#[derive(Debug, Serialize)]
struct FactorReport {
    factor_name: String,
    mean_ic: f64,
    icir: f64,
    t_stat_ic: f64,
    pct_positive_ic: f64,
    rank_autocorr_lag1: f64,
    return_spread_annual: f64,
    monotonicity: f64,
    bucket_annualized_returns: Vec<f64>,
    bucket_sharpes: Vec<f64>,
    long_short_annualized_return: f64,
    long_short_sharpe: f64,
    avg_turnover: f64,
}

/// Full report output.
#[derive(Debug, Serialize)]
struct FullReport {
    generated_at: String,
    input_file: String,
    n_periods: usize,
    n_assets: usize,
    n_factors: usize,
    normalization_method: String,
    n_buckets: usize,
    rebalance_freq: usize,
    factor_reports: Vec<FactorReport>,
    fama_macbeth_premiums: HashMap<String, f64>,
    fama_macbeth_t_stats: HashMap<String, f64>,
    fama_macbeth_nw_t_stats: HashMap<String, f64>,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Cli::parse();

    if args.verbose > 0 {
        eprintln!("Loading data from: {}", args.input.display());
    }

    // Validate factors argument
    if args.factors.is_empty() {
        anyhow::bail!("At least one factor name must be specified with --factors");
    }

    let panel = load_csv(&args.input, &args.factors)?;
    let n_periods = panel.dates.len();
    let n_assets = panel.assets.len();
    let n_factors = args.factors.len();

    if args.verbose > 0 {
        eprintln!("Loaded {} periods x {} assets", n_periods, n_assets);
    }

    let norm_method = args.normalization.to_normalization(args.winsor_sigma);

    // Normalize factor panels cross-sectionally
    let mut normalized_factor_panels: HashMap<String, Array2<f64>> = HashMap::new();
    for factor_name in &args.factors {
        if let Some(raw_mat) = panel.factor_matrices.get(factor_name) {
            let mut norm_mat = Array2::<f64>::from_elem((n_periods, n_assets), f64::NAN);
            for t in 0..n_periods {
                let row: Vec<f64> = raw_mat.row(t).to_vec();
                let normalized = normalize_factor(&row, norm_method);
                for i in 0..n_assets {
                    norm_mat[[t, i]] = normalized[i];
                }
            }
            normalized_factor_panels.insert(factor_name.clone(), norm_mat);
        }
    }

    // Per-factor analysis
    let mut factor_reports = Vec::new();

    for factor_name in &args.factors {
        let factor_mat = match normalized_factor_panels.get(factor_name) {
            Some(m) => m,
            None => {
                eprintln!("Warning: factor '{}' not found in data", factor_name);
                continue;
            }
        };

        if args.verbose > 0 {
            eprintln!("Analyzing factor: {}", factor_name);
        }

        // IC time series
        let mut ic_series: Vec<f64> = Vec::new();
        for t in 0..(n_periods.saturating_sub(1)) {
            let factor_t: Vec<f64> = factor_mat.row(t).to_vec();
            let ret_t1: Vec<f64> = panel.returns.row(t + 1).to_vec();
            let ic = compute_ic(&factor_t, &ret_t1);
            ic_series.push(ic);
        }

        let ic_stats = compute_ic_statistics(&ic_series);
        let rank_ac = rank_autocorrelation(factor_mat, 1);

        // Fraction of periods with positive IC
        let valid_ics: Vec<f64> = ic_series.iter().copied().filter(|v| v.is_finite()).collect();
        let pct_pos_ic = if !valid_ics.is_empty() {
            valid_ics.iter().filter(|&&v| v > 0.0).count() as f64 / valid_ics.len() as f64
        } else {
            f64::NAN
        };

        // Quintile backtest
        let backtest_config = BacktestConfig {
            n_buckets: args.n_buckets,
            rebalance_freq: args.rebalance_freq,
            transaction_cost: args.transaction_cost,
            ..Default::default()
        };

        let (bucket_ann_rets, bucket_sharpes, ls_ret, ls_sharpe, return_spread, monotonicity, avg_turnover) =
            match run_factor_backtest(factor_mat, &panel.returns, &backtest_config) {
                Ok(result) => {
                    let ann_rets: Vec<f64> = result.bucket_stats.iter().map(|s| s.annualized_return).collect();
                    let sharpes: Vec<f64> = result.bucket_stats.iter().map(|s| s.sharpe_ratio).collect();
                    let (ls_r, ls_s) = result.long_short_stats
                        .as_ref()
                        .map(|s| (s.annualized_return, s.sharpe_ratio))
                        .unwrap_or((f64::NAN, f64::NAN));
                    let avg_turn = result.bucket_stats.iter().map(|s| s.avg_turnover).filter(|v| v.is_finite()).sum::<f64>()
                        / result.bucket_stats.iter().filter(|s| s.avg_turnover.is_finite()).count().max(1) as f64;
                    (ann_rets, sharpes, ls_r, ls_s, result.return_spread, result.monotonicity, avg_turn)
                }
                Err(e) => {
                    eprintln!("Backtest error for {}: {}", factor_name, e);
                    (vec![f64::NAN; args.n_buckets], vec![f64::NAN; args.n_buckets], f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
                }
            };

        factor_reports.push(FactorReport {
            factor_name: factor_name.clone(),
            mean_ic: ic_stats.ic,
            icir: ic_stats.icir,
            t_stat_ic: ic_stats.t_statistic,
            pct_positive_ic: pct_pos_ic,
            rank_autocorr_lag1: rank_ac,
            return_spread_annual: return_spread,
            monotonicity,
            bucket_annualized_returns: bucket_ann_rets,
            bucket_sharpes,
            long_short_annualized_return: ls_ret,
            long_short_sharpe: ls_sharpe,
            avg_turnover,
        });
    }

    // Fama-MacBeth regressions
    let mut fm_premiums: HashMap<String, f64> = HashMap::new();
    let mut fm_t_stats: HashMap<String, f64> = HashMap::new();
    let mut fm_nw_t_stats: HashMap<String, f64> = HashMap::new();

    if args.fama_macbeth && n_factors > 0 {
        if args.verbose > 0 {
            eprintln!("Running Fama-MacBeth regressions...");
        }

        let factor_panels_ordered: Vec<Array2<f64>> = args.factors
            .iter()
            .filter_map(|f| normalized_factor_panels.get(f).cloned())
            .collect();

        let factor_names_present: Vec<String> = args.factors
            .iter()
            .filter(|f| normalized_factor_panels.contains_key(*f))
            .cloned()
            .collect();

        if !factor_panels_ordered.is_empty() {
            match fama_macbeth(
                &panel.returns,
                &factor_panels_ordered,
                &factor_names_present,
                args.nw_lags,
            ) {
                Ok(result) => {
                    for (i, name) in factor_names_present.iter().enumerate() {
                        fm_premiums.insert(name.clone(), result.avg_factor_premiums[i]);
                        fm_t_stats.insert(name.clone(), result.t_statistics[i]);
                        fm_nw_t_stats.insert(name.clone(), result.t_statistics_nw[i]);
                    }
                }
                Err(e) => {
                    eprintln!("Fama-MacBeth error: {}", e);
                }
            }
        }
    }

    // Build full report
    let report = FullReport {
        generated_at: chrono::Utc::now().to_rfc3339(),
        input_file: args.input.display().to_string(),
        n_periods,
        n_assets,
        n_factors,
        normalization_method: format!("{:?}", args.normalization),
        n_buckets: args.n_buckets,
        rebalance_freq: args.rebalance_freq,
        factor_reports,
        fama_macbeth_premiums: fm_premiums,
        fama_macbeth_t_stats: fm_t_stats,
        fama_macbeth_nw_t_stats: fm_nw_t_stats,
    };

    // Output report
    let json_output = serde_json::to_string_pretty(&report)?;

    match args.output {
        Some(out_path) => {
            std::fs::write(&out_path, &json_output)?;
            eprintln!("Report written to: {}", out_path.display());
        }
        None => {
            println!("{}", json_output);
        }
    }

    // Print text summary to stderr
    if args.verbose > 0 {
        eprintln!("\n--- Factor Summary ---");
        for fr in &report.factor_reports {
            eprintln!(
                "{}: IC={:.4}, ICIR={:.3}, t={:.3}, spread={:.2}%",
                fr.factor_name,
                fr.mean_ic,
                fr.icir,
                fr.t_stat_ic,
                fr.return_spread_annual * 100.0,
            );
        }
    }

    Ok(())
}
